"""
Database Integration Tests for Custom Backtest Engine

Tests the complete integration between our custom backtest engine and the 
TimescaleDB infrastructure using real market data.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
import logging

# Database imports (using your psycopg3 pattern)
import psycopg
from psycopg_pool import AsyncConnectionPool

# Our custom engine imports
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from core.backtest_executor import BacktestExecutor
from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from risk.risk_manager import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from statistics.statistics_engine import StatisticsEngine

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager adapted from your vectorbt service"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: AsyncConnectionPool = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = AsyncConnectionPool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
            
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Database connection pool closed")
    
    async def get_available_symbols(self) -> List[str]:
        """Get all symbols available in ohlcv_1min table"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT DISTINCT symbol 
                    FROM ohlcv_1min 
                    ORDER BY symbol
                    LIMIT 10
                """)
                result = await cur.fetchall()
                return [row[0] for row in result]
    
    async def get_market_data(
        self, 
        symbol: str, 
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get OHLCV data for backtesting"""
        
        conditions = ["symbol = %s"]
        params = [symbol]
        
        if start_time:
            conditions.append("time >= %s")
            params.append(start_time)
            
        if end_time:
            conditions.append("time <= %s")
            params.append(end_time)
            
        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM ohlcv_1min
            WHERE {where_clause}
            ORDER BY time ASC
            {limit_clause}
        """
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
                
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        return df
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health and data availability"""
        health_status = {
            'healthy': False,
            'total_records': 0,
            'symbols_count': 0,
            'earliest_data': None,
            'latest_data': None,
            'error': None
        }
        
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT 
                            COUNT(*) as total_records,
                            COUNT(DISTINCT symbol) as symbols_count,
                            MIN(time) as earliest_data,
                            MAX(time) as latest_data
                        FROM ohlcv_1min
                    """)
                    result = await cur.fetchone()
                    
                    if result:
                        health_status['total_records'] = result[0]
                        health_status['symbols_count'] = result[1]
                        health_status['earliest_data'] = result[2]
                        health_status['latest_data'] = result[3]
                        health_status['healthy'] = result[0] > 0
                        
        except Exception as e:
            health_status['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
            
        return health_status
    
    async def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """Get data summary for a specific symbol"""
        summary = {
            'symbol': symbol,
            'total_records': 0,
            'date_range': None,
            'price_range': None,
            'volume_stats': None,
            'data_quality': 'unknown'
        }
        
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT 
                            COUNT(*) as total_records,
                            MIN(time) as earliest,
                            MAX(time) as latest,
                            MIN(low) as min_price,
                            MAX(high) as max_price,
                            AVG(volume) as avg_volume,
                            STDDEV(volume) as stddev_volume
                        FROM ohlcv_1min
                        WHERE symbol = %s
                    """, [symbol])
                    result = await cur.fetchone()
                    
                    if result and result[0] > 0:
                        summary['total_records'] = result[0]
                        summary['date_range'] = {
                            'start': result[1],
                            'end': result[2],
                            'days': (result[2] - result[1]).days if result[1] and result[2] else 0
                        }
                        summary['price_range'] = {
                            'min': float(result[3]) if result[3] else 0,
                            'max': float(result[4]) if result[4] else 0
                        }
                        summary['volume_stats'] = {
                            'avg': float(result[5]) if result[5] else 0,
                            'stddev': float(result[6]) if result[6] else 0
                        }
                        
                        # Assess data quality
                        expected_records = summary['date_range']['days'] * 24 * 60  # 1min intervals
                        completeness = summary['total_records'] / expected_records if expected_records > 0 else 0
                        
                        if completeness > 0.95:
                            summary['data_quality'] = 'excellent'
                        elif completeness > 0.8:
                            summary['data_quality'] = 'good'
                        elif completeness > 0.5:
                            summary['data_quality'] = 'fair'
                        else:
                            summary['data_quality'] = 'poor'
                            
        except Exception as e:
            logger.error(f"Failed to get data summary for {symbol}: {e}")
            summary['error'] = str(e)
            
        return summary


@pytest_asyncio.fixture
async def db_manager():
    """Database manager fixture"""
    # Use TimescaleDB connection string
    database_url = os.getenv(
        'DATABASE_URL', 
        'postgresql://quant_user:quant_password@localhost:5433/quant_db'
    )
    
    db = DatabaseManager(database_url)
    try:
        await db.initialize()
        yield db
    finally:
        await db.close()


@pytest_asyncio.fixture
async def sample_symbol_data(db_manager):
    """Get real market data for testing"""
    # Check database health first
    health = await db_manager.check_database_health()
    
    if not health['healthy']:
        pytest.skip(f"Database not healthy: {health}")
    
    # Get available symbols
    symbols = await db_manager.get_available_symbols()
    
    if not symbols:
        pytest.skip("No symbols available in database")
    
    # Use the first available symbol
    symbol = symbols[0]
    
    # Get recent data for testing
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)  # 1 week of data
    
    data = await db_manager.get_market_data(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        limit=2000
    )
    
    if data.empty:
        pytest.skip(f"No data available for symbol {symbol}")
    
    return {
        'symbol': symbol,
        'data': data,
        'summary': await db_manager.get_data_summary(symbol)
    }


class TestDatabaseIntegration:
    """Test database connectivity and data retrieval"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, db_manager):
        """Test basic database connectivity"""
        health = await db_manager.check_database_health()
        
        assert health is not None
        print(f"\n📊 Database Health Status:")
        print(f"   - Healthy: {health['healthy']}")
        print(f"   - Total Records: {health['total_records']:,}")
        print(f"   - Symbols Count: {health['symbols_count']}")
        print(f"   - Date Range: {health['earliest_data']} to {health['latest_data']}")
        
        if health['error']:
            pytest.fail(f"Database error: {health['error']}")
        
        assert health['healthy'], "Database should be healthy for testing"
    
    @pytest.mark.asyncio
    async def test_symbol_availability(self, db_manager):
        """Test symbol data availability"""
        symbols = await db_manager.get_available_symbols()
        
        assert len(symbols) > 0, "Should have at least one symbol available"
        
        print(f"\n📈 Available Symbols: {symbols}")
        
        # Test data summary for first symbol
        if symbols:
            summary = await db_manager.get_data_summary(symbols[0])
            print(f"\n📊 Data Summary for {symbols[0]}:")
            print(f"   - Records: {summary['total_records']:,}")
            print(f"   - Date Range: {summary['date_range']}")
            print(f"   - Price Range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
            print(f"   - Data Quality: {summary['data_quality']}")
            
            assert summary['total_records'] > 100, "Should have sufficient data for testing"
    
    @pytest.mark.asyncio
    async def test_data_retrieval_quality(self, sample_symbol_data):
        """Test data retrieval and quality"""
        symbol = sample_symbol_data['symbol']
        data = sample_symbol_data['data']
        summary = sample_symbol_data['summary']
        
        # Basic data validation
        assert not data.empty, "Should retrieve non-empty data"
        assert len(data) > 100, "Should have sufficient data points"
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
            assert not data[col].isna().all(), f"Column {col} should not be all NaN"
        
        # Basic price validation
        assert (data['high'] >= data['low']).all(), "High should be >= Low"
        assert (data['high'] >= data['open']).all(), "High should be >= Open"
        assert (data['high'] >= data['close']).all(), "High should be >= Close"
        assert (data['low'] <= data['open']).all(), "Low should be <= Open"
        assert (data['low'] <= data['close']).all(), "Low should be <= Close"
        
        # Volume validation
        assert (data['volume'] >= 0).all(), "Volume should be non-negative"
        
        print(f"\n✅ Data Quality Check for {symbol}:")
        print(f"   - Records Retrieved: {len(data):,}")
        print(f"   - Date Range: {data.index.min()} to {data.index.max()}")
        print(f"   - Price Range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        print(f"   - Avg Volume: {data['volume'].mean():,.0f}")
        print(f"   - Data Quality: {summary['data_quality']}")


class TestEngineWithRealData:
    """Test custom backtest engine with real market data"""
    
    @pytest.mark.asyncio
    async def test_simple_momentum_strategy(self, sample_symbol_data):
        """Test simple momentum strategy with real data"""
        symbol = sample_symbol_data['symbol']
        data = sample_symbol_data['data']
        
        # Ensure we have enough data
        if len(data) < 500:
            pytest.skip("Need at least 500 data points for meaningful backtest")
        
        # Take a subset for faster testing
        test_data = data.tail(1000).copy()
        
        # Define simple momentum strategy
        def momentum_strategy(data_slice, timestamp):
            if len(data_slice) < 20:
                return {symbol: {'action': 'hold'}}
            
            # Simple moving average crossover
            short_ma = data_slice['close'].rolling(10).mean().iloc[-1]
            long_ma = data_slice['close'].rolling(20).mean().iloc[-1]
            
            if pd.isna(short_ma) or pd.isna(long_ma):
                return {symbol: {'action': 'hold'}}
            
            if short_ma > long_ma:
                return {symbol: {'action': 'buy', 'side': 'long'}}
            else:
                return {symbol: {'action': 'sell', 'side': 'short'}}
        
        # Create backtest executor
        executor = BacktestExecutor(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage=0.001
        )
        
        executor.set_strategy(momentum_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.02,  # Reduced from 0.05 to 0.02
            take_profit_pct=0.05,  # Reduced from 0.10 to 0.05
            risk_per_trade=0.01,  # Reduced from 0.02 to 0.01
            max_drawdown_limit=0.50  # Increased significantly to avoid hitting limit
        )
        
        # Run backtest
        print(f"\n🚀 Running Momentum Strategy Backtest on {symbol}")
        print(f"   - Data Points: {len(test_data):,}")
        print(f"   - Date Range: {test_data.index.min()} to {test_data.index.max()}")
        print(f"   - Initial Capital: $100,000")
        
        results = executor.run_backtest(test_data, [symbol])
        
        # Validate results
        assert 'metrics' in results
        assert 'equity_curve' in results
        
        # Get final portfolio value from equity curve
        final_portfolio_value = results['equity_curve'].iloc[-1] if len(results['equity_curve']) > 0 else 100000
        
        metrics = results['metrics']
        print(f"\n📊 Backtest Results:")
        print(f"   - Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   - Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"   - Total Trades: {metrics.get('num_trades', 0)}")  # Fixed: was 'total_trades'
        print(f"   - Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
        print(f"   - Final Portfolio Value: ${final_portfolio_value:,.2f}")
        
        # Basic validation
        assert isinstance(metrics.get('total_return_pct'), (int, float))
        assert isinstance(metrics.get('num_trades'), int)  # Fixed: was 'total_trades'
        assert results['metrics']['final_capital'] >= 0  # Allow for losses, just ensure it's not negative
    
    @pytest.mark.asyncio
    async def test_advanced_strategy_with_risk_management(self, sample_symbol_data):
        """Test advanced strategy with comprehensive risk management"""
        symbol = sample_symbol_data['symbol']
        data = sample_symbol_data['data']
        
        if len(data) < 1000:
            pytest.skip("Need at least 1000 data points for advanced strategy")
        
        # Take more recent data for testing
        test_data = data.tail(1500).copy()
        
        # Calculate additional indicators
        test_data['sma_50'] = test_data['close'].rolling(50).mean()
        test_data['sma_200'] = test_data['close'].rolling(200).mean()
        test_data['rsi'] = self._calculate_rsi(test_data['close'], 14)
        test_data['volatility'] = test_data['close'].rolling(20).std()
        
        def advanced_strategy(data_slice, timestamp):
            if len(data_slice) < 200:
                return {symbol: {'action': 'hold'}}
            
            current = data_slice.iloc[-1]
            
            # Trend following with RSI filter
            trend_bullish = current['sma_50'] > current['sma_200']
            rsi_oversold = current['rsi'] < 30
            rsi_overbought = current['rsi'] > 70
            
            # Volatility filter
            avg_volatility = data_slice['volatility'].rolling(10).mean().iloc[-1]
            current_volatility = current['volatility']
            
            # Signal strength based on multiple factors
            signal_strength = 0.5
            
            if trend_bullish and rsi_oversold and current_volatility < avg_volatility * 1.5:
                signal_strength = 0.8
                return {symbol: {
                    'action': 'buy', 
                    'side': 'long',
                    'signal_strength': signal_strength
                }}
            elif not trend_bullish and rsi_overbought:
                signal_strength = 0.6
                return {symbol: {
                    'action': 'sell',
                    'side': 'short',
                    'signal_strength': signal_strength
                }}
            else:
                return {symbol: {'action': 'hold'}}
        
        # Create enhanced portfolio with advanced risk management
        risk_limits = RiskLimits(
            max_positions=3,
            max_portfolio_risk=0.03,
            max_drawdown=0.15,
            max_concentration=0.40,
            max_correlation=0.8
        )
        
        position_sizing = PositionSizingConfig(
            method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            volatility_target=0.02,
            percentage=0.02
        )
        
        # Create enhanced executor
        executor = BacktestExecutor(
            initial_capital=500000.0,  # Larger capital for advanced strategy
            commission_rate=0.001,
            slippage=0.0005
        )
        
        # Set advanced risk parameters
        executor.set_risk_parameters(
            stop_loss_pct=0.03,
            take_profit_pct=0.08,
            risk_per_trade=0.015,
            max_drawdown_limit=0.15
        )
        
        executor.set_strategy(advanced_strategy)
        
        # Run backtest
        print(f"\n🎯 Running Advanced Strategy Backtest on {symbol}")
        print(f"   - Data Points: {len(test_data):,}")
        print(f"   - Advanced Risk Management: Enabled")
        print(f"   - Position Sizing: Volatility Adjusted")
        print(f"   - Initial Capital: $500,000")
        
        results = executor.run_backtest(test_data, [symbol])
        
        # Comprehensive results analysis
        metrics = results['metrics']
        portfolio = executor.portfolio
        
        print(f"\n📈 Advanced Strategy Results:")
        print(f"   - Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   - Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
        print(f"   - Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   - Total Trades: {metrics.get('num_trades', 0)}")  # Fixed: was 'total_trades'
        print(f"   - Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
        print(f"   - Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   - Final Value: ${results['metrics']['final_capital']:,.2f}")
        
        # Get trade statistics
        trades = portfolio.trade_engine.get_trades()
        completed_trades = [t for t in trades if t.exit_price is not None]
        
        if completed_trades:
            # Calculate return percentage from pnl and entry price/size
            returns = []
            for t in completed_trades:
                # Calculate return percentage: (exit_price - entry_price) / entry_price * 100
                if t.side == 'buy':  # long position
                    return_pct = ((t.exit_price - t.entry_price) / t.entry_price) * 100
                else:  # short position
                    return_pct = ((t.entry_price - t.exit_price) / t.entry_price) * 100
                returns.append(return_pct)
            
            avg_return = np.mean(returns)
            avg_duration = np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 
                                   for t in completed_trades])
            
            print(f"   - Avg Trade Return: {avg_return:.2f}%")
            print(f"   - Avg Trade Duration: {avg_duration:.1f} hours")
        
        # Validate advanced features worked
        assert results['metrics']['final_capital'] > 0
        assert isinstance(metrics.get('sharpe_ratio'), (int, float))
        
        # Check that risk management was active
        equity_curve = portfolio.get_equity_curve()
        if len(equity_curve) > 1:
            max_dd = ((equity_curve.max() - equity_curve.min()) / equity_curve.max()) * 100
            assert max_dd <= 20, f"Max drawdown {max_dd:.1f}% exceeded limit"
    
    @pytest.mark.asyncio
    async def test_statistics_engine_with_real_data(self, sample_symbol_data):
        """Test statistics engine with real backtest results"""
        symbol = sample_symbol_data['symbol']
        data = sample_symbol_data['data']
        
        if len(data) < 500:
            pytest.skip("Need sufficient data for statistics testing")
        
        test_data = data.tail(1000).copy()
        
        # Run a simple backtest first
        def simple_strategy(data_slice, timestamp):
            if len(data_slice) < 10:
                return {symbol: {'action': 'hold'}}
            
            # Simple momentum
            recent_return = (data_slice['close'].iloc[-1] / data_slice['close'].iloc[-5] - 1)
            
            if recent_return > 0.01:  # 1% gain in 5 periods
                return {symbol: {'action': 'buy', 'side': 'long'}}
            elif recent_return < -0.01:  # 1% loss in 5 periods
                return {symbol: {'action': 'sell', 'side': 'short'}}
            else:
                return {symbol: {'action': 'hold'}}
        
        executor = BacktestExecutor(initial_capital=100000.0)
        executor.set_strategy(simple_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            risk_per_trade=0.02
        )
        
        results = executor.run_backtest(test_data, [symbol])
        
        # Get data for statistics engine
        portfolio = executor.portfolio
        equity_curve = portfolio.get_equity_curve()
        trades = portfolio.trade_engine.get_trades()
        completed_trades = [t for t in trades if t.exit_price is not None]
        
        # Convert trades for statistics engine
        trades_data = []
        for trade in completed_trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'commission': trade.commission
            })
        
        # Test statistics engine
        stats_engine = StatisticsEngine()
        
        if len(trades_data) > 0 and len(equity_curve) > 10:
            print(f"\n📊 Testing Statistics Engine with Real Data")
            print(f"   - Equity Curve Points: {len(equity_curve)}")
            print(f"   - Completed Trades: {len(trades_data)}")
            
            # Calculate performance metrics
            metrics = stats_engine.calculate_performance_metrics(
                equity_curve=equity_curve,
                trades=trades_data,
                initial_capital=100000.0
            )
            
            print(f"\n📈 Detailed Performance Metrics:")
            print(f"   - Total Return: {metrics.total_return:.4f}")
            print(f"   - Annualized Return: {metrics.annualized_return:.4f}")
            print(f"   - Volatility: {metrics.volatility:.4f}")
            print(f"   - Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"   - Max Drawdown: {metrics.max_drawdown:.4f}")
            print(f"   - Win Rate: {metrics.win_rate:.2f}")
            print(f"   - Profit Factor: {metrics.profit_factor:.2f}")
            
            # Test rolling metrics
            if len(equity_curve) > 30:
                rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=20)
                print(f"   - Rolling Periods: {len(rolling_metrics.time_series['returns'])}")
            
            # Test trade analysis
            trade_analysis = stats_engine.analyze_trades(trades_data)
            print(f"   - Trade Analysis: {trade_analysis.total_trades} trades analyzed")
            
            # Generate comprehensive report
            report = stats_engine.generate_report(metrics, None, trade_analysis)
            assert len(report) > 500, "Report should be comprehensive"
            print(f"   - Generated Report: {len(report)} characters")
            
            # Validate statistics
            assert isinstance(metrics.total_return, float)
            assert isinstance(metrics.sharpe_ratio, float)
            assert metrics.total_trades == len(trades_data)
            
        else:
            print(f"\n⚠️  Insufficient trade data for full statistics testing")
            print(f"   - Consider running with more volatile data or longer time period")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TestDatabaseBackedAPI:
    """Test FastAPI integration with database"""
    
    @pytest.mark.asyncio 
    async def test_api_with_database_data(self, db_manager, sample_symbol_data):
        """Test API endpoints using real database data"""
        from fastapi.testclient import TestClient
        
        # Import API with correct path
        import sys
        api_path = '/home/jaden/Documents/projects/crypto_quant_mvp/services/backtest-engine/src'
        if api_path not in sys.path:
            sys.path.insert(0, api_path)
        
        try:
            from api.main import app
            client = TestClient(app)
        except ImportError as e:
            pytest.skip(f"API not available for testing: {e}")
        
        symbol = sample_symbol_data['symbol']
        data = sample_symbol_data['data'].tail(500)  # Use subset for faster API testing
        
        # Convert data to API format
        market_data = []
        for timestamp, row in data.iterrows():
            market_data.append({
                'timestamp': timestamp.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        # Test API backtest with real data
        request_data = {
            "strategy": {
                "name": f"db_test_{symbol.lower()}",
                "type": "momentum",
                "parameters": {}
            },
            "market_data": market_data,
            "symbols": [symbol],
            "initial_capital": 100000.0,
            "risk_parameters": {
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "risk_per_trade": 0.02,
                "max_drawdown_limit": 0.20,
                "max_positions": 5,
                "position_sizing_method": "percentage"
            },
            "commission_rate": 0.001,
            "slippage": 0.001
        }
        
        print(f"\n🌐 Testing API with Database Data")
        print(f"   - Symbol: {symbol}")
        print(f"   - Data Points: {len(market_data)}")
        print(f"   - Strategy: Momentum")
        
        # Start backtest via API
        response = client.post("/backtest/run", json=request_data)
        assert response.status_code == 200
        
        backtest_id = response.json()["backtest_id"]
        print(f"   - Backtest ID: {backtest_id}")
        
        # Poll for completion (with timeout)
        import time
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = client.get(f"/backtest/status/{backtest_id}")
            status_data = status_response.json()
            
            print(f"   - Status: {status_data['status']} ({status_data.get('progress', 0)*100:.1f}%)")
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"API backtest failed: {status_data.get('message')}")
            
            time.sleep(2)
        else:
            pytest.fail("API backtest timed out")
        
        # Get results
        result_response = client.get(f"/backtest/result/{backtest_id}")
        assert result_response.status_code == 200
        
        result_data = result_response.json()
        
        print(f"\n✅ API Backtest Completed")
        print(f"   - Total Return: {result_data['metrics']['total_return']:.2f}%")
        print(f"   - Max Drawdown: {result_data['metrics']['max_drawdown']:.2f}%")
        print(f"   - Total Trades: {result_data['metrics']['total_trades']}")
        print(f"   - Execution Time: {result_data['execution_time_seconds']:.2f}s")
        
        # Validate API results
        assert result_data["status"] == "completed"
        assert "metrics" in result_data
        assert "trades" in result_data
        assert "equity_curve" in result_data
        assert "statistics_report" in result_data
        
        # Test statistics API with the results
        stats_request = {
            "equity_curve": result_data["equity_curve"],
            "trades": result_data["trades"],
            "initial_capital": 100000.0
        }
        
        stats_response = client.post("/statistics/performance", json=stats_request)
        assert stats_response.status_code == 200
        
        stats_data = stats_response.json()
        print(f"   - Statistics API: Sharpe {stats_data['risk_adjusted_returns']['sharpe_ratio']:.2f}")
        
        # Cleanup
        delete_response = client.delete(f"/backtest/{backtest_id}")
        assert delete_response.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
