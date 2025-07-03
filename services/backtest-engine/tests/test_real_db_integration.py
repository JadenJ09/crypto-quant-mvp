"""
Working Database Integration Test

Tests our custom backtest engine with real TimescaleDB market data.
This is a simplified test that focuses on actual functionality.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Database imports
import psycopg
from psycopg_pool import AsyncConnectionPool

# Our custom engine imports  
import sys
sys.path.append('/home/jaden/Documents/projects/crypto_quant_mvp/services/backtest-engine/src')

from core.backtest_executor import BacktestExecutor
from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from risk.risk_manager import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from statistics.statistics_engine import StatisticsEngine

logger = logging.getLogger(__name__)


class SimpleDBManager:
    """Simplified database manager for testing"""
    
    def __init__(self):
        self.database_url = 'postgresql://quant_user:quant_password@localhost:5433/quant_db'
        self.pool = None
        
    async def connect(self):
        """Connect to database"""
        try:
            self.pool = AsyncConnectionPool(
                self.database_url,
                min_size=1,
                max_size=3
            )
            print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
            
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
            print("✅ Database connection closed")
            
    async def get_symbols(self):
        """Get available symbols"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT DISTINCT symbol 
                    FROM ohlcv_1min 
                    ORDER BY symbol 
                    LIMIT 10
                """)
                symbols = await cur.fetchall()
                return [s[0] for s in symbols]
                
    async def get_market_data(self, symbol: str, hours: int = 168):
        """Get recent market data for a symbol"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT time, open, high, low, close, volume
                    FROM ohlcv_1min 
                    WHERE symbol = %s 
                    AND time >= NOW() - INTERVAL '%s hours'
                    ORDER BY time DESC
                    LIMIT 2000
                """, (symbol, hours))
                
                rows = await cur.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df


def create_momentum_strategy():
    """Create a simple momentum strategy"""
    def momentum_strategy(data, timestamp):
        if len(data) < 20:
            return {'action': 'hold'}
        
        # Simple momentum with SMA
        sma_short = data['close'].rolling(5).mean().iloc[-1]
        sma_long = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if pd.isna(sma_short) or pd.isna(sma_long):
            return {'action': 'hold'}
        
        # Buy signal: short SMA above long SMA and price above short SMA
        if sma_short > sma_long and current_price > sma_short:
            return {'action': 'buy', 'side': 'long'}
        # Sell signal: opposite condition
        elif sma_short < sma_long:
            return {'action': 'sell', 'side': 'short'}
        else:
            return {'action': 'hold'}
    
    return momentum_strategy


async def test_database_connection():
    """Test 1: Basic database connectivity"""
    print("\n🔍 Test 1: Database Connection")
    print("=" * 50)
    
    db = SimpleDBManager()
    
    try:
        await db.connect()
        
        # Get symbols
        symbols = await db.get_symbols()
        print(f"✅ Available symbols: {symbols[:5]}...")
        
        # Test data retrieval
        if symbols:
            symbol = symbols[0]
            data = await db.get_market_data(symbol, hours=24)
            print(f"✅ Retrieved {len(data)} records for {symbol}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        await db.close()
        return False


async def test_engine_with_real_data():
    """Test 2: Run backtest with real market data"""
    print("\n🚀 Test 2: Engine with Real Data")
    print("=" * 50)
    
    db = SimpleDBManager()
    
    try:
        await db.connect()
        
        # Get symbols and data
        symbols = await db.get_symbols()
        symbol = symbols[0] if symbols else 'BTCUSD'
        
        print(f"📊 Testing with symbol: {symbol}")
        
        # Get 1 week of data
        data = await db.get_market_data(symbol, hours=168)
        
        if data.empty:
            print(f"❌ No data available for {symbol}")
            return False
            
        print(f"✅ Data loaded: {len(data)} records")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Create and run backtest
        executor = BacktestExecutor(initial_capital=100000.0)
        executor.set_strategy(create_momentum_strategy())
        executor.set_risk_parameters(
            stop_loss_pct=0.02,      # 2% stop loss
            take_profit_pct=0.05,    # 5% take profit  
            risk_per_trade=0.01      # 1% risk per trade
        )
        
        print("🔄 Running backtest...")
        results = executor.run_backtest(data, [symbol])
        
        # Display results
        metrics = results['metrics']
        print(f"\n📈 BACKTEST RESULTS:")
        print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"   Number of Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Get portfolio statistics
        portfolio = executor.portfolio
        equity_curve = portfolio.get_equity_curve()
        trades = portfolio.trade_engine.get_trades()
        
        print(f"   Equity Curve Length: {len(equity_curve)}")
        print(f"   Final Equity: ${equity_curve.iloc[-1]:,.2f}")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        await db.close()
        return False


async def test_statistics_with_real_data():
    """Test 3: Statistics engine with real trading results"""
    print("\n📊 Test 3: Statistics Engine")
    print("=" * 50)
    
    db = SimpleDBManager()
    
    try:
        await db.connect()
        
        # Get data and run backtest
        symbols = await db.get_symbols()
        symbol = symbols[0] if symbols else 'BTCUSD'
        data = await db.get_market_data(symbol, hours=168)
        
        if data.empty:
            print(f"❌ No data for statistics test")
            return False
        
        # Run backtest with more trades
        executor = BacktestExecutor(initial_capital=50000.0)
        executor.set_strategy(create_momentum_strategy())
        executor.set_risk_parameters(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_per_trade=0.02
        )
        
        results = executor.run_backtest(data, [symbol])
        
        # Get trading data
        portfolio = executor.portfolio
        equity_curve = portfolio.get_equity_curve()
        all_trades = portfolio.trade_engine.get_trades()
        completed_trades = [t for t in all_trades if t.exit_price is not None]
        
        print(f"✅ Generated {len(completed_trades)} completed trades")
        
        if len(completed_trades) > 0:
            # Prepare trades for statistics engine
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
            
            # Create statistics engine
            stats_engine = StatisticsEngine()
            
            # Calculate performance metrics
            perf_metrics = stats_engine.calculate_performance_metrics(
                equity_curve=equity_curve,
                trades=trades_data,
                initial_capital=50000.0
            )
            
            print(f"\n📈 ADVANCED STATISTICS:")
            print(f"   Sharpe Ratio: {perf_metrics.sharpe_ratio:.3f}")
            print(f"   Sortino Ratio: {perf_metrics.sortino_ratio:.3f}")
            print(f"   Calmar Ratio: {perf_metrics.calmar_ratio:.3f}")
            print(f"   Max Drawdown: {perf_metrics.max_drawdown:.3f}")
            print(f"   VaR (95%): {perf_metrics.var_95:.3f}")
            
            # Trade analysis
            trade_analysis = stats_engine.analyze_trades(trades_data)
            print(f"   Average Hold Time: {trade_analysis.average_hold_time}")
            print(f"   Profit Factor: {trade_analysis.profit_factor:.3f}")
            
            # Generate comprehensive report
            rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=50)
            report = stats_engine.generate_report(perf_metrics, rolling_metrics, trade_analysis)
            
            print(f"\n📋 Report Generated: {len(report)} characters")
            
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        await db.close()
        return False


async def run_all_tests():
    """Run all database integration tests"""
    print("🧪 CUSTOM BACKTEST ENGINE - DATABASE INTEGRATION TESTS")
    print("=" * 60)
    print(f"📅 Running on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🗄️  Database: TimescaleDB (localhost:5433)")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Engine with Real Data", test_engine_with_real_data),
        ("Statistics Engine", test_statistics_with_real_data),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS SUCCESSFUL! Custom engine works with TimescaleDB!")
    else:
        print("⚠️  Some tests failed. Check output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
