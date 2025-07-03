#!/usr/bin/env python3
"""
Comprehensive TimescaleDB Integration Test for Custom Backtest Engine

This script tests the complete integration between our custom backtest engine 
and the TimescaleDB infrastructure using real market data.

Usage:
    python test_timescale_integration.py
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Database imports
import psycopg
from psycopg_pool import AsyncConnectionPool

# Our custom engine imports
from core.backtest_executor import BacktestExecutor
from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from risk.risk_manager import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from statistics.statistics_engine import StatisticsEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimescaleDBIntegrationTester:
    """Comprehensive integration tester for TimescaleDB and custom backtest engine"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: AsyncConnectionPool = None
        self.api_base_url = "http://localhost:8003"
        
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
            logger.error(f"❌ Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Database connection pool closed")
    
    async def check_timescale_health(self) -> Dict[str, Any]:
        """Check TimescaleDB health and data availability"""
        health_status = {
            'healthy': False,
            'total_records': 0,
            'symbols_count': 0,
            'earliest_data': None,
            'latest_data': None,
            'available_tables': [],
            'error': None
        }
        
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Check available tables
                    await cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name LIKE '%ohlcv%'
                        ORDER BY table_name
                    """)
                    tables = await cur.fetchall()
                    health_status['available_tables'] = [table[0] for table in tables]
                    
                    # Check main ohlcv table data
                    if tables:
                        main_table = tables[0][0]  # Use first available table
                        await cur.execute(f"""
                            SELECT 
                                COUNT(*) as total_records,
                                COUNT(DISTINCT symbol) as symbols_count,
                                MIN(time) as earliest_data,
                                MAX(time) as latest_data
                            FROM {main_table}
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
            logger.error(f"❌ Database health check failed: {e}")
        
        return health_status
    
    async def get_available_symbols(self, limit: int = 5) -> List[str]:
        """Get available symbols from TimescaleDB"""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Try different possible table names
                    table_names = ['ohlcv_1min', 'ohlcv_5min', 'ohlcv_1h', 'ohlcv']
                    
                    for table_name in table_names:
                        try:
                            await cur.execute(f"""
                                SELECT DISTINCT symbol 
                                FROM {table_name} 
                                ORDER BY symbol
                                LIMIT %s
                            """, (limit,))
                            result = await cur.fetchall()
                            if result:
                                symbols = [row[0] for row in result]
                                logger.info(f"✅ Found {len(symbols)} symbols in {table_name}")
                                return symbols
                        except Exception as e:
                            logger.debug(f"Table {table_name} not found or inaccessible: {e}")
                            continue
                    
                    logger.warning("❌ No accessible OHLCV tables found")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Failed to get symbols: {e}")
            return []
    
    async def get_market_data(
        self, 
        symbol: str, 
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get OHLCV data for backtesting"""
        
        # Try different table names
        table_names = ['ohlcv_1min', 'ohlcv_5min', 'ohlcv_1h', 'ohlcv']
        
        for table_name in table_names:
            try:
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
                    FROM {table_name}
                    WHERE {where_clause}
                    ORDER BY time ASC
                    {limit_clause}
                """
                
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params)
                        rows = await cur.fetchall()
                        
                if rows:
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    
                    # Convert to float for calculations
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    logger.info(f"✅ Retrieved {len(df)} records from {table_name} for {symbol}")
                    return df
                    
            except Exception as e:
                logger.debug(f"Failed to get data from {table_name}: {e}")
                continue
        
        logger.warning(f"❌ No data found for {symbol} in any table")
        return pd.DataFrame()
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check if backtest engine API is healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/health/", timeout=10)
            if response.status_code == 200:
                return {"healthy": True, "data": response.json()}
            else:
                return {"healthy": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def test_backtest_engine_integration(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Test the custom backtest engine with real TimescaleDB data"""
        
        if data.empty:
            return {"success": False, "error": "No data provided"}
        
        try:
            # Define a simple strategy
            def simple_momentum_strategy(data_slice, timestamp):
                if len(data_slice) < 20:
                    return {symbol: {'action': 'hold'}}
                
                # Simple moving average crossover
                sma_short = data_slice['close'].rolling(5).mean()
                sma_long = data_slice['close'].rolling(20).mean()
                
                if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
                    return {symbol: {'action': 'buy', 'side': 'long'}}
                elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
                    return {symbol: {'action': 'sell', 'side': 'short'}}
                else:
                    return {symbol: {'action': 'hold'}}
            
            # Create backtest executor
            executor = BacktestExecutor(initial_capital=100000.0)
            executor.set_strategy(simple_momentum_strategy)
            executor.set_risk_parameters(
                stop_loss_pct=0.02,  # 2% stop loss
                take_profit_pct=0.04,  # 4% take profit
                risk_per_trade=0.01  # 1% risk per trade
            )
            
            # Run backtest
            results = executor.run_backtest(data, [symbol])
            
            # Get statistics
            stats_engine = StatisticsEngine(results['portfolio_history'])
            performance_stats = stats_engine.calculate_performance_metrics()
            
            return {
                "success": True,
                "data_points": len(data),
                "trades_executed": len(results['trades']),
                "final_capital": results['final_capital'],
                "total_return": results['metrics']['total_return_pct'],
                "max_drawdown": performance_stats.get('max_drawdown', 0),
                "sharpe_ratio": performance_stats.get('sharpe_ratio', 0),
                "win_rate": performance_stats.get('win_rate', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self):
        """Run comprehensive integration test"""
        logger.info("🚀 Starting comprehensive TimescaleDB integration test")
        
        # Test 1: Database Health Check
        logger.info("📊 Testing database health...")
        health = await self.check_timescale_health()
        
        if not health['healthy']:
            logger.error(f"❌ Database unhealthy: {health['error']}")
            return False
        
        logger.info(f"✅ Database healthy: {health['total_records']} records, {health['symbols_count']} symbols")
        logger.info(f"📅 Data range: {health['earliest_data']} to {health['latest_data']}")
        logger.info(f"🗃️ Available tables: {', '.join(health['available_tables'])}")
        
        # Test 2: API Health Check
        logger.info("🔗 Testing API health...")
        api_health = self.check_api_health()
        
        if not api_health['healthy']:
            logger.warning(f"⚠️ API not healthy: {api_health['error']}")
        else:
            logger.info("✅ API healthy")
        
        # Test 3: Get Available Symbols
        logger.info("📈 Getting available symbols...")
        symbols = await self.get_available_symbols(limit=3)
        
        if not symbols:
            logger.error("❌ No symbols found")
            return False
        
        logger.info(f"✅ Found symbols: {', '.join(symbols)}")
        
        # Test 4: Run Backtest Integration for Each Symbol
        for symbol in symbols:
            logger.info(f"🔄 Testing backtest integration for {symbol}...")
            
            # Get recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            data = await self.get_market_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if data.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                continue
            
            # Run backtest
            result = await self.test_backtest_engine_integration(symbol, data)
            
            if result['success']:
                logger.info(f"✅ Backtest successful for {symbol}:")
                logger.info(f"   📊 Data points: {result['data_points']}")
                logger.info(f"   💰 Final capital: ${result['final_capital']:,.2f}")
                logger.info(f"   📈 Total return: {result['total_return']:.2f}%")
                logger.info(f"   🎯 Trades executed: {result['trades_executed']}")
                logger.info(f"   📉 Max drawdown: {result['max_drawdown']:.2f}%")
                logger.info(f"   ⚡ Sharpe ratio: {result['sharpe_ratio']:.2f}")
                logger.info(f"   🏆 Win rate: {result['win_rate']:.2f}%")
            else:
                logger.error(f"❌ Backtest failed for {symbol}: {result['error']}")
        
        logger.info("🎉 Comprehensive integration test completed!")
        return True


async def main():
    """Main test function"""
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL', 'postgresql://quant_user:quant_password@localhost:5433/quant_db')
    
    logger.info(f"🔗 Connecting to database: {database_url}")
    
    # Create tester
    tester = TimescaleDBIntegrationTester(database_url)
    
    try:
        # Initialize database connection
        await tester.initialize()
        
        # Run comprehensive test
        success = await tester.run_comprehensive_test()
        
        if success:
            logger.info("✅ All tests passed! Your custom backtest engine is ready for production.")
        else:
            logger.error("❌ Some tests failed. Check the logs above.")
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        
    finally:
        # Clean up
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
