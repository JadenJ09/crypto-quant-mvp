#!/usr/bin/env python3
"""
Integration test for VectorBT Service

This script tests the complete integration of the VectorBT service
with TimescaleDB and verifies technical indicators are calculated correctly.
"""

import asyncio
import logging
import os
import psycopg
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://quant_user:quant_password@localhost:5433/quant_db')

async def test_database_connection():
    """Test basic database connection"""
    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT version();")
                result = await cur.fetchone()
                logger.info(f"✅ Database connection successful: {result[0][:50]}...")
                return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def test_ohlcv_tables():
    """Test that OHLCV tables exist and have data"""
    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            async with conn.cursor() as cur:
                # Check 1m table
                await cur.execute("SELECT COUNT(*) FROM ohlcv_1min;")
                count_1m = await cur.fetchone()
                logger.info(f"✅ ohlcv_1min table has {count_1m[0]} records")
                
                # Check symbols
                await cur.execute("SELECT DISTINCT symbol FROM ohlcv_1min ORDER BY symbol;")
                symbols = await cur.fetchall()
                logger.info(f"✅ Available symbols: {[s[0] for s in symbols]}")
                
                # Check timeframe tables
                timeframes = ['5min', '15min', '1hour', '4hour', '1day', '7day']
                for tf in timeframes:
                    await cur.execute(f"SELECT COUNT(*) FROM ohlcv_{tf};")
                    count = await cur.fetchone()
                    logger.info(f"✅ ohlcv_{tf} table has {count[0]} records")
                
                return len(symbols) > 0
    except Exception as e:
        logger.error(f"❌ Table check failed: {e}")
        return False

async def insert_test_data():
    """Insert test 1m OHLCV data for testing"""
    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            async with conn.cursor() as cur:
                # Generate test data for the last hour
                now = datetime.now().replace(second=0, microsecond=0)
                start_time = now - timedelta(hours=1)
                
                test_data = []
                current_time = start_time
                price = 50000.0  # Starting price
                
                while current_time <= now:
                    # Generate realistic OHLCV data
                    open_price = price
                    high_price = price * (1 + np.random.uniform(0, 0.002))
                    low_price = price * (1 - np.random.uniform(0, 0.002))
                    close_price = price * (1 + np.random.uniform(-0.001, 0.001))
                    volume = np.random.uniform(1000, 10000)
                    
                    test_data.append((
                        current_time, 'TESTUSDT', 
                        open_price, high_price, low_price, close_price, volume
                    ))
                    
                    price = close_price  # Next candle starts where this one ended
                    current_time += timedelta(minutes=1)
                
                # Insert test data
                await cur.executemany("""
                    INSERT INTO ohlcv_1min (time, symbol, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, time) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, test_data)
                
                await conn.commit()
                logger.info(f"✅ Inserted {len(test_data)} test records for TESTUSDT")
                return True
                
    except Exception as e:
        logger.error(f"❌ Test data insertion failed: {e}")
        return False

async def test_indicator_calculation():
    """Test that indicators can be calculated"""
    try:
        # Import vectorbt service components
        import sys
        sys.path.append('/app')  # Add app directory to path
        
        from app.config import Settings
        from app.database import DatabaseManager
        from app.indicators_processor import TechnicalIndicatorsProcessor
        
        # Initialize components
        settings = Settings()
        db_manager = DatabaseManager(DATABASE_URL)
        processor = TechnicalIndicatorsProcessor(settings)
        processor.set_db_manager(db_manager)
        
        await db_manager.initialize()
        
        # Test bulk processing for TESTUSDT
        logger.info("🧮 Testing indicator calculation...")
        await processor.process_symbol_bulk('TESTUSDT')
        
        # Check if indicators were calculated
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT time, symbol, close, rsi_14, ema_21, sma_20, macd_line
                    FROM ohlcv_5min 
                    WHERE symbol = 'TESTUSDT' 
                    ORDER BY time DESC 
                    LIMIT 5
                """)
                results = await cur.fetchall()
                
                if results:
                    logger.info("✅ Technical indicators calculated successfully:")
                    for row in results:
                        logger.info(f"   {row[0]} | Close: {row[2]:.2f} | RSI: {row[3]:.2f if row[3] else 'N/A'} | EMA21: {row[4]:.2f if row[4] else 'N/A'}")
                else:
                    logger.warning("⚠️  No indicator data found")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Indicator calculation test failed: {e}")
        return False

async def test_database_monitoring():
    """Test database monitoring functionality"""
    try:
        # Import database monitor
        import sys
        sys.path.append('/app')
        
        from app.config import Settings
        from app.database import DatabaseManager
        from app.database_monitor import DatabaseMonitor
        
        settings = Settings()
        db_manager = DatabaseManager(DATABASE_URL)
        monitor = DatabaseMonitor(settings, db_manager)
        
        await db_manager.initialize()
        
        # Test monitoring status
        logger.info("🔍 Testing database monitoring...")
        
        # Initialize monitoring
        await monitor._initialize_last_check()
        status = monitor.get_monitoring_status()
        
        logger.info(f"✅ Monitoring status: {status['symbols_monitored']} symbols")
        logger.info(f"✅ Polling interval: {status['polling_interval']}s")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database monitoring test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    logger.info("🚀 Starting VectorBT Service Integration Tests")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("OHLCV Tables", test_ohlcv_tables),
        ("Test Data Insertion", insert_test_data),
        ("Indicator Calculation", test_indicator_calculation),
        ("Database Monitoring", test_database_monitoring),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            logger.error(f"   ❌ FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🏁 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT service is ready.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
