"""
Database connection utilities for TimescaleDB integration

This module provides database connection management compatible with the
existing crypto_quant_mvp infrastructure while enabling custom backtest
engine integration.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager

# Database imports
import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class TimescaleDBConnector:
    """TimescaleDB connection manager for backtest engine integration"""
    
    def __init__(self, database_url: str = None):
        """Initialize TimescaleDB connector
        
        Args:
            database_url: Database connection string. If None, uses environment variables.
        """
        self.database_url = database_url or self._get_database_url()
        self.pool: Optional[AsyncConnectionPool] = None
        self._table_cache: Dict[str, bool] = {}
        
    def _get_database_url(self) -> str:
        """Get database URL from environment variables"""
        return os.getenv(
            'DATABASE_URL',
            'postgresql://quant_user:quant_password@timescaledb:5433/quant_db'
        )
    
    async def initialize(self):
        """Initialize async connection pool"""
        try:
            self.pool = AsyncConnectionPool(
                self.database_url,
                min_size=2,
                max_size=10,
                kwargs={'row_factory': dict_row}  # Return dict rows for easier handling
            )
            logger.info("✅ TimescaleDB connection pool initialized")
            
            # Test connection
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT version()")
                    version = await cur.fetchone()
                    logger.info(f"✅ Connected to: {version['version']}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB connection: {e}")
            raise
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ TimescaleDB connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.connection() as conn:
            yield conn
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        if table_name in self._table_cache:
            return self._table_cache[table_name]
            
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        )
                    """, (table_name,))
                    result = await cur.fetchone()
                    exists = result['exists']
                    self._table_cache[table_name] = exists
                    return exists
        except Exception as e:
            logger.error(f"❌ Error checking table {table_name}: {e}")
            return False
    
    async def get_available_tables(self) -> List[str]:
        """Get list of available OHLCV tables"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name LIKE '%ohlcv%'
                        ORDER BY table_name
                    """)
                    results = await cur.fetchall()
                    return [row['table_name'] for row in results]
        except Exception as e:
            logger.error(f"❌ Error getting available tables: {e}")
            return []
    
    async def get_available_symbols(self, table_name: str = None, limit: int = 10) -> List[str]:
        """Get available symbols from OHLCV tables"""
        if not table_name:
            tables = await self.get_available_tables()
            if not tables:
                return []
            table_name = tables[0]  # Use first available table
        
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"""
                        SELECT DISTINCT symbol 
                        FROM {table_name} 
                        ORDER BY symbol
                        LIMIT %s
                    """, (limit,))
                    results = await cur.fetchall()
                    return [row['symbol'] for row in results]
        except Exception as e:
            logger.error(f"❌ Error getting symbols from {table_name}: {e}")
            return []
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """Get OHLCV market data for backtesting
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe ('1min', '5min', '1h', '1d')
            start_time: Start datetime for data
            end_time: End datetime for data
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with OHLCV data indexed by time
        """
        table_name = f"ohlcv_{timeframe}"
        
        # Check if table exists
        if not await self.check_table_exists(table_name):
            # Try alternative table names
            alternative_tables = await self.get_available_tables()
            if alternative_tables:
                table_name = alternative_tables[0]
                logger.warning(f"⚠️ Table ohlcv_{timeframe} not found, using {table_name}")
            else:
                logger.error(f"❌ No OHLCV tables found")
                return pd.DataFrame()
        
        # Build query
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
        
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    results = await cur.fetchall()
                    
            if not results:
                logger.warning(f"⚠️ No data found for {symbol} in {table_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Convert to numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"✅ Retrieved {len(df)} records for {symbol} from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_data_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get information about available data"""
        info = {
            'available_tables': [],
            'total_records': 0,
            'symbols_count': 0,
            'earliest_data': None,
            'latest_data': None,
            'symbol_info': {}
        }
        
        try:
            # Get available tables
            tables = await self.get_available_tables()
            info['available_tables'] = tables
            
            if not tables:
                return info
            
            # Get data statistics from the first table
            table_name = tables[0]
            
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    # Overall statistics
                    query = f"""
                        SELECT 
                            COUNT(*) as total_records,
                            COUNT(DISTINCT symbol) as symbols_count,
                            MIN(time) as earliest_data,
                            MAX(time) as latest_data
                        FROM {table_name}
                    """
                    
                    if symbol:
                        query += " WHERE symbol = %s"
                        await cur.execute(query, (symbol,))
                    else:
                        await cur.execute(query)
                    
                    result = await cur.fetchone()
                    if result:
                        info['total_records'] = result['total_records']
                        info['symbols_count'] = result['symbols_count']
                        info['earliest_data'] = result['earliest_data']
                        info['latest_data'] = result['latest_data']
                    
                    # Per-symbol information
                    if symbol:
                        await cur.execute(f"""
                            SELECT 
                                symbol,
                                COUNT(*) as records,
                                MIN(time) as earliest,
                                MAX(time) as latest,
                                AVG(close) as avg_price,
                                MIN(close) as min_price,
                                MAX(close) as max_price
                            FROM {table_name}
                            WHERE symbol = %s
                            GROUP BY symbol
                        """, (symbol,))
                        
                        result = await cur.fetchone()
                        if result:
                            info['symbol_info'] = dict(result)
                    
        except Exception as e:
            logger.error(f"❌ Error getting data info: {e}")
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'healthy': False,
            'database_connected': False,
            'tables_available': False,
            'data_available': False,
            'error': None,
            'details': {}
        }
        
        try:
            # Test database connection
            async with self.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    health['database_connected'] = True
            
            # Check available tables
            tables = await self.get_available_tables()
            health['tables_available'] = len(tables) > 0
            health['details']['available_tables'] = tables
            
            # Check data availability
            if tables:
                data_info = await self.get_data_info()
                health['data_available'] = data_info['total_records'] > 0
                health['details']['data_info'] = data_info
            
            # Overall health status
            health['healthy'] = (
                health['database_connected'] and 
                health['tables_available'] and 
                health['data_available']
            )
            
        except Exception as e:
            health['error'] = str(e)
            logger.error(f"❌ Health check failed: {e}")
        
        return health


# Global connector instance
_connector: Optional[TimescaleDBConnector] = None


async def get_timescale_connector() -> TimescaleDBConnector:
    """Get singleton TimescaleDB connector"""
    global _connector
    
    if _connector is None:
        _connector = TimescaleDBConnector()
        await _connector.initialize()
    
    return _connector


async def close_timescale_connector():
    """Close global TimescaleDB connector"""
    global _connector
    
    if _connector:
        await _connector.close()
        _connector = None
