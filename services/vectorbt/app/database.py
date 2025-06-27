"""
Database management for VectorBT Service
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import psycopg
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[AsyncConnectionPool] = None
        
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
                """)
                result = await cur.fetchall()
                return [row[0] for row in result]
                
    async def get_1min_data(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get 1-minute OHLCV data for a symbol"""
        
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
            SELECT time, symbol, open, high, low, close, volume
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
        df = pd.DataFrame(rows, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        return df
        
    async def upsert_timeframe_data(
        self, 
        table_name: str,
        data: List[Dict[str, Any]]
    ) -> int:
        """Upsert OHLCV data with technical indicators to timeframe table"""
        
        if not data:
            return 0
            
        # Build column list dynamically based on first record
        columns = list(data[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        
        # Build upsert query
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (symbol, time) DO UPDATE SET
            {', '.join([f'{col} = EXCLUDED.{col}' for col in columns if col not in ['symbol', 'time']])}
        """
        
        # Prepare data for bulk insert
        values = []
        for record in data:
            values.append([record[col] for col in columns])
            
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, values)
                return len(values)
                
    async def get_latest_timeframe_data(
        self,
        table_name: str,
        symbol: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """Get latest data from a timeframe table for indicator calculation"""
        
        query = f"""
            SELECT *
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT %s
        """
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, [symbol, limit])
                rows = await cur.fetchall()
                
                # Get column names
                columns = [desc[0] for desc in cur.description]
                
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)  # Sort ascending for indicators
            
        return df
        
    async def get_timeframe_data_range(
        self,
        table_name: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get timeframe data for a specific time range"""
        
        query = f"""
            SELECT *
            FROM {table_name}
            WHERE symbol = %s 
            AND time >= %s 
            AND time <= %s
            ORDER BY time ASC
        """
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, [symbol, start_time, end_time])
                rows = await cur.fetchall()
                
                # Get column names
                columns = [desc[0] for desc in cur.description]
                
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        return df
        
    async def check_data_exists(self, table_name: str, symbol: str) -> bool:
        """Check if data exists for symbol in table"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT 1 FROM {table_name} 
                    WHERE symbol = %s 
                    LIMIT 1
                """, [symbol])
                result = await cur.fetchone()
                return result is not None
            
    async def get_latest_1m_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the latest timestamp for a symbol in ohlcv_1min table"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT time 
                    FROM ohlcv_1min 
                    WHERE symbol = %s 
                    ORDER BY time DESC 
                    LIMIT 1
                """, [symbol])
                result = await cur.fetchone()
                return result[0] if result else None
                
    async def get_1m_data_since(
        self, 
        symbol: str, 
        since_time: datetime
    ) -> pd.DataFrame:
        """Get 1m OHLCV data for a symbol since a specific timestamp"""
        
        query = """
            SELECT time, symbol, open, high, low, close, volume
            FROM ohlcv_1min
            WHERE symbol = %s AND time > %s
            ORDER BY time ASC
        """
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, [symbol, since_time])
                rows = await cur.fetchall()
                
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        return df
