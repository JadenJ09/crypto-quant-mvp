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

    # =============================
    # AUTOMATED PIPELINE METHODS
    # =============================
    
    async def check_database_health(self) -> Dict[str, Any]:
        """
        Comprehensive database health check for automated pipeline orchestration
        Returns health status with detailed metrics and processing recommendations
        """
        health_status = {
            'healthy': False,
            'total_1m_records': 0,
            'symbols_count': 0,
            'earliest_data': None,
            'latest_data': None,
            'has_sufficient_data': False,
            'needs_restore': False,
            'needs_gap_fill': False,
            'needs_incremental_processing': False,
            'needs_full_bulk_processing': False,
            'gap_periods': [],
            'processing_recommendation': 'none',
            'error': None
        }
        
        try:
            if not self.pool:
                raise Exception("Database pool not initialized")
                
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Check if ohlcv_1min table exists and has data
                    await cur.execute("""
                        SELECT COUNT(*) as total_records,
                               COUNT(DISTINCT symbol) as symbols_count,
                               MIN(time) as earliest_data,
                               MAX(time) as latest_data
                        FROM ohlcv_1min
                    """)
                    result = await cur.fetchone()
                    
                    if result:
                        health_status['total_1m_records'] = result[0]
                        health_status['symbols_count'] = result[1]
                        health_status['earliest_data'] = result[2]
                        health_status['latest_data'] = result[3]
                        
                    # Check for sufficient data (at least 100K records for meaningful analysis)
                    health_status['has_sufficient_data'] = health_status['total_1m_records'] >= 100000
                    
                    # Check if database is completely empty (needs restoration)
                    health_status['needs_restore'] = health_status['total_1m_records'] == 0
                    
                    if health_status['total_1m_records'] > 0:
                        # Detect gaps and determine processing strategy
                        gap_analysis = await self._analyze_processing_gaps()
                        health_status.update(gap_analysis)
                        
                        # Determine processing recommendation
                        health_status['processing_recommendation'] = self._determine_processing_strategy(health_status)
                    
                    # Overall health assessment
                    health_status['healthy'] = (
                        health_status['total_1m_records'] > 0 and
                        not health_status['needs_restore']
                    )
                    
        except Exception as e:
            health_status['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
            
        return health_status
    
    async def _analyze_processing_gaps(self) -> Dict[str, Any]:
        """
        Analyze gaps between 1m data and processed multi-timeframe data
        Returns detailed gap analysis for intelligent processing decisions
        """
        gap_analysis = {
            'needs_gap_fill': False,
            'needs_incremental_processing': False,
            'needs_full_bulk_processing': False,
            'gap_periods': [],
            'timeframe_coverage': {},
            'total_gap_hours': 0
        }
        
        try:
            if not self.pool:
                raise Exception("Database pool not initialized")
                
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Get 1m data coverage
                    await cur.execute("""
                        SELECT 
                            MIN(time) as earliest_1m,
                            MAX(time) as latest_1m,
                            COUNT(*) as total_1m_records
                        FROM ohlcv_1min
                    """)
                    base_coverage = await cur.fetchone()
                    
                    if not base_coverage or base_coverage[2] == 0:
                        return gap_analysis
                    
                    earliest_1m, latest_1m, total_1m = base_coverage
                    
                    # Check each timeframe for coverage gaps
                    timeframes = {
                        '5min': 5, '15min': 15, '1hour': 60, 
                        '4hour': 240, '1day': 1440, '7day': 10080
                    }
                    
                    total_gap_minutes = 0
                    significant_gaps = []
                    
                    for tf_name, tf_minutes in timeframes.items():
                        table_name = f'ohlcv_{tf_name}'
                        
                        # Check if table exists and has data
                        await cur.execute(f"""
                            SELECT 
                                COUNT(*) as tf_records,
                                MIN(time) as earliest_tf,
                                MAX(time) as latest_tf,
                                COUNT(CASE WHEN rsi_14 IS NOT NULL THEN 1 END) as indicator_records
                            FROM {table_name}
                        """)
                        tf_result = await cur.fetchone()
                        
                        if tf_result:
                            tf_records, earliest_tf, latest_tf, indicator_records = tf_result
                            
                            # Calculate expected records based on 1m data
                            if earliest_1m and latest_1m:
                                time_span_minutes = (latest_1m - earliest_1m).total_seconds() / 60
                                expected_records = max(1, int(time_span_minutes / tf_minutes))
                                coverage_ratio = tf_records / expected_records if expected_records > 0 else 0
                                indicator_ratio = indicator_records / tf_records if tf_records > 0 else 0
                            else:
                                coverage_ratio = 0
                                indicator_ratio = 0
                            
                            gap_analysis['timeframe_coverage'][tf_name] = {
                                'records': tf_records,
                                'expected_records': expected_records,
                                'coverage_ratio': coverage_ratio,
                                'indicator_records': indicator_records,
                                'indicator_ratio': indicator_ratio,
                                'earliest': earliest_tf,
                                'latest': latest_tf
                            }
                            
                            # Identify significant gaps (less than 70% coverage)
                            if coverage_ratio < 0.7:
                                gap_size_minutes = (expected_records - tf_records) * tf_minutes
                                total_gap_minutes += gap_size_minutes
                                significant_gaps.append({
                                    'timeframe': tf_name,
                                    'gap_size_hours': gap_size_minutes / 60,
                                    'coverage_ratio': coverage_ratio
                                })
                    
                    # Check for recent 1m data gaps (for data-recovery needs)
                    await cur.execute("""
                        WITH time_gaps AS (
                            SELECT 
                                symbol,
                                time,
                                LAG(time) OVER (PARTITION BY symbol ORDER BY time) as prev_time,
                                EXTRACT(EPOCH FROM (time - LAG(time) OVER (PARTITION BY symbol ORDER BY time)))/60 as gap_minutes
                            FROM ohlcv_1min 
                            WHERE time >= NOW() - INTERVAL '7 days'
                        )
                        SELECT 
                            COUNT(*) as gap_count,
                            COALESCE(SUM(gap_minutes - 1), 0) as total_gap_minutes
                        FROM time_gaps
                        WHERE gap_minutes > 5
                    """)
                    gap_result = await cur.fetchone()
                    
                    if gap_result:
                        recent_gap_count, recent_gap_minutes = gap_result
                        gap_analysis['needs_gap_fill'] = recent_gap_count > 10
                        
                    # Set processing recommendations
                    gap_analysis['gap_periods'] = significant_gaps
                    gap_analysis['total_gap_hours'] = total_gap_minutes / 60
                    
                    # Decision logic for processing strategy
                    if gap_analysis['total_gap_hours'] > 168:  # More than 1 week of gaps
                        gap_analysis['needs_full_bulk_processing'] = True
                    elif gap_analysis['total_gap_hours'] > 0:
                        gap_analysis['needs_incremental_processing'] = True
                        
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            gap_analysis['error'] = str(e)
            
        return gap_analysis
    
    def _determine_processing_strategy(self, health_status: Dict[str, Any]) -> str:
        """Determine the optimal processing strategy based on health status"""
        
        if health_status['needs_restore']:
            return 'restore_from_backup'
        elif health_status['needs_gap_fill']:
            return 'gap_fill_then_incremental'
        elif health_status['needs_full_bulk_processing']:
            return 'full_bulk_processing'
        elif health_status['needs_incremental_processing']:
            return 'incremental_processing'
        elif health_status['healthy']:
            return 'realtime_only'
        else:
            return 'investigate_manually'
    
    async def get_incremental_processing_periods(self, max_period_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get specific time periods that need incremental processing
        Breaks down large gaps into manageable chunks
        """
        processing_periods = []
        
        try:
            if not self.pool:
                raise Exception("Database pool not initialized")
                
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Find periods where 1m data exists but multi-timeframe data is missing/incomplete
                    symbols = await self.get_available_symbols()
                    
                    for symbol in symbols:
                        # Get 1m data range for this symbol
                        await cur.execute("""
                            SELECT MIN(time), MAX(time)
                            FROM ohlcv_1min
                            WHERE symbol = %s
                        """, [symbol])
                        result = await cur.fetchone()
                        
                        if not result or not result[0]:
                            continue
                            
                        min_1m, max_1m = result
                        
                        # Check 5min data coverage (most important timeframe)
                        await cur.execute("""
                            SELECT MIN(time), MAX(time)
                            FROM ohlcv_5min
                            WHERE symbol = %s AND rsi_14 IS NOT NULL
                        """, [symbol])
                        tf_result = await cur.fetchone()
                        
                        if tf_result and tf_result[0]:
                            min_tf, max_tf = tf_result
                            
                            # Find gaps at the beginning
                            if min_1m < min_tf:
                                gap_start = min_1m
                                gap_end = min(min_tf, min_1m + timedelta(hours=max_period_hours))
                                processing_periods.append({
                                    'symbol': symbol,
                                    'start_time': gap_start,
                                    'end_time': gap_end,
                                    'type': 'beginning_gap',
                                    'estimated_hours': (gap_end - gap_start).total_seconds() / 3600
                                })
                            
                            # Find gaps at the end
                            if max_1m > max_tf:
                                gap_start = max(max_tf, max_1m - timedelta(hours=max_period_hours))
                                gap_end = max_1m
                                processing_periods.append({
                                    'symbol': symbol,
                                    'start_time': gap_start,
                                    'end_time': gap_end,
                                    'type': 'recent_gap',
                                    'estimated_hours': (gap_end - gap_start).total_seconds() / 3600
                                })
                        else:
                            # No processed data at all - need full processing (but chunked)
                            current_start = min_1m
                            while current_start < max_1m:
                                current_end = min(current_start + timedelta(hours=max_period_hours), max_1m)
                                processing_periods.append({
                                    'symbol': symbol,
                                    'start_time': current_start,
                                    'end_time': current_end,
                                    'type': 'full_processing_chunk',
                                    'estimated_hours': (current_end - current_start).total_seconds() / 3600
                                })
                                current_start = current_end
                                
        except Exception as e:
            logger.error(f"Failed to get incremental processing periods: {e}")
            
        # Sort by estimated processing time (smaller chunks first)
        processing_periods.sort(key=lambda x: x['estimated_hours'])
        
        return processing_periods
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics for monitoring dashboard"""
        stats = {
            'timeframes': {},
            'symbols': [],
            'processing_progress': {},
            'recommendations': []
        }
        
        try:
            if not self.pool:
                raise Exception("Database pool not initialized")
                
            symbols = await self.get_available_symbols()
            stats['symbols'] = symbols
            
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Get stats for each timeframe
                    timeframes = ['1min', '5min', '15min', '1hour', '4hour', '1day', '7day']
                    
                    for tf in timeframes:
                        table_name = f'ohlcv_{tf}'
                        await cur.execute(f"""
                            SELECT 
                                COUNT(*) as total_records,
                                COUNT(DISTINCT symbol) as symbols_count,
                                MIN(time) as earliest,
                                MAX(time) as latest,
                                COUNT(CASE WHEN rsi_14 IS NOT NULL THEN 1 END) as indicator_records
                            FROM {table_name}
                        """)
                        result = await cur.fetchone()
                        
                        if result:
                            total_records, symbols_count, earliest, latest, indicator_records = result
                            indicator_coverage = (indicator_records / total_records * 100) if total_records > 0 else 0
                            
                            stats['timeframes'][tf] = {
                                'total_records': total_records,
                                'symbols_count': symbols_count,
                                'earliest': earliest,
                                'latest': latest,
                                'indicator_records': indicator_records,
                                'indicator_coverage_pct': round(indicator_coverage, 2)
                            }
                    
                    # Calculate processing progress
                    base_records = stats['timeframes'].get('1min', {}).get('total_records', 0)
                    if base_records > 0:
                        for tf in ['5min', '15min', '1hour']:
                            tf_records = stats['timeframes'].get(tf, {}).get('total_records', 0)
                            expected_ratio = {'5min': 0.2, '15min': 0.067, '1hour': 0.017}[tf]
                            actual_ratio = tf_records / base_records
                            progress_pct = min(100, (actual_ratio / expected_ratio) * 100)
                            
                            stats['processing_progress'][tf] = {
                                'expected_records': int(base_records * expected_ratio),
                                'actual_records': tf_records,
                                'progress_pct': round(progress_pct, 2)
                            }
                            
                            # Add recommendations
                            if progress_pct < 80:
                                stats['recommendations'].append({
                                    'type': 'processing_lag',
                                    'timeframe': tf,
                                    'message': f'{tf} processing is {progress_pct:.1f}% complete',
                                    'action': 'incremental_processing'
                                })
                                
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            stats['error'] = str(e)
            
        return stats
    
    async def wait_for_database_ready(self, timeout: int = 300) -> bool:
        """
        Wait for database to be ready and healthy
        Returns True if database becomes ready within timeout
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                if not self.pool:
                    logger.debug("Database pool not initialized yet")
                    await asyncio.sleep(10)
                    continue
                    
                # Simple connection test
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT 1")
                        result = await cur.fetchone()
                        
                        if result and result[0] == 1:
                            logger.info("✅ Database is ready and responding")
                            return True
                            
            except Exception as e:
                logger.debug(f"Database not ready: {e}")
                
            await asyncio.sleep(10)  # Check every 10 seconds
            
        logger.error(f"❌ Database did not become ready within {timeout} seconds")
        return False
