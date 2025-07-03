#!/usr/bin/env python3
"""
High-Performance Bulk Historical Data Loader
Optimized for maximum speed bulk loading of historical crypto data.
Direct API → Database bypass for initial data collection.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceClient:
    """Optimized Binance API client for bulk data fetching."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        
        # Rate limiting: 1200 requests/minute = 20 requests/second
        self.requests_per_second = 18  # Conservative limit
        self.last_request_time = 0
        
        if api_key:
            self.session.headers.update({'X-MBX-APIKEY': api_key})
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits."""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_klines_batch(self, symbol: str, interval: str, start_time: int, 
                        end_time: int, limit: int = 1000) -> List[List]:
        """
        Fetch a batch of klines with optimal parameters and robust error handling.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '1h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max records per request (max 1000)
        
        Returns:
            List of kline data
        """
        self._rate_limit()
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v3/klines",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                klines = response.json()
                
                start_dt = datetime.fromtimestamp(start_time/1000)
                end_dt = datetime.fromtimestamp(end_time/1000)
                
                logger.info(f"✅ Fetched {len(klines)} klines for {symbol} "
                           f"({start_dt.strftime('%Y-%m-%d %H:%M')} to "
                           f"{end_dt.strftime('%Y-%m-%d %H:%M')})")
                
                return klines
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"⚠️  API error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Final API error for {symbol} after {max_retries} attempts: {e}")
                    raise
    
    def get_symbol_complete_history(self, symbol: str, start_date: str, 
                                  end_date: str, interval: str = '1m') -> List[Dict]:
        """
        Fetch complete history for a single symbol with deduplication.
        
        This is faster than fetching all symbols for each timeframe because:
        1. Better API rate limit utilization per symbol
        2. More efficient database bulk inserts
        3. Better error recovery (per symbol)
        4. Linear progress tracking
        """
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        
        all_records = []
        seen_timestamps = set()  # Track timestamps to prevent duplicates
        current_time = start_dt
        
        # Process in chunks to stay within API limits
        # For 1m data: 1000 records = ~16.67 hours
        chunk_duration = timedelta(hours=16)  # Conservative chunk size
        
        while current_time < end_dt:
            chunk_end = min(current_time + chunk_duration, end_dt)
            
            start_ms = int(current_time.timestamp() * 1000)
            end_ms = int(chunk_end.timestamp() * 1000)
            
            try:
                klines = self.get_klines_batch(symbol, interval, start_ms, end_ms)
                
                # Convert to standardized format with deduplication
                for kline in klines:
                    timestamp = kline[0]  # Timestamp in milliseconds
                    
                    # Skip duplicates
                    if timestamp in seen_timestamps:
                        continue
                    
                    seen_timestamps.add(timestamp)
                    
                    record = {
                        'time': datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
                        'symbol': symbol,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    }
                    all_records.append(record)
                
                current_time = chunk_end
                
                # Progress logging
                progress = ((current_time - start_dt).total_seconds() / 
                           (end_dt - start_dt).total_seconds()) * 100
                logger.info(f"Progress for {symbol}: {progress:.1f}% "
                           f"({len(all_records)} unique records)")
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk for {symbol}: {e}")
                # Skip this chunk and continue
                current_time = chunk_end
                continue
        
        logger.info(f"Completed {symbol}: {len(all_records)} total records (deduplicated)")
        return all_records


class BulkDatabaseWriter:
    """Optimized database writer for high-performance bulk inserts."""
    
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        # Create connection pool for optimal performance
        self.pool = ConnectionPool(connection_url, min_size=1, max_size=5)
    
    def cleanup(self):
        """Properly close the database connection pool."""
        if hasattr(self, 'pool') and self.pool:
            logger.info("🔒 Closing database connection pool...")
            self.pool.close()
            logger.info("✅ Database connections closed")

    def bulk_insert_symbol_data(self, records: List[Dict], batch_size: int = 5000):
        """
        Bulk insert data for a symbol using high-performance COPY.
        Updated to use psycopg3 connection pool and COPY API.
        
        Args:
            records: List of OHLCV records
            batch_size: Records per batch for memory management
        """
        if not records:
            return
        
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    # Process in batches to manage memory
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        
                        # Prepare data for bulk insert using COPY
                        data_rows = []
                        for record in batch:
                            data_rows.append((
                                record['time'],
                                record['symbol'],
                                record['open'],
                                record['high'],
                                record['low'],
                                record['close'],
                                record['volume']
                            ))
                        
                        # Use COPY for high performance bulk insert with ON CONFLICT handling
                        # Create a unique temporary table name for this batch to avoid conflicts
                        temp_table_name = f"temp_ohlcv_batch_{int(time.time() * 1000000)}"
                        
                        cursor.execute(f"""
                            CREATE TEMP TABLE {temp_table_name} (
                                time TIMESTAMPTZ,
                                symbol TEXT,
                                open DOUBLE PRECISION,
                                high DOUBLE PRECISION,
                                low DOUBLE PRECISION,
                                close DOUBLE PRECISION,
                                volume DOUBLE PRECISION
                            ) ON COMMIT DROP
                        """)
                        
                        # Use COPY to load data into temp table
                        with cursor.copy(
                            f"COPY {temp_table_name} (time, symbol, open, high, low, close, volume) FROM STDIN"
                        ) as copy:
                            for row in data_rows:
                                copy.write_row(row)
                        
                        # Insert from temp table with conflict resolution
                        cursor.execute(f"""
                            INSERT INTO ohlcv_1min (time, symbol, open, high, low, close, volume)
                            SELECT time, symbol, open, high, low, close, volume FROM {temp_table_name}
                            ON CONFLICT (symbol, time) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        """)
                        
                        conn.commit()
                        logger.info(f"Inserted batch of {len(batch)} records")
                    
        except psycopg.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during bulk insert: {e}")
            raise


class BulkHistoricalLoader:
    """Main bulk loading orchestrator."""
    
    def __init__(self):
        self.binance_client = BinanceClient(
            api_key=os.getenv('BINANCE_API_KEY'),
            secret_key=os.getenv('BINANCE_SECRET_KEY')
        )
        self.db_writer = BulkDatabaseWriter(
            connection_url=os.getenv('DATABASE_URL')
        )
        
        # Major crypto symbols for initial load
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 
            'DOGEUSDT', 'SOLUSDT', 'TRXUSDT'
        ]
    
    def check_existing_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, any]:
        """
        Check what data already exists in the database for a symbol and date range.
        
        Returns:
            Dict with existing data information and missing ranges
        """
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        
        with self.db_writer.pool.connection() as conn:
            with conn.cursor() as cursor:
                # Check if any data exists for this symbol in the date range
                cursor.execute("""
                    SELECT 
                        MIN(time) as first_record,
                        MAX(time) as last_record,
                        COUNT(*) as record_count
                    FROM ohlcv_1min 
                    WHERE symbol = %s 
                    AND time >= %s 
                    AND time <= %s
                """, (symbol, start_dt, end_dt))
                
                result = cursor.fetchone()
                
                if result and result[2] > 0:  # record_count > 0
                    first_record, last_record, record_count = result
                    
                    # Calculate expected records (1 minute intervals)
                    expected_records = int((end_dt - start_dt).total_seconds() / 60) + 1
                    
                    return {
                        'has_data': True,
                        'first_record': first_record,
                        'last_record': last_record,
                        'record_count': record_count,
                        'expected_records': expected_records,
                        'is_complete': record_count >= expected_records * 0.99,  # 99% threshold for "complete"
                        'completion_rate': (record_count / expected_records) * 100 if expected_records > 0 else 0
                    }
                else:
                    return {
                        'has_data': False,
                        'record_count': 0,
                        'expected_records': int((end_dt - start_dt).total_seconds() / 60) + 1,
                        'is_complete': False,
                        'completion_rate': 0
                    }
    
    def detect_missing_ranges(self, symbol: str, start_date: str, end_date: str) -> List[Tuple[datetime, datetime]]:
        """
        Detect missing time ranges for a symbol that need to be fetched.
        
        Returns:
            List of (start_datetime, end_datetime) tuples for missing ranges
        """
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        
        missing_ranges = []
        
        with self.db_writer.pool.connection() as conn:
            with conn.cursor() as cursor:
                # Find gaps in the data using a window function approach
                cursor.execute("""
                    WITH time_series AS (
                        SELECT generate_series(%s, %s, interval '1 minute') AS expected_time
                    ),
                    existing_data AS (
                        SELECT time 
                        FROM ohlcv_1min 
                        WHERE symbol = %s 
                        AND time >= %s 
                        AND time <= %s
                    ),
                    missing_times AS (
                        SELECT ts.expected_time
                        FROM time_series ts
                        LEFT JOIN existing_data ed ON ts.expected_time = ed.time
                        WHERE ed.time IS NULL
                        ORDER BY ts.expected_time
                    ),
                    gap_groups AS (
                        SELECT 
                            expected_time,
                            expected_time - (ROW_NUMBER() OVER (ORDER BY expected_time) * interval '1 minute') AS group_key
                        FROM missing_times
                    )
                    SELECT 
                        MIN(expected_time) AS gap_start,
                        MAX(expected_time) AS gap_end,
                        COUNT(*) AS gap_size
                    FROM gap_groups
                    GROUP BY group_key
                    ORDER BY gap_start
                    LIMIT 50;  -- Limit to avoid too many small gaps
                """, (start_dt, end_dt, symbol, start_dt, end_dt))
                
                gaps = cursor.fetchall()
                
                for gap_start, gap_end, gap_size in gaps:
                    if gap_size >= 1:  # Include even single-minute gaps for completeness
                        missing_ranges.append((gap_start, gap_end))
                        logger.info(f"Found gap for {symbol}: {gap_start} to {gap_end} ({gap_size} minutes)")
        
        return missing_ranges

    def load_symbol_sequential(self, symbol: str, start_date: str, end_date: str) -> int:
        """Load complete history for a single symbol with smart gap detection."""
        logger.info(f"🚀 Starting smart load for {symbol}")
        symbol_start_time = time.time()
        
        # Check existing data first
        existing_data = self.check_existing_data(symbol, start_date, end_date)
        
        logger.info(f"📊 {symbol} existing data analysis:")
        logger.info(f"   • Records in range: {existing_data.get('record_count', 0):,}")
        logger.info(f"   • Expected records: {existing_data.get('expected_records', 0):,}")
        logger.info(f"   • Completion rate: {existing_data.get('completion_rate', 0):.1f}%")
        
        if existing_data['has_data'] and existing_data['completion_rate'] >= 99.9:
            logger.info(f"✅ {symbol} data is already complete ({existing_data['record_count']} records). Skipping...")
            return existing_data['record_count']
        elif existing_data['has_data']:
            logger.info(f"🔧 {symbol} data is incomplete - checking for specific gaps...")
            
            # For partial data, detect specific missing ranges
            missing_ranges = self.detect_missing_ranges(symbol, start_date, end_date)
            
            if not missing_ranges:
                logger.info(f"✅ {symbol} no gaps found despite low completion rate - might be weekend/market hours")
                return existing_data['record_count']
            
            # Fill gaps only
            total_new_records = 0
            for i, (gap_start, gap_end) in enumerate(missing_ranges, 1):
                logger.info(f"🔧 Filling gap {i}/{len(missing_ranges)} for {symbol}: "
                           f"{gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')}")
                
                gap_records = self.binance_client.get_symbol_complete_history(
                    symbol=symbol,
                    start_date=gap_start.isoformat(),
                    end_date=gap_end.isoformat(),
                    interval='1m'
                )
                
                if gap_records:
                    self.db_writer.bulk_insert_symbol_data(gap_records)
                    total_new_records += len(gap_records)
                    logger.info(f"✅ Filled gap {i}: {len(gap_records)} records")
            
            final_count = existing_data['record_count'] + total_new_records
            logger.info(f"📊 {symbol} final count: {final_count:,} total records (+{total_new_records:,} new)")
            return final_count
        else:
            logger.info(f"📭 {symbol} has no existing data - will fetch complete range")
        
        # Fetch all data for this symbol (complete range)
        records = self.binance_client.get_symbol_complete_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1m'
        )
        
        # Bulk insert to database
        if records:
            logger.info(f"💾 Saving {len(records):,} records for {symbol} to database...")
            self.db_writer.bulk_insert_symbol_data(records)
        else:
            logger.warning(f"⚠️  No records fetched for {symbol}")
        
        duration = time.time() - symbol_start_time
        
        if records:
            rate = len(records) / duration if duration > 0 else 0
            logger.info(f"✅ {symbol} completed: {len(records):,} records in {duration:.1f}s ({rate:.0f} records/sec)")
        
        return len(records)
    
    def load_all_symbols_sequential(self, start_date: str, end_date: str) -> Dict[str, int]:
        """
        Load all symbols sequentially (Symbol-Sequential approach).
        
        This is the OPTIMAL approach for bulk historical data loading because:
        1. Better API rate limit utilization per symbol
        2. More efficient database bulk inserts  
        3. Better error recovery (per symbol)
        4. Linear progress tracking
        5. Prevents API rate limit conflicts
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
        
        Returns:
            Dict mapping symbol to record count
        """
        logger.info(f"🚀 Starting SYMBOL-SEQUENTIAL bulk load for {len(self.symbols)} symbols")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        logger.info("💡 Strategy: One symbol at a time for optimal API usage")
        
        results = {}
        start_time = time.time()
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"📊 Processing symbol {i}/{len(self.symbols)}: {symbol}")
            
            try:
                # Load complete history for this symbol
                record_count = self.load_symbol_sequential(symbol, start_date, end_date)
                results[symbol] = record_count
                
                # Progress update
                progress = (i / len(self.symbols)) * 100
                elapsed = time.time() - start_time
                
                logger.info(f"✅ {symbol}: {record_count:,} records loaded")
                logger.info(f"📈 Progress: {progress:.1f}% ({i}/{len(self.symbols)} symbols)")
                
                if i < len(self.symbols):
                    eta = (elapsed / i) * (len(self.symbols) - i)
                    logger.info(f"⏱️  ETA: {eta/60:.1f} minutes remaining")
                
                # Small delay between symbols to be respectful to API
                if i < len(self.symbols):
                    logger.info("⏳ Cooling down for 2 seconds...")
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ {symbol} failed: {e}")
                results[symbol] = 0
                # Continue with next symbol instead of failing completely
                logger.info("🔄 Continuing with next symbol...")
                continue
        
        total_duration = time.time() - start_time
        total_records = sum(results.values())
        successful_symbols = sum(1 for count in results.values() if count > 0)
        
        logger.info("🎉 SYMBOL-SEQUENTIAL BULK LOAD COMPLETED!")
        logger.info(f"📊 Total records: {total_records:,}")
        logger.info(f"✅ Successful symbols: {successful_symbols}/{len(self.symbols)}")
        logger.info(f"⏱️  Total time: {total_duration/60:.1f} minutes")
        
        if total_records > 0:
            logger.info(f"🚀 Average rate: {total_records/total_duration:.0f} records/second")
            logger.info("🎯 Data ready for ML/RL pipeline!")
        
        return results


def main():
    """Main entry point for bulk historical data loading."""
    
    # Configuration from environment with safer defaults
    start_date = os.getenv('START_DATE', '2024-01-01')

    # Default end date: 1 hour before current time to prevent incomplete data
    default_end = datetime.now() - timedelta(hours=1)
    end_date = os.getenv('END_DATE', default_end.strftime('%Y-%m-%d'))
    
    # Debug: Show environment variables
    logger.info("🔍 ENVIRONMENT VARIABLE CHECK:")
    logger.info(f"   • START_DATE env: {os.getenv('START_DATE')}")
    logger.info(f"   • END_DATE env: {os.getenv('END_DATE')}")
    logger.info(f"   • Resolved start_date: {start_date}")
    logger.info(f"   • Resolved end_date: {end_date}")
    
    # Add timezone if not present
    if 'T' not in start_date:
        start_date += 'T00:00:00'
    if 'T' not in end_date:
        end_date += 'T23:59:59'
    
    logger.info("🚀 HIGH-PERFORMANCE BULK HISTORICAL DATA LOADER")
    logger.info("💡 Strategy: Symbol-Sequential with Smart Gap Detection")
    logger.info(f"📅 Final date range: {start_date} to {end_date}")
    
    # Calculate expected duration
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    duration_hours = (end_dt - start_dt).total_seconds() / 3600
    expected_records_per_symbol = int(duration_hours * 60)  # 1 minute intervals
    
    logger.info(f"⏱️  Duration: {duration_hours:.1f} hours")
    logger.info(f"📊 Expected records per symbol: ~{expected_records_per_symbol:,}")
    
    try:
        loader = BulkHistoricalLoader()
        # Use sequential loading instead of parallel
        results = loader.load_all_symbols_sequential(
            start_date=start_date,
            end_date=end_date
        )
        
        # Summary report
        logger.info("📈 FINAL LOAD SUMMARY:")
        successful_symbols = 0
        for symbol, count in sorted(results.items()):
            status = "✅" if count > 0 else "❌"
            if count > 0:
                successful_symbols += 1
            logger.info(f"  {status} {symbol}: {count:,} records")
        
        # Final statistics
        total_records = sum(results.values())
        success_rate = (successful_symbols / len(results)) * 100
        
        logger.info(f"🎯 Success rate: {success_rate:.1f}% ({successful_symbols}/{len(results)} symbols)")
        logger.info(f"📊 Total records loaded: {total_records:,}")
        
        if successful_symbols == len(results):
            logger.info("🎉 ALL SYMBOLS LOADED SUCCESSFULLY!")
            
            # Final gap check
            logger.info("🔍 Performing final gap check...")
            total_gaps_found = 0
            for symbol in loader.symbols:
                final_check = loader.check_existing_data(symbol, start_date, end_date)
                if final_check['completion_rate'] < 99.5:
                    gaps = loader.detect_missing_ranges(symbol, start_date, end_date)
                    if gaps:
                        total_gaps_found += len(gaps)
                        logger.warning(f"⚠️  {symbol} still has {len(gaps)} gaps after bulk load")
            
            if total_gaps_found > 0:
                logger.warning(f"⚠️  Found {total_gaps_found} remaining gaps across all symbols")
                logger.info("💡 Consider running data-recovery service to fill remaining gaps")
                logger.info("   Command: docker compose -f docker-compose.dev.yml --profile data-recovery up")
            else:
                logger.info("✅ No gaps detected - data is complete!")
                
        elif successful_symbols > 0:
            logger.info("⚠️  Partial success - some symbols failed")
        else:
            logger.error("❌ ALL SYMBOLS FAILED - check configuration")
            return False
            
    except Exception as e:
        logger.error(f"💥 Bulk load failed: {e}")
        return False
    finally:
        # Always cleanup database connections
        try:
            loader.db_writer.cleanup()
        except Exception as e:
            logger.warning(f"Warning during cleanup: {e}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
