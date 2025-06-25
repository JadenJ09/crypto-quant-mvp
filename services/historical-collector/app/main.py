#!/usr/bin/env python3
"""
Hybrid Historical Data Collection Service
Combines fast bulk loading with intelligent gap detection and filling.

Strategy:
1. Use bulk-loader for fast initial 1m data collection (Symbol-Sequential)
2. Detect gaps in collected data per symbol
3. Targeted backfill of missing data only
4. Optimal for ML/RL pipeline (1m base data + TimescaleDB aggregation)
"""

import os
import psycopg
import requests
import urllib3.util.retry
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Set
from confluent_kafka import Producer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridHistoricalCollector:
    """
    Hybrid approach for historical data collection:
    1. Fast bulk collection (Symbol-Sequential)
    2. Intelligent gap detection
    3. Targeted backfill of missing data only
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
        self.kafka_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "agg.ohlcv.1m")
        
        # Binance API configuration
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Collection parameters
        self.start_date = os.getenv("START_DATE", "2025-01-01")
        self.end_date = os.getenv("END_DATE", datetime.now().strftime('%Y-%m-%d'))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        
        # Major crypto symbols optimized for ML/RL pipeline
        self.symbols = [
            # Tier 1: Major pairs (highest liquidity) - prioritize for gap filling
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            
            # Tier 2: Large caps 
            'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            
            # # Tier 3: Mid caps (good for ML model diversity)
            # 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
            
            # # Tier 4: Active trading pairs
            # 'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'ETCUSDT'
        ]
        
        # Initialize Kafka producer for backfilled data
        self.kafka_producer = Producer({
            'bootstrap.servers': self.kafka_servers,
            'client.id': 'hybrid-historical-collector'
        })
        
        # Rate limiting for API calls
        self.requests_per_second = 18  # Conservative limit
        self.last_request_time = 0
        
        logger.info("🚀 Hybrid Historical Collector initialized")
        logger.info(f"📅 Collection period: {self.start_date} to {self.end_date}")
        logger.info(f"🎯 Symbols: {len(self.symbols)} symbols")
        logger.info(f"⚡ Max workers: {self.max_workers}")
    
    def get_db_connection(self):
        """Get database connection using psycopg3"""
        return psycopg.connect(self.database_url)
    
    def _rate_limit(self):
        """Smart rate limiting to maximize throughput while staying safe."""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def detect_symbol_gaps(self, symbol: str, start_date: str, end_date: str) -> List[Tuple[datetime, datetime]]:
        """
        Detect missing 1-minute intervals for a specific symbol.
        Returns list of (start_time, end_time) tuples representing continuous gap periods.
        
        This is optimized for:
        - Bulk gap detection per symbol
        - Continuous gap period identification
        - Minimal API calls for backfill
        """
        gaps = []
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Convert dates to datetime objects
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    # Query to find missing minute intervals
                    query = """
                    WITH RECURSIVE expected_times AS (
                        -- Generate all expected 1-minute intervals
                        SELECT %s::timestamp AS expected_time
                        UNION ALL
                        SELECT expected_time + INTERVAL '1 minute'
                        FROM expected_times
                        WHERE expected_time < %s::timestamp
                    ),
                    actual_times AS (
                        -- Get all existing data points for this symbol
                        SELECT DISTINCT time 
                        FROM ohlcv_1min 
                        WHERE symbol = %s 
                        AND time >= %s::timestamp
                        AND time <= %s::timestamp
                    ),
                    missing_times AS (
                        -- Find missing intervals
                        SELECT et.expected_time
                        FROM expected_times et
                        LEFT JOIN actual_times at ON et.expected_time = at.time
                        WHERE at.time IS NULL
                        ORDER BY et.expected_time
                    )
                    SELECT expected_time FROM missing_times;
                    """
                    
                    cur.execute(query, (start_dt, end_dt, symbol, start_dt, end_dt))
                    missing_times = [row[0] for row in cur.fetchall()]
                    
                    if not missing_times:
                        logger.info(f"✅ {symbol}: No gaps found - data is complete")
                        return gaps
                    
                    # Group consecutive missing times into continuous gaps
                    if missing_times:
                        gap_start = missing_times[0]
                        gap_end = missing_times[0]
                        
                        for i in range(1, len(missing_times)):
                            current_time = missing_times[i]
                            expected_next = gap_end + timedelta(minutes=1)
                            
                            if current_time == expected_next:
                                # Extend current gap
                                gap_end = current_time
                            else:
                                # End current gap and start new one
                                gaps.append((gap_start, gap_end))
                                gap_start = current_time
                                gap_end = current_time
                        
                        # Add the last gap
                        gaps.append((gap_start, gap_end))
                        
                        logger.info(f"🔍 {symbol}: Found {len(gaps)} gap periods "
                                   f"(total {len(missing_times)} missing minutes)")
                        
                        # Log gap summary
                        for i, (start, end) in enumerate(gaps[:5]):  # Show first 5 gaps
                            duration = int((end - start).total_seconds() / 60) + 1
                            logger.info(f"   Gap {i+1}: {start} to {end} ({duration} minutes)")
                        
                        if len(gaps) > 5:
                            logger.info(f"   ... and {len(gaps) - 5} more gaps")
                    
        except Exception as e:
            logger.error(f"❌ Error detecting gaps for {symbol}: {e}")
            return []
        
        return gaps
    
    def get_binance_klines_optimized(self, symbol: str, start_time: datetime, 
                                   end_time: datetime) -> Optional[List]:
        """
        Optimized Binance API call for gap filling.
        
        Designed for:
        - Minimal API calls
        - Robust error handling
        - Rate limit compliance
        """
        self._rate_limit()
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000  # Maximum allowed by Binance
        }
        
        # Add API key headers if available
        headers = {}
        if self.binance_api_key:
            headers['X-MBX-APIKEY'] = self.binance_api_key
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1)) + (attempt * 0.5)
                    logger.info(f"⏳ Retrying {symbol} API call in {delay:.1f}s "
                               f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=(10, 30)
                )
                response.raise_for_status()
                
                klines_data = response.json()
                
                if klines_data:
                    logger.info(f"✅ Retrieved {len(klines_data)} klines for {symbol} "
                               f"({start_time} to {end_time})")
                
                return klines_data
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🌐 Connection error for {symbol}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed to connect after {max_retries} attempts")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⏰ Rate limit hit for {symbol}")
                    if attempt < max_retries - 1:
                        time.sleep(60)  # Wait 1 minute for rate limit
                else:
                    logger.error(f"❌ HTTP error for {symbol}: {e}")
                    return None
                    
            except Exception as e:
                logger.warning(f"⚠️  Request error for {symbol}: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def fill_symbol_gaps(self, symbol: str, gaps: List[Tuple[datetime, datetime]]) -> int:
        """
        Fill gaps for a specific symbol by fetching missing data from Binance API.
        
        Returns:
            Number of records successfully backfilled
        """
        if not gaps:
            return 0
        
        total_filled = 0
        
        logger.info(f"🚀 Starting gap fill for {symbol}: {len(gaps)} gap periods")
        
        for i, (gap_start, gap_end) in enumerate(gaps):
            try:
                # Fetch data for this gap period
                klines_data = self.get_binance_klines_optimized(symbol, gap_start, gap_end)
                
                if not klines_data:
                    logger.warning(f"⚠️  No data received for {symbol} gap {i+1}")
                    continue
                
                # Convert to OHLCV format and send to Kafka
                ohlcv_records = self.convert_binance_to_ohlcv(klines_data, symbol)
                
                if ohlcv_records:
                    self.send_to_kafka(ohlcv_records)
                    total_filled += len(ohlcv_records)
                    
                    logger.info(f"✅ {symbol} gap {i+1}/{len(gaps)}: "
                               f"{len(ohlcv_records)} records filled")
                
                # Small delay between gap fills to be respectful to API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error filling gap {i+1} for {symbol}: {e}")
                continue
        
        logger.info(f"🎉 {symbol} gap fill complete: {total_filled} records filled")
        return total_filled
    
    def convert_binance_to_ohlcv(self, klines_data: List, symbol: str) -> List[Dict]:
        """Convert Binance klines format to OHLCV records."""
        ohlcv_records = []
        
        for kline in klines_data:
            record = {
                'timestamp': kline[0],  # Keep as milliseconds for Kafka
                'symbol': symbol,
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            }
            ohlcv_records.append(record)
        
        return ohlcv_records
    
    def send_to_kafka(self, records: List[Dict]):
        """Send OHLCV records to Kafka topic."""
        for record in records:
            try:
                message = json.dumps(record)
                self.kafka_producer.produce(
                    self.kafka_topic,
                    value=message,
                    key=record['symbol']
                )
            except Exception as e:
                logger.error(f"❌ Failed to send record to Kafka: {e}")
        
        # Flush to ensure delivery
        self.kafka_producer.flush(timeout=10)
    
    def collect_symbol_hybrid(self, symbol: str) -> Dict[str, int]:
        """
        Hybrid collection for a single symbol:
        1. Detect gaps in existing data
        2. Fill only the missing gaps
        3. Report collection statistics
        """
        logger.info(f"🔍 Processing {symbol} with hybrid approach")
        start_time = time.time()
        
        # Step 1: Detect gaps in existing data
        gaps = self.detect_symbol_gaps(symbol, self.start_date, self.end_date)
        
        # Step 2: Fill gaps if any exist
        filled_count = 0
        if gaps:
            filled_count = self.fill_symbol_gaps(symbol, gaps)
        
        duration = time.time() - start_time
        
        result = {
            'gaps_detected': len(gaps),
            'records_filled': filled_count,
            'duration_seconds': round(duration, 2)
        }
        
        logger.info(f"✅ {symbol} complete: {len(gaps)} gaps, "
                   f"{filled_count} records filled in {duration:.1f}s")
        
        return result
    
    def run_hybrid_collection(self) -> Dict[str, Dict]:
        """
        Run hybrid historical collection for all symbols.
        
        Strategy:
        1. Process symbols in parallel (Symbol-Sequential approach)
        2. Each symbol: detect gaps → fill gaps
        3. Minimal API calls (only for missing data)
        4. Optimal for ML/RL pipeline
        """
        logger.info("🚀 HYBRID HISTORICAL COLLECTION STARTED")
        logger.info(f"📊 Processing {len(self.symbols)} symbols")
        logger.info(f"📅 Period: {self.start_date} to {self.end_date}")
        logger.info(f"⚡ Workers: {self.max_workers}")
        logger.info("🎯 Strategy: Gap detection + targeted backfill")
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all symbol processing tasks
            future_to_symbol = {
                executor.submit(self.collect_symbol_hybrid, symbol): symbol
                for symbol in self.symbols
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[symbol] = result
                    
                    progress = (completed / len(self.symbols)) * 100
                    logger.info(f"🎉 {symbol}: {result['gaps_detected']} gaps, "
                               f"{result['records_filled']} filled "
                               f"(Progress: {progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} failed: {e}")
                    results[symbol] = {
                        'gaps_detected': 0,
                        'records_filled': 0,
                        'duration_seconds': 0,
                        'error': str(e)
                    }
        
        # Final summary
        total_duration = time.time() - start_time
        total_gaps = sum(r['gaps_detected'] for r in results.values())
        total_filled = sum(r['records_filled'] for r in results.values())
        successful_symbols = len([r for r in results.values() if 'error' not in r])
        
        logger.info("🎉 HYBRID COLLECTION COMPLETED!")
        logger.info(f"📊 Processed: {successful_symbols}/{len(self.symbols)} symbols")
        logger.info(f"🔍 Total gaps detected: {total_gaps}")
        logger.info(f"💾 Total records filled: {total_filled:,}")
        logger.info(f"⏱️  Total time: {total_duration:.1f}s")
        
        if total_filled > 0:
            logger.info(f"🚀 Fill rate: {total_filled/total_duration:.0f} records/second")
            logger.info("✅ Data collection optimized for ML/RL pipeline")
        else:
            logger.info("✅ All data was already complete - no gaps to fill!")
        
        return results


def main():
    """Main entry point for hybrid historical data collection."""
    
    logger.info("🚀 HYBRID HISTORICAL DATA COLLECTOR")
    logger.info("🎯 Strategy: Fast bulk load + intelligent gap filling")
    logger.info("💡 Optimized for ML/RL trading pipeline")
    
    try:
        collector = HybridHistoricalCollector()
        results = collector.run_hybrid_collection()
        
        # Detailed summary
        logger.info("📈 COLLECTION SUMMARY:")
        for symbol, result in sorted(results.items()):
            if 'error' in result:
                logger.info(f"  ❌ {symbol}: {result['error']}")
            else:
                logger.info(f"  ✅ {symbol}: {result['gaps_detected']} gaps, "
                           f"{result['records_filled']} filled")
        
        logger.info("🎯 Next steps:")
        logger.info("  • Data ready for real-time technical indicators")
        logger.info("  • Use TimescaleDB time_bucket() for higher timeframes")
        logger.info("  • Start ML model training with consistent 1m data")
        
    except Exception as e:
        logger.error(f"💥 Hybrid collection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
