import os
import psycopg
import requests
import urllib3.util.retry
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from confluent_kafka import Producer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataRecoveryService:
    """Service to detect gaps in OHLCV data and backfill using Binance REST API"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
        self.kafka_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "agg.ohlcv.1m")
        
        # Binance API configuration
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Symbols to monitor (same as ingestor)
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 
            'DOGEUSDT', 'SOLUSDT', 'TRXUSDT'
        ]
        
        # Initialize Kafka producer for backfilled data (using confluent-kafka)
        self.kafka_producer = Producer({
            'bootstrap.servers': self.kafka_servers,
            'client.id': 'data-recovery-service'
        })
        
        if self.binance_api_key:
            logger.info("Data Recovery Service initialized with API key authentication")
        else:
            logger.info("Data Recovery Service initialized with public API access")
            logger.warning("Consider adding BINANCE_API_KEY for higher rate limits")
    
    def get_db_connection(self):
        """Get database connection using psycopg3"""
        return psycopg.connect(self.database_url)
    
    def detect_gaps(self, hours_back: int = 24, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Tuple[datetime, str]]:
        """
        Detect missing 1-minute intervals for each symbol in the last N hours or from a specific start date
        Returns list of (datetime, symbol) tuples for missing data
        
        Args:
            hours_back: Number of hours to look back from now (ignored if start_date is provided)
            start_date: Start date in YYYY-MM-DD format (e.g., "2025-01-01")
            end_date: End date in YYYY-MM-DD format (e.g., "2025-06-25"), defaults to NOW() if not provided
        """
        gaps = []
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                for symbol in self.symbols:
                    if start_date:
                        # Use specific start date
                        start_time = datetime.strptime(start_date, "%Y-%m-%d")
                        if end_date:
                            end_time = datetime.strptime(end_date, "%Y-%m-%d")
                        else:
                            end_time = datetime.now()
                        
                        query = """
                        WITH expected_times AS (
                            SELECT generate_series(
                                date_trunc('minute', %s::timestamp),
                                date_trunc('minute', %s::timestamp),
                                INTERVAL '1 minute'
                            ) AS expected_time
                        ),
                        actual_times AS (
                            SELECT DISTINCT time 
                            FROM ohlcv_1min 
                            WHERE symbol = %s 
                            AND time >= %s::timestamp
                            AND time <= %s::timestamp
                        )
                        SELECT et.expected_time
                        FROM expected_times et
                        LEFT JOIN actual_times at ON et.expected_time = at.time
                        WHERE at.time IS NULL
                        ORDER BY et.expected_time;
                        """
                        
                        cur.execute(query, (start_time, end_time, symbol, start_time, end_time))
                        if end_date:
                            logger.info(f"Checking for gaps in {symbol} from {start_date} to {end_date}")
                        else:
                            logger.info(f"Checking for gaps in {symbol} from {start_date} to now")
                    else:
                        # Use hours_back from now
                        query = """
                        WITH expected_times AS (
                            SELECT generate_series(
                                date_trunc('minute', NOW() - INTERVAL '%s hours'),
                                date_trunc('minute', NOW()),
                                INTERVAL '1 minute'
                            ) AS expected_time
                        ),
                        actual_times AS (
                            SELECT DISTINCT time 
                            FROM ohlcv_1min 
                            WHERE symbol = %s 
                            AND time >= NOW() - INTERVAL '%s hours'
                        )
                        SELECT et.expected_time
                        FROM expected_times et
                        LEFT JOIN actual_times at ON et.expected_time = at.time
                        WHERE at.time IS NULL
                        ORDER BY et.expected_time;
                        """
                        
                        cur.execute(query, (hours_back, symbol, hours_back))
                        logger.info(f"Checking for gaps in {symbol} for last {hours_back} hours")
                    
                    missing_times = cur.fetchall()
                    
                    for (missing_time,) in missing_times:
                        gaps.append((missing_time, symbol))
                    
                    if missing_times:
                        logger.info(f"Found {len(missing_times)} missing intervals for {symbol}")
        
        return gaps
    
    def get_binance_klines(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[List]:
        """
        Fetch historical 1-minute klines from Binance REST API with robust retry logic
        """
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
        base_delay = 2  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1)) + (attempt * 0.5)  # Add jitter
                    logger.info(f"Retrying Binance API call for {symbol} in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                
                # Add session for connection pooling and DNS caching
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=10,
                    max_retries=urllib3.util.retry.Retry(
                        total=0,  # We handle retries manually
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504]
                    )
                )
                session.mount('https://', adapter)
                
                response = session.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=(10, 30)  # (connect_timeout, read_timeout)
                )
                response.raise_for_status()
                
                klines_data = response.json()
                logger.info(f"Retrieved {len(klines_data)} klines for {symbol}")
                
                return klines_data
                
            except requests.exceptions.ConnectionError as e:
                if "Temporary failure in name resolution" in str(e) or "Failed to resolve" in str(e):
                    logger.warning(f"DNS resolution failed for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                    # For DNS issues, wait longer before retry
                    if attempt < max_retries - 1:
                        time.sleep(min(30, base_delay * (2 ** attempt)))
                else:
                    logger.warning(f"Connection error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to Binance API for {symbol} after {max_retries} attempts")
                    return None
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Request timeout for {symbol} after {max_retries} attempts")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for {symbol} (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        logger.error(f"Rate limit exceeded for {symbol} after {max_retries} attempts")
                        return None
                else:
                    logger.error(f"HTTP error for {symbol}: {e}")
                    return None
                    
            except requests.RequestException as e:
                logger.warning(f"Request error for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Request failed for {symbol} after {max_retries} attempts")
                    return None
        
        return None
    
    def convert_binance_to_ohlcv(self, klines_data: List, symbol: str) -> List[Dict]:
        """
        Convert Binance klines format to our OHLCV format
        
        Binance klines format:
        [
          [
            1499040000000,      // Open time
            "0.01634790",       // Open price
            "0.80000000",       // High price
            "0.01575800",       // Low price
            "0.01577100",       // Close price
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore
          ]
        ]
        """
        ohlcv_records = []
        
        for kline in klines_data:
            record = {
                "time": datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                "symbol": symbol,
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            }
            ohlcv_records.append(record)
        
        return ohlcv_records
    
    def send_to_kafka(self, ohlcv_records: List[Dict]):
        """Send backfilled OHLCV data to Kafka topic"""
        sent_count = 0
        
        for record in ohlcv_records:
            try:
                # Use symbol as key for partitioning
                key = record["symbol"]
                value = json.dumps(record)
                
                # Send to Kafka using confluent-kafka
                self.kafka_producer.produce(
                    topic=self.kafka_topic,
                    key=key,
                    value=value
                )
                
                sent_count += 1
                
            except Exception as e:
                logger.error(f"Error sending record to Kafka: {e}")
        
        # Flush to ensure all messages are sent
        self.kafka_producer.flush()
        
        logger.info(f"Sent {sent_count} backfilled records to Kafka topic: {self.kafka_topic}")
        return sent_count
    
    def backfill_gaps(self, gaps: List[Tuple[datetime, str]], batch_size: int = 60) -> int:
        """
        Backfill missing data gaps by fetching from Binance API
        
        Args:
            gaps: List of (datetime, symbol) tuples
            batch_size: Number of minutes to fetch in one API call
        
        Returns:
            Number of records successfully backfilled
        """
        total_backfilled = 0
        
        # Group gaps by symbol for efficient API calls
        gaps_by_symbol = {}
        for gap_time, symbol in gaps:
            if symbol not in gaps_by_symbol:
                gaps_by_symbol[symbol] = []
            gaps_by_symbol[symbol].append(gap_time)
        
        for symbol, missing_times in gaps_by_symbol.items():
            logger.info(f"Backfilling {len(missing_times)} gaps for {symbol}")
            
            # Sort times and group into continuous ranges
            missing_times.sort()
            
            # Process in batches to avoid API rate limits
            for i in range(0, len(missing_times), batch_size):
                batch_times = missing_times[i:i + batch_size]
                
                if not batch_times:
                    continue
                
                start_time = batch_times[0]
                end_time = batch_times[-1] + timedelta(minutes=1)
                
                # Fetch data from Binance
                klines_data = self.get_binance_klines(symbol, start_time, end_time)
                
                if klines_data:
                    # Convert to our format
                    ohlcv_records = self.convert_binance_to_ohlcv(klines_data, symbol)
                    
                    # Send to Kafka (db-writer will pick it up)
                    sent_count = self.send_to_kafka(ohlcv_records)
                    total_backfilled += sent_count
                    
                    logger.info(f"Backfilled {sent_count} records for {symbol} ({start_time} to {end_time})")
                
                # Rate limiting - Binance allows 1200 requests per minute
                time.sleep(0.1)  # 100ms delay between requests
        
        return total_backfilled
    
    def run_gap_detection_cycle(self, hours_back: int = 24, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Run one cycle of gap detection and backfilling"""
        if start_date:
            if end_date:
                logger.info(f"Starting gap detection from {start_date} to {end_date}...")
            else:
                logger.info(f"Starting gap detection from {start_date} to now...")
        else:
            logger.info(f"Starting gap detection for last {hours_back} hours...")
        
        # Detect gaps
        gaps = self.detect_gaps(hours_back, start_date, end_date)
        
        if not gaps:
            logger.info("No gaps detected!")
            return
        
        logger.info(f"Found {len(gaps)} total gaps across all symbols")
        
        # Backfill gaps
        backfilled_count = self.backfill_gaps(gaps)
        
        logger.info(f"Gap detection cycle completed. Backfilled {backfilled_count} records.")
    
    def run_continuous_monitoring(self, check_interval_minutes: int = 10):
        """Run continuous gap detection and backfilling"""
        logger.info(f"Starting continuous monitoring (checking every {check_interval_minutes} minutes)")
        
        while True:
            try:
                # Check for gaps in the last 2 hours (more frequent checks)
                self.run_gap_detection_cycle(hours_back=2)
                
                # Wait before next check
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous monitoring...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main entry point"""
    service = DataRecoveryService()
    
    # Check if running in one-time mode or continuous mode
    mode = os.getenv("RECOVERY_MODE", "continuous")
    start_date = os.getenv("START_DATE")  # Optional: YYYY-MM-DD format
    
    # If START_DATE is provided, force oneshot mode
    if start_date:
        mode = "oneshot"
        logger.info(f"START_DATE provided ({start_date}), forcing oneshot mode")
    
    if mode == "oneshot":
        # Run one-time gap detection
        hours_back = int(os.getenv("HOURS_BACK", "24"))
        end_date = os.getenv("END_DATE")  # Optional: YYYY-MM-DD format
        
        if start_date:
            logger.info(f"Running one-time recovery from {start_date} to {end_date or 'now'}")
            service.run_gap_detection_cycle(start_date=start_date, end_date=end_date)
        else:
            logger.info(f"Running one-time recovery for last {hours_back} hours")
            service.run_gap_detection_cycle(hours_back)
        
        logger.info("One-shot recovery completed. Exiting.")
    else:
        # Run continuous monitoring
        check_interval = int(os.getenv("CHECK_INTERVAL_MINUTES", "10"))
        service.run_continuous_monitoring(check_interval)


if __name__ == "__main__":
    main()
