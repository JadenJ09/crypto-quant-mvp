# ==============================================================================
# File: services/db-writer/app/main.py
# ==============================================================================
# The core application logic for the db-writer service.
# REFACTORED FOR PERFORMANCE AND ROBUSTNESS:
# - Uses psycopg_pool for resilient database connection management.
# - Implements a high-throughput UPSERT strategy using a temporary table and COPY command.

import json
import logging
import os
import sys
import time
from datetime import datetime
import io
import csv

import psycopg
from psycopg_pool import ConnectionPool
from confluent_kafka import Consumer, KafkaException, KafkaError
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "agg.ohlcv.1m"
KAFKA_GROUP_ID = "ohlcv_db_writer_group" # Consumer group ID
DB_CONN_STRING = os.environ.get("DATABASE_URL")
# Set a higher min_size for the pool in a production environment
db_pool = ConnectionPool(DB_CONN_STRING, min_size=2, max_size=10)
logging.info("Database connection pool created.")


# --- Pydantic Data Validation Model for OHLCV Bar ---
class OHLCVBar(BaseModel):
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

# --- Database Operations ---

def create_temp_table(conn):
    """Creates an UNLOGGED temporary table for high-speed staging of data."""
    temp_table_sql = """
    CREATE UNLOGGED TABLE IF NOT EXISTS ohlcv_1min_temp (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION
    );
    """
    try:
        with conn.cursor() as cur:
            cur.execute(temp_table_sql)
            conn.commit()
            logging.info("Temporary table 'ohlcv_1min_temp' is ready.")
    except psycopg.Error as e:
        logging.error(f"Failed to create temporary table: {e}")
        conn.rollback()
        raise

def upsert_ohlcv_batch(pool: ConnectionPool, bars: list[OHLCVBar]):
    """
    Inserts or updates a batch of OHLCV bars using a high-performance
    COPY to a temporary table followed by an INSERT ... ON CONFLICT.
    """
    if not bars:
        return 0

    # The fields must match the order in ohlcv_1min_temp
    cols = ["time", "symbol", "open", "high", "low", "close", "volume"]
    
    sql_upsert_from_temp = """
    INSERT INTO ohlcv_1min (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume FROM ohlcv_1min_temp
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;
    """
    
    try:
        # Get a connection from the pool
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Use COPY to efficiently load data into the temp table
                with io.StringIO() as buffer:
                    writer = csv.writer(buffer, delimiter='\t', quotechar='"')
                    for bar in bars:
                        writer.writerow([getattr(bar, col) for col in cols])
                    
                    buffer.seek(0)
                    with cur.copy(f"COPY ohlcv_1min_temp ({', '.join(cols)}) FROM STDIN") as copy:
                        copy.write(buffer.read())

                # 2. Upsert from the temp table into the main table
                cur.execute(sql_upsert_from_temp)
                
                # 3. Clean up the temp table for the next batch
                cur.execute("TRUNCATE ohlcv_1min_temp")

                conn.commit()
                return len(bars)
    except psycopg.Error as e:
        logging.error(f"Database error during batch upsert: {e}")
        # The pool handles connection state, no need for conn.rollback() here
        # as the 'with' block will release the connection properly.
        return 0
    except Exception as e:
        logging.error(f"An unexpected error occurred during upsert: {e}")
        return 0


def main():
    """
    Main consumer loop. Polls Kafka for messages and writes them to the DB in batches.
    """
    consumer_config = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    }
    consumer = Consumer(consumer_config)

    # --- Retry logic for Kafka subscription (unchanged) ---
    max_retries = 5
    retry_delay_seconds = 10
    retries = 0
    subscribed = False
    while not subscribed and retries < max_retries:
        try:
            consumer.subscribe([KAFKA_TOPIC])
            logging.info(f"Attempting to subscribe to Kafka topic: {KAFKA_TOPIC}")
            test_msg = consumer.poll(timeout=5.0)
            if test_msg is None:
                logging.info(f"Successfully subscribed to Kafka topic: {KAFKA_TOPIC} (polled None, topic likely exists or will be auto-created).")
                subscribed = True
            elif test_msg.error():
                if test_msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    logging.warning(f"Kafka topic {KAFKA_TOPIC} not available yet (UNKNOWN_TOPIC_OR_PART). Retrying ({retries+1}/{max_retries})...")
                    retries += 1
                    time.sleep(retry_delay_seconds)
                else:
                    raise KafkaException(test_msg.error())
            else:
                logging.info(f"Successfully subscribed to Kafka topic: {KAFKA_TOPIC} (polled a message).")
                subscribed = True
                logging.info(f"Received initial message during subscription check: {test_msg.value()[:100]}...")
        except KafkaException as e:
            if e.args[0].code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                logging.warning(f"Kafka topic {KAFKA_TOPIC} not available (UNKNOWN_TOPIC_OR_PART on subscribe attempt). Retrying ({retries+1}/{max_retries})...")
                retries += 1
                time.sleep(retry_delay_seconds)
            else:
                logging.error(f"Unhandled KafkaException during subscription: {e}")
                raise
        except Exception as e:
            logging.error(f"Unexpected error during Kafka subscription: {e}")
            raise

    if not subscribed:
        logging.error(f"Failed to subscribe to Kafka topic {KAFKA_TOPIC} after {max_retries} retries. Exiting.")
        sys.exit(1)
    # --- End of retry logic ---

    # --- Ensure temp table exists before starting the loop ---
    try:
        with db_pool.connection() as conn:
            create_temp_table(conn)
    except Exception as e:
        logging.error(f"Could not initialize database environment. Exiting. Error: {e}")
        sys.exit(1)
    # ---

    batch = []
    batch_size = 500 # Increased batch size for better COPY performance
    last_write_time = time.time()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                if batch and (time.time() - last_write_time > 5):
                    logging.info(f"Timeout reached. Writing incomplete batch of {len(batch)} bars...")
                    rows_written = upsert_ohlcv_batch(db_pool, batch)
                    if rows_written > 0:
                        consumer.commit(asynchronous=False)
                    batch.clear()
                    last_write_time = time.time()
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())

            try:
                data = json.loads(msg.value().decode('utf-8'))
                bar = OHLCVBar(**data)
                batch.append(bar)

                if len(batch) >= batch_size:
                    logging.info(f"Batch size reached. Writing {len(batch)} bars to DB...")
                    rows_written = upsert_ohlcv_batch(db_pool, batch)
                    if rows_written > 0:
                        consumer.commit(asynchronous=False)
                    batch.clear()
                    last_write_time = time.time()

            except (json.JSONDecodeError, ValidationError) as e:
                logging.warning(f"Skipping malformed message. Error: {e}. Value: {msg.value()}")

    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    finally:
        if batch:
            logging.info(f"Writing final batch of {len(batch)} bars...")
            rows_written = upsert_ohlcv_batch(db_pool, batch)
            if rows_written > 0:
                consumer.commit(asynchronous=False)
        
        db_pool.close()
        consumer.close()
        logging.info("Database connection pool and Kafka consumer closed.")


if __name__ == "__main__":
    main()