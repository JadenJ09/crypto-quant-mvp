# ==============================================================================
# File: services/db-writer/app/main.py
# ==============================================================================
# The core application logic for the db-writer service.

import json
import logging
import os
import sys
import time
from datetime import datetime

import psycopg
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

# --- Pydantic Data Validation Model for OHLCV Bar ---
class OHLCVBar(BaseModel):
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

# --- Main Application Logic ---

def get_db_connection():
    """Establishes and returns a connection to the database."""
    while True:
        try:
            conn = psycopg.connect(DB_CONN_STRING)
            logging.info("Successfully connected to TimescaleDB.")
            return conn
        except psycopg.OperationalError as e:
            logging.error(f"Could not connect to database: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def upsert_ohlcv_batch(conn, bars: list[OHLCVBar]):
    """
    Inserts or updates a batch of OHLCV bars into the database using an UPSERT command.
    UPSERT is crucial for data idempotency - if we accidentally consume the same message
    twice, this command will simply update the existing row instead of creating a duplicate.
    """
    if not bars:
        return 0

    sql_upsert = """
    INSERT INTO ohlcv_1min (time, symbol, open, high, low, close, volume)
    VALUES (%(time)s, %(symbol)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;
    """
    try:
        with conn.cursor() as cur:
            # executemany is highly efficient for batch operations.
            cur.executemany(sql_upsert, [bar.model_dump() for bar in bars])
            conn.commit()
            return len(bars)
    except psycopg.Error as e:
        logging.error(f"Database error during batch upsert: {e}")
        conn.rollback()
        return 0

def main():
    """
    Main consumer loop. Polls Kafka for messages and writes them to the DB in batches.
    """
    consumer_config = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest', # Start reading from the beginning of the topic if no offset is stored
        'enable.auto.commit': False # We will commit offsets manually for better control
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([KAFKA_TOPIC])
    logging.info(f"Subscribed to Kafka topic: {KAFKA_TOPIC}")

    conn = get_db_connection()
    batch = []
    batch_size = 100
    last_write_time = time.time()

    try:
        while True:
            msg = consumer.poll(timeout=1.0) # Poll for messages with a 1-second timeout

            if msg is None:
                # If no message, check if it's time to write the current batch
                if batch and (time.time() - last_write_time > 5):
                    logging.info("Timeout reached. Writing incomplete batch...")
                    upsert_ohlcv_batch(conn, batch)
                    consumer.commit(asynchronous=False)
                    batch.clear()
                    last_write_time = time.time()
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event, not an error
                    continue
                else:
                    raise KafkaException(msg.error())

            try:
                # Decode and validate the message payload
                data = json.loads(msg.value().decode('utf-8'))
                bar = OHLCVBar(**data)
                batch.append(bar)

                if len(batch) >= batch_size:
                    logging.info(f"Batch size reached. Writing {len(batch)} bars to DB...")
                    rows_written = upsert_ohlcv_batch(conn, batch)
                    if rows_written > 0:
                        consumer.commit(asynchronous=False) # Commit offset after successful write
                    batch.clear()
                    last_write_time = time.time()

            except (json.JSONDecodeError, ValidationError) as e:
                logging.warning(f"Skipping malformed message. Error: {e}. Value: {msg.value()}")

    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    finally:
        # Final batch write and cleanup
        if batch:
            logging.info(f"Writing final batch of {len(batch)} bars...")
            upsert_ohlcv_batch(conn, batch)
            consumer.commit(asynchronous=False)
        conn.close()
        consumer.close()
        logging.info("Database connection and Kafka consumer closed.")


if __name__ == "__main__":
    main()