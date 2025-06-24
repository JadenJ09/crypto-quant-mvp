# services/ingestor/app/main.py
import asyncio
import json
import logging
import os
import signal
from typing import List

from confluent_kafka import Producer
from pydantic import BaseModel, ValidationError
import websockets

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "raw.trades.crypto"
BINANCE_WS_BASE_URL = "wss://fstream.binance.com/ws/"

# Symbols defined in the PRD.
SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "xrpusdt",
    "trxusdt", "dogeusdt", "adausdt"
]

# --- Pydantic Data Validation Model ---
# This class defines the expected structure of a trade message from Binance.
# If a message doesn't match this structure, Pydantic will raise an error.
class BinanceTrade(BaseModel):
    e: str  # Event type
    E: int  # Event time
    s: str  # Symbol
    a: int  # Aggregate trade ID
    p: str  # Price
    q: str  # Quantity
    f: int  # First trade ID
    l: int  # Last trade ID
    T: int  # Trade time
    m: bool # Is the buyer the market maker?

# --- Kafka Producer ---
# Create a single, reusable Kafka producer instance.
producer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'quant-ingestor-producer',
    # For higher throughput in production, you might tune these:
    # 'linger.ms': '10',
    # 'batch.size': '131072', # 128KB
}
producer = Producer(producer_config)
logging.info(f"Initialized Kafka producer for brokers at {KAFKA_BOOTSTRAP_SERVERS}")

# --- Core Application Logic ---

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        logging.error(f"Message delivery failed for {msg.key()}: {err}")

async def handle_trade(payload: str):
    """
    Receives a raw JSON payload, validates it, and produces it to Kafka.
    """
    try:
        data = json.loads(payload)
        # Validate the data against our Pydantic model.
        trade = BinanceTrade(**data)
        
        # We use the symbol as the message key. This ensures all trades for the
        # same symbol go to the same Kafka partition, preserving order.
        key = trade.s.encode('utf-8')
        value = payload.encode('utf-8')

        # Asynchronously produce the message to our Kafka topic.
        producer.produce(KAFKA_TOPIC, key=key, value=value, callback=delivery_report)
        # producer.poll(0) is important to trigger the delivery callback.
        producer.poll(0)
        
    except ValidationError as e:
        logging.warning(f"Message validation failed: {e} - Payload: {payload[:200]}")
    except json.JSONDecodeError:
        logging.warning(f"Failed to decode JSON: {payload[:200]}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in handle_trade: {e}")

async def subscribe_to_symbol(symbol: str):
    """
    Establishes a persistent WebSocket connection for a single symbol.
    """
    stream_name = f"{symbol.lower()}@aggTrade"
    ws_url = f"{BINANCE_WS_BASE_URL}{stream_name}"
    logging.info(f"Subscribing to {ws_url}")
    
    while True: # Main reconnection loop
        try:
            async with websockets.connect(ws_url) as websocket:
                logging.info(f"Successfully connected to {symbol} stream.")
                # The 'for' loop handles messages as they arrive.
                # If the connection drops, it will exit and the 'except' block will handle it.
                async for message in websocket:
                    await handle_trade(message)
        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as e:
            logging.warning(f"Connection to {symbol} closed: {e}. Reconnecting in 5 seconds...")
        except Exception as e:
            logging.error(f"Error with {symbol} subscription: {e}. Reconnecting in 5 seconds...")
        
        await asyncio.sleep(5) # Wait before attempting to reconnect.

async def main():
    """
    The main entry point. Creates and runs all symbol subscription tasks concurrently.
    """
    # Create a list of asyncio tasks, one for each symbol subscription.
    tasks = [asyncio.create_task(subscribe_to_symbol(symbol)) for symbol in SYMBOLS]
    logging.info(f"Starting subscriptions for {len(SYMBOLS)} symbols.")
    
    # asyncio.gather runs all tasks in parallel. It will complete if one of the tasks errors out.
    await asyncio.gather(*tasks)

# --- Graceful Shutdown ---
async def shutdown(signal, loop):
    """Gracefully handles shutdown signals."""
    logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    for task in tasks:
        task.cancel()

    logging.info("Cancelling outstanding tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)
    # Flush any final messages from the Kafka producer
    producer.flush(10) 
    logging.info("Flushed Kafka producer. Shutting down.")
    loop.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    try:
        loop.create_task(main())
        loop.run_forever()
    finally:
        loop.close()
