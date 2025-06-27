# services/api/app/main.py
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import psycopg
from psycopg_pool import AsyncConnectionPool
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://quant_user:quant_password@timescaledb:5432/quant_db")

# --- Database Connection Pool ---
# In an async application, managing a pool of connections is more efficient
# than creating a new connection for every request.
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    global db_pool
    logging.info("API starting up... Creating database connection pool.")
    try:
        # Use AsyncConnectionPool from psycopg_pool
        db_pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=2,
            max_size=10
        )
        # Wait for the pool to be ready
        await db_pool.wait()
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.critical(f"Failed to create database connection pool: {e}")
        logging.critical(f"DATABASE_URL: {DATABASE_URL}")
        # In a real app, you might exit if the DB isn't available
    yield
    # This code runs on shutdown
    if db_pool:
        logging.info("API shutting down... Closing database connection pool.")
        await db_pool.close()

# --- Pydantic Models (Data Transfer Objects) ---
# This defines the shape of the data we will return to the client.
# It ensures our API response is consistent and well-structured.
class OHLCVOut(BaseModel):
    time: datetime
    open: float = Field(..., alias='o')
    high: float = Field(..., alias='h')
    low: float = Field(..., alias='l')
    close: float = Field(..., alias='c')
    volume: float = Field(..., alias='v')

    class Config:
        # Allows Pydantic to create the model from database row objects.
        from_attributes = True
        # Allows using aliases for shorter JSON field names.
        populate_by_name = True

class CandlestickData(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class TimeframeInfo(BaseModel):
    label: str
    value: str
    table: str
    description: str

class SymbolInfo(BaseModel):
    symbol: str
    name: str
    exchange: str
    base_currency: str
    quote_currency: str

# --- FastAPI Application ---
app = FastAPI(
    title="Crypto-Quant-MVP API",
    description="Serves OHLCV data and backtesting results.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware: Allows our frontend (running on localhost:3000, for example)
# to make requests to this API (running on localhost:8001).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints (Routers) ---
@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "Crypto-Quant-MVP API is running"}

# Timeframe mapping for table selection
TIMEFRAME_TABLES = {
    "1m": "ohlcv_1min",
    "5m": "ohlcv_5min", 
    "15m": "ohlcv_15min",
    "1h": "ohlcv_1hour",
    "4h": "ohlcv_4hour",
    "1d": "ohlcv_1day",
    "7d": "ohlcv_7day"
}

@app.get("/timeframes", response_model=List[TimeframeInfo])
async def get_available_timeframes():
    """Get list of available timeframes for charting."""
    timeframes = [
        {"label": "1 Minute", "value": "1m", "table": "ohlcv_1min", "description": "1-minute candlesticks"},
        {"label": "5 Minutes", "value": "5m", "table": "ohlcv_5min", "description": "5-minute candlesticks"},
        {"label": "15 Minutes", "value": "15m", "table": "ohlcv_15min", "description": "15-minute candlesticks"},
        {"label": "1 Hour", "value": "1h", "table": "ohlcv_1hour", "description": "1-hour candlesticks"},
        {"label": "4 Hours", "value": "4h", "table": "ohlcv_4hour", "description": "4-hour candlesticks"},
        {"label": "1 Day", "value": "1d", "table": "ohlcv_1day", "description": "Daily candlesticks"},
        {"label": "7 Days", "value": "7d", "table": "ohlcv_7day", "description": "Weekly candlesticks"}
    ]
    return timeframes

@app.get("/symbols", response_model=List[SymbolInfo])
async def get_available_symbols():
    """Get list of available trading symbols."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    
    query = """
        SELECT DISTINCT symbol 
        FROM ohlcv_1min 
        ORDER BY symbol;
    """
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(query)
                results = await acur.fetchall()
                
                symbols = []
                for row in results:
                    symbol = row[0]
                    # Parse symbol (assuming format like BTCUSDT)
                    if symbol.endswith('USDT'):
                        base = symbol[:-4]
                        quote = 'USDT'
                    elif symbol.endswith('BTC'):
                        base = symbol[:-3]
                        quote = 'BTC'
                    elif symbol.endswith('ETH'):
                        base = symbol[:-3]
                        quote = 'ETH'
                    else:
                        base = symbol
                        quote = 'USD'
                    
                    symbols.append({
                        "symbol": symbol,
                        "name": f"{base}/{quote}",
                        "exchange": "Binance",
                        "base_currency": base,
                        "quote_currency": quote
                    })
                
                return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching symbols.")

@app.get("/ohlcv/{symbol}", response_model=List[OHLCVOut])
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = Query("1m", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 7d)"),
    limit: int = Query(1000, gt=0, le=5000),
    start_time: Optional[datetime] = Query(None, description="Start time for data range"),
    end_time: Optional[datetime] = Query(None, description="End time for data range")
):
    """
    Fetches OHLCV data for a given symbol and timeframe from TimescaleDB.
    """
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    
    # Validate timeframe
    if timeframe not in TIMEFRAME_TABLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timeframe. Must be one of: {', '.join(TIMEFRAME_TABLES.keys())}"
        )
    
    table_name = TIMEFRAME_TABLES[timeframe]
    
    # Build query based on time range
    if start_time and end_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time DESC
            LIMIT %s;
        """
        params = (symbol.upper(), start_time, end_time, limit)
    elif start_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s
            ORDER BY time DESC
            LIMIT %s;
        """
        params = (symbol.upper(), start_time, limit)
    elif end_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time <= %s
            ORDER BY time DESC
            LIMIT %s;
        """
        params = (symbol.upper(), end_time, limit)
    else:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT %s;
        """
        params = (symbol.upper(), limit)
    
    logging.info(f"Fetching {limit} records for symbol {symbol.upper()} with timeframe {timeframe}")
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(query, params)
                # Fetch all results from the cursor
                results = await acur.fetchall()
                # We need to map the column names to the Pydantic model fields
                columns = [desc[0] for desc in acur.description]
                data = [dict(zip(columns, row)) for row in results]
                # Pydantic will automatically validate and serialize this list of dicts
                return data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} with timeframe {timeframe}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching data.")

@app.get("/candlesticks/{symbol}", response_model=List[CandlestickData])
async def get_candlestick_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 7d)"),
    start_time: Optional[datetime] = Query(None, description="Start time for data range"),
    end_time: Optional[datetime] = Query(None, description="End time for data range")
):
    """
    Fetches candlestick data optimized for TradingView charts.
    Returns data in ascending time order (oldest first) for proper chart rendering.
    """
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    
    # Validate timeframe
    if timeframe not in TIMEFRAME_TABLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timeframe. Must be one of: {', '.join(TIMEFRAME_TABLES.keys())}"
        )
    
    table_name = TIMEFRAME_TABLES[timeframe]
    
    # Build query based on time range (order by time ASC for candlestick charts)
    if start_time and end_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC;
        """
        params = (symbol.upper(), start_time, end_time)
    elif start_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s
            ORDER BY time ASC;
        """
        params = (symbol.upper(), start_time)
    elif end_time:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time <= %s
            ORDER BY time ASC;
        """
        params = (symbol.upper(), end_time)
    else:
        # Get all data for the symbol, ordered by time ASC
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time ASC;
        """
        params = (symbol.upper(),)
    
    logging.info(f"Fetching candlestick data: all records for {symbol.upper()} ({timeframe})")
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(query, params)
                results = await acur.fetchall()
                
                candlesticks = []
                for row in results:
                    candlesticks.append({
                        "time": row[0],  # timestamp as time
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5]
                    })
                
                return candlesticks
    except Exception as e:
        logging.error(f"Error fetching candlestick data for {symbol} ({timeframe}): {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching candlestick data.")


