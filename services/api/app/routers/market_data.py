# ==============================================================================
# File: services/api/app/routers/market_data.py
# Description: Router for market data endpoints (OHLCV, symbols, timeframes)
# ==============================================================================

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import logging

from ..models import (
    OHLCVOut, 
    CandlestickData, 
    TimeframeInfo, 
    SymbolInfo,
    IndicatorList
)
from ..dependencies import get_db_pool

router = APIRouter(prefix="/market", tags=["market_data"])

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

@router.get("/timeframes", response_model=List[TimeframeInfo])
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

@router.get("/symbols", response_model=List[SymbolInfo])
async def get_available_symbols():
    """Get list of available trading symbols."""
    db_pool = get_db_pool()
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

@router.get("/ohlcv/{symbol}", response_model=List[OHLCVOut])
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
    db_pool = get_db_pool()
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
                results = await acur.fetchall()
                columns = [desc[0] for desc in acur.description]
                data = [dict(zip(columns, row)) for row in results]
                return data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} with timeframe {timeframe}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching data.")

@router.get("/candlesticks/{symbol}", response_model=List[CandlestickData])
async def get_candlestick_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 7d)")
):
    """
    Fetches all candlestick data for a given symbol and timeframe with indicators.
    """
    db_pool = get_db_pool()
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    
    # Validate timeframe
    if timeframe not in TIMEFRAME_TABLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timeframe. Must be one of: {', '.join(TIMEFRAME_TABLES.keys())}"
        )
    
    table_name = TIMEFRAME_TABLES[timeframe]
    
    # Determine if this timeframe has SMA columns (1m doesn't have SMA)
    has_sma = timeframe != "1m"
    
    # Simple query to get all data for the symbol
    if has_sma:
        query = f"""
            SELECT time, open, high, low, close, volume, sma_20, sma_50, sma_100
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time ASC;
        """
    else:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time ASC;
        """
    
    params = (symbol.upper(),)
    
    logging.info(f"Fetching all candlestick data for {symbol.upper()} ({timeframe})")
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(query, params)
                results = await acur.fetchall()
                
                candlesticks = []
                for row in results:
                    if has_sma:
                        candlesticks.append({
                            "time": row[0],
                            "open": row[1],
                            "high": row[2], 
                            "low": row[3],
                            "close": row[4],
                            "volume": row[5],
                            "sma_20": row[6],
                            "sma_50": row[7],
                            "sma_100": row[8]
                        })
                    else:
                        candlesticks.append({
                            "time": row[0],
                            "open": row[1],
                            "high": row[2],
                            "low": row[3], 
                            "close": row[4],
                            "volume": row[5],
                            "sma_20": None,
                            "sma_50": None,
                            "sma_100": None
                        })
                
                return candlesticks
    except Exception as e:
        logging.error(f"Error fetching candlestick data for {symbol} ({timeframe}): {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching candlestick data.")
