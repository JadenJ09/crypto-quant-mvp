# ==============================================================================
# File: services/api/app/data_access.py
# Description: Handle data fetching from the database for analysis.
# ==============================================================================

import logging
import pandas as pd
import psycopg
from typing import List, Optional, Dict, Any
from datetime import datetime

def fetch_ohlcv_as_dataframe(conn_info: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetches OHLCV data for a given symbol and timeframe from TimescaleDB
    and returns it as a pandas DataFrame, which is the required format for vectorbt.
    
    NOTE: This uses a SYNCHRONOUS connection because pandas and vectorbt are
    synchronous libraries. This function is intended to be run in a separate thread.
    """
    table_name = f"ohlcv_{timeframe}"
    query = f"""
        SELECT time, open, close, high, low, volume
        FROM {table_name}
        WHERE symbol = %s
        ORDER BY time ASC;
    """
    logging.info(f"Executing synchronous query on table {table_name} for symbol {symbol}")
    
    try:
        with psycopg.connect(conn_info) as conn:
            # pandas.read_sql is highly optimized for this task.
            df = pd.read_sql(query, conn, params=(symbol.upper(),), index_col='time')
            if df.empty:
                logging.warning(f"No data returned for symbol {symbol} and timeframe {timeframe}")
            else:
                logging.info(f"Fetched {len(df)} rows for {symbol} into pandas DataFrame.")
            return df
    except Exception as e:
        logging.error(f"Error fetching data as DataFrame: {e}")
        # Return an empty DataFrame on error
        return pd.DataFrame()

def fetch_ohlcv_with_indicators(
    conn_info: str, 
    symbol: str, 
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetches OHLCV data with technical indicators for backtesting.
    
    Args:
        conn_info: Database connection string
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '5min', '1hour', '1day')
        start_date: Optional start date filter
        end_date: Optional end date filter
        indicators: List of indicator names to fetch (None = fetch all available)
        
    Returns:
        DataFrame with OHLCV data and requested indicators
    """
    table_name = f"ohlcv_{timeframe}"
    
    # Base columns
    base_columns = ["time", "open", "high", "low", "close", "volume"]
    
    # Available indicators based on your schema
    available_indicators = [
        "rsi_14", "rsi_21", "rsi_30",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent_b",
        "ema_9", "ema_21", "ema_50", "ema_100", "ema_200",
        "sma_20", "sma_50", "sma_100", "sma_200",
        "vwap", "vwema_ribbon_fast", "vwema_ribbon_slow",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
        "atr_14", "stoch_k", "stoch_d", "williams_r", "cci_20",
        "momentum_10", "momentum_20", "roc_10", "roc_20",
        "adx_14", "di_plus"
    ]
    
    # Determine which indicators to fetch
    if indicators is None:
        # Fetch all available indicators
        indicator_columns = available_indicators
    else:
        # Only fetch requested indicators that are available
        indicator_columns = [ind for ind in indicators if ind in available_indicators]
    
    # Build the SELECT clause
    all_columns = base_columns + indicator_columns
    columns_str = ", ".join(all_columns)
    
    # Build WHERE clause
    where_conditions = ["symbol = %s"]
    params: List[Any] = [symbol.upper()]
    
    if start_date:
        where_conditions.append("time >= %s")
        params.append(start_date)
    
    if end_date:
        where_conditions.append("time <= %s")
        params.append(end_date)
    
    where_clause = " AND ".join(where_conditions)
    
    query = f"""
        SELECT {columns_str}
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY time ASC;
    """
    
    logging.info(f"Fetching OHLCV with indicators for {symbol} ({timeframe})")
    logging.debug(f"Query: {query}")
    
    try:
        with psycopg.connect(conn_info) as conn:
            df = pd.read_sql(query, conn, params=params, index_col='time')
            if df.empty:
                logging.warning(f"No data returned for symbol {symbol} and timeframe {timeframe}")
            else:
                logging.info(f"Fetched {len(df)} rows with {len(indicator_columns)} indicators")
            return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV with indicators: {e}")
        return pd.DataFrame()

def get_available_indicators(conn_info: str, timeframe: str) -> List[str]:
    """
    Get list of available indicator columns for a specific timeframe table.
    
    Args:
        conn_info: Database connection string
        timeframe: Timeframe (e.g., '5min', '1hour', '1day')
        
    Returns:
        List of available indicator column names
    """
    table_name = f"ohlcv_{timeframe}"
    
    query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s 
        AND column_name NOT IN ('time', 'symbol', 'open', 'high', 'low', 'close', 'volume')
        ORDER BY column_name;
    """
    
    try:
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (table_name,))
                results = cur.fetchall()
                return [row[0] for row in results]
    except Exception as e:
        logging.error(f"Error fetching available indicators: {e}")
        return []

def validate_symbol_and_timeframe(conn_info: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Validate that symbol and timeframe exist in the database and return metadata.
    
    Args:
        conn_info: Database connection string
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Dictionary with validation results and metadata
    """
    table_name = f"ohlcv_{timeframe}"
    
    # Check if table exists
    table_exists_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
    """
    
    # Check if symbol has data
    symbol_exists_query = f"""
        SELECT 
            COUNT(*) as record_count,
            MIN(time) as earliest_date,
            MAX(time) as latest_date
        FROM {table_name}
        WHERE symbol = %s;
    """
    
    try:
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                # Check table existence
                cur.execute(table_exists_query, (table_name,))
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    return {
                        "valid": False,
                        "error": f"Timeframe '{timeframe}' is not available",
                        "table_exists": False
                    }
                
                # Check symbol data
                cur.execute(symbol_exists_query, (symbol.upper(),))
                result = cur.fetchone()
                record_count, earliest_date, latest_date = result
                
                return {
                    "valid": record_count > 0,
                    "table_exists": True,
                    "record_count": record_count,
                    "earliest_date": earliest_date,
                    "latest_date": latest_date,
                    "error": None if record_count > 0 else f"No data found for symbol '{symbol}'"
                }
                
    except Exception as e:
        logging.error(f"Error validating symbol and timeframe: {e}")
        return {
            "valid": False,
            "error": f"Database error: {str(e)}",
            "table_exists": None
        }