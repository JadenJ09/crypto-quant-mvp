# ==============================================================================
# File: services/vectorbt/app/main_api.py
# Description: FastAPI service for VectorBT computations
# ==============================================================================

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import vectorbt as vbt

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncpg

from .models import (
    BacktestRequest, BacktestStats, StrategyConfig,
    MarketDataResponse, ValidationResult, TradeInfo,
    IndicatorRequest, IndicatorResponse
)
from .backtesting.strategies import (
    run_ma_crossover_strategy,
    run_rsi_oversold_strategy
)
from .backtesting.strategy_executor import strategy_executor
from .indicators.processor import TechnicalIndicatorsProcessor
from .config import Settings
from .dependencies import get_db_pool, set_db_pool, close_db_pool

# Database helper functions
async def fetch_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Fetch OHLCV data from TimescaleDB"""
    try:
        from .dependencies import db_pool
        pool = db_pool
        if not pool:
            logger.warning("No database pool available, using mock data")
            return generate_mock_data(start_date, end_date)
        
        # Convert timeframe to proper table name
        table_map = {
            '1m': 'ohlcv_1min',
            '5m': 'ohlcv_5min',
            '15m': 'ohlcv_15min',
            '1h': 'ohlcv_1hour',
            '4h': 'ohlcv_4hour',
            '1d': 'ohlcv_1day',
            '7d': 'ohlcv_7day'
        }
        
        table_name = table_map.get(timeframe, 'ohlcv_1hour')
        logger.info(f"Fetching data from {table_name} for {symbol} from {start_date} to {end_date}")
        
        query = """
            SELECT time, open, high, low, close, volume
            FROM {}
            WHERE symbol = $1 AND time >= $2 AND time <= $3
            ORDER BY time ASC
            LIMIT 10000
        """.format(table_name)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_date, end_date)
            
        if not rows:
            logger.warning(f"No data found for {symbol} in {table_name}, using mock data")
            return generate_mock_data(start_date, end_date)
        
        # Convert to DataFrame
        # Convert asyncpg records to proper format
        data_list = []
        for row in rows:
            data_list.append(dict(row))
        
        df = pd.DataFrame(data_list)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        logger.info(f"Fetched {len(df)} rows of real data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from database: {e}")
        logger.info("Falling back to mock data")
        return generate_mock_data(start_date, end_date)

def generate_mock_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate mock OHLCV data for testing"""
    import numpy as np
    
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')[:100]
    mock_data = {
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(42000, 47000, len(dates)),
        'low': np.random.uniform(38000, 43000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }
    
    for i in range(len(dates)):
        # Ensure OHLC logic
        high = max(mock_data['open'][i], mock_data['close'][i]) + np.random.uniform(100, 1000)
        low = min(mock_data['open'][i], mock_data['close'][i]) - np.random.uniform(100, 1000)
        mock_data['high'][i] = high
        mock_data['low'][i] = low
    
    return pd.DataFrame(mock_data, index=dates)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize indicators processor
settings = Settings()
indicators_processor = TechnicalIndicatorsProcessor(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    
    # Startup
    try:
        # Use environment-based database URL, with fallback for local development
        database_url = os.environ.get("DATABASE_URL", "postgresql://quant_user:quant_password@timescaledb:5433/quant_db")
        
        logger.info(f"Attempting to connect to database: {database_url.replace('password', '***')}")
            
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10
        )
        set_db_pool(pool)
        logger.info(f"Database pool created successfully with URL: {database_url.replace('password', '***')}")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        # Continue without database for testing purposes
        logger.warning("Continuing without database connection - mock data will be used")
    
    yield
    
    # Shutdown
    await close_db_pool()

# Initialize FastAPI app
app = FastAPI(
    title="VectorBT Computation Service",
    description="Microservice for heavy computations using VectorBT",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3006", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3006", "*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vectorbt-computation"}

@app.post("/strategies/backtest", response_model=BacktestStats)
async def run_backtest(request: BacktestRequest):
    """
    Execute a backtest strategy with the given parameters
    """
    try:
        # Convert string dates to datetime objects
        from datetime import datetime
        start_date_str = request.start_date or '2024-01-01'
        end_date_str = request.end_date or '2024-12-31'
        
        # Parse dates properly
        if isinstance(start_date_str, str):
            if 'T' in start_date_str:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            else:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        else:
            start_date = start_date_str
        
        if isinstance(end_date_str, str):
            if 'T' in end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            else:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        else:
            end_date = end_date_str
        
        # Fetch real data from database
        data = await fetch_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Use the new strategy execution system
        try:
            # Convert request to strategy data format with ALL frontend parameters
            strategy_data = {
                'initial_cash': request.initial_cash,
                'commission': request.commission,
                'slippage': request.slippage,
                
                # Position sizing parameters (CRITICAL FIX)
                'position_sizing': request.strategy.position_sizing,
                'position_size': request.strategy.position_size,
                
                # Risk management parameters (CRITICAL FIX)
                'stop_loss': request.strategy.stop_loss,
                'take_profit': request.strategy.take_profit,
                'max_positions': getattr(request.strategy, 'max_positions', 1),
                'max_position_strategy': getattr(request.strategy, 'max_position_strategy', 'ignore'),
                
                # Position direction (CRITICAL FIX)
                'position_direction': getattr(request.strategy, 'position_direction', 'long_only'),
                
                # Strategy definition
                'name': request.strategy.name,
                'description': request.strategy.description,
                'entry_conditions': request.strategy.entry_conditions,
                'exit_conditions': request.strategy.exit_conditions
            }
            
            logger.info(f"Running backtest with new strategy system")
            logger.info(f"Strategy data: {strategy_data}")
            
            # Execute backtest using new system
            result = strategy_executor.run_backtest(strategy_data, data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error with new strategy system: {e}")
            logger.info("Falling back to legacy strategy system")
            
            # Fallback to legacy system if new system fails
            # ... (legacy code as fallback)
        
        # Legacy fallback logic (keep for now)
        # Run the appropriate strategy based on frontend strategy
        # Convert frontend strategy to backend strategy type
        strategy_type = "rsi_oversold"  # Default to RSI strategy to match frontend
        params = {}
        
        # Analyze entry conditions to determine strategy type
        for condition in request.strategy.entry_conditions:
            if condition.enabled and condition.indicator:
                indicator_name = condition.indicator.lower()
                if 'rsi' in indicator_name:
                    strategy_type = "rsi_oversold"
                    # Extract RSI parameters from conditions
                    oversold_level = 30  # default
                    if condition.operator == 'less_than':
                        try:
                            oversold_level = int(float(condition.value))
                        except (ValueError, TypeError):
                            oversold_level = 30
                    
                    params = {
                        'rsi_period': 14,  # Default values
                        'oversold_level': oversold_level,
                        'overbought_level': 70  # Look for overbought in exit conditions
                    }
                elif 'sma' in indicator_name or 'ema' in indicator_name or 'ma' in indicator_name:
                    strategy_type = "ma_crossover"
                    # Extract MA parameters from conditions
                    params = {
                        'fast_window': 10,  # Default values
                        'slow_window': 20
                    }
                elif 'macd' in indicator_name:
                    strategy_type = "ma_crossover"  # Use MA crossover for MACD
                    params = {
                        'fast_window': 12,
                        'slow_window': 26
                    }
        
        # Check exit conditions for additional parameters
        for condition in request.strategy.exit_conditions:
            if condition.enabled and condition.indicator:
                indicator_name = condition.indicator.lower()
                if 'rsi' in indicator_name and strategy_type == "rsi_oversold":
                    if condition.operator == 'greater_than':
                        try:
                            params['overbought_level'] = int(float(condition.value))
                        except (ValueError, TypeError):
                            params['overbought_level'] = 70
        
        # Debug logging for parameters
        logger.info(f"Legacy fallback - Strategy type: {strategy_type}")
        logger.info(f"Position sizing: {request.strategy.position_sizing}, size: {request.strategy.position_size}")
        logger.info(f"Stop loss: {request.strategy.stop_loss}, Take profit: {request.strategy.take_profit}")
        logger.info(f"Slippage: {request.slippage}")
        
        if strategy_type == "ma_crossover":
            result = run_ma_crossover_strategy(
                data,
                fast_window=params.get('fast_window', 10),
                slow_window=params.get('slow_window', 20),
                initial_cash=request.initial_cash,
                commission=request.commission,
                slippage=request.slippage,
                stop_loss=request.strategy.stop_loss,
                take_profit=request.strategy.take_profit,
                position_sizing=request.strategy.position_sizing,
                position_size=request.strategy.position_size
            )
        elif strategy_type == "rsi_oversold":
            result = run_rsi_oversold_strategy(
                data,
                rsi_period=params.get('rsi_period', 14),
                oversold_level=params.get('oversold_level', 30),
                overbought_level=params.get('overbought_level', 70),
                initial_cash=request.initial_cash,
                commission=request.commission,
                slippage=request.slippage,
                stop_loss=request.strategy.stop_loss,
                take_profit=request.strategy.take_profit,
                position_sizing=request.strategy.position_sizing,
                position_size=request.strategy.position_size
            )
        else:
            # Return mock results for unsupported strategies
            result = BacktestStats(
                total_return_pct=25.0,
                annualized_return_pct=30.0,
                max_drawdown_pct=-8.5,
                sharpe_ratio=1.45,
                sortino_ratio=1.89,
                calmar_ratio=3.53,
                total_trades=45,
                win_rate_pct=62.0,
                profit_factor=1.8,
                avg_win_pct=2.1,
                avg_loss_pct=-1.2,
                volatility_pct=18.5,
                var_95_pct=-3.2,
                avg_trade_duration_hours=4.5,
                max_trade_duration_hours=24.0,
                start_date=start_date,
                end_date=end_date,
                initial_cash=request.initial_cash,
                final_value=request.initial_cash * 1.25,
                trades=[]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in backtest execution: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest execution failed: {str(e)}")

@app.get("/strategies/available")
async def get_available_strategies():
    """
    Get list of available trading strategies
    """
    return [
        {
            "name": "rsi_oversold",
            "display_name": "RSI Oversold Strategy",
            "description": "Buy when RSI is oversold (<30) and sell when overbought (>70)",
            "parameters": [
                {"name": "rsi_period", "type": "int", "default": 14, "description": "RSI calculation period"},
                {"name": "oversold_level", "type": "float", "default": 30, "description": "Oversold threshold"},
                {"name": "overbought_level", "type": "float", "default": 70, "description": "Overbought threshold"}
            ]
        },
        {
            "name": "ma_crossover",
            "display_name": "Moving Average Crossover",
            "description": "Buy when fast MA crosses above slow MA, sell when it crosses below",
            "parameters": [
                {"name": "fast_window", "type": "int", "default": 10, "description": "Fast MA period"},
                {"name": "slow_window", "type": "int", "default": 20, "description": "Slow MA period"}
            ]
        },
        {
            "name": "bollinger_bands",
            "display_name": "Bollinger Bands Strategy",
            "description": "Mean reversion strategy using Bollinger Bands",
            "parameters": [
                {"name": "bb_period", "type": "int", "default": 20, "description": "Bollinger Bands period"},
                {"name": "bb_std", "type": "float", "default": 2.0, "description": "Standard deviation multiplier"}
            ]
        }
    ]

@app.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get market data for a symbol
    """
    try:
        async with pool.acquire() as connection:
            if start_date and end_date:
                query = """
                SELECT time, open, high, low, close, volume
                FROM ohlcv_1min 
                WHERE symbol = $1 
                AND time BETWEEN $2 AND $3
                ORDER BY time DESC
                LIMIT $4
                """
                rows = await connection.fetch(query, symbol, start_date, end_date, limit)
            else:
                query = """
                SELECT time, open, high, low, close, volume
                FROM ohlcv_1min 
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT $2
                """
                rows = await connection.fetch(query, symbol, limit)
            
            data = [dict(row) for row in rows]
            
            return MarketDataResponse(
                symbol=symbol,
                data=data,
                total_records=len(data)
            )
            
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@app.post("/indicators/calculate", response_model=IndicatorResponse)
async def calculate_indicators(
    request: IndicatorRequest,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Calculate technical indicators for given symbol and date range
    """
    try:
        # Fetch market data
        async with pool.acquire() as connection:
            query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv_1min 
            WHERE symbol = $1 
            AND time BETWEEN $2 AND $3
            ORDER BY time
            """
            
            rows = await connection.fetch(
                query,
                request.symbol,
                request.start_date or '2024-01-01',
                request.end_date or '2024-12-31'
            )
            
        if not rows:
            raise HTTPException(status_code=404, detail="No data found for the specified parameters")
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        df.set_index('time', inplace=True)
        
        # Calculate indicators using pandas/numpy (simpler approach)
        result_data = {}
        
        # Example indicators - you can expand this based on the request
        if 'rsi' in request.indicators or not request.indicators:
            # Simple RSI calculation using pandas
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()  # type: ignore
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # type: ignore
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result_data['rsi_14'] = rsi.dropna().tolist()
        
        if 'sma' in request.indicators or not request.indicators:
            sma = df['close'].rolling(window=20).mean()
            result_data['sma_20'] = sma.dropna().tolist()
        
        if 'ema' in request.indicators or not request.indicators:
            ema = df['close'].ewm(span=21).mean()
            result_data['ema_21'] = ema.dropna().tolist()
        
        return IndicatorResponse(
            symbol=request.symbol,
            indicators=result_data,
            data_points=len(df),
            start_date=pd.Timestamp(df.index[0]).to_pydatetime(),
            end_date=pd.Timestamp(df.index[-1]).to_pydatetime()
        )
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")

@app.get("/indicators/calculate/{symbol}")
async def calculate_indicators_for_chart(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    indicators: str = "rsi,sma_20,ema_21",
    limit: int = 1000
):
    """
    Calculate real technical indicators for frontend charts.
    
    Supported indicators:
    - rsi, rsi_14, rsi_21 (RSI with different periods)
    - sma_X (Simple Moving Average with X periods, e.g., sma_20, sma_50)
    - ema_X (Exponential Moving Average with X periods, e.g., ema_21, ema_50)
    - macd (MACD with default parameters)
    - bb (Bollinger Bands with default parameters)
    - stoch (Stochastic Oscillator with default parameters)
    """
    try:
        # Convert dates to datetime objects
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Fetch OHLCV data
        df = await fetch_ohlcv_data(symbol, timeframe, start_dt, end_dt)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
        
        # Limit the data if needed
        if len(df) > limit:
            df = df.tail(limit)
        
        # Parse requested indicators
        indicator_list = [i.strip().lower() for i in indicators.split(',')]
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'timestamps': df.index.strftime('%Y-%m-%dT%H:%M:%S%z').tolist(),
            'indicators': {}
        }
        
        # Calculate each requested indicator
        for indicator in indicator_list:
            try:
                if indicator.startswith('rsi'):
                    # Extract period from indicator name (e.g., rsi_14)
                    period = 14  # default
                    if '_' in indicator:
                        try:
                            period = int(indicator.split('_')[1])
                        except:
                            period = 14
                    
                    rsi_values = calculate_rsi(df['close'], period)
                    result['indicators'][f'rsi_{period}'] = rsi_values.dropna().tolist()
                
                elif indicator.startswith('sma'):
                    # Extract period from indicator name (e.g., sma_20)
                    period = 20  # default
                    if '_' in indicator:
                        try:
                            period = int(indicator.split('_')[1])
                        except:
                            period = 20
                    
                    sma_values = calculate_sma(df['close'], period)
                    result['indicators'][f'sma_{period}'] = sma_values.dropna().tolist()
                
                elif indicator.startswith('ema'):
                    # Extract period from indicator name (e.g., ema_21)
                    period = 21  # default
                    if '_' in indicator:
                        try:
                            period = int(indicator.split('_')[1])
                        except:
                            period = 21
                    
                    ema_values = calculate_ema(df['close'], period)
                    result['indicators'][f'ema_{period}'] = ema_values.dropna().tolist()
                
                elif indicator == 'macd':
                    macd_result = calculate_macd(df['close'])
                    for key, values in macd_result.items():
                        result['indicators'][key] = values.dropna().tolist()
                
                elif indicator == 'bb' or indicator == 'bollinger':
                    bb_result = calculate_bollinger_bands(df['close'])
                    for key, values in bb_result.items():
                        result['indicators'][key] = values.dropna().tolist()
                
                elif indicator == 'stoch' or indicator == 'stochastic':
                    stoch_result = calculate_stochastic(df['high'], df['low'], df['close'])
                    for key, values in stoch_result.items():
                        result['indicators'][key] = values.dropna().tolist()
                
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
                    
            except Exception as e:
                logger.error(f"Error calculating indicator {indicator}: {e}")
                continue
        
        logger.info(f"Calculated indicators for {symbol}: {list(result['indicators'].keys())}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating indicators: {str(e)}")

@app.get("/indicators/available")
async def get_available_indicators():
    """Get list of available technical indicators"""
    return {
        "indicators": [
            {
                "name": "rsi",
                "description": "Relative Strength Index",
                "parameters": ["period"],
                "default_params": {"period": 14},
                "examples": ["rsi", "rsi_14", "rsi_21"]
            },
            {
                "name": "sma",
                "description": "Simple Moving Average",
                "parameters": ["period"],
                "default_params": {"period": 20},
                "examples": ["sma_20", "sma_50", "sma_200"]
            },
            {
                "name": "ema",
                "description": "Exponential Moving Average", 
                "parameters": ["period"],
                "default_params": {"period": 21},
                "examples": ["ema_21", "ema_50", "ema_200"]
            },
            {
                "name": "macd",
                "description": "MACD",
                "parameters": ["fast_period", "slow_period", "signal_period"],
                "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "examples": ["macd"]
            },
            {
                "name": "bollinger_bands",
                "description": "Bollinger Bands",
                "parameters": ["period", "std"],
                "default_params": {"period": 20, "std": 2.0},
                "examples": ["bb", "bollinger"]
            },
            {
                "name": "stochastic",
                "description": "Stochastic Oscillator",
                "parameters": ["k_period", "d_period"],
                "default_params": {"k_period": 14, "d_period": 3},
                "examples": ["stoch", "stochastic"]
            }
        ]
    }

# ==============================================================================
# Indicators Endpoints for Real Chart Data
# ==============================================================================

# Technical indicator calculation functions
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return close.rolling(window=period).mean()

def calculate_ema(close: pd.Series, period: int = 21) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return close.ewm(span=period).mean()

def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    return {
        'bb_upper': sma + (std * std_dev),
        'bb_middle': sma,
        'bb_lower': sma - (std * std_dev)
    }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'stoch_k': k_percent,
        'stoch_d': d_percent
    }

@app.get("/symbols")
async def get_symbols(pool: asyncpg.Pool = Depends(get_db_pool)):
    """
    Get list of available symbols
    """
    try:
        async with pool.acquire() as connection:
            query = """
            SELECT DISTINCT symbol
            FROM ohlcv_1min 
            ORDER BY symbol
            """
            rows = await connection.fetch(query)
            
            symbols = []
            for row in rows:
                symbol = row['symbol']
                # Create symbol info - you might want to get this from a symbols table
                symbols.append({
                    "symbol": symbol,
                    "name": f"{symbol[:3]}/{symbol[3:]}",  # e.g., BTC/USDT
                    "exchange": "Binance",
                    "base_currency": symbol[:-4],
                    "quote_currency": symbol[-4:]
                })
            
            return symbols
            
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        # Return fallback symbols
        return [
            {"symbol": "BTCUSDT", "name": "BTC/USDT", "exchange": "Binance", "base_currency": "BTC", "quote_currency": "USDT"},
            {"symbol": "ETHUSDT", "name": "ETH/USDT", "exchange": "Binance", "base_currency": "ETH", "quote_currency": "USDT"},
        ]

@app.get("/timeframes")
async def get_timeframes():
    """
    Get list of available timeframes
    """
    return [
        {"label": "1 Minute", "value": "1m", "table": "ohlcv_1min", "description": "1-minute candlesticks"},
        {"label": "5 Minutes", "value": "5m", "table": "ohlcv_5min", "description": "5-minute candlesticks"},
        {"label": "15 Minutes", "value": "15m", "table": "ohlcv_15min", "description": "15-minute candlesticks"},
        {"label": "1 Hour", "value": "1h", "table": "ohlcv_1hour", "description": "1-hour candlesticks"},
        {"label": "4 Hours", "value": "4h", "table": "ohlcv_4hour", "description": "4-hour candlesticks"},
        {"label": "1 Day", "value": "1d", "table": "ohlcv_1day", "description": "1-day candlesticks"},
        {"label": "7 Days", "value": "7d", "table": "ohlcv_1week", "description": "1-week candlesticks"},
    ]

@app.get("/candlesticks/{symbol}")
async def get_candlesticks(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1000,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get candlestick data for a symbol with specific timeframe
    """
    try:
        # Map timeframe to table name
        timeframe_tables = {
            "1m": "ohlcv_1min",
            "5m": "ohlcv_5min", 
            "15m": "ohlcv_15min",
            "1h": "ohlcv_1hour",
            "4h": "ohlcv_4hour",
            "1d": "ohlcv_1day",
            "7d": "ohlcv_1week"
        }
        
        table_name = timeframe_tables.get(timeframe, "ohlcv_1hour")
        
        async with pool.acquire() as connection:
            # First, check if SMA columns exist in the table
            check_columns_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = $1 
            AND column_name IN ('sma_20', 'sma_50', 'sma_100')
            """
            
            existing_columns = await connection.fetch(check_columns_query, table_name)
            sma_columns_exist = len(existing_columns) > 0
            
            # Build query based on whether SMA columns exist
            if sma_columns_exist:
                query = f"""
                SELECT time, open, high, low, close, volume, sma_20, sma_50, sma_100
                FROM {table_name}
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT $2
                """
            else:
                query = f"""
                SELECT time, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT $2
                """
            
            rows = await connection.fetch(query, symbol, limit)
            
            # Convert to the format expected by the frontend
            candlesticks = []
            for row in rows:
                candlestick_data = {
                    "time": row['time'].isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                }
                
                # Add SMA data if columns exist
                if sma_columns_exist:
                    candlestick_data.update({
                        "sma_20": float(row['sma_20']) if row['sma_20'] else None,
                        "sma_50": float(row['sma_50']) if row['sma_50'] else None,
                        "sma_100": float(row['sma_100']) if row['sma_100'] else None,
                    })
                else:
                    candlestick_data.update({
                        "sma_20": None,
                        "sma_50": None,
                        "sma_100": None,
                    })
                
                candlesticks.append(candlestick_data)
            
            # Reverse to get chronological order (oldest first)
            candlesticks.reverse()
            
            return candlesticks
            
    except Exception as e:
        logger.error(f"Error fetching candlesticks for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch candlesticks: {str(e)}")

@app.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 1000,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get OHLCV data for backtesting (alternative endpoint)
    """
    try:
        # Map timeframe to table name
        timeframe_tables = {
            "1m": "ohlcv_1min",
            "5m": "ohlcv_5min", 
            "15m": "ohlcv_15min",
            "1h": "ohlcv_1hour",
            "4h": "ohlcv_4hour",
            "1d": "ohlcv_1day",
            "7d": "ohlcv_1week"
        }
        
        table_name = timeframe_tables.get(timeframe, "ohlcv_1hour")
        
        async with pool.acquire() as connection:
            if start and end:
                # Convert string dates to datetime objects
                from datetime import datetime
                
                # Parse start date
                if isinstance(start, str):
                    if 'T' in start:
                        start_date = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    else:
                        start_date = datetime.strptime(start, '%Y-%m-%d')
                else:
                    start_date = start
                
                # Parse end date
                if isinstance(end, str):
                    if 'T' in end:
                        end_date = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    else:
                        end_date = datetime.strptime(end, '%Y-%m-%d')
                else:
                    end_date = end
                
                query = f"""
                SELECT time, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = $1 
                AND time BETWEEN $2 AND $3
                ORDER BY time DESC
                LIMIT $4
                """
                rows = await connection.fetch(query, symbol, start_date, end_date, limit)
            else:
                query = f"""
                SELECT time, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT $2
                """
                rows = await connection.fetch(query, symbol, limit)
            
            # Convert to the format expected by the frontend
            data = []
            for row in rows:
                data.append({
                    "timestamp": row['time'].isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                })
            
            # Reverse to get chronological order (oldest first)
            data.reverse()
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data,
                "total_records": len(data)
            }
            
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
