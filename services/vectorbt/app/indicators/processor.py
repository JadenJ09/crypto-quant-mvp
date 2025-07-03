"""
Technical Indicators Processor using VectorBT, Pandas, NumPy, and TA Library

This module handles:
1. Multi-timeframe OHLCV resampling from 1m data
2. Technical indicators calculation using ta library, vectorbt, pandas, and numpy
3. Efficient batch processing for historical data
4. Real-time processing for streaming data
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import vectorbt as vbt

# Try importing ta library - fallback if not available
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("TA library not available. Using numpy/pandas implementations only.")

logger = logging.getLogger(__name__)


class TechnicalIndicatorsProcessor:
    """Processes technical indicators using TA library, VectorBT, pandas, and numpy"""
    
    def __init__(self, settings):
        self.settings = settings
        self.db_manager = None  # Will be injected
        
        # Configure vectorbt for performance if available
        try:
            if hasattr(vbt, 'settings') and hasattr(vbt.settings, 'set_theme'):
                vbt.settings.set_theme('dark')
        except (AttributeError, Exception):
            # Newer versions of vectorbt may not have this method or may fail
            pass
        
        # Set default frequency for pandas operations
        self.default_freq = 'T'  # Minute frequency
        
    def set_db_manager(self, db_manager):
        """Set database manager (dependency injection)"""
        self.db_manager = db_manager
        
    async def process_symbol_bulk(self, symbol: str):
        """Process all historical data for a symbol"""
        logger.info(f"📊 Starting bulk processing for {symbol}")
        
        if not self.db_manager:
            raise ValueError("Database manager not set. Call set_db_manager() first.")
        
        try:
            # Get all 1m data for symbol
            df_1m = await self.db_manager.get_1min_data(symbol)
            
            if df_1m.empty:
                logger.warning(f"No 1m data found for {symbol}")
                return
                
            logger.info(f"Processing {len(df_1m)} 1m records for {symbol}")
            
            # Process each timeframe
            for timeframe, config in self.settings.TIMEFRAMES.items():
                await self._process_timeframe_bulk(symbol, df_1m, timeframe, config)
                
            logger.info(f"✅ Bulk processing completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in bulk processing for {symbol}: {e}")
            raise
            
    async def _process_timeframe_bulk(
        self, 
        symbol: str, 
        df_1m: pd.DataFrame,
        timeframe: str,
        config: Dict[str, Any]
    ):
        """Process a specific timeframe for bulk data"""
        
        try:
            # Resample 1m data to target timeframe
            df_resampled = self._resample_ohlcv(df_1m, config['resample_freq'])
            
            if len(df_resampled) == 0:
                logger.warning(f"No data to resample for {symbol} {timeframe}")
                return
            
            # Always calculate basic OHLCV aggregation, but indicators only if enough data
            if len(df_resampled) >= config['min_periods']:
                # Calculate technical indicators
                df_with_indicators = self._calculate_indicators(df_resampled, timeframe)
            else:
                logger.warning(f"Insufficient data for {symbol} {timeframe} indicators: {len(df_resampled)} < {config['min_periods']}")
                logger.info(f"Creating basic OHLCV data for {symbol} {timeframe}: {len(df_resampled)} records")
                # Use resampled data without indicators (indicators will be NULL)
                df_with_indicators = df_resampled.copy()
            
            # Prepare data for database
            records = self._prepare_records_for_db(df_with_indicators, symbol)
            
            if records:
                # Batch insert to database
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    await self.db_manager.upsert_timeframe_data(config['table'], batch)
                    
                logger.info(f"✅ Processed {len(records)} {timeframe} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing {timeframe} for {symbol}: {e}")
            
    async def process_new_ohlcv(self, ohlcv_message: Dict[str, Any]):
        """Process new 1m OHLCV data point in real-time"""
        
        if not self.db_manager:
            logger.error("Database manager not set. Cannot process real-time data.")
            return
            
        symbol = ohlcv_message.get('symbol')
        if not symbol:
            logger.error("Symbol not found in OHLCV message")
            return
        
        timestamp = pd.to_datetime(ohlcv_message.get('time'))
        
        logger.debug(f"Processing new 1m data for {symbol} at {timestamp}")
        
        # For each timeframe, check if we need to update
        for timeframe, config in self.settings.TIMEFRAMES.items():
            try:
                await self._process_timeframe_realtime(symbol, timestamp, timeframe, config)
            except Exception as e:
                logger.error(f"Error processing realtime {timeframe} for {symbol}: {e}")
                
    async def _process_timeframe_realtime(
        self,
        symbol: str,
        new_timestamp: pd.Timestamp,
        timeframe: str,
        config: Dict[str, Any]
    ):
        """Process realtime update for specific timeframe"""
        
        # Determine if we need to calculate/update this timeframe
        timeframe_boundary = self._get_timeframe_boundary(new_timestamp, timeframe)
        
        # Get recent data for calculations (need enough periods for indicators)
        lookback_hours = self._get_lookback_hours(timeframe)
        start_time = new_timestamp - timedelta(hours=lookback_hours)
        
        # Get 1m data for the lookback period
        df_1m = await self.db_manager.get_1min_data(symbol, start_time, new_timestamp)
        
        if df_1m.empty or len(df_1m) < 50:  # Need minimum data
            return
            
        # Resample to target timeframe
        df_resampled = self._resample_ohlcv(df_1m, config['resample_freq'])
        
        if df_resampled.empty:
            return
            
        # Calculate indicators for the latest periods only
        df_with_indicators = self._calculate_indicators(df_resampled, timeframe)
        
        # Get only the latest complete timeframe period
        latest_period = df_with_indicators.iloc[-1:].copy()
        
        if not latest_period.empty:
            # Ensure we have a DataFrame, not a Series
            if isinstance(latest_period, pd.Series):
                latest_period = latest_period.to_frame().T
            records = self._prepare_records_for_db(latest_period, symbol)
            if records:
                await self.db_manager.upsert_timeframe_data(config['table'], records)
                logger.debug(f"Updated {timeframe} for {symbol}")
                
    def _resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample 1m OHLCV data to target frequency"""
        
        if df.empty:
            return pd.DataFrame()
            
        try:
            # Ensure time column is datetime index
            if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('time')
            
            # Resample OHLCV data
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return pd.DataFrame()
            
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate technical indicators using TA library and numpy/pandas implementations"""
        
        if df.empty or len(df) < 20:
            return df
            
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            if not TA_AVAILABLE:
                logger.warning("TA library not available. Using basic numpy/pandas implementations.")
                return self._calculate_indicators_numpy(result_df, timeframe)
            
            # RSI indicators
            for period in self.settings.rsi_periods_list:
                if len(df) >= period:
                    try:
                        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=period)
                        result_df[f'rsi_{period}'] = rsi_indicator.rsi()
                    except Exception as e:
                        logger.warning(f"Error calculating RSI {period}: {e}")
                    
            # EMA indicators
            for period in self.settings.ema_periods_list:
                if len(df) >= period:
                    try:
                        ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=period)
                        result_df[f'ema_{period}'] = ema_indicator.ema_indicator()
                    except Exception as e:
                        logger.warning(f"Error calculating EMA {period}: {e}")
                    
            # SMA indicators
            for period in self.settings.sma_periods_list:
                if len(df) >= period:
                    try:
                        sma_indicator = ta.trend.SMAIndicator(close=df['close'], window=period)
                        result_df[f'sma_{period}'] = sma_indicator.sma_indicator()
                    except Exception as e:
                        logger.warning(f"Error calculating SMA {period}: {e}")
                    
            # MACD
            if len(df) >= self.settings.macd_slow:
                try:
                    macd = ta.trend.MACD(
                        close=df['close'],
                        window_fast=self.settings.macd_fast,
                        window_slow=self.settings.macd_slow,
                        window_sign=self.settings.macd_signal
                    )
                    result_df['macd_line'] = macd.macd()
                    result_df['macd_signal'] = macd.macd_signal()
                    result_df['macd_histogram'] = macd.macd_diff()
                except Exception as e:
                    logger.warning(f"Error calculating MACD: {e}")
                
            # Bollinger Bands
            if len(df) >= self.settings.bb_period:
                try:
                    bb = ta.volatility.BollingerBands(
                        close=df['close'],
                        window=self.settings.bb_period,
                        window_dev=self.settings.bb_std
                    )
                    result_df['bb_upper'] = bb.bollinger_hband()
                    result_df['bb_middle'] = bb.bollinger_mavg()
                    result_df['bb_lower'] = bb.bollinger_lband()
                    result_df['bb_width'] = bb.bollinger_wband()
                    result_df['bb_percent_b'] = bb.bollinger_pband()
                except Exception as e:
                    logger.warning(f"Error calculating Bollinger Bands: {e}")
                
            # ATR
            if len(df) >= self.settings.atr_period:
                try:
                    result_df['atr_14'] = ta.volatility.AverageTrueRange(
                        high=df['high'], low=df['low'], close=df['close'],
                        window=self.settings.atr_period
                    ).average_true_range()
                except Exception as e:
                    logger.warning(f"Error calculating ATR: {e}")
                
            # Stochastic
            if len(df) >= self.settings.stoch_k_period:
                try:
                    stoch = ta.momentum.StochasticOscillator(
                        high=df['high'], low=df['low'], close=df['close'],
                        window=self.settings.stoch_k_period,
                        smooth_window=self.settings.stoch_d_period
                    )
                    result_df['stoch_k'] = stoch.stoch()
                    result_df['stoch_d'] = stoch.stoch_signal()
                except Exception as e:
                    logger.warning(f"Error calculating Stochastic: {e}")
                
            # Williams %R
            if len(df) >= 14:
                try:
                    result_df['williams_r'] = ta.momentum.WilliamsRIndicator(
                        high=df['high'], low=df['low'], close=df['close'], lbp=14
                    ).williams_r()
                except Exception as e:
                    logger.warning(f"Error calculating Williams %R: {e}")
                
            # CCI
            if len(df) >= 20:
                try:
                    result_df['cci_20'] = ta.trend.CCIIndicator(
                        high=df['high'], low=df['low'], close=df['close'], window=20
                    ).cci()
                except Exception as e:
                    logger.warning(f"Error calculating CCI: {e}")
                
            # Momentum and ROC
            if len(df) >= 10:
                try:
                    roc_10 = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()
                    result_df['momentum_10'] = roc_10
                    result_df['roc_10'] = roc_10  # Same calculation
                except Exception as e:
                    logger.warning(f"Error calculating ROC 10: {e}")
                
            if len(df) >= 20:
                try:
                    roc_20 = ta.momentum.ROCIndicator(close=df['close'], window=20).roc()
                    result_df['momentum_20'] = roc_20
                    result_df['roc_20'] = roc_20
                except Exception as e:
                    logger.warning(f"Error calculating ROC 20: {e}")
                
            # VWAP (only for intraday timeframes)
            if timeframe in ['5min', '15min', '1hour']:
                try:
                    # Calculate VWAP using cumulative values
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                    result_df['vwap'] = vwap
                except Exception as e:
                    logger.warning(f"Error calculating VWAP: {e}")
                
            # Volume indicators  
            if len(df) >= self.settings.volume_sma_period:
                try:
                    # Use simple moving average on volume directly
                    result_df['volume_sma_20'] = ta.trend.SMAIndicator(
                        close=df['volume'], 
                        window=self.settings.volume_sma_period
                    ).sma_indicator()
                    result_df['volume_ratio'] = df['volume'] / result_df['volume_sma_20']
                except Exception as e:
                    logger.warning(f"Error calculating Volume SMA: {e}")
                
            # OBV
            try:
                result_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                    close=df['close'], volume=df['volume']
                ).on_balance_volume()
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
            
            # Accumulation/Distribution Line
            try:
                result_df['ad_line'] = ta.volume.AccDistIndexIndicator(
                    high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
                ).acc_dist_index()
            except Exception as e:
                logger.warning(f"Error calculating A/D Line: {e}")
            
            # CMF (Chaikin Money Flow)
            if len(df) >= self.settings.cmf_period:
                try:
                    result_df['cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(
                        high=df['high'], low=df['low'], close=df['close'], volume=df['volume'],
                        window=self.settings.cmf_period
                    ).chaikin_money_flow()
                except Exception as e:
                    logger.warning(f"Error calculating CMF: {e}")
                
            # Volatility indicators using numpy
            if len(df) >= 20:
                try:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 10:
                        # Parkinson volatility
                        hl_ratio = np.log(df['high'] / df['low'])
                        parkinson_vol = np.sqrt(hl_ratio.rolling(20).var() * 252)
                        result_df['volatility_parkinson'] = parkinson_vol
                        
                        # Simple volatility
                        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                        result_df['volatility_garman_klass'] = rolling_vol
                except Exception as e:
                    logger.warning(f"Error calculating volatility indicators: {e}")
                    
            # VWEMA Ribbon
            if 'volume' in df.columns and len(df) >= 26:
                try:
                    def vwema(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
                        """Calculates the Volume-Weighted Exponential Moving Average (VWEMA)."""
                        return (close * volume).ewm(span=window, adjust=False).mean() / volume.ewm(span=window, adjust=False).mean()

                    result_df['vwema_ribbon_fast'] = vwema(result_df['close'], result_df['volume'], window=12)
                    result_df['vwema_ribbon_slow'] = vwema(result_df['close'], result_df['volume'], window=26)
                except Exception as e:
                    logger.warning(f"Error calculating VWEMA Ribbon for {timeframe}: {e}")

            # Ichimoku Cloud
            if len(df) >= 52:
                try:
                    ichimoku = ta.trend.IchimokuIndicator(
                        high=df['high'],
                        low=df['low'],
                        window1=9,    # Tenkan-sen
                        window2=26,   # Kijun-sen
                        window3=52    # Senkou Span B
                    )
                    result_df['ichimoku_tenkan'] = ichimoku.ichimoku_conversion_line()
                    result_df['ichimoku_kijun'] = ichimoku.ichimoku_base_line()
                    result_df['ichimoku_senkou_a'] = ichimoku.ichimoku_a()
                    result_df['ichimoku_senkou_b'] = ichimoku.ichimoku_b()
                except Exception as e:
                    logger.warning(f"Error calculating Ichimoku Cloud for {timeframe}: {e}")

            # ADX (Average Directional Movement Index)
            if len(df) >= 28: # ADX typically requires 2x the period
                try:
                    adx_indicator = ta.trend.ADXIndicator(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14
                    )
                    result_df['adx_14'] = adx_indicator.adx()
                    result_df['di_plus'] = adx_indicator.adx_pos()
                    result_df['di_minus'] = adx_indicator.adx_neg()
                except Exception as e:
                    logger.warning(f"Error calculating ADX for {timeframe}: {e}")

            # Historical Volatility (as a proxy for GARCH)
            if len(df) >= 21:
                try:
                    log_returns = np.log(df['close'] / df['close'].shift(1))
                    # Using a rolling standard deviation of log returns
                    result_df['garch_volatility'] = log_returns.rolling(window=21).std()
                except Exception as e:
                    logger.warning(f"Error calculating Historical Volatility for {timeframe}: {e}")
 
                    
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_indicators_numpy(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Fallback indicator calculations using only numpy/pandas"""
        
        try:
            # Simple RSI implementation
            for period in self.settings.rsi_periods_list:
                if len(df) >= period:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Simple EMA implementation
            for period in self.settings.ema_periods_list:
                if len(df) >= period:
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Simple SMA implementation
            for period in self.settings.sma_periods_list:
                if len(df) >= period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Simple Bollinger Bands
            if len(df) >= self.settings.bb_period:
                sma = df['close'].rolling(window=self.settings.bb_period).mean()
                std = df['close'].rolling(window=self.settings.bb_period).std()
                df['bb_upper'] = sma + (std * self.settings.bb_std)
                df['bb_middle'] = sma
                df['bb_lower'] = sma - (std * self.settings.bb_std)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Simple ATR
            if len(df) >= self.settings.atr_period:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr_14'] = true_range.rolling(window=self.settings.atr_period).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error in numpy indicator calculations: {e}")
            return df
            
    def _prepare_records_for_db(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Prepare DataFrame records for database insertion"""
        
        records = []
        
        for timestamp, row in df.iterrows():
            record = {
                'time': timestamp,
                'symbol': symbol
            }
            
            # Add all numeric columns
            for col in df.columns:
                if col != 'symbol':
                    try:
                        value = row[col]
                        
                        # Handle pandas scalar values properly
                        if isinstance(value, pd.Series):
                            # Extract scalar value from series
                            if len(value) > 0:
                                scalar_val = value.iloc[0]
                            else:
                                record[col] = None
                                continue
                        else:
                            scalar_val = value
                        
                        # Check for NaN values
                        try:
                            is_nan = pd.isna(scalar_val)
                        except (TypeError, ValueError):
                            is_nan = scalar_val is None
                        
                        if is_nan:
                            record[col] = None
                        elif isinstance(scalar_val, (int, float, np.integer, np.floating)):
                            record[col] = float(scalar_val)
                        elif hasattr(scalar_val, 'item'):  # numpy scalar
                            record[col] = float(scalar_val.item())
                        else:
                            # Try to convert to float
                            record[col] = float(scalar_val) if scalar_val is not None else None
                            
                    except (ValueError, TypeError, IndexError, AttributeError) as e:
                        logger.debug(f"Error converting column {col} value: {e}")
                        record[col] = None
                        
            records.append(record)
            
        return records
        
    def _get_timeframe_boundary(self, timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """Get the timeframe boundary for a given timestamp"""
        
        if timeframe == '5min':
            return timestamp.floor('5min')
        elif timeframe == '15min':
            return timestamp.floor('15min')
        elif timeframe == '1hour':
            return timestamp.floor('h')
        elif timeframe == '4hour':
            return timestamp.floor('4h')
        elif timeframe == '1day':
            return timestamp.floor('d')
        elif timeframe == '7day':
            # For weekly data, floor to the beginning of the week (Monday)
            # Since floor('W') doesn't work reliably, calculate manually
            days_since_monday = timestamp.weekday()
            week_start = timestamp - pd.Timedelta(days=days_since_monday)
            return week_start.floor('d')
        else:
            return timestamp
            
    def _get_lookback_hours(self, timeframe: str) -> int:
        """Get lookback hours needed for indicator calculation"""
        
        lookback_map = {
            '5min': 24,      # 1 days
            '15min': 72,    # 3 days
            '1hour': 168,    # 7 days
            '4hour': 720,   # 30 days
            '1day': 1440,    # 60 days
            '7day': 1680    # 180 days
        }
        
        return lookback_map.get(timeframe, 48)
