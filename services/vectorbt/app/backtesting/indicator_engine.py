# ==============================================================================
# File: services/vectorbt/app/backtesting/indicator_engine.py
# Description: Centralized technical indicator calculation engine
# ==============================================================================

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class IndicatorEngine:
    """
    Centralized engine for calculating technical indicators.
    
    Features:
    - Caching for performance
    - Parameter validation
    - Support for all major indicators
    - Custom indicator support
    - Batch calculation
    """
    
    def __init__(self, cache_size: int = 128):
        self._cache_size = cache_size
        self._indicator_cache: Dict[str, pd.Series] = {}
        
    def _generate_cache_key(self, indicator_name: str, data_hash: str, **params) -> str:
        """Generate a unique cache key for indicator calculation"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{indicator_name}_{data_hash}_{param_str}"
    
    def _get_data_hash(self, data: pd.Series) -> str:
        """Generate hash for data series"""
        return hashlib.md5(str(data.values).encode()).hexdigest()[:16]
    
    def _validate_data(self, data: Union[pd.Series, pd.DataFrame], required_cols: List[str] = None) -> bool:
        """Validate input data"""
        if isinstance(data, pd.DataFrame):
            if required_cols:
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
        elif isinstance(data, pd.Series):
            if data.empty:
                raise ValueError("Data series is empty")
        
        return True
    
    # Moving Averages
    def sma(self, data: pd.Series, window: int = 20, **kwargs) -> pd.Series:
        """Simple Moving Average"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("sma", self._get_data_hash(data), window=window)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        result = data.rolling(window=window).mean()
        self._indicator_cache[cache_key] = result
        return result
    
    def ema(self, data: pd.Series, window: int = 20, **kwargs) -> pd.Series:
        """Exponential Moving Average"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("ema", self._get_data_hash(data), window=window)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        result = data.ewm(span=window).mean()
        self._indicator_cache[cache_key] = result
        return result
    
    def wma(self, data: pd.Series, window: int = 20, **kwargs) -> pd.Series:
        """Weighted Moving Average"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("wma", self._get_data_hash(data), window=window)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        weights = np.arange(1, window + 1)
        result = data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        self._indicator_cache[cache_key] = result
        return result
    
    # Oscillators
    def rsi(self, data: pd.Series, window: int = 14, **kwargs) -> pd.Series:
        """Relative Strength Index"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("rsi", self._get_data_hash(data), window=window)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        try:
            # Use vectorbt RSI if available
            rsi_result = vbt.RSI.run(data, window=window)
            result = rsi_result.rsi if hasattr(rsi_result, 'rsi') else rsi_result
        except:
            # Fallback to manual calculation
            delta = data.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gains = gains.rolling(window=window).mean()
            avg_losses = losses.rolling(window=window).mean()
            
            rs = avg_gains / avg_losses
            result = 100 - (100 / (1 + rs))
        
        self._indicator_cache[cache_key] = result
        return result
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3, **kwargs) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        self._validate_data(high)
        self._validate_data(low)
        self._validate_data(close)
        
        data_hash = self._get_data_hash(pd.concat([high, low, close]))
        cache_key = self._generate_cache_key("stoch", data_hash, k_window=k_window, d_window=d_window)
        
        if cache_key in self._indicator_cache:
            cached = self._indicator_cache[cache_key]
            return cached['%K'], cached['%D']
        
        try:
            stoch_result = vbt.STOCH.run(high, low, close, k_window=k_window, d_window=d_window)
            k_percent = stoch_result.percent_k if hasattr(stoch_result, 'percent_k') else stoch_result[0]
            d_percent = stoch_result.percent_d if hasattr(stoch_result, 'percent_d') else stoch_result[1]
        except:
            # Fallback calculation
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
        
        result = {'%K': k_percent, '%D': d_percent}
        self._indicator_cache[cache_key] = result
        return k_percent, d_percent
    
    # MACD
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, 
             **kwargs) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("macd", self._get_data_hash(data), 
                                            fast=fast, slow=slow, signal=signal)
        
        if cache_key in self._indicator_cache:
            cached = self._indicator_cache[cache_key]
            return cached['macd'], cached['signal'], cached['histogram']
        
        try:
            macd_result = vbt.MACD.run(data, fast_window=fast, slow_window=slow, signal_window=signal)
            macd_line = macd_result.macd if hasattr(macd_result, 'macd') else macd_result[0]
            signal_line = macd_result.signal if hasattr(macd_result, 'signal') else macd_result[1]
            histogram = macd_result.histogram if hasattr(macd_result, 'histogram') else macd_result[2]
        except:
            # Fallback calculation
            ema_fast = self.ema(data, fast)
            ema_slow = self.ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.ema(macd_line, signal)
            histogram = macd_line - signal_line
        
        result = {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
        self._indicator_cache[cache_key] = result
        return macd_line, signal_line, histogram
    
    # Bollinger Bands
    def bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2.0, 
                       **kwargs) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        self._validate_data(data)
        cache_key = self._generate_cache_key("bb", self._get_data_hash(data), 
                                            window=window, std_dev=std_dev)
        
        if cache_key in self._indicator_cache:
            cached = self._indicator_cache[cache_key]
            return cached['upper'], cached['middle'], cached['lower']
        
        try:
            bb_result = vbt.BBANDS.run(data, window=window, alpha=std_dev)
            upper = bb_result.upper if hasattr(bb_result, 'upper') else bb_result[0]
            middle = bb_result.middle if hasattr(bb_result, 'middle') else bb_result[1]
            lower = bb_result.lower if hasattr(bb_result, 'lower') else bb_result[2]
        except:
            # Fallback calculation
            middle = self.sma(data, window)
            rolling_std = data.rolling(window=window).std()
            upper = middle + (rolling_std * std_dev)
            lower = middle - (rolling_std * std_dev)
        
        result = {'upper': upper, 'middle': middle, 'lower': lower}
        self._indicator_cache[cache_key] = result
        return upper, middle, lower
    
    # Volume Indicators
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, 
             window: Optional[int] = None, **kwargs) -> pd.Series:
        """Volume Weighted Average Price"""
        self._validate_data(close)
        self._validate_data(volume)
        
        typical_price = (high + low + close) / 3
        
        if window is None:
            # Session VWAP (cumulative)
            cumulative_volume = volume.cumsum()
            cumulative_pv = (typical_price * volume).cumsum()
            vwap = cumulative_pv / cumulative_volume
        else:
            # Rolling VWAP
            rolling_volume = volume.rolling(window=window).sum()
            rolling_pv = (typical_price * volume).rolling(window=window).sum()
            vwap = rolling_pv / rolling_volume
        
        return vwap
    
    def obv(self, close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
        """On-Balance Volume"""
        self._validate_data(close)
        self._validate_data(volume)
        
        price_change = close.diff()
        volume_direction = volume.where(price_change > 0, -volume)
        volume_direction = volume_direction.where(price_change != 0, 0)
        
        return volume_direction.cumsum()
    
    # Trend Indicators
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, 
            **kwargs) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        self._validate_data(high)
        self._validate_data(low)
        self._validate_data(close)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
        minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
        
        # Smooth the values
        atr = true_range.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx, plus_di, minus_di
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, 
            **kwargs) -> pd.Series:
        """Average True Range"""
        self._validate_data(high)
        self._validate_data(low)
        self._validate_data(close)
        
        cache_key = self._generate_cache_key("atr", self._get_data_hash(close), window=window)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        try:
            atr_result = vbt.ATR.run(high, low, close, window=window)
            result = atr_result.atr if hasattr(atr_result, 'atr') else atr_result
        except:
            # Fallback calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result = true_range.rolling(window=window).mean()
        
        self._indicator_cache[cache_key] = result
        return result
    
    # Custom Indicators
    def custom_indicator(self, func, data: Union[pd.Series, pd.DataFrame], 
                        name: str, **params) -> Union[pd.Series, Tuple[pd.Series, ...]]:
        """Calculate custom indicator using provided function"""
        cache_key = self._generate_cache_key(f"custom_{name}", 
                                            self._get_data_hash(data if isinstance(data, pd.Series) else data.iloc[:, 0]), 
                                            **params)
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        result = func(data, **params)
        self._indicator_cache[cache_key] = result
        return result
    
    # Batch Calculation
    def calculate_multiple(self, data: pd.DataFrame, 
                          indicators: List[Dict[str, Any]]) -> Dict[str, Union[pd.Series, Tuple[pd.Series, ...]]]:
        """
        Calculate multiple indicators in batch.
        
        Args:
            data: OHLCV DataFrame
            indicators: List of indicator specifications
                       Each dict should have 'name' and optionally 'params'
        
        Returns:
            Dictionary of indicator name -> result
        """
        results = {}
        
        for indicator_spec in indicators:
            name = indicator_spec['name']
            params = indicator_spec.get('params', {})
            
            try:
                if name == 'sma':
                    results[name] = self.sma(data['close'], **params)
                elif name == 'ema':
                    results[name] = self.ema(data['close'], **params)
                elif name == 'rsi':
                    results[name] = self.rsi(data['close'], **params)
                elif name == 'macd':
                    macd, signal, histogram = self.macd(data['close'], **params)
                    results['macd'] = macd
                    results['macd_signal'] = signal
                    results['macd_histogram'] = histogram
                elif name == 'bollinger_bands':
                    upper, middle, lower = self.bollinger_bands(data['close'], **params)
                    results['bb_upper'] = upper
                    results['bb_middle'] = middle
                    results['bb_lower'] = lower
                elif name == 'stochastic':
                    k, d = self.stochastic(data['high'], data['low'], data['close'], **params)
                    results['stoch_k'] = k
                    results['stoch_d'] = d
                elif name == 'atr':
                    results[name] = self.atr(data['high'], data['low'], data['close'], **params)
                elif name == 'adx':
                    adx, plus_di, minus_di = self.adx(data['high'], data['low'], data['close'], **params)
                    results['adx'] = adx
                    results['plus_di'] = plus_di
                    results['minus_di'] = minus_di
                elif name == 'vwap':
                    results[name] = self.vwap(data['high'], data['low'], data['close'], data['volume'], **params)
                elif name == 'obv':
                    results[name] = self.obv(data['close'], data['volume'], **params)
                else:
                    logger.warning(f"Unknown indicator: {name}")
                    
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                
        return results
    
    def clear_cache(self):
        """Clear the indicator cache"""
        self._indicator_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._indicator_cache),
            "max_size": self._cache_size,
            "keys": list(self._indicator_cache.keys())
        }

# Global indicator engine instance
indicator_engine = IndicatorEngine()
