"""
Signal Engine - Technical indicator calculation and signal generation

This module handles signal generation including:
- Technical indicator calculation using the ta library
- Signal logic and filtering
- Strategy signal coordination
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
try:
    import ta
    from ta.trend import sma_indicator, ema_indicator, MACD
    from ta.momentum import RSIIndicator  
    from ta.volatility import BollingerBands
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not available, using custom implementations")

from numba import njit


class SignalEngine:
    """Technical signal generation engine"""
    
    def __init__(self):
        self.indicators = {}
        self.signals = {}
        
    def add_sma(self, data: pd.DataFrame, column: str = 'close', 
                window: int = 20, name: Optional[str] = None) -> pd.Series:
        """
        Add Simple Moving Average indicator
        
        Args:
            data: OHLCV data
            column: Column to calculate SMA on
            window: Moving average window
            name: Custom name for indicator
            
        Returns:
            SMA series
        """
        if name is None:
            name = f'sma_{window}'
        
        if TA_AVAILABLE:
            sma = sma_indicator(data[column], window=window)
        else:
            sma = pd.Series(calculate_sma(np.array(data[column].values, dtype=float), window), index=data.index)
        
        self.indicators[name] = sma
        return sma
    
    def add_ema(self, data: pd.DataFrame, column: str = 'close',
                window: int = 20, name: Optional[str] = None) -> pd.Series:
        """
        Add Exponential Moving Average indicator
        
        Args:
            data: OHLCV data
            column: Column to calculate EMA on  
            window: Moving average window
            name: Custom name for indicator
            
        Returns:
            EMA series
        """
        if name is None:
            name = f'ema_{window}'
        
        if TA_AVAILABLE:
            ema = ema_indicator(data[column], window=window)
        else:
            ema = pd.Series(calculate_ema(np.array(data[column].values, dtype=float), window), index=data.index)
        
        self.indicators[name] = ema
        return ema
    
    def add_rsi(self, data: pd.DataFrame, column: str = 'close',
                window: int = 14, name: Optional[str] = None) -> pd.Series:
        """
        Add Relative Strength Index indicator
        
        Args:
            data: OHLCV data
            column: Column to calculate RSI on
            window: RSI window
            name: Custom name for indicator
            
        Returns:
            RSI series
        """
        if name is None:
            name = f'rsi_{window}'
        
        if TA_AVAILABLE:
            rsi_indicator = RSIIndicator(close=data[column], window=window)
            rsi = rsi_indicator.rsi()
        else:
            rsi = pd.Series(calculate_rsi(np.array(data[column].values, dtype=float), window), index=data.index)
        
        self.indicators[name] = rsi
        return rsi
    
    def add_macd(self, data: pd.DataFrame, column: str = 'close',
                 window_slow: int = 26, window_fast: int = 12, window_sign: int = 9,
                 name: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Add MACD indicator
        
        Args:
            data: OHLCV data
            column: Column to calculate MACD on
            window_slow: Slow EMA window
            window_fast: Fast EMA window  
            window_sign: Signal line EMA window
            name: Custom name prefix for indicator
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if name is None:
            name = 'macd'
        
        if TA_AVAILABLE:
            macd_indicator = MACD(close=data[column], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
            macd_line = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_diff = macd_indicator.macd_diff()
        else:
            # Simple MACD implementation
            ema_fast = pd.Series(calculate_ema(np.array(data[column].values, dtype=float), window_fast), index=data.index)
            ema_slow = pd.Series(calculate_ema(np.array(data[column].values, dtype=float), window_slow), index=data.index)
            macd_line = ema_fast - ema_slow
            macd_signal = pd.Series(calculate_ema(np.array(macd_line.values, dtype=float), window_sign), index=data.index)
            macd_diff = macd_line - macd_signal
        
        result = {
            f'{name}_line': macd_line,
            f'{name}_signal': macd_signal, 
            f'{name}_histogram': macd_diff
        }
        
        self.indicators.update(result)
        return result
    
    def add_bollinger_bands(self, data: pd.DataFrame, column: str = 'close',
                           window: int = 20, window_dev: int = 2,
                           name: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Add Bollinger Bands indicator
        
        Args:
            data: OHLCV data
            column: Column to calculate bands on
            window: Moving average window
            window_dev: Standard deviation multiplier
            name: Custom name prefix for indicator
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        if name is None:
            name = 'bb'
        
        if TA_AVAILABLE:
            bb_indicator = BollingerBands(close=data[column], window=window, window_dev=window_dev)
            bb_upper = bb_indicator.bollinger_hband()
            bb_middle = bb_indicator.bollinger_mavg()
            bb_lower = bb_indicator.bollinger_lband()
        else:
            # Simple Bollinger Bands implementation
            sma = pd.Series(calculate_sma(np.array(data[column].values, dtype=float), window), index=data.index)
            std = data[column].rolling(window=window).std()
            bb_upper = sma + (std * window_dev)
            bb_middle = sma
            bb_lower = sma - (std * window_dev)
        
        result = {
            f'{name}_upper': bb_upper,
            f'{name}_middle': bb_middle,
            f'{name}_lower': bb_lower
        }
        
        self.indicators.update(result)
        return result
    
    def generate_ma_crossover_signals(self, fast_ma: str, slow_ma: str,
                                     symbol: str = 'default') -> pd.Series:
        """
        Generate moving average crossover signals
        
        Args:
            fast_ma: Name of fast moving average indicator
            slow_ma: Name of slow moving average indicator
            symbol: Trading symbol
            
        Returns:
            Signal series (1 for buy, -1 for sell, 0 for hold)
        """
        if fast_ma not in self.indicators or slow_ma not in self.indicators:
            raise ValueError(f"Indicators {fast_ma} or {slow_ma} not found")
        
        fast = self.indicators[fast_ma]
        slow = self.indicators[slow_ma]
        
        # Generate crossover signals
        signals = pd.Series(0, index=fast.index, dtype=int)
        
        # Buy when fast MA crosses above slow MA
        buy_signals = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        signals[buy_signals] = 1
        
        # Sell when fast MA crosses below slow MA  
        sell_signals = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        signals[sell_signals] = -1
        
        signal_name = f'{symbol}_ma_crossover'
        self.signals[signal_name] = signals
        
        return signals
    
    def generate_rsi_signals(self, rsi_name: str, oversold: float = 30,
                           overbought: float = 70, symbol: str = 'default') -> pd.Series:
        """
        Generate RSI overbought/oversold signals
        
        Args:
            rsi_name: Name of RSI indicator
            oversold: Oversold threshold
            overbought: Overbought threshold
            symbol: Trading symbol
            
        Returns:
            Signal series (1 for buy, -1 for sell, 0 for hold)
        """
        if rsi_name not in self.indicators:
            raise ValueError(f"RSI indicator {rsi_name} not found")
        
        rsi = self.indicators[rsi_name]
        signals = pd.Series(0, index=rsi.index, dtype=int)
        
        # Buy when RSI crosses above oversold level
        buy_signals = (rsi > oversold) & (rsi.shift(1) <= oversold)
        signals[buy_signals] = 1
        
        # Sell when RSI crosses below overbought level
        sell_signals = (rsi < overbought) & (rsi.shift(1) >= overbought)
        signals[sell_signals] = -1
        
        signal_name = f'{symbol}_rsi_signals'
        self.signals[signal_name] = signals
        
        return signals
    
    def generate_bollinger_signals(self, bb_prefix: str = 'bb', 
                                  symbol: str = 'default') -> pd.Series:
        """
        Generate Bollinger Band signals
        
        Args:
            bb_prefix: Prefix for Bollinger Band indicators
            symbol: Trading symbol
            
        Returns:
            Signal series (1 for buy, -1 for sell, 0 for hold)
        """
        upper_name = f'{bb_prefix}_upper'
        lower_name = f'{bb_prefix}_lower'
        
        if upper_name not in self.indicators or lower_name not in self.indicators:
            raise ValueError(f"Bollinger Band indicators not found")
        
        upper = self.indicators[upper_name]
        lower = self.indicators[lower_name]
        
        # Assume we have price data in indicators (this would need to be improved)
        # For now, use the middle band as price proxy
        middle_name = f'{bb_prefix}_middle'
        if middle_name in self.indicators:
            price = self.indicators[middle_name]  # This is just the SMA, need actual price
        else:
            raise ValueError("Need price data for Bollinger Band signals")
        
        signals = pd.Series(0, index=upper.index, dtype=int)
        
        # Buy when price touches lower band (oversold)
        buy_signals = price <= lower
        signals[buy_signals] = 1
        
        # Sell when price touches upper band (overbought)
        sell_signals = price >= upper
        signals[sell_signals] = -1
        
        signal_name = f'{symbol}_bollinger_signals'
        self.signals[signal_name] = signals
        
        return signals
    
    def combine_signals(self, signal_names: List[str], method: str = 'majority',
                       weights: Optional[List[float]] = None) -> pd.Series:
        """
        Combine multiple signals using specified method
        
        Args:
            signal_names: List of signal names to combine
            method: Combination method ('majority', 'unanimous', 'weighted')
            weights: Weights for weighted combination
            
        Returns:
            Combined signal series
        """
        signals_list = []
        for name in signal_names:
            if name in self.signals:
                signals_list.append(self.signals[name])
        
        if not signals_list:
            raise ValueError("No valid signals found")
        
        signals_df = pd.concat(signals_list, axis=1)
        
        if method == 'majority':
            # Take majority vote
            combined = signals_df.apply(lambda row: 1 if row.sum() > 0 else (-1 if row.sum() < 0 else 0), axis=1)
        
        elif method == 'unanimous':
            # Require all signals to agree
            combined = signals_df.apply(lambda row: row.iloc[0] if (row == row.iloc[0]).all() else 0, axis=1)
        
        elif method == 'weighted':
            if weights is None or len(weights) != len(signals_list):
                raise ValueError("Weights must be provided and match number of signals")
            
            weighted_sum = signals_df.mul(weights, axis=1).sum(axis=1)
            combined = weighted_sum.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined
    
    def get_signal_summary(self, symbol: str = 'default') -> Dict[str, Any]:
        """
        Get summary of all signals for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with signal statistics
        """
        symbol_signals = {k: v for k, v in self.signals.items() if symbol in k}
        
        summary = {}
        for name, signal in symbol_signals.items():
            summary[name] = {
                'total_signals': (signal != 0).sum(),
                'buy_signals': (signal == 1).sum(),
                'sell_signals': (signal == -1).sum(),
                'last_signal': signal.iloc[-1] if len(signal) > 0 else 0,
                'last_signal_date': signal.index[-1] if len(signal) > 0 else None
            }
        
        return summary
    
    def reset(self):
        """Reset all indicators and signals"""
        self.indicators.clear()
        self.signals.clear()


@njit
def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Fast SMA calculation with numba
    
    Args:
        prices: Price array
        window: Moving average window
        
    Returns:
        SMA array
    """
    n = len(prices)
    sma = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        sma[i] = np.mean(prices[i - window + 1:i + 1])
    
    return sma


@njit
def calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Fast EMA calculation with numba
    
    Args:
        prices: Price array
        window: EMA window
        
    Returns:
        EMA array
    """
    n = len(prices)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (window + 1)
    
    # Initialize with first non-NaN value
    ema[0] = prices[0]
    
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@njit
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Fast RSI calculation with numba
    
    Args:
        prices: Price array
        window: RSI window
        
    Returns:
        RSI array
    """
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    if n < window + 1:
        return rsi
    
    # Calculate price changes
    changes = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    if avg_loss == 0:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate RSI for remaining values
    for i in range(window + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi
