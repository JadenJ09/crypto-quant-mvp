# ==============================================================================
# File: services/vectorbt/app/backtesting/concrete_strategies.py
# Description: Concrete strategy implementations using the new architecture
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .strategy_base import BaseStrategy, StrategyConfig, StrategySignals
from .indicator_engine import indicator_engine

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, fast_window: int = 10, slow_window: int = 20, 
                 config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="MA Crossover",
                description=f"Fast MA({fast_window}) vs Slow MA({slow_window}) crossover strategy"
            )
        
        super().__init__(config)
        self.fast_window = fast_window
        self.slow_window = slow_window
        
        # Store parameters in config
        self.config.parameters.update({
            'fast_window': fast_window,
            'slow_window': slow_window
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate moving averages"""
        indicators = {}
        
        # Calculate moving averages
        indicators['fast_ma'] = indicator_engine.sma(data['close'], window=self.fast_window)
        indicators['slow_ma'] = indicator_engine.sma(data['close'], window=self.slow_window)
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate crossover signals"""
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']
        
        # Entry: fast MA crosses above slow MA
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Exit: fast MA crosses below slow MA
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'fast_window': self.fast_window,
                'slow_window': self.slow_window
            }
        )


class RSIStrategy(BaseStrategy):
    """RSI Oversold/Overbought Strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold_level: float = 30, 
                 overbought_level: float = 70, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="RSI Strategy",
                description=f"RSI({rsi_period}) oversold({oversold_level})/overbought({overbought_level}) strategy"
            )
        
        super().__init__(config)
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        
        # Store parameters in config
        self.config.parameters.update({
            'rsi_period': rsi_period,
            'oversold_level': oversold_level,
            'overbought_level': overbought_level
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI"""
        indicators = {}
        indicators['rsi'] = indicator_engine.rsi(data['close'], window=self.rsi_period)
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate RSI signals"""
        rsi = indicators['rsi']
        
        # Entry: RSI below oversold level
        entries = rsi < self.oversold_level
        
        # Exit: RSI above overbought level
        exits = rsi > self.overbought_level
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'rsi_period': self.rsi_period,
                'oversold_level': self.oversold_level,
                'overbought_level': self.overbought_level
            }
        )


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, 
                 config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Bollinger Bands",
                description=f"Bollinger Bands({bb_period}, {bb_std}) mean reversion strategy"
            )
        
        super().__init__(config)
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Store parameters in config
        self.config.parameters.update({
            'bb_period': bb_period,
            'bb_std': bb_std
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        indicators = {}
        
        try:
            upper, middle, lower = indicator_engine.bollinger_bands(
                data['close'], window=self.bb_period, std_dev=self.bb_std
            )
            indicators['bb_upper'] = upper
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = lower
        except:
            # Fallback calculation
            middle = indicator_engine.sma(data['close'], self.bb_period)
            rolling_std = data['close'].rolling(window=self.bb_period).std()
            indicators['bb_upper'] = middle + (rolling_std * self.bb_std)
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = middle - (rolling_std * self.bb_std)
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate Bollinger Bands signals"""
        close = data['close']
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        bb_lower = indicators['bb_lower']
        
        # Entry: price touches lower band
        entries = close <= bb_lower
        
        # Exit: price reaches middle band
        exits = close >= bb_middle
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'bb_period': self.bb_period,
                'bb_std': self.bb_std
            }
        )


class MACDStrategy(BaseStrategy):
    """MACD Strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="MACD Strategy",
                description=f"MACD({fast},{slow},{signal}) crossover strategy"
            )
        
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        
        # Store parameters in config
        self.config.parameters.update({
            'fast': fast,
            'slow': slow,
            'signal': signal
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        indicators = {}
        
        try:
            macd, signal_line, histogram = indicator_engine.macd(
                data['close'], fast=self.fast, slow=self.slow, signal=self.signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = histogram
        except:
            # Fallback calculation
            ema_fast = indicator_engine.ema(data['close'], self.fast)
            ema_slow = indicator_engine.ema(data['close'], self.slow)
            macd = ema_fast - ema_slow
            signal_line = indicator_engine.ema(macd, self.signal)
            histogram = macd - signal_line
            indicators['macd'] = macd
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = histogram
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate MACD signals"""
        macd = indicators['macd']
        signal_line = indicators['macd_signal']
        
        # Entry: MACD crosses above signal line
        entries = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        
        # Exit: MACD crosses below signal line
        exits = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'fast': self.fast,
                'slow': self.slow,
                'signal': self.signal
            }
        )


class MultiIndicatorStrategy(BaseStrategy):
    """Multi-Indicator Combined Strategy"""
    
    def __init__(self, indicators_config: List[Dict[str, Any]], 
                 config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Multi-Indicator Strategy",
                description="Combined multi-indicator strategy"
            )
        
        super().__init__(config)
        self.indicators_config = indicators_config
        
        # Store parameters in config
        self.config.parameters.update({
            'indicators_config': indicators_config
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all required indicators"""
        indicators = {}
        
        # Use the indicator engine's batch calculation
        indicators.update(indicator_engine.calculate_multiple(data, self.indicators_config))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate combined signals"""
        # This is a placeholder implementation
        # In practice, this would implement complex logic to combine multiple indicators
        
        # For now, use a simple approach: if any indicator gives a signal, use it
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        
        # Example: Combine RSI and MACD signals
        if 'rsi' in indicators and 'macd' in indicators and 'macd_signal' in indicators:
            # Entry: RSI oversold AND MACD bullish
            rsi_signal = indicators['rsi'] < 30
            macd_signal = indicators['macd'] > indicators['macd_signal']
            entries = rsi_signal & macd_signal
            
            # Exit: RSI overbought OR MACD bearish
            rsi_exit = indicators['rsi'] > 70
            macd_exit = indicators['macd'] < indicators['macd_signal']
            exits = rsi_exit | macd_exit
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'indicators_used': list(indicators.keys())
            }
        )


# Strategy factory for easy creation
class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_type: str, **params) -> BaseStrategy:
        """Create a strategy instance based on type and parameters"""
        
        if strategy_type == "ma_crossover":
            return MovingAverageCrossoverStrategy(
                fast_window=params.get('fast_window', 10),
                slow_window=params.get('slow_window', 20)
            )
        
        elif strategy_type == "rsi_oversold":
            return RSIStrategy(
                rsi_period=params.get('rsi_period', 14),
                oversold_level=params.get('oversold_level', 30),
                overbought_level=params.get('overbought_level', 70)
            )
        
        elif strategy_type == "bollinger_bands":
            return BollingerBandsStrategy(
                bb_period=params.get('bb_period', 20),
                bb_std=params.get('bb_std', 2.0)
            )
        
        elif strategy_type == "macd":
            return MACDStrategy(
                fast=params.get('fast', 12),
                slow=params.get('slow', 26),
                signal=params.get('signal', 9)
            )
        
        elif strategy_type == "multi_indicator":
            return MultiIndicatorStrategy(
                indicators_config=params.get('indicators_config', [])
            )
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> List[Dict[str, Any]]:
        """Get list of available strategies with their metadata"""
        return [
            {
                "type": "ma_crossover",
                "name": "Moving Average Crossover",
                "description": "Buy when fast MA crosses above slow MA",
                "parameters": [
                    {"name": "fast_window", "type": "int", "default": 10, "min": 1, "max": 100},
                    {"name": "slow_window", "type": "int", "default": 20, "min": 1, "max": 200}
                ]
            },
            {
                "type": "rsi_oversold",
                "name": "RSI Oversold/Overbought",
                "description": "Buy when RSI oversold, sell when overbought",
                "parameters": [
                    {"name": "rsi_period", "type": "int", "default": 14, "min": 1, "max": 50},
                    {"name": "oversold_level", "type": "float", "default": 30, "min": 10, "max": 40},
                    {"name": "overbought_level", "type": "float", "default": 70, "min": 60, "max": 90}
                ]
            },
            {
                "type": "bollinger_bands",
                "name": "Bollinger Bands Mean Reversion",
                "description": "Buy at lower band, sell at middle band",
                "parameters": [
                    {"name": "bb_period", "type": "int", "default": 20, "min": 5, "max": 50},
                    {"name": "bb_std", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0}
                ]
            },
            {
                "type": "macd",
                "name": "MACD Crossover",
                "description": "Buy when MACD crosses above signal line",
                "parameters": [
                    {"name": "fast", "type": "int", "default": 12, "min": 1, "max": 50},
                    {"name": "slow", "type": "int", "default": 26, "min": 1, "max": 100},
                    {"name": "signal", "type": "int", "default": 9, "min": 1, "max": 50}
                ]
            }
        ]
