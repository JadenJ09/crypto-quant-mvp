# ==============================================================================
# File: services/vectorbt/app/backtesting/strategy_builder.py
# Description: Strategy builder for custom strategies from frontend
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

from .strategy_base import (
    BaseStrategy, StrategyConfig, StrategySignals, SignalCondition, 
    SignalType, OperatorType, PositionSizing, PositionDirection, PositionSizingConfig, RiskManagementConfig, ExecutionConfig
)
from .indicator_engine import indicator_engine
from ..models import BacktestStats, TradeInfo

logger = logging.getLogger(__name__)

class CustomStrategyBuilder(BaseStrategy):
    """
    Builder for custom strategies defined by frontend conditions.
    
    This class processes entry and exit conditions from the frontend
    and creates a working strategy.
    """
    
    def __init__(self, entry_conditions: List[Any], 
                 exit_conditions: List[Any], 
                 config: StrategyConfig):
        super().__init__(config)
        self.entry_conditions = self._parse_conditions(entry_conditions, SignalType.ENTRY)
        self.exit_conditions = self._parse_conditions(exit_conditions, SignalType.EXIT)
        
    def _parse_conditions(self, conditions: List[Any], 
                         signal_type: SignalType) -> List[SignalCondition]:
        """Parse frontend conditions into SignalCondition objects"""
        parsed_conditions = []
        
        for i, condition in enumerate(conditions):
            try:
                # Map frontend operator names to OperatorType
                operator_map = {
                    'greater_than': OperatorType.GREATER_THAN,
                    'less_than': OperatorType.LESS_THAN,
                    'equals': OperatorType.EQUALS,
                    'crosses_above': OperatorType.CROSSES_ABOVE,
                    'crosses_below': OperatorType.CROSSES_BELOW,
                }
                
                # Handle both dict and Pydantic model instances
                if hasattr(condition, 'operator'):
                    # Pydantic model instance
                    operator = operator_map.get(condition.operator, OperatorType.GREATER_THAN)
                    signal_condition = SignalCondition(
                        id=getattr(condition, 'id', f"{signal_type.value}_{i}"),
                        signal_type=signal_type,
                        indicator=getattr(condition, 'indicator', 'close'),
                        operator=operator,
                        value=getattr(condition, 'value', 0),
                        enabled=getattr(condition, 'enabled', True),
                        weight=1.0  # Default weight
                    )
                else:
                    # Dictionary
                    operator = operator_map.get(condition.get('operator', 'greater_than'), 
                                              OperatorType.GREATER_THAN)
                    signal_condition = SignalCondition(
                        id=condition.get('id', f"{signal_type.value}_{i}"),
                        signal_type=signal_type,
                        indicator=condition.get('indicator', 'close'),
                        operator=operator,
                        value=condition.get('value', 0),
                        enabled=condition.get('enabled', True),
                        weight=condition.get('weight', 1.0)
                    )
                
                parsed_conditions.append(signal_condition)
                
            except Exception as e:
                logger.error(f"Error parsing condition {i}: {e}")
                continue
                
        return parsed_conditions
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all required indicators based on conditions"""
        required_indicators = set()
        
        # Collect all required indicators from conditions
        for condition in self.entry_conditions + self.exit_conditions:
            if condition.enabled:
                required_indicators.add(condition.indicator)
        
        indicators = {}
        
        for indicator_name in required_indicators:
            try:
                indicators.update(self._calculate_single_indicator(data, indicator_name))
            except Exception as e:
                logger.error(f"Error calculating indicator {indicator_name}: {e}")
                
        return indicators
    
    def _calculate_single_indicator(self, data: pd.DataFrame, 
                                   indicator_name: str) -> Dict[str, pd.Series]:
        """Calculate a single indicator and return all its components"""
        indicators = {}
        
        # Extract parameters from indicator name (e.g., "sma_20" -> sma with period 20)
        parts = indicator_name.split('_')
        base_indicator = parts[0].lower()
        
        # Default parameters
        period = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
        
        if base_indicator in ['sma', 'ma']:
            indicators[indicator_name] = indicator_engine.sma(data['close'], window=period)
            
        elif base_indicator == 'ema':
            indicators[indicator_name] = indicator_engine.ema(data['close'], window=period)
            
        elif base_indicator == 'rsi':
            indicators[indicator_name] = indicator_engine.rsi(data['close'], window=period)
            
        elif base_indicator == 'macd':
            # Extract MACD parameters
            fast = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 12
            slow = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 26
            signal = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 9
            
            try:
                macd, signal_line, histogram = indicator_engine.macd(data['close'], fast=fast, slow=slow, signal=signal)
                indicators['macd'] = macd
                indicators['macd_signal'] = signal_line
                indicators['macd_histogram'] = histogram
            except:
                # Fallback to simple calculation
                ema_fast = indicator_engine.ema(data['close'], fast)
                ema_slow = indicator_engine.ema(data['close'], slow)
                macd = ema_fast - ema_slow
                signal_line = indicator_engine.ema(macd, signal)
                histogram = macd - signal_line
                indicators['macd'] = macd
                indicators['macd_signal'] = signal_line
                indicators['macd_histogram'] = histogram
            
        elif base_indicator in ['bb', 'bollinger']:
            # Bollinger Bands
            std_dev = float(parts[2]) if len(parts) > 2 else 2.0
            try:
                upper, middle, lower = indicator_engine.bollinger_bands(data['close'], window=period, std_dev=std_dev)
                indicators['bb_upper'] = upper
                indicators['bb_middle'] = middle  
                indicators['bb_lower'] = lower
            except:
                # Fallback calculation
                middle = indicator_engine.sma(data['close'], period)
                rolling_std = data['close'].rolling(window=period).std()
                indicators['bb_upper'] = middle + (rolling_std * std_dev)
                indicators['bb_middle'] = middle
                indicators['bb_lower'] = middle - (rolling_std * std_dev)
            
        elif base_indicator == 'stoch':
            # Stochastic
            d_period = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 3
            try:
                k, d = indicator_engine.stochastic(data['high'], data['low'], data['close'], 
                                                 k_window=period, d_window=d_period)
                indicators['stoch_k'] = k
                indicators['stoch_d'] = d
            except:
                # Fallback calculation
                lowest_low = data['low'].rolling(window=period).min()
                highest_high = data['high'].rolling(window=period).max()
                k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
                d = k.rolling(window=d_period).mean()
                indicators['stoch_k'] = k
                indicators['stoch_d'] = d
                
        elif base_indicator == 'atr':
            try:
                indicators[indicator_name] = indicator_engine.atr(data['high'], data['low'], data['close'], window=period)
            except:
                # Fallback calculation
                tr1 = data['high'] - data['low']
                tr2 = abs(data['high'] - data['close'].shift(1))
                tr3 = abs(data['low'] - data['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                indicators[indicator_name] = true_range.rolling(window=period).mean()
                
        elif base_indicator == 'close':
            indicators['close'] = data['close']
            
        elif base_indicator == 'open':
            indicators['open'] = data['open']
            
        elif base_indicator == 'high':
            indicators['high'] = data['high']
            
        elif base_indicator == 'low':
            indicators['low'] = data['low']
            
        elif base_indicator == 'volume':
            indicators['volume'] = data['volume']
            
        else:
            logger.warning(f"Unknown indicator: {indicator_name}")
            # Fallback to close price
            indicators[indicator_name] = data['close']
            
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """Generate trading signals based on conditions"""
        # Initialize signal series
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        
        # Generate entry signals
        entry_signals = []
        for condition in self.entry_conditions:
            if condition.enabled:
                signal = self._evaluate_condition(condition, data, indicators)
                entry_signals.append(signal * condition.weight)
        
        # Combine entry signals (AND logic for now - can be made configurable)
        if entry_signals:
            # For now, use simple AND logic (all conditions must be true)
            entries = pd.concat(entry_signals, axis=1).all(axis=1)
        
        # Generate exit signals
        exit_signals = []
        for condition in self.exit_conditions:
            if condition.enabled:
                signal = self._evaluate_condition(condition, data, indicators)
                exit_signals.append(signal * condition.weight)
        
        # Combine exit signals (OR logic for exits)
        if exit_signals:
            # For exits, use OR logic (any condition can trigger exit)
            exits = pd.concat(exit_signals, axis=1).any(axis=1)
        
        return StrategySignals(
            entries=entries,
            exits=exits,
            metadata={
                'entry_conditions': len([c for c in self.entry_conditions if c.enabled]),
                'exit_conditions': len([c for c in self.exit_conditions if c.enabled])
            }
        )
    
    def _evaluate_condition(self, condition: SignalCondition, data: pd.DataFrame, 
                           indicators: Dict[str, pd.Series]) -> pd.Series:
        """Evaluate a single condition"""
        # Get the indicator data
        if condition.indicator in indicators:
            indicator_data = indicators[condition.indicator]
        elif condition.indicator in data.columns:
            indicator_data = data[condition.indicator]
        else:
            logger.warning(f"Indicator {condition.indicator} not found, using close price")
            indicator_data = data['close']
        
        # Get comparison value
        if isinstance(condition.value, str):
            if condition.value in indicators:
                compare_value = indicators[condition.value]
            elif condition.value in data.columns:
                compare_value = data[condition.value]
            else:
                # Try to parse as float
                try:
                    compare_value = float(condition.value)
                except:
                    logger.warning(f"Could not parse value {condition.value}, using 0")
                    compare_value = 0
        else:
            compare_value = condition.value
        
        # Apply operator
        if condition.operator == OperatorType.GREATER_THAN:
            return indicator_data > compare_value
        elif condition.operator == OperatorType.LESS_THAN:
            return indicator_data < compare_value
        elif condition.operator == OperatorType.EQUALS:
            return indicator_data == compare_value
        elif condition.operator == OperatorType.CROSSES_ABOVE:
            if isinstance(compare_value, pd.Series):
                return (indicator_data > compare_value) & (indicator_data.shift(1) <= compare_value.shift(1))
            else:
                return (indicator_data > compare_value) & (indicator_data.shift(1) <= compare_value)
        elif condition.operator == OperatorType.CROSSES_BELOW:
            if isinstance(compare_value, pd.Series):
                return (indicator_data < compare_value) & (indicator_data.shift(1) >= compare_value.shift(1))
            else:
                return (indicator_data < compare_value) & (indicator_data.shift(1) >= compare_value)
        else:
            logger.warning(f"Unknown operator {condition.operator}, using greater_than")
            return indicator_data > compare_value


def create_custom_strategy_from_frontend(strategy_data: Dict[str, Any]) -> CustomStrategyBuilder:
    """
    Create a custom strategy from frontend strategy data.
    
    Args:
        strategy_data: Dictionary containing strategy configuration from frontend
        
    Returns:
        CustomStrategyBuilder instance
    """
    # Extract basic configuration
    initial_cash = strategy_data.get('initial_cash', 100000.0)
    commission = strategy_data.get('commission', 0.001)
    slippage = strategy_data.get('slippage', 0.0005)
    
    # Position sizing
    position_sizing_method = strategy_data.get('position_sizing', 'percentage')
    position_size = strategy_data.get('position_size', 0.1)
    
    # Convert percentage position size to decimal if needed
    # Frontend sends percentage (e.g., 5.0 for 5%), but backend expects decimal (0.05)
    if position_sizing_method == 'percentage' and position_size > 1:
        position_size = position_size / 100.0
        logger.info(f"Converted position size from percentage to decimal: {position_size}")
    
    # Risk management
    stop_loss = strategy_data.get('stop_loss')
    take_profit = strategy_data.get('take_profit')
    max_positions = strategy_data.get('max_positions', 1)
    max_position_strategy = strategy_data.get('max_position_strategy', 'ignore')
    
    # Position direction
    position_direction = strategy_data.get('position_direction', 'long_only')
    
    # Create configuration objects - Map frontend values to backend enums
    position_sizing_map = {
        'fixed': PositionSizing.FIXED_AMOUNT,
        'percentage': PositionSizing.PERCENTAGE,
        'percent_equity': PositionSizing.PERCENT_EQUITY,
        'volatility': PositionSizing.VOLATILITY_ADJUSTED,
        'kelly': PositionSizing.KELLY_CRITERION
    }
    
    # Map position direction
    direction_map = {
        'long_only': PositionDirection.LONG_ONLY,
        'short_only': PositionDirection.SHORT_ONLY,
        'both': PositionDirection.BOTH
    }
    
    position_sizing_enum = position_sizing_map.get(position_sizing_method, PositionSizing.PERCENTAGE)
    position_direction_enum = direction_map.get(position_direction, PositionDirection.LONG_ONLY)
    
    # DEBUG: Log the position direction mapping
    logger.info(f"Frontend position_direction: '{position_direction}' -> Backend enum: {position_direction_enum}")
    
    position_sizing_config = PositionSizingConfig(
        method=position_sizing_enum,
        size=position_size
    )
    
    risk_management_config = RiskManagementConfig(
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_positions=max_positions,
        max_position_strategy=max_position_strategy
    )
    
    execution_config = ExecutionConfig(
        commission=commission,
        slippage=slippage
    )
    
    strategy_config = StrategyConfig(
        name=strategy_data.get('name', 'Custom Strategy'),
        description=strategy_data.get('description', 'Custom strategy from frontend'),
        initial_cash=initial_cash,
        position_direction=position_direction_enum,
        position_sizing=position_sizing_config,
        risk_management=risk_management_config,
        execution=execution_config
    )
    
    # Extract conditions
    entry_conditions = strategy_data.get('entry_conditions', [])
    exit_conditions = strategy_data.get('exit_conditions', [])
    
    return CustomStrategyBuilder(entry_conditions, exit_conditions, strategy_config)
