# ==============================================================================
# File: services/vectorbt/app/backtesting/strategy_base.py
# Description: Base strategy classes and configuration for scalable strategy system
# ==============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Enums for better type safety
class PositionSizing(Enum):
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage" 
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"

class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OperatorType(Enum):
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUALS = "equals"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    OUTSIDE = "outside"

class PositionDirection(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"

@dataclass
class RiskManagementConfig:
    """Risk management parameters"""
    stop_loss: Optional[float] = None  # Percentage
    take_profit: Optional[float] = None  # Percentage
    max_position_size: float = 1.0  # Maximum fraction of portfolio
    max_drawdown: Optional[float] = None  # Maximum allowed drawdown
    max_consecutive_losses: Optional[int] = None
    risk_per_trade: float = 0.02  # 2% risk per trade default
    max_positions: int = 1  # Maximum concurrent positions
    max_position_strategy: str = "ignore"  # "ignore", "replace_oldest", "replace_worst"

@dataclass
class ExecutionConfig:
    """Execution parameters"""
    commission: float = 0.001  # 0.1% default
    slippage: float = 0.0005  # 0.05% default
    market_impact: float = 0.0  # Market impact cost
    latency_ms: int = 0  # Execution latency in milliseconds

@dataclass
class PositionSizingConfig:
    """Position sizing configuration"""
    method: PositionSizing = PositionSizing.PERCENTAGE
    size: float = 0.1  # 10% of portfolio default
    max_leverage: float = 1.0
    volatility_lookback: int = 20  # For volatility-adjusted sizing
    kelly_lookback: int = 252  # For Kelly criterion

@dataclass
class StrategyConfig:
    """Base configuration for all strategies"""
    # Strategy identification
    name: str = "base_strategy"
    description: str = ""
    version: str = "1.0.0"
    
    # Financial parameters
    initial_cash: float = 100000.0
    currency: str = "USD"
    
    # Position direction
    position_direction: PositionDirection = PositionDirection.LONG_ONLY
    
    # Configuration objects
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Strategy-specific parameters (to be extended by subclasses)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class SignalCondition:
    """Represents a single signal condition"""
    id: str
    signal_type: SignalType
    indicator: str
    operator: OperatorType
    value: Union[float, str, List[float]]
    enabled: bool = True
    weight: float = 1.0  # For weighted signal combination
    confidence_threshold: float = 0.0  # Minimum confidence to trigger

@dataclass
class StrategySignals:
    """Container for strategy signals"""
    entries: pd.Series
    exits: pd.Series
    stop_losses: Optional[pd.Series] = None
    take_profits: Optional[pd.Series] = None
    confidence: Optional[pd.Series] = None  # Signal confidence scores
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides the common framework for strategy implementation,
    including indicator calculation, signal generation, and backtesting.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicators: Dict[str, pd.Series] = {}
        self.signals: Optional[StrategySignals] = None
        self.portfolio = None
        self.results = None
        self._indicator_cache: Dict[str, pd.Series] = {}
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all required technical indicators for the strategy.
        
        Args:
            data: OHLCV price data
            
        Returns:
            Dictionary of indicator name -> Series
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> StrategySignals:
        """
        Generate entry and exit signals based on indicators.
        
        Args:
            data: OHLCV price data
            indicators: Calculated indicators
            
        Returns:
            StrategySignals object containing all signals
        """
        pass
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            # Basic validation
            if self.config.initial_cash <= 0:
                raise ValueError("Initial cash must be positive")
            
            # Validate position size based on sizing method
            if self.config.position_sizing.method == PositionSizing.FIXED_AMOUNT:
                # For fixed amount, size should be positive dollar amount
                if self.config.position_sizing.size <= 0:
                    raise ValueError("Fixed amount position size must be positive")
            else:
                # For percentage-based methods, size should be between 0 and 1
                if not (0 < self.config.position_sizing.size <= 1):
                    raise ValueError("Position size must be between 0 and 1")
            
            if self.config.execution.commission < 0:
                raise ValueError("Commission cannot be negative")
                
            return True
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before strategy execution"""
        # Ensure data is sorted by time
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        data = data.dropna()
        
        return data
    
    @staticmethod
    def kelly_criterion(p: float, b: float) -> float:
        """
        Calculate Kelly criterion position size using the simplified formula.
        
        Args:
            p: Probability of winning (0-1)
            b: Ratio of average win to average loss (odds)
            
        Returns:
            Optimal fraction of capital to allocate
        """
        q = 1 - p
        return (b * p - q) / b
    
    def calculate_position_size(self, data: pd.DataFrame, signal_strength: float = 1.0) -> pd.Series:
        """
        Calculate position size based on configuration.
        
        Args:
            data: Price data
            signal_strength: Signal strength (0-1), used for scaling
            
        Returns:
            Position size series
        """
        method = self.config.position_sizing.method
        base_size = self.config.position_sizing.size
        
        if method == PositionSizing.FIXED_AMOUNT:
            # Fixed dollar amount - return the dollar amount, not shares
            return pd.Series(base_size, index=data.index)
        
        elif method == PositionSizing.PERCENTAGE:
            # Fixed percentage of portfolio
            return pd.Series(base_size * signal_strength, index=data.index)
        
        elif method == PositionSizing.PERCENT_EQUITY:
            # Percentage of current equity (dynamic)
            return pd.Series(base_size * signal_strength, index=data.index)
        
        elif method == PositionSizing.VOLATILITY_ADJUSTED:
            # Volatility-adjusted sizing
            returns = data['close'].pct_change()
            volatility = returns.rolling(self.config.position_sizing.volatility_lookback).std()
            # Inverse volatility scaling
            vol_scalar = (volatility.mean() / volatility).fillna(1.0)
            return pd.Series(base_size * vol_scalar * signal_strength, index=data.index)
        
        elif method == PositionSizing.KELLY_CRITERION:
            # Kelly criterion using dynamic calculation
            returns = data['close'].pct_change().dropna()
            lookback = self.config.position_sizing.kelly_lookback
            
            kelly_sizes = []
            for i in range(len(returns)):
                if i < lookback:
                    # Not enough data, use base size
                    kelly_sizes.append(base_size)
                else:
                    # Calculate Kelly using recent performance
                    recent_returns = returns.iloc[i-lookback:i]
                    
                    win_returns = recent_returns[recent_returns > 0]
                    loss_returns = recent_returns[recent_returns < 0]
                    
                    if len(win_returns) == 0 or len(loss_returns) == 0:
                        kelly_sizes.append(base_size)
                        continue
                    
                    # Calculate Kelly parameters
                    win_probability = len(win_returns) / len(recent_returns)
                    avg_win = win_returns.mean()
                    avg_loss = abs(loss_returns.mean())
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                    
                    # Use our Kelly function with safety cap
                    kelly_f = self.kelly_criterion(win_probability, win_loss_ratio)
                    # Apply safety constraints
                    kelly_f = max(0.0, min(kelly_f, base_size))  # Cap at base_size for safety
                    kelly_sizes.append(kelly_f)
            
            # Pad with base_size for any missing values
            while len(kelly_sizes) < len(data):
                kelly_sizes.insert(0, base_size)
                
            return pd.Series(kelly_sizes, index=data.index) * signal_strength
        
        else:
            # Default to percentage
            return pd.Series(base_size * signal_strength, index=data.index)
    
    def apply_risk_management(self, signals: StrategySignals, data: pd.DataFrame) -> StrategySignals:
        """Apply risk management rules to signals"""
        # Apply stop loss and take profit
        if self.config.risk_management.stop_loss:
            # Add stop loss signals
            signals.stop_losses = self._calculate_stop_loss_signals(data, signals.entries)
        
        if self.config.risk_management.take_profit:
            # Add take profit signals
            signals.take_profits = self._calculate_take_profit_signals(data, signals.entries)
        
        # Apply max position size constraint
        max_pos = self.config.risk_management.max_position_size
        # Note: position_sizes would be handled in the portfolio construction phase
        
        return signals
    
    def apply_max_position_constraint(self, entry_signals: pd.Series, exit_signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply maximum position constraint with different strategies.
        
        When max_positions is set to N and we already have N open positions:
        - "ignore": Ignore new entry signals (most conservative)
        - "replace_oldest": Close the oldest position and open the new one (FIFO)
        - "replace_worst": Close the worst performing position and open the new one
        
        Args:
            entry_signals: Entry signals series
            exit_signals: Exit signals series
            
        Returns:
            Tuple of (modified_entry_signals, modified_exit_signals)
        """
        max_positions = self.config.risk_management.max_positions
        strategy = self.config.risk_management.max_position_strategy
        
        if max_positions <= 0:
            # No position limit
            return entry_signals, exit_signals
        
        # Track position states
        modified_entries = entry_signals.copy()
        modified_exits = exit_signals.copy()
        
        position_count = 0
        position_entry_times = []  # Track when positions were opened
        position_returns = {}  # Track position performance for "replace_worst"
        
        logger.info(f"Applying max position constraint: {max_positions} positions, strategy: {strategy}")
        
        for timestamp in entry_signals.index:
            # Process exits first - they free up position slots
            if exit_signals.loc[timestamp]:
                if position_count > 0:
                    position_count -= 1
                    # Remove from tracking (FIFO for simplicity)
                    if position_entry_times:
                        exited_time = position_entry_times.pop(0)
                        if exited_time in position_returns:
                            del position_returns[exited_time]
                    logger.debug(f"Position exited at {timestamp}, count now: {position_count}")
            
            # Process entries - they consume position slots
            if entry_signals.loc[timestamp]:
                if position_count < max_positions:
                    # Can open new position - slot available
                    position_count += 1
                    position_entry_times.append(timestamp)
                    position_returns[timestamp] = 0.0  # Initialize return tracking
                    logger.debug(f"New position opened at {timestamp}, count now: {position_count}")
                else:
                    # At max positions - apply strategy
                    logger.debug(f"Max positions ({max_positions}) reached at {timestamp}, applying strategy: {strategy}")
                    
                    if strategy == "ignore":
                        # Ignore the new signal
                        modified_entries.loc[timestamp] = False
                        logger.debug(f"Ignoring entry signal at {timestamp}")
                        
                    elif strategy == "replace_oldest":
                        # Close oldest position and open new one
                        if position_entry_times:
                            oldest_time = position_entry_times.pop(0)
                            modified_exits.loc[timestamp] = True  # Force exit at same time
                            position_entry_times.append(timestamp)  # Add new position
                            if oldest_time in position_returns:
                                del position_returns[oldest_time]
                            position_returns[timestamp] = 0.0
                            logger.debug(f"Replaced oldest position (from {oldest_time}) with new position at {timestamp}")
                        else:
                            # Fallback to ignore if no positions to replace
                            modified_entries.loc[timestamp] = False
                            
                    elif strategy == "replace_worst":
                        # Close worst performing position and open new one
                        if position_returns:
                            worst_time = min(position_returns.keys(), key=lambda k: position_returns[k])
                            position_entry_times.remove(worst_time)
                            modified_exits.loc[timestamp] = True  # Force exit at same time
                            position_entry_times.append(timestamp)  # Add new position
                            worst_return = position_returns[worst_time]
                            del position_returns[worst_time]
                            position_returns[timestamp] = 0.0
                            logger.debug(f"Replaced worst position (from {worst_time}, return: {worst_return:.2%}) with new position at {timestamp}")
                        else:
                            # Fallback to ignore if no positions to replace
                            modified_entries.loc[timestamp] = False
                    else:
                        # Default to ignore for unknown strategies
                        modified_entries.loc[timestamp] = False
                        logger.warning(f"Unknown max position strategy: {strategy}, defaulting to ignore")
        
        logger.info(f"Max position constraint applied. Final modifications: {modified_entries.sum()} entries, {modified_exits.sum()} exits")
        return modified_entries, modified_exits
    
    def apply_position_direction_filter(self, signals: StrategySignals) -> StrategySignals:
        """
        Filter signals based on position direction preference.
        
        Position Direction Options:
        - LONG_ONLY: Only long positions (buy low, sell high)
        - SHORT_ONLY: Only short positions (sell high, buy low)  
        - BOTH: Both long and short positions allowed
        
        Args:
            signals: Original strategy signals
            
        Returns:
            Filtered signals based on position direction
        """
        direction = self.config.position_direction
        
        if direction == PositionDirection.LONG_ONLY:
            # Keep only long signals (original behavior)
            logger.debug("Applying LONG_ONLY position direction filter")
            return signals
            
        elif direction == PositionDirection.SHORT_ONLY:
            # Reverse signals for short-only trading
            logger.debug("Applying SHORT_ONLY position direction filter")
            filtered_signals = StrategySignals(
                entries=signals.exits.copy(),  # Exit signals become entry signals for shorts
                exits=signals.entries.copy(),   # Entry signals become exit signals for shorts
                # Note: stop loss and take profit logic is already handled correctly in the calculation methods
                stop_losses=signals.stop_losses,
                take_profits=signals.take_profits,
                confidence=signals.confidence,
                metadata={**signals.metadata, "position_direction": "short_only"}
            )
            return filtered_signals
            
        elif direction == PositionDirection.BOTH:
            # Allow both long and short signals
            logger.debug("Applying BOTH position direction filter")
            
            # For "BOTH", we keep original signals as long signals
            # and also create short opportunities
            # This is more nuanced and depends on the strategy
            # For simplicity, we'll allow the original signals to determine direction
            # More sophisticated strategies can override this method
            
            filtered_signals = StrategySignals(
                entries=signals.entries.copy(),  # Keep original entry logic
                exits=signals.exits.copy(),      # Keep original exit logic
                stop_losses=signals.stop_losses,
                take_profits=signals.take_profits,
                confidence=signals.confidence,
                metadata={**signals.metadata, "position_direction": "both"}
            )
            return filtered_signals
        else:
            logger.warning(f"Unknown position direction: {direction}, defaulting to LONG_ONLY")
            return signals
    
    def _calculate_stop_loss_signals(self, data: pd.DataFrame, entries: pd.Series) -> pd.Series:
        """
        Calculate stop loss signals with precise percentage enforcement.
        
        Args:
            data: OHLCV price data
            entries: Entry signals series
            
        Returns:
            Stop loss signals series
        """
        stop_loss_pct = self.config.risk_management.stop_loss
        if stop_loss_pct is None or stop_loss_pct <= 0:
            return pd.Series(False, index=data.index)
            
        # Convert percentage to decimal (5% -> 0.05)
        stop_loss_decimal = stop_loss_pct / 100.0
        
        stop_signals = pd.Series(False, index=data.index)
        
        # Track positions and their stop loss levels
        entry_prices = data['close'][entries]
        for entry_time, entry_price in entry_prices.items():
            # Calculate exact stop loss price
            if self.config.position_direction == PositionDirection.SHORT_ONLY:
                # For short positions, stop loss is above entry price
                stop_price = entry_price * (1 + stop_loss_decimal)
                future_data = data.loc[entry_time:]
                # Use close price for precise stop loss (not high/low which can overshoot)
                stop_hits = future_data['close'] >= stop_price
            else:
                # For long positions, stop loss is below entry price  
                stop_price = entry_price * (1 - stop_loss_decimal)
                future_data = data.loc[entry_time:]
                # Use close price for precise stop loss (not high/low which can overshoot)
                stop_hits = future_data['close'] <= stop_price
            
            if stop_hits.any():
                first_stop = stop_hits.idxmax()
                stop_signals.loc[first_stop] = True
        
        return stop_signals
    
    def _calculate_take_profit_signals(self, data: pd.DataFrame, entries: pd.Series) -> pd.Series:
        """
        Calculate take profit signals with precise percentage enforcement.
        
        Args:
            data: OHLCV price data
            entries: Entry signals series
            
        Returns:
            Take profit signals series
        """
        take_profit_pct = self.config.risk_management.take_profit
        if take_profit_pct is None or take_profit_pct <= 0:
            return pd.Series(False, index=data.index)
            
        # Convert percentage to decimal (10% -> 0.10)
        take_profit_decimal = take_profit_pct / 100.0
        
        profit_signals = pd.Series(False, index=data.index)
        
        entry_prices = data['close'][entries]
        for entry_time, entry_price in entry_prices.items():
            # Calculate exact take profit price
            if self.config.position_direction == PositionDirection.SHORT_ONLY:
                # For short positions, take profit is below entry price
                profit_price = entry_price * (1 - take_profit_decimal)
                future_data = data.loc[entry_time:]
                # Use close price for precise take profit (not high/low which can overshoot)
                profit_hits = future_data['close'] <= profit_price
            else:
                # For long positions, take profit is above entry price
                profit_price = entry_price * (1 + take_profit_decimal)
                future_data = data.loc[entry_time:]
                # Use close price for precise take profit (not high/low which can overshoot)
                profit_hits = future_data['close'] >= profit_price
            
            if profit_hits.any():
                first_profit = profit_hits.idxmax()
                profit_signals.loc[first_profit] = True
        
        return profit_signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy metadata and information"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "parameters": self.config.parameters,
            "position_sizing": self.config.position_sizing.method.value,
            "position_direction": self.config.position_direction.value,
            "risk_management": {
                "stop_loss": self.config.risk_management.stop_loss,
                "take_profit": self.config.risk_management.take_profit,
                "max_position_size": self.config.risk_management.max_position_size,
                "max_positions": self.config.risk_management.max_positions,
                "max_position_strategy": self.config.risk_management.max_position_strategy,
            },
            "execution": {
                "commission": self.config.execution.commission,
                "slippage": self.config.execution.slippage,
            },
            "tags": self.config.tags,
            "created_at": self.config.created_at.isoformat(),
            "modified_at": self.config.modified_at.isoformat(),
        }
    
    def __str__(self) -> str:
        return f"{self.config.name} v{self.config.version}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.config.name}>"
