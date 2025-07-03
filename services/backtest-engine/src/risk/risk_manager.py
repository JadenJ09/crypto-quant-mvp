"""
Advanced Risk Management Module

This module provides sophisticated risk management capabilities including:
- Multiple position sizing methods (Kelly, fixed, percentage)
- Portfolio-level risk limits (max drawdown, max positions)
- Multiple take profit levels
- Advanced stop loss methods (ATR-based, volatility-based)
- Dynamic position sizing based on volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"
    KELLY_CRITERION = "kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    ATR_BASED = "atr_based"


class StopLossMethod(Enum):
    """Stop loss calculation methods"""
    PERCENTAGE = "percentage"
    ATR_MULTIPLE = "atr_multiple"
    VOLATILITY_BASED = "volatility_based"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class RiskLimits:
    """Portfolio-level risk limits"""
    max_positions: int = 5
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per trade
    max_drawdown: float = 0.15  # 15% max drawdown
    max_concentration: float = 0.30  # 30% max single position size
    max_correlation: float = 0.70  # Max correlation between positions
    
    
@dataclass
class TakeProfitLevel:
    """Individual take profit level"""
    price_pct: float  # Percentage from entry price
    quantity_pct: float  # Percentage of position to close
    
    
@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    method: PositionSizingMethod = PositionSizingMethod.PERCENTAGE
    fixed_amount: float = 1000.0
    percentage: float = 0.02  # 2% of portfolio
    kelly_lookback: int = 100
    kelly_max: float = 0.25  # Max 25% of portfolio
    volatility_target: float = 0.02  # 2% volatility target
    atr_period: int = 14
    atr_multiplier: float = 2.0


class RiskManager:
    """
    Advanced risk management system for backtesting
    
    Features:
    - Multiple position sizing methods
    - Portfolio-level risk constraints
    - Multiple take profit levels
    - Advanced stop loss calculations
    - Dynamic risk adjustment
    """
    
    def __init__(self, 
                 risk_limits: Optional[RiskLimits] = None,
                 position_sizing: Optional[PositionSizingConfig] = None):
        """
        Initialize risk manager
        
        Args:
            risk_limits: Portfolio-level risk limits
            position_sizing: Position sizing configuration
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.position_sizing = position_sizing or PositionSizingConfig()
        
        # Track portfolio state
        self.max_portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.position_correlations = {}
        
    def calculate_position_size(self,
                              signal_strength: float,
                              current_price: float,
                              portfolio_value: float,
                              volatility: Optional[float] = None,
                              atr: Optional[float] = None,
                              kelly_data: Optional[Dict] = None) -> float:
        """
        Calculate position size based on configured method
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            portfolio_value: Current portfolio value
            volatility: Asset volatility (for volatility-based sizing)
            atr: Average True Range (for ATR-based sizing)
            kelly_data: Historical returns data for Kelly criterion
            
        Returns:
            Position size in units
        """
        
        if self.position_sizing.method == PositionSizingMethod.FIXED_AMOUNT:
            return self._fixed_amount_sizing(current_price)
            
        elif self.position_sizing.method == PositionSizingMethod.PERCENTAGE:
            return self._percentage_sizing(current_price, portfolio_value, signal_strength)
            
        elif self.position_sizing.method == PositionSizingMethod.KELLY_CRITERION:
            return self._kelly_sizing(current_price, portfolio_value, kelly_data)
            
        elif self.position_sizing.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_sizing(current_price, portfolio_value, volatility)
            
        elif self.position_sizing.method == PositionSizingMethod.ATR_BASED:
            return self._atr_sizing(current_price, portfolio_value, atr)
            
        else:
            logger.warning(f"Unknown sizing method: {self.position_sizing.method}")
            return self._percentage_sizing(current_price, portfolio_value, signal_strength)
    
    def _fixed_amount_sizing(self, current_price: float) -> float:
        """Fixed dollar amount position sizing"""
        return self.position_sizing.fixed_amount / current_price
    
    def _percentage_sizing(self, current_price: float, portfolio_value: float, 
                          signal_strength: float) -> float:
        """Percentage of portfolio position sizing"""
        base_size = (self.position_sizing.percentage * portfolio_value) / current_price
        return base_size * signal_strength  # Scale by signal strength
    
    def _kelly_sizing(self, current_price: float, portfolio_value: float,
                     kelly_data: Optional[Dict]) -> float:
        """Kelly Criterion position sizing"""
        if not kelly_data or 'returns' not in kelly_data:
            logger.warning("Kelly data not available, falling back to percentage sizing")
            return self._percentage_sizing(current_price, portfolio_value, 1.0)
        
        returns = kelly_data['returns']
        if len(returns) < 10:  # Need minimum data
            return self._percentage_sizing(current_price, portfolio_value, 1.0)
        
        # Calculate Kelly fraction
        mean_return = np.mean(returns)
        var_return = np.var(returns)
        
        if var_return == 0:
            kelly_fraction = 0
        else:
            kelly_fraction = mean_return / var_return
        
        # Ensure Kelly fraction is positive and cap it
        kelly_fraction = max(0, kelly_fraction)  # No negative positions
        kelly_fraction = np.clip(kelly_fraction, 0, self.position_sizing.kelly_max)
        
        return (kelly_fraction * portfolio_value) / current_price
    
    def _volatility_sizing(self, current_price: float, portfolio_value: float,
                          volatility: Optional[float]) -> float:
        """Volatility-adjusted position sizing"""
        if volatility is None or volatility == 0:
            return self._percentage_sizing(current_price, portfolio_value, 1.0)
        
        # Scale position size inversely with volatility
        vol_adjustment = self.position_sizing.volatility_target / volatility
        vol_adjustment = np.clip(vol_adjustment, 0.1, 3.0)  # Reasonable bounds
        
        base_size = (self.position_sizing.percentage * portfolio_value) / current_price
        return base_size * vol_adjustment
    
    def _atr_sizing(self, current_price: float, portfolio_value: float,
                   atr: Optional[float]) -> float:
        """ATR-based position sizing"""
        if atr is None or atr == 0:
            return self._percentage_sizing(current_price, portfolio_value, 1.0)
        
        # Risk amount per trade
        risk_amount = self.risk_limits.max_portfolio_risk * portfolio_value
        
        # Stop loss distance based on ATR
        stop_distance = atr * self.position_sizing.atr_multiplier
        
        # Position size = Risk amount / Stop distance
        return risk_amount / stop_distance
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           direction: str,
                           method: StopLossMethod = StopLossMethod.PERCENTAGE,
                           stop_pct: float = 0.02,
                           atr: Optional[float] = None,
                           volatility: Optional[float] = None) -> float:
        """
        Calculate stop loss price based on method
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            method: Stop loss calculation method
            stop_pct: Stop loss percentage (for percentage method)
            atr: Average True Range
            volatility: Asset volatility
            
        Returns:
            Stop loss price
        """
        
        if method == StopLossMethod.PERCENTAGE:
            if direction == 'long':
                return entry_price * (1 - stop_pct)
            else:  # short
                return entry_price * (1 + stop_pct)
                
        elif method == StopLossMethod.ATR_MULTIPLE:
            if atr is None:
                logger.warning("ATR not provided, falling back to percentage method")
                return self.calculate_stop_loss(entry_price, direction, 
                                              StopLossMethod.PERCENTAGE, stop_pct)
            
            atr_multiple = 2.0  # Default ATR multiplier
            if direction == 'long':
                return entry_price - (atr * atr_multiple)
            else:  # short
                return entry_price + (atr * atr_multiple)
                
        elif method == StopLossMethod.VOLATILITY_BASED:
            if volatility is None:
                logger.warning("Volatility not provided, falling back to percentage method")
                return self.calculate_stop_loss(entry_price, direction,
                                              StopLossMethod.PERCENTAGE, stop_pct)
            
            # Use 2 standard deviations as stop distance
            vol_multiple = 2.0
            stop_distance = entry_price * volatility * vol_multiple
            
            if direction == 'long':
                return entry_price - stop_distance
            else:  # short
                return entry_price + stop_distance
                
        else:
            logger.warning(f"Unknown stop loss method: {method}")
            return self.calculate_stop_loss(entry_price, direction,
                                          StopLossMethod.PERCENTAGE, stop_pct)
    
    def create_take_profit_levels(self,
                                entry_price: float,
                                direction: str,
                                levels: List[Tuple[float, float]]) -> List[TakeProfitLevel]:
        """
        Create take profit levels
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'  
            levels: List of (price_pct, quantity_pct) tuples
            
        Returns:
            List of TakeProfitLevel objects
        """
        tp_levels = []
        
        for price_pct, quantity_pct in levels:
            if direction == 'long':
                tp_price_pct = price_pct  # Profit above entry
            else:  # short
                tp_price_pct = -price_pct  # Profit below entry
                
            tp_levels.append(TakeProfitLevel(
                price_pct=tp_price_pct,
                quantity_pct=quantity_pct
            ))
        
        return tp_levels
    
    def check_portfolio_risk_limits(self,
                                  portfolio_value: float,
                                  positions: Dict,
                                  proposed_position_size: float,
                                  asset_price: float) -> Tuple[bool, str]:
        """
        Check if proposed trade violates portfolio risk limits
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions dictionary
            proposed_position_size: Size of proposed position
            asset_price: Price of asset for proposed position
            
        Returns:
            (is_allowed, reason) tuple
        """
        
        # Check maximum number of positions
        if len(positions) >= self.risk_limits.max_positions:
            return False, f"Maximum positions limit reached ({self.risk_limits.max_positions})"
        
        # Check maximum concentration
        position_value = abs(proposed_position_size * asset_price)
        concentration = position_value / portfolio_value
        
        if concentration > self.risk_limits.max_concentration:
            return False, f"Position too large: {concentration:.2%} > {self.risk_limits.max_concentration:.2%}"
        
        # Check maximum drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        
        if self.current_drawdown > self.risk_limits.max_drawdown:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
        
        return True, "Risk limits satisfied"
    
    def calculate_portfolio_risk(self, positions: Dict, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics
        
        Args:
            positions: Current positions
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of risk metrics
        """
        total_exposure = sum(abs(pos.get('size', 0) * pos.get('current_price', 0)) 
                           for pos in positions.values())
        
        long_exposure = sum(pos.get('size', 0) * pos.get('current_price', 0)
                          for pos in positions.values() if pos.get('size', 0) > 0)
        
        short_exposure = sum(abs(pos.get('size', 0) * pos.get('current_price', 0))
                           for pos in positions.values() if pos.get('size', 0) < 0)
        
        return {
            'total_exposure': total_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'leverage': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'positions_count': len(positions)
        }
    
    def update_risk_state(self, portfolio_value: float):
        """Update risk manager state with current portfolio value"""
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
