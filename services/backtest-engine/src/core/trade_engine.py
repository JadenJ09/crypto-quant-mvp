"""
Trade Engine - Core trade execution and position management

This module handles the fundamental trade execution logic including:
- Order processing and fills
- Position opening and closing
- Stop loss and take profit execution
- Trade tracking and validation
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
import numpy as np
from numba import njit
import pandas as pd


class OrderType(Enum):
    """Order types supported by the engine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Trade(NamedTuple):
    """Individual trade record"""
    trade_id: int
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    commission: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'


class Position:
    """Active position tracking"""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float, 
                 entry_time: pd.Timestamp, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.unrealized_pnl = 0.0
        
    def update_unrealized_pnl(self, current_price: float) -> float:
        """Update and return unrealized PnL"""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        if self.stop_loss is None:
            return False
        
        if self.side == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should trigger"""
        if self.take_profit is None:
            return False
        
        if self.side == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit


class TradeEngine:
    """Core trade execution engine"""
    
    def __init__(self, commission_rate: float = 0.001, slippage: float = 0.0001):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
    def open_position(self, symbol: str, side: str, size: float, price: float,
                     timestamp: pd.Timestamp, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            price: Entry price
            timestamp: Entry timestamp
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            bool: True if position opened successfully
        """
        # Close existing position for the symbol if it exists
        if symbol in self.positions:
            self.close_position(symbol, price, timestamp, "signal")
        
        # Apply slippage
        if side == 'long':
            execution_price = price * (1 + self.slippage)
        else:  # short
            execution_price = price * (1 - self.slippage)
        
        # Create new position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=execution_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: pd.Timestamp,
                      exit_reason: str = "signal") -> Optional[Trade]:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for exit ('signal', 'stop_loss', 'take_profit')
            
        Returns:
            Trade: Trade record if position was closed
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Apply slippage
        if position.side == 'long':
            execution_price = price * (1 - self.slippage)
        else:  # short
            execution_price = price * (1 + self.slippage)
        
        # Calculate PnL
        if position.side == 'long':
            pnl = (execution_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - execution_price) * position.size
        
        # Calculate commission
        commission = (position.entry_price * position.size * self.commission_rate +
                     execution_price * position.size * self.commission_rate)
        
        # Net PnL after commission
        net_pnl = pnl - commission
        
        # Create trade record
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=execution_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            commission=commission,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        self.trade_counter += 1
        
        # Remove position
        del self.positions[symbol]
        
        return trade
    
    def close_position_partial(self, symbol: str, size: float, price: float,
                             timestamp: pd.Timestamp, exit_reason: str = "partial_close") -> Optional[Trade]:
        """
        Partially close an existing position
        
        Args:
            symbol: Trading symbol
            size: Size to close (negative for shorts)
            price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for exit
            
        Returns:
            Trade: Trade record for the partial close
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Validate partial close size
        if position.side == 'long':
            if size <= 0 or size > position.size:
                return None
        else:  # short
            if size >= 0 or abs(size) > abs(position.size):
                return None
        
        # Apply slippage
        if position.side == 'long':
            execution_price = price * (1 - self.slippage)
        else:  # short
            execution_price = price * (1 + self.slippage)
        
        # Calculate PnL for partial close
        if position.side == 'long':
            pnl = (execution_price - position.entry_price) * size
        else:  # short
            pnl = (position.entry_price - execution_price) * abs(size)
        
        # Calculate commission for partial close
        commission = execution_price * abs(size) * self.commission_rate
        net_pnl = pnl - commission
        
        # Create trade record for partial close
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=position.side,
            size=abs(size),
            entry_price=position.entry_price,
            exit_price=execution_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            commission=commission,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        self.trade_counter += 1
        
        # Reduce position size
        if position.side == 'long':
            position.size -= size
        else:  # short
            position.size += abs(size)  # Reduce negative size
        
        # Remove position if fully closed
        if abs(position.size) < 1e-8:  # Account for floating point precision
            del self.positions[symbol]
        
        return trade
    
    def check_stop_losses(self, symbol: str, low_price: float, high_price: float,
                         timestamp: pd.Timestamp) -> Optional[Trade]:
        """
        Check if stop loss should trigger based on bar low/high
        
        Args:
            symbol: Trading symbol
            low_price: Bar low price
            high_price: Bar high price
            timestamp: Current timestamp
            
        Returns:
            Trade: Trade record if stop loss triggered
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Check stop loss
        if position.side == 'long' and position.stop_loss is not None:
            if low_price <= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, timestamp, "stop_loss")
        elif position.side == 'short' and position.stop_loss is not None:
            if high_price >= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, timestamp, "stop_loss")
        
        return None
    
    def check_take_profits(self, symbol: str, low_price: float, high_price: float,
                          timestamp: pd.Timestamp) -> Optional[Trade]:
        """
        Check if take profit should trigger based on bar low/high
        
        Args:
            symbol: Trading symbol
            low_price: Bar low price
            high_price: Bar high price
            timestamp: Current timestamp
            
        Returns:
            Trade: Trade record if take profit triggered
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Check take profit
        if position.side == 'long' and position.take_profit is not None:
            if high_price >= position.take_profit:
                return self.close_position(symbol, position.take_profit, timestamp, "take_profit")
        elif position.side == 'short' and position.take_profit is not None:
            if low_price <= position.take_profit:
                return self.close_position(symbol, position.take_profit, timestamp, "take_profit")
        
        return None
    
    def update_positions(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update unrealized PnL for all positions
        
        Args:
            prices: Current prices for symbols
            
        Returns:
            Dict of symbol -> unrealized PnL
        """
        unrealized_pnls = {}
        for symbol, position in self.positions.items():
            if symbol in prices:
                unrealized_pnls[symbol] = position.update_unrealized_pnl(prices[symbol])
        return unrealized_pnls
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions"""
        return self.positions.copy()
    
    def get_trades(self) -> List[Trade]:
        """Get all completed trades"""
        return self.trades.copy()
    
    def reset(self):
        """Reset engine state"""
        self.positions.clear()
        self.trades.clear()
        self.trade_counter = 0


@njit
def calculate_stop_loss_price(entry_price: float, side: str, stop_loss_pct: float) -> float:
    """
    Calculate stop loss price with numba optimization
    
    Args:
        entry_price: Entry price
        side: 'long' or 'short'
        stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
        
    Returns:
        Stop loss price
    """
    if side == 'long':
        return entry_price * (1.0 - stop_loss_pct)
    else:  # short
        return entry_price * (1.0 + stop_loss_pct)


@njit
def calculate_take_profit_price(entry_price: float, side: str, take_profit_pct: float) -> float:
    """
    Calculate take profit price with numba optimization
    
    Args:
        entry_price: Entry price
        side: 'long' or 'short'
        take_profit_pct: Take profit percentage (e.g., 0.10 for 10%)
        
    Returns:
        Take profit price
    """
    if side == 'long':
        return entry_price * (1.0 + take_profit_pct)
    else:  # short
        return entry_price * (1.0 - take_profit_pct)


@njit
def calculate_position_size(capital: float, risk_per_trade: float, entry_price: float,
                           stop_loss_price: float) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        capital: Available capital
        risk_per_trade: Risk per trade as decimal (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        
    Returns:
        Position size
    """
    risk_amount = capital * risk_per_trade
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff == 0:
        return 0.0
    
    return risk_amount / price_diff
