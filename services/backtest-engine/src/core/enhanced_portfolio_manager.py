"""
Enhanced Portfolio Manager with Advanced Risk Management Integration

This module integrates the risk management system with portfolio operations,
providing sophisticated position sizing, risk monitoring, and portfolio-level controls.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trade_engine import TradeEngine, Trade, Position
from risk.risk_manager import RiskManager, RiskLimits, PositionSizingConfig, TakeProfitLevel

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Portfolio state snapshot"""
    timestamp: pd.Timestamp
    total_value: float
    available_cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    positions_count: int
    risk_metrics: Dict[str, float]


class EnhancedPortfolioManager:
    """
    Enhanced portfolio manager with advanced risk management
    
    Features:
    - Integrated risk management system
    - Multiple position sizing methods
    - Portfolio-level risk constraints
    - Advanced performance tracking
    - Multiple take profit levels
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_limits: Optional[RiskLimits] = None,
                 position_sizing: Optional[PositionSizingConfig] = None):
        """
        Initialize enhanced portfolio manager
        
        Args:
            initial_capital: Starting capital
            risk_limits: Portfolio risk limits
            position_sizing: Position sizing configuration
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_cash = initial_capital
        
        # Core components
        self.trade_engine = TradeEngine()
        self.risk_manager = RiskManager(risk_limits, position_sizing)
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[pd.Timestamp] = []
        self.portfolio_states: List[PortfolioState] = []
        
        # Risk tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Position tracking with take profit levels
        self.take_profit_levels: Dict[str, List[TakeProfitLevel]] = {}
        
        # Market data for risk calculations
        self.market_data: Dict[str, pd.DataFrame] = {}
        
    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """Add market data for risk calculations"""
        self.market_data[symbol] = data.copy()
        
    def calculate_volatility(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate asset volatility"""
        if symbol not in self.market_data:
            return None
            
        data = self.market_data[symbol]
        if len(data) < period:
            return None
            
        returns = data['close'].pct_change().dropna()
        if len(returns) < period:
            return None
            
        return returns.rolling(period).std().iloc[-1]
    
    def calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if symbol not in self.market_data:
            return None
            
        data = self.market_data[symbol]
        if len(data) < period + 1:
            return None
            
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else None
    
    def get_kelly_data(self, symbol: str, lookback: int = 100) -> Optional[Dict]:
        """Get data for Kelly criterion calculation"""
        if symbol not in self.market_data:
            return None
            
        data = self.market_data[symbol]
        if len(data) < lookback:
            return None
            
        returns = data['close'].pct_change().dropna().tail(lookback)
        
        return {
            'returns': returns.values,
            'mean_return': returns.mean(),
            'std_return': returns.std()
        }
    
    def update_portfolio(self, timestamp: pd.Timestamp, market_prices: Dict[str, float]):
        """
        Update portfolio state with current market prices
        
        Args:
            timestamp: Current timestamp
            market_prices: Current market prices for all symbols
        """
        # Update positions with current prices
        unrealized_pnls = self.trade_engine.update_positions(market_prices)
        total_unrealized_pnl = sum(unrealized_pnls.values())
        
        # Calculate realized PnL
        trades = self.trade_engine.get_trades()
        total_realized_pnl = sum(trade.pnl for trade in trades if trade.exit_price is not None)
        
        # Update capital
        self.current_capital = self.initial_capital + total_realized_pnl + total_unrealized_pnl
        
        # Update cash (capital not in positions)
        positions = self.trade_engine.get_all_positions()
        position_values = sum(abs(pos.size * market_prices.get(pos.symbol, pos.entry_price)) 
                            for pos in positions.values())
        self.available_cash = self.current_capital - position_values
        
        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update risk manager state
        self.risk_manager.update_risk_state(self.current_capital)
        
        # Calculate portfolio risk metrics
        positions_dict = {symbol: {
            'size': pos.size,
            'current_price': market_prices.get(symbol, pos.entry_price)
        } for symbol, pos in positions.items()}
        
        risk_metrics = self.risk_manager.calculate_portfolio_risk(positions_dict, self.current_capital)
        
        # Create portfolio state snapshot
        portfolio_state = PortfolioState(
            timestamp=timestamp,
            total_value=self.current_capital,
            available_cash=self.available_cash,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=total_realized_pnl,
            drawdown=self.current_drawdown,
            positions_count=len(positions),
            risk_metrics=risk_metrics
        )
        
        # Update tracking arrays
        self.equity_curve.append(self.current_capital)
        self.timestamps.append(timestamp)
        self.portfolio_states.append(portfolio_state)
        
        # Check and execute take profit levels
        self._check_take_profit_levels(market_prices, timestamp)
    
    def open_position(self,
                     symbol: str,
                     side: str,
                     signal_strength: float,
                     entry_price: float,
                     timestamp: pd.Timestamp,
                     stop_loss_pct: Optional[float] = None,
                     take_profit_levels: Optional[List[Tuple[float, float]]] = None) -> Tuple[bool, str]:
        """
        Open position with advanced risk management
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            signal_strength: Signal strength (0-1)
            entry_price: Entry price
            timestamp: Entry timestamp
            stop_loss_pct: Stop loss percentage
            take_profit_levels: List of (price_pct, quantity_pct) tuples
            
        Returns:
            (success, message) tuple
        """
        # Get market data for risk calculations
        volatility = self.calculate_volatility(symbol)
        atr = self.calculate_atr(symbol)
        kelly_data = self.get_kelly_data(symbol)
        
        # Calculate position size using risk manager
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            current_price=entry_price,
            portfolio_value=self.current_capital,
            volatility=volatility,
            atr=atr,
            kelly_data=kelly_data
        )
        
        # Check portfolio risk limits
        positions = self.trade_engine.get_all_positions()
        positions_dict = {s: {'size': p.size, 'current_price': entry_price} 
                         for s, p in positions.items()}
        
        is_allowed, reason = self.risk_manager.check_portfolio_risk_limits(
            self.current_capital, positions_dict, position_size, entry_price
        )
        
        if not is_allowed:
            return False, f"Risk limit violation: {reason}"
        
        # Calculate stop loss price
        stop_loss_price = None
        if stop_loss_pct is not None:
            stop_loss_price = self.risk_manager.calculate_stop_loss(
                entry_price, side, stop_pct=stop_loss_pct
            )
        
        # Open position in trade engine
        success = self.trade_engine.open_position(
            symbol=symbol,
            side=side,
            size=position_size,
            price=entry_price,
            timestamp=timestamp,
            stop_loss=stop_loss_price,
            take_profit=None  # We'll handle multiple TP levels separately
        )
        
        if not success:
            return False, "Failed to open position in trade engine"
        
        # Set up take profit levels
        if take_profit_levels:
            tp_levels = self.risk_manager.create_take_profit_levels(
                entry_price, side, take_profit_levels
            )
            self.take_profit_levels[symbol] = tp_levels
        
        return True, f"Position opened: {position_size:.4f} units at {entry_price}"
    
    def _check_take_profit_levels(self, market_prices: Dict[str, float], timestamp: pd.Timestamp):
        """Check and execute take profit levels"""
        for symbol, tp_levels in list(self.take_profit_levels.items()):
            if symbol not in market_prices:
                continue
                
            current_price = market_prices[symbol]
            position = self.trade_engine.get_position(symbol)
            
            if position is None:
                # Position closed, remove TP levels
                del self.take_profit_levels[symbol]
                continue
            
            # Check each take profit level
            remaining_levels = []
            for tp_level in tp_levels:
                should_trigger = False
                
                if position.side == 'long':
                    target_price = position.entry_price * (1 + tp_level.price_pct)
                    should_trigger = current_price >= target_price
                else:  # short
                    target_price = position.entry_price * (1 + tp_level.price_pct)  # price_pct is negative for shorts
                    should_trigger = current_price <= target_price
                
                if should_trigger:
                    # Execute partial close
                    close_size = abs(position.size) * tp_level.quantity_pct
                    if position.side == 'short':
                        close_size = -close_size
                        
                    self.trade_engine.close_position_partial(
                        symbol=symbol,
                        size=close_size,
                        price=current_price,
                        timestamp=timestamp,
                        exit_reason=f"Take profit level {tp_level.price_pct:.2%}"
                    )
                    
                    logger.info(f"Take profit triggered for {symbol}: {tp_level.quantity_pct:.1%} at {current_price}")
                else:
                    remaining_levels.append(tp_level)
            
            # Update remaining levels
            if remaining_levels:
                self.take_profit_levels[symbol] = remaining_levels
            else:
                del self.take_profit_levels[symbol]
    
    def close_position(self, symbol: str, exit_price: float, timestamp: pd.Timestamp, 
                      reason: str = "Manual close") -> Tuple[bool, str]:
        """Close position and clean up take profit levels"""
        trade = self.trade_engine.close_position(symbol, exit_price, timestamp, reason)
        success = trade is not None
        
        if success and symbol in self.take_profit_levels:
            del self.take_profit_levels[symbol]
            
        return success, "Position closed" if success else "Failed to close position"
    
    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get comprehensive portfolio summary"""
        positions = self.trade_engine.get_all_positions()
        trades = self.trade_engine.get_trades()
        
        # Calculate basic metrics
        total_trades = len([t for t in trades if t.exit_price is not None])
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
            
            sharpe_ratio = 0
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            total_return = 0
            sharpe_ratio = 0
            returns = pd.Series()
        
        # Calculate risk metrics if positions exist and prices available
        risk_metrics = {}
        if positions and current_prices:
            positions_dict = {s: {'size': p.size, 'current_price': current_prices.get(s, p.entry_price)} 
                            for s, p in positions.items()}
            risk_metrics = self.risk_manager.calculate_portfolio_risk(positions_dict, self.current_capital)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'available_cash': self.available_cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / max(total_trades, 1),
            'open_positions': len(positions),
            'risk_metrics': risk_metrics
        }
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if len(self.timestamps) == 0:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve[:len(self.timestamps)],
            'drawdown': [state.drawdown for state in self.portfolio_states]
        })
