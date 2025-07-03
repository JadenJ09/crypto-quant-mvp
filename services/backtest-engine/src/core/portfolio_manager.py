"""
Portfolio Manager - Portfolio tracking and performance management

This module handles portfolio-level operations including:
- Capital allocation and tracking
- Portfolio-wide risk management
- Performance calculation and monitoring
- Multi-asset position coordination
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .trade_engine import TradeEngine, Trade, Position


class PortfolioManager:
    """Portfolio management and tracking"""
    
    def __init__(self, initial_capital: float = 100000.0, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.trade_engine = TradeEngine()
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[pd.Timestamp] = []
        self.drawdown_series: List[float] = [0.0]
        self.peak_capital = initial_capital
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
    def update_capital(self, timestamp: pd.Timestamp, market_prices: Dict[str, float]):
        """
        Update portfolio capital based on current positions and market prices
        
        Args:
            timestamp: Current timestamp
            market_prices: Current market prices for all symbols
        """
        # Calculate unrealized PnL
        unrealized_pnls = self.trade_engine.update_positions(market_prices)
        total_unrealized_pnl = sum(unrealized_pnls.values())
        
        # Calculate realized PnL from trades
        realized_pnl = sum(trade.pnl for trade in self.trade_engine.get_trades())
        
        # Update current capital
        self.current_capital = self.initial_capital + realized_pnl + total_unrealized_pnl
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        self.timestamps.append(timestamp)
        
        # Ensure both lists have the same length
        if len(self.equity_curve) != len(self.timestamps):
            # Trim to match the shorter one
            min_len = min(len(self.equity_curve), len(self.timestamps))
            self.equity_curve = self.equity_curve[:min_len]
            self.timestamps = self.timestamps[:min_len]
        
        # Update drawdown metrics
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        self.drawdown_series.append(self.current_drawdown)
    
    def can_open_position(self, symbol: str) -> bool:
        """
        Check if a new position can be opened
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: True if position can be opened
        """
        current_positions = self.trade_engine.get_all_positions()
        
        # Check if already have position in this symbol
        if symbol in current_positions:
            return True  # Can replace existing position
        
        # Check max positions limit
        if len(current_positions) >= self.max_positions:
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate appropriate position size based on risk management
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Risk per trade as decimal (default 2%)
            
        Returns:
            Position size in base currency
        """
        from .trade_engine import calculate_position_size
        
        available_capital = self.current_capital
        return calculate_position_size(
            available_capital, risk_per_trade, entry_price, stop_loss_price
        )
    
    def open_position(self, symbol: str, side: str, entry_price: float,
                     timestamp: pd.Timestamp, stop_loss_pct: Optional[float] = None,
                     take_profit_pct: Optional[float] = None,
                     risk_per_trade: float = 0.02) -> bool:
        """
        Open a new position with proper sizing
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            timestamp: Entry timestamp
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            risk_per_trade: Risk per trade as decimal
            
        Returns:
            bool: True if position opened successfully
        """
        if not self.can_open_position(symbol):
            return False
        
        # Calculate stop loss and take profit prices
        stop_loss_price = None
        take_profit_price = None
        
        if stop_loss_pct is not None:
            from .trade_engine import calculate_stop_loss_price
            stop_loss_price = calculate_stop_loss_price(entry_price, side, stop_loss_pct)
        
        if take_profit_pct is not None:
            from .trade_engine import calculate_take_profit_price
            take_profit_price = calculate_take_profit_price(entry_price, side, take_profit_pct)
        
        # Calculate position size
        if stop_loss_price is not None:
            position_size = self.calculate_position_size(
                symbol, entry_price, stop_loss_price, risk_per_trade
            )
        else:
            # Use fixed percentage of capital if no stop loss
            position_size = self.current_capital * risk_per_trade / entry_price
        
        if position_size <= 0:
            return False
        
        # Open position in trade engine
        return self.trade_engine.open_position(
            symbol=symbol,
            side=side,
            size=position_size,
            price=entry_price,
            timestamp=timestamp,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price
        )
    
    def close_position(self, symbol: str, exit_price: float, 
                      timestamp: pd.Timestamp, exit_reason: str = "signal") -> Optional[Trade]:
        """Close a position"""
        return self.trade_engine.close_position(symbol, exit_price, timestamp, exit_reason)
    
    def close_all_positions(self, market_prices: Dict[str, float], 
                           timestamp: pd.Timestamp) -> List[Trade]:
        """
        Close all open positions
        
        Args:
            market_prices: Current market prices
            timestamp: Exit timestamp
            
        Returns:
            List of trade records
        """
        trades = []
        positions = self.trade_engine.get_all_positions()
        
        for symbol in list(positions.keys()):
            if symbol in market_prices:
                trade = self.close_position(symbol, market_prices[symbol], timestamp, "end")
                if trade:
                    trades.append(trade)
        
        return trades
    
    def process_bar(self, symbol: str, bar_data: Dict, timestamp: pd.Timestamp) -> List[Trade]:
        """
        Process a single bar of data for stop loss/take profit checks
        
        Args:
            symbol: Trading symbol
            bar_data: Bar data with 'open', 'high', 'low', 'close'
            timestamp: Bar timestamp
            
        Returns:
            List of triggered trades
        """
        trades = []
        
        # Check stop losses first (higher priority)
        stop_loss_trade = self.trade_engine.check_stop_losses(
            symbol, bar_data['low'], bar_data['high'], timestamp
        )
        if stop_loss_trade:
            trades.append(stop_loss_trade)
            return trades  # Position closed, no need to check take profit
        
        # Check take profits
        take_profit_trade = self.trade_engine.check_take_profits(
            symbol, bar_data['low'], bar_data['high'], timestamp
        )
        if take_profit_trade:
            trades.append(take_profit_trade)
        
        return trades
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        
        Returns:
            Dictionary of portfolio performance metrics
        """
        if len(self.equity_curve) < 2:
            return {}
        
        # Ensure timestamps and equity curve have same length
        min_len = min(len(self.equity_curve), len(self.timestamps))
        if min_len == 0:
            return {}
        
        equity_curve = self.equity_curve[:min_len]
        timestamps = self.timestamps[:min_len]
        
        equity_series = pd.Series(equity_curve, index=timestamps)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-8) if volatility > 0 else 0.0
        
        # Trade metrics
        trades = self.trade_engine.get_trades()
        num_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown * 100,
            'current_drawdown_pct': self.current_drawdown * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_positions': len(self.trade_engine.get_all_positions())
        }
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series"""
        min_len = min(len(self.equity_curve), len(self.timestamps))
        if min_len == 0:
            return pd.Series(dtype=float)
        return pd.Series(self.equity_curve[:min_len], index=self.timestamps[:min_len])
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown series as pandas Series"""
        min_len = min(len(self.drawdown_series), len(self.timestamps))
        if min_len == 0:
            return pd.Series(dtype=float)
        return pd.Series(self.drawdown_series[:min_len], index=self.timestamps[:min_len])
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as pandas DataFrame"""
        trades = self.trade_engine.get_trades()
        if not trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'side': t.side,
                'size': t.size,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'commission': t.commission,
                'exit_reason': t.exit_reason,
                'duration': (t.exit_time - t.entry_time).total_seconds() / 3600  # hours
            }
            for t in trades
        ])
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.timestamps = []
        self.drawdown_series = [0.0]
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.trade_engine.reset()
