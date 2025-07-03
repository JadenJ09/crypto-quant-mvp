"""
Core Engine Components

This module contains the core components of the backtesting engine:
- TradeEngine: Trade execution and position management
- PortfolioManager: Portfolio-level tracking and risk management  
- BacktestExecutor: Main backtesting coordination
"""

from .trade_engine import (
    TradeEngine, 
    Trade, 
    Position, 
    OrderType, 
    OrderSide, 
    OrderStatus,
    calculate_stop_loss_price,
    calculate_take_profit_price,
    calculate_position_size
)

from .portfolio_manager import PortfolioManager

from .backtest_executor import BacktestExecutor

__all__ = [
    'TradeEngine',
    'Trade', 
    'Position',
    'OrderType',
    'OrderSide', 
    'OrderStatus',
    'calculate_stop_loss_price',
    'calculate_take_profit_price',
    'calculate_position_size',
    'PortfolioManager',
    'BacktestExecutor'
]
