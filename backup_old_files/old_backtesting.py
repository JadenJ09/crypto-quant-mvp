# ==============================================================================
# File: services/api/app/backtesting.py
# Description: Comprehensive backtesting engine with multiple strategies and ML support
# ==============================================================================
import vectorbt as vbt
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging

# Configure vectorbt settings for better performance
vbt.settings.array_wrapper['freq'] = 'T'  # Set default frequency
vbt.settings.portfolio['call_seq'] = 'auto'  # Auto call sequence

class TradeInfo(BaseModel):
    """Individual trade information"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    size: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_minutes: Optional[int] = None
    entry_reason: str
    exit_reason: Optional[str] = None

class BacktestStats(BaseModel):
    """Enhanced Pydantic model for comprehensive backtest results."""
    # Performance Metrics
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return_pct: float = Field(..., description="Annualized return percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    
    # Trade Statistics
    total_trades: int = Field(..., description="Total number of trades")
    win_rate_pct: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    avg_win_pct: float = Field(..., description="Average winning trade percentage")
    avg_loss_pct: float = Field(..., description="Average losing trade percentage")
    
    # Risk Metrics
    volatility_pct: float = Field(..., description="Annual volatility percentage")
    var_95_pct: float = Field(..., description="Value at Risk (95%)")
    
    # Duration Statistics
    avg_trade_duration_hours: float = Field(..., description="Average trade duration in hours")
    max_trade_duration_hours: float = Field(..., description="Maximum trade duration in hours")
    
    # Additional Metrics
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_cash: float = Field(..., description="Initial capital")
    final_value: float = Field(..., description="Final portfolio value")
    
    # Trade Details
    trades: List[TradeInfo] = Field(default_factory=list, description="Individual trade details")

class StrategyConfig(BaseModel):
    """Configuration for trading strategies"""
    strategy_type: str = Field(..., description="Type of strategy (ma_crossover, rsi_oversold, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Strategy-specific parameters")
    
    # Position sizing
    position_size_type: str = Field(default="fixed_amount", description="Position sizing method")
    position_size_value: float = Field(default=1000.0, description="Position size value")
    
    # Risk management
    stop_loss_pct: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, description="Take profit percentage")
    max_positions: int = Field(default=1, description="Maximum concurrent positions")

class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (5min, 1hour, etc.)")
    start_date: Optional[datetime] = Field(None, description="Backtest start date")
    end_date: Optional[datetime] = Field(None, description="Backtest end date")
    initial_cash: float = Field(default=100000.0, description="Initial capital")
    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    commission: float = Field(default=0.001, description="Commission rate (0.1%)")

def calculate_enhanced_stats(portfolio, trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio statistics
    
    Args:
        portfolio: vectorbt Portfolio object
        trades_df: Optional DataFrame with individual trade details
        
    Returns:
        Dictionary with enhanced statistics
    """
    stats = portfolio.stats()
    
    # Basic metrics
    total_return = stats.get('Total Return [%]', 0.0)
    max_drawdown = abs(stats.get('Max Drawdown [%]', 0.0))  # Make positive
    sharpe = stats.get('Sharpe Ratio', 0.0)
    
    # Enhanced metrics
    returns = portfolio.returns()
    annual_vol = returns.std() * np.sqrt(252 * 24 * 60 / 5) * 100  # Assuming 5min data
    
    # Risk metrics
    var_95 = np.percentile(returns.dropna(), 5) * 100 if len(returns.dropna()) > 0 else 0.0
    
    # Sortino ratio (using downside deviation)
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252 * 24 * 60 / 5) if len(negative_returns) > 0 else 0.0
    sortino = (returns.mean() * 252 * 24 * 60 / 5) / downside_vol if downside_vol > 0 else 0.0
    
    # Calmar ratio
    calmar = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0.0
    
    # Trade statistics
    trade_stats = {
        'total_trades': stats.get('Total Trades', 0),
        'win_rate_pct': stats.get('Win Rate [%]', 0.0),
        'profit_factor': stats.get('Profit Factor', 0.0),
        'avg_win_pct': stats.get('Avg Winning Trade [%]', 0.0),
        'avg_loss_pct': abs(stats.get('Avg Losing Trade [%]', 0.0)),
    }
    
    # Duration statistics (if trades available)
    duration_stats = {
        'avg_trade_duration_hours': 0.0,
        'max_trade_duration_hours': 0.0
    }
    
    if trades_df is not None and not trades_df.empty:
        durations = trades_df['Exit Timestamp'] - trades_df['Entry Timestamp']
        duration_hours = durations.dt.total_seconds() / 3600
        duration_stats['avg_trade_duration_hours'] = float(duration_hours.mean())
        duration_stats['max_trade_duration_hours'] = float(duration_hours.max())
    
    return {
        'total_return_pct': total_return,
        'annualized_return_pct': returns.mean() * 252 * 24 * 60 / 5 * 100,  # Assuming 5min data
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'volatility_pct': annual_vol,
        'var_95_pct': var_95,
        **trade_stats,
        **duration_stats
    }

def run_ma_crossover_strategy(
    price_data: pd.DataFrame, 
    fast_window: int = 10, 
    slow_window: int = 20,
    initial_cash: float = 100000.0,
    commission: float = 0.001
) -> BacktestStats:
    """
    Enhanced Moving Average Crossover backtest with better error handling and metrics.
    """
    try:
        # Calculate indicators
        fast_ma = vbt.MA.run(price_data['close'], window=fast_window, short_name='fast')
        slow_ma = vbt.MA.run(price_data['close'], window=slow_window, short_name='slow')
        
        # Generate signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        # Get trade details
        trades = portfolio.trades.records_readable
        trade_list = []
        
        if not trades.empty:
            for _, trade in trades.iterrows():
                trade_info = TradeInfo(
                    entry_time=trade['Entry Timestamp'],
                    exit_time=trade['Exit Timestamp'] if pd.notna(trade['Exit Timestamp']) else None,
                    entry_price=trade['Entry Price'],
                    exit_price=trade['Exit Price'] if pd.notna(trade['Exit Price']) else None,
                    size=trade['Size'],
                    side='long',  # MA crossover is long-only
                    pnl=trade['PnL'],
                    pnl_pct=trade['Return [%]'],
                    duration_minutes=int((trade['Exit Timestamp'] - trade['Entry Timestamp']).total_seconds() / 60) if pd.notna(trade['Exit Timestamp']) else None,
                    entry_reason=f"Fast MA({fast_window}) crossed above Slow MA({slow_window})",
                    exit_reason=f"Fast MA({fast_window}) crossed below Slow MA({slow_window})" if pd.notna(trade['Exit Timestamp']) else None
                )
                trade_list.append(trade_info)
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=trade_list
        )
        
    except Exception as e:
        logging.error(f"Error in MA crossover strategy: {e}")
        # Return empty stats on error
        return BacktestStats(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            volatility_pct=0.0,
            var_95_pct=0.0,
            avg_trade_duration_hours=0.0,
            max_trade_duration_hours=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=initial_cash,
            final_value=initial_cash,
            trades=[]
        )

def run_rsi_oversold_strategy(
    price_data: pd.DataFrame,
    rsi_period: int = 14,
    oversold_level: float = 30,
    overbought_level: float = 70,
    initial_cash: float = 100000.0,
    commission: float = 0.001
) -> BacktestStats:
    """
    RSI Oversold/Overbought strategy
    """
    try:
        # Check if RSI data is available
        rsi_column = f'rsi_{rsi_period}'
        if rsi_column not in price_data.columns:
            # Calculate RSI if not available
            rsi = vbt.RSI.run(price_data['close'], window=rsi_period).rsi
        else:
            rsi = price_data[rsi_column]
        
        # Generate signals
        entries = rsi < oversold_level
        exits = rsi > overbought_level
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        # Get trade details
        trades = portfolio.trades.records_readable
        trade_list = []
        
        if not trades.empty:
            for _, trade in trades.iterrows():
                entry_rsi = rsi.loc[trade['Entry Timestamp']] if trade['Entry Timestamp'] in rsi.index else None
                exit_rsi = rsi.loc[trade['Exit Timestamp']] if pd.notna(trade['Exit Timestamp']) and trade['Exit Timestamp'] in rsi.index else None
                
                trade_info = TradeInfo(
                    entry_time=trade['Entry Timestamp'],
                    exit_time=trade['Exit Timestamp'] if pd.notna(trade['Exit Timestamp']) else None,
                    entry_price=trade['Entry Price'],
                    exit_price=trade['Exit Price'] if pd.notna(trade['Exit Price']) else None,
                    size=trade['Size'],
                    side='long',
                    pnl=trade['PnL'],
                    pnl_pct=trade['Return [%]'],
                    duration_minutes=int((trade['Exit Timestamp'] - trade['Entry Timestamp']).total_seconds() / 60) if pd.notna(trade['Exit Timestamp']) else None,
                    entry_reason=f"RSI({rsi_period}) < {oversold_level} (RSI: {entry_rsi:.2f})" if entry_rsi else f"RSI({rsi_period}) < {oversold_level}",
                    exit_reason=f"RSI({rsi_period}) > {overbought_level} (RSI: {exit_rsi:.2f})" if exit_rsi else f"RSI({rsi_period}) > {overbought_level}"
                )
                trade_list.append(trade_info)
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=trade_list
        )
        
    except Exception as e:
        logging.error(f"Error in RSI strategy: {e}")
        return BacktestStats(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            volatility_pct=0.0,
            var_95_pct=0.0,
            avg_trade_duration_hours=0.0,
            max_trade_duration_hours=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=initial_cash,
            final_value=initial_cash,
            trades=[]
        )

def run_bollinger_bands_strategy(
    price_data: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    initial_cash: float = 100000.0,
    commission: float = 0.001
) -> BacktestStats:
    """
    Bollinger Bands Mean Reversion Strategy
    """
    try:
        # Check if Bollinger Bands data is available
        if all(col in price_data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_upper = price_data['bb_upper']
            bb_lower = price_data['bb_lower']
            bb_middle = price_data['bb_middle']
        else:
            # Calculate Bollinger Bands if not available
            bb = vbt.BBANDS.run(price_data['close'], window=bb_period, alpha=bb_std)
            bb_upper = bb.upper
            bb_lower = bb.lower
            bb_middle = bb.middle
        
        # Generate signals - Buy when price touches lower band, sell when it reaches middle
        entries = price_data['close'] <= bb_lower
        exits = price_data['close'] >= bb_middle
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=[]  # Simplified for now
        )
        
    except Exception as e:
        logging.error(f"Error in Bollinger Bands strategy: {e}")
        return BacktestStats(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            volatility_pct=0.0,
            var_95_pct=0.0,
            avg_trade_duration_hours=0.0,
            max_trade_duration_hours=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=initial_cash,
            final_value=initial_cash,
            trades=[]
        )

def run_custom_multi_indicator_strategy(
    price_data: pd.DataFrame,
    strategy_config: Dict[str, Any],
    initial_cash: float = 100000.0,
    commission: float = 0.001
) -> BacktestStats:
    """
    Custom multi-indicator strategy that can combine various technical indicators
    This is where ML-generated signals would be integrated
    """
    try:
        # This is a placeholder for more complex strategies
        # In the future, this would integrate with ML models
        
        # For now, implement a simple multi-indicator approach
        conditions = strategy_config.get('conditions', [])
        
        # Initialize entry/exit signals
        entries = pd.Series(False, index=price_data.index)
        exits = pd.Series(False, index=price_data.index)
        
        # Process each condition (this is a simplified example)
        for condition in conditions:
            indicator = condition.get('indicator')
            operator = condition.get('operator')
            value = condition.get('value')
            signal_type = condition.get('signal_type', 'entry')
            
            if indicator in price_data.columns:
                if operator == 'greater_than':
                    condition_met = price_data[indicator] > value
                elif operator == 'less_than':
                    condition_met = price_data[indicator] < value
                elif operator == 'crosses_above':
                    # This would need more sophisticated implementation
                    condition_met = price_data[indicator] > value
                else:
                    continue
                
                if signal_type == 'entry':
                    entries |= condition_met
                elif signal_type == 'exit':
                    exits |= condition_met
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=[]  # Simplified for now
        )
        
    except Exception as e:
        logging.error(f"Error in custom multi-indicator strategy: {e}")
        return BacktestStats(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            volatility_pct=0.0,
            var_95_pct=0.0,
            avg_trade_duration_hours=0.0,
            max_trade_duration_hours=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=initial_cash,
            final_value=initial_cash,
            trades=[]
        )

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'ma_crossover': run_ma_crossover_strategy,
    'rsi_oversold': run_rsi_oversold_strategy,
    'bollinger_bands': run_bollinger_bands_strategy,
    'multi_indicator': run_custom_multi_indicator_strategy,
}

def get_available_strategies() -> List[Dict[str, Any]]:
    """
    Get list of available backtesting strategies with their parameters
    """
    return [
        {
            "name": "ma_crossover",
            "display_name": "Moving Average Crossover",
            "description": "Buy when fast MA crosses above slow MA, sell when fast MA crosses below slow MA",
            "parameters": [
                {"name": "fast_window", "type": "int", "default": 10, "min": 1, "max": 100, "description": "Fast MA period"},
                {"name": "slow_window", "type": "int", "default": 20, "min": 1, "max": 200, "description": "Slow MA period"}
            ]
        },
        {
            "name": "rsi_oversold",
            "display_name": "RSI Oversold/Overbought",
            "description": "Buy when RSI < oversold level, sell when RSI > overbought level",
            "parameters": [
                {"name": "rsi_period", "type": "int", "default": 14, "min": 1, "max": 50, "description": "RSI calculation period"},
                {"name": "oversold_level", "type": "float", "default": 30, "min": 10, "max": 40, "description": "RSI oversold threshold"},
                {"name": "overbought_level", "type": "float", "default": 70, "min": 60, "max": 90, "description": "RSI overbought threshold"}
            ]
        },
        {
            "name": "bollinger_bands",
            "display_name": "Bollinger Bands Mean Reversion",
            "description": "Buy when price touches lower band, sell when price reaches middle band",
            "parameters": [
                {"name": "bb_period", "type": "int", "default": 20, "min": 5, "max": 50, "description": "Bollinger Bands period"},
                {"name": "bb_std", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "description": "Standard deviation multiplier"}
            ]
        },
        {
            "name": "multi_indicator",
            "display_name": "Multi-Indicator Custom Strategy",
            "description": "Custom strategy combining multiple technical indicators",
            "parameters": [
                {"name": "conditions", "type": "array", "description": "Array of indicator conditions"}
            ]
        }
    ]