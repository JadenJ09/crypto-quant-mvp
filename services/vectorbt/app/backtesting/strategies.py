# ==============================================================================
# File: services/vectorbt/app/backtesting/strategies.py
# Description: All vectorbt strategy implementations
# ==============================================================================

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Configure vectorbt settings for better performance
try:
    vbt.settings.array_wrapper['freq'] = 'T'  # Set default frequency
    vbt.settings.portfolio['call_seq'] = 'auto'  # Auto call sequence
except:
    # Handle older versions of vectorbt
    pass

from .utils import calculate_enhanced_stats, sanitize_float
from .indicator_engine import indicator_engine
from ..models import BacktestStats, TradeInfo

def run_ma_crossover_strategy(
    price_data: pd.DataFrame, 
    fast_window: int = 10, 
    slow_window: int = 20,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    position_sizing: str = "percentage",
    position_size: float = 0.1,
    max_positions: int = 1
) -> BacktestStats:
    """
    Enhanced Moving Average Crossover backtest with better error handling and metrics.
    """
    try:
        # Calculate indicators using IndicatorEngine
        fast_ma = indicator_engine.sma(price_data['close'], window=fast_window)
        slow_ma = indicator_engine.sma(price_data['close'], window=slow_window)
        
        # Generate signals - crossover detection
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Apply stop loss and take profit if specified
        if stop_loss or take_profit:
            logger.info(f"Applying stop loss: {stop_loss}% and take profit: {take_profit}%")
            
            # Create price-based exit signals for stop loss and take profit
            close_prices = price_data['close']
            
            # Track entry prices for stop loss/take profit calculation
            entry_prices = close_prices.where(entries).ffill()
            
            # Stop loss exits: price drops below entry_price * (1 - stop_loss/100)
            stop_loss_exits = pd.Series(False, index=price_data.index)
            if stop_loss:
                stop_loss_level = entry_prices * (1 - stop_loss / 100)
                stop_loss_exits = (close_prices < stop_loss_level) & (entry_prices.notna())
            
            # Take profit exits: price rises above entry_price * (1 + take_profit/100)
            take_profit_exits = pd.Series(False, index=price_data.index)
            if take_profit:
                take_profit_level = entry_prices * (1 + take_profit / 100)
                take_profit_exits = (close_prices > take_profit_level) & (entry_prices.notna())
            
            # Combine original exits with stop loss and take profit
            exits = exits | stop_loss_exits | take_profit_exits
        
        # Calculate position size based on sizing method
        if position_sizing == "percentage":
            # Percentage of portfolio per position
            # Convert percentage to decimal (10% -> 0.1)
            if position_size > 1.0:
                # Frontend likely sent percentage (10 for 10%), convert to decimal
                size = position_size / 100.0
            else:
                # Already a decimal (0.1 for 10%)
                size = position_size
            # Adjust for max positions
            if max_positions > 1:
                size = size / max_positions  # Divide available capital among max positions
        elif position_sizing == "fixed":
            # Fixed dollar amount - convert to fraction of portfolio
            size = position_size / initial_cash
            # Adjust for max positions  
            if max_positions > 1:
                size = min(size, 1.0 / max_positions)  # Don't exceed available capital
        else:
            # Default to percentage
            if position_size > 1.0:
                size = position_size / 100.0  # Convert percentage to decimal
            else:
                size = position_size
            if max_positions > 1:
                size = size / max_positions
        
        logger.info(f"Position sizing: {position_sizing}, size: {position_size}, calculated size: {size}, max_positions: {max_positions}")
        
        # DEBUG: Log price data statistics
        logger.info(f"Price data shape: {price_data.shape}")
        logger.info(f"Close price stats - min: {price_data['close'].min():.2f}, max: {price_data['close'].max():.2f}, mean: {price_data['close'].mean():.2f}")
        logger.info(f"Sample close prices: {price_data['close'].head(3).tolist()}")
        
        # Advanced portfolio construction with stop loss and take profit
        if stop_loss or take_profit:
            logger.info(f"Risk management enabled - Stop loss: {stop_loss}%, Take profit: {take_profit}%")
        
        # Standard portfolio construction (risk management is handled in exit signals above)
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission,
            slippage=slippage if slippage else 0.0,
            size=size,  # Apply position sizing
            size_type='targetpercent'  # Use target percentage of portfolio
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        # Get trade details with better error handling
        trades = portfolio.trades.records_readable
        trade_list = []
        
        if not trades.empty:
            # Debug: Log available columns
            logger.info(f"Available trade columns: {list(trades.columns)}")
            
            for _, trade in trades.iterrows():
                try:
                    # Try different possible column names
                    entry_price = None
                    exit_price = None
                    pnl = None
                    pnl_pct = None
                    
                    # Handle different possible column names for VectorBT
                    for col in trades.columns:
                        if 'avg entry price' in col.lower():
                            entry_price = trade[col]
                        elif 'avg exit price' in col.lower():
                            exit_price = trade[col]
                        elif col.lower() in ['pnl', 'p&l']:
                            pnl = trade[col]
                        elif col.lower() == 'return':
                            pnl_pct = trade[col]
                    
                    # Fallback to direct access with correct field names
                    if entry_price is None:
                        entry_price = trade.get('Avg Entry Price', trade.get('Entry Price', trade.get('entry_price', 0)))
                    if exit_price is None:
                        exit_price = trade.get('Avg Exit Price', trade.get('Exit Price', trade.get('exit_price', None)))
                    if pnl is None:
                        pnl = trade.get('PnL', trade.get('pnl', 0))
                    if pnl_pct is None:
                        pnl_pct = trade.get('Return', trade.get('Return [%]', trade.get('return_pct', 0)))
                    
                    # VectorBT returns the actual trade return as decimal (0.007 = 0.7%)
                    # Convert to percentage for frontend display
                    if pnl_pct is not None:
                        pnl_pct = pnl_pct * 100.0
                    
                    # Calculate size properly - VectorBT Size is the actual crypto amount
                    trade_size = trade.get('Size', trade.get('size', 0))
                    # VectorBT Size field already represents the correct crypto units
                    actual_size = trade_size
                    
                    trade_info = TradeInfo(
                        entry_time=trade.get('Entry Timestamp', trade.get('entry_time', datetime.now())),
                        exit_time=trade.get('Exit Timestamp', trade.get('exit_time', None)) if pd.notna(trade.get('Exit Timestamp', trade.get('exit_time', None))) else None,
                        entry_price=entry_price,
                        exit_price=exit_price if pd.notna(exit_price) else None,
                        size=actual_size,
                        side='long',  # MA crossover is long-only
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        duration_minutes=int((trade.get('Exit Timestamp', trade.get('exit_time', datetime.now())) - trade.get('Entry Timestamp', trade.get('entry_time', datetime.now()))).total_seconds() / 60) if pd.notna(trade.get('Exit Timestamp', trade.get('exit_time', None))) else None,
                        entry_reason=f"Fast MA({fast_window}) crossed above Slow MA({slow_window})",
                        exit_reason=f"Fast MA({fast_window}) crossed below Slow MA({slow_window})" if pd.notna(trade.get('Exit Timestamp', trade.get('exit_time', None))) else None
                    )
                    trade_list.append(trade_info)
                except Exception as e:
                    logger.error(f"Error processing trade: {e}")
                    continue
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=trade_list,
            # Include indicator data for frontend charting
            indicators={
                "sma_fast": fast_ma.dropna().tolist(),
                "sma_slow": slow_ma.dropna().tolist()
            },
            timestamps=[ts.isoformat() for ts in fast_ma.dropna().index.tolist()]
        )
        
    except Exception as e:
        logging.error(f"Error in MA crossover strategy: {e}")
        return _create_empty_stats(initial_cash)

def run_rsi_oversold_strategy(
    price_data: pd.DataFrame,
    rsi_period: int = 14,
    oversold_level: float = 30,
    overbought_level: float = 70,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    position_sizing: str = "percentage",
    position_size: float = 0.2,
    max_positions: int = 1
) -> BacktestStats:
    """
    RSI Oversold/Overbought strategy
    """
    try:
        # Calculate RSI using IndicatorEngine
        rsi = indicator_engine.rsi(price_data['close'], window=rsi_period)
        
        # Generate signals
        entries = rsi < oversold_level
        exits = rsi > overbought_level
        
        # Apply stop loss and take profit if specified
        if stop_loss or take_profit:
            logger.info(f"RSI Strategy - Applying stop loss: {stop_loss}% and take profit: {take_profit}%")
            
            # Create price-based exit signals for stop loss and take profit
            close_prices = price_data['close']
            
            # Track entry prices for stop loss/take profit calculation
            entry_prices = close_prices.where(entries).ffill()
            
            # Stop loss exits: price drops below entry_price * (1 - stop_loss/100)
            stop_loss_exits = pd.Series(False, index=price_data.index)
            if stop_loss:
                stop_loss_level = entry_prices * (1 - stop_loss / 100)
                stop_loss_exits = (close_prices < stop_loss_level) & (entry_prices.notna())
            
            # Take profit exits: price rises above entry_price * (1 + take_profit/100)
            take_profit_exits = pd.Series(False, index=price_data.index)
            if take_profit:
                take_profit_level = entry_prices * (1 + take_profit / 100)
                take_profit_exits = (close_prices > take_profit_level) & (entry_prices.notna())
            
            # Combine original exits with stop loss and take profit
            exits = exits | stop_loss_exits | take_profit_exits
        
        # Calculate position size based on sizing method (same logic as MA crossover)
        if position_sizing == "percentage":
            # Percentage of portfolio per position
            # Convert percentage to decimal (10% -> 0.1)
            if position_size > 1.0:
                # Frontend likely sent percentage (10 for 10%), convert to decimal
                size = position_size / 100.0
            else:
                # Already a decimal (0.1 for 10%)
                size = position_size
            # Adjust for max positions
            if max_positions > 1:
                size = size / max_positions  # Divide available capital among max positions
        elif position_sizing == "fixed_amount":
            # Fixed dollar amount - convert to fraction of portfolio
            size = position_size / initial_cash
            # Adjust for max positions  
            if max_positions > 1:
                size = min(size, 1.0 / max_positions)  # Don't exceed available capital
        else:
            # Default to percentage
            if position_size > 1.0:
                size = position_size / 100.0  # Convert percentage to decimal
            else:
                size = position_size
            if max_positions > 1:
                size = size / max_positions
        
        logger.info(f"RSI Strategy - Position sizing: {position_sizing}, size: {position_size}, calculated size: {size}, max_positions: {max_positions}")
        
        # DEBUG: Log price data statistics
        logger.info(f"RSI Strategy - Price data shape: {price_data.shape}")
        logger.info(f"RSI Strategy - Close price stats - min: {price_data['close'].min():.2f}, max: {price_data['close'].max():.2f}, mean: {price_data['close'].mean():.2f}")
        logger.info(f"RSI Strategy - Sample close prices: {price_data['close'].head(3).tolist()}")
        
        # Run backtest with risk management parameters
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission,
            slippage=slippage if slippage else 0.0,
            size=size,
            size_type='targetpercent'
        )
        
        # Get enhanced statistics
        enhanced_stats = calculate_enhanced_stats(portfolio)
        
        # Get trade details with better error handling
        trades = portfolio.trades.records_readable
        trade_list = []
        
        if not trades.empty:
            for _, trade in trades.iterrows():
                try:
                    # Try different possible column names
                    entry_price = None
                    exit_price = None
                    pnl = None
                    pnl_pct = None
                    
                    # Handle different possible column names for VectorBT
                    for col in trades.columns:
                        if 'avg entry price' in col.lower():
                            entry_price = trade[col]
                        elif 'avg exit price' in col.lower():
                            exit_price = trade[col]
                        elif col.lower() in ['pnl', 'p&l']:
                            pnl = trade[col]
                        elif col.lower() == 'return':
                            pnl_pct = trade[col]
                    
                    # Fallback to direct access with correct field names
                    if entry_price is None:
                        entry_price = trade.get('Avg Entry Price', trade.get('Entry Price', trade.get('entry_price', 0)))
                    if exit_price is None:
                        exit_price = trade.get('Avg Exit Price', trade.get('Exit Price', trade.get('exit_price', None)))
                    if pnl is None:
                        pnl = trade.get('PnL', trade.get('pnl', 0))
                    if pnl_pct is None:
                        pnl_pct = trade.get('Return', trade.get('Return [%]', trade.get('return_pct', 0)))
                    
                    # VectorBT returns the actual trade return as decimal (0.007 = 0.7%)
                    # Convert to percentage for frontend display
                    if pnl_pct is not None:
                        pnl_pct = pnl_pct * 100.0
                    
                    # Calculate size properly - VectorBT Size is the actual crypto amount
                    trade_size = trade.get('Size', trade.get('size', 0))
                    # VectorBT Size field already represents the correct crypto units
                    actual_size = trade_size
                    
                    entry_timestamp = trade.get('Entry Timestamp', trade.get('entry_time', datetime.now()))
                    exit_timestamp = trade.get('Exit Timestamp', trade.get('exit_time', None))
                    
                    entry_rsi = rsi.loc[entry_timestamp] if entry_timestamp in rsi.index else None
                    exit_rsi = rsi.loc[exit_timestamp] if pd.notna(exit_timestamp) and exit_timestamp in rsi.index else None
                    
                    trade_info = TradeInfo(
                        entry_time=entry_timestamp,
                        exit_time=exit_timestamp if pd.notna(exit_timestamp) else None,
                        entry_price=entry_price,
                        exit_price=exit_price if pd.notna(exit_price) else None,
                        size=actual_size,
                        side='long',
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        duration_minutes=int((exit_timestamp - entry_timestamp).total_seconds() / 60) if pd.notna(exit_timestamp) else None,
                        entry_reason=f"RSI({rsi_period}) < {oversold_level} (RSI: {entry_rsi:.2f})" if entry_rsi else f"RSI({rsi_period}) < {oversold_level}",
                        exit_reason=f"RSI({rsi_period}) > {overbought_level} (RSI: {exit_rsi:.2f})" if exit_rsi else f"RSI({rsi_period}) > {overbought_level}"
                    )
                    trade_list.append(trade_info)
                except Exception as e:
                    logger.error(f"Error processing RSI trade: {e}")
                    continue
        
        return BacktestStats(
            **enhanced_stats,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            initial_cash=initial_cash,
            final_value=portfolio.value().iloc[-1],
            trades=trade_list,
            # Include indicator data for frontend charting
            indicators={
                "rsi": rsi.dropna().tolist()
            },
            timestamps=[ts.isoformat() for ts in rsi.dropna().index.tolist()]
        )
        
    except Exception as e:
        logging.error(f"Error in RSI strategy: {e}")
        return _create_empty_stats(initial_cash)

def run_bollinger_bands_strategy(
    price_data: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    position_sizing: str = "fixed_amount",
    position_size: float = 10000.0
) -> BacktestStats:
    """
    Bollinger Bands Mean Reversion Strategy
    """
    try:
        # Calculate Bollinger Bands using IndicatorEngine
        bb_upper, bb_middle, bb_lower = indicator_engine.bollinger_bands(
            price_data['close'], window=bb_period, std_dev=bb_std
        )
        
        # Generate signals - Buy when price touches lower band, sell when it reaches middle
        entries = price_data['close'] <= bb_lower
        exits = price_data['close'] >= bb_middle
        
        # Calculate position sizing
        size = None
        if position_sizing == "fixed_amount":
            # Fixed dollar amount per trade
            size = position_size / price_data['close']
        elif position_sizing == "percent_equity":
            # Percentage of current equity
            size = (position_size / 100.0) * initial_cash / price_data['close']
        # Default (None) uses all available cash
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data['close'], 
            entries, 
            exits, 
            init_cash=initial_cash,
            fees=commission,
            size=size
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
        return _create_empty_stats(initial_cash)

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
        return _create_empty_stats(initial_cash)

def _create_empty_stats(initial_cash: float) -> BacktestStats:
    """Create empty backtest stats for error cases"""
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
