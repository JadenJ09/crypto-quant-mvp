# ==============================================================================
# File: services/vectorbt/app/backtesting/strategy_executor.py
# Description: Strategy execution engine that handles backtesting
# ==============================================================================

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

from .strategy_base import BaseStrategy, StrategySignals
from .concrete_strategies import StrategyFactory
from .strategy_builder import create_custom_strategy_from_frontend
from .utils import calculate_enhanced_stats, sanitize_float
from ..models import BacktestStats, TradeInfo

logger = logging.getLogger(__name__)

class StrategyExecutor:
    """
    Main execution engine for running strategy backtests.
    
    This class handles:
    - Strategy initialization
    - Data preprocessing
    - Signal generation
    - Portfolio simulation
    - Results processing
    """
    
    def __init__(self):
        self.strategy: Optional[BaseStrategy] = None
        self.data: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, pd.Series] = {}
        self.signals: Optional[StrategySignals] = None
        self.portfolio = None
        self.results: Optional[BacktestStats] = None
    
    def run_backtest(self, strategy_data: Dict[str, Any], price_data: pd.DataFrame) -> BacktestStats:
        """
        Run a complete backtest for the given strategy and data.
        
        Args:
            strategy_data: Strategy configuration from frontend
            price_data: OHLCV price data
            
        Returns:
            BacktestStats with complete backtest results
        """
        try:
            # Step 1: Initialize strategy
            strategy = self._initialize_strategy(strategy_data)
            
            # Step 2: Validate and preprocess data
            data = strategy.preprocess_data(price_data)
            
            # DEBUG: Log price data at strategy executor level
            logger.info(f"Strategy Executor - Input price_data shape: {price_data.shape}")
            logger.info(f"Strategy Executor - Input close price stats - min: {price_data['close'].min():.2f}, max: {price_data['close'].max():.2f}, mean: {price_data['close'].mean():.2f}")
            logger.info(f"Strategy Executor - Preprocessed data shape: {data.shape}")
            logger.info(f"Strategy Executor - Preprocessed close price stats - min: {data['close'].min():.2f}, max: {data['close'].max():.2f}, mean: {data['close'].mean():.2f}")
            
            # Step 3: Validate strategy parameters
            if not strategy.validate_parameters():
                raise ValueError("Strategy parameter validation failed")
            
            # Step 4: Calculate indicators
            logger.info("Calculating indicators...")
            indicators = strategy.calculate_indicators(data)
            
            # Step 5: Generate signals
            logger.info("Generating signals...")
            signals = strategy.generate_signals(data, indicators)
            
            # Step 6: Apply position direction filter (CRITICAL FIX)
            logger.info(f"Strategy config position_direction: {strategy.config.position_direction}")
            logger.info(f"Position direction value: {strategy.config.position_direction.value}")
            logger.info(f"Applying position direction filter: {strategy.config.position_direction.value}")
            signals = strategy.apply_position_direction_filter(signals)
            logger.info(f"After position direction filter - Entries: {signals.entries.sum()}, Exits: {signals.exits.sum()}")
            
            # DEBUG: Log signal details for troubleshooting
            if signals.entries.sum() > 0:
                first_entry = signals.entries.idxmax()
                logger.info(f"First entry signal at: {first_entry}")
            if signals.exits.sum() > 0:
                first_exit = signals.exits.idxmax()  
                logger.info(f"First exit signal at: {first_exit}")
            
            # Step 7: Apply max position constraints (CRITICAL FIX)
            logger.info("Applying max position constraints...")
            max_positions = getattr(strategy, 'max_positions', strategy.config.risk_management.max_positions)
            if max_positions and max_positions > 0:
                signals.entries, signals.exits = strategy.apply_max_position_constraint(signals.entries, signals.exits)
                logger.info(f"Max positions constraint applied: {max_positions}")
            
            # Step 8: Apply risk management (stop loss/take profit)
            logger.info("Applying risk management...")
            signals = strategy.apply_risk_management(signals, data)
            
            # Step 9: Calculate position sizing
            logger.info("Calculating position sizing...")
            position_sizes = strategy.calculate_position_size(data)
            
            # Step 10: Run portfolio simulation
            logger.info("Running portfolio simulation...")
            portfolio = self._run_portfolio_simulation(strategy, data, signals, position_sizes)
            
            # Step 11: Calculate enhanced statistics
            logger.info("Calculating statistics...")
            enhanced_stats = calculate_enhanced_stats(portfolio)
            
            # Step 12: Extract trade details
            logger.info("Extracting trade details...")
            trades = self._extract_trade_details(portfolio, data, indicators, strategy)
            
            # Step 13: Create results with indicator data
            indicator_data = {}
            timestamps = []
            
            # Convert indicators to frontend format
            if indicators:
                timestamps = [ts.isoformat() for ts in data.index]
                for name, values in indicators.items():
                    if hasattr(values, 'dropna'):
                        indicator_data[name] = values.dropna().tolist()
                    else:
                        indicator_data[name] = values if isinstance(values, list) else [values]
            
            results = BacktestStats(
                **enhanced_stats,
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_cash=strategy.config.initial_cash,
                final_value=portfolio.value().iloc[-1],
                trades=trades,
                indicators=indicator_data if indicator_data else None,
                timestamps=timestamps if timestamps else None
            )
            
            # Store for later access
            self.strategy = strategy
            self.data = data
            self.indicators = indicators
            self.signals = signals
            self.portfolio = portfolio
            self.results = results
            
            logger.info(f"Backtest completed successfully. Total trades: {len(trades)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest execution: {e}")
            return self._create_empty_stats(strategy_data.get('initial_cash', 100000.0))
    
    def _initialize_strategy(self, strategy_data: Dict[str, Any]) -> BaseStrategy:
        """Initialize strategy based on strategy data"""
        
        # Check if this is a custom strategy from the frontend
        if 'entry_conditions' in strategy_data and 'exit_conditions' in strategy_data:
            logger.info("Creating custom strategy from frontend conditions")
            strategy = create_custom_strategy_from_frontend(strategy_data)
            # Store max_positions as an attribute for later access
            setattr(strategy, 'max_positions', strategy_data.get('max_positions', 1))
            return strategy
        
        # Check if this is a predefined strategy type
        elif 'type' in strategy_data:
            strategy_type = strategy_data['type']
            params = strategy_data.get('parameters', {})
            logger.info(f"Creating predefined strategy: {strategy_type}")
            strategy = StrategyFactory.create_strategy(strategy_type, **params)
            setattr(strategy, 'max_positions', strategy_data.get('max_positions', 1))
            return strategy
        
        # Legacy support: try to infer strategy type from entry conditions
        elif 'entry_conditions' in strategy_data:
            logger.info("Inferring strategy type from conditions")
            strategy = self._infer_strategy_from_conditions(strategy_data)
            setattr(strategy, 'max_positions', strategy_data.get('max_positions', 1))
            return strategy
        
        else:
            # Default to MA crossover
            logger.warning("No strategy type specified, defaulting to MA crossover")
            strategy = StrategyFactory.create_strategy("ma_crossover")
            setattr(strategy, 'max_positions', strategy_data.get('max_positions', 1))
            return strategy
    
    def _infer_strategy_from_conditions(self, strategy_data: Dict[str, Any]) -> BaseStrategy:
        """Infer strategy type from entry conditions (legacy support)"""
        entry_conditions = strategy_data.get('entry_conditions', [])
        
        if not entry_conditions:
            return StrategyFactory.create_strategy("ma_crossover")
        
        # Look at first enabled condition
        for condition in entry_conditions:
            if condition.get('enabled', True):
                indicator = condition.get('indicator', '').lower()
                
                if 'rsi' in indicator:
                    return StrategyFactory.create_strategy("rsi_oversold")
                elif 'macd' in indicator:
                    return StrategyFactory.create_strategy("macd")
                elif 'bb' in indicator or 'bollinger' in indicator:
                    return StrategyFactory.create_strategy("bollinger_bands")
                elif 'sma' in indicator or 'ema' in indicator or 'ma' in indicator:
                    return StrategyFactory.create_strategy("ma_crossover")
        
        # Default fallback
        return StrategyFactory.create_strategy("ma_crossover")
    
    def _run_portfolio_simulation(self, strategy: BaseStrategy, data: pd.DataFrame, 
                                 signals: StrategySignals, position_sizes: pd.Series) -> Any:
        """Run the portfolio simulation using vectorbt"""
        
        try:
            # Always use the calculated position_sizes, not the config default
            size_type = 'percent'  # Default to percentage
            size = position_sizes  # Use the calculated position sizes
            
            # Log for debugging Kelly Criterion issues
            logger.info(f"Position sizing method: {strategy.config.position_sizing.method.value}")
            logger.info(f"Position size from config: {strategy.config.position_sizing.size}")
            logger.info(f"Calculated position sizes - mean: {position_sizes.mean():.4f}, std: {position_sizes.std():.4f}, min: {position_sizes.min():.4f}, max: {position_sizes.max():.4f}")
            
            # CRITICAL FIX: VectorBT has limitations with percentage sizing and direction changes
            # Use amount sizing for short/both directions to avoid VectorBT errors
            method_value = strategy.config.position_sizing.method.value
            
            if method_value == 'fixed_amount' or method_value == 'fixed':
                size_type = 'amount'
                size = position_sizes
                logger.info(f"Using FIXED AMOUNT position sizing: {size.mean():.2f} USD")
            elif strategy.config.position_direction.value in ['short_only', 'both']:
                # CRITICAL FIX: For short/both directions, convert percentage to amount
                # to avoid VectorBT "SizeType.Percent does not support position reversal" error
                size_type = 'amount'
                # Convert percentage to dollar amount, then to shares
                initial_cash = strategy.config.initial_cash
                percentage = position_sizes.iloc[0]  # Get the percentage (e.g., 0.2 for 20%)
                # Calculate dollar amount: percentage * initial_cash
                dollar_amount = percentage * initial_cash
                # Convert dollar amount to number of shares based on close price
                # shares = dollar_amount / price_per_share
                shares_per_bar = dollar_amount / data['close']
                size = shares_per_bar
                logger.info(f"Using AMOUNT position sizing for {strategy.config.position_direction.value}: ${dollar_amount:.2f} -> {shares_per_bar.mean():.4f} shares/units (converted from {percentage:.1%})")
            elif method_value == 'kelly_criterion':
                # Kelly Criterion uses percentage of portfolio (0 to 1)
                size_type = 'percent'
                size = position_sizes
                logger.info(f"Using KELLY CRITERION position sizing: {size.mean():.4f} (fraction of portfolio)")
            else:
                # For all other methods (percentage, volatility, etc.)
                size_type = 'percent'
                size = position_sizes
                logger.info(f"Using PERCENTAGE position sizing: {size.mean():.4f} (fraction of portfolio)")
            
            # Prepare risk management parameters
            portfolio_kwargs = {
                'close': data['close'],
                'entries': signals.entries,
                'exits': signals.exits,
                'init_cash': strategy.config.initial_cash,
                'fees': strategy.config.execution.commission,
                'slippage': strategy.config.execution.slippage,
                'size': size,
                'size_type': size_type,
                'freq': '1min'  # Default frequency
            }
            
            # Add OHLC data for stop loss/take profit calculations
            if 'high' in data.columns and 'low' in data.columns:
                portfolio_kwargs['high'] = data['high']
                portfolio_kwargs['low'] = data['low']
                portfolio_kwargs['open'] = data['open']
            
            # Apply risk management BEFORE portfolio simulation for better control
            if (strategy.config.risk_management.stop_loss is not None or 
                strategy.config.risk_management.take_profit is not None):
                logger.info("Applying custom stop loss/take profit signals for better precision...")
                signals = self._apply_custom_stop_loss_take_profit(signals, data, strategy)
            
            # Add basic VectorBT stop loss as backup (but custom signals take precedence)
            if strategy.config.risk_management.stop_loss is not None:
                stop_loss_pct = strategy.config.risk_management.stop_loss / 100.0  # Convert percentage to decimal
                portfolio_kwargs['sl_stop'] = stop_loss_pct
                logger.info(f"Adding VectorBT backup stop loss: {stop_loss_pct:.4f} ({strategy.config.risk_management.stop_loss}%)")
            
            # Add basic VectorBT take profit as backup
            if strategy.config.risk_management.take_profit is not None:
                take_profit_pct = strategy.config.risk_management.take_profit / 100.0  # Convert percentage to decimal
                portfolio_kwargs['tp_stop'] = take_profit_pct
                logger.info(f"Adding VectorBT backup take profit: {take_profit_pct:.4f} ({strategy.config.risk_management.take_profit}%)")
            
            # CRITICAL FIX: Handle short positions properly in VectorBT
            if strategy.config.position_direction.value == 'short_only':
                # For short-only strategies, we need to tell VectorBT to go short
                portfolio_kwargs['direction'] = 'shortonly'  # VectorBT parameter for short positions
                logger.info("Setting VectorBT direction to 'shortonly' for short-only strategy")
            elif strategy.config.position_direction.value == 'both':
                # For both directions, allow VectorBT to go both ways
                portfolio_kwargs['direction'] = 'both'  # VectorBT parameter for both directions
                logger.info("Setting VectorBT direction to 'both' for both-direction strategy")
            else:
                # Default to long-only
                portfolio_kwargs['direction'] = 'longonly'  # VectorBT parameter for long positions
                logger.info("Setting VectorBT direction to 'longonly' for long-only strategy")
            
            # Run portfolio simulation with risk management
            portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)
            
            # DEBUG: Log portfolio simulation results
            logger.info(f"Portfolio simulation - data['close'] stats: min: {data['close'].min():.2f}, max: {data['close'].max():.2f}, mean: {data['close'].mean():.2f}")
            logger.info(f"Portfolio simulation - Portfolio value: {portfolio.value().iloc[-1]:.2f}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error in portfolio simulation: {e}")
            logger.error(f"Portfolio kwargs: {list(portfolio_kwargs.keys())}")
            logger.error(f"Entries shape: {signals.entries.shape}, sum: {signals.entries.sum()}")
            logger.error(f"Exits shape: {signals.exits.shape}, sum: {signals.exits.sum()}")
            logger.error(f"Size shape: {size.shape if hasattr(size, 'shape') else 'scalar'}")
            logger.error(f"Falling back to basic simulation without risk management")
            # Fallback to simple simulation
            return vbt.Portfolio.from_signals(
                data['close'],
                signals.entries,
                signals.exits,
                init_cash=strategy.config.initial_cash,
                fees=strategy.config.execution.commission
            )
    
    def _extract_trade_details(self, portfolio, data: pd.DataFrame, 
                              indicators: Dict[str, pd.Series], 
                              strategy: BaseStrategy) -> List[TradeInfo]:
        """Extract detailed trade information"""
        trade_list = []
        
        try:
            trades = portfolio.trades.records_readable
            
            if trades.empty:
                return trade_list
            
            logger.info(f"Processing {len(trades)} trades")
            
            # DEBUG: Log the trade columns and a sample trade
            logger.info(f"Trade columns: {list(trades.columns)}")
            if not trades.empty:
                logger.info(f"First trade raw data: {dict(trades.iloc[0])}")
            
            for _, trade in trades.iterrows():
                try:
                    # Extract trade information with multiple column name attempts
                    entry_time = self._safe_get_trade_field(trade, ['Entry Timestamp', 'entry_time', 'Entry Time'])
                    exit_time = self._safe_get_trade_field(trade, ['Exit Timestamp', 'exit_time', 'Exit Time'])
                    entry_price = self._safe_get_trade_field(trade, ['Avg Entry Price', 'Entry Price', 'entry_price'])
                    exit_price = self._safe_get_trade_field(trade, ['Avg Exit Price', 'Exit Price', 'exit_price'])
                    size = self._safe_get_trade_field(trade, ['Size', 'size', 'Amount'])
                    pnl = self._safe_get_trade_field(trade, ['PnL', 'pnl', 'P&L', 'Profit'])
                    pnl_pct = self._safe_get_trade_field(trade, ['Return', 'Return [%]', 'return_pct', 'Return%'])
                    direction = self._safe_get_trade_field(trade, ['Direction', 'direction', 'Side'])
                    
                    # Map VectorBT direction to frontend format
                    if direction and str(direction).lower() == 'short':
                        side = 'short'
                    else:
                        side = 'long'  # Default to long if direction is unclear
                    
                    # Calculate duration
                    duration_minutes = None
                    if entry_time and exit_time and pd.notna(exit_time):
                        try:
                            duration = pd.to_datetime(exit_time) - pd.to_datetime(entry_time)
                            duration_minutes = int(duration.total_seconds() / 60)
                        except:
                            duration_minutes = None
                    
                    # Generate entry/exit reasons based on strategy type
                    entry_reason, exit_reason = self._generate_trade_reasons(
                        strategy, entry_time, exit_time, indicators
                    )
                    
                    trade_info = TradeInfo(
                        entry_time=entry_time if entry_time else datetime.now(),
                        exit_time=exit_time if pd.notna(exit_time) else None,
                        entry_price=float(entry_price) if entry_price else 0.0,
                        exit_price=float(exit_price) if pd.notna(exit_price) else None,
                        size=float(size) if size else 0.0,
                        side=side,  # Use extracted direction from VectorBT
                        pnl=float(pnl) if pnl else 0.0,
                        pnl_pct=float(pnl_pct) * 100.0 if pnl_pct else 0.0,  # Convert decimal to percentage
                        duration_minutes=duration_minutes,
                        entry_reason=entry_reason,
                        exit_reason=exit_reason
                    )
                    
                    trade_list.append(trade_info)
                    
                except Exception as e:
                    logger.error(f"Error processing individual trade: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error extracting trade details: {e}")
        
        return trade_list
    
    def _safe_get_trade_field(self, trade, possible_field_names: List[str]):
        """Safely get a field from trade record with multiple possible names"""
        for field_name in possible_field_names:
            if field_name in trade.index:
                return trade[field_name]
        
        # If no exact match, try case-insensitive search
        trade_index_lower = [col.lower() for col in trade.index]
        for field_name in possible_field_names:
            field_lower = field_name.lower()
            for i, col_lower in enumerate(trade_index_lower):
                if field_lower in col_lower or col_lower in field_lower:
                    return trade.iloc[i]
        
        return None
    
    def _generate_trade_reasons(self, strategy: BaseStrategy, entry_time, exit_time, 
                               indicators: Dict[str, pd.Series]) -> tuple:
        """Generate human-readable entry and exit reasons"""
        strategy_name = strategy.config.name
        
        # Default reasons
        entry_reason = f"{strategy_name} entry signal"
        exit_reason = f"{strategy_name} exit signal" if exit_time else None
        
        try:
            # Customize based on strategy type
            if "MA Crossover" in strategy_name:
                fast_window = strategy.config.parameters.get('fast_window', 10)
                slow_window = strategy.config.parameters.get('slow_window', 20)
                entry_reason = f"Fast MA({fast_window}) crossed above Slow MA({slow_window})"
                if exit_time:
                    exit_reason = f"Fast MA({fast_window}) crossed below Slow MA({slow_window})"
            
            elif "RSI" in strategy_name:
                rsi_period = strategy.config.parameters.get('rsi_period', 14)
                oversold = strategy.config.parameters.get('oversold_level', 30)
                overbought = strategy.config.parameters.get('overbought_level', 70)
                
                # Try to get actual RSI value at entry/exit
                if 'rsi' in indicators and entry_time in indicators['rsi'].index:
                    entry_rsi = indicators['rsi'].loc[entry_time]
                    entry_reason = f"RSI({rsi_period}) < {oversold} (RSI: {entry_rsi:.1f})"
                else:
                    entry_reason = f"RSI({rsi_period}) < {oversold}"
                
                if exit_time and 'rsi' in indicators and exit_time in indicators['rsi'].index:
                    exit_rsi = indicators['rsi'].loc[exit_time]
                    exit_reason = f"RSI({rsi_period}) > {overbought} (RSI: {exit_rsi:.1f})"
                elif exit_time:
                    exit_reason = f"RSI({rsi_period}) > {overbought}"
            
            elif "MACD" in strategy_name:
                entry_reason = "MACD crossed above signal line"
                if exit_time:
                    exit_reason = "MACD crossed below signal line"
            
            elif "Bollinger" in strategy_name:
                entry_reason = "Price touched lower Bollinger Band"
                if exit_time:
                    exit_reason = "Price reached middle Bollinger Band"
                    
        except Exception as e:
            logger.error(f"Error generating trade reasons: {e}")
        
        return entry_reason, exit_reason
    
    def _create_empty_stats(self, initial_cash: float) -> BacktestStats:
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
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current strategy"""
        if self.strategy:
            return self.strategy.get_strategy_info()
        return {}
    
    def get_indicators(self) -> Dict[str, pd.Series]:
        """Get calculated indicators"""
        return self.indicators
    
    def get_signals(self) -> Optional[StrategySignals]:
        """Get generated signals"""
        return self.signals
    
    def _apply_max_positions_constraint(self, signals: StrategySignals, max_positions: int) -> StrategySignals:
        """Apply max positions constraint by filtering entry signals"""
        if max_positions <= 0:
            return signals
        
        # Create modified entry signals that respect max positions
        entries = signals.entries.copy()
        exits = signals.exits.copy()
        
        # Track open positions
        open_positions = 0
        filtered_entries = pd.Series(False, index=entries.index)
        
        for timestamp in entries.index:
            # Check if we have an exit signal at this timestamp
            if exits.loc[timestamp]:
                open_positions = max(0, open_positions - 1)
            
            # Check if we have an entry signal and room for more positions
            if entries.loc[timestamp] and open_positions < max_positions:
                filtered_entries.loc[timestamp] = True
                open_positions += 1
        
        # Return new signals with filtered entries
        return StrategySignals(
            entries=filtered_entries,
            exits=exits,
            stop_losses=signals.stop_losses,
            take_profits=signals.take_profits
        )
    
    def _apply_custom_stop_loss_take_profit(self, signals: StrategySignals, data: pd.DataFrame, strategy: BaseStrategy) -> StrategySignals:
        """
        Apply custom stop loss and take profit logic for precise control, especially for short positions.
        This method ensures stop losses trigger at exactly the specified percentage.
        """
        stop_loss_pct = strategy.config.risk_management.stop_loss
        take_profit_pct = strategy.config.risk_management.take_profit
        direction = strategy.config.position_direction.value
        
        if stop_loss_pct is None and take_profit_pct is None:
            return signals
        
        logger.info(f"Applying custom stop loss/take profit for {direction} positions")
        
        # Convert to decimal
        stop_loss_decimal = (stop_loss_pct / 100.0) if stop_loss_pct else None
        take_profit_decimal = (take_profit_pct / 100.0) if take_profit_pct else None
        
        # Create new exit signals that combine original exits with stop loss/take profit
        enhanced_exits = signals.exits.copy()
        
        # Track entry points and monitor for stop loss/take profit
        current_position = None
        current_entry_price = None
        current_entry_time = None
        
        for timestamp in data.index:
            # Check for new entries
            if signals.entries.loc[timestamp]:
                if current_position is None:  # No position open
                    current_position = direction if direction != 'both' else 'long'  # Default to long for 'both'
                    current_entry_price = data.loc[timestamp, 'close']
                    current_entry_time = timestamp
                    logger.debug(f"Position opened at {timestamp}: {current_position} @ ${current_entry_price:.2f}")
            
            # Check for exits (original strategy exits)
            if signals.exits.loc[timestamp] and current_position is not None:
                current_position = None
                current_entry_price = None
                current_entry_time = None
                logger.debug(f"Position closed by strategy signal at {timestamp}")
                continue
            
            # Check for stop loss/take profit if we have an open position
            if current_position is not None and current_entry_price is not None:
                try:
                    current_price = pd.to_numeric(data.at[timestamp, 'close'], errors='coerce')
                    entry_price = pd.to_numeric(current_entry_price, errors='coerce')
                    
                    if pd.isna(current_price) or pd.isna(entry_price):
                        continue
                    
                    if current_position == 'short' or (current_position == 'both' and direction == 'short_only'):
                        # SHORT POSITION LOGIC
                        price_change_pct = (current_price - entry_price) / entry_price
                        pnl_pct = -price_change_pct  # Short profit = opposite of price change
                        
                        # Stop loss for short: when price goes UP too much (pnl becomes too negative)
                        if stop_loss_decimal and pnl_pct <= -stop_loss_decimal:
                            enhanced_exits.at[timestamp] = True
                            logger.info(f"SHORT STOP LOSS triggered at {timestamp}: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Loss={pnl_pct:.2%}")
                            current_position = None
                            current_entry_price = None
                            current_entry_time = None
                            continue
                        
                        # Take profit for short: when price goes DOWN enough (pnl becomes positive enough)
                        if take_profit_decimal and pnl_pct >= take_profit_decimal:
                            enhanced_exits.at[timestamp] = True
                            logger.info(f"SHORT TAKE PROFIT triggered at {timestamp}: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Profit={pnl_pct:.2%}")
                            current_position = None
                            current_entry_price = None
                            current_entry_time = None
                            continue
                            
                    else:
                        # LONG POSITION LOGIC
                        price_change_pct = (current_price - entry_price) / entry_price
                        pnl_pct = price_change_pct  # Long profit = same as price change
                        
                        # Stop loss for long: when price goes DOWN too much
                        if stop_loss_decimal and pnl_pct <= -stop_loss_decimal:
                            enhanced_exits.at[timestamp] = True
                            logger.info(f"LONG STOP LOSS triggered at {timestamp}: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Loss={pnl_pct:.2%}")
                            current_position = None
                            current_entry_price = None
                            current_entry_time = None
                            continue
                        
                        # Take profit for long: when price goes UP enough
                        if take_profit_decimal and pnl_pct >= take_profit_decimal:
                            enhanced_exits.at[timestamp] = True
                            logger.info(f"LONG TAKE PROFIT triggered at {timestamp}: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Profit={pnl_pct:.2%}")
                            current_position = None
                            current_entry_price = None
                            current_entry_time = None
                            continue
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error processing stop loss/take profit at {timestamp}: {e}")
                    continue
        
        # Create enhanced signals with the new exits
        enhanced_signals = StrategySignals(
            entries=signals.entries,
            exits=enhanced_exits,
            stop_losses=signals.stop_losses,
            take_profits=signals.take_profits,
            confidence=signals.confidence,
            metadata={**signals.metadata, "custom_risk_management": True}
        )
        
        original_exits = signals.exits.sum()
        enhanced_exits_count = enhanced_exits.sum()
        logger.info(f"Custom risk management: {original_exits} original exits -> {enhanced_exits_count} enhanced exits")
        
        return enhanced_signals

# Create a global instance for API use
strategy_executor = StrategyExecutor()
