"""
Backtest Executor - Main backtesting coordination and execution

This module coordinates the entire backtesting process including:
- Data processing and validation
- Signal generation and execution
- Risk management application
- Performance tracking and reporting
"""

from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .portfolio_manager import PortfolioManager
from .trade_engine import Trade


class BacktestExecutor:
    """Main backtesting execution engine"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0001,
                 max_positions: int = 10):
        """
        Initialize backtest executor
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage: Slippage rate per trade
            max_positions: Maximum concurrent positions
        """
        self.portfolio = PortfolioManager(initial_capital, max_positions)
        self.portfolio.trade_engine.commission_rate = commission_rate
        self.portfolio.trade_engine.slippage = slippage
        
        # Configuration
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.max_positions = max_positions
        
        # Strategy configuration
        self.strategy_func: Optional[Callable] = None
        self.risk_params = {
            'max_position_size': 0.1,  # 10% of capital per position
            'stop_loss_pct': 0.05,     # 5% stop loss
            'take_profit_pct': None,   # No take profit by default
            'risk_per_trade': 0.02,    # 2% risk per trade
            'max_drawdown_limit': 0.20 # 20% max drawdown limit
        }
        
        # Tracking
        self.logger = logging.getLogger(__name__)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def set_strategy(self, strategy_func: Callable):
        """
        Set the strategy function
        
        Args:
            strategy_func: Function that takes (data, timestamp) and returns signals
        """
        self.strategy_func = strategy_func
    
    def set_risk_parameters(self, **params):
        """
        Update risk management parameters
        
        Args:
            **params: Risk parameters to update
        """
        self.risk_params.update(params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and quality
        
        Args:
            data: Input OHLCV data
            
        Returns:
            bool: True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Need: {required_columns}")
            return False
        
        # Check for null values
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains null values")
        
        # Check OHLC relationships
        invalid_bars = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_bars.any():
            self.logger.warning(f"Found {invalid_bars.sum()} invalid OHLC bars")
        
        return True
    
    def run_backtest(self, data: pd.DataFrame, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete backtest
        
        Args:
            data: OHLCV data with MultiIndex (timestamp, symbol) or single symbol
            symbols: List of symbols to trade (auto-detected if None)
            
        Returns:
            Dictionary with backtest results
        """
        if self.strategy_func is None:
            raise ValueError("Strategy function not set. Use set_strategy() first.")
        
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Auto-detect symbols if not provided
        if symbols is None:
            if isinstance(data.index, pd.MultiIndex):
                symbols = data.index.get_level_values(1).unique().tolist()
            else:
                symbols = ['default']  # Single symbol data
        
        self.logger.info(f"Starting backtest with {len(symbols)} symbols")
        self.logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        
        # Track execution time
        self.start_time = datetime.now()
        
        # Main execution loop
        if isinstance(data.index, pd.MultiIndex):
            self._run_multi_symbol_backtest(data, symbols)
        else:
            self._run_single_symbol_backtest(data, symbols[0])
        
        self.end_time = datetime.now()
        
        # Generate final results
        results = self._generate_results()
        
        self.logger.info(f"Backtest completed in {self.end_time - self.start_time}")
        self.logger.info(f"Total return: {results['metrics']['total_return_pct']:.2f}%")
        self.logger.info(f"Number of trades: {results['metrics']['num_trades']}")
        
        return results
    
    def _run_single_symbol_backtest(self, data: pd.DataFrame, symbol: str):
        """Run backtest for single symbol"""
        for i in range(len(data)):
            timestamp = data.index[i]
            
            # Ensure timestamp is pd.Timestamp
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)
            
            # Create bar data using iloc for safe access
            bar_data = {
                'open': float(data.iloc[i]['open']),
                'high': float(data.iloc[i]['high']), 
                'low': float(data.iloc[i]['low']),
                'close': float(data.iloc[i]['close']),
                'volume': float(data.iloc[i]['volume'])
            }
            
            # Process existing positions (stop loss/take profit)
            trades = self.portfolio.process_bar(symbol, bar_data, timestamp)
            
            # Check drawdown limit
            if self.portfolio.current_drawdown >= self.risk_params['max_drawdown_limit']:
                self.logger.warning(f"Max drawdown limit reached: {self.portfolio.current_drawdown:.2%}")
                continue
            
            # Generate strategy signals
            strategy_data = data.iloc[:i+1]  # Data up to current timestamp
            if self.strategy_func:
                signals = self.strategy_func(strategy_data, timestamp)
            else:
                signals = {}
            
            # Process signals
            self._process_signals(signals, symbol, bar_data['close'], timestamp)
            
            # Update portfolio
            market_prices = {symbol: bar_data['close']}
            self.portfolio.update_capital(timestamp, market_prices)
    
    def _run_multi_symbol_backtest(self, data: pd.DataFrame, symbols: List[str]):
        """Run backtest for multiple symbols"""
        # Get unique timestamps
        timestamps = data.index.get_level_values(0).unique().sort_values()
        
        for timestamp in timestamps:
            # Ensure timestamp is pd.Timestamp
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)
            
            # Get current bar data for all symbols
            current_bars = {}
            market_prices = {}
            
            for symbol in symbols:
                try:
                    # Access data directly with column names
                    open_price = data.loc[(timestamp, symbol), 'open']
                    high_price = data.loc[(timestamp, symbol), 'high']
                    low_price = data.loc[(timestamp, symbol), 'low']
                    close_price = data.loc[(timestamp, symbol), 'close']
                    volume = data.loc[(timestamp, symbol), 'volume']
                    
                    bar_data = {
                        'open': float(pd.to_numeric(open_price, errors='coerce')),
                        'high': float(pd.to_numeric(high_price, errors='coerce')),
                        'low': float(pd.to_numeric(low_price, errors='coerce')), 
                        'close': float(pd.to_numeric(close_price, errors='coerce')),
                        'volume': float(pd.to_numeric(volume, errors='coerce'))
                    }
                    current_bars[symbol] = bar_data
                    market_prices[symbol] = bar_data['close']
                except (KeyError, IndexError, ValueError):
                    continue  # Symbol not available at this timestamp
            
            # Process existing positions for all symbols
            all_trades = []
            for symbol in current_bars:
                trades = self.portfolio.process_bar(symbol, current_bars[symbol], timestamp)
                all_trades.extend(trades)
            
            # Check drawdown limit
            if self.portfolio.current_drawdown >= self.risk_params['max_drawdown_limit']:
                self.logger.warning(f"Max drawdown limit reached: {self.portfolio.current_drawdown:.2%}")
                continue
            
            # Generate strategy signals for all symbols
            try:
                strategy_data = data.loc[:timestamp]  # Data up to current timestamp
            except:
                # Handle index slicing issues
                ts_index = data.index.get_level_values(0)
                mask = ts_index <= timestamp
                strategy_data = data[mask]
                
            if self.strategy_func:
                signals = self.strategy_func(strategy_data, timestamp)
            else:
                signals = {}
            
            # Process signals for each symbol
            for symbol in symbols:
                if symbol in current_bars and symbol in signals:
                    self._process_signals(
                        {symbol: signals[symbol]}, 
                        symbol, 
                        current_bars[symbol]['close'], 
                        timestamp
                    )
            
            # Update portfolio
            self.portfolio.update_capital(timestamp, market_prices)
    
    def _process_signals(self, signals: Dict[str, Any], symbol: str, 
                        current_price: float, timestamp: pd.Timestamp):
        """
        Process strategy signals and execute trades
        
        Args:
            signals: Strategy signals
            symbol: Trading symbol
            current_price: Current market price
            timestamp: Current timestamp
        """
        if symbol not in signals:
            return
        
        signal = signals[symbol]
        
        # Handle different signal formats
        if isinstance(signal, dict):
            action = signal.get('action', 'hold')
            side = signal.get('side', 'long')
            confidence = signal.get('confidence', 1.0)
        elif isinstance(signal, str):
            action = signal
            side = 'long'
            confidence = 1.0
        elif isinstance(signal, (int, float)):
            if signal > 0:
                action = 'buy'
                side = 'long'
            elif signal < 0:
                action = 'sell'
                side = 'short'
            else:
                action = 'hold'
                side = 'long'
            confidence = abs(signal)
        else:
            return  # Unknown signal format
        
        # Execute based on action
        if action in ['buy', 'long']:
            self._execute_buy_signal(symbol, 'long', current_price, timestamp, confidence)
        elif action in ['sell', 'short']:
            self._execute_buy_signal(symbol, 'short', current_price, timestamp, confidence)
        elif action == 'close':
            self.portfolio.close_position(symbol, current_price, timestamp, "signal")
    
    def _execute_buy_signal(self, symbol: str, side: str, price: float,
                           timestamp: pd.Timestamp, confidence: float = 1.0):
        """Execute buy/sell signal with risk management"""
        
        # Apply confidence to position sizing
        risk_per_trade = self.risk_params['risk_per_trade'] * confidence
        
        # Open position with risk management
        success = self.portfolio.open_position(
            symbol=symbol,
            side=side,
            entry_price=price,
            timestamp=timestamp,
            stop_loss_pct=self.risk_params['stop_loss_pct'],
            take_profit_pct=self.risk_params['take_profit_pct'],
            risk_per_trade=risk_per_trade
        )
        
        if success:
            self.logger.debug(f"Opened {side} position in {symbol} at {price}")
        else:
            self.logger.debug(f"Failed to open {side} position in {symbol}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        
        # Close all remaining positions at the end
        if self.portfolio.timestamps:
            final_timestamp = self.portfolio.timestamps[-1]
            final_prices = {}
            
            # Get final prices for all open positions
            for symbol in self.portfolio.trade_engine.get_all_positions():
                position = self.portfolio.trade_engine.get_position(symbol)
                if position:
                    final_prices[symbol] = position.entry_price
            
            self.portfolio.close_all_positions(final_prices, final_timestamp)
        
        # Generate comprehensive results
        metrics = self.portfolio.get_portfolio_metrics()
        trades_df = self.portfolio.get_trades_dataframe()
        equity_curve = self.portfolio.get_equity_curve()
        drawdown_series = self.portfolio.get_drawdown_series()
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_curve,
            'drawdown_series': drawdown_series,
            'config': {
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage': self.slippage,
                'max_positions': self.max_positions,
                'risk_params': self.risk_params
            },
            'execution_time': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration': self.end_time - self.start_time if self.end_time and self.start_time else None
            }
        }
    
    def run_optimization(self, data: pd.DataFrame, param_grid: Dict[str, List],
                        symbols: Optional[List[str]] = None, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Run parameter optimization
        
        Args:
            data: OHLCV data
            param_grid: Dictionary of parameters to optimize
            symbols: Trading symbols
            metric: Optimization metric
            
        Returns:
            DataFrame with optimization results
        """
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for i, combo in enumerate(combinations):
            self.logger.info(f"Testing combination {i+1}/{len(combinations)}")
            
            # Set parameters
            param_dict = dict(zip(param_names, combo))
            self.set_risk_parameters(**param_dict)
            
            # Run backtest
            try:
                result = self.run_backtest(data, symbols)
                metrics = result['metrics']
                
                # Add parameters to results
                row = param_dict.copy()
                row.update(metrics)
                results.append(row)
                
            except Exception as e:
                self.logger.error(f"Error in combination {i+1}: {e}")
                continue
        
        # Convert to DataFrame and sort by metric
        results_df = pd.DataFrame(results)
        if metric in results_df.columns:
            results_df = results_df.sort_values(metric, ascending=False)
        
        return results_df
