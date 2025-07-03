# ==============================================================================
# File: services/vectorbt/app/backtesting/utils.py
# Description: Utility functions for backtesting calculations
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

def calculate_enhanced_stats(portfolio, trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio statistics
    
    Args:
        portfolio: vectorbt Portfolio object
        trades_df: Optional DataFrame with individual trade details
        
    Returns:
        Dictionary with enhanced statistics
    """
    try:
        stats = portfolio.stats()
        
        # Basic metrics with sanitization
        total_return = sanitize_float(stats.get('Total Return [%]', 0.0))
        max_drawdown = sanitize_float(abs(stats.get('Max Drawdown [%]', 0.0)))  # Make positive
        sharpe = sanitize_float(stats.get('Sharpe Ratio', 0.0))
        
        # Enhanced metrics
        returns = portfolio.returns()
        annual_vol = sanitize_float(returns.std() * np.sqrt(252 * 24 * 60 / 5) * 100) if len(returns.dropna()) > 0 else 0.0
        
        # Risk metrics
        var_95 = sanitize_float(np.percentile(returns.dropna(), 5) * 100) if len(returns.dropna()) > 0 else 0.0
        
        # Sortino ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        downside_vol = sanitize_float(negative_returns.std() * np.sqrt(252 * 24 * 60 / 5)) if len(negative_returns) > 0 else 0.0
        sortino = sanitize_float((returns.mean() * 252 * 24 * 60 / 5) / downside_vol) if downside_vol > 0 else 0.0
        
        # Calmar ratio
        calmar = sanitize_float((total_return / 100) / (max_drawdown / 100)) if max_drawdown > 0 else 0.0
        
        # Trade statistics
        trade_stats = {
            'total_trades': int(stats.get('Total Trades', 0)),
            'win_rate_pct': sanitize_float(stats.get('Win Rate [%]', 0.0)),
            'profit_factor': sanitize_float(stats.get('Profit Factor', 0.0)),
            'avg_win_pct': sanitize_float(stats.get('Avg Winning Trade [%]', 0.0)),
            'avg_loss_pct': sanitize_float(abs(stats.get('Avg Losing Trade [%]', 0.0))),
        }
        
        # Enhanced trade statistics from VectorBT trades data
        enhanced_trade_stats = {}
        if hasattr(portfolio, 'trades') and hasattr(portfolio.trades, 'records_readable'):
            trades_readable = portfolio.trades.records_readable
            if not trades_readable.empty:
                # Position direction counts
                direction_col = None
                for col in ['Direction', 'direction', 'Side', 'side']:
                    if col in trades_readable.columns:
                        direction_col = col
                        break
                
                if direction_col:
                    long_count = (trades_readable[direction_col].str.lower() == 'long').sum() if direction_col else 0
                    short_count = (trades_readable[direction_col].str.lower() == 'short').sum() if direction_col else 0
                else:
                    # Fallback: assume all trades are long if no direction column
                    long_count = len(trades_readable)
                    short_count = 0
                
                enhanced_trade_stats['long_positions'] = int(long_count)
                enhanced_trade_stats['short_positions'] = int(short_count)
                
                # Largest win/loss calculations
                pnl_col = None
                pnl_pct_col = None
                for col in ['PnL', 'pnl', 'P&L', 'Profit']:
                    if col in trades_readable.columns:
                        pnl_col = col
                        break
                for col in ['Return', 'Return [%]', 'return_pct', 'Return%']:
                    if col in trades_readable.columns:
                        pnl_pct_col = col
                        break
                
                if pnl_col:
                    pnl_values = pd.to_numeric(trades_readable[pnl_col], errors='coerce').dropna()
                    if not pnl_values.empty:
                        enhanced_trade_stats['largest_win'] = sanitize_float(pnl_values.max())
                        enhanced_trade_stats['largest_loss'] = sanitize_float(abs(pnl_values.min()))  # Make positive
                    else:
                        enhanced_trade_stats['largest_win'] = 0.0
                        enhanced_trade_stats['largest_loss'] = 0.0
                else:
                    enhanced_trade_stats['largest_win'] = 0.0
                    enhanced_trade_stats['largest_loss'] = 0.0
                
                if pnl_pct_col:
                    pnl_pct_values = pd.to_numeric(trades_readable[pnl_pct_col], errors='coerce').dropna()
                    if not pnl_pct_values.empty:
                        # Convert to percentage if in decimal form
                        if pnl_pct_values.abs().max() <= 1.0:
                            pnl_pct_values = pnl_pct_values * 100
                        enhanced_trade_stats['largest_win_pct'] = sanitize_float(pnl_pct_values.max())
                        enhanced_trade_stats['largest_loss_pct'] = sanitize_float(abs(pnl_pct_values.min()))  # Make positive
                    else:
                        enhanced_trade_stats['largest_win_pct'] = 0.0
                        enhanced_trade_stats['largest_loss_pct'] = 0.0
                else:
                    enhanced_trade_stats['largest_win_pct'] = 0.0
                    enhanced_trade_stats['largest_loss_pct'] = 0.0
            else:
                # No trades available
                enhanced_trade_stats.update({
                    'long_positions': 0,
                    'short_positions': 0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'largest_win_pct': 0.0,
                    'largest_loss_pct': 0.0
                })
        else:
            # No trades data available
            enhanced_trade_stats.update({
                'long_positions': 0,
                'short_positions': 0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss_pct': 0.0
            })
        
        # Duration statistics (if trades available)
        duration_stats = {
            'avg_trade_duration_hours': 0.0,
            'max_trade_duration_hours': 0.0
        }
        
        if trades_df is not None and not trades_df.empty:
            durations = trades_df['Exit Timestamp'] - trades_df['Entry Timestamp']
            duration_hours = durations.dt.total_seconds() / 3600
            duration_stats['avg_trade_duration_hours'] = sanitize_float(duration_hours.mean())
            duration_stats['max_trade_duration_hours'] = sanitize_float(duration_hours.max())
        
        enhanced_stats = {
            'total_return_pct': total_return,
            'annualized_return_pct': sanitize_float(returns.mean() * 252 * 24 * 60 / 5 * 100) if len(returns.dropna()) > 0 else 0.0,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'volatility_pct': annual_vol,
            'var_95_pct': var_95,
            **trade_stats,
            **enhanced_trade_stats,
            **duration_stats
        }
        
        # Final sanitization of the entire dictionary
        return sanitize_stats_dict(enhanced_stats)
    except Exception as e:
        logging.error(f"Error calculating enhanced stats: {e}")
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'volatility_pct': 0.0,
            'var_95_pct': 0.0,
            'total_trades': 0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'long_positions': 0,
            'short_positions': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'largest_win_pct': 0.0,
            'largest_loss_pct': 0.0,
            'avg_trade_duration_hours': 0.0,
            'max_trade_duration_hours': 0.0
        }

def validate_price_data(price_data: pd.DataFrame) -> bool:
    """
    Validate that price data has the required columns and format
    
    Args:
        price_data: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if price_data.empty:
        logging.error("Price data is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in price_data.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for NaN values in critical columns
    critical_columns = ['open', 'high', 'low', 'close']
    for col in critical_columns:
        if price_data[col].isna().any():
            logging.warning(f"Found NaN values in {col} column")
    
    # Basic sanity checks
    if (price_data['high'] < price_data['low']).any():
        logging.error("Found high < low in price data")
        return False
    
    if (price_data['high'] < price_data['open']).any() or (price_data['high'] < price_data['close']).any():
        logging.error("Found high < open/close in price data")
        return False
    
    if (price_data['low'] > price_data['open']).any() or (price_data['low'] > price_data['close']).any():
        logging.error("Found low > open/close in price data")
        return False
    
    return True

def create_sample_price_data(n_periods: int = 100, symbol: str = "TEST") -> pd.DataFrame:
    """
    Create sample OHLCV data for testing
    
    Args:
        n_periods: Number of periods to generate
        symbol: Symbol name
        
    Returns:
        DataFrame with sample OHLCV data
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate time index
    start_time = datetime.now() - timedelta(minutes=n_periods)
    time_index = pd.date_range(start=start_time, periods=n_periods, freq='1min')
    
    # Generate realistic price data using random walk
    np.random.seed(42)  # For reproducible tests
    
    initial_price = 100.0
    returns = np.random.normal(0, 0.001, n_periods)  # 0.1% volatility
    
    # Generate close prices using cumulative returns
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i in range(n_periods):
        close = close_prices[i]
        # Generate realistic OHLC relationships
        noise = np.random.normal(0, 0.0005, 4)
        
        open_price = close_prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + abs(noise[0]))
        low = min(open_price, close) * (1 - abs(noise[1]))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'time': time_index[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    return df

def sanitize_float(value: float) -> float:
    """
    Sanitize float values for JSON serialization
    Converts NaN, inf, -inf to 0.0
    """
    if pd.isna(value) or np.isinf(value) or np.isnan(value):
        return 0.0
    return float(value)

def sanitize_stats_dict(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all float values in a stats dictionary for JSON serialization
    """
    sanitized = {}
    for key, value in stats_dict.items():
        if isinstance(value, (int, float)):
            sanitized[key] = sanitize_float(float(value))
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [sanitize_float(float(v)) if isinstance(v, (int, float)) else v for v in value]
        else:
            sanitized[key] = value
    return sanitized
