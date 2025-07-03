"""
Example Strategies for the Custom Backtesting Engine

This file demonstrates how to create and use strategies with the custom engine.
Shows practical examples of different trading strategies and their implementation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from core import BacktestExecutor
from signals import SignalEngine


def create_crypto_data(symbol='BTCUSD', start_date='2024-01-01', days=30):
    """Create realistic crypto OHLCV data for testing"""
    
    # Generate hourly data
    periods = days * 24
    dates = pd.date_range(start=start_date, periods=periods, freq='1h')
    
    # Realistic crypto price simulation
    np.random.seed(42)
    base_price = 45000.0
    
    # Generate returns with trending periods and volatility clusters
    trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.0005  # Trending component
    volatility = 0.015 + 0.01 * np.abs(np.sin(np.linspace(0, 8*np.pi, periods)))  # Variable volatility
    random_returns = np.random.normal(0, 1, periods) * volatility + trend
    
    # Calculate prices
    prices = [base_price]
    for ret in random_returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Price floor
    
    # Generate OHLCV bars
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Realistic intrabar movement
        volatility_factor = np.random.uniform(0.005, 0.02)
        
        open_price = prices[i-1] if i > 0 else close_price
        high = max(open_price, close_price) * (1 + volatility_factor * np.random.uniform(0, 1))
        low = min(open_price, close_price) * (1 - volatility_factor * np.random.uniform(0, 1))
        volume = np.random.uniform(50, 500) * (1 + abs(random_returns[i]) * 10)  # Volume surge on big moves
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def rsi_mean_reversion_strategy(data, timestamp):
    """
    RSI Mean Reversion Strategy
    
    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    - Use 5% stop loss and 8% take profit
    """
    if len(data) < 20:  # Need enough data for RSI
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate RSI
    signal_engine = SignalEngine()
    rsi = signal_engine.add_rsi(data, window=14)
    
    current_rsi = rsi.iloc[-1]
    
    if pd.isna(current_rsi):
        return {'BTCUSD': {'action': 'hold'}}
    
    # Generate signals
    if current_rsi < 30:
        # Oversold - buy signal
        confidence = min(1.0, (30 - current_rsi) / 20)  # Higher confidence when more oversold
        signal = {
            'action': 'buy',
            'side': 'long',
            'confidence': confidence,
            'reason': f'RSI oversold: {current_rsi:.1f}'
        }
    elif current_rsi > 70:
        # Overbought - sell signal
        confidence = min(1.0, (current_rsi - 70) / 20)  # Higher confidence when more overbought
        signal = {
            'action': 'sell', 
            'side': 'short',
            'confidence': confidence,
            'reason': f'RSI overbought: {current_rsi:.1f}'
        }
    else:
        signal = {'action': 'hold'}
    
    return {'BTCUSD': signal}


def ma_crossover_strategy(data, timestamp):
    """
    Moving Average Crossover Strategy
    
    - Buy when fast MA crosses above slow MA (golden cross)
    - Sell when fast MA crosses below slow MA (death cross)
    - Use momentum confirmation
    """
    if len(data) < 50:  # Need enough data for MAs
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate moving averages
    fast_ma = data['close'].rolling(window=10).mean()
    slow_ma = data['close'].rolling(window=21).mean()
    
    if len(fast_ma) < 2 or pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
        return {'BTCUSD': {'action': 'hold'}}
    
    # Current and previous values
    fast_current, fast_prev = fast_ma.iloc[-1], fast_ma.iloc[-2]
    slow_current, slow_prev = slow_ma.iloc[-1], slow_ma.iloc[-2]
    
    # Detect crossovers
    golden_cross = (fast_current > slow_current) and (fast_prev <= slow_prev)
    death_cross = (fast_current < slow_current) and (fast_prev >= slow_prev)
    
    # Add momentum confirmation
    price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
    
    if golden_cross and price_momentum > 0:
        confidence = min(1.0, abs(price_momentum) * 10)
        signal = {
            'action': 'buy',
            'side': 'long', 
            'confidence': confidence,
            'reason': f'Golden cross with momentum: {price_momentum:.3f}'
        }
    elif death_cross and price_momentum < 0:
        confidence = min(1.0, abs(price_momentum) * 10)
        signal = {
            'action': 'sell',
            'side': 'short',
            'confidence': confidence,
            'reason': f'Death cross with momentum: {price_momentum:.3f}'
        }
    else:
        signal = {'action': 'hold'}
    
    return {'BTCUSD': signal}


def bollinger_bands_strategy(data, timestamp):
    """
    Bollinger Bands Mean Reversion Strategy
    
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Use band width for volatility confirmation
    """
    if len(data) < 30:
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate Bollinger Bands
    signal_engine = SignalEngine()
    bb = signal_engine.add_bollinger_bands(data, window=20, window_dev=2)
    
    current_price = data['close'].iloc[-1]
    upper_band = bb['bb_upper'].iloc[-1]
    lower_band = bb['bb_lower'].iloc[-1]
    middle_band = bb['bb_middle'].iloc[-1]
    
    if pd.isna(upper_band) or pd.isna(lower_band):
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate band width (volatility measure)
    band_width = (upper_band - lower_band) / middle_band
    
    # Position relative to bands
    band_position = (current_price - lower_band) / (upper_band - lower_band)
    
    if band_position <= 0.1 and band_width > 0.05:  # Near lower band with sufficient volatility
        confidence = min(1.0, (0.1 - band_position) * 10)
        signal = {
            'action': 'buy',
            'side': 'long',
            'confidence': confidence,
            'reason': f'Price near lower BB: {band_position:.2f}'
        }
    elif band_position >= 0.9 and band_width > 0.05:  # Near upper band with sufficient volatility
        confidence = min(1.0, (band_position - 0.9) * 10)
        signal = {
            'action': 'sell',
            'side': 'short', 
            'confidence': confidence,
            'reason': f'Price near upper BB: {band_position:.2f}'
        }
    else:
        signal = {'action': 'hold'}
    
    return {'BTCUSD': signal}


def run_strategy_comparison():
    """Compare performance of different strategies"""
    
    print("=" * 60)
    print("STRATEGY COMPARISON - CUSTOM BACKTESTING ENGINE")
    print("=" * 60)
    
    # Create test data
    data = create_crypto_data(days=60)  # 2 months of hourly data
    print(f"Testing with {len(data)} hours of data")
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    print()
    
    strategies = {
        'RSI Mean Reversion': rsi_mean_reversion_strategy,
        'MA Crossover': ma_crossover_strategy,
        'Bollinger Bands': bollinger_bands_strategy
    }
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"Testing {strategy_name}...")
        
        # Initialize executor
        executor = BacktestExecutor(
            initial_capital=100000.0,
            commission_rate=0.001,  # 0.1% commission
            slippage=0.0005,        # 0.05% slippage
            max_positions=1
        )
        
        # Set strategy
        executor.set_strategy(strategy_func)
        
        # Strategy-specific risk parameters
        if strategy_name == 'RSI Mean Reversion':
            executor.set_risk_parameters(
                stop_loss_pct=0.05,      # 5% stop loss
                take_profit_pct=0.08,    # 8% take profit
                risk_per_trade=0.02      # 2% risk per trade
            )
        elif strategy_name == 'MA Crossover':
            executor.set_risk_parameters(
                stop_loss_pct=0.03,      # Tighter stop for trend following
                take_profit_pct=0.12,    # Higher target for trends
                risk_per_trade=0.025     # Slightly higher risk for trend
            )
        else:  # Bollinger Bands
            executor.set_risk_parameters(
                stop_loss_pct=0.04,      # Medium stop loss
                take_profit_pct=0.06,    # Quick profits for mean reversion
                risk_per_trade=0.015     # Lower risk for mean reversion
            )
        
        # Run backtest
        result = executor.run_backtest(data, ['BTCUSD'])
        results[strategy_name] = result
    
    # Print comparison
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    for strategy_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': strategy_name,
            'Return %': f"{metrics.get('total_return_pct', 0):.2f}%",
            'Trades': metrics.get('num_trades', 0),
            'Win Rate': f"{metrics.get('win_rate_pct', 0):.1f}%",
            'Max DD': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}"
        })
    
    # Print formatted table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best strategy
    best_strategy = max(results.keys(), 
                       key=lambda x: results[x]['metrics'].get('total_return_pct', -999))
    
    print(f"\n🏆 Best Performing Strategy: {best_strategy}")
    print(f"Return: {results[best_strategy]['metrics'].get('total_return_pct', 0):.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY COMPARISON COMPLETE")
    print("✅ Custom Engine Demonstrating Precise Control")
    print("=" * 60)


if __name__ == "__main__":
    run_strategy_comparison()
