#!/usr/bin/env python3
"""
Statistics Engine Demonstration

This script demonstrates the advanced statistics capabilities of the custom backtesting engine,
showcasing performance metrics, risk analysis, trade attribution, and comprehensive reporting
that far exceed VectorBT's capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.backtest_executor import BacktestExecutor
from src.statistics.statistics_engine import StatisticsEngine


def create_sample_data(periods=1000, start_price=50000):
    """Create realistic cryptocurrency price data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='1h')
    
    # Generate realistic OHLCV data with trending and volatility
    returns = np.random.normal(0.0001, 0.02, periods)  # Slight upward bias
    prices = [start_price]
    
    for i in range(1, periods):
        # Add some momentum and mean reversion
        momentum = 0.1 * returns[i-1] if i > 0 else 0
        mean_reversion = -0.05 * (prices[-1] / start_price - 1)
        
        price_change = returns[i] + momentum + mean_reversion
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, start_price * 0.5))  # Prevent unrealistic crashes
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


def momentum_strategy(data, timestamp):
    """
    Enhanced momentum strategy with multiple signals
    """
    if len(data) < 50:
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate technical indicators
    sma_short = data['close'].rolling(10).mean()
    sma_long = data['close'].rolling(30).mean()
    rsi = calculate_rsi(data['close'], 14)
    current_price = data['close'].iloc[-1]
    
    # Generate signals based on multiple conditions
    momentum_signal = sma_short.iloc[-1] > sma_long.iloc[-1]
    rsi_signal = 30 < rsi.iloc[-1] < 70  # Avoid overbought/oversold
    trend_signal = data['close'].iloc[-1] > data['close'].iloc[-5]
    
    # Combine signals for strength
    signal_strength = sum([momentum_signal, rsi_signal, trend_signal])
    
    if signal_strength >= 2:
        return {
            'BTCUSD': {
                'action': 'buy',
                'side': 'long',
                'signal_strength': signal_strength / 3.0
            }
        }
    elif signal_strength <= 1:
        return {
            'BTCUSD': {
                'action': 'sell',
                'side': 'short',
                'signal_strength': (3 - signal_strength) / 3.0
            }
        }
    else:
        return {'BTCUSD': {'action': 'hold'}}


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def run_statistics_demo():
    """Run comprehensive statistics engine demonstration"""
    
    print("🚀 CUSTOM BACKTESTING ENGINE - STATISTICS DEMO")
    print("=" * 80)
    
    # 1. Generate realistic market data
    print("\n📊 Generating realistic market data...")
    data = create_sample_data(periods=2000, start_price=45000)
    print(f"Generated {len(data)} hourly data points from {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    # 2. Set up backtesting with advanced risk management
    print("\n⚙️  Setting up advanced backtesting...")
    executor = BacktestExecutor(initial_capital=100000.0)
    executor.set_strategy(momentum_strategy)
    
    # Advanced risk parameters  
    executor.set_risk_parameters(
        stop_loss_pct=0.03,          # 3% stop loss
        take_profit_pct=0.08,        # 8% take profit
        risk_per_trade=0.025,        # 2.5% risk per trade
        max_drawdown_limit=0.25,     # 25% max drawdown (higher for demo)
        max_positions=3,             # Max concurrent positions
        position_sizing_method='volatility_adjusted'  # Dynamic sizing
    )
    
    # 3. Run the backtest
    print("\n🎯 Running backtest with momentum strategy...")
    results = executor.run_backtest(data, ['BTCUSD'])
    
    # 4. Initialize statistics engine
    print("\n📈 Analyzing performance with statistics engine...")
    stats_engine = StatisticsEngine()
    
    # Extract data for analysis
    portfolio = executor.portfolio  # Correct attribute name
    equity_curve = portfolio.get_equity_curve()
    
    # Get completed trades from trade engine
    all_trades = portfolio.trade_engine.get_trades()
    completed_trades = [trade for trade in all_trades if trade.exit_price is not None]
    
    # Convert trades to expected format for statistics engine
    trades_data = []
    for trade in completed_trades:
        trades_data.append({
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'symbol': trade.symbol,
            'side': trade.side,
            'size': trade.size,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'commission': trade.commission
        })
    
    # 5. Calculate comprehensive performance metrics
    metrics = stats_engine.calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades_data,
        initial_capital=100000.0
    )
    
    # 6. Calculate rolling statistics
    rolling_metrics = stats_engine.calculate_rolling_metrics(
        equity_curve=equity_curve,
        window=168  # 1 week rolling (168 hours)
    )
    
    # 7. Perform trade analysis
    trade_analysis = stats_engine.analyze_trades(trades_data)
    
    # 8. Generate comprehensive report
    report = stats_engine.generate_report(
        metrics=metrics,
        rolling_metrics=rolling_metrics,
        trade_analysis=trade_analysis
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(report)
    
    # 9. Additional insights
    print("\n" + "=" * 80)
    print("🔍 ADDITIONAL INSIGHTS")
    print("=" * 80)
    
    print(f"\n💰 PORTFOLIO PERFORMANCE:")
    print(f"Initial Capital:        ${executor.initial_capital:>10,.2f}")
    print(f"Final Equity:           ${portfolio.current_capital:>10,.2f}")
    print(f"Total Return:           {metrics.total_return:>10.2%}")
    print(f"Annualized Return:      {metrics.annualized_return:>10.2%}")
    print(f"Best Month:             {metrics.best_month:>10.2%}")
    print(f"Worst Month:            {metrics.worst_month:>10.2%}")
    
    print(f"\n⚠️  RISK ANALYSIS:")
    print(f"Maximum Drawdown:       {metrics.max_drawdown:>10.2%}")
    print(f"Current Drawdown:       {metrics.current_drawdown:>10.2%}")
    print(f"Volatility (Annual):    {metrics.annualized_volatility:>10.2%}")
    print(f"VaR (95%):              {metrics.var_95:>10.2%}")
    print(f"CVaR (95%):             {metrics.cvar_95:>10.2%}")
    print(f"Skewness:               {metrics.skewness:>10.2f}")
    print(f"Kurtosis:               {metrics.kurtosis:>10.2f}")
    
    print(f"\n📊 RISK-ADJUSTED METRICS:")
    print(f"Sharpe Ratio:           {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:          {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:           {metrics.calmar_ratio:>10.2f}")
    print(f"Omega Ratio:            {metrics.omega_ratio:>10.2f}")
    
    print(f"\n🎯 TRADING PERFORMANCE:")
    print(f"Total Trades:           {metrics.total_trades:>10d}")
    print(f"Win Rate:               {metrics.win_rate:>10.1%}")
    print(f"Profit Factor:          {metrics.profit_factor:>10.2f}")
    print(f"Average Win:            {metrics.average_win:>10.2%}")
    print(f"Average Loss:           {metrics.average_loss:>10.2%}")
    print(f"Largest Win:            {metrics.largest_win:>10.2%}")
    print(f"Largest Loss:           {metrics.largest_loss:>10.2%}")
    
    print(f"\n📈 ROLLING ANALYTICS (Latest Values):")
    if len(rolling_metrics.returns) > 0:
        print(f"Rolling Return (7d):    {rolling_metrics.returns[-1]:>10.2%}")
        print(f"Rolling Volatility:     {rolling_metrics.volatility[-1]:>10.2%}")
        print(f"Rolling Sharpe:         {rolling_metrics.sharpe_ratio[-1]:>10.2f}")
        print(f"Rolling Drawdown:       {rolling_metrics.drawdown[-1]:>10.2%}")
    
    # 10. Performance comparison insight
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
    print(f"Market Return (B&H):    {market_return:>10.2%}")
    print(f"Strategy Return:        {metrics.total_return:>10.2%}")
    print(f"Excess Return:          {metrics.total_return - market_return:>10.2%}")
    
    if metrics.total_return > market_return:
        print("✅ Strategy OUTPERFORMED buy-and-hold!")
    else:
        print("❌ Strategy underperformed buy-and-hold")
    
    print(f"\n📊 DATA QUALITY:")
    print(f"Data Points:            {len(data):>10d}")
    print(f"Equity Curve Points:    {len(equity_curve):>10d}")
    print(f"Completed Trades:       {len(trades_data):>10d}")
    print(f"Analysis Period:        {(data.index[-1] - data.index[0]).days:>10d} days")
    
    print("\n" + "=" * 80)
    print("🎉 STATISTICS ENGINE DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\nKey advantages over VectorBT:")
    print("✅ Complete transparency in all calculations")
    print("✅ Advanced risk metrics (VaR, CVaR, skewness, kurtosis)")
    print("✅ Rolling analytics with customizable windows")
    print("✅ Comprehensive trade attribution analysis")
    print("✅ Robust error handling and edge case management")
    print("✅ Flexible reporting system with detailed insights")
    print("\nThe custom statistics engine provides institutional-grade")
    print("performance analysis that far exceeds VectorBT's capabilities!")


if __name__ == "__main__":
    run_statistics_demo()
