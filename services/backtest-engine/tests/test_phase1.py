"""
Test the basic functionality of the Phase 1 custom backtesting engine

This test validates:
- Core trade engine functionality
- Portfolio management
- Basic signal generation
- Simple backtest execution
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from core import BacktestExecutor, TradeEngine, PortfolioManager
from signals import SignalEngine


def create_test_data(start_date='2024-01-01', periods=100, freq='1H'):
    """Create synthetic OHLCV data for testing"""
    
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate synthetic price data with some trend and volatility
    np.random.seed(42)  # For reproducible results
    
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, periods)  # Small positive drift with volatility
    prices = [base_price]
    
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # Generate OHLCV from base prices
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intrabar volatility
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def simple_ma_strategy(data, timestamp):
    """
    Simple moving average crossover strategy
    
    Args:
        data: Historical OHLCV data up to current timestamp
        timestamp: Current timestamp
        
    Returns:
        Dictionary with signals for symbols
    """
    if len(data) < 20:  # Need enough data for indicators
        return {'BTCUSD': {'action': 'hold'}}
    
    # Calculate short and long moving averages
    short_ma = data['close'].rolling(window=10).mean()
    long_ma = data['close'].rolling(window=20).mean()
    
    # Get current values (latest)
    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    prev_short = short_ma.iloc[-2] if len(short_ma) > 1 else current_short
    prev_long = long_ma.iloc[-2] if len(long_ma) > 1 else current_long
    
    # Generate signals
    if current_short > current_long and prev_short <= prev_long:
        # Golden cross - buy signal
        signal = {'action': 'buy', 'side': 'long', 'confidence': 1.0}
    elif current_short < current_long and prev_short >= prev_long:
        # Death cross - sell signal  
        signal = {'action': 'sell', 'side': 'short', 'confidence': 1.0}
    else:
        # No signal
        signal = {'action': 'hold'}
    
    return {'BTCUSD': signal}


def test_trade_engine():
    """Test basic trade engine functionality"""
    print("Testing Trade Engine...")
    
    engine = TradeEngine(commission_rate=0.001, slippage=0.0001)
    
    # Test opening a position
    timestamp = pd.Timestamp('2024-01-01 10:00:00')
    success = engine.open_position(
        symbol='BTCUSD',
        side='long',
        size=1.0,
        price=50000.0,
        timestamp=timestamp,
        stop_loss=47500.0,  # 5% stop loss
        take_profit=55000.0  # 10% take profit
    )
    
    assert success, "Failed to open position"
    assert len(engine.positions) == 1, "Position not added"
    
    position = engine.get_position('BTCUSD')
    assert position is not None, "Position not found"
    assert position.side == 'long', "Wrong position side"
    assert position.stop_loss == 47500.0, "Wrong stop loss"
    
    # Test stop loss trigger
    trade = engine.check_stop_losses('BTCUSD', 47000.0, 48000.0, timestamp)
    assert trade is not None, "Stop loss should have triggered"
    assert trade.exit_reason == 'stop_loss', "Wrong exit reason"
    assert len(engine.positions) == 0, "Position should be closed"
    
    print("✓ Trade Engine tests passed")


def test_portfolio_manager():
    """Test portfolio management functionality"""
    print("Testing Portfolio Manager...")
    
    portfolio = PortfolioManager(initial_capital=100000.0, max_positions=5)
    
    # Test opening position with risk management
    timestamp = pd.Timestamp('2024-01-01 10:00:00')
    success = portfolio.open_position(
        symbol='BTCUSD',
        side='long',
        entry_price=50000.0,
        timestamp=timestamp,
        stop_loss_pct=0.05,  # 5% stop loss
        risk_per_trade=0.02  # 2% risk per trade
    )
    
    assert success, "Failed to open position"
    
    # Update capital with market prices
    market_prices = {'BTCUSD': 51000.0}  # 2% gain
    portfolio.update_capital(timestamp, market_prices)
    
    assert portfolio.current_capital > portfolio.initial_capital, "Capital should increase with profits"
    
    # Test closing position
    trade = portfolio.close_position('BTCUSD', 51000.0, timestamp, 'signal')
    assert trade is not None, "Failed to close position"
    assert trade.pnl > 0, "Should have positive PnL"
    
    print("✓ Portfolio Manager tests passed")


def test_signal_engine():
    """Test signal generation"""
    print("Testing Signal Engine...")
    
    # Create test data
    data = create_test_data(periods=50)
    
    signal_engine = SignalEngine()
    
    # Add indicators
    sma_short = signal_engine.add_sma(data, window=10, name='sma_10')
    sma_long = signal_engine.add_sma(data, window=20, name='sma_20')
    rsi = signal_engine.add_rsi(data, window=14)
    
    assert len(sma_short) == len(data), "SMA length mismatch"
    assert len(rsi) == len(data), "RSI length mismatch"
    
    # Generate signals
    ma_signals = signal_engine.generate_ma_crossover_signals('sma_10', 'sma_20', 'BTCUSD')
    
    assert len(ma_signals) == len(data), "Signal length mismatch"
    assert ma_signals.dtype == int, "Signals should be integers"
    
    print("✓ Signal Engine tests passed")


def test_backtest_executor():
    """Test complete backtest execution"""
    print("Testing Backtest Executor...")
    
    # Create test data
    data = create_test_data(periods=100)
    
    # Initialize executor
    executor = BacktestExecutor(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage=0.0001
    )
    
    # Set strategy
    executor.set_strategy(simple_ma_strategy)
    
    # Set risk parameters
    executor.set_risk_parameters(
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        risk_per_trade=0.02
    )
    
    # Run backtest
    results = executor.run_backtest(data, ['BTCUSD'])
    
    assert 'metrics' in results, "Results should contain metrics"
    assert 'trades' in results, "Results should contain trades"
    assert 'equity_curve' in results, "Results should contain equity curve"
    
    metrics = results['metrics']
    assert 'total_return_pct' in metrics, "Should have total return"
    assert 'num_trades' in metrics, "Should have number of trades"
    assert 'win_rate_pct' in metrics, "Should have win rate"
    
    print(f"✓ Backtest completed:")
    print(f"  - Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  - Number of Trades: {metrics['num_trades']}")
    print(f"  - Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"  - Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")


def test_stop_loss_precision():
    """Test that stop losses are executed at precise levels"""
    print("Testing Stop Loss Precision...")
    
    executor = BacktestExecutor(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    
    # Create simple data where price drops exactly to stop loss level
    timestamps = pd.date_range('2024-01-01', periods=5, freq='1H')
    data = pd.DataFrame({
        'open': [50000, 49000, 47500, 47500, 48000],
        'high': [50500, 49500, 47600, 47600, 48500],
        'low': [49500, 47500, 47500, 47400, 47800],  # Touches stop loss at bar 1
        'close': [49000, 47500, 47500, 47500, 48000],
        'volume': [1000] * 5
    }, index=timestamps)
    
    # Manual test of stop loss
    portfolio = PortfolioManager(initial_capital=100000.0)
    portfolio.trade_engine.commission_rate = 0.0
    portfolio.trade_engine.slippage = 0.0
    
    # Open long position
    success = portfolio.trade_engine.open_position(
        symbol='BTCUSD',
        side='long', 
        size=1.0,
        price=50000.0,
        timestamp=timestamps[0],
        stop_loss=47500.0  # Exactly 5% stop loss
    )
    
    assert success, "Failed to open position"
    
    # Check stop loss on second bar where low touches stop loss
    trade = portfolio.trade_engine.check_stop_losses(
        'BTCUSD', 
        data.iloc[1]['low'],  # 47500 - exactly touches stop loss
        data.iloc[1]['high'],
        timestamps[1]
    )
    
    assert trade is not None, "Stop loss should have triggered"
    assert trade.exit_price == 47500.0, f"Stop loss should execute at 47500, got {trade.exit_price}"
    
    # Calculate actual loss percentage
    loss_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
    expected_loss = -0.05  # -5%
    
    assert abs(loss_pct - expected_loss) < 0.001, f"Loss should be exactly -5%, got {loss_pct:.3%}"
    
    print(f"✓ Stop loss precision test passed")
    print(f"  - Entry Price: ${trade.entry_price:,.2f}")
    print(f"  - Stop Loss Price: ${trade.exit_price:,.2f}")
    print(f"  - Actual Loss: {loss_pct:.3%}")


if __name__ == "__main__":
    print("=" * 50)
    print("CUSTOM BACKTESTING ENGINE - PHASE 1 TESTS")
    print("=" * 50)
    
    try:
        test_trade_engine()
        test_portfolio_manager()
        test_signal_engine()
        test_backtest_executor()
        test_stop_loss_precision()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Phase 1 implementation is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
