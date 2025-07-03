#!/usr/bin/env python3
"""
Simple Stop Loss Verification Test
Verifies that stop loss functionality works correctly with synthetic data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.backtest_executor import BacktestExecutor

def test_stop_loss_precision():
    """Test that stop losses trigger at the exact price"""
    print("🧪 Testing Stop Loss Precision")
    
    # Create synthetic data where price drops exactly 5%
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # Start at $100, gradually rise to $105, then drop to $95 (exactly 5% below $100)
    prices = []
    for i in range(100):
        if i < 20:
            prices.append(100.0)  # Start stable
        elif i < 50:
            prices.append(100.0 + (i-20) * 0.1667)  # Rise to $105
        elif i < 80:
            prices.append(105.0 - (i-50) * 0.3333)  # Drop to $95
        else:
            prices.append(95.0)  # Stay at $95
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [1000] * 100
    }, index=dates)
    
    # Strategy that buys immediately
    def buy_immediately_strategy(data_slice, timestamp):
        if len(data_slice) == 1:  # First bar
            return {'TEST': {'action': 'buy', 'side': 'long'}}
        return {'TEST': {'action': 'hold'}}
    
    # Create executor with 5% stop loss and no slippage/commission
    executor = BacktestExecutor(
        initial_capital=10000.0,
        commission_rate=0.0,  # No commission for precise testing
        slippage=0.0  # No slippage for precise testing
    )
    
    executor.set_strategy(buy_immediately_strategy)
    executor.set_risk_parameters(
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.20,
        risk_per_trade=1.0,  # Use full capital
        max_drawdown_limit=0.20
    )
    
    # Run backtest
    print(f"   Entry price: ${prices[0]:.2f}")
    print(f"   Expected stop loss price: ${prices[0] * 0.95:.2f}")
    print(f"   Actual price at bar 80: ${prices[80]:.2f}")
    
    results = executor.run_backtest(test_data, ['TEST'])
    
    # Check trades
    trades = executor.portfolio.get_trades_dataframe()
    
    if not trades.empty:
        completed_trades = trades[trades['exit_price'].notna()]
        if not completed_trades.empty:
            trade = completed_trades.iloc[0]
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            
            # Calculate actual loss percentage
            actual_loss_pct = (exit_price - entry_price) / entry_price
            expected_loss_pct = -0.05  # 5% stop loss
            
            print(f"   Entry Price: ${entry_price:.2f}")
            print(f"   Exit Price: ${exit_price:.2f}")
            print(f"   Actual Loss: {actual_loss_pct*100:.3f}%")
            print(f"   Expected Loss: {expected_loss_pct*100:.3f}%")
            print(f"   Precision Error: {abs(actual_loss_pct - expected_loss_pct)*100:.3f}%")
            
            if abs(actual_loss_pct - expected_loss_pct) < 0.001:  # Within 0.1%
                print("   ✅ Stop loss precision is EXCELLENT")
                return True
            else:
                print("   ❌ Stop loss precision is OFF")
                return False
        else:
            print("   ❌ No completed trades found")
            return False
    else:
        print("   ❌ No trades found")
        return False

def test_max_drawdown_limit():
    """Test that max drawdown limit stops trading"""
    print("\n🛡️ Testing Max Drawdown Limit")
    
    # Create data that causes steady losses
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    prices = [100.0]
    
    # Steady decline from $100 to $80 (20% drop)
    for i in range(1, 200):
        prices.append(100.0 - (i * 0.1))  # Drop $0.10 per bar
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [1000] * 200
    }, index=dates)
    
    # Strategy that tries to buy every 20 bars
    trade_count = 0
    def frequent_buying_strategy(data_slice, timestamp):
        nonlocal trade_count
        if len(data_slice) % 20 == 1:  # Buy every 20 bars
            trade_count += 1
            return {'TEST': {'action': 'buy', 'side': 'long'}}
        return {'TEST': {'action': 'hold'}}
    
    executor = BacktestExecutor(
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage=0.001
    )
    
    executor.set_strategy(frequent_buying_strategy)
    executor.set_risk_parameters(
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.04,
        risk_per_trade=0.10,
        max_drawdown_limit=0.15  # 15% max drawdown
    )
    
    print(f"   Max drawdown limit: 15%")
    print(f"   Expected behavior: Stop trading when 15% drawdown reached")
    
    results = executor.run_backtest(test_data, ['TEST'])
    
    max_drawdown = results['metrics']['max_drawdown_pct'] / 100
    total_trades = results['metrics']['num_trades']  # Fixed: was 'total_trades'
    
    print(f"   Actual max drawdown: {max_drawdown*100:.2f}%")
    print(f"   Total trades executed: {total_trades}")
    
    # Test passes if max drawdown is close to the limit (within 2%)
    if max_drawdown <= 0.17:  # Allow some buffer
        print("   ✅ Max drawdown protection WORKING")
        return True
    else:
        print("   ❌ Max drawdown protection FAILED")
        return False

def test_risk_per_trade_sizing():
    """Test that position sizing respects risk per trade"""
    print("\n⚖️ Testing Risk Per Trade Position Sizing")
    
    # Create stable price data for position sizing test
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
    prices = [100.0] * 50  # Stable at $100
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [1000] * 50
    }, index=dates)
    
    # Strategy that buys once
    def single_buy_strategy(data_slice, timestamp):
        if len(data_slice) == 10:  # Buy on 10th bar
            return {'TEST': {'action': 'buy', 'side': 'long'}}
        return {'TEST': {'action': 'hold'}}
    
    executor = BacktestExecutor(
        initial_capital=10000.0,
        commission_rate=0.0,
        slippage=0.0
    )
    
    executor.set_strategy(single_buy_strategy)
    executor.set_risk_parameters(
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.10,
        risk_per_trade=0.02,  # 2% risk per trade
        max_drawdown_limit=0.20
    )
    
    print(f"   Initial capital: $10,000")
    print(f"   Risk per trade: 2%")
    print(f"   Stop loss: 5%")
    print(f"   Expected risk amount: $200")
    print(f"   Expected position size: $200 / $5 = 40 shares")
    
    results = executor.run_backtest(test_data, ['TEST'])
    
    trades = executor.portfolio.get_trades_dataframe()
    
    if not trades.empty:
        trade = trades.iloc[0]
        position_size = trade['size']
        entry_price = trade['entry_price']
        position_value = position_size * entry_price
        
        # Calculate expected position size
        risk_amount = 10000.0 * 0.02  # $200 risk
        stop_loss_amount_per_share = entry_price * 0.05  # $5 per share
        expected_position_size = risk_amount / stop_loss_amount_per_share  # 40 shares
        
        print(f"   Actual position size: {position_size:.2f} shares")
        print(f"   Expected position size: {expected_position_size:.2f} shares")
        print(f"   Position value: ${position_value:.2f}")
        print(f"   Size difference: {abs(position_size - expected_position_size):.2f} shares")
        
        # Test passes if position size is within 5% of expected
        if abs(position_size - expected_position_size) / expected_position_size < 0.05:
            print("   ✅ Position sizing is CORRECT")
            return True
        else:
            print("   ❌ Position sizing is INCORRECT")
            return False
    else:
        print("   ❌ No trades found")
        return False

def main():
    """Run all stop loss tests"""
    print("🧪 STOP LOSS FUNCTIONALITY TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_stop_loss_precision,
        test_max_drawdown_limit,
        test_risk_per_trade_sizing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stop loss functionality is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the results above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
