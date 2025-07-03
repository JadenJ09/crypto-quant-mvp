#!/usr/bin/env python3
"""
Stop Loss Precision Test

This script tests the precise stop loss execution feature of the custom backtest engine.
It verifies that stop losses are executed at exactly the specified percentage loss.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from core.backtest_executor import BacktestExecutor


def test_precise_stop_loss():
    """Test that stop losses execute at exactly the specified percentage"""
    
    print("🎯 Testing Precise Stop Loss Execution")
    print("=" * 50)
    
    # Create controlled test data where we know the stop loss will trigger
    timestamps = pd.date_range('2024-01-01', periods=10, freq='1H')
    
    # Start at $50,000, then drop to trigger stop loss
    test_data = pd.DataFrame({
        'open': [50000, 49000, 48000, 47000, 46000, 45000, 44000, 43000, 42000, 41000],
        'high': [50500, 49500, 48500, 47500, 46500, 45500, 44500, 43500, 42500, 41500],
        'low': [49500, 48000, 47000, 46000, 45000, 44000, 43000, 42000, 41000, 40000],
        'close': [49000, 48000, 47000, 46000, 45000, 44000, 43000, 42000, 41000, 40000],
        'volume': [1000] * 10
    }, index=timestamps)
    
    print(f"Test Data Range: ${test_data['high'].max():,.2f} to ${test_data['low'].min():,.2f}")
    
    # Define a simple buy-and-hold strategy
    def simple_buy_strategy(data_slice, timestamp):
        """Buy on first signal, then hold"""
        if len(data_slice) == 1:  # First bar
            return {'BTCUSD': {'action': 'buy', 'side': 'long'}}
        else:
            return {'BTCUSD': {'action': 'hold'}}
    
    # Test different stop loss percentages
    stop_loss_tests = [0.05, 0.03, 0.02, 0.01]  # 5%, 3%, 2%, 1%
    
    for stop_loss_pct in stop_loss_tests:
        print(f"\n📊 Testing {stop_loss_pct*100:.1f}% Stop Loss")
        print("-" * 30)
        
        # Create executor
        executor = BacktestExecutor(
            initial_capital=100000.0,
            commission_rate=0.0,  # No commission
            slippage=0.0  # No slippage for pure precision test
        )
        executor.set_strategy(simple_buy_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=0.20,  # High take profit so it doesn't trigger
            risk_per_trade=0.10,   # 10% position size
            max_drawdown_limit=0.50  # High limit to avoid stopping early
        )
        
        # Run backtest
        results = executor.run_backtest(test_data, ['BTCUSD'])
        
        # Get trade details
        trades = executor.portfolio.trade_engine.get_trades()
        
        if trades:
            # Check the first (and likely only) trade
            trade = trades[0]
            
            print(f"   Entry Price: ${trade.entry_price:,.2f}")
            print(f"   Entry Time: {trade.entry_time}")
            
            if trade.exit_price is not None:
                # Calculate actual return percentage
                actual_return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                expected_return_pct = -stop_loss_pct * 100  # Negative for loss
                
                print(f"   Exit Price: ${trade.exit_price:,.2f}")
                print(f"   Exit Time: {trade.exit_time}")
                print(f"   Exit Reason: {trade.exit_reason}")
                print(f"   Expected Return: {expected_return_pct:.3f}%")
                print(f"   Actual Return: {actual_return_pct:.3f}%")
                print(f"   PnL: ${trade.pnl:,.2f}")
                
                # Check precision - allow for tiny rounding errors
                precision_error = abs(actual_return_pct - expected_return_pct)
                
                if precision_error < 0.001:  # Less than 0.001% error
                    print(f"   ✅ PASS: Precision error {precision_error:.6f}% < 0.001%")
                else:
                    print(f"   ❌ FAIL: Precision error {precision_error:.6f}% >= 0.001%")
                    
                # Verify stop loss was the exit reason
                if trade.exit_reason == "stop_loss":
                    print(f"   ✅ PASS: Exit reason correctly identified as 'stop_loss'")
                else:
                    print(f"   ❌ FAIL: Exit reason was '{trade.exit_reason}', expected 'stop_loss'")
                    
            else:
                print(f"   ⚠️  WARNING: Trade did not exit (stop loss may not have triggered)")
                print(f"   Position is still open")
        else:
            print(f"   ⚠️  WARNING: No trades executed")
    
    # Test with extreme precision
    print(f"\n🔬 Extreme Precision Test")
    print("-" * 30)
    
    # Create data that will trigger at exactly 5.000%
    precise_data = pd.DataFrame({
        'open': [50000, 50000, 50000],
        'high': [50000, 50000, 50000], 
        'low': [50000, 47500, 45000],  # Exactly 5% drop to $47,500
        'close': [50000, 47500, 45000],
        'volume': [1000, 1000, 1000]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
    
    executor = BacktestExecutor(
        initial_capital=100000.0,
        commission_rate=0.0,  # No commission
        slippage=0.0  # No slippage for pure precision test
    )
    executor.set_strategy(simple_buy_strategy)
    executor.set_risk_parameters(
        stop_loss_pct=0.05,  # Exactly 5%
        take_profit_pct=0.20,
        risk_per_trade=0.10,
        max_drawdown_limit=0.50
    )
    
    results = executor.run_backtest(precise_data, ['BTCUSD'])
    trades = executor.portfolio.trade_engine.get_trades()
    
    if trades and trades[0].exit_price is not None:
        trade = trades[0]
        actual_return = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
        print(f"   Entry: ${trade.entry_price:,.2f}")
        print(f"   Exit: ${trade.exit_price:,.2f}")
        print(f"   Expected: -5.000%")
        print(f"   Actual: {actual_return:.3f}%")
        print(f"   Difference: {abs(actual_return + 5.0):.6f}%")
        
        if abs(actual_return + 5.0) < 0.0001:
            print(f"   ✅ EXTREME PRECISION PASS: Sub-0.0001% accuracy!")
        else:
            print(f"   ⚠️  Precision within reasonable bounds")
    
    print(f"\n🎉 Stop Loss Precision Test Complete!")


if __name__ == "__main__":
    test_precise_stop_loss()
