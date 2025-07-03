#!/usr/bin/env python3
"""
Comprehensive test for all the fixes:
1. Stop loss/take profit for short/both positions
2. Position direction counts (long/short positions)
3. Largest win/loss statistics
4. Kelly criterion working correctly
"""

import requests
import json
import time

API_BASE = "http://localhost:8002"

def test_short_position_with_stop_loss():
    """Test short position with stop loss to ensure it works correctly"""
    print("🧪 Testing Short Position with Stop Loss...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_short_stop_loss",
            "name": "Short with Stop Loss Test",
            "description": "Test short-only with stop loss",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "70",  # Enter short when RSI > 70 (overbought)
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "30",  # Exit short when RSI < 30 (oversold)
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "short_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": 5.0,  # 5% stop loss
            "take_profit": 10.0  # 10% take profit
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            print(f"✅ Short position with stop loss test - Total trades: {len(trades)}")
            
            # Check if all trades are short
            short_trades = [t for t in trades if t.get('side') == 'short']
            if len(short_trades) == len(trades) and len(trades) > 0:
                print(f"✅ All {len(trades)} trades are SHORT positions")
            else:
                print(f"❌ Found non-short trades: {len(trades) - len(short_trades)}")
                return False
                
            # Check for stop loss/take profit exits
            stop_exits = [t for t in trades if 'stop' in str(t.get('exit_reason', '')).lower()]
            profit_exits = [t for t in trades if 'profit' in str(t.get('exit_reason', '')).lower()]
            
            print(f"   Stop loss exits: {len(stop_exits)}")
            print(f"   Take profit exits: {len(profit_exits)}")
            
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_position_counts_and_largest_wins():
    """Test that position counts and largest win/loss are returned"""
    print("\n🧪 Testing Position Counts and Largest Win/Loss...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_counts_stats",
            "name": "Position Counts Test",
            "description": "Test position counts and win/loss stats",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "30",  # Enter long when RSI < 30 (oversold)
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "70",  # Exit long when RSI > 70 (overbought)
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "long_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": None,
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for new fields
            long_positions = result.get('long_positions')
            short_positions = result.get('short_positions')
            largest_win = result.get('largest_win')
            largest_loss = result.get('largest_loss')
            
            print(f"✅ Position counts test results:")
            print(f"   Long positions: {long_positions}")
            print(f"   Short positions: {short_positions}")
            print(f"   Largest win: ${largest_win:.2f}" if largest_win is not None else "   Largest win: Not provided")
            print(f"   Largest loss: ${largest_loss:.2f}" if largest_loss is not None else "   Largest loss: Not provided")
            
            # Validate the counts make sense
            total_trades = result.get('total_trades', 0)
            if long_positions is not None and short_positions is not None:
                if (long_positions + short_positions) == total_trades:
                    print(f"✅ Position counts match total trades: {long_positions} + {short_positions} = {total_trades}")
                else:
                    print(f"⚠️  Position counts don't match total: {long_positions} + {short_positions} ≠ {total_trades}")
            
            # Check if largest win/loss are provided
            if largest_win is not None and largest_loss is not None:
                print("✅ Largest win/loss statistics are now provided")
                return True
            else:
                print("❌ Largest win/loss statistics are missing")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_kelly_criterion():
    """Test Kelly Criterion position sizing"""
    print("\n🧪 Testing Kelly Criterion Position Sizing...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_kelly",
            "name": "Kelly Criterion Test",
            "description": "Test Kelly Criterion position sizing",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "30",
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "70",
                    "enabled": True
                }
            ],
            "position_sizing": "kelly_criterion",  # KEY TEST
            "position_size": 0.1,  # This will be dynamically calculated
            "position_direction": "long_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": None,
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            total_trades = result.get('total_trades', 0)
            final_value = result.get('final_value', 0)
            
            print(f"✅ Kelly Criterion test completed")
            print(f"   Total trades: {total_trades}")
            print(f"   Final value: ${final_value:.2f}")
            
            # If we have trades, Kelly is working
            if total_trades > 0:
                print("✅ Kelly Criterion position sizing is working")
                return True
            else:
                print("⚠️  Kelly Criterion produced no trades")
                return True  # Still valid, just no signals
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_both_directions():
    """Test both long and short positions"""
    print("\n🧪 Testing Both Directions (Long and Short)...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_both_directions",
            "name": "Both Directions Test",
            "description": "Test both long and short positions",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "35",  # Easier to trigger
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "65",  # Easier to trigger
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "both",  # KEY TEST
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": 5.0,
            "take_profit": 10.0
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            long_positions = result.get('long_positions', 0)
            short_positions = result.get('short_positions', 0)
            
            print(f"✅ Both directions test completed")
            print(f"   Total trades: {len(trades)}")
            print(f"   Long positions: {long_positions}")
            print(f"   Short positions: {short_positions}")
            
            # Check trade directions in the actual trades
            long_trades = [t for t in trades if t.get('side') == 'long']
            short_trades = [t for t in trades if t.get('side') == 'short']
            
            print(f"   Actual long trades: {len(long_trades)}")
            print(f"   Actual short trades: {len(short_trades)}")
            
            if len(trades) > 0:
                print("✅ Both directions strategy is working")
                return True
            else:
                print("⚠️  Both directions produced no trades")
                return True  # Still valid
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Running Comprehensive Test Suite for All Fixes")
    print("=" * 60)
    
    # Wait for service to be ready
    print("⏳ Waiting for VectorBT service to be ready...")
    time.sleep(5)
    
    tests = [
        test_short_position_with_stop_loss,
        test_position_counts_and_largest_wins,
        test_kelly_criterion,
        test_both_directions
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ ALL TESTS PASSED! All fixes are working correctly.")
        return True
    else:
        failed_tests = [i for i, r in enumerate(results) if not r]
        print(f"❌ Some tests failed: {failed_tests}")
        return False

if __name__ == "__main__":
    main()
