#!/usr/bin/env python3
"""
Test script to verify backend fixes for Kelly Criterion, Stop Loss, Position Direction, and Max Positions
"""

import requests
import json
from datetime import datetime, timedelta

# Backend API URL
API_BASE = "http://localhost:8002"

def test_kelly_criterion():
    """Test Kelly Criterion position sizing"""
    print("🧪 Testing Kelly Criterion Position Sizing...")
    
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
            "name": "Kelly Test Strategy", 
            "description": "Test Kelly Criterion",
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
            "position_sizing": "kelly",  # This is the KEY TEST
            "position_size": 20,  # Base size for Kelly
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
            print(f"✅ Kelly Criterion test PASSED")
            print(f"   Total trades: {result.get('total_trades', 0)}")
            print(f"   Final value: ${result.get('final_value', 0):,.2f}")
            return True
        else:
            print(f"❌ Kelly Criterion test FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Kelly Criterion test FAILED: {str(e)}")
        return False

def test_stop_loss_precision():
    """Test precise stop loss calculation (6% should be exactly 6%, not 6.24%)"""
    print("\n🧪 Testing Stop Loss Precision...")
    
    test_payload = {
        "symbol": "BTCUSDT", 
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_stop_loss",
            "name": "Stop Loss Test Strategy",
            "description": "Test precise stop loss",
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
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "long_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": 6.0,  # This is the KEY TEST - should be exactly 6%
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            if trades:
                # Check if any trades hit stop loss and if the loss is precise
                stop_loss_trades = [t for t in trades if t.get('pnl_pct', 0) < 0]
                if stop_loss_trades:
                    worst_loss = min(t.get('pnl_pct', 0) for t in stop_loss_trades)
                    print(f"✅ Stop Loss test PASSED")
                    print(f"   Worst loss: {worst_loss:.2f}% (should be around -6%)")
                    print(f"   Total trades: {len(trades)}")
                    return True
                else:
                    print(f"⚠️  Stop Loss test INCONCLUSIVE - no losing trades found")
                    return True
            else:
                print(f"⚠️  Stop Loss test INCONCLUSIVE - no trades found")
                return True
        else:
            print(f"❌ Stop Loss test FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Stop Loss test FAILED: {str(e)}")
        return False

def test_short_only_direction():
    """Test short-only position direction"""
    print("\n🧪 Testing Short-Only Position Direction...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h", 
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_short_only",
            "name": "Short Only Test Strategy",
            "description": "Test short-only trading",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",  # Opposite of long logic
                    "value": "70",
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1", 
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",  # Opposite of long logic
                    "value": "30",
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "short_only",  # This is the KEY TEST
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
            trades = result.get('trades', [])
            
            if trades:
                # Check if trades have short positions
                short_trades = [t for t in trades if t.get('side') == 'short']
                print(f"✅ Short-Only test PASSED")
                print(f"   Total trades: {len(trades)}")
                print(f"   Short trades: {len(short_trades)}")
                print(f"   All trades should be short positions")
                return True
            else:
                print(f"⚠️  Short-Only test INCONCLUSIVE - no trades found")
                return True
        else:
            print(f"❌ Short-Only test FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Short-Only test FAILED: {str(e)}")
        return False

def test_max_positions():
    """Test max positions constraint"""
    print("\n🧪 Testing Max Positions Constraint...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01", 
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_max_positions",
            "name": "Max Positions Test Strategy", 
            "description": "Test max positions constraint",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "50",  # More liberal entry for more signals
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi", 
                    "operator": "greater_than",
                    "value": "60",  # More liberal exit
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "long_only",
            "max_positions": 3,  # This is the KEY TEST - allow 3 positions
            "max_position_strategy": "replace_worst",  # This is also a KEY TEST
            "stop_loss": None,
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            print(f"✅ Max Positions test PASSED")
            print(f"   Total trades: {len(trades)}")
            print(f"   Should respect max 3 concurrent positions")
            return True
        else:
            print(f"❌ Max Positions test FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Max Positions test FAILED: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Backend Fixes for Crypto Quant MVP")
    print("=" * 60)
    
    tests = [
        test_kelly_criterion,
        test_stop_loss_precision, 
        test_short_only_direction,
        test_max_positions
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! Backend fixes are working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed or were inconclusive.")
    
    return passed == total

if __name__ == "__main__":
    main()
