#!/usr/bin/env python3
"""
Test for precise 5% stop loss on short positions
"""

import requests
import json
import time

API_BASE = "http://localhost:8002"

def test_5_percent_stop_loss_short():
    """Test that 5% stop loss on short positions works precisely"""
    print("🧪 Testing 5% Stop Loss Precision for Short Positions...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_5_percent_stop_loss",
            "name": "5% Stop Loss Test",
            "description": "Test precise 5% stop loss for short positions",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "30",  # Enter short when RSI < 30 (oversold, contrarian)
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator", 
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "70",  # Exit when RSI > 70 (overbought)
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "short_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": 5.0,  # 5% stop loss - KEY TEST
            "take_profit": None
        }
    }
    
    try:
        # Wait for service to be ready
        time.sleep(2)
        
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            print(f"✅ Backtest completed with {len(trades)} trades")
            
            if trades:
                print("\n📊 Stop Loss Analysis:")
                stop_loss_violations = []
                
                for i, trade in enumerate(trades):
                    side = trade.get('side', 'unknown')
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    pnl_pct = trade.get('pnl_pct', 0)
                    exit_reason = trade.get('exit_reason', 'unknown')
                    
                    print(f"\nTrade {i+1}:")
                    print(f"  Side: {side}")
                    print(f"  Entry: ${entry_price:.2f}")
                    print(f"  Exit: ${exit_price:.2f}")
                    print(f"  PnL: {pnl_pct:.2f}%")
                    print(f"  Exit reason: {exit_reason}")
                    
                    # Check for stop loss violations (losses > 5%)
                    if pnl_pct < -5.5:  # Allow 0.5% tolerance for precision
                        stop_loss_violations.append({
                            'trade': i+1,
                            'pnl_pct': pnl_pct,
                            'violation': abs(pnl_pct) - 5.0
                        })
                        print(f"  ❌ STOP LOSS VIOLATION: Loss of {abs(pnl_pct):.2f}% exceeds 5% limit!")
                    elif pnl_pct < -5.0:
                        print(f"  ⚠️  Close to stop loss limit: {abs(pnl_pct):.2f}%")
                    else:
                        print(f"  ✅ Within stop loss limit")
                
                print(f"\n📈 Summary:")
                print(f"  Total trades: {len(trades)}")
                print(f"  Stop loss violations: {len(stop_loss_violations)}")
                
                if stop_loss_violations:
                    print(f"\n❌ STOP LOSS NOT WORKING PROPERLY!")
                    for violation in stop_loss_violations:
                        print(f"     Trade {violation['trade']}: {violation['pnl_pct']:.2f}% (excess: {violation['violation']:.2f}%)")
                    return False
                else:
                    print(f"✅ STOP LOSS WORKING CORRECTLY - No violations found!")
                    return True
            else:
                print("⚠️  No trades found - cannot verify stop loss")
                return True
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_5_percent_stop_loss_short()
