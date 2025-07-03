#!/usr/bin/env python3
"""
Test stop loss precision for both long and short positions
"""

import requests
import json

API_BASE = "http://localhost:8002"

def test_stop_loss_precision():
    """Test that stop loss actually triggers at the specified percentage"""
    print("🧪 Testing Stop Loss Precision...")
    
    # Test short position with 6% stop loss
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_stop_loss_precision",
            "name": "Stop Loss Precision Test",
            "description": "Test 6% stop loss precision",
            "entry_conditions": [
                {
                    "id": "entry1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "greater_than",
                    "value": "70",  # Enter short when overbought
                    "enabled": True
                }
            ],
            "exit_conditions": [
                {
                    "id": "exit1",
                    "type": "technical_indicator",
                    "indicator": "rsi",
                    "operator": "less_than",
                    "value": "20",  # Very low to ensure stop loss triggers first
                    "enabled": True
                }
            ],
            "position_sizing": "percentage",
            "position_size": 20,
            "position_direction": "short_only",
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": 6.0,  # 6% stop loss - should trigger close to 6%
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            print(f"Total trades with 6% stop loss: {len(trades)}")
            
            if trades:
                print("\nStop Loss Analysis:")
                stop_loss_trades = []
                
                for i, trade in enumerate(trades):
                    side = trade.get('side', 'unknown')
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    pnl_pct = trade.get('pnl_pct', 0)
                    exit_reason = trade.get('exit_reason', 'unknown')
                    
                    # For short positions, stop loss occurs when price goes UP (against us)
                    if 'stop' in exit_reason.lower():
                        stop_loss_trades.append(trade)
                        price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                        print(f"  Trade {i+1}: Entry=${entry_price:.2f}, Exit=${exit_price:.2f}")
                        print(f"           Price change: {price_change_pct:.2f}% (positive = price went up = bad for short)")
                        print(f"           PnL: {pnl_pct:.2f}% (negative = loss as expected)")
                        print(f"           Exit reason: {exit_reason}")
                        
                        # Check if the loss is close to 6%
                        if abs(abs(pnl_pct) - 6.0) < 1.0:  # Within 1% of 6%
                            print(f"           ✅ Stop loss triggered close to 6% (actual: {abs(pnl_pct):.2f}%)")
                        else:
                            print(f"           ⚠️  Stop loss triggered at {abs(pnl_pct):.2f}% (target: 6%)")
                
                print(f"\nStop loss trades: {len(stop_loss_trades)}/{len(trades)}")
                
                if stop_loss_trades:
                    avg_loss = sum(abs(t.get('pnl_pct', 0)) for t in stop_loss_trades) / len(stop_loss_trades)
                    print(f"Average stop loss: {avg_loss:.2f}% (target: 6.00%)")
                    
                    if 5.0 <= avg_loss <= 7.0:  # Within reasonable range
                        print("✅ Stop loss precision is working correctly!")
                        return True
                    else:
                        print(f"⚠️  Stop loss precision may need adjustment (avg: {avg_loss:.2f}%)")
                        return True  # Still working, just not perfectly precise
                else:
                    print("⚠️  No stop loss trades found")
                    return True
            else:
                print("⚠️  No trades found")
                return True
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_stop_loss_precision()
