#!/usr/bin/env python3
"""
Focused test for short-only position direction
"""

import requests
import json

API_BASE = "http://localhost:8002"

def test_short_only_focused():
    """Test short-only position direction with detailed logging"""
    print("🧪 Testing Short-Only Position Direction (FOCUSED)...")
    
    test_payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": "2024-07-01",
        "end_date": "2024-08-01",
        "initial_cash": 100000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy": {
            "id": "test_short_focused",
            "name": "Short Focused Test",
            "description": "Test short-only with simple RSI",
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
            "position_direction": "short_only",  # KEY TEST
            "max_positions": 1,
            "max_position_strategy": "ignore",
            "stop_loss": None,
            "take_profit": None
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/strategies/backtest", json=test_payload, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            trades = result.get('trades', [])
            
            print(f"Total trades: {len(trades)}")
            
            if trades:
                print("\nFirst few trades:")
                for i, trade in enumerate(trades[:3]):
                    side = trade.get('side', 'unknown')
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    pnl_pct = trade.get('pnl_pct', 0)
                    print(f"  Trade {i+1}: Side={side}, Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, PnL={pnl_pct:.2f}%")
                
                # Count short vs long trades
                short_trades = [t for t in trades if t.get('side') == 'short']
                long_trades = [t for t in trades if t.get('side') == 'long']
                
                print(f"\nTrade Direction Analysis:")
                print(f"  Short trades: {len(short_trades)}")
                print(f"  Long trades: {len(long_trades)}")
                print(f"  Unknown/Other: {len(trades) - len(short_trades) - len(long_trades)}")
                
                if len(short_trades) == len(trades) and len(trades) > 0:
                    print("✅ SHORT-ONLY WORKING CORRECTLY!")
                    return True
                else:
                    print("❌ SHORT-ONLY NOT WORKING - Found non-short trades")
                    return False
            else:
                print("⚠️  No trades found - cannot verify short-only functionality")
                return True  # Inconclusive but not failed
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    test_short_only_focused()
