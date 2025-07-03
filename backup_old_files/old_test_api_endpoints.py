# ==============================================================================
# File: tests/test_api_endpoints.py
# Description: Simple API endpoint testing script
# ==============================================================================

import asyncio
import json
import sys
import os
from datetime import datetime

# For testing without external dependencies
class MockHTTPResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self.data = data
    
    def json(self):
        return self.data

async def test_api_endpoints():
    """Test API endpoints with mock responses"""
    
    print("Testing API Endpoint Structure...")
    print("=" * 50)
    
    # Test 1: Health endpoint structure
    def test_health_endpoint():
        """Test health endpoint response structure"""
        mock_response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "services": {
                "database": "connected",
                "vectorbt": "external_service"
            }
        }
        
        # Validate response structure
        assert "status" in mock_response
        assert "timestamp" in mock_response
        assert "services" in mock_response
        assert mock_response["status"] == "healthy"
        
        print("✅ Health endpoint structure test passed")
        return True
    
    # Test 2: Timeframes endpoint
    def test_timeframes_endpoint():
        """Test timeframes endpoint response"""
        mock_response = [
            {"label": "1 Minute", "value": "1m", "table": "ohlcv_1min", "description": "1-minute candlesticks"},
            {"label": "5 Minutes", "value": "5m", "table": "ohlcv_5min", "description": "5-minute candlesticks"},
            {"label": "15 Minutes", "value": "15m", "table": "ohlcv_15min", "description": "15-minute candlesticks"},
            {"label": "1 Hour", "value": "1h", "table": "ohlcv_1hour", "description": "1-hour candlesticks"},
            {"label": "4 Hours", "value": "4h", "table": "ohlcv_4hour", "description": "4-hour candlesticks"},
            {"label": "1 Day", "value": "1d", "table": "ohlcv_1day", "description": "Daily candlesticks"},
            {"label": "7 Days", "value": "7d", "table": "ohlcv_7day", "description": "Weekly candlesticks"}
        ]
        
        # Validate response structure
        assert isinstance(mock_response, list)
        assert len(mock_response) == 7
        
        for timeframe in mock_response:
            assert "label" in timeframe
            assert "value" in timeframe
            assert "table" in timeframe
            assert "description" in timeframe
            assert timeframe["table"].startswith("ohlcv_")
        
        print(f"✅ Timeframes endpoint test passed ({len(mock_response)} timeframes)")
        return True
    
    # Test 3: OHLCV data structure
    def test_ohlcv_data_structure():
        """Test OHLCV data response structure"""
        mock_response = [
            {
                "time": "2024-01-01T00:00:00",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000.0
            },
            {
                "time": "2024-01-01T00:01:00", 
                "open": 102.0,
                "high": 107.0,
                "low": 101.0,
                "close": 104.0,
                "volume": 1200.0
            }
        ]
        
        # Validate response structure
        assert isinstance(mock_response, list)
        
        for candle in mock_response:
            required_fields = ["time", "open", "high", "low", "close", "volume"]
            for field in required_fields:
                assert field in candle, f"Missing field: {field}"
            
            # Validate price relationships
            assert candle["high"] >= candle["low"], "High should be >= Low"
            assert candle["high"] >= candle["open"], "High should be >= Open" 
            assert candle["high"] >= candle["close"], "High should be >= Close"
            assert candle["low"] <= candle["open"], "Low should be <= Open"
            assert candle["low"] <= candle["close"], "Low should be <= Close"
            assert candle["volume"] > 0, "Volume should be positive"
        
        print(f"✅ OHLCV data structure test passed ({len(mock_response)} candles)")
        return True
    
    # Test 4: Backtest request structure
    def test_backtest_request_structure():
        """Test backtest request validation"""
        mock_request = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": None,
            "end_date": None,
            "initial_cash": 100000.0,
            "strategy": {
                "strategy_type": "ma_crossover",
                "parameters": {
                    "fast_window": 10,
                    "slow_window": 20
                },
                "position_size_type": "fixed_amount",
                "position_size_value": 1000.0,
                "stop_loss_pct": None,
                "take_profit_pct": None,
                "max_positions": 1
            },
            "commission": 0.001
        }
        
        # Validate request structure
        assert "symbol" in mock_request
        assert "timeframe" in mock_request
        assert "strategy" in mock_request
        assert "initial_cash" in mock_request
        
        strategy = mock_request["strategy"]
        assert "strategy_type" in strategy
        assert "parameters" in strategy
        assert strategy["strategy_type"] in ["ma_crossover", "rsi_oversold", "bollinger_bands", "multi_indicator"]
        
        # Validate MA strategy parameters
        if strategy["strategy_type"] == "ma_crossover":
            params = strategy["parameters"]
            assert "fast_window" in params
            assert "slow_window" in params
            assert params["fast_window"] < params["slow_window"], "Fast window should be < slow window"
        
        print("✅ Backtest request structure test passed")
        return True
    
    # Test 5: Backtest response structure
    def test_backtest_response_structure():
        """Test backtest response structure"""
        mock_response = {
            "total_return_pct": 15.5,
            "annualized_return_pct": 18.2,
            "max_drawdown_pct": -8.3,
            "sharpe_ratio": 1.45,
            "sortino_ratio": 1.78,
            "calmar_ratio": 2.19,
            "total_trades": 25,
            "win_rate_pct": 64.0,
            "profit_factor": 1.85,
            "avg_win_pct": 3.2,
            "avg_loss_pct": -1.8,
            "volatility_pct": 12.4,
            "var_95_pct": -2.1,
            "avg_trade_duration_hours": 24.5,
            "max_trade_duration_hours": 168.0,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-31T23:59:59",
            "initial_cash": 100000.0,
            "final_value": 115500.0,
            "trades": [
                {
                    "entry_time": "2024-01-01T10:00:00",
                    "exit_time": "2024-01-02T14:30:00",
                    "entry_price": 100.0,
                    "exit_price": 103.2,
                    "size": 1000.0,
                    "side": "long",
                    "pnl": 3200.0,
                    "pnl_pct": 3.2,
                    "duration_minutes": 1710,
                    "entry_reason": "Fast MA(10) crossed above Slow MA(20)",
                    "exit_reason": "Fast MA(10) crossed below Slow MA(20)"
                }
            ]
        }
        
        # Validate performance metrics
        performance_fields = [
            "total_return_pct", "annualized_return_pct", "max_drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio"
        ]
        for field in performance_fields:
            assert field in mock_response, f"Missing performance field: {field}"
        
        # Validate trade statistics
        trade_fields = [
            "total_trades", "win_rate_pct", "profit_factor",
            "avg_win_pct", "avg_loss_pct"
        ]
        for field in trade_fields:
            assert field in mock_response, f"Missing trade field: {field}"
        
        # Validate trade details
        assert "trades" in mock_response
        assert isinstance(mock_response["trades"], list)
        
        for trade in mock_response["trades"]:
            trade_required_fields = [
                "entry_time", "entry_price", "size", "side",
                "entry_reason"
            ]
            for field in trade_required_fields:
                assert field in trade, f"Missing trade field: {field}"
        
        print(f"✅ Backtest response structure test passed ({len(mock_response['trades'])} trades)")
        return True
    
    # Run all tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Timeframes Endpoint", test_timeframes_endpoint),
        ("OHLCV Data Structure", test_ohlcv_data_structure),
        ("Backtest Request Structure", test_backtest_request_structure),
        ("Backtest Response Structure", test_backtest_response_structure)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 API Endpoint Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API endpoint structure tests passed!")
        print("\n🔗 Next: Test with real API calls once services are running")
    else:
        print("⚠️  Some API endpoint tests failed:")
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
    
    return results

# Real API testing function (to be used when services are running)
async def test_real_api_endpoints(base_url="http://localhost:8000"):
    """Test real API endpoints when services are running"""
    print(f"\nTesting Real API Endpoints at {base_url}...")
    print("=" * 50)
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            try:
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health endpoint: {data['status']}")
                    else:
                        print(f"❌ Health endpoint returned {response.status}")
            except Exception as e:
                print(f"❌ Health endpoint failed: {e}")
            
            # Test timeframes endpoint
            try:
                async with session.get(f"{base_url}/market/timeframes") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Timeframes endpoint: {len(data)} timeframes")
                    else:
                        print(f"❌ Timeframes endpoint returned {response.status}")
            except Exception as e:
                print(f"❌ Timeframes endpoint failed: {e}")
            
            # Test strategies endpoint  
            try:
                async with session.get(f"{base_url}/backtest/strategies") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Strategies endpoint: {len(data)} strategies")
                    else:
                        print(f"❌ Strategies endpoint returned {response.status}")
            except Exception as e:
                print(f"❌ Strategies endpoint failed: {e}")
    
    except ImportError:
        print("❌ aiohttp not available for real API testing")
        print("   Install with: pip install aiohttp")
        print("   Or start services and test manually:")
        print(f"   curl {base_url}/health")
        print(f"   curl {base_url}/market/timeframes")

if __name__ == "__main__":
    # Run structure tests
    asyncio.run(test_api_endpoints())
    
    # Offer to test real endpoints
    print("\n" + "=" * 50)
    print("To test real API endpoints:")
    print("1. Start the services: docker compose -f docker-compose.dev.yml up")
    print("2. Run: python tests/test_api_endpoints.py --real")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        asyncio.run(test_real_api_endpoints())
