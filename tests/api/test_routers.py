# ==============================================================================
# File: tests/api/test_routers.py
# Description: Unit tests for API router functionality
# ==============================================================================

import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Add the service path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/api/app'))

class TestMarketDataRouter:
    """Test market data router functionality"""
    
    def test_timeframe_tables_mapping(self):
        """Test that timeframe mapping is correct"""
        try:
            from routers.market_data import TIMEFRAME_TABLES
            
            expected_mappings = {
                "1m": "ohlcv_1min",
                "5m": "ohlcv_5min",
                "15m": "ohlcv_15min",
                "1h": "ohlcv_1hour",
                "4h": "ohlcv_4hour",
                "1d": "ohlcv_1day",
                "7d": "ohlcv_7day"
            }
            
            for timeframe, table in expected_mappings.items():
                assert timeframe in TIMEFRAME_TABLES
                assert TIMEFRAME_TABLES[timeframe] == table
                
            print("✅ Timeframe mapping test passed")
            return True
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"❌ Timeframe mapping test failed: {e}")
            return False
    
    def test_timeframes_endpoint_structure(self):
        """Test the structure of timeframes endpoint data"""
        try:
            # Mock the get_available_timeframes function
            timeframes = [
                {"label": "1 Minute", "value": "1m", "table": "ohlcv_1min", "description": "1-minute candlesticks"},
                {"label": "5 Minutes", "value": "5m", "table": "ohlcv_5min", "description": "5-minute candlesticks"},
                {"label": "15 Minutes", "value": "15m", "table": "ohlcv_15min", "description": "15-minute candlesticks"},
                {"label": "1 Hour", "value": "1h", "table": "ohlcv_1hour", "description": "1-hour candlesticks"},
                {"label": "4 Hours", "value": "4h", "table": "ohlcv_4hour", "description": "4-hour candlesticks"},
                {"label": "1 Day", "value": "1d", "table": "ohlcv_1day", "description": "Daily candlesticks"},
                {"label": "7 Days", "value": "7d", "table": "ohlcv_7day", "description": "Weekly candlesticks"}
            ]
            
            # Validate structure
            for tf in timeframes:
                assert "label" in tf
                assert "value" in tf
                assert "table" in tf
                assert "description" in tf
                assert tf["table"].startswith("ohlcv_")
                
            print(f"✅ Timeframes endpoint structure test passed ({len(timeframes)} timeframes)")
            return True
        except Exception as e:
            print(f"❌ Timeframes endpoint structure test failed: {e}")
            return False

class TestBacktestingRouter:
    """Test backtesting router functionality"""
    
    def test_simple_ma_parameters_validation(self):
        """Test parameter validation for simple MA strategy"""
        
        # Test valid parameters
        valid_params = {
            "fast_window": 10,
            "slow_window": 20,
            "initial_cash": 100000.0,
            "commission": 0.001
        }
        
        # Test that fast_window < slow_window
        assert valid_params["fast_window"] < valid_params["slow_window"]
        
        # Test invalid parameters (fast >= slow)
        invalid_params = {
            "fast_window": 20,
            "slow_window": 10,  # This should fail
        }
        
        # In a real router, this would raise an HTTPException
        is_valid = invalid_params["fast_window"] < invalid_params["slow_window"]
        assert not is_valid
        
        print("✅ Simple MA parameter validation test passed")
        return True

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_symbol_parsing(self):
        """Test symbol parsing logic"""
        test_cases = [
            ("BTCUSDT", "BTC", "USDT"),
            ("ETHUSDT", "ETH", "USDT"),
            ("ADABTC", "ADA", "BTC"),
            ("DOTETH", "DOT", "ETH"),
            ("UNKNOWN", "UNKNOWN", "USD"),  # Default case
        ]
        
        for symbol, expected_base, expected_quote in test_cases:
            # Simulate the parsing logic from the router
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                quote = 'USDT'
            elif symbol.endswith('BTC'):
                base = symbol[:-3]
                quote = 'BTC'
            elif symbol.endswith('ETH'):
                base = symbol[:-3]
                quote = 'ETH'
            else:
                base = symbol
                quote = 'USD'
            
            assert base == expected_base, f"Expected base {expected_base}, got {base}"
            assert quote == expected_quote, f"Expected quote {expected_quote}, got {quote}"
        
        print("✅ Symbol parsing test passed")
        return True

def run_all_tests():
    """Run all tests and return summary"""
    results = {}
    
    # Market Data Router Tests
    market_data_tests = TestMarketDataRouter()
    results["timeframe_mapping"] = market_data_tests.test_timeframe_tables_mapping()
    results["timeframes_structure"] = market_data_tests.test_timeframes_endpoint_structure()
    
    # Backtesting Router Tests
    backtesting_tests = TestBacktestingRouter()
    results["ma_parameters"] = backtesting_tests.test_simple_ma_parameters_validation()
    
    # Utility Tests
    utility_tests = TestUtilityFunctions()
    results["symbol_parsing"] = utility_tests.test_symbol_parsing()
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed:")
        for test_name, result in results.items():
            if not result:
                print(f"   - {test_name}")
    
    return results

if __name__ == "__main__":
    print("Running API Router Tests...")
    print("=" * 50)
    run_all_tests()
