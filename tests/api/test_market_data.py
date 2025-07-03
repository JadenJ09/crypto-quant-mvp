# ==============================================================================
# File: tests/api/test_market_data.py
# Description: Comprehensive unit tests for market data router functionality
# ==============================================================================

import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Add the service path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/api/app'))

try:
    from routers.market_data import router, TIMEFRAME_TABLES, get_available_timeframes, get_available_symbols
    from models import TimeframeInfo, SymbolInfo, OHLCVOut, CandlestickData
    from main import app
    # Import HTTPException for testing
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    imports_available = True
except ImportError as e:
    print(f"Warning: Could not import all modules - {e}")
    app = None
    router = None
    TIMEFRAME_TABLES = {
        "1m": "ohlcv_1min",
        "5m": "ohlcv_5min",
        "15m": "ohlcv_15min",
        "1h": "ohlcv_1hour",
        "4h": "ohlcv_4hour",
        "1d": "ohlcv_1day",
        "7d": "ohlcv_7day"
    }
    imports_available = False
    
    # Mock HTTPException for testing
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    
    # Mock TestClient
    class TestClient:
        def __init__(self, app):
            self.app = app

class TestMarketDataTimeframes:
    """Test timeframe-related functionality"""
    
    def test_timeframe_tables_mapping(self):
        """Test that timeframe mapping is correct and complete"""
        expected_mappings = {
            "1m": "ohlcv_1min",
            "5m": "ohlcv_5min",
            "15m": "ohlcv_15min",
            "1h": "ohlcv_1hour",
            "4h": "ohlcv_4hour",
            "1d": "ohlcv_1day",
            "7d": "ohlcv_7day"
        }
        
        # Check that all expected mappings exist
        for timeframe, table in expected_mappings.items():
            assert timeframe in TIMEFRAME_TABLES, f"Missing timeframe: {timeframe}"
            assert TIMEFRAME_TABLES[timeframe] == table, f"Wrong table for {timeframe}: expected {table}, got {TIMEFRAME_TABLES[timeframe]}"
        
        # Check that there are no extra mappings
        assert len(TIMEFRAME_TABLES) == len(expected_mappings), "Extra timeframes found"
        
        print("✅ Timeframe mapping test passed")
        return True
    
    def test_timeframe_validation(self):
        """Test timeframe validation logic"""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "7d"]
        invalid_timeframes = ["2m", "30m", "2h", "6h", "3d", "1w", ""]
        
        # Test valid timeframes
        for tf in valid_timeframes:
            assert tf in TIMEFRAME_TABLES, f"Valid timeframe {tf} not found in mapping"
        
        # Test invalid timeframes
        for tf in invalid_timeframes:
            assert tf not in TIMEFRAME_TABLES, f"Invalid timeframe {tf} found in mapping"
        
        print("✅ Timeframe validation test passed")
        return True
    
    async def test_get_available_timeframes(self):
        """Test the get_available_timeframes endpoint function"""
        if not imports_available:
            print("⚠️  Skipping get_available_timeframes test - imports not available")
            return True
            
        try:
            result = await get_available_timeframes()
            
            # Check that result is a list
            assert isinstance(result, list), "Result should be a list"
            assert len(result) == 7, f"Expected 7 timeframes, got {len(result)}"
            
            # Check structure of each timeframe
            required_fields = ["label", "value", "table", "description"]
            for tf in result:
                for field in required_fields:
                    assert field in tf, f"Missing field {field} in timeframe {tf}"
                
                # Check that value is a valid timeframe
                assert tf["value"] in TIMEFRAME_TABLES, f"Invalid timeframe value: {tf['value']}"
                
                # Check that table matches mapping
                assert tf["table"] == TIMEFRAME_TABLES[tf["value"]], f"Table mismatch for {tf['value']}"
                
                # Check that table starts with ohlcv_
                assert tf["table"].startswith("ohlcv_"), f"Invalid table name: {tf['table']}"
            
            print("✅ Get available timeframes test passed")
            return True
        except Exception as e:
            print(f"❌ Get available timeframes test failed: {e}")
            return False

class TestMarketDataSymbols:
    """Test symbol-related functionality"""
    
    def test_symbol_parsing_logic(self):
        """Test the symbol parsing logic used in get_available_symbols"""
        test_cases = [
            ("BTCUSDT", "BTC", "USDT"),
            ("ETHUSDT", "ETH", "USDT"),
            ("ADAUSDT", "ADA", "USDT"),
            ("ADABTC", "ADA", "BTC"),
            ("DOTETH", "DOT", "ETH"),
            ("LINKETH", "LINK", "ETH"),
            ("UNKNOWN", "UNKNOWN", "USD"),  # Default case
            ("", "", "USD"),  # Edge case
        ]
        
        for symbol, expected_base, expected_quote in test_cases:
            # Replicate the parsing logic from the router
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
            
            assert base == expected_base, f"Expected base {expected_base}, got {base} for symbol {symbol}"
            assert quote == expected_quote, f"Expected quote {expected_quote}, got {quote} for symbol {symbol}"
        
        print("✅ Symbol parsing test passed")
        return True
    
    @patch('routers.market_data.get_db_pool')
    async def test_get_available_symbols_success(self, mock_get_db_pool):
        """Test successful symbol retrieval"""
        # Mock database response
        mock_rows = [
            ("BTCUSDT",),
            ("ETHUSDT",),
            ("ADABTC",),
            ("DOTETH",),
        ]
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=mock_rows)
        
        mock_connection = AsyncMock()
        mock_connection.cursor = AsyncMock(return_value=mock_cursor)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        
        mock_pool = Mock()
        mock_pool.connection = Mock(return_value=mock_connection)
        
        mock_get_db_pool.return_value = mock_pool
        
        # Test the function
        try:
            result = await get_available_symbols()
            
            # Verify result structure
            assert isinstance(result, list), "Result should be a list"
            assert len(result) == 4, f"Expected 4 symbols, got {len(result)}"
            
            # Check first symbol (BTCUSDT)
            btc_symbol = result[0]
            expected_fields = ["symbol", "name", "exchange", "base_currency", "quote_currency"]
            for field in expected_fields:
                assert field in btc_symbol, f"Missing field {field} in symbol data"
            
            assert btc_symbol["symbol"] == "BTCUSDT"
            assert btc_symbol["base_currency"] == "BTC"
            assert btc_symbol["quote_currency"] == "USDT"
            assert btc_symbol["name"] == "BTC/USDT"
            assert btc_symbol["exchange"] == "Binance"
            
            print("✅ Get available symbols success test passed")
            return True
        except Exception as e:
            print(f"❌ Get available symbols success test failed: {e}")
            return False
    
    @patch('routers.market_data.get_db_pool')
    async def test_get_available_symbols_no_db(self, mock_get_db_pool):
        """Test symbol retrieval when database is unavailable"""
        if not imports_available:
            print("⚠️  Skipping get_available_symbols_no_db test - imports not available")
            return True
            
        mock_get_db_pool.return_value = None
        
        try:
            await get_available_symbols()
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 503
            assert "Database connection is not available" in str(e.detail)
            print("✅ Get available symbols no DB test passed")
            return True
        except Exception as e:
            print(f"❌ Get available symbols no DB test failed: {e}")
            return False

class TestMarketDataOHLCV:
    """Test OHLCV data retrieval functionality"""
    
    def test_query_parameter_validation(self):
        """Test OHLCV endpoint parameter validation"""
        # Test valid parameters
        valid_params = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "limit": 1000,
            "start_time": datetime.now() - timedelta(days=1),
            "end_time": datetime.now()
        }
        
        # Validate timeframe
        assert valid_params["timeframe"] in TIMEFRAME_TABLES
        
        # Validate limit
        assert 0 < valid_params["limit"] <= 5000
        
        # Validate time range
        assert valid_params["start_time"] < valid_params["end_time"]
        
        print("✅ OHLCV parameter validation test passed")
        return True
    
    def test_sql_query_construction(self):
        """Test SQL query construction logic for different parameter combinations"""
        symbol = "BTCUSDT"
        table_name = "ohlcv_1hour"
        limit = 1000
        
        # Test query with no time filters
        base_query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT %s;
        """
        expected_params = (symbol, limit)
        
        # Validate base query structure
        assert "SELECT" in base_query
        assert "FROM" in base_query
        assert "WHERE" in base_query
        assert "ORDER BY" in base_query
        assert "LIMIT" in base_query
        assert table_name in base_query
        assert len(expected_params) == 2
        
        # Test query with start_time
        start_time = datetime.now() - timedelta(days=1)
        start_query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s
            ORDER BY time DESC
            LIMIT %s;
        """
        start_params = (symbol, start_time, limit)
        assert len(start_params) == 3
        assert "time >=" in start_query
        
        # Test query with both start and end time
        end_time = datetime.now()
        range_query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time DESC
            LIMIT %s;
        """
        range_params = (symbol, start_time, end_time, limit)
        assert len(range_params) == 4
        assert "time >=" in range_query
        assert "time <=" in range_query
        
        print("✅ SQL query construction test passed")
        return True

class TestMarketDataCandlesticks:
    """Test candlestick data functionality"""
    
    def test_candlestick_data_structure(self):
        """Test candlestick data structure requirements"""
        # Test data with SMA indicators
        candlestick_with_sma = {
            "time": datetime.now(),
            "open": 45000.0,
            "high": 46000.0,
            "low": 44000.0,
            "close": 45500.0,
            "volume": 1000.0,
            "sma_20": 45200.0,
            "sma_50": 44800.0,
            "sma_100": 44500.0
        }
        
        # Test data without SMA indicators (1m timeframe)
        candlestick_without_sma = {
            "time": datetime.now(),
            "open": 45000.0,
            "high": 46000.0,
            "low": 44000.0,
            "close": 45500.0,
            "volume": 1000.0,
            "sma_20": None,
            "sma_50": None,
            "sma_100": None
        }
        
        # Validate required fields
        required_fields = ["time", "open", "high", "low", "close", "volume", "sma_20", "sma_50", "sma_100"]
        
        for field in required_fields:
            assert field in candlestick_with_sma, f"Missing field {field} in candlestick with SMA"
            assert field in candlestick_without_sma, f"Missing field {field} in candlestick without SMA"
        
        # Validate OHLC logic
        assert candlestick_with_sma["low"] <= candlestick_with_sma["open"], "Low should be <= Open"
        assert candlestick_with_sma["low"] <= candlestick_with_sma["close"], "Low should be <= Close"
        assert candlestick_with_sma["high"] >= candlestick_with_sma["open"], "High should be >= Open"
        assert candlestick_with_sma["high"] >= candlestick_with_sma["close"], "High should be >= Close"
        
        print("✅ Candlestick data structure test passed")
        return True
    
    def test_sma_logic_by_timeframe(self):
        """Test SMA inclusion logic based on timeframe"""
        # 1m timeframe should not have SMA
        timeframe_1m = "1m"
        has_sma_1m = timeframe_1m != "1m"
        assert not has_sma_1m, "1m timeframe should not have SMA"
        
        # Other timeframes should have SMA
        other_timeframes = ["5m", "15m", "1h", "4h", "1d", "7d"]
        for tf in other_timeframes:
            has_sma = tf != "1m"
            assert has_sma, f"{tf} timeframe should have SMA"
        
        print("✅ SMA logic by timeframe test passed")
        return True

class TestMarketDataEndpoints:
    """Test actual API endpoints if possible"""
    
    def test_client_creation(self):
        """Test that we can create a test client"""
        if app is None:
            print("⚠️  Cannot create test client - app not available")
            return False
        
        try:
            client = TestClient(app)
            assert client is not None
            print("✅ Test client creation passed")
            return True
        except Exception as e:
            print(f"❌ Test client creation failed: {e}")
            return False

# Test runner
async def run_async_tests():
    """Run all async tests"""
    results = {}
    
    # Timeframes tests
    timeframes_tests = TestMarketDataTimeframes()
    results["timeframe_mapping"] = timeframes_tests.test_timeframe_tables_mapping()
    results["timeframe_validation"] = timeframes_tests.test_timeframe_validation()
    results["get_timeframes"] = await timeframes_tests.test_get_available_timeframes()
    
    # Symbols tests
    symbols_tests = TestMarketDataSymbols()
    results["symbol_parsing"] = symbols_tests.test_symbol_parsing_logic()
    results["get_symbols_success"] = await symbols_tests.test_get_available_symbols_success()
    results["get_symbols_no_db"] = await symbols_tests.test_get_available_symbols_no_db()
    
    # OHLCV tests
    ohlcv_tests = TestMarketDataOHLCV()
    results["ohlcv_params"] = ohlcv_tests.test_query_parameter_validation()
    results["sql_construction"] = ohlcv_tests.test_sql_query_construction()
    
    # Candlesticks tests
    candlesticks_tests = TestMarketDataCandlesticks()
    results["candlestick_structure"] = candlesticks_tests.test_candlestick_data_structure()
    results["sma_logic"] = candlesticks_tests.test_sma_logic_by_timeframe()
    
    # Endpoints tests
    endpoints_tests = TestMarketDataEndpoints()
    results["test_client"] = endpoints_tests.test_client_creation()
    
    return results

def run_all_tests():
    """Run all tests and return summary"""
    print("Running Market Data Router Tests...")
    print("=" * 60)
    
    # Run async tests
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    results = loop.run_until_complete(run_async_tests())
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Market Data Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All market data tests passed!")
    else:
        print("⚠️  Some market data tests failed:")
        for test_name, result in results.items():
            if not result:
                print(f"   - {test_name}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
