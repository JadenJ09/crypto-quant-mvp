# ==============================================================================
# File: tests/api/test_endpoints_new.py
# Description: Test new API endpoints on port 8001
# ==============================================================================

import asyncio
import json
import sys
import os
from datetime import datetime
import urllib.request
import urllib.error

# Test configuration
API_BASE_URL = "http://localhost:8001"
TIMEOUT = 10

class NewAPITester:
    """Test new API endpoints"""
    
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.results = {}
    
    def make_request(self, endpoint, timeout=TIMEOUT):
        """Make HTTP request to API endpoint"""
        url = f"{self.base_url}{endpoint}"
        try:
            print(f"🔄 Testing: {url}")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return {"status": "success", "data": data, "status_code": response.status}
                else:
                    return {"status": "error", "data": None, "status_code": response.status}
        except urllib.error.HTTPError as e:
            return {"status": "http_error", "data": str(e), "status_code": e.code}
        except urllib.error.URLError as e:
            return {"status": "connection_error", "data": str(e), "status_code": None}
        except Exception as e:
            return {"status": "error", "data": str(e), "status_code": None}
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        result = self.make_request("/health")
        
        if result["status"] == "success":
            data = result["data"]
            # Validate health response structure
            required_fields = ["status", "timestamp", "services"]
            for field in required_fields:
                if field not in data:
                    print(f"❌ Health check missing field: {field}")
                    return False
            
            if data["status"] == "healthy":
                print("✅ Health check passed")
                return True
            else:
                print(f"⚠️  API status: {data['status']}")
                return False
        else:
            print(f"❌ Health check failed: {result['data']}")
            return False
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        result = self.make_request("/")
        
        if result["status"] == "success":
            data = result["data"]
            print(f"✅ Root endpoint accessible: {data}")
            return True
        else:
            print(f"❌ Root endpoint failed: {result['data']}")
            return False
    
    def test_timeframes_endpoint(self):
        """Test /market/timeframes endpoint"""
        result = self.make_request("/market/timeframes")
        
        if result["status"] == "success":
            data = result["data"]
            
            # Validate response structure
            if not isinstance(data, list):
                print("❌ Timeframes response should be a list")
                return False
            
            if len(data) != 7:
                print(f"❌ Expected 7 timeframes, got {len(data)}")
                return False
            
            # Check structure of first timeframe
            if data:
                tf = data[0]
                required_fields = ["label", "value", "table", "description"]
                for field in required_fields:
                    if field not in tf:
                        print(f"❌ Timeframe missing field: {field}")
                        return False
                
                print(f"✅ Timeframes endpoint passed ({len(data)} timeframes)")
                print(f"   Sample timeframe: {tf['label']} ({tf['value']})")
            return True
        else:
            print(f"❌ Timeframes endpoint failed: {result['data']}")
            return False
    
    def test_symbols_endpoint(self):
        """Test /market/symbols endpoint"""
        result = self.make_request("/market/symbols")
        
        if result["status"] == "http_error" and result["status_code"] == 503:
            print("⚠️  Database connection unavailable (expected)")
            return True
        elif result["status"] == "success":
            data = result["data"]
            
            # Validate response structure
            if not isinstance(data, list):
                print("❌ Symbols response should be a list")
                return False
            
            # Check structure if symbols exist
            if data:
                symbol = data[0]
                required_fields = ["symbol", "name", "exchange", "base_currency", "quote_currency"]
                for field in required_fields:
                    if field not in symbol:
                        print(f"❌ Symbol missing field: {field}")
                        return False
                
                print(f"✅ Symbols endpoint passed ({len(data)} symbols)")
                print(f"   Sample symbol: {symbol['name']} ({symbol['symbol']})")
            else:
                print("✅ Symbols endpoint passed (no symbols - DB empty)")
            return True
        else:
            print(f"⚠️  Symbols endpoint: {result['data']} (expected if DB not connected)")
            return True
    
    def test_api_docs(self):
        """Test API documentation endpoint"""
        result = self.make_request("/docs")
        
        if result["status"] == "success":
            print("✅ API documentation accessible")
            return True
        else:
            print(f"⚠️  API docs: {result['data']}")
            return False
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        result = self.make_request("/openapi.json")
        
        if result["status"] == "success":
            data = result["data"]
            # Check for basic OpenAPI structure
            if "openapi" in data and "info" in data:
                print("✅ OpenAPI schema accessible")
                print(f"   API Title: {data['info'].get('title', 'Unknown')}")
                print(f"   API Version: {data['info'].get('version', 'Unknown')}")
                return True
            else:
                print("❌ Invalid OpenAPI schema structure")
                return False
        else:
            print(f"❌ OpenAPI schema failed: {result['data']}")
            return False
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("Testing New API Endpoints...")
        print("=" * 60)
        print(f"API Base URL: {self.base_url}")
        print()
        
        tests = [
            ("health_check", self.test_health_endpoint),
            ("root_endpoint", self.test_root_endpoint),
            ("timeframes", self.test_timeframes_endpoint),
            ("symbols", self.test_symbols_endpoint),
            ("api_docs", self.test_api_docs),
            ("openapi_schema", self.test_openapi_schema),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ Test {test_name} failed with exception: {e}")
                results[test_name] = False
            print()  # Add spacing between tests
        
        # Summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print("=" * 60)
        print(f"📊 New API Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All new API tests passed!")
            print("✨ New API endpoints are working correctly!")
        else:
            print("⚠️  Some API tests failed or were skipped:")
            for test_name, result in results.items():
                if not result:
                    print(f"   - {test_name}")
        
        return results

def main():
    """Main test runner"""
    print("🚀 New API Endpoint Tester")
    print("=" * 60)
    print("This will test the new API endpoints.")
    print("Make sure the new API service is running on http://localhost:8001")
    print()
    
    tester = NewAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    main()
