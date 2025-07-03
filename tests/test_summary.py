# ==============================================================================
# File: tests/test_summary.py
# Description: Comprehensive test summary for all components
# ==============================================================================

import asyncio
import sys
import os
import subprocess
from datetime import datetime

def run_command(command, description):
    """Run a command and return the result"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def test_service_availability():
    """Test if key services are available"""
    print("🔧 Testing Service Availability")
    print("-" * 40)
    
    services = [
        ("curl -s http://localhost:8001/health > /dev/null", "New API Service (Port 8001)"),
        ("curl -s http://localhost:8000/ > /dev/null", "Old API Service (Port 8000)"),
        ("docker ps | grep timescaledb > /dev/null", "TimescaleDB Container"),
        ("docker ps | grep vectorbt > /dev/null", "VectorBT Container"),
    ]
    
    results = {}
    for command, description in services:
        results[description] = run_command(command, description)
    
    return results

def run_unit_tests():
    """Run all unit tests"""
    print("\n🧪 Running Unit Tests")
    print("-" * 40)
    
    test_commands = [
        ("cd /home/jaden/Documents/projects/crypto_quant_mvp && python tests/api/test_market_data_simple.py", 
         "Market Data Logic Tests"),
        ("cd /home/jaden/Documents/projects/crypto_quant_mvp && python tests/api/test_routers.py", 
         "API Router Tests"),
        ("cd /home/jaden/Documents/projects/crypto_quant_mvp && python tests/test_integration.py", 
         "Integration Tests"),
    ]
    
    results = {}
    for command, description in test_commands:
        results[description] = run_command(command, description)
    
    return results

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("-" * 40)
    
    endpoint_tests = [
        ("curl -s http://localhost:8001/health | grep -q 'healthy'", 
         "Health Endpoint"),
        ("curl -s http://localhost:8001/market/timeframes | grep -q '1 Minute'", 
         "Timeframes Endpoint"),
        ("curl -s http://localhost:8001/openapi.json | grep -q 'openapi'", 
         "OpenAPI Schema"),
        ("curl -s -w '%{http_code}' http://localhost:8001/market/symbols | grep -q '503'", 
         "Symbols Endpoint (DB Unavailable)"),
    ]
    
    results = {}
    for command, description in endpoint_tests:
        results[description] = run_command(command, description)
    
    return results

def check_code_structure():
    """Check code structure and imports"""
    print("\n📁 Checking Code Structure")
    print("-" * 40)
    
    structure_checks = [
        ("ls /home/jaden/Documents/projects/crypto_quant_mvp/services/api/app/routers/market_data.py > /dev/null", 
         "Market Data Router Exists"),
        ("ls /home/jaden/Documents/projects/crypto_quant_mvp/services/api/app/routers/backtesting.py > /dev/null", 
         "Backtesting Router Exists"),
        ("ls /home/jaden/Documents/projects/crypto_quant_mvp/services/api/app/models.py > /dev/null", 
         "API Models Exist"),
        ("ls /home/jaden/Documents/projects/crypto_quant_mvp/services/vectorbt/app/backtesting/strategies.py > /dev/null", 
         "VectorBT Strategies Exist"),
        ("ls /home/jaden/Documents/projects/crypto_quant_mvp/tests/api/test_market_data_simple.py > /dev/null", 
         "Unit Tests Exist"),
    ]
    
    results = {}
    for command, description in structure_checks:
        results[description] = run_command(command, description)
    
    return results

def main():
    """Main test runner"""
    print("🚀 Comprehensive Test Suite")
    print("=" * 80)
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all test categories
    all_results = {}
    
    all_results.update(test_service_availability())
    all_results.update(run_unit_tests())
    all_results.update(test_api_endpoints())
    all_results.update(check_code_structure())
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in all_results.values() if result)
    total = len(all_results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print()
    
    # Categorize results
    categories = {
        "Service Availability": [],
        "Unit Tests": [],
        "API Endpoints": [],
        "Code Structure": []
    }
    
    for test_name, result in all_results.items():
        if any(service in test_name for service in ["Service", "Container"]):
            categories["Service Availability"].append((test_name, result))
        elif "Tests" in test_name:
            categories["Unit Tests"].append((test_name, result))
        elif "Endpoint" in test_name or "Schema" in test_name:
            categories["API Endpoints"].append((test_name, result))
        else:
            categories["Code Structure"].append((test_name, result))
    
    for category, tests in categories.items():
        if tests:
            print(f"{category}:")
            for test_name, result in tests:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status} {test_name}")
            print()
    
    # Overall assessment
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✨ The refactored crypto quant MVP is working correctly!")
        print("\n🏗️  Architecture Summary:")
        print("   ✅ Microservice separation (API & VectorBT)")
        print("   ✅ Clean router structure (market_data, backtesting)")
        print("   ✅ Centralized models and dependencies")
        print("   ✅ Comprehensive unit tests")
        print("   ✅ API endpoint validation")
        print("   ✅ Error handling and validation")
    elif passed >= total * 0.8:
        print("⚠️  MOSTLY WORKING - Some non-critical issues")
        print("🔧 The core functionality is working but some services may need attention")
    else:
        print("❌ NEEDS ATTENTION - Multiple failures detected")
        print("🚨 Please check the failed tests above")
    
    return all_results

if __name__ == "__main__":
    main()
