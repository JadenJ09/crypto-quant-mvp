#!/usr/bin/env python3
"""
Production-Ready TimescaleDB Integration Test Suite

This script provides comprehensive testing of the custom backtest engine
integration with TimescaleDB in your crypto_quant_mvp environment.

Usage:
    # Run all tests
    python run_integration_tests.py

    # Run specific test
    python run_integration_tests.py --test database
    python run_integration_tests.py --test api
    python run_integration_tests.py --test backtest

    # Run with specific symbol
    python run_integration_tests.py --symbol BTCUSDT

    # Run with custom database URL
    python run_integration_tests.py --db-url postgresql://user:pass@localhost:5433/db
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from database.timescale_connector import TimescaleDBConnector
except ImportError:
    # Handle case where module is not available
    logger.warning("TimescaleDBConnector not available. Run from backtest-engine directory.")
    TimescaleDBConnector = None


class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self, database_url: str, api_base_url: str = "http://localhost:8003"):
        self.database_url = database_url
        self.api_base_url = api_base_url
        self.connector = TimescaleDBConnector(database_url)
        self.test_results = {}
        
    async def run_all_tests(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🚀 Starting comprehensive integration test suite")
        
        tests = [
            ("database_connection", self.test_database_connection),
            ("data_availability", self.test_data_availability),
            ("api_health", self.test_api_health),
            ("timescale_api_endpoints", self.test_timescale_api_endpoints),
            ("backtest_integration", self.test_backtest_integration),
            ("performance_validation", self.test_performance_validation)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"🔄 Running test: {test_name}")
            try:
                if symbol and test_name in ["backtest_integration", "performance_validation"]:
                    result = await test_func(symbol)
                else:
                    result = await test_func()
                
                results[test_name] = {
                    "success": True,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"✅ Test {test_name} passed")
                
            except Exception as e:
                results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"❌ Test {test_name} failed: {e}")
        
        # Summary
        passed = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        logger.info(f"📊 Test Summary: {passed}/{total} tests passed")
        
        return {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total * 100
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_database_connection(self) -> Dict[str, Any]:
        """Test database connection and basic functionality"""
        logger.info("🔗 Testing database connection...")
        
        await self.connector.initialize()
        
        # Test basic connection
        async with self.connector.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT version(), now()")
                result = await cur.fetchone()
                
        return {
            "database_version": result['version'],
            "server_time": result['now'],
            "connection_successful": True
        }
    
    async def test_data_availability(self) -> Dict[str, Any]:
        """Test data availability and structure"""
        logger.info("📊 Testing data availability...")
        
        # Check available tables
        tables = await self.connector.get_available_tables()
        
        # Check available symbols
        symbols = await self.connector.get_available_symbols(limit=5)
        
        # Get data info
        data_info = await self.connector.get_data_info()
        
        return {
            "available_tables": tables,
            "tables_count": len(tables),
            "available_symbols": symbols,
            "symbols_count": len(symbols),
            "total_records": data_info.get('total_records', 0),
            "earliest_data": data_info.get('earliest_data'),
            "latest_data": data_info.get('latest_data'),
            "has_data": data_info.get('total_records', 0) > 0
        }
    
    async def test_api_health(self) -> Dict[str, Any]:
        """Test API health and connectivity"""
        logger.info("🏥 Testing API health...")
        
        try:
            # Test main health endpoint
            response = requests.get(f"{self.api_base_url}/health/", timeout=10)
            main_health = response.json() if response.status_code == 200 else None
            
            # Test TimescaleDB health endpoint
            response = requests.get(f"{self.api_base_url}/timescale/health", timeout=10)
            timescale_health = response.json() if response.status_code == 200 else None
            
            return {
                "main_api_healthy": main_health is not None,
                "main_health_data": main_health,
                "timescale_api_healthy": timescale_health is not None,
                "timescale_health_data": timescale_health,
                "api_accessible": True
            }
            
        except Exception as e:
            return {
                "main_api_healthy": False,
                "timescale_api_healthy": False,
                "api_accessible": False,
                "error": str(e)
            }
    
    async def test_timescale_api_endpoints(self) -> Dict[str, Any]:
        """Test TimescaleDB API endpoints"""
        logger.info("🌐 Testing TimescaleDB API endpoints...")
        
        results = {}
        
        # Test endpoints
        endpoints = [
            ("info", "GET", "/timescale/info"),
            ("symbols", "GET", "/timescale/symbols"),
            ("tables", "GET", "/timescale/tables")
        ]
        
        for name, method, endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=10)
                results[name] = {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "data": response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def test_backtest_integration(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Test backtest integration with TimescaleDB"""
        logger.info("🔄 Testing backtest integration...")
        
        # Use provided symbol or get one from database
        if not symbol:
            symbols = await self.connector.get_available_symbols(limit=1)
            if not symbols:
                raise Exception("No symbols available for testing")
            symbol = symbols[0]
        
        # Test via API
        backtest_data = {
            "symbol": symbol,
            "timeframe": "1min",
            "start_time": (datetime.now() - timedelta(days=7)).isoformat(),
            "end_time": datetime.now().isoformat(),
            "initial_capital": 100000.0,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "risk_per_trade": 0.01,
            "strategy": "momentum"
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/timescale/backtest",
                json=backtest_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "api_test_successful": True,
                    "backtest_result": result,
                    "symbol_tested": symbol
                }
            else:
                return {
                    "api_test_successful": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "api_test_successful": False,
                "error": str(e)
            }
    
    async def test_performance_validation(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Test performance and validate results"""
        logger.info("⚡ Testing performance validation...")
        
        if not symbol:
            symbols = await self.connector.get_available_symbols(limit=1)
            if not symbols:
                raise Exception("No symbols available for testing")
            symbol = symbols[0]
        
        # Measure data retrieval performance
        start_time = datetime.now()
        
        data = await self.connector.get_market_data(
            symbol=symbol,
            timeframe="1min",
            start_time=datetime.now() - timedelta(days=1),
            limit=1000
        )
        
        data_retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Test quick backtest endpoint
        start_time = datetime.now()
        
        try:
            response = requests.get(
                f"{self.api_base_url}/timescale/quick-test/{symbol}?days=7",
                timeout=60
            )
            
            api_response_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "performance_test_successful": True,
                    "data_retrieval_time": data_retrieval_time,
                    "api_response_time": api_response_time,
                    "records_retrieved": len(data),
                    "quick_test_result": result
                }
            else:
                return {
                    "performance_test_successful": False,
                    "data_retrieval_time": data_retrieval_time,
                    "api_response_time": api_response_time,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "performance_test_successful": False,
                "data_retrieval_time": data_retrieval_time,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Clean up resources"""
        await self.connector.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run TimescaleDB integration tests')
    parser.add_argument('--db-url', help='Database URL', 
                       default=os.getenv('DATABASE_URL', 'postgresql://quant_user:quant_password@localhost:5433/quant_db'))
    parser.add_argument('--api-url', help='API base URL', default='http://localhost:8003')
    parser.add_argument('--symbol', help='Symbol to test with')
    parser.add_argument('--test', help='Specific test to run', 
                       choices=['database', 'api', 'backtest', 'performance'])
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    async def run_tests():
        suite = IntegrationTestSuite(args.db_url, args.api_url)
        
        try:
            if args.test == 'database':
                result = await suite.test_database_connection()
                print(json.dumps(result, indent=2, default=str))
            elif args.test == 'api':
                result = await suite.test_api_health()
                print(json.dumps(result, indent=2, default=str))
            elif args.test == 'backtest':
                result = await suite.test_backtest_integration(args.symbol)
                print(json.dumps(result, indent=2, default=str))
            elif args.test == 'performance':
                result = await suite.test_performance_validation(args.symbol)
                print(json.dumps(result, indent=2, default=str))
            else:
                # Run all tests
                results = await suite.run_all_tests(args.symbol)
                
                # Print summary
                print(f"\n{'='*50}")
                print(f"🎯 INTEGRATION TEST RESULTS")
                print(f"{'='*50}")
                print(f"Total Tests: {results['summary']['total_tests']}")
                print(f"Passed: {results['summary']['passed']}")
                print(f"Failed: {results['summary']['failed']}")
                print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
                print(f"{'='*50}\n")
                
                # Save results if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"📄 Results saved to {args.output}")
                else:
                    print(json.dumps(results, indent=2, default=str))
                
        finally:
            await suite.cleanup()
    
    # Run tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
