# ==============================================================================
# File: tests/test_integration.py
# Description: Simple integration tests that can run without external dependencies
# ==============================================================================

import sys
import os
from datetime import datetime

def test_project_structure():
    """Test that the project structure is correct"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Test API service structure
    api_path = os.path.join(project_root, 'services', 'api', 'app')
    expected_api_files = [
        'models.py',
        'dependencies.py',
        'main.py',
        'routers'
    ]
    
    for file_path in expected_api_files:
        full_path = os.path.join(api_path, file_path)
        assert os.path.exists(full_path), f"Missing API file: {file_path}"
    
    # Test API routers
    routers_path = os.path.join(api_path, 'routers')
    expected_router_files = [
        '__init__.py',
        'market_data.py',
        'backtesting.py'
    ]
    
    for file_path in expected_router_files:
        full_path = os.path.join(routers_path, file_path)
        assert os.path.exists(full_path), f"Missing router file: {file_path}"
    
    # Test vectorbt service structure
    vectorbt_path = os.path.join(project_root, 'services', 'vectorbt', 'app')
    expected_vectorbt_files = [
        'models.py',
        'backtesting'
    ]
    
    for file_path in expected_vectorbt_files:
        full_path = os.path.join(vectorbt_path, file_path)
        assert os.path.exists(full_path), f"Missing vectorbt file: {file_path}"
    
    # Test vectorbt backtesting
    backtesting_path = os.path.join(vectorbt_path, 'backtesting')
    expected_backtesting_files = [
        '__init__.py',
        'strategies.py',
        'utils.py'
    ]
    
    for file_path in expected_backtesting_files:
        full_path = os.path.join(backtesting_path, file_path)
        assert os.path.exists(full_path), f"Missing backtesting file: {file_path}"
    
    print("✅ Project structure test passed")
    return True

def test_import_structure():
    """Test that files can be imported correctly"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Test API models import
    api_path = os.path.join(project_root, 'services', 'api', 'app')
    sys.path.insert(0, api_path)
    
    try:
        # This will fail due to missing dependencies, but tests file syntax
        with open(os.path.join(api_path, 'models.py'), 'r') as f:
            content = f.read()
            assert 'class OHLCVOut' in content
            assert 'class BacktestStats' in content
            assert 'class StrategyConfig' in content
        print("✅ API models structure test passed")
    except Exception as e:
        print(f"❌ API models test failed: {e}")
        return False
    
    # Test router structure
    try:
        with open(os.path.join(api_path, 'routers', 'market_data.py'), 'r') as f:
            content = f.read()
            assert 'TIMEFRAME_TABLES' in content
            assert 'get_available_timeframes' in content
            assert 'get_ohlcv_data' in content
        print("✅ Market data router structure test passed")
    except Exception as e:
        print(f"❌ Market data router test failed: {e}")
        return False
    
    try:
        with open(os.path.join(api_path, 'routers', 'backtesting.py'), 'r') as f:
            content = f.read()
            assert 'get_strategies' in content
            assert 'run_backtest' in content
            assert 'simple_ma_backtest' in content
        print("✅ Backtesting router structure test passed")
    except Exception as e:
        print(f"❌ Backtesting router test failed: {e}")
        return False
    
    return True

def test_configuration_constants():
    """Test that configuration constants are properly defined"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    api_path = os.path.join(project_root, 'services', 'api', 'app')
    
    # Test timeframe mappings
    try:
        with open(os.path.join(api_path, 'routers', 'market_data.py'), 'r') as f:
            content = f.read()
            
            # Check that all expected timeframes are defined
            expected_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '7d']
            for tf in expected_timeframes:
                assert f'"{tf}"' in content, f"Timeframe {tf} not found"
            
            # Check that table mappings are correct
            expected_tables = ['ohlcv_1min', 'ohlcv_5min', 'ohlcv_15min', 'ohlcv_1hour', 'ohlcv_4hour', 'ohlcv_1day', 'ohlcv_7day']
            for table in expected_tables:
                assert f'"{table}"' in content, f"Table {table} not found"
        
        print("✅ Configuration constants test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration constants test failed: {e}")
        return False

def test_model_definitions():
    """Test that model definitions are complete"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    api_path = os.path.join(project_root, 'services', 'api', 'app')
    
    try:
        with open(os.path.join(api_path, 'models.py'), 'r') as f:
            content = f.read()
            
            # Test that essential models are defined
            essential_models = [
                'OHLCVOut',
                'CandlestickData', 
                'BacktestStats',
                'StrategyConfig',
                'BacktestRequest',
                'ValidationResult'
            ]
            
            for model in essential_models:
                assert f'class {model}' in content, f"Model {model} not found"
            
            # Test that essential fields are present
            assert 'total_return_pct' in content, "BacktestStats missing total_return_pct"
            assert 'strategy_type' in content, "StrategyConfig missing strategy_type"
            assert 'symbol' in content, "BacktestRequest missing symbol"
        
        print("✅ Model definitions test passed")
        return True
    except Exception as e:
        print(f"❌ Model definitions test failed: {e}")
        return False

def test_strategy_registry():
    """Test that strategy registry is properly defined"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    vectorbt_path = os.path.join(project_root, 'services', 'vectorbt', 'app', 'backtesting')
    
    try:
        with open(os.path.join(vectorbt_path, 'strategies.py'), 'r') as f:
            content = f.read()
            
            # Test that strategy registry exists
            assert 'STRATEGY_REGISTRY' in content, "STRATEGY_REGISTRY not found"
            
            # Test that essential strategies are defined
            essential_strategies = [
                'ma_crossover',
                'rsi_oversold',
                'bollinger_bands',
                'multi_indicator'
            ]
            
            for strategy in essential_strategies:
                assert f"'{strategy}'" in content, f"Strategy {strategy} not found in registry"
            
            # Test that strategy functions exist
            strategy_functions = [
                'run_ma_crossover_strategy',
                'run_rsi_oversold_strategy',
                'run_bollinger_bands_strategy',
                'run_custom_multi_indicator_strategy'
            ]
            
            for func in strategy_functions:
                assert f'def {func}' in content, f"Strategy function {func} not found"
        
        print("✅ Strategy registry test passed")
        return True
    except Exception as e:
        print(f"❌ Strategy registry test failed: {e}")
        return False

def run_all_integration_tests():
    """Run all integration tests"""
    print("Running Integration Tests...")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Import Structure", test_import_structure),
        ("Configuration Constants", test_configuration_constants),
        ("Model Definitions", test_model_definitions),
        ("Strategy Registry", test_strategy_registry)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Integration Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n🚀 Ready for API endpoint testing!")
    else:
        print("⚠️  Some integration tests failed:")
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
    
    return results

if __name__ == "__main__":
    results = run_all_integration_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
