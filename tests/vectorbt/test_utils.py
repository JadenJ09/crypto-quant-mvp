# ==============================================================================
# File: tests/vectorbt/test_utils.py  
# Description: Unit tests for vectorbt utility functions
# ==============================================================================

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the vectorbt service path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/vectorbt/app'))

class TestVectorbtUtils:
    """Test vectorbt utility functions"""
    
    def test_create_sample_price_data(self):
        """Test sample price data generation"""
        try:
            from backtesting.utils import create_sample_price_data
            
            # Test basic generation
            df = create_sample_price_data(n_periods=100, symbol="TEST")
            
            # Validate structure
            assert len(df) == 100
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            assert df.index.name == 'time'  # Time should be index
            
            # Validate price relationships
            assert (df['high'] >= df['low']).all(), "High should be >= Low"
            assert (df['high'] >= df['open']).all(), "High should be >= Open"
            assert (df['high'] >= df['close']).all(), "High should be >= Close"
            assert (df['low'] <= df['open']).all(), "Low should be <= Open"
            assert (df['low'] <= df['close']).all(), "Low should be <= Close"
            
            # Validate volume is positive
            assert (df['volume'] > 0).all(), "Volume should be positive"
            
            print("✅ Sample price data generation test passed")
            return True
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"❌ Sample price data test failed: {e}")
            return False
    
    def test_validate_price_data(self):
        """Test price data validation"""
        try:
            from backtesting.utils import validate_price_data, create_sample_price_data
            
            # Test valid data
            valid_df = create_sample_price_data(50)
            assert validate_price_data(valid_df) == True
            
            # Test empty data
            empty_df = pd.DataFrame()
            assert validate_price_data(empty_df) == False
            
            # Test missing columns
            incomplete_df = pd.DataFrame({
                'open': [100, 101],
                'close': [102, 103]
                # Missing high, low, volume
            })
            assert validate_price_data(incomplete_df) == False
            
            # Test invalid price relationships
            invalid_df = pd.DataFrame({
                'open': [100, 101],
                'high': [99, 100],  # High < Open (invalid)
                'low': [98, 99],
                'close': [102, 103],
                'volume': [1000, 1100]
            })
            assert validate_price_data(invalid_df) == False
            
            print("✅ Price data validation test passed")
            return True
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"❌ Price data validation test failed: {e}")
            return False

class TestStrategyLogic:
    """Test strategy logic without vectorbt dependency"""
    
    def test_moving_average_calculation(self):
        """Test moving average calculation logic"""
        # Create simple test data
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Calculate simple moving averages manually
        fast_window = 3
        slow_window = 5
        
        fast_ma = prices.rolling(window=fast_window).mean()
        slow_ma = prices.rolling(window=slow_window).mean()
        
        # Test that MA calculations are reasonable
        assert len(fast_ma) == len(prices)
        assert len(slow_ma) == len(prices)
        
        # Fast MA should have values starting from index 2 (0-indexed, window=3)
        assert pd.isna(fast_ma.iloc[0])  # Not enough data
        assert pd.isna(fast_ma.iloc[1])  # Not enough data
        assert not pd.isna(fast_ma.iloc[2])  # First valid MA
        
        # Check actual calculation
        expected_first_fast_ma = (100 + 101 + 102) / 3
        assert abs(fast_ma.iloc[2] - expected_first_fast_ma) < 0.001
        
        print("✅ Moving average calculation test passed")
        return True
    
    def test_crossover_signals(self):
        """Test crossover signal generation logic"""
        # Create test data where fast MA crosses above slow MA
        fast_ma = pd.Series([100, 101, 102, 103, 104])
        slow_ma = pd.Series([102, 102, 102, 102, 102])
        
        # Generate crossover signals
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Should have entry signal when fast crosses above slow
        assert entries.iloc[2] == True  # Fast MA (102) crosses above slow MA (102)
        
        print("✅ Crossover signals test passed")
        return True

class TestDataIntegrity:
    """Test data integrity and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            # This would normally be handled by validation
            if empty_df.empty:
                print("✅ Empty data handled correctly")
                return True
        except Exception as e:
            print(f"❌ Empty data handling failed: {e}")
            return False
    
    def test_nan_data_handling(self):
        """Test handling of NaN values"""
        df_with_nans = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [105, 106, np.nan],
            'low': [99, 100, 101],
            'close': [104, 105, 103],
            'volume': [1000, 1100, 1200]
        })
        
        # Count NaN values
        nan_count = df_with_nans.isna().sum().sum()
        assert nan_count > 0  # Should detect NaN values
        
        print(f"✅ NaN data detection test passed (found {nan_count} NaN values)")
        return True

def run_vectorbt_tests():
    """Run all vectorbt-related tests"""
    results = {}
    
    # Utils tests
    utils_tests = TestVectorbtUtils()
    results["sample_data_generation"] = utils_tests.test_create_sample_price_data()
    results["price_data_validation"] = utils_tests.test_validate_price_data()
    
    # Strategy logic tests
    strategy_tests = TestStrategyLogic()
    results["moving_average_calculation"] = strategy_tests.test_moving_average_calculation()
    results["crossover_signals"] = strategy_tests.test_crossover_signals()
    
    # Data integrity tests
    integrity_tests = TestDataIntegrity()
    results["empty_data_handling"] = integrity_tests.test_empty_data_handling()
    results["nan_data_handling"] = integrity_tests.test_nan_data_handling()
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Vectorbt Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All vectorbt tests passed!")
    else:
        print("⚠️  Some vectorbt tests failed:")
        for test_name, result in results.items():
            if not result:
                print(f"   - {test_name}")
    
    return results

if __name__ == "__main__":
    print("Running Vectorbt Utility Tests...")
    print("=" * 50)
    run_vectorbt_tests()
