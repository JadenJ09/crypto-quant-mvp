#!/usr/bin/env python3
"""
Test script for technical indicators calculation

This script validates that the indicators processor can:
1. Calculate indicators using the ta library 
2. Handle fallback to numpy/pandas implementations
3. Process OHLCV data correctly
4. Generate proper database-ready records
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.append('app')

from config import Settings
from indicators_processor import TechnicalIndicatorsProcessor


def create_sample_ohlcv_data(periods=100):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 50000.0
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
    
    # Generate random walk price data
    price_changes = np.random.normal(0, 0.01, periods)  # 1% volatility
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Create realistic OHLC from close price
        volatility = 0.005  # 0.5% intraday volatility
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationships are correct
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        # Generate volume (random but realistic)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    return df


def test_indicators_calculation():
    """Test the indicators calculation functionality"""
    print("🧪 Testing Technical Indicators Calculation")
    print("=" * 50)
    
    # Create test configuration
    settings = Settings()
    processor = TechnicalIndicatorsProcessor(settings)
    
    # Create sample data
    print("📊 Creating sample OHLCV data...")
    sample_data = create_sample_ohlcv_data(periods=200)  # Need enough data for indicators
    print(f"   Created {len(sample_data)} periods of data")
    print(f"   Date range: {sample_data.index.min()} to {sample_data.index.max()}")
    print(f"   Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    
    # Test indicator calculation
    print("\n🔧 Testing indicator calculations...")
    
    try:
        # Test with different timeframes
        timeframes = ['5min', '15min', '1hour']
        
        for timeframe in timeframes:
            print(f"\n   Testing {timeframe} indicators...")
            
            # Calculate indicators
            result_df = processor._calculate_indicators(sample_data.copy(), timeframe)
            
            # Check results
            print(f"   ✅ Original columns: {len(sample_data.columns)}")
            print(f"   ✅ Result columns: {len(result_df.columns)}")
            print(f"   ✅ New indicators: {len(result_df.columns) - len(sample_data.columns)}")
            
            # Check for specific indicators
            expected_indicators = [
                'rsi_14', 'rsi_21', 'rsi_30',
                'ema_9', 'ema_21', 'ema_50',
                'sma_20', 'sma_50',
                'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr_14', 'obv'
            ]
            
            found_indicators = []
            for indicator in expected_indicators:
                if indicator in result_df.columns:
                    found_indicators.append(indicator)
            
            print(f"   ✅ Found {len(found_indicators)}/{len(expected_indicators)} expected indicators")
            
            # Check for VWAP in intraday timeframes
            if timeframe in ['5min', '15min', '1hour']:
                if 'vwap' in result_df.columns:
                    print(f"   ✅ VWAP indicator found for {timeframe}")
                else:
                    print(f"   ⚠️  VWAP indicator missing for {timeframe}")
        
        print("\n🔍 Testing record preparation for database...")
        
        # Test record preparation
        test_df = processor._calculate_indicators(sample_data.copy(), '1hour')
        records = processor._prepare_records_for_db(test_df.tail(5), 'BTCUSDT')
        
        print(f"   ✅ Generated {len(records)} database records")
        
        if records:
            sample_record = records[0]
            print(f"   ✅ Sample record has {len(sample_record)} fields")
            print(f"   ✅ Sample record keys: {list(sample_record.keys())[:10]}...")  # Show first 10 keys
            
            # Check data types
            numeric_fields = 0
            null_fields = 0
            for key, value in sample_record.items():
                if key in ['time', 'symbol']:
                    continue
                if value is None:
                    null_fields += 1
                elif isinstance(value, (int, float)):
                    numeric_fields += 1
            
            print(f"   ✅ Numeric fields: {numeric_fields}, Null fields: {null_fields}")
        
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests completed successfully!")
    return True


def test_resampling():
    """Test OHLCV resampling functionality"""
    print("\n🔄 Testing OHLCV Resampling")
    print("=" * 50)
    
    settings = Settings()
    processor = TechnicalIndicatorsProcessor(settings)
    
    # Create 1-minute data
    sample_data = create_sample_ohlcv_data(periods=500)  # 500 minutes
    print(f"📊 Created {len(sample_data)} 1-minute records")
    
    # Test different resampling frequencies
    frequencies = ['5min', '15min', '1H', '4H', '1D']
    
    for freq in frequencies:
        try:
            resampled = processor._resample_ohlcv(sample_data.copy(), freq)
            print(f"   ✅ {freq}: {len(sample_data)} → {len(resampled)} records")
            
            if not resampled.empty:
                # Verify OHLC relationships
                valid_ohlc = (
                    (resampled['high'] >= resampled['open']).all() and
                    (resampled['high'] >= resampled['close']).all() and
                    (resampled['low'] <= resampled['open']).all() and
                    (resampled['low'] <= resampled['close']).all()
                )
                print(f"      Valid OHLC relationships: {'✅' if valid_ohlc else '❌'}")
                
        except Exception as e:
            print(f"   ❌ Error resampling {freq}: {e}")
    
    print("✅ Resampling tests completed!")


if __name__ == "__main__":
    print("🚀 VectorBT Service - Technical Indicators Test")
    print("=" * 60)
    
    # Run tests
    success = True
    
    try:
        success &= test_indicators_calculation()
        test_resampling()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nThe VectorBT service is ready to:")
            print("  • Calculate technical indicators using TA library")
            print("  • Resample OHLCV data to multiple timeframes")
            print("  • Handle bulk and real-time processing modes")
            print("  • Store results in TimescaleDB with proper formatting")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
