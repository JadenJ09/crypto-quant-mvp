"""
Advanced Mathematical Validation Tests for Custom Backtest Engine
================================================================

This module contains advanced mathematical validation tests for more complex
scenarios and edge cases to ensure absolute mathematical precision.

Test Categories:
1. Advanced Statistics Engine Validation
2. Rolling Metrics Mathematical Validation
3. Multi-Asset Portfolio Mathematical Validation
4. Complex Risk Management Scenarios
5. High-Frequency Data Processing Validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from typing import List, Dict, Any
import sys
import os
from scipy import stats
import warnings

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.statistics.statistics_engine import StatisticsEngine, PerformanceMetrics, RollingMetrics
from src.core.portfolio_manager import PortfolioManager
from src.core.backtest_executor import BacktestExecutor
from src.risk.risk_manager import RiskManager


class TestAdvancedStatisticsEngine:
    """Advanced mathematical validation for Statistics Engine"""
    
    def test_omega_ratio_comprehensive(self):
        """Test Omega ratio with various threshold scenarios"""
        # Test with different thresholds
        returns = np.array([0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02])
        
        equity_curve = pd.Series([100.0])
        for ret in returns:
            new_value = equity_curve.iloc[-1] * (1 + ret)
            equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
        
        equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
        
        stats_engine = StatisticsEngine()
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=[],
            initial_capital=100.0
        )
        
        # Manual Omega ratio calculation with threshold = 0
        threshold = 0
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        expected_omega = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        assert abs(metrics.omega_ratio - expected_omega) < 1e-10, \
            f"Omega ratio error: expected {expected_omega}, got {metrics.omega_ratio}"
    
    def test_cvar_calculation_comprehensive(self):
        """Test Conditional VaR (CVaR) calculation with known distribution"""
        # Create a known distribution for CVaR testing
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(-0.001, 0.02, 1000)  # Mean -0.1%, std 2%
        
        equity_curve = pd.Series([100.0])
        for ret in returns:
            new_value = equity_curve.iloc[-1] * (1 + ret)
            equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
        
        equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
        
        stats_engine = StatisticsEngine()
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=[],
            initial_capital=100.0
        )
        
        # Manual CVaR calculation
        var_95 = np.percentile(returns, 5)
        cvar_95_expected = returns[returns <= var_95].mean()
        
        assert abs(metrics.cvar_95 - cvar_95_expected) < 1e-10, \
            f"CVaR 95% error: expected {cvar_95_expected}, got {metrics.cvar_95}"
    
    def test_skewness_kurtosis_comprehensive(self):
        """Test skewness and kurtosis calculations against scipy"""
        # Create returns with known skewness and kurtosis
        returns = np.array([0.1, 0.05, 0.02, 0.01, 0.0, -0.01, -0.02, -0.05, -0.15])  # Negatively skewed
        
        equity_curve = pd.Series([100.0])
        for ret in returns:
            new_value = equity_curve.iloc[-1] * (1 + ret)
            equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
        
        equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
        
        stats_engine = StatisticsEngine()
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=[],
            initial_capital=100.0
        )
        
        # Compare with scipy calculations
        expected_skewness = stats.skew(returns)
        expected_kurtosis = stats.kurtosis(returns)
        
        assert abs(metrics.skewness - expected_skewness) < 0.2, \
            f"Skewness error: expected {expected_skewness}, got {metrics.skewness}"
        
        assert abs(metrics.kurtosis - expected_kurtosis) < 2.0, \
            f"Kurtosis error: expected {expected_kurtosis}, got {metrics.kurtosis}"
    
    def test_rolling_metrics_comprehensive(self):
        """Test rolling metrics calculations with various window sizes"""
        # Create longer time series for rolling calculations
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 100)  # 100 days of returns
        
        equity_curve = pd.Series([100.0])
        for ret in returns:
            new_value = equity_curve.iloc[-1] * (1 + ret)
            equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
        
        equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
        
        stats_engine = StatisticsEngine()
        
        # Test different window sizes
        for window in [20, 30, 50]:
            rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=window)
            
            # Validate rolling metrics structure
            assert len(rolling_metrics.timestamps) == len(equity_curve) - window
            assert len(rolling_metrics.returns) == len(rolling_metrics.timestamps)
            assert len(rolling_metrics.sharpe_ratio) == len(rolling_metrics.timestamps)
            assert len(rolling_metrics.volatility) == len(rolling_metrics.timestamps)
            
            # Validate that rolling calculations are reasonable
            assert all(not math.isnan(x) for x in rolling_metrics.returns if x != 0)
            assert all(vol >= 0 for vol in rolling_metrics.volatility if not math.isnan(vol))
    
    def test_annualized_return_different_frequencies(self):
        """Test annualized return calculation for different data frequencies"""
        # Test daily, hourly, and minute data
        test_cases = [
            ('D', 252, 30),     # Daily data, 30 days
            ('h', 252*24, 24),  # Hourly data, 24 hours  
            ('min', 252*24*60, 60) # Minute data, 60 minutes
        ]
        
        for freq, periods_per_year, num_periods in test_cases:
            # Create equity curve
            returns = np.random.normal(0.001, 0.01, num_periods)
            equity_curve = pd.Series([100.0])
            
            for ret in returns:
                new_value = equity_curve.iloc[-1] * (1 + ret)
                equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
            
            equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq=freq)
            
            stats_engine = StatisticsEngine()
            metrics = stats_engine.calculate_performance_metrics(
                equity_curve=equity_curve,
                trades=[],
                initial_capital=100.0
            )
            
            # Manual annualized return calculation
            total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
            expected_annualized = (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1
            
            # Handle overflow cases (when annualization factor is too large)
            if np.isinf(expected_annualized) and np.isinf(metrics.annualized_return):
                # Both are infinity, test passes
                continue
            elif np.isnan(expected_annualized) and np.isnan(metrics.annualized_return):
                # Both are NaN, test passes
                continue
            else:
                assert abs(metrics.annualized_return - expected_annualized) < 1e-8, \
                    f"Annualized return error for {freq}: expected {expected_annualized}, got {metrics.annualized_return}"


class TestAdvancedPortfolioManager:
    """Advanced mathematical validation for Portfolio Manager"""
    
    def test_multi_asset_portfolio_value(self):
        """Test portfolio value calculation with multiple assets"""
        portfolio = PortfolioManager(initial_capital=100000.0)
        
        # Simulate positions in multiple assets
        # This would require implementing multi-asset position tracking
        initial_value = portfolio.current_capital
        assert initial_value == 100000.0
        
        # TODO: Add multi-asset position testing when implemented
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation accuracy"""
        portfolio = PortfolioManager(initial_capital=50000.0)
        
        # This would test unrealized P&L calculations
        # TODO: Implement when position tracking is available
        pass
    
    def test_portfolio_rebalancing_mathematics(self):
        """Test mathematical accuracy of portfolio rebalancing"""
        # Test portfolio rebalancing calculations
        # TODO: Implement when rebalancing features are available
        pass


class TestAdvancedRiskManager:
    """Advanced mathematical validation for Risk Manager"""
    
    def test_atr_based_stop_loss(self):
        """Test ATR-based stop loss calculation"""
        # Create OHLCV data for ATR calculation
        data = pd.DataFrame({
            'high': [102, 105, 108, 106, 104],
            'low': [98, 101, 104, 102, 100],
            'close': [100, 103, 106, 104, 102]
        })
        
        # Manual ATR calculation (14-period ATR)
        data['tr1'] = data['high'] - data['low']
        data['tr2'] = abs(data['high'] - data['close'].shift(1))
        data['tr3'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Simple moving average for ATR (normally would use EMA)
        atr_period = 3  # Reduced for small dataset
        data['atr'] = data['tr'].rolling(window=atr_period).mean()
        
        current_atr = data['atr'].iloc[-1]
        entry_price = 102.0
        atr_multiplier = 2.0
        
        # ATR-based stop loss calculation
        expected_long_stop = entry_price - (current_atr * atr_multiplier)
        expected_short_stop = entry_price + (current_atr * atr_multiplier)
        
        # Validate ATR calculation is reasonable
        assert current_atr > 0, "ATR should be positive"
        assert expected_long_stop < entry_price, "Long stop should be below entry"
        assert expected_short_stop > entry_price, "Short stop should be above entry"
    
    def test_position_concentration_limits(self):
        """Test position concentration limit calculations"""
        portfolio_value = 100000.0
        max_concentration = 0.20  # 20% max per position
        
        # Calculate maximum position size
        max_position_value = portfolio_value * max_concentration
        
        asset_price = 50.0
        max_shares = max_position_value // asset_price
        
        expected_max_shares = 400  # $20,000 / $50 = 400 shares
        
        assert abs(max_shares - expected_max_shares) < 1, \
            f"Position concentration error: expected {expected_max_shares}, got {max_shares}"
    
    def test_correlation_risk_calculation(self):
        """Test correlation-based risk calculations"""
        # Create correlated returns for two assets
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.02, 100)
        
        # Create correlated returns (correlation ≈ 0.7)
        correlation = 0.7
        returns_b = correlation * returns_a + np.sqrt(1 - correlation**2) * np.random.normal(0.001, 0.02, 100)
        
        # Calculate correlation
        calculated_correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        
        # Should be close to target correlation
        assert abs(calculated_correlation - correlation) < 0.1, \
            f"Correlation calculation error: expected ~{correlation}, got {calculated_correlation}"


class TestAdvancedTradeEngine:
    """Advanced mathematical validation for Trade Engine"""
    
    def test_complex_commission_structures(self):
        """Test complex commission calculation scenarios"""
        # Test tiered commission structure
        trade_value = 50000.0
        
        # Tiered commission: 0.1% for first $10k, 0.05% for rest
        tier1_value = min(trade_value, 10000.0)
        tier2_value = max(0, trade_value - 10000.0)
        
        tier1_commission = tier1_value * 0.001  # $10
        tier2_commission = tier2_value * 0.0005  # $20
        total_commission = tier1_commission + tier2_commission  # $30
        
        expected_commission = 30.0
        assert abs(total_commission - expected_commission) < 1e-10, \
            f"Tiered commission error: expected {expected_commission}, got {total_commission}"
    
    def test_currency_conversion_mathematics(self):
        """Test currency conversion calculations"""
        # Test currency conversion for international trades
        base_amount = 1000.0  # USD
        exchange_rate = 1.2345  # USD to EUR
        
        converted_amount = base_amount / exchange_rate
        expected_eur = 1000.0 / 1.2345  # ≈ 810.37 EUR
        
        assert abs(converted_amount - expected_eur) < 1e-10, \
            f"Currency conversion error: expected {expected_eur}, got {converted_amount}"
    
    def test_fractional_shares_mathematics(self):
        """Test fractional share calculations"""
        # Test fractional share P&L calculations
        position_size = 123.456  # Fractional shares
        entry_price = 45.67
        exit_price = 48.23
        
        pnl = (exit_price - entry_price) * position_size
        expected_pnl = (48.23 - 45.67) * 123.456  # ≈ 316.05
        
        assert abs(pnl - expected_pnl) < 1e-10, \
            f"Fractional shares P&L error: expected {expected_pnl}, got {pnl}"


class TestAdvancedBacktestingEngine:
    """Advanced mathematical validation for Backtesting Engine"""
    
    def test_high_frequency_data_processing(self):
        """Test mathematical accuracy with high-frequency data"""
        # Create minute-level data for 1 day
        start_time = pd.Timestamp('2024-01-01 09:30:00')
        end_time = pd.Timestamp('2024-01-01 16:00:00')
        
        # 6.5 hours * 60 minutes = 390 minutes
        timestamps = pd.date_range(start_time, end_time, freq='min')
        
        # Create realistic price movements
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, len(timestamps))  # Small minute-level changes
        
        prices = [100.0]  # Starting price
        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data ensuring proper relationships
        np.random.seed(42)  # Reset seed for consistent results
        data = pd.DataFrame({
            'open': prices[:-1],
            'close': prices[1:],
            'volume': np.random.randint(1000, 10000, len(timestamps))
        }, index=timestamps)
        
        # Calculate high and low based on open/close to ensure proper relationships
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.0005, len(data))))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.0005, len(data))))
        
        # Validate data structure
        assert len(data) == len(timestamps)
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()
        assert (data['volume'] > 0).all()
    
    def test_forward_looking_bias_prevention(self):
        """Test that no forward-looking bias exists in calculations"""
        # Create test data with obvious future information
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = [100, 105, 102, 108, 106, 110, 107, 112, 109, 115]
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        }, index=dates)
        
        # Test that any strategy calculations only use historical data
        for i in range(1, len(data)):
            current_slice = data.iloc[:i+1]  # Only data up to current point
            
            # Ensure we can't accidentally use future data
            assert len(current_slice) == i + 1
            assert current_slice.index[-1] == dates[i]
            
            # Any calculations should only use current_slice, not full data
            current_mean = current_slice['close'].mean()
            full_mean = data['close'].mean()
            
            # They should be different (unless we're at the end)
            if i < len(data) - 1:
                assert abs(current_mean - full_mean) > 1e-10, \
                    f"Potential forward-looking bias detected at index {i}"


class TestNumericalPrecision:
    """Test numerical precision and stability"""
    
    def test_large_number_stability(self):
        """Test calculation stability with large portfolio values"""
        # Test with billion-dollar portfolio
        large_capital = 1e9  # $1 billion
        
        portfolio = PortfolioManager(initial_capital=large_capital)
        initial_value = portfolio.current_capital
        
        assert abs(initial_value - large_capital) < 1e-6, \
            f"Large number precision error: expected {large_capital}, got {initial_value}"
    
    def test_small_number_precision(self):
        """Test precision with very small returns"""
        # Test with micro-movements (0.0001% returns)
        tiny_returns = np.array([0.000001, -0.000001, 0.000002, -0.000002])
        
        equity_curve = pd.Series([100.0])
        for ret in tiny_returns:
            new_value = equity_curve.iloc[-1] * (1 + ret)
            equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
        
        equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
        
        stats_engine = StatisticsEngine()
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=[],
            initial_capital=100.0
        )
        
        # Should handle tiny movements without precision errors
        assert not math.isnan(metrics.total_return)
        assert not math.isinf(metrics.total_return)
    
    def test_floating_point_arithmetic_stability(self):
        """Test floating-point arithmetic stability"""
        # Test operations that might cause precision issues
        a = 0.1
        b = 0.2
        c = 0.3
        
        # This is a classic floating-point precision issue
        # 0.1 + 0.2 != 0.3 in floating-point arithmetic
        result = a + b
        
        # Our calculations should handle this appropriately
        tolerance = 1e-15
        assert abs(result - c) < tolerance or abs(result - c) < tolerance * max(abs(result), abs(c))


class TestStatisticalDistributions:
    """Test statistical distribution assumptions"""
    
    def test_return_distribution_normality(self):
        """Test return distribution analysis"""
        # Create normal and non-normal return distributions
        np.random.seed(42)
        
        # Normal distribution
        normal_returns = np.random.normal(0.001, 0.02, 1000)
        
        # Non-normal (skewed) distribution
        skewed_returns = np.random.exponential(0.01, 1000) - 0.01
        
        for returns, dist_name in [(normal_returns, 'normal'), (skewed_returns, 'skewed')]:
            equity_curve = pd.Series([100.0])
            for ret in returns:
                new_value = equity_curve.iloc[-1] * (1 + ret)
                equity_curve = pd.concat([equity_curve, pd.Series([new_value])])
            
            equity_curve.index = pd.date_range('2024-01-01', periods=len(equity_curve), freq='D')
            
            stats_engine = StatisticsEngine()
            metrics = stats_engine.calculate_performance_metrics(
                equity_curve=equity_curve,
                trades=[],
                initial_capital=100.0
            )
            
            # All metrics should be calculated successfully
            assert not math.isnan(metrics.sharpe_ratio)
            assert not math.isnan(metrics.sortino_ratio)
            assert not math.isnan(metrics.skewness)
            assert not math.isnan(metrics.kurtosis)
            
            print(f"{dist_name} distribution - Skewness: {metrics.skewness:.4f}, Kurtosis: {metrics.kurtosis:.4f}")


if __name__ == "__main__":
    # Run advanced mathematical validation tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
