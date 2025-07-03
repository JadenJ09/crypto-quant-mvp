"""
Phase 3 Tests: Statistics Engine

Test suite for advanced statistics and performance analysis features including:
- Performance metrics calculation (returns, Sharpe, Sortino, Calmar)
- Risk metrics (VaR, CVaR, drawdown analysis)
- Rolling statistics and time-based analysis
- Trade attribution and analysis
- Benchmark comparison and relative performance
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from statistics.statistics_engine import (
    StatisticsEngine, PerformanceMetrics, RollingMetrics, TradeAnalysis
)


def create_sample_equity_curve(periods: int = 252, initial_value: float = 100000.0, 
                             annual_return: float = 0.10, annual_volatility: float = 0.15) -> pd.Series:
    """Create sample equity curve for testing"""
    np.random.seed(42)
    
    # Generate daily returns
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    returns = np.random.normal(daily_return, daily_volatility, periods)
    
    # Create equity curve
    equity_values = [initial_value]
    for ret in returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    
    # Create timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=periods + 1, freq='D')
    
    return pd.Series(equity_values, index=timestamps)


def create_sample_trades(num_trades: int = 50) -> list:
    """Create sample trade data for testing"""
    np.random.seed(42)
    
    trades = []
    start_date = pd.Timestamp('2024-01-01')
    
    for i in range(num_trades):
        # Random trade characteristics
        side = np.random.choice(['long', 'short'])
        entry_price = np.random.uniform(100, 200)
        
        # Generate PnL with some winning bias
        pnl = np.random.normal(50, 200)  # Slight positive bias
        
        # Entry and exit times
        entry_time = start_date + pd.Timedelta(days=i * 5)
        exit_time = entry_time + pd.Timedelta(hours=np.random.randint(1, 48))
        
        # Exit price based on PnL
        if side == 'long':
            exit_price = entry_price + pnl
        else:
            exit_price = entry_price - pnl
        
        trade = {
            'trade_id': i,
            'symbol': 'BTCUSD',
            'side': side,
            'size': np.random.uniform(0.1, 2.0),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'commission': abs(pnl) * 0.001,  # 0.1% commission
            'exit_reason': np.random.choice(['signal', 'stop_loss', 'take_profit'])
        }
        
        trades.append(trade)
    
    return trades


class TestStatisticsEngine(unittest.TestCase):
    """Test the statistics engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.stats_engine = StatisticsEngine(risk_free_rate=0.02)
        self.equity_curve = create_sample_equity_curve()
        self.trades = create_sample_trades()
        self.initial_capital = 100000.0
    
    def test_performance_metrics_calculation(self):
        """Test comprehensive performance metrics calculation"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Check that all metrics are calculated
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # Basic return metrics
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.annualized_return, float)
        self.assertIsInstance(metrics.cumulative_return, float)
        
        # Risk metrics
        self.assertGreater(metrics.volatility, 0)
        self.assertGreater(metrics.annualized_volatility, 0)
        self.assertLessEqual(metrics.max_drawdown, 0)  # Should be negative or zero
        
        # Risk-adjusted metrics
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.sortino_ratio, float)
        self.assertIsInstance(metrics.calmar_ratio, float)
        
        # Trade statistics
        self.assertEqual(metrics.total_trades, len([t for t in self.trades if t.get('exit_price') is not None]))
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
        
        print(f"✅ Performance Metrics - Total Return: {metrics.total_return:.2%}, Sharpe: {metrics.sharpe_ratio:.2f}")
    
    def test_returns_calculation(self):
        """Test return calculations are accurate"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Manual calculation
        expected_total_return = (self.equity_curve.iloc[-1] - self.equity_curve.iloc[0]) / self.equity_curve.iloc[0]
        
        self.assertAlmostEqual(metrics.total_return, expected_total_return, places=6)
        
        # Cumulative return should equal total return for simple case
        expected_cumulative = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        self.assertAlmostEqual(metrics.cumulative_return, expected_cumulative, places=6)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation accuracy"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Manual drawdown calculation
        running_max = self.equity_curve.expanding().max()
        drawdown_series = (self.equity_curve - running_max) / running_max
        expected_max_drawdown = drawdown_series.min()
        expected_current_drawdown = drawdown_series.iloc[-1]
        
        self.assertAlmostEqual(metrics.max_drawdown, expected_max_drawdown, places=6)
        self.assertAlmostEqual(metrics.current_drawdown, expected_current_drawdown, places=6)
        
        print(f"✅ Drawdown Analysis - Max DD: {metrics.max_drawdown:.2%}, Current DD: {metrics.current_drawdown:.2%}")
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Manual Sharpe calculation
        returns = self.equity_curve.pct_change().dropna()
        excess_returns = returns - self.stats_engine.risk_free_rate / 252
        expected_sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        self.assertAlmostEqual(metrics.sharpe_ratio, expected_sharpe, places=2)
        
        # Sharpe should be reasonable (not extreme values)
        self.assertGreater(metrics.sharpe_ratio, -5)
        self.assertLess(metrics.sharpe_ratio, 5)
    
    def test_trade_statistics(self):
        """Test trade statistics calculation"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Verify trade counts
        completed_trades = [t for t in self.trades if t.get('exit_price') is not None]
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) < 0]
        
        self.assertEqual(metrics.total_trades, len(completed_trades))
        self.assertEqual(metrics.winning_trades, len(winning_trades))
        self.assertEqual(metrics.losing_trades, len(losing_trades))
        
        # Win rate should match
        expected_win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        self.assertAlmostEqual(metrics.win_rate, expected_win_rate, places=6)
        
        # Profit factor validation
        if winning_trades and losing_trades:
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            expected_profit_factor = gross_profit / gross_loss
            self.assertAlmostEqual(metrics.profit_factor, expected_profit_factor, places=6)
        
        print(f"✅ Trade Statistics - Trades: {metrics.total_trades}, Win Rate: {metrics.win_rate:.1%}")
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation"""
        rolling_metrics = self.stats_engine.calculate_rolling_metrics(
            self.equity_curve, window=50
        )
        
        self.assertIsInstance(rolling_metrics, RollingMetrics)
        
        # Check that rolling metrics have reasonable length
        expected_length = len(self.equity_curve) - 50
        self.assertEqual(len(rolling_metrics.returns), expected_length)
        self.assertEqual(len(rolling_metrics.sharpe_ratio), expected_length)
        self.assertEqual(len(rolling_metrics.volatility), expected_length)
        self.assertEqual(len(rolling_metrics.drawdown), expected_length)
        
        # All arrays should have same length
        self.assertEqual(len(rolling_metrics.timestamps), len(rolling_metrics.returns))
        self.assertEqual(len(rolling_metrics.timestamps), len(rolling_metrics.sharpe_ratio))
        
        print(f"✅ Rolling Metrics - Window: 50, Data Points: {len(rolling_metrics.returns)}")
    
    def test_trade_analysis(self):
        """Test detailed trade analysis"""
        trade_analysis = self.stats_engine.analyze_trades(self.trades)
        
        self.assertIsInstance(trade_analysis, TradeAnalysis)
        
        # Count trades by direction
        long_trades = [t for t in self.trades if t.get('side') == 'long']
        short_trades = [t for t in self.trades if t.get('side') == 'short']
        
        self.assertEqual(trade_analysis.long_trades, len(long_trades))
        self.assertEqual(trade_analysis.short_trades, len(short_trades))
        
        # Win rates should be valid
        self.assertGreaterEqual(trade_analysis.long_win_rate, 0)
        self.assertLessEqual(trade_analysis.long_win_rate, 1)
        self.assertGreaterEqual(trade_analysis.short_win_rate, 0)
        self.assertLessEqual(trade_analysis.short_win_rate, 1)
        
        # Hold time should be positive
        self.assertGreaterEqual(trade_analysis.average_hold_time, pd.Timedelta(0))
        
        print(f"✅ Trade Analysis - Long: {trade_analysis.long_trades}, Short: {trade_analysis.short_trades}")
    
    def test_risk_metrics(self):
        """Test risk metrics calculation (VaR, CVaR, skewness, kurtosis)"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # VaR should be negative (represents loss)
        self.assertLessEqual(metrics.var_95, 0)
        
        # CVaR should be more negative than VaR
        self.assertLessEqual(metrics.cvar_95, metrics.var_95)
        
        # Skewness and kurtosis should be reasonable
        self.assertGreater(metrics.skewness, -10)
        self.assertLess(metrics.skewness, 10)
        self.assertGreater(metrics.kurtosis, -10)
        self.assertLess(metrics.kurtosis, 50)
        
        print(f"✅ Risk Metrics - VaR: {metrics.var_95:.2%}, CVaR: {metrics.cvar_95:.2%}")
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality"""
        # Create benchmark data
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.01, len(self.equity_curve)),
            index=self.equity_curve.index
        )
        
        self.stats_engine.set_benchmark(benchmark_returns)
        
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        # Benchmark metrics should be calculated
        self.assertIsNotNone(metrics.alpha)
        self.assertIsNotNone(metrics.beta)
        self.assertIsNotNone(metrics.information_ratio)
        self.assertIsNotNone(metrics.tracking_error)
        
        # Beta should be reasonable
        if metrics.beta is not None:
            self.assertGreater(metrics.beta, -5)
            self.assertLess(metrics.beta, 5)
        
        print(f"✅ Benchmark Comparison - Alpha: {metrics.alpha:.2%}, Beta: {metrics.beta:.2f}")
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, self.trades, self.initial_capital
        )
        
        rolling_metrics = self.stats_engine.calculate_rolling_metrics(self.equity_curve, window=30)
        trade_analysis = self.stats_engine.analyze_trades(self.trades)
        
        report = self.stats_engine.generate_report(metrics, rolling_metrics, trade_analysis)
        
        # Report should be a string with meaningful content
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 1000)  # Should be substantial
        
        # Check for key sections
        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("RETURNS", report)
        self.assertIn("RISK METRICS", report)
        self.assertIn("TRADE STATISTICS", report)
        self.assertIn("TRADE ANALYSIS", report)
        
        # Check for key metrics in report
        self.assertIn("Total Return:", report)
        self.assertIn("Sharpe Ratio:", report)
        self.assertIn("Max Drawdown:", report)
        self.assertIn("Win Rate:", report)
        
        print(f"✅ Report Generation - Length: {len(report)} characters")
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        # Empty equity curve
        empty_equity = pd.Series([], dtype=float)
        empty_trades = []
        
        metrics = self.stats_engine.calculate_performance_metrics(
            empty_equity, empty_trades, self.initial_capital
        )
        
        # Should return valid metrics with zero values
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        
        # Test with minimal data
        minimal_equity = pd.Series([100000, 101000], 
                                 index=pd.date_range('2024-01-01', periods=2, freq='D'))
        
        metrics = self.stats_engine.calculate_performance_metrics(
            minimal_equity, empty_trades, self.initial_capital
        )
        
        self.assertAlmostEqual(metrics.total_return, 0.01, places=6)
        
        print("✅ Empty Data Handling - Graceful degradation working")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # All losing trades
        losing_trades = []
        for i in range(10):
            trade = {
                'trade_id': i,
                'symbol': 'BTCUSD',
                'side': 'long',
                'entry_price': 100.0,
                'exit_price': 95.0,
                'entry_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'exit_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i, hours=1),
                'pnl': -5.0,
                'commission': 0.1
            }
            losing_trades.append(trade)
        
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, losing_trades, self.initial_capital
        )
        
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.winning_trades, 0)
        self.assertEqual(metrics.losing_trades, len(losing_trades))
        
        # All winning trades
        winning_trades = []
        for i in range(10):
            trade = {
                'trade_id': i,
                'symbol': 'BTCUSD',
                'side': 'long',
                'entry_price': 100.0,
                'exit_price': 105.0,
                'entry_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'exit_time': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i, hours=1),
                'pnl': 5.0,
                'commission': 0.1
            }
            winning_trades.append(trade)
        
        metrics = self.stats_engine.calculate_performance_metrics(
            self.equity_curve, winning_trades, self.initial_capital
        )
        
        self.assertEqual(metrics.win_rate, 1.0)
        self.assertEqual(metrics.winning_trades, len(winning_trades))
        self.assertEqual(metrics.losing_trades, 0)
        
        print("✅ Edge Cases - All wins/losses handled correctly")
    
    def test_periods_per_year_inference(self):
        """Test automatic detection of data frequency"""
        # Daily data
        daily_equity = pd.Series(
            [100000, 101000, 102000],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )
        
        metrics = self.stats_engine.calculate_performance_metrics(
            daily_equity, [], self.initial_capital
        )
        
        # Should calculate annualized metrics appropriately
        self.assertIsInstance(metrics.annualized_return, float)
        self.assertIsInstance(metrics.annualized_volatility, float)
        
        # Hourly data
        hourly_equity = pd.Series(
            [100000, 101000, 102000],
            index=pd.date_range('2024-01-01', periods=3, freq='H')
        )
        
        metrics = self.stats_engine.calculate_performance_metrics(
            hourly_equity, [], self.initial_capital
        )
        
        self.assertIsInstance(metrics.annualized_return, float)
        
        print("✅ Frequency Detection - Daily and hourly data handled correctly")


class TestStatisticsEngineIntegration(unittest.TestCase):
    """Integration tests for statistics engine with other components"""
    
    def test_full_pipeline_integration(self):
        """Test full integration with enhanced portfolio manager"""
        # This would test integration with the actual portfolio manager
        # For now, we'll simulate the expected data structures
        
        # Simulate portfolio manager output
        equity_curve = create_sample_equity_curve(periods=100)
        trades = create_sample_trades(num_trades=20)
        
        stats_engine = StatisticsEngine()
        
        # Calculate all metrics
        metrics = stats_engine.calculate_performance_metrics(equity_curve, trades, 100000.0)
        rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=30)
        trade_analysis = stats_engine.analyze_trades(trades)
        
        # Generate comprehensive report
        report = stats_engine.generate_report(metrics, rolling_metrics, trade_analysis)
        
        # Verify comprehensive analysis
        self.assertGreater(metrics.total_trades, 0)
        self.assertGreater(len(rolling_metrics.returns), 0)
        self.assertGreater(trade_analysis.long_trades + trade_analysis.short_trades, 0)
        self.assertGreater(len(report), 2000)
        
        print("✅ Full Pipeline Integration - All components working together")
        print(f"   Metrics: {metrics.total_trades} trades, {metrics.win_rate:.1%} win rate")
        print(f"   Rolling: {len(rolling_metrics.returns)} data points")
        print(f"   Report: {len(report)} characters")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
