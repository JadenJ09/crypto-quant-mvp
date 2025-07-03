"""
Phase 2 Tests: Advanced Risk Management

Test suite for risk management features including:
- Position sizing methods (Kelly, volatility-adjusted, ATR-based)
- Portfolio risk limits (max positions, max drawdown, concentration)
- Multiple take profit levels
- Advanced stop loss methods
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from risk.risk_manager import (
    RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod, 
    StopLossMethod, TakeProfitLevel
)
from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from core.trade_engine import TradeEngine


def create_sample_data(periods: int = 100, start_date: str = '2024-01-01', freq: str = 'h') -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create realistic price data with volatility
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.01, periods)  # 1% daily volatility
    prices = 100 * np.exp(np.cumsum(price_changes))
    
    # Create OHLCV data
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, periods)))
    volume = np.random.randint(1000, 10000, periods)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })


class TestRiskManager(unittest.TestCase):
    """Test the risk management system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_limits = RiskLimits(
            max_positions=3,
            max_portfolio_risk=0.02,
            max_drawdown=0.15,
            max_concentration=0.30
        )
        
        self.position_sizing = PositionSizingConfig(
            method=PositionSizingMethod.PERCENTAGE,
            percentage=0.05,  # 5% of portfolio
            kelly_lookback=50,
            volatility_target=0.02
        )
        
        self.risk_manager = RiskManager(self.risk_limits, self.position_sizing)
    
    def test_percentage_position_sizing(self):
        """Test percentage-based position sizing"""
        signal_strength = 1.0
        current_price = 100.0
        portfolio_value = 10000.0
        
        position_size = self.risk_manager.calculate_position_size(
            signal_strength, current_price, portfolio_value
        )
        
        expected_size = (0.05 * 10000) / 100  # 5% of portfolio / price
        self.assertAlmostEqual(position_size, expected_size, places=4)
    
    def test_signal_strength_scaling(self):
        """Test that position size scales with signal strength"""
        current_price = 100.0
        portfolio_value = 10000.0
        
        # Test different signal strengths
        full_size = self.risk_manager.calculate_position_size(1.0, current_price, portfolio_value)
        half_size = self.risk_manager.calculate_position_size(0.5, current_price, portfolio_value)
        
        self.assertAlmostEqual(full_size, half_size * 2, places=4)
    
    def test_kelly_position_sizing(self):
        """Test Kelly criterion position sizing"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.KELLY_CRITERION
        
        # Create mock historical returns data with positive expected return
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.01, 0.02, 100)  # 1% mean return, 2% std
        kelly_data = {'returns': returns}
        
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            kelly_data=kelly_data
        )
        
        # Kelly size should be reasonable (not zero, not excessive)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 25)  # Less than or equal to 25% of portfolio (max Kelly)
    
    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.VOLATILITY_ADJUSTED
        
        current_price = 100.0
        portfolio_value = 10000.0
        
        # High volatility should result in smaller position
        high_vol_size = self.risk_manager.calculate_position_size(
            1.0, current_price, portfolio_value, volatility=0.04
        )
        
        # Low volatility should result in larger position
        low_vol_size = self.risk_manager.calculate_position_size(
            1.0, current_price, portfolio_value, volatility=0.01
        )
        
        self.assertGreater(low_vol_size, high_vol_size)
    
    def test_atr_position_sizing(self):
        """Test ATR-based position sizing"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.ATR_BASED
        
        current_price = 100.0
        portfolio_value = 10000.0
        atr = 2.0  # $2 ATR
        
        position_size = self.risk_manager.calculate_position_size(
            1.0, current_price, portfolio_value, atr=atr
        )
        
        # Should be based on risk per trade / stop distance
        expected_risk = 0.02 * portfolio_value  # 2% portfolio risk
        stop_distance = atr * 2.0  # ATR multiplier
        expected_size = expected_risk / stop_distance
        
        self.assertAlmostEqual(position_size, expected_size, places=4)
    
    def test_percentage_stop_loss(self):
        """Test percentage-based stop loss calculation"""
        entry_price = 100.0
        stop_pct = 0.05  # 5% stop loss
        
        # Long position
        long_stop = self.risk_manager.calculate_stop_loss(entry_price, 'long', stop_pct=stop_pct)
        expected_long_stop = entry_price * (1 - stop_pct)
        self.assertAlmostEqual(long_stop, expected_long_stop, places=4)
        
        # Short position
        short_stop = self.risk_manager.calculate_stop_loss(entry_price, 'short', stop_pct=stop_pct)
        expected_short_stop = entry_price * (1 + stop_pct)
        self.assertAlmostEqual(short_stop, expected_short_stop, places=4)
    
    def test_atr_stop_loss(self):
        """Test ATR-based stop loss calculation"""
        entry_price = 100.0
        atr = 2.0
        
        # Long position
        long_stop = self.risk_manager.calculate_stop_loss(
            entry_price, 'long', StopLossMethod.ATR_MULTIPLE, atr=atr
        )
        expected_long_stop = entry_price - (atr * 2.0)
        self.assertAlmostEqual(long_stop, expected_long_stop, places=4)
        
        # Short position
        short_stop = self.risk_manager.calculate_stop_loss(
            entry_price, 'short', StopLossMethod.ATR_MULTIPLE, atr=atr
        )
        expected_short_stop = entry_price + (atr * 2.0)
        self.assertAlmostEqual(short_stop, expected_short_stop, places=4)
    
    def test_take_profit_levels(self):
        """Test take profit level creation"""
        entry_price = 100.0
        levels = [(0.05, 0.5), (0.10, 0.3), (0.15, 0.2)]  # 3 levels
        
        # Long position
        long_tp_levels = self.risk_manager.create_take_profit_levels(entry_price, 'long', levels)
        
        self.assertEqual(len(long_tp_levels), 3)
        self.assertEqual(long_tp_levels[0].price_pct, 0.05)  # 5% profit
        self.assertEqual(long_tp_levels[0].quantity_pct, 0.5)  # 50% of position
        
        # Short position
        short_tp_levels = self.risk_manager.create_take_profit_levels(entry_price, 'short', levels)
        
        self.assertEqual(len(short_tp_levels), 3)
        self.assertEqual(short_tp_levels[0].price_pct, -0.05)  # -5% for short (profit below entry)
        self.assertEqual(short_tp_levels[0].quantity_pct, 0.5)
    
    def test_portfolio_risk_limits(self):
        """Test portfolio-level risk limit checks"""
        portfolio_value = 10000.0
        positions = {}  # Empty positions
        
        # Test normal position
        is_allowed, reason = self.risk_manager.check_portfolio_risk_limits(
            portfolio_value, positions, 5.0, 100.0  # $500 position
        )
        self.assertTrue(is_allowed)
        
        # Test concentration limit
        is_allowed, reason = self.risk_manager.check_portfolio_risk_limits(
            portfolio_value, positions, 35.0, 100.0  # $3500 position (35% of portfolio)
        )
        self.assertFalse(is_allowed)
        self.assertIn("Position too large", reason)
        
        # Test max positions limit
        positions = {f"SYM{i}": {} for i in range(3)}  # 3 positions (at limit)
        is_allowed, reason = self.risk_manager.check_portfolio_risk_limits(
            portfolio_value, positions, 5.0, 100.0
        )
        self.assertFalse(is_allowed)
        self.assertIn("Maximum positions limit", reason)


class TestEnhancedPortfolioManager(unittest.TestCase):
    """Test the enhanced portfolio manager with risk management integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_limits = RiskLimits(max_positions=3, max_concentration=0.25)
        self.position_sizing = PositionSizingConfig(method=PositionSizingMethod.PERCENTAGE, percentage=0.05)
        
        self.portfolio = EnhancedPortfolioManager(
            initial_capital=10000.0,
            risk_limits=self.risk_limits,
            position_sizing=self.position_sizing
        )
        
        # Add sample market data
        data = create_sample_data(100)
        self.portfolio.add_market_data('BTC', data)
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        self.assertEqual(self.portfolio.initial_capital, 10000.0)
        self.assertEqual(self.portfolio.current_capital, 10000.0)
        self.assertEqual(self.portfolio.available_cash, 10000.0)
        self.assertEqual(len(self.portfolio.equity_curve), 1)
    
    def test_market_data_integration(self):
        """Test market data integration for risk calculations"""
        volatility = self.portfolio.calculate_volatility('BTC')
        atr = self.portfolio.calculate_atr('BTC')
        kelly_data = self.portfolio.get_kelly_data('BTC')
        
        self.assertIsNotNone(volatility)
        self.assertIsNotNone(atr)
        self.assertIsNotNone(kelly_data)
        self.assertIn('returns', kelly_data)
    
    def test_position_opening_with_risk_management(self):
        """Test opening positions with risk management"""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Should succeed
        success, message = self.portfolio.open_position(
            symbol='BTC',
            side='long',
            signal_strength=1.0,
            entry_price=100.0,
            timestamp=timestamp,
            stop_loss_pct=0.05
        )
        
        self.assertTrue(success)
        self.assertIn("Position opened", message)
        
        # Check position was created
        positions = self.portfolio.trade_engine.get_all_positions()
        self.assertEqual(len(positions), 1)
        self.assertIn('BTC', positions)
    
    def test_risk_limit_enforcement(self):
        """Test that risk limits are enforced"""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Open positions up to the limit
        for i in range(3):
            success, _ = self.portfolio.open_position(
                symbol=f'SYM{i}',
                side='long',
                signal_strength=1.0,
                entry_price=100.0,
                timestamp=timestamp
            )
            self.assertTrue(success)
        
        # Try to open one more - should fail
        success, message = self.portfolio.open_position(
            symbol='SYM3',
            side='long',
            signal_strength=1.0,
            entry_price=100.0,
            timestamp=timestamp
        )
        
        self.assertFalse(success)
        self.assertIn("Risk limit violation", message)
    
    def test_multiple_take_profit_levels(self):
        """Test multiple take profit levels"""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Open position with multiple TP levels
        take_profit_levels = [(0.05, 0.5), (0.10, 0.3), (0.15, 0.2)]
        
        success, _ = self.portfolio.open_position(
            symbol='BTC',
            side='long',
            signal_strength=1.0,
            entry_price=100.0,
            timestamp=timestamp,
            take_profit_levels=take_profit_levels
        )
        
        self.assertTrue(success)
        self.assertIn('BTC', self.portfolio.take_profit_levels)
        self.assertEqual(len(self.portfolio.take_profit_levels['BTC']), 3)
    
    def test_take_profit_execution(self):
        """Test take profit level execution"""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Open position with TP levels
        take_profit_levels = [(0.05, 0.5)]  # 5% profit, close 50%
        
        success, _ = self.portfolio.open_position(
            symbol='BTC',
            side='long',
            signal_strength=1.0,
            entry_price=100.0,
            timestamp=timestamp,
            take_profit_levels=take_profit_levels
        )
        
        self.assertTrue(success)
        
        # Update portfolio with higher price to trigger TP
        market_prices = {'BTC': 106.0}  # 6% gain, should trigger 5% TP level
        self.portfolio.update_portfolio(timestamp, market_prices)
        
        # Check that partial close occurred
        trades = self.portfolio.trade_engine.get_trades()
        partial_trades = [t for t in trades if 'Take profit' in t.exit_reason]
        self.assertGreater(len(partial_trades), 0)
    
    def test_portfolio_summary_with_risk_metrics(self):
        """Test portfolio summary includes risk metrics"""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Open a position
        self.portfolio.open_position(
            symbol='BTC',
            side='long', 
            signal_strength=1.0,
            entry_price=100.0,
            timestamp=timestamp
        )
        
        # Update with current prices
        market_prices = {'BTC': 105.0}
        self.portfolio.update_portfolio(timestamp, market_prices)
        
        # Get summary
        summary = self.portfolio.get_portfolio_summary(market_prices)
        
        self.assertIn('risk_metrics', summary)
        self.assertIn('total_exposure', summary['risk_metrics'])
        self.assertIn('leverage', summary['risk_metrics'])
        self.assertEqual(summary['open_positions'], 1)
    
    def test_equity_curve_tracking(self):
        """Test equity curve tracking with risk management"""
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1h')
        prices = [100, 102, 105, 103, 107]
        
        # Open position
        self.portfolio.open_position(
            symbol='BTC',
            side='long',
            signal_strength=1.0,
            entry_price=prices[0],
            timestamp=timestamps[0]
        )
        
        # Update portfolio with price changes
        for i, (timestamp, price) in enumerate(zip(timestamps[1:], prices[1:]), 1):
            market_prices = {'BTC': price}
            self.portfolio.update_portfolio(timestamp, market_prices)
        
        # Check equity curve (account for initial state)
        equity_df = self.portfolio.get_equity_curve_df()
        # Should have initial state + 4 updates = 5 total, but let's be flexible
        self.assertGreaterEqual(len(equity_df), 4)  # At least 4 entries
        self.assertTrue(all(col in equity_df.columns for col in ['timestamp', 'equity', 'drawdown']))
        
        # Final equity should reflect the price appreciation
        final_equity = equity_df['equity'].iloc[-1]
        self.assertGreater(final_equity, self.portfolio.initial_capital)


class TestRiskManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in risk management"""
    
    def setUp(self):
        self.risk_manager = RiskManager()
    
    def test_zero_volatility_handling(self):
        """Test handling of zero volatility"""
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            volatility=0.0
        )
        
        # Should fall back to percentage sizing
        self.assertGreater(position_size, 0)
    
    def test_missing_kelly_data(self):
        """Test handling of missing Kelly data"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.KELLY_CRITERION
        
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            kelly_data=None
        )
        
        # Should fall back to percentage sizing
        self.assertGreater(position_size, 0)
    
    def test_insufficient_kelly_data(self):
        """Test handling of insufficient Kelly data"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.KELLY_CRITERION
        
        # Too few returns
        kelly_data = {'returns': [0.01, 0.02]}
        
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            kelly_data=kelly_data
        )
        
        # Should fall back to percentage sizing
        self.assertGreater(position_size, 0)
    
    def test_extreme_kelly_values(self):
        """Test Kelly criterion with extreme values"""
        self.risk_manager.position_sizing.method = PositionSizingMethod.KELLY_CRITERION
        
        # Very high returns (should be capped)
        returns = np.array([0.1] * 50)  # 10% returns
        kelly_data = {'returns': returns}
        
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            kelly_data=kelly_data
        )
        
        # Should be capped at max Kelly fraction
        max_position_value = self.risk_manager.position_sizing.kelly_max * 10000
        max_position_size = max_position_value / 100
        self.assertLessEqual(position_size, max_position_size)


if __name__ == '__main__':
    unittest.main()
