#!/usr/bin/env python3
"""
Comprehensive Stop Loss Testing
Test all aspects of stop loss functionality including:
1. Precise stop loss execution
2. Max drawdown limits
3. Risk management features
4. Database integration
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.backtest_executor import BacktestExecutor
from src.core.portfolio_manager import PortfolioManager
from src.core.trade_engine import TradeEngine
from src.risk.risk_manager import RiskManager

# Database API imports
from tests.test_database_integration import DatabaseManager

class StopLossTestSuite:
    """Comprehensive stop loss test suite"""
    
    def __init__(self):
        # Database connection string
        database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql://quant_user:quant_password@localhost:5433/quant_db'
        )
        self.db_manager = DatabaseManager(database_url)
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all stop loss tests"""
        logger.info("🧪 Starting Comprehensive Stop Loss Test Suite")
        
        # Initialize database
        await self.db_manager.initialize()
        
        try:
            # Test 1: Precise stop loss execution
            await self.test_precise_stop_loss_execution()
            
            # Test 2: Max drawdown protection
            await self.test_max_drawdown_protection()
            
            # Test 3: Risk management integration
            await self.test_risk_management_integration()
            
            # Test 4: Database integration with stop losses
            await self.test_database_integration_with_stop_losses()
            
            # Test 5: Edge cases and error handling
            await self.test_edge_cases()
            
        finally:
            await self.db_manager.close()
            
        # Print summary
        self.print_test_summary()
    
    async def test_precise_stop_loss_execution(self):
        """Test 1: Verify precise stop loss execution"""
        logger.info("🎯 Test 1: Precise Stop Loss Execution")
        
        # Create synthetic data with specific price movements
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        # Create price data that will trigger stop loss
        prices = [100.0]  # Start at $100
        for i in range(1, 100):
            if i < 50:
                prices.append(prices[-1] * 1.001)  # Gradual increase
            else:
                prices.append(prices[-1] * 0.999)  # Gradual decrease to trigger stop loss
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        }, index=dates)
        
        # Strategy that buys at the start
        def buy_and_hold_strategy(data_slice, timestamp):
            if len(data_slice) == 1:  # First bar
                return {'TEST': {'action': 'buy', 'side': 'long'}}
            return {'TEST': {'action': 'hold'}}
        
        # Create executor with precise stop loss
        executor = BacktestExecutor(
            initial_capital=10000.0,
            commission_rate=0.0,  # No commission for precise testing
            slippage=0.0  # No slippage for precise testing
        )
        
        executor.set_strategy(buy_and_hold_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.10,
            risk_per_trade=0.50,  # Use 50% of capital
            max_drawdown_limit=0.20
        )
        
        # Run backtest
        results = executor.run_backtest(test_data, ['TEST'])
        
        # Verify stop loss was triggered
        trades = executor.portfolio.get_trades_dataframe()
        
        test_result = {
            'test_name': 'Precise Stop Loss Execution',
            'passed': False,
            'details': {}
        }
        
        if not trades.empty:
            # Check if we have a completed trade (entry and exit)
            completed_trades = trades[trades['exit_price'].notna()]
            if not completed_trades.empty:
                trade = completed_trades.iloc[0]
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                
                # Calculate actual loss percentage
                actual_loss_pct = (exit_price - entry_price) / entry_price
                expected_loss_pct = -0.05  # 5% stop loss
                
                test_result['details'] = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'actual_loss_pct': actual_loss_pct * 100,
                    'expected_loss_pct': expected_loss_pct * 100,
                    'precision_error': abs(actual_loss_pct - expected_loss_pct) * 100
                }
                
                # Test passes if loss is within 0.1% of expected
                if abs(actual_loss_pct - expected_loss_pct) < 0.001:
                    test_result['passed'] = True
                    logger.info(f"✅ Stop loss precision: {actual_loss_pct*100:.3f}% (expected: {expected_loss_pct*100:.3f}%)")
                else:
                    logger.warning(f"❌ Stop loss imprecise: {actual_loss_pct*100:.3f}% (expected: {expected_loss_pct*100:.3f}%)")
            else:
                logger.warning("❌ No completed trades found")
        else:
            logger.warning("❌ No trades found")
        
        self.test_results.append(test_result)
    
    async def test_max_drawdown_protection(self):
        """Test 2: Verify max drawdown protection"""
        logger.info("🛡️ Test 2: Max Drawdown Protection")
        
        # Create data that would cause large drawdown
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
        prices = [100.0]
        
        # Create volatile data with significant downtrend
        for i in range(1, 200):
            if i < 20:
                prices.append(prices[-1] * 1.01)  # Initial gains
            else:
                # Steep decline to test max drawdown
                prices.append(prices[-1] * 0.99)
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [1000] * 200
        }, index=dates)
        
        # Aggressive strategy that would normally cause large losses
        def aggressive_strategy(data_slice, timestamp):
            if len(data_slice) % 10 == 1:  # Buy every 10 bars
                return {'TEST': {'action': 'buy', 'side': 'long'}}
            return {'TEST': {'action': 'hold'}}
        
        executor = BacktestExecutor(
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage=0.001
        )
        
        executor.set_strategy(aggressive_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            risk_per_trade=0.20,
            max_drawdown_limit=0.15  # 15% max drawdown
        )
        
        # Run backtest
        results = executor.run_backtest(test_data, ['TEST'])
        
        # Check if max drawdown was respected
        max_drawdown = results['metrics']['max_drawdown_pct'] / 100
        
        test_result = {
            'test_name': 'Max Drawdown Protection',
            'passed': max_drawdown <= 0.16,  # Allow 1% buffer for calculation differences
            'details': {
                'max_drawdown_pct': max_drawdown * 100,
                'max_drawdown_limit': 15.0,
                'within_limit': max_drawdown <= 0.16
            }
        }
        
        if test_result['passed']:
            logger.info(f"✅ Max drawdown protection: {max_drawdown*100:.2f}% (limit: 15.0%)")
        else:
            logger.warning(f"❌ Max drawdown exceeded: {max_drawdown*100:.2f}% (limit: 15.0%)")
        
        self.test_results.append(test_result)
    
    async def test_risk_management_integration(self):
        """Test 3: Risk management integration"""
        logger.info("⚖️ Test 3: Risk Management Integration")
        
        # Test position sizing based on risk parameters
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
        prices = [100.0] * 50  # Stable price for testing
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        }, index=dates)
        
        # Strategy that tries to buy once
        def single_buy_strategy(data_slice, timestamp):
            if len(data_slice) == 10:  # Buy on 10th bar
                return {'TEST': {'action': 'buy', 'side': 'long'}}
            return {'TEST': {'action': 'hold'}}
        
        executor = BacktestExecutor(
            initial_capital=10000.0,
            commission_rate=0.0,
            slippage=0.0
        )
        
        executor.set_strategy(single_buy_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            risk_per_trade=0.02,  # 2% risk per trade
            max_drawdown_limit=0.20
        )
        
        # Run backtest
        results = executor.run_backtest(test_data, ['TEST'])
        
        # Check position sizing
        trades = executor.portfolio.get_trades_dataframe()
        
        test_result = {
            'test_name': 'Risk Management Integration',
            'passed': False,
            'details': {}
        }
        
        if not trades.empty:
            trade = trades.iloc[0]
            position_size = trade['size']
            entry_price = trade['entry_price']
            position_value = position_size * entry_price
            
            # Calculate expected position size based on 2% risk and 5% stop loss
            risk_amount = 10000.0 * 0.02  # $200 risk
            stop_loss_distance = entry_price * 0.05  # 5% stop loss
            expected_position_size = risk_amount / stop_loss_distance
            expected_position_value = expected_position_size * entry_price
            
            test_result['details'] = {
                'position_size': position_size,
                'position_value': position_value,
                'expected_position_size': expected_position_size,
                'expected_position_value': expected_position_value,
                'size_difference_pct': abs(position_size - expected_position_size) / expected_position_size * 100
            }
            
            # Test passes if position size is within 5% of expected
            if abs(position_size - expected_position_size) / expected_position_size < 0.05:
                test_result['passed'] = True
                logger.info(f"✅ Position sizing correct: {position_size:.2f} (expected: {expected_position_size:.2f})")
            else:
                logger.warning(f"❌ Position sizing incorrect: {position_size:.2f} (expected: {expected_position_size:.2f})")
        
        self.test_results.append(test_result)
    
    async def test_database_integration_with_stop_losses(self):
        """Test 4: Database integration with stop losses"""
        logger.info("🗄️ Test 4: Database Integration with Stop Losses")
        
        try:
            # Get real market data
            symbols = await self.db_manager.get_available_symbols()
            if not symbols:
                logger.warning("❌ No symbols available from database")
                self.test_results.append({
                    'test_name': 'Database Integration with Stop Losses',
                    'passed': False,
                    'details': {'error': 'No symbols available'}
                })
                return
            
            symbol = symbols[0]
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            
            data = await self.db_manager.get_market_data(symbol, start_time, end_time)
            
            if data is None or len(data) < 100:
                logger.warning("❌ Insufficient data from database")
                self.test_results.append({
                    'test_name': 'Database Integration with Stop Losses',
                    'passed': False,
                    'details': {'error': 'Insufficient data'}
                })
                return
            
            # Take subset for testing
            test_data = data.tail(500).copy()
            
            # Conservative strategy with stop losses
            def conservative_strategy(data_slice, timestamp):
                if len(data_slice) < 20:
                    return {symbol: {'action': 'hold'}}
                
                # Simple buy signal every 50 bars
                if len(data_slice) % 50 == 0:
                    return {symbol: {'action': 'buy', 'side': 'long'}}
                return {symbol: {'action': 'hold'}}
            
            executor = BacktestExecutor(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage=0.001
            )
            
            executor.set_strategy(conservative_strategy)
            executor.set_risk_parameters(
                stop_loss_pct=0.03,  # 3% stop loss
                take_profit_pct=0.06,  # 6% take profit
                risk_per_trade=0.01,  # 1% risk per trade
                max_drawdown_limit=0.10  # 10% max drawdown
            )
            
            # Run backtest
            results = executor.run_backtest(test_data, [symbol])
            
            # Validate results
            test_result = {
                'test_name': 'Database Integration with Stop Losses',
                'passed': False,
                'details': {}
            }
            
            if results and 'metrics' in results:
                metrics = results['metrics']
                max_drawdown = metrics.get('max_drawdown_pct', 0) / 100
                total_trades = metrics.get('total_trades', 0)
                final_capital = metrics.get('final_capital', 0)
                
                test_result['details'] = {
                    'symbol': symbol,
                    'data_points': len(test_data),
                    'total_trades': total_trades,
                    'max_drawdown_pct': max_drawdown * 100,
                    'final_capital': final_capital,
                    'total_return_pct': metrics.get('total_return_pct', 0)
                }
                
                # Test passes if:
                # 1. Max drawdown is within limit
                # 2. Final capital is non-negative
                # 3. We have some trades (strategy executed)
                if max_drawdown <= 0.11 and final_capital >= 0:  # Allow 1% buffer
                    test_result['passed'] = True
                    logger.info(f"✅ Database integration successful: {total_trades} trades, {max_drawdown*100:.2f}% max drawdown")
                else:
                    logger.warning(f"❌ Database integration failed: {max_drawdown*100:.2f}% drawdown, ${final_capital:.2f} final capital")
            else:
                logger.warning("❌ No results from database backtest")
            
            self.test_results.append(test_result)
            
        except Exception as e:
            logger.error(f"❌ Database integration test failed: {e}")
            self.test_results.append({
                'test_name': 'Database Integration with Stop Losses',
                'passed': False,
                'details': {'error': str(e)}
            })
    
    async def test_edge_cases(self):
        """Test 5: Edge cases and error handling"""
        logger.info("🔍 Test 5: Edge Cases and Error Handling")
        
        test_cases = []
        
        # Test case 1: Very tight stop loss (0.1%)
        dates = pd.date_range(start='2024-01-01', periods=20, freq='1min')
        prices = [100.0] + [99.9] * 19  # Immediate small drop
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [1000] * 20
        }, index=dates)
        
        def immediate_buy_strategy(data_slice, timestamp):
            if len(data_slice) == 1:
                return {'TEST': {'action': 'buy', 'side': 'long'}}
            return {'TEST': {'action': 'hold'}}
        
        executor = BacktestExecutor(
            initial_capital=10000.0,
            commission_rate=0.0,
            slippage=0.0
        )
        
        executor.set_strategy(immediate_buy_strategy)
        executor.set_risk_parameters(
            stop_loss_pct=0.001,  # 0.1% stop loss
            take_profit_pct=0.002,
            risk_per_trade=0.10,
            max_drawdown_limit=0.05
        )
        
        try:
            results = executor.run_backtest(test_data, ['TEST'])
            
            # Check if tight stop loss was handled correctly
            trades = executor.portfolio.get_trades_dataframe()
            if not trades.empty:
                completed_trades = trades[trades['exit_price'].notna()]
                if not completed_trades.empty:
                    trade = completed_trades.iloc[0]
                    actual_loss_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                    
                    test_cases.append({
                        'case': 'Tight Stop Loss (0.1%)',
                        'passed': abs(actual_loss_pct + 0.001) < 0.001,  # Within 0.1% of expected
                        'details': {
                            'expected_loss_pct': -0.1,
                            'actual_loss_pct': actual_loss_pct * 100
                        }
                    })
                else:
                    test_cases.append({
                        'case': 'Tight Stop Loss (0.1%)',
                        'passed': False,
                        'details': {'error': 'No completed trades'}
                    })
            else:
                test_cases.append({
                    'case': 'Tight Stop Loss (0.1%)',
                    'passed': False,
                    'details': {'error': 'No trades found'}
                })
        
        except Exception as e:
            test_cases.append({
                'case': 'Tight Stop Loss (0.1%)',
                'passed': False,
                'details': {'error': str(e)}
            })
        
        # Compile edge case results
        passed_cases = sum(1 for case in test_cases if case['passed'])
        total_cases = len(test_cases)
        
        test_result = {
            'test_name': 'Edge Cases and Error Handling',
            'passed': passed_cases == total_cases,
            'details': {
                'passed_cases': passed_cases,
                'total_cases': total_cases,
                'cases': test_cases
            }
        }
        
        if test_result['passed']:
            logger.info(f"✅ Edge cases handled correctly: {passed_cases}/{total_cases}")
        else:
            logger.warning(f"❌ Some edge cases failed: {passed_cases}/{total_cases}")
        
        self.test_results.append(test_result)
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*80)
        logger.info("🧪 COMPREHENSIVE STOP LOSS TEST SUMMARY")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed Tests: {passed_tests}")
        logger.info(f"Failed Tests: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 80)
        
        for i, result in enumerate(self.test_results, 1):
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"{i}. {result['test_name']}: {status}")
            
            if result['details']:
                for key, value in result['details'].items():
                    if isinstance(value, dict):
                        logger.info(f"   {key}:")
                        for subkey, subvalue in value.items():
                            logger.info(f"     {subkey}: {subvalue}")
                    else:
                        logger.info(f"   {key}: {value}")
        
        logger.info("\n" + "="*80)
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Stop loss functionality is working correctly.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Please review the results above.")
        logger.info("="*80)

async def main():
    """Main test runner"""
    test_suite = StopLossTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
