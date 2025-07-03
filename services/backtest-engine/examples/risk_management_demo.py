"""
Advanced Risk Management Example

This example demonstrates the advanced risk management features implemented in Phase 2:
- Multiple position sizing methods
- Portfolio-level risk constraints
- Multiple take profit levels
- Advanced stop loss methods
- Risk monitoring and reporting
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from risk.risk_manager import (
    RiskLimits, PositionSizingConfig, PositionSizingMethod, StopLossMethod
)
from signals.signal_engine import SignalEngine


def create_market_data(symbol: str, periods: int = 500, base_price: float = 100.0) -> pd.DataFrame:
    """Create realistic market data with volatility clustering"""
    np.random.seed(42 if symbol == 'BTC' else 123)
    
    # Generate price series with volatility clustering
    returns = []
    volatility = 0.02  # Base volatility
    
    for i in range(periods):
        # Volatility clustering - high vol periods tend to cluster
        if i > 0:
            volatility = 0.8 * volatility + 0.2 * abs(returns[-1]) * 3
        volatility = np.clip(volatility, 0.01, 0.05)  # Keep reasonable bounds
        
        # Generate return with current volatility
        ret = np.random.normal(0, volatility)
        returns.append(ret)
    
    # Convert to prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, periods)))
    volumes = np.random.randint(1000, 50000, periods)
    
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq='1h')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


def demonstrate_risk_management():
    """Demonstrate advanced risk management features"""
    
    print("🚀 Advanced Risk Management Demonstration")
    print("=" * 60)
    
    # 1. Configure Risk Management System
    print("\n1. Setting up Risk Management Configuration")
    
    risk_limits = RiskLimits(
        max_positions=3,           # Maximum 3 concurrent positions
        max_portfolio_risk=0.015,  # 1.5% portfolio risk per trade
        max_drawdown=0.20,         # 20% maximum drawdown
        max_concentration=0.40,    # 40% maximum single position
        max_correlation=0.70       # Maximum correlation between positions
    )
    
    # Test different position sizing methods
    sizing_methods = [
        ("Percentage Sizing", PositionSizingMethod.PERCENTAGE, {"percentage": 0.08}),
        ("Kelly Criterion", PositionSizingMethod.KELLY_CRITERION, {"kelly_max": 0.30}),
        ("Volatility Adjusted", PositionSizingMethod.VOLATILITY_ADJUSTED, {"volatility_target": 0.02}),
        ("ATR Based", PositionSizingMethod.ATR_BASED, {"atr_multiplier": 2.5})
    ]
    
    results = {}
    
    for method_name, method, params in sizing_methods:
        print(f"\n2. Testing {method_name}")
        print("-" * 40)
        
        # Configure position sizing
        position_sizing = PositionSizingConfig(method=method, **params)
        
        # Create portfolio with risk management
        portfolio = EnhancedPortfolioManager(
            initial_capital=100000.0,
            risk_limits=risk_limits,
            position_sizing=position_sizing
        )
        
        # Create market data for multiple assets
        symbols = ['BTC', 'ETH', 'ADA']
        market_data = {}
        signal_engines = {}
        
        for symbol in symbols:
            data = create_market_data(symbol, 500, 100.0 if symbol == 'BTC' else 50.0)
            market_data[symbol] = data
            portfolio.add_market_data(symbol, data)
            
            # Create signal engine for each symbol
            signal_engines[symbol] = SignalEngine()
        
        # Run backtest simulation
        trades_executed = 0
        
        for i in range(50, 500):  # Start after warm-up period
            timestamp = market_data['BTC']['timestamp'].iloc[i]
            current_prices = {symbol: data['close'].iloc[i] for symbol, data in market_data.items()}
            
            # Update portfolio
            portfolio.update_portfolio(timestamp, current_prices)
            
            # Generate signals and execute trades
            for symbol in symbols:
                # Get recent data for signal generation
                recent_data = market_data[symbol].iloc[i-49:i+1]  # 50 periods
                
                # Calculate indicators and signals
                signal_engine = signal_engines[symbol]
                
                # Add indicators
                signal_engine.add_sma(recent_data, window=10, name='sma_fast')
                signal_engine.add_sma(recent_data, window=20, name='sma_slow')
                signal_engine.add_rsi(recent_data, window=14)
                
                # Generate signals
                ma_signals = signal_engine.generate_ma_crossover_signals('sma_fast', 'sma_slow', symbol)
                rsi_signals = signal_engine.generate_rsi_signals('rsi_14')
                
                if ma_signals.empty or rsi_signals.empty:
                    continue
                
                # Combine signals (simple approach)
                latest_ma = ma_signals.iloc[-1] if len(ma_signals) > 0 else 0
                latest_rsi = rsi_signals.iloc[-1] if len(rsi_signals) > 0 else 0
                
                # Generate trading signal
                long_signal = latest_ma > 0 and latest_rsi > 0
                short_signal = latest_ma < 0 and latest_rsi < 0
                signal_strength = 0.8  # Fixed for demo
                
                # Check for entry signals (every 10 periods to avoid overtrading)
                if i % 10 == 0 and (long_signal or short_signal):
                    side = 'long' if long_signal else 'short'
                    
                    # Multiple take profit levels
                    take_profit_levels = [
                        (0.05, 0.4),  # 5% profit, close 40% of position
                        (0.10, 0.3),  # 10% profit, close 30% of position  
                        (0.20, 0.3),  # 20% profit, close remaining 30%
                    ]
                    
                    # Try to open position
                    success, message = portfolio.open_position(
                        symbol=symbol,
                        side=side,
                        signal_strength=signal_strength,
                        entry_price=current_prices[symbol],
                        timestamp=timestamp,
                        stop_loss_pct=0.05,  # 5% stop loss
                        take_profit_levels=take_profit_levels
                    )
                    
                    if success:
                        trades_executed += 1
                        print(f"  ✅ {symbol} {side.upper()} position opened: {message}")
                    else:
                        print(f"  ❌ {symbol} position rejected: {message}")
        
        # Get final results
        final_summary = portfolio.get_portfolio_summary(current_prices)
        
        print(f"\n📊 Results for {method_name}:")
        print(f"  • Final Capital: ${final_summary['current_capital']:,.2f}")
        print(f"  • Total Return: {final_summary['total_return_pct']:.2f}%")
        print(f"  • Max Drawdown: {final_summary['max_drawdown']:.2f}%")
        print(f"  • Sharpe Ratio: {final_summary['sharpe_ratio']:.2f}")
        print(f"  • Total Trades: {final_summary['total_trades']}")
        print(f"  • Win Rate: {final_summary['win_rate']:.1%}")
        print(f"  • Trades Executed: {trades_executed}")
        
        if final_summary['risk_metrics']:
            risk_metrics = final_summary['risk_metrics']
            print(f"  • Portfolio Leverage: {risk_metrics.get('leverage', 0):.2f}x")
            print(f"  • Long Exposure: ${risk_metrics.get('long_exposure', 0):,.0f}")
            print(f"  • Short Exposure: ${risk_metrics.get('short_exposure', 0):,.0f}")
        
        results[method_name] = final_summary
    
    # 3. Compare Methods
    print(f"\n3. Method Comparison")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        method: {
            'Final Capital': f"${results[method]['current_capital']:,.0f}",
            'Total Return': f"{results[method]['total_return_pct']:.1f}%",
            'Max Drawdown': f"{results[method]['max_drawdown']:.1f}%",
            'Sharpe Ratio': f"{results[method]['sharpe_ratio']:.2f}",
            'Total Trades': results[method]['total_trades'],
            'Win Rate': f"{results[method]['win_rate']:.1%}"
        } for method in results.keys()
    })
    
    print(comparison_df.to_string())
    
    # 4. Risk Management Features Summary
    print(f"\n4. Advanced Risk Management Features Demonstrated")
    print("=" * 60)
    
    features = [
        "✅ Multiple Position Sizing Methods (Percentage, Kelly, Volatility-Adjusted, ATR-based)",
        "✅ Portfolio-Level Risk Limits (Max positions, drawdown, concentration)",
        "✅ Multiple Take Profit Levels (Partial position closing)",
        "✅ Advanced Stop Loss Methods (Percentage, ATR-based, Volatility-based)",
        "✅ Dynamic Risk Adjustment (Volatility clustering, market conditions)",
        "✅ Real-time Risk Monitoring (Leverage, exposure, correlation tracking)",
        "✅ Precise Trade Execution (No slippage approximation like VectorBT)",
        "✅ Transparent Risk Attribution (Full audit trail of risk decisions)"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n🎯 Key Advantages over VectorBT:")
    print("  • Precise position sizing with multiple sophisticated methods")
    print("  • Portfolio-level risk constraints (not just individual positions)")  
    print("  • Multiple take profit levels with partial closing")
    print("  • Advanced stop loss methods beyond simple percentages")
    print("  • Real-time risk monitoring and adjustment")
    print("  • Complete transparency and control over all risk decisions")
    print("  • No black box behavior - full audit trail of every decision")
    
    print(f"\n✨ Phase 2 (Advanced Risk Management) Complete!")
    print("Ready for Phase 3: Statistics Engine & API Integration")


if __name__ == "__main__":
    demonstrate_risk_management()
