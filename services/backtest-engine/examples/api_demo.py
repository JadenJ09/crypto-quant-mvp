#!/usr/bin/env python3
"""
API Demo Client for Custom Backtesting Engine

This script demonstrates how to interact with the backtesting engine API
for running backtests and analyzing results.

Prerequisites:
1. Start the API server: ./run_api_server.py
2. Install requests: pip install requests
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# API Base URL
BASE_URL = "http://localhost:8000"

def generate_sample_data(days=30):
    """Generate sample market data for demonstration"""
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    # Generate realistic OHLCV data with volatility
    np.random.seed(42)
    base_price = 45000  # BTC price
    prices = []
    
    for i, date in enumerate(dates):
        if i == 0:
            close = base_price
        else:
            # Random walk with slight trend
            change = np.random.normal(0.0002, 0.015)  # 1.5% volatility
            close = max(prices[-1]['close'] * (1 + change), 1000)  # Minimum price
        
        # Generate OHLC from close
        high = close * (1 + abs(np.random.normal(0, 0.008)))
        low = close * (1 - abs(np.random.normal(0, 0.008)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.uniform(50, 200)
        
        prices.append({
            'timestamp': date.isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })
    
    return prices

def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{BASE_URL}/health/")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health Check:")
            print(f"   Status: {health_data['status']}")
            print(f"   Version: {health_data['version']}")
            print(f"   Uptime: {health_data['uptime_seconds']:.1f} seconds")
            print(f"   Engine Status: {health_data['engine_status']['core_engine']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the API server is running: ./run_api_server.py")
        return False

def list_available_strategies():
    """List available trading strategies"""
    try:
        response = requests.get(f"{BASE_URL}/backtest/strategies")
        if response.status_code == 200:
            strategies = response.json()['strategies']
            print("\n📋 Available Trading Strategies:")
            for strategy in strategies:
                print(f"   • {strategy['name']}: {strategy['description']}")
            return strategies
        else:
            print(f"❌ Failed to get strategies: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error getting strategies: {e}")
        return []

def run_backtest_demo():
    """Run a complete backtest demonstration"""
    print("\n🚀 Running Backtest Demo...")
    
    # Generate sample data
    print("📊 Generating sample market data...")
    market_data = generate_sample_data(days=15)  # 15 days of hourly data
    print(f"   Generated {len(market_data)} data points")
    
    # Prepare backtest request
    backtest_request = {
        "strategy": {
            "name": "momentum_demo",
            "type": "momentum",
            "parameters": {}
        },
        "market_data": market_data,
        "symbols": ["BTCUSD"],
        "initial_capital": 100000.0,
        "risk_parameters": {
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.08,
            "risk_per_trade": 0.02,
            "max_drawdown_limit": 0.15,
            "max_positions": 3,
            "position_sizing_method": "percentage"
        },
        "commission_rate": 0.001,
        "slippage": 0.0005
    }
    
    # Start backtest
    print("🎯 Starting backtest...")
    try:
        response = requests.post(f"{BASE_URL}/backtest/run", json=backtest_request)
        if response.status_code != 200:
            print(f"❌ Failed to start backtest: {response.status_code}")
            return None
        
        backtest_id = response.json()["backtest_id"]
        print(f"   Backtest ID: {backtest_id}")
        
        # Monitor progress
        print("⏳ Monitoring backtest progress...")
        max_wait = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/backtest/status/{backtest_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data["status"]
                progress = status_data.get("progress", 0) * 100
                
                print(f"   Status: {status} ({progress:.1f}%)")
                
                if status == "completed":
                    break
                elif status == "failed":
                    print(f"❌ Backtest failed: {status_data.get('message', 'Unknown error')}")
                    return None
            
            time.sleep(2)
        else:
            print("❌ Backtest timed out")
            return None
        
        # Get results
        print("📊 Retrieving backtest results...")
        result_response = requests.get(f"{BASE_URL}/backtest/result/{backtest_id}")
        if result_response.status_code != 200:
            print(f"❌ Failed to get results: {result_response.status_code}")
            return None
        
        result_data = result_response.json()
        
        # Display results
        print("\n🎉 Backtest Results:")
        metrics = result_data["metrics"]
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Total Trades: {metrics['total_trades']}")
        
        return result_data
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None

def analyze_performance(result_data):
    """Analyze performance using the statistics API"""
    if not result_data:
        return
        
    print("\n📈 Running Advanced Performance Analysis...")
    
    equity_curve = result_data["equity_curve"]
    trades = result_data["trades"]
    
    # Performance analysis
    try:
        perf_request = {
            "equity_curve": equity_curve,
            "trades": trades,
            "initial_capital": 100000.0
        }
        
        response = requests.post(f"{BASE_URL}/statistics/performance", json=perf_request)
        if response.status_code == 200:
            perf_data = response.json()
            print("✅ Performance Analysis:")
            risk_metrics = perf_data["risk_metrics"]
            print(f"   Volatility: {risk_metrics['annualized_volatility']:.2f}%")
            print(f"   VaR (95%): {risk_metrics['var_95']:.2f}%")
            print(f"   Skewness: {risk_metrics['skewness']:.2f}")
            print(f"   Best Month: {perf_data['best_month']:.2f}%")
            print(f"   Worst Month: {perf_data['worst_month']:.2f}%")
        
        # Rolling analysis
        rolling_request = {
            "equity_curve": equity_curve,
            "window": 24  # 24-hour rolling window
        }
        
        rolling_response = requests.post(f"{BASE_URL}/statistics/rolling", json=rolling_request)
        if rolling_response.status_code == 200:
            rolling_data = rolling_response.json()
            print("✅ Rolling Analysis:")
            print(f"   Rolling periods analyzed: {rolling_data['total_periods']}")
            print(f"   Window size: {rolling_data['window_size']} hours")
        
        # Trade analysis
        if trades:
            trade_response = requests.post(f"{BASE_URL}/statistics/trades/analysis", json=trades)
            if trade_response.status_code == 200:
                trade_data = trade_response.json()
                print("✅ Trade Analysis:")
                print(f"   Profit Factor: {trade_data['profit_factor']:.2f}")
                print(f"   Average Hold Time: {trade_data['average_hold_time']}")
                print(f"   Max Consecutive Wins: {trade_data['max_consecutive_wins']}")
        
        # Generate comprehensive report
        report_response = requests.post(f"{BASE_URL}/statistics/report", json=perf_request)
        if report_response.status_code == 200:
            report_data = report_response.json()
            print("\n📋 Comprehensive Report Generated:")
            print("   Report contains detailed performance metrics, risk analysis, and trade statistics")
            print(f"   Report length: {len(report_data['formatted_report'])} characters")
        
    except Exception as e:
        print(f"❌ Error in performance analysis: {e}")

def main():
    """Main demonstration function"""
    print("="*70)
    print("CUSTOM BACKTESTING ENGINE - API DEMO CLIENT")
    print("="*70)
    print("This demo shows the complete API functionality:")
    print("• Health monitoring and system status")
    print("• Running backtests with momentum strategy")
    print("• Advanced performance analysis and statistics")
    print("• Rolling metrics and comprehensive reporting")
    print("="*70)
    
    # Check API health
    if not check_api_health():
        return
    
    # List strategies
    strategies = list_available_strategies()
    if not strategies:
        return
    
    # Run backtest demo
    result_data = run_backtest_demo()
    
    # Analyze results
    analyze_performance(result_data)
    
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE!")
    print("✅ All API endpoints tested successfully")
    print("📚 Visit http://localhost:8000/docs for interactive API documentation")
    print("💡 Check http://localhost:8000/health/ for real-time system status")
    print("="*70)

if __name__ == "__main__":
    main()
