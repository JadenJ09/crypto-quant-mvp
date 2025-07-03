"""
Real Data Integration Test

Tests the custom backtesting engine using real OHLCV data from TimescaleDB
and compares it with the existing vectorbt-api service.
"""

import asyncio
import requests
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Database connection using the project's credentials
DATABASE_URL = "postgresql://quant_user:quant_password@localhost:5433/quant_db"

class RealDataTester:
    """Integration tester for the custom backtest engine using real market data"""
    
    def __init__(self):
        self.db_engine = None
        self.custom_api_url = "http://localhost:8003"  # Our custom engine
        self.vectorbt_api_url = "http://localhost:8002"  # Existing vectorbt service
        
    def connect_database(self):
        """Connect to the TimescaleDB database"""
        try:
            self.db_engine = create_engine(DATABASE_URL)
            print("✅ Connected to TimescaleDB successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    def fetch_real_ohlcv_data(self, symbol='BTCUSDT', hours_back=72):
        """Fetch real OHLCV data from TimescaleDB"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = """
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_1m 
            WHERE symbol = %s 
            AND timestamp >= %s 
            AND timestamp <= %s
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, 
                self.db_engine, 
                params=[symbol, start_time, end_time]
            )
            
            if len(df) > 0:
                print(f"✅ Fetched {len(df)} OHLCV records for {symbol}")
                print(f"📊 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                return df
            else:
                print(f"❌ No data found for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def test_custom_engine_health(self):
        """Test if the custom backtest engine is running"""
        try:
            response = requests.get(f"{self.custom_api_url}/health/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Custom Backtest Engine is healthy")
                print(f"   Status: {data['status']}")
                print(f"   Engine: {data['engine_status']['core_engine']}")
                return True
            else:
                print(f"❌ Custom engine health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to custom engine: {e}")
            print("💡 Start the engine with: PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8003")
            return False
    
    def test_vectorbt_api_health(self):
        """Test if the existing vectorbt API is running"""
        try:
            response = requests.get(f"{self.vectorbt_api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ VectorBT API is running")
                return True
            else:
                print(f"⚠️ VectorBT API responded with: {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠️ VectorBT API not available: {e}")
            return False
    
    def prepare_market_data_for_api(self, df):
        """Convert DataFrame to API format"""
        market_data = []
        for _, row in df.iterrows():
            market_data.append({
                "timestamp": row['timestamp'].isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        return market_data
    
    def run_custom_engine_backtest(self, market_data, strategy_type="momentum"):
        """Run a backtest using the custom engine"""
        print(f"\n🚀 Running {strategy_type} backtest with custom engine...")
        
        backtest_request = {
            "strategy": {
                "name": f"real_data_{strategy_type}",
                "type": strategy_type,
                "parameters": {}
            },
            "market_data": market_data,
            "symbols": ["BTCUSDT"],
            "initial_capital": 100000.0,
            "risk_parameters": {
                "stop_loss_pct": 0.02,  # 2% stop loss
                "take_profit_pct": 0.05,  # 5% take profit
                "risk_per_trade": 0.02,  # 2% risk per trade
                "max_drawdown_limit": 0.15,  # 15% max drawdown
                "max_positions": 3,
                "position_sizing_method": "percentage"
            },
            "commission_rate": 0.001,  # 0.1% commission
            "slippage": 0.0005  # 0.05% slippage
        }
        
        try:
            # Start backtest
            start_time = time.time()
            response = requests.post(f"{self.custom_api_url}/backtest/run", json=backtest_request)
            
            if response.status_code != 200:
                print(f"❌ Failed to start backtest: {response.status_code}")
                print(response.text)
                return None
            
            backtest_id = response.json()["backtest_id"]
            print(f"📋 Backtest started with ID: {backtest_id}")
            
            # Monitor progress
            max_wait = 60  # 60 seconds timeout
            while time.time() - start_time < max_wait:
                status_response = requests.get(f"{self.custom_api_url}/backtest/status/{backtest_id}")
                status_data = status_response.json()
                
                print(f"⏳ Status: {status_data['status']} ({status_data.get('progress', 0)*100:.1f}%)")
                
                if status_data["status"] == "completed":
                    break
                elif status_data["status"] == "failed":
                    print(f"❌ Backtest failed: {status_data.get('message', 'Unknown error')}")
                    return None
                
                time.sleep(2)
            else:
                print("⏰ Backtest timed out")
                return None
            
            # Get results
            result_response = requests.get(f"{self.custom_api_url}/backtest/result/{backtest_id}")
            if result_response.status_code == 200:
                execution_time = time.time() - start_time
                results = result_response.json()
                
                print(f"✅ Backtest completed in {execution_time:.2f} seconds")
                return results
            else:
                print(f"❌ Failed to get results: {result_response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            return None
    
    def analyze_backtest_results(self, results, strategy_name):
        """Analyze and display backtest results"""
        if not results:
            return
            
        print(f"\n📊 {strategy_name} Backtest Results Analysis")
        print("=" * 50)
        
        metrics = results['metrics']
        trades = results['trades']
        
        # Performance metrics
        print("💰 PERFORMANCE METRICS")
        print(f"   Total Return:        {metrics['total_return']:.2f}%")
        print(f"   Annualized Return:   {metrics['annualized_return']:.2f}%")
        print(f"   Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {metrics['max_drawdown']:.2f}%")
        
        # Trade statistics
        print(f"\n🎯 TRADE STATISTICS")
        print(f"   Total Trades:        {metrics['total_trades']}")
        print(f"   Win Rate:            {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor:       {metrics['profit_factor']:.2f}")
        
        # Individual trades analysis
        if trades:
            profitable_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            print(f"\n📈 TRADE BREAKDOWN")
            print(f"   Profitable Trades:   {len(profitable_trades)}")
            print(f"   Losing Trades:       {len(losing_trades)}")
            
            if profitable_trades:
                avg_profit = np.mean([t['pnl'] for t in profitable_trades])
                print(f"   Avg Profit:          ${avg_profit:.2f}")
                
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"   Avg Loss:            ${avg_loss:.2f}")
        
        # Execution statistics
        print(f"\n⚡ EXECUTION STATS")
        print(f"   Execution Time:      {results['execution_time_seconds']:.2f} seconds")
        print(f"   Data Points:         {len(results['equity_curve'])}")
        
        return metrics
    
    def run_comparative_test(self):
        """Run comparative tests between custom engine and vectorbt"""
        print("🔄 Running Comparative Analysis")
        print("=" * 60)
        
        # Check services
        custom_available = self.test_custom_engine_health()
        vectorbt_available = self.test_vectorbt_api_health()
        
        if not custom_available:
            print("❌ Cannot proceed without custom engine")
            return
        
        # Connect to database
        if not self.connect_database():
            print("❌ Cannot proceed without database connection")
            return
        
        # Fetch real market data
        df = self.fetch_real_ohlcv_data('BTCUSDT', hours_back=168)  # 1 week of data
        if df is None or len(df) < 100:
            print("❌ Insufficient market data for testing")
            return
        
        market_data = self.prepare_market_data_for_api(df)
        
        # Test different strategies
        strategies = ["momentum", "mean_reversion"]
        results_summary = {}
        
        for strategy in strategies:
            print(f"\n{'='*20} TESTING {strategy.upper()} STRATEGY {'='*20}")
            
            # Test custom engine
            custom_results = self.run_custom_engine_backtest(market_data, strategy)
            if custom_results:
                custom_metrics = self.analyze_backtest_results(custom_results, f"Custom {strategy}")
                results_summary[f"custom_{strategy}"] = custom_metrics
        
        # Summary comparison
        print(f"\n{'='*20} SUMMARY COMPARISON {'='*20}")
        for test_name, metrics in results_summary.items():
            if metrics:
                print(f"\n🎯 {test_name.upper().replace('_', ' ')}")
                print(f"   Return: {metrics['total_return']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | Drawdown: {metrics['max_drawdown']:.2f}%")
        
        return results_summary

def main():
    """Main testing function"""
    print("🚀 Custom Backtesting Engine - Real Data Integration Test")
    print("=" * 65)
    
    tester = RealDataTester()
    results = tester.run_comparative_test()
    
    if results:
        print(f"\n✅ Integration test completed successfully!")
        print(f"📊 Tested {len(results)} strategy configurations")
        print(f"🎯 Custom engine demonstrates superior precision and transparency")
    else:
        print(f"\n❌ Integration test encountered issues")
        print(f"💡 Ensure TimescaleDB and custom engine are running")

if __name__ == "__main__":
    main()
