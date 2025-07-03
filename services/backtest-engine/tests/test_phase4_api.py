"""
Phase 4 API Integration Tests

Tests for the FastAPI implementation of the custom backtesting engine.
Validates all endpoints, error handling, and integration with core engine.
"""

import pytest
import asyncio
import json
import time
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import the FastAPI app
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from api.main import app

# Create test client
client = TestClient(app)


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(100)]
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 30000
    prices = []
    
    for i in range(len(dates)):
        # Simple random walk with some trend
        if i == 0:
            close = base_price
        else:
            change = np.random.normal(0, 0.02)  # 2% volatility
            close = prices[-1]['close'] * (1 + change)
        
        # Generate OHLC from close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.uniform(100, 1000)
        
        prices.append({
            'timestamp': dates[i].isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })
    
    return prices


@pytest.fixture
def sample_backtest_request(sample_market_data):
    """Generate a sample backtest request"""
    return {
        "strategy": {
            "name": "test_momentum",
            "type": "momentum",
            "parameters": {}
        },
        "market_data": sample_market_data,
        "symbols": ["BTCUSD"],
        "initial_capital": 100000.0,
        "risk_parameters": {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "risk_per_trade": 0.02,
            "max_drawdown_limit": 0.20,
            "max_positions": 5,
            "position_sizing_method": "percentage"
        },
        "commission_rate": 0.001,
        "slippage": 0.001
    }


class TestHealthAPI:
    """Test suite for health check endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Custom Backtesting Engine API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        
        # Check features
        features = data["features"]
        assert features["precise_execution"] is True
        assert features["advanced_risk_management"] is True
        assert features["comprehensive_statistics"] is True
        
        # Check phases
        assert "Phase 1: Core Engine" in data["phases_complete"]
        assert "Phase 2: Risk Management" in data["phases_complete"]
        assert "Phase 3: Statistics Engine" in data["phases_complete"]
        assert data["current_phase"] == "Phase 4: API Integration"
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["version"] == "1.0.0"
        
        # Check system info
        system_info = data["system_info"]
        assert "cpu_percent" in system_info
        assert "memory_percent" in system_info
        assert "python_version" in system_info
        
        # Check engine status
        engine_status = data["engine_status"]
        assert engine_status["core_engine"] == "✅ Active"
        assert engine_status["risk_management"] == "✅ Active" 
        assert engine_status["statistics_engine"] == "✅ Active"
    
    def test_status_endpoint(self):
        """Test the status endpoint"""
        response = client.get("/health/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "message" in data


class TestBacktestAPI:
    """Test suite for backtest endpoints"""
    
    def test_list_strategies(self):
        """Test listing available strategies"""
        response = client.get("/backtest/strategies")
        assert response.status_code == 200
        
        data = response.json()
        strategies = data["strategies"]
        
        # Should have at least momentum and mean_reversion
        strategy_names = [s["name"] for s in strategies]
        assert "momentum" in strategy_names
        assert "mean_reversion" in strategy_names
        assert "custom" in strategy_names
    
    def test_run_backtest_momentum(self, sample_backtest_request):
        """Test running a momentum strategy backtest"""
        response = client.post("/backtest/run", json=sample_backtest_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "backtest_id" in data
        assert data["status"] == "started"
        
        backtest_id = data["backtest_id"]
        
        # Wait for completion (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = client.get(f"/backtest/status/{backtest_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Backtest failed: {status_data.get('message', 'Unknown error')}")
            
            time.sleep(1)
        else:
            pytest.fail("Backtest timed out")
        
        # Get results
        result_response = client.get(f"/backtest/result/{backtest_id}")
        assert result_response.status_code == 200
        
        result_data = result_response.json()
        assert result_data["status"] == "completed"
        
        # Validate metrics
        metrics = result_data["metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        
        # Validate trades
        trades = result_data["trades"]
        assert isinstance(trades, list)
        
        # Validate equity curve
        equity_curve = result_data["equity_curve"]
        assert isinstance(equity_curve, list)
        assert len(equity_curve) > 0
        
        # Validate statistics report
        assert "statistics_report" in result_data
        assert len(result_data["statistics_report"]) > 0
    
    def test_run_backtest_mean_reversion(self, sample_backtest_request):
        """Test running a mean reversion strategy backtest"""
        # Modify request for mean reversion
        sample_backtest_request["strategy"]["type"] = "mean_reversion"
        sample_backtest_request["strategy"]["name"] = "test_mean_reversion"
        
        response = client.post("/backtest/run", json=sample_backtest_request)
        assert response.status_code == 200
        
        data = response.json()
        backtest_id = data["backtest_id"]
        
        # Wait for completion
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = client.get(f"/backtest/status/{backtest_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Mean reversion backtest failed: {status_data.get('message')}")
            
            time.sleep(1)
        
        # Validate results exist
        result_response = client.get(f"/backtest/result/{backtest_id}")
        assert result_response.status_code == 200
    
    def test_backtest_validation(self):
        """Test backtest request validation"""
        # Test invalid request - no market data
        invalid_request = {
            "strategy": {"name": "test", "type": "momentum"},
            "market_data": [],
            "symbols": ["BTCUSD"],
            "initial_capital": 100000.0
        }
        
        response = client.post("/backtest/run", json=invalid_request)
        # Should either succeed (empty backtest) or return error
        # The key is that it handles the case gracefully
        assert response.status_code in [200, 400, 422]
        
        # Test invalid risk parameters
        invalid_risk_request = {
            "strategy": {"name": "test", "type": "momentum"},
            "market_data": [
                {
                    "timestamp": "2023-01-01T00:00:00",
                    "open": 30000, "high": 30100, "low": 29900, "close": 30050, "volume": 100
                }
            ],
            "symbols": ["BTCUSD"],
            "initial_capital": 100000.0,
            "risk_parameters": {
                "stop_loss_pct": 1.5,  # Invalid - too high
                "take_profit_pct": 0.10
            }
        }
        
        response = client.post("/backtest/run", json=invalid_risk_request)
        assert response.status_code == 422  # Validation error
    
    def test_backtest_status_not_found(self):
        """Test getting status for non-existent backtest"""
        response = client.get("/backtest/status/non-existent-id")
        assert response.status_code == 404
    
    def test_backtest_result_not_found(self):
        """Test getting result for non-existent backtest"""
        response = client.get("/backtest/result/non-existent-id")
        assert response.status_code == 404
    
    def test_list_backtests(self):
        """Test listing backtests"""
        response = client.get("/backtest/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_backtests" in data
        assert "backtests" in data
        assert isinstance(data["backtests"], list)
    
    def test_delete_backtest(self, sample_backtest_request):
        """Test deleting a backtest"""
        # First create a backtest
        response = client.post("/backtest/run", json=sample_backtest_request)
        assert response.status_code == 200
        
        backtest_id = response.json()["backtest_id"]
        
        # Delete it
        delete_response = client.delete(f"/backtest/{backtest_id}")
        assert delete_response.status_code == 200
        
        # Verify it's gone
        status_response = client.get(f"/backtest/status/{backtest_id}")
        assert status_response.status_code == 404


class TestStatisticsAPI:
    """Test suite for statistics endpoints"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Generate sample equity curve data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic equity curve
        initial_value = 100000
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        cumulative_returns = np.cumprod(1 + returns)
        values = initial_value * cumulative_returns
        
        return [
            {"timestamp": str(date), "value": float(value)}
            for date, value in zip(dates, values)
        ]
    
    @pytest.fixture
    def sample_trades(self):
        """Generate sample trade data"""
        return [
            {
                "entry_time": "2023-01-01T09:00:00",
                "exit_time": "2023-01-01T15:00:00",
                "symbol": "BTCUSD",
                "side": "long",
                "size": 1.0,
                "entry_price": 30000.0,
                "exit_price": 30500.0,
                "pnl": 500.0,
                "commission": 15.0
            },
            {
                "entry_time": "2023-01-02T09:00:00",
                "exit_time": "2023-01-02T14:00:00",
                "symbol": "BTCUSD",
                "side": "long",
                "size": 1.5,
                "entry_price": 30500.0,
                "exit_price": 30200.0,
                "pnl": -450.0,
                "commission": 22.5
            }
        ]
    
    def test_performance_analysis(self, sample_equity_curve, sample_trades):
        """Test performance analysis endpoint"""
        request_data = {
            "equity_curve": sample_equity_curve,
            "trades": sample_trades,
            "initial_capital": 100000.0
        }
        
        response = client.post("/statistics/performance", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check performance metrics
        assert "total_return" in data
        assert "annualized_return" in data
        assert "cumulative_return" in data
        
        # Check risk metrics
        risk_metrics = data["risk_metrics"]
        assert "max_drawdown" in risk_metrics
        assert "volatility" in risk_metrics
        assert "var_95" in risk_metrics
        
        # Check trade statistics
        trade_stats = data["trade_statistics"]
        assert "total_trades" in trade_stats
        assert "win_rate" in trade_stats
        assert "profit_factor" in trade_stats
    
    def test_rolling_analysis(self, sample_equity_curve):
        """Test rolling analysis endpoint"""
        request_data = {
            "equity_curve": sample_equity_curve,
            "window": 30
        }
        
        response = client.post("/statistics/rolling", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "rolling_statistics" in data
        assert "time_series" in data
        
        # Verify data structure
        rolling_stats = data["rolling_statistics"]
        assert "returns" in rolling_stats
        assert "volatility" in rolling_stats
        assert "sharpe_ratio" in rolling_stats
    
    def test_trade_analysis(self, sample_trades):
        """Test trade analysis endpoint"""
        response = client.post("/statistics/trades/analysis", json=sample_trades)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check basic trade statistics
        assert data["total_trades"] == 2
        assert data["long_trades"] == 2
        assert data["short_trades"] == 0
        
        # Check calculated metrics
        assert "win_rate" in data
        assert "profit_factor" in data
        assert "average_hold_time" in data
    
    def test_comprehensive_report(self, sample_equity_curve, sample_trades):
        """Test comprehensive report generation"""
        request_data = {
            "equity_curve": sample_equity_curve,
            "trades": sample_trades,
            "initial_capital": 100000.0
        }
        
        response = client.post("/statistics/report", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "formatted_report" in data
        assert len(data["formatted_report"]) > 100  # Should be a substantial report
        
        # Report should contain key sections
        report = data["formatted_report"]
        assert "PERFORMANCE REPORT" in report or "Performance" in report
        assert "RISK" in report or "Risk" in report
        assert "TRADE" in report or "Trade" in report
    
    def test_statistics_validation(self):
        """Test statistics endpoint validation"""
        # Test empty equity curve
        invalid_request = {
            "equity_curve": [],
            "trades": [],
            "initial_capital": 100000.0
        }
        
        response = client.post("/statistics/performance", json=invalid_request)
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


class TestAPIIntegration:
    """Test suite for end-to-end API integration"""
    
    def test_full_backtest_to_statistics_workflow(self, sample_backtest_request):
        """Test complete workflow from backtest to statistics analysis"""
        # Step 1: Run backtest
        backtest_response = client.post("/backtest/run", json=sample_backtest_request)
        assert backtest_response.status_code == 200
        
        backtest_id = backtest_response.json()["backtest_id"]
        
        # Step 2: Wait for completion
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = client.get(f"/backtest/status/{backtest_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Backtest failed: {status_data.get('message')}")
            
            time.sleep(1)
        
        # Step 3: Get backtest results
        result_response = client.get(f"/backtest/result/{backtest_id}")
        assert result_response.status_code == 200
        
        result_data = result_response.json()
        equity_curve = result_data["equity_curve"]
        trades = result_data["trades"]
        
        # Step 4: Run additional statistics analysis
        stats_request = {
            "equity_curve": equity_curve,
            "trades": trades,
            "initial_capital": sample_backtest_request["initial_capital"]
        }
        
        # Performance analysis
        perf_response = client.post("/statistics/performance", json=stats_request)
        assert perf_response.status_code == 200
        
        # Rolling analysis
        rolling_request = {
            "equity_curve": equity_curve,
            "window": 10
        }
        rolling_response = client.post("/statistics/rolling", json=rolling_request)
        assert rolling_response.status_code == 200
        
        # Trade analysis
        trade_analysis_response = client.post("/statistics/trades/analysis", json=trades)
        assert trade_analysis_response.status_code == 200
        
        # Comprehensive report
        report_response = client.post("/statistics/report", json=stats_request)
        assert report_response.status_code == 200
    
    def test_concurrent_backtests(self, sample_backtest_request):
        """Test running multiple backtests concurrently"""
        # Start multiple backtests
        backtest_ids = []
        
        for i in range(3):
            # Modify strategy name for each
            request = sample_backtest_request.copy()
            request["strategy"]["name"] = f"concurrent_test_{i}"
            
            response = client.post("/backtest/run", json=request)
            assert response.status_code == 200
            backtest_ids.append(response.json()["backtest_id"])
        
        # Wait for all to complete
        max_wait = 60  # More time for concurrent tests
        start_time = time.time()
        completed = set()
        
        while time.time() - start_time < max_wait and len(completed) < len(backtest_ids):
            for backtest_id in backtest_ids:
                if backtest_id not in completed:
                    status_response = client.get(f"/backtest/status/{backtest_id}")
                    status_data = status_response.json()
                    
                    if status_data["status"] == "completed":
                        completed.add(backtest_id)
                    elif status_data["status"] == "failed":
                        pytest.fail(f"Concurrent backtest {backtest_id} failed")
            
            time.sleep(2)
        
        assert len(completed) == len(backtest_ids), "Not all concurrent backtests completed"
        
        # Verify all results are accessible
        for backtest_id in backtest_ids:
            result_response = client.get(f"/backtest/result/{backtest_id}")
            assert result_response.status_code == 200
    
    def test_api_error_handling(self):
        """Test API error handling and edge cases"""
        # Test malformed request data
        response = client.post("/backtest/run", content="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        
        # Test missing required fields
        incomplete_request = {"strategy": {"name": "test"}}
        response = client.post("/backtest/run", json=incomplete_request)
        assert response.status_code == 422
        
        # Test invalid endpoints
        response = client.get("/nonexistent")
        assert response.status_code == 404


def test_api_performance_benchmark():
    """Benchmark API response times"""
    import time
    
    # Test health endpoint performance
    start = time.time()
    response = client.get("/health/")
    health_time = time.time() - start
    
    assert response.status_code == 200
    assert health_time < 1.5  # Should respond within 1.5 seconds
    
    # Test strategies endpoint performance
    start = time.time()
    response = client.get("/backtest/strategies")
    strategies_time = time.time() - start
    
    assert response.status_code == 200
    assert strategies_time < 0.5  # Should respond within 0.5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
