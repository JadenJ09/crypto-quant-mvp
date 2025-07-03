"""
Backtesting API endpoints

Provides REST API access to the custom backtesting engine for running
backtests, configuring strategies, and managing risk parameters.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import our custom engine
import sys
import os

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.backtest_executor import BacktestExecutor
from statistics.statistics_engine import StatisticsEngine

router = APIRouter()

# Store for running backtests
active_backtests: Dict[str, Dict] = {}
completed_backtests: Dict[str, Dict] = {}

# Thread pool for running backtests
executor = ThreadPoolExecutor(max_workers=4)


class MarketData(BaseModel):
    """Market data model"""
    timestamp: str
    open: float
    high: float  
    low: float
    close: float
    volume: float


class StrategyConfig(BaseModel):
    """Strategy configuration model"""
    name: str
    type: str = "custom"
    parameters: Dict[str, Any] = {}
    code: Optional[str] = None  # Python code for custom strategies


class RiskParameters(BaseModel):
    """Risk management parameters"""
    stop_loss_pct: float = Field(default=0.05, ge=0.001, le=0.5)
    take_profit_pct: float = Field(default=0.10, ge=0.001, le=1.0)
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)
    max_drawdown_limit: float = Field(default=0.20, ge=0.05, le=0.5)
    max_positions: int = Field(default=5, ge=1, le=20)
    position_sizing_method: str = Field(default='percentage')


class BacktestRequest(BaseModel):
    """Backtest request model"""
    strategy: StrategyConfig
    market_data: List[MarketData]
    symbols: List[str]
    initial_capital: float = Field(default=100000.0, gt=1000.0)
    risk_parameters: RiskParameters = RiskParameters()
    commission_rate: float = Field(default=0.001, ge=0.0, le=0.01)
    slippage: float = Field(default=0.001, ge=0.0, le=0.01)


class BacktestStatus(BaseModel):
    """Backtest status model"""
    id: str
    status: str  # 'running', 'completed', 'failed'
    progress: float = 0.0
    start_time: datetime
    end_time: Optional[datetime] = None
    message: str = ""


class BacktestResult(BaseModel):
    """Backtest result model"""
    id: str
    status: str
    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    statistics_report: str
    execution_time_seconds: float


def create_simple_strategy(strategy_config: StrategyConfig):
    """Create a strategy function from configuration"""
    
    if strategy_config.type == "momentum":
        def momentum_strategy(data, timestamp):
            if len(data) < 2:
                return {'BTCUSD': {'action': 'hold'}}
            
            # Simple momentum
            if data['close'].iloc[-1] > data['close'].iloc[-2]:
                return {'BTCUSD': {'action': 'buy', 'side': 'long'}}
            else:
                return {'BTCUSD': {'action': 'sell', 'side': 'short'}}
        return momentum_strategy
    
    elif strategy_config.type == "mean_reversion":
        def mean_reversion_strategy(data, timestamp):
            if len(data) < 20:
                return {'BTCUSD': {'action': 'hold'}}
            
            # Simple mean reversion
            sma = data['close'].rolling(20).mean()
            current_price = data['close'].iloc[-1]
            sma_current = sma.iloc[-1]
            
            if current_price < sma_current * 0.98:  # 2% below SMA
                return {'BTCUSD': {'action': 'buy', 'side': 'long'}}
            elif current_price > sma_current * 1.02:  # 2% above SMA
                return {'BTCUSD': {'action': 'sell', 'side': 'short'}}
            else:
                return {'BTCUSD': {'action': 'hold'}}
        return mean_reversion_strategy
    
    else:
        # Default simple strategy
        def simple_strategy(data, timestamp):
            if len(data) < 2:
                return {'BTCUSD': {'action': 'hold'}}
            
            return {'BTCUSD': {'action': 'buy', 'side': 'long'}}
        return simple_strategy


def run_backtest_sync(backtest_id: str, request: BacktestRequest):
    """Run backtest synchronously in thread pool"""
    try:
        # Update status
        active_backtests[backtest_id]['status'] = 'running'
        active_backtests[backtest_id]['progress'] = 0.1
        
        # Convert market data to DataFrame
        data_records = []
        for md in request.market_data:
            data_records.append({
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume,
                'timestamp': pd.to_datetime(md.timestamp)
            })
        
        df = pd.DataFrame(data_records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        active_backtests[backtest_id]['progress'] = 0.3
        
        # Create strategy
        strategy_func = create_simple_strategy(request.strategy)
        
        # Create backtest executor
        bt_executor = BacktestExecutor(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage=request.slippage
        )
        
        bt_executor.set_strategy(strategy_func)
        bt_executor.set_risk_parameters(
            stop_loss_pct=request.risk_parameters.stop_loss_pct,
            take_profit_pct=request.risk_parameters.take_profit_pct,
            risk_per_trade=request.risk_parameters.risk_per_trade,
            max_drawdown_limit=request.risk_parameters.max_drawdown_limit,
            max_positions=request.risk_parameters.max_positions,
            position_sizing_method=request.risk_parameters.position_sizing_method
        )
        
        active_backtests[backtest_id]['progress'] = 0.5
        
        # Run backtest
        start_time = datetime.now()
        results = bt_executor.run_backtest(df, request.symbols)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        active_backtests[backtest_id]['progress'] = 0.8
        
        # Get statistics
        portfolio = bt_executor.portfolio
        equity_curve = portfolio.get_equity_curve()
        
        # Get completed trades
        all_trades = portfolio.trade_engine.get_trades()
        completed_trades = [trade for trade in all_trades if trade.exit_price is not None]
        
        # Convert trades to expected format for statistics engine
        trades_data = []
        for trade in completed_trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'commission': trade.commission
            })
        
        # Generate statistics
        stats_engine = StatisticsEngine()
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=trades_data,
            initial_capital=request.initial_capital
        )
        
        rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=30)
        trade_analysis = stats_engine.analyze_trades(trades_data)
        
        report = stats_engine.generate_report(metrics, rolling_metrics, trade_analysis)
        
        active_backtests[backtest_id]['progress'] = 1.0
        
        # Prepare result
        result = {
            'id': backtest_id,
            'status': 'completed',
            'metrics': {
                'total_return': float(metrics.total_return),
                'annualized_return': float(metrics.annualized_return),
                'sharpe_ratio': float(metrics.sharpe_ratio),
                'max_drawdown': float(metrics.max_drawdown),
                'win_rate': float(metrics.win_rate),
                'profit_factor': float(metrics.profit_factor),
                'total_trades': int(metrics.total_trades)
            },
            'trades': trades_data,
            'equity_curve': [
                {'timestamp': str(idx), 'value': float(val)} 
                for idx, val in equity_curve.items()
            ],
            'statistics_report': report,
            'execution_time_seconds': execution_time
        }
        
        # Move to completed
        completed_backtests[backtest_id] = result
        active_backtests[backtest_id]['status'] = 'completed'
        active_backtests[backtest_id]['end_time'] = datetime.now()
        active_backtests[backtest_id]['message'] = 'Backtest completed successfully'
        
    except Exception as e:
        # Handle error
        active_backtests[backtest_id]['status'] = 'failed'
        active_backtests[backtest_id]['end_time'] = datetime.now()
        active_backtests[backtest_id]['message'] = f'Backtest failed: {str(e)}'


@router.post("/run", response_model=Dict[str, str])
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Start a new backtest
    
    Runs a backtest with the specified strategy, market data, and risk parameters.
    Returns immediately with a backtest ID for status tracking.
    """
    try:
        # Generate unique ID
        backtest_id = str(uuid.uuid4())
        
        # Initialize status
        active_backtests[backtest_id] = {
            'id': backtest_id,
            'status': 'starting',
            'progress': 0.0,
            'start_time': datetime.now(),
            'end_time': None,
            'message': 'Backtest initialized'
        }
        
        # Start backtest in background
        background_tasks.add_task(run_backtest_sync, backtest_id, request)
        
        return {
            'backtest_id': backtest_id,
            'status': 'started',
            'message': 'Backtest started successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")


@router.get("/status/{backtest_id}", response_model=BacktestStatus)
async def get_backtest_status(backtest_id: str):
    """Get the status of a running or completed backtest"""
    
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    status_data = active_backtests[backtest_id]
    return BacktestStatus(**status_data)


@router.get("/result/{backtest_id}", response_model=BacktestResult)
async def get_backtest_result(backtest_id: str):
    """Get the complete results of a completed backtest"""
    
    if backtest_id not in completed_backtests:
        if backtest_id in active_backtests:
            status = active_backtests[backtest_id]['status']
            if status == 'running' or status == 'starting':
                raise HTTPException(status_code=202, detail="Backtest still running")
            elif status == 'failed':
                raise HTTPException(status_code=500, detail="Backtest failed")
        raise HTTPException(status_code=404, detail="Backtest result not found")
    
    return BacktestResult(**completed_backtests[backtest_id])


@router.get("/list")
async def list_backtests():
    """List all backtests with their status"""
    
    all_backtests = []
    
    # Add active backtests
    for backtest_id, status in active_backtests.items():
        all_backtests.append({
            'id': backtest_id,
            'status': status['status'],
            'start_time': status['start_time'],
            'progress': status.get('progress', 0.0)
        })
    
    return {
        'total_backtests': len(all_backtests),
        'backtests': all_backtests
    }


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest and its results"""
    
    deleted = False
    
    if backtest_id in active_backtests:
        del active_backtests[backtest_id]
        deleted = True
    
    if backtest_id in completed_backtests:
        del completed_backtests[backtest_id]
        deleted = True
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return {'message': f'Backtest {backtest_id} deleted successfully'}


@router.get("/strategies")
async def list_available_strategies():
    """List available built-in strategies"""
    
    return {
        'strategies': [
            {
                'name': 'momentum',
                'description': 'Simple momentum strategy - buy on price increase, sell on decrease',
                'parameters': {}
            },
            {
                'name': 'mean_reversion', 
                'description': 'Mean reversion strategy - buy below SMA, sell above SMA',
                'parameters': {
                    'sma_period': 20,
                    'threshold_pct': 0.02
                }
            },
            {
                'name': 'custom',
                'description': 'Custom strategy with user-defined logic',
                'parameters': {
                    'code': 'Python function that takes (data, timestamp) and returns signals'
                }
            }
        ]
    }
