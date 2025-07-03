# ==============================================================================
# File: services/api/app/routers/backtesting.py
# Description: Router for backtesting endpoints - lightweight orchestration layer
# ==============================================================================

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import logging
import httpx
import asyncio
from datetime import datetime

from ..models import (
    BacktestStats, 
    BacktestRequest, 
    StrategyConfig, 
    StrategyInfo,
    ValidationResult
)
from ..dependencies import get_vectorbt_service_url

router = APIRouter(prefix="/backtest", tags=["backtesting"])

@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies():
    """Get list of available backtesting strategies from vectorbt service"""
    vectorbt_url = get_vectorbt_service_url()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{vectorbt_url}/strategies/available")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error connecting to vectorbt service: {e}")
        # Return default strategies if service is not available
        return [
            {
                "name": "rsi_oversold",
                "display_name": "RSI Oversold Strategy",
                "description": "Buy when RSI is oversold (<30) and sell when overbought (>70)",
                "parameters": [
                    {"name": "rsi_period", "type": "int", "default": 14, "description": "RSI calculation period"},
                    {"name": "oversold_level", "type": "float", "default": 30, "description": "Oversold threshold"},
                    {"name": "overbought_level", "type": "float", "default": 70, "description": "Overbought threshold"}
                ]
            },
            {
                "name": "ma_crossover",
                "display_name": "Moving Average Crossover",
                "description": "Buy when fast MA crosses above slow MA, sell when it crosses below",
                "parameters": [
                    {"name": "fast_window", "type": "int", "default": 10, "description": "Fast MA period"},
                    {"name": "slow_window", "type": "int", "default": 20, "description": "Slow MA period"}
                ]
            }
        ]
    except httpx.HTTPStatusError as e:
        logging.error(f"Error from vectorbt service: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Error retrieving strategies"
        )

@router.get("/indicators/{timeframe}")
async def get_indicators_for_timeframe(timeframe: str):
    """Get available technical indicators for a specific timeframe from vectorbt service"""
    vectorbt_url = get_vectorbt_service_url()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{vectorbt_url}/indicators/{timeframe}")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error connecting to vectorbt service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Backtesting service is not available"
        )
    except httpx.HTTPStatusError as e:
        logging.error(f"Error from vectorbt service: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Error retrieving indicators"
        )

@router.post("/run", response_model=BacktestStats)
async def run_backtest(request: BacktestRequest):
    """
    Run a comprehensive backtest - orchestrates call to vectorbt service
    """
    # Determine which strategy format we're using
    strategy_data = None
    if request.frontend_strategy:
        strategy_data = request.frontend_strategy.dict()
        logging.info(f"Orchestrating backtest for {request.symbol} ({request.timeframe}) with frontend strategy: {request.frontend_strategy.name}")
    elif request.legacy_strategy:
        strategy_data = request.legacy_strategy.dict()
        logging.info(f"Orchestrating backtest for {request.symbol} ({request.timeframe}) with legacy strategy: {request.legacy_strategy.strategy_type}")
    else:
        # Convert strategy to dict regardless of type
        strategy_data = request.strategy.dict()
        logging.info(f"Orchestrating backtest for {request.symbol} ({request.timeframe}) with strategy")
    
    vectorbt_url = get_vectorbt_service_url()
    
    # Prepare request for vectorbt service (use VectorBT format)
    vectorbt_request = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "initial_cash": request.initial_cash,
        "commission": request.commission,
        "slippage": request.slippage,
        "strategy": strategy_data
    }
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for backtests
            response = await client.post(
                f"{vectorbt_url}/strategies/backtest",  # Fixed endpoint URL
                json=vectorbt_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logging.error("Backtest timeout")
        raise HTTPException(
            status_code=408,
            detail="Backtest timed out - try reducing data range or complexity"
        )
    except httpx.RequestError as e:
        logging.error(f"Error connecting to vectorbt service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Backtesting service is not available"
        )
    except httpx.HTTPStatusError as e:
        logging.error(f"Error from vectorbt service: {e.response.status_code}: {e.response.text}")
        if e.response.status_code == 400:
            # Pass through validation errors
            error_detail = "Invalid backtest parameters"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("detail", error_detail)
            except:
                pass
            raise HTTPException(status_code=400, detail=error_detail)
        else:
            raise HTTPException(
                status_code=500,
                detail="Error running backtest"
            )

@router.get("/simple_ma/{symbol}")
async def simple_ma_backtest(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe"),
    fast_window: int = Query(10, ge=1, le=100, description="Fast MA window"),
    slow_window: int = Query(20, ge=1, le=200, description="Slow MA window"),
    initial_cash: float = Query(100000.0, gt=0, description="Initial capital"),
    commission: float = Query(0.001, ge=0, le=0.1, description="Commission rate")
):
    """
    Simple Moving Average crossover backtest - P1-T7 implementation
    This is the MVP endpoint mentioned in your PRD
    """
    if fast_window >= slow_window:
        raise HTTPException(
            status_code=400,
            detail="Fast window must be smaller than slow window"
        )
    
    # Convert to BacktestRequest format for consistency
    request = BacktestRequest(
        symbol=symbol,
        timeframe=timeframe,
        start_date=None,
        end_date=None,
        initial_cash=initial_cash,
        commission=commission,
        strategy=StrategyConfig(
            strategy_type="ma_crossover",
            parameters={
                "fast_window": fast_window,
                "slow_window": slow_window
            },
            stop_loss_pct=None,
            take_profit_pct=None
        )
    )
    
    return await run_backtest(request)

@router.get("/validate/{symbol}")
async def validate_backtest_inputs(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe to validate")
):
    """
    Validate symbol and timeframe for backtesting - delegates to vectorbt service
    """
    vectorbt_url = get_vectorbt_service_url()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{vectorbt_url}/validate/{symbol}?timeframe={timeframe}")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error connecting to vectorbt service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Backtesting service is not available"
        )
    except httpx.HTTPStatusError as e:
        logging.error(f"Error from vectorbt service: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Error validating inputs"
        )
