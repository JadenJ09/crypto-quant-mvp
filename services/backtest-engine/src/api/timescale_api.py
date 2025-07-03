"""
TimescaleDB Data API for Custom Backtest Engine

This module provides REST API endpoints for retrieving market data from
TimescaleDB and running backtests with the custom engine.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

# Database imports
from database.timescale_connector import get_timescale_connector, TimescaleDBConnector

# Core engine imports
from core.backtest_executor import BacktestExecutor
from core.enhanced_portfolio_manager import EnhancedPortfolioManager
from risk.risk_manager import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from statistics.statistics_engine import StatisticsEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/timescale", tags=["TimescaleDB Integration"])


# Pydantic models
class MarketDataRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTCUSDT')")
    timeframe: str = Field(default="1min", description="Data timeframe")
    start_time: Optional[datetime] = Field(None, description="Start datetime")
    end_time: Optional[datetime] = Field(None, description="End datetime")
    limit: Optional[int] = Field(default=1000, description="Maximum records")


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1min", description="Data timeframe")
    start_time: Optional[datetime] = Field(None, description="Start datetime")
    end_time: Optional[datetime] = Field(None, description="End datetime")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.04, description="Take profit percentage")
    risk_per_trade: float = Field(default=0.01, description="Risk per trade")
    strategy: str = Field(default="momentum", description="Strategy type")


class HealthResponse(BaseModel):
    healthy: bool
    database_connected: bool
    tables_available: bool
    data_available: bool
    error: Optional[str] = None
    details: Dict[str, Any] = {}


class DataInfoResponse(BaseModel):
    available_tables: List[str]
    total_records: int
    symbols_count: int
    earliest_data: Optional[datetime]
    latest_data: Optional[datetime]
    symbol_info: Dict[str, Any] = {}


class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    records_count: int
    earliest_time: Optional[datetime]
    latest_time: Optional[datetime]
    data: List[Dict[str, Any]]


class BacktestResponse(BaseModel):
    success: bool
    symbol: str
    timeframe: str
    data_points: int
    trades_executed: int
    initial_capital: float
    final_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    error: Optional[str] = None


@router.get("/health", response_model=HealthResponse)
async def timescale_health_check():
    """Check TimescaleDB health and connectivity"""
    try:
        connector = await get_timescale_connector()
        health = await connector.health_check()
        
        return HealthResponse(
            healthy=health['healthy'],
            database_connected=health['database_connected'],
            tables_available=health['tables_available'],
            data_available=health['data_available'],
            error=health.get('error'),
            details=health.get('details', {})
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            healthy=False,
            database_connected=False,
            tables_available=False,
            data_available=False,
            error=str(e)
        )


@router.get("/info", response_model=DataInfoResponse)
async def get_data_info(symbol: Optional[str] = None):
    """Get information about available data"""
    try:
        connector = await get_timescale_connector()
        info = await connector.get_data_info(symbol)
        
        return DataInfoResponse(
            available_tables=info['available_tables'],
            total_records=info['total_records'],
            symbols_count=info['symbols_count'],
            earliest_data=info['earliest_data'],
            latest_data=info['latest_data'],
            symbol_info=info['symbol_info']
        )
    except Exception as e:
        logger.error(f"❌ Failed to get data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_available_symbols(
    table_name: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get available trading symbols"""
    try:
        connector = await get_timescale_connector()
        symbols = await connector.get_available_symbols(table_name, limit)
        
        return {
            "symbols": symbols,
            "count": len(symbols)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables")
async def get_available_tables():
    """Get available OHLCV tables"""
    try:
        connector = await get_timescale_connector()
        tables = await connector.get_available_tables()
        
        return {
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest):
    """Get market data for backtesting"""
    try:
        connector = await get_timescale_connector()
        
        # Get market data
        data = await connector.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit
        )
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {request.symbol} in {request.timeframe} timeframe"
            )
        
        # Convert DataFrame to response format
        data_records = []
        for timestamp, row in data.iterrows():
            data_records.append({
                "time": timestamp.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return MarketDataResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            records_count=len(data_records),
            earliest_time=data.index.min(),
            latest_time=data.index.max(),
            data=data_records
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest with TimescaleDB data"""
    try:
        connector = await get_timescale_connector()
        
        # Get market data
        data = await connector.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=2000  # Sufficient for backtesting
        )
        
        if data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {request.symbol}"
            )
        
        if len(data) < 50:  # Minimum data points for meaningful backtest
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(data)} records (minimum 50 required)"
            )
        
        # Define strategy based on request
        def create_strategy(strategy_type: str):
            if strategy_type == "momentum":
                def momentum_strategy(data_slice, timestamp):
                    if len(data_slice) < 20:
                        return {request.symbol: {'action': 'hold'}}
                    
                    # Simple momentum strategy
                    sma_short = data_slice['close'].rolling(5).mean()
                    sma_long = data_slice['close'].rolling(20).mean()
                    
                    if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
                        return {request.symbol: {'action': 'buy', 'side': 'long'}}
                    elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
                        return {request.symbol: {'action': 'sell', 'side': 'short'}}
                    else:
                        return {request.symbol: {'action': 'hold'}}
                
                return momentum_strategy
            
            elif strategy_type == "mean_reversion":
                def mean_reversion_strategy(data_slice, timestamp):
                    if len(data_slice) < 20:
                        return {request.symbol: {'action': 'hold'}}
                    
                    # RSI-based mean reversion
                    delta = data_slice['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    if rsi.iloc[-1] < 30:  # Oversold
                        return {request.symbol: {'action': 'buy', 'side': 'long'}}
                    elif rsi.iloc[-1] > 70:  # Overbought
                        return {request.symbol: {'action': 'sell', 'side': 'short'}}
                    else:
                        return {request.symbol: {'action': 'hold'}}
                
                return mean_reversion_strategy
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy_type}")
        
        # Create and configure backtest executor
        executor = BacktestExecutor(initial_capital=request.initial_capital)
        executor.set_strategy(create_strategy(request.strategy))
        executor.set_risk_parameters(
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            risk_per_trade=request.risk_per_trade
        )
        
        # Run backtest
        results = executor.run_backtest(data, [request.symbol])
        
        # Calculate performance statistics
        stats_engine = StatisticsEngine()
        
        # Convert trades DataFrame to list of dictionaries for statistics engine
        trades_data = []
        if not results['trades'].empty:
            for _, trade in results['trades'].iterrows():
                trades_data.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'size': trade['size'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl': trade['pnl'],
                    'commission': trade['commission']
                })
        
        performance_stats = stats_engine.calculate_performance_metrics(
            equity_curve=results['equity_curve'],
            trades=trades_data,
            initial_capital=request.initial_capital
        )
        
        return BacktestResponse(
            success=True,
            symbol=request.symbol,
            timeframe=request.timeframe,
            data_points=len(data),
            trades_executed=len(results['trades']),
            initial_capital=request.initial_capital,
            final_capital=results['metrics']['final_capital'],
            total_return_pct=results['metrics']['total_return_pct'],
            max_drawdown_pct=performance_stats.max_drawdown * 100,  # Convert to percentage
            sharpe_ratio=performance_stats.sharpe_ratio,
            win_rate=performance_stats.win_rate * 100  # Convert to percentage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Backtest execution failed: {e}")
        return BacktestResponse(
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            data_points=0,
            trades_executed=0,
            initial_capital=request.initial_capital,
            final_capital=0,
            total_return_pct=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            win_rate=0,
            error=str(e)
        )


@router.get("/quick-test/{symbol}")
async def quick_backtest_test(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365),
    timeframe: str = Query(default="1min")
):
    """Quick backtest test for a symbol"""
    try:
        # Create backtest request
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        request = BacktestRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            initial_capital=100000.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            risk_per_trade=0.01,
            strategy="momentum"
        )
        
        # Run backtest
        result = await run_backtest(request)
        
        return {
            "test_completed": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "days_tested": days,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
