"""
Statistics API endpoints

Provides REST API access to the advanced statistics engine for
performance analysis, risk metrics, and comprehensive reporting.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import our statistics engine
import sys
import os

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from statistics.statistics_engine import StatisticsEngine

router = APIRouter()


class TradeData(BaseModel):
    """Trade data model"""
    entry_time: str
    exit_time: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    commission: float = 0.0


class EquityPoint(BaseModel):
    """Equity curve data point"""
    timestamp: str
    value: float


class PerformanceAnalysisRequest(BaseModel):
    """Request model for performance analysis"""
    equity_curve: List[EquityPoint]
    trades: List[TradeData]
    initial_capital: float = Field(default=100000.0, gt=0)
    benchmark_returns: Optional[List[float]] = None


class RollingAnalysisRequest(BaseModel):
    """Request model for rolling analysis"""
    equity_curve: List[EquityPoint]
    window: int = Field(default=30, ge=5, le=365)


class RiskMetricsResponse(BaseModel):
    """Risk metrics response model"""
    max_drawdown: float
    current_drawdown: float
    volatility: float
    annualized_volatility: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model"""
    total_return: float
    annualized_return: float
    cumulative_return: float
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    risk_metrics: RiskMetricsResponse
    risk_adjusted_returns: Dict[str, float]
    trade_statistics: Dict[str, Any]


class TradeAnalysisResponse(BaseModel):
    """Trade analysis response model"""
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate: float
    long_win_rate: float
    short_win_rate: float
    profit_factor: float
    long_profit_factor: float
    short_profit_factor: float
    average_hold_time: str
    winning_hold_time: str
    losing_hold_time: str
    max_consecutive_wins: int
    max_consecutive_losses: int
    entry_efficiency: float
    exit_efficiency: float


class ComprehensiveReportResponse(BaseModel):
    """Comprehensive analysis response model"""
    performance_metrics: PerformanceMetricsResponse
    trade_analysis: TradeAnalysisResponse
    rolling_metrics_summary: Dict[str, Any]
    formatted_report: str
    analysis_timestamp: datetime


def convert_equity_curve(equity_points: List[EquityPoint]) -> pd.Series:
    """Convert equity points to pandas Series"""
    timestamps = [pd.to_datetime(ep.timestamp) for ep in equity_points]
    values = [ep.value for ep in equity_points]
    return pd.Series(values, index=timestamps)


def convert_trades(trades: List[TradeData]) -> List[Dict[str, Any]]:
    """Convert trade data to dictionary format"""
    return [
        {
            'entry_time': pd.to_datetime(trade.entry_time),
            'exit_time': pd.to_datetime(trade.exit_time),
            'symbol': trade.symbol,
            'side': trade.side,
            'size': trade.size,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'commission': trade.commission
        }
        for trade in trades
    ]


@router.post("/performance", response_model=PerformanceMetricsResponse)
async def analyze_performance(request: PerformanceAnalysisRequest):
    """
    Comprehensive performance analysis
    
    Analyzes portfolio performance including returns, risk metrics,
    and trade statistics using the advanced statistics engine.
    """
    try:
        # Convert data
        equity_curve = convert_equity_curve(request.equity_curve)
        trades_data = convert_trades(request.trades)
        
        # Initialize statistics engine
        stats_engine = StatisticsEngine()
        
        # Calculate performance metrics
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=trades_data,
            initial_capital=request.initial_capital
        )
        
        # Build response
        risk_metrics = RiskMetricsResponse(
            max_drawdown=float(metrics.max_drawdown),
            current_drawdown=float(metrics.current_drawdown),
            volatility=float(metrics.volatility),
            annualized_volatility=float(metrics.annualized_volatility),
            var_95=float(metrics.var_95),
            cvar_95=float(metrics.cvar_95),
            skewness=float(metrics.skewness),
            kurtosis=float(metrics.kurtosis)
        )
        
        risk_adjusted_returns = {
            'sharpe_ratio': float(metrics.sharpe_ratio),
            'sortino_ratio': float(metrics.sortino_ratio),
            'calmar_ratio': float(metrics.calmar_ratio),
            'omega_ratio': float(metrics.omega_ratio)
        }
        
        trade_statistics = {
            'total_trades': int(metrics.total_trades),
            'winning_trades': int(metrics.winning_trades),
            'losing_trades': int(metrics.losing_trades),
            'win_rate': float(metrics.win_rate),
            'profit_factor': float(metrics.profit_factor),
            'average_win': float(metrics.average_win),
            'average_loss': float(metrics.average_loss),
            'largest_win': float(metrics.largest_win),
            'largest_loss': float(metrics.largest_loss)
        }
        
        if metrics.alpha is not None:
            risk_adjusted_returns.update({
                'alpha': float(metrics.alpha),
                'beta': float(metrics.beta),
                'information_ratio': float(metrics.information_ratio),
                'tracking_error': float(metrics.tracking_error)
            })
        
        return PerformanceMetricsResponse(
            total_return=float(metrics.total_return),
            annualized_return=float(metrics.annualized_return),
            cumulative_return=float(metrics.cumulative_return),
            best_month=float(metrics.best_month),
            worst_month=float(metrics.worst_month),
            positive_months=int(metrics.positive_months),
            negative_months=int(metrics.negative_months),
            risk_metrics=risk_metrics,
            risk_adjusted_returns=risk_adjusted_returns,
            trade_statistics=trade_statistics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")


@router.post("/trades/analysis", response_model=TradeAnalysisResponse)
async def analyze_trades(trades: List[TradeData]):
    """
    Detailed trade analysis
    
    Analyzes trading performance including win rates, hold times,
    and efficiency metrics by trade direction and overall.
    """
    try:
        # Convert trades
        trades_data = convert_trades(trades)
        
        # Initialize statistics engine
        stats_engine = StatisticsEngine()
        
        # Analyze trades
        analysis = stats_engine.analyze_trades(trades_data)
        
        return TradeAnalysisResponse(
            total_trades=len(trades_data),
            long_trades=int(analysis.long_trades),
            short_trades=int(analysis.short_trades),
            win_rate=float(analysis.long_win_rate * analysis.long_trades + analysis.short_win_rate * analysis.short_trades) / len(trades_data) if len(trades_data) > 0 else 0.0,
            long_win_rate=float(analysis.long_win_rate),
            short_win_rate=float(analysis.short_win_rate),
            profit_factor=float(analysis.long_profit_factor * analysis.long_trades + analysis.short_profit_factor * analysis.short_trades) / len(trades_data) if len(trades_data) > 0 else 0.0,
            long_profit_factor=float(analysis.long_profit_factor),
            short_profit_factor=float(analysis.short_profit_factor),
            average_hold_time=str(analysis.average_hold_time),
            winning_hold_time=str(analysis.winning_hold_time),
            losing_hold_time=str(analysis.losing_hold_time),
            max_consecutive_wins=int(analysis.max_consecutive_wins),
            max_consecutive_losses=int(analysis.max_consecutive_losses),
            entry_efficiency=float(analysis.entry_efficiency),
            exit_efficiency=float(analysis.exit_efficiency)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade analysis failed: {str(e)}")


@router.post("/rolling", response_model=Dict[str, Any])
async def rolling_analysis(request: RollingAnalysisRequest):
    """
    Rolling performance analysis
    
    Calculates rolling statistics with customizable window size
    for time-series performance analysis.
    """
    try:
        # Convert data
        equity_curve = convert_equity_curve(request.equity_curve)
        
        # Initialize statistics engine
        stats_engine = StatisticsEngine()
        
        # Calculate rolling metrics
        rolling_metrics = stats_engine.calculate_rolling_metrics(
            equity_curve=equity_curve,
            window=request.window
        )
        
        # Prepare response
        response = {
            'window_size': request.window,
            'total_periods': len(rolling_metrics.returns),
            'rolling_statistics': {
                'returns': {
                    'mean': float(np.mean(rolling_metrics.returns)) if rolling_metrics.returns else 0.0,
                    'std': float(np.std(rolling_metrics.returns)) if rolling_metrics.returns else 0.0,
                    'min': float(np.min(rolling_metrics.returns)) if rolling_metrics.returns else 0.0,
                    'max': float(np.max(rolling_metrics.returns)) if rolling_metrics.returns else 0.0,
                    'latest': float(rolling_metrics.returns[-1]) if rolling_metrics.returns else 0.0
                },
                'volatility': {
                    'mean': float(np.mean(rolling_metrics.volatility)) if rolling_metrics.volatility else 0.0,
                    'std': float(np.std(rolling_metrics.volatility)) if rolling_metrics.volatility else 0.0,
                    'min': float(np.min(rolling_metrics.volatility)) if rolling_metrics.volatility else 0.0,
                    'max': float(np.max(rolling_metrics.volatility)) if rolling_metrics.volatility else 0.0,
                    'latest': float(rolling_metrics.volatility[-1]) if rolling_metrics.volatility else 0.0
                },
                'sharpe_ratio': {
                    'mean': float(np.mean(rolling_metrics.sharpe_ratio)) if rolling_metrics.sharpe_ratio else 0.0,
                    'std': float(np.std(rolling_metrics.sharpe_ratio)) if rolling_metrics.sharpe_ratio else 0.0,
                    'min': float(np.min(rolling_metrics.sharpe_ratio)) if rolling_metrics.sharpe_ratio else 0.0,
                    'max': float(np.max(rolling_metrics.sharpe_ratio)) if rolling_metrics.sharpe_ratio else 0.0,
                    'latest': float(rolling_metrics.sharpe_ratio[-1]) if rolling_metrics.sharpe_ratio else 0.0
                },
                'drawdown': {
                    'mean': float(np.mean(rolling_metrics.drawdown)) if rolling_metrics.drawdown else 0.0,
                    'std': float(np.std(rolling_metrics.drawdown)) if rolling_metrics.drawdown else 0.0,
                    'min': float(np.min(rolling_metrics.drawdown)) if rolling_metrics.drawdown else 0.0,
                    'max': float(np.max(rolling_metrics.drawdown)) if rolling_metrics.drawdown else 0.0,
                    'latest': float(rolling_metrics.drawdown[-1]) if rolling_metrics.drawdown else 0.0
                }
            }
        }
        
        # Include time series data if requested
        if len(rolling_metrics.timestamps) <= 1000:  # Limit response size
            response['time_series'] = {
                'timestamps': [str(ts) for ts in rolling_metrics.timestamps],
                'returns': rolling_metrics.returns,
                'volatility': rolling_metrics.volatility,
                'sharpe_ratio': rolling_metrics.sharpe_ratio,
                'drawdown': rolling_metrics.drawdown
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rolling analysis failed: {str(e)}")


@router.post("/report", response_model=ComprehensiveReportResponse)
async def comprehensive_report(request: PerformanceAnalysisRequest):
    """
    Generate comprehensive performance report
    
    Creates a complete analysis including performance metrics, trade analysis,
    rolling statistics, and a formatted text report.
    """
    try:
        # Convert data
        equity_curve = convert_equity_curve(request.equity_curve)
        trades_data = convert_trades(request.trades)
        
        # Initialize statistics engine
        stats_engine = StatisticsEngine()
        
        # Calculate all metrics
        metrics = stats_engine.calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=trades_data,
            initial_capital=request.initial_capital
        )
        
        rolling_metrics = stats_engine.calculate_rolling_metrics(equity_curve, window=30)
        trade_analysis = stats_engine.analyze_trades(trades_data)
        
        # Generate formatted report
        formatted_report = stats_engine.generate_report(metrics, rolling_metrics, trade_analysis)
        
        # Build comprehensive response
        risk_metrics = RiskMetricsResponse(
            max_drawdown=float(metrics.max_drawdown),
            current_drawdown=float(metrics.current_drawdown),
            volatility=float(metrics.volatility),
            annualized_volatility=float(metrics.annualized_volatility),
            var_95=float(metrics.var_95),
            cvar_95=float(metrics.cvar_95),
            skewness=float(metrics.skewness),
            kurtosis=float(metrics.kurtosis)
        )
        
        risk_adjusted_returns = {
            'sharpe_ratio': float(metrics.sharpe_ratio),
            'sortino_ratio': float(metrics.sortino_ratio),
            'calmar_ratio': float(metrics.calmar_ratio),
            'omega_ratio': float(metrics.omega_ratio)
        }
        
        trade_statistics = {
            'total_trades': int(metrics.total_trades),
            'winning_trades': int(metrics.winning_trades),
            'losing_trades': int(metrics.losing_trades),
            'win_rate': float(metrics.win_rate),
            'profit_factor': float(metrics.profit_factor),
            'average_win': float(metrics.average_win),
            'average_loss': float(metrics.average_loss),
            'largest_win': float(metrics.largest_win),
            'largest_loss': float(metrics.largest_loss)
        }
        
        performance_metrics = PerformanceMetricsResponse(
            total_return=float(metrics.total_return),
            annualized_return=float(metrics.annualized_return),
            cumulative_return=float(metrics.cumulative_return),
            best_month=float(metrics.best_month),
            worst_month=float(metrics.worst_month),
            positive_months=int(metrics.positive_months),
            negative_months=int(metrics.negative_months),
            risk_metrics=risk_metrics,
            risk_adjusted_returns=risk_adjusted_returns,
            trade_statistics=trade_statistics
        )
        
        trade_analysis_response = TradeAnalysisResponse(
            total_trades=len(trades_data),
            long_trades=int(trade_analysis.long_trades),
            short_trades=int(trade_analysis.short_trades),
            win_rate=float(trade_analysis.long_win_rate * trade_analysis.long_trades + trade_analysis.short_win_rate * trade_analysis.short_trades) / len(trades_data) if len(trades_data) > 0 else 0.0,
            long_win_rate=float(trade_analysis.long_win_rate),
            short_win_rate=float(trade_analysis.short_win_rate),
            profit_factor=float(trade_analysis.long_profit_factor * trade_analysis.long_trades + trade_analysis.short_profit_factor * trade_analysis.short_trades) / len(trades_data) if len(trades_data) > 0 else 0.0,
            long_profit_factor=float(trade_analysis.long_profit_factor),
            short_profit_factor=float(trade_analysis.short_profit_factor),
            average_hold_time=str(trade_analysis.average_hold_time),
            winning_hold_time=str(trade_analysis.winning_hold_time),
            losing_hold_time=str(trade_analysis.losing_hold_time),
            max_consecutive_wins=int(trade_analysis.max_consecutive_wins),
            max_consecutive_losses=int(trade_analysis.max_consecutive_losses),
            entry_efficiency=float(trade_analysis.entry_efficiency),
            exit_efficiency=float(trade_analysis.exit_efficiency)
        )
        
        rolling_summary = {
            'window_size': 30,
            'total_periods': len(rolling_metrics.returns),
            'avg_rolling_return': float(np.mean(rolling_metrics.returns)) if rolling_metrics.returns else 0.0,
            'avg_rolling_volatility': float(np.mean(rolling_metrics.volatility)) if rolling_metrics.volatility else 0.0,
            'avg_rolling_sharpe': float(np.mean(rolling_metrics.sharpe_ratio)) if rolling_metrics.sharpe_ratio else 0.0,
            'latest_rolling_return': float(rolling_metrics.returns[-1]) if rolling_metrics.returns else 0.0,
            'latest_rolling_volatility': float(rolling_metrics.volatility[-1]) if rolling_metrics.volatility else 0.0,
            'latest_rolling_sharpe': float(rolling_metrics.sharpe_ratio[-1]) if rolling_metrics.sharpe_ratio else 0.0
        }
        
        return ComprehensiveReportResponse(
            performance_metrics=performance_metrics,
            trade_analysis=trade_analysis_response,
            rolling_metrics_summary=rolling_summary,
            formatted_report=formatted_report,
            analysis_timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive report generation failed: {str(e)}")


@router.get("/metrics/available")
async def list_available_metrics():
    """List all available performance and risk metrics"""
    
    return {
        'performance_metrics': [
            'total_return', 'annualized_return', 'cumulative_return',
            'best_month', 'worst_month', 'positive_months', 'negative_months'
        ],
        'risk_metrics': [
            'max_drawdown', 'current_drawdown', 'volatility', 'annualized_volatility',
            'var_95', 'cvar_95', 'skewness', 'kurtosis'
        ],
        'risk_adjusted_metrics': [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio'
        ],
        'benchmark_metrics': [
            'alpha', 'beta', 'information_ratio', 'tracking_error'
        ],
        'trade_metrics': [
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'profit_factor', 'average_win', 'average_loss', 'largest_win', 'largest_loss'
        ],
        'trade_analysis': [
            'long_trades', 'short_trades', 'long_win_rate', 'short_win_rate',
            'long_profit_factor', 'short_profit_factor', 'average_hold_time',
            'winning_hold_time', 'losing_hold_time', 'max_consecutive_wins',
            'max_consecutive_losses', 'entry_efficiency', 'exit_efficiency'
        ],
        'rolling_metrics': [
            'rolling_returns', 'rolling_volatility', 'rolling_sharpe_ratio', 'rolling_drawdown'
        ]
    }
