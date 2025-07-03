"""
API endpoints for the custom backtesting engine

This module provides FastAPI endpoints for:
- Running backtests
- Real-time performance analysis
- Statistics and reporting
- Risk management configuration
"""

from .backtest_api import router as backtest_router
from .statistics_api import router as statistics_router
from .health_api import router as health_router

__all__ = ['backtest_router', 'statistics_router', 'health_router']
