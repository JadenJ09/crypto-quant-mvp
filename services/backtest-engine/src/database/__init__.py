"""
Database integration module for Custom Backtest Engine

This module provides database connectivity and data access utilities
for integrating with the crypto_quant_mvp TimescaleDB infrastructure.
"""

from .timescale_connector import TimescaleDBConnector, get_timescale_connector, close_timescale_connector

__all__ = [
    'TimescaleDBConnector',
    'get_timescale_connector', 
    'close_timescale_connector'
]
