"""
Signal Generation Components

This module contains components for technical analysis and signal generation:
- SignalEngine: Main signal generation engine with technical indicators
- Custom numba-optimized indicator calculations
"""

from .signal_engine import (
    SignalEngine,
    calculate_sma,
    calculate_ema, 
    calculate_rsi
)

__all__ = [
    'SignalEngine',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi'
]
