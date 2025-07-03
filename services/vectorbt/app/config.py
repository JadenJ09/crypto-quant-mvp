"""
Configuration management for VectorBT Service
"""

import os
from typing import List, Dict, Any, ClassVar
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Service configuration settings"""
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/crypto_data" 
    
    # Service Configuration
    service_mode: str = "indicators"  # 'indicators' or 'bulk'
    batch_size: int = 1000
    max_workers: int = 4
    polling_interval: int = 30  # Seconds between DB polls
    log_level: str = "INFO"
    
    # Date Range Configuration (for bulk processing)
    start_date: str = "2025-06-01"
    end_date: str = "2025-06-29"
    
    # Technical Indicator Periods
    rsi_periods: str = "14,21,30"
    ema_periods: str = "9,21,50,100,200"
    sma_periods: str = "20,50,100,200"
    
    # MACD Parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Other Indicators
    atr_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    volume_sma_period: int = 20
    cmf_period: int = 20
    mfi_period: int = 14
    volatility_window: int = 252
    
    # Timeframe definitions (ClassVar since it's not configurable)
    TIMEFRAMES: ClassVar[Dict[str, Dict[str, Any]]] = {
        '5min': {
            'table': 'ohlcv_5min',
            'resample_freq': '5min',
            'min_periods': 288  # Need at least 288 periods for indicators
        },
        '15min': {
            'table': 'ohlcv_15min', 
            'resample_freq': '15min',
            'min_periods': 288
        },
        '1hour': {
            'table': 'ohlcv_1hour',
            'resample_freq': '1h',
            'min_periods': 240
        },
        '4hour': {
            'table': 'ohlcv_4hour',
            'resample_freq': '4h', 
            'min_periods': 180
        },
        '1day': {
            'table': 'ohlcv_1day',
            'resample_freq': '1D',
            'min_periods': 50
        },
        '7day': {
            'table': 'ohlcv_7day',
            'resample_freq': 'W',
            'min_periods': 20
        }
    }
    
    @property
    def rsi_periods_list(self) -> List[int]:
        """Parse RSI periods from string"""
        return [int(p.strip()) for p in self.rsi_periods.split(',')]
    
    @property
    def ema_periods_list(self) -> List[int]:
        """Parse EMA periods from string"""
        return [int(p.strip()) for p in self.ema_periods.split(',')]
    
    @property
    def sma_periods_list(self) -> List[int]:
        """Parse SMA periods from string"""
        return [int(p.strip()) for p in self.sma_periods.split(',')]
    
    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False}
