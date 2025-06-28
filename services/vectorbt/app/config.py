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
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Service Configuration
    service_mode: str = Field("indicators", env="SERVICE_MODE")  # 'indicators' or 'bulk'
    batch_size: int = Field(1000, env="BATCH_SIZE")
    max_workers: int = Field(4, env="MAX_WORKERS")
    polling_interval: int = Field(30, env="POLLING_INTERVAL")  # Seconds between DB polls
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Technical Indicator Periods
    rsi_periods: str = Field("14,21,30", env="RSI_PERIODS")
    ema_periods: str = Field("9,21,50,100,200", env="EMA_PERIODS")
    sma_periods: str = Field("20,50,100,200", env="SMA_PERIODS")
    
    # MACD Parameters
    macd_fast: int = Field(12, env="MACD_FAST")
    macd_slow: int = Field(26, env="MACD_SLOW")
    macd_signal: int = Field(9, env="MACD_SIGNAL")
    
    # Bollinger Bands
    bb_period: int = Field(20, env="BB_PERIOD")
    bb_std: float = Field(2.0, env="BB_STD")
    
    # Other Indicators
    atr_period: int = Field(14, env="ATR_PERIOD")
    stoch_k_period: int = Field(14, env="STOCH_K_PERIOD")
    stoch_d_period: int = Field(3, env="STOCH_D_PERIOD")
    volume_sma_period: int = Field(20, env="VOLUME_SMA_PERIOD")
    cmf_period: int = Field(20, env="CMF_PERIOD")
    mfi_period: int = Field(14, env="MFI_PERIOD")
    volatility_window: int = Field(252, env="VOLATILITY_WINDOW")
    
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
            'resample_freq': '7D',
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
    
    model_config = {"env_file": ".env"}
