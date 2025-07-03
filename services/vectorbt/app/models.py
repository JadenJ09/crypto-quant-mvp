# ==============================================================================
# File: services/vectorbt/app/models.py
# Description: Models specific to the vectorbt service
# ==============================================================================

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

# Enums for strategy and indicator types
class StrategyType(str, Enum):
    MA_CROSSOVER = "MA_CROSSOVER"
    RSI_OVERSOLD = "RSI_OVERSOLD" 
    MEAN_REVERSION = "MEAN_REVERSION"
    BOLLINGER_BANDS = "BOLLINGER_BANDS"

class IndicatorType(str, Enum):
    SMA = "SMA"
    EMA = "EMA"
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER_BANDS = "BOLLINGER_BANDS"
    STOCHASTIC = "STOCHASTIC"

# Frontend strategy models to match the TypeScript interfaces
class EntryCondition(BaseModel):
    """Entry condition model"""
    id: str
    type: str  # 'technical_indicator', 'price_action', 'ml_signal'
    indicator: Optional[str] = None
    operator: str  # 'greater_than', 'less_than', 'crosses_above', etc.
    value: Any  # Can be number or string
    enabled: bool = True

class ExitCondition(BaseModel):
    """Exit condition model"""
    id: str
    type: str  # 'technical_indicator', 'price_action', 'stop_loss', etc.
    indicator: Optional[str] = None
    operator: str
    value: Any
    enabled: bool = True

class FrontendStrategy(BaseModel):
    """Frontend strategy configuration to match TypeScript interface"""
    id: str
    name: str
    description: str
    entry_conditions: List[EntryCondition] = Field(default_factory=list)
    exit_conditions: List[ExitCondition] = Field(default_factory=list)
    
    # Position sizing configuration
    position_sizing: str = Field(default="percentage")  # 'fixed', 'percentage', 'kelly', 'volatility'
    position_size: float = Field(default=0.1)
    
    # Risk management configuration
    max_positions: int = Field(default=1)
    max_position_strategy: str = Field(default="ignore")  # 'ignore', 'replace_oldest', 'replace_worst'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Position direction configuration
    position_direction: str = Field(default="long_only")  # 'long_only', 'short_only', 'both'

class BacktestRequest(BaseModel):
    """Updated request model to handle frontend data"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (5min, 1hour, etc.)")
    start_date: Optional[str] = Field(None, description="Backtest start date")
    end_date: Optional[str] = Field(None, description="Backtest end date")
    initial_cash: float = Field(default=100000.0, description="Initial capital")
    commission: float = Field(default=0.001, description="Commission rate (0.1%)")
    slippage: float = Field(default=0.0005, description="Slippage rate")
    strategy: FrontendStrategy = Field(..., description="Strategy configuration")

class IndicatorsRequest(BaseModel):
    """Request model for indicators calculation"""
    symbol: str
    indicators: List[IndicatorType]
    start_date: datetime
    end_date: datetime
    parameters: Optional[Dict[str, Dict[str, Any]]] = None

class IndicatorRequest(BaseModel):
    """Request model for calculating indicators"""
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    indicators: List[str] = []  # List of indicator names to calculate
    parameters: Optional[Dict[str, Any]] = None

# Response models
class IndicatorsResponse(BaseModel):
    """Response model for indicators calculation"""
    symbol: str
    start_date: datetime
    end_date: datetime
    indicators: Dict[str, List[float]]
    timestamps: List[str]

class MarketDataResponse(BaseModel):
    """Market data response model"""
    symbol: str
    data: List[Dict[str, Any]]
    total_records: int

class ValidationError(BaseModel):
    """Validation error model"""
    field: str
    message: str

class IndicatorResponse(BaseModel):
    """Response model for indicator calculations"""
    symbol: str
    indicators: Dict[str, Any]  # Dictionary of indicator name -> values
    data_points: int
    start_date: datetime
    end_date: datetime

class TradeInfo(BaseModel):
    """Individual trade information"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    size: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_minutes: Optional[int] = None
    entry_reason: str
    exit_reason: Optional[str] = None

class BacktestStats(BaseModel):
    """Comprehensive backtest results"""
    # Performance Metrics
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return_pct: float = Field(..., description="Annualized return percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    
    # Trade Statistics
    total_trades: int = Field(..., description="Total number of trades")
    win_rate_pct: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    avg_win_pct: float = Field(..., description="Average winning trade percentage")
    avg_loss_pct: float = Field(..., description="Average losing trade percentage")
    
    # Position Direction Statistics (NEW)
    long_positions: int = Field(default=0, description="Number of long positions opened")
    short_positions: int = Field(default=0, description="Number of short positions opened")
    
    # Win/Loss Statistics (NEW)
    largest_win: float = Field(default=0.0, description="Largest winning trade (absolute value)")
    largest_loss: float = Field(default=0.0, description="Largest losing trade (absolute value)")
    largest_win_pct: float = Field(default=0.0, description="Largest winning trade percentage")
    largest_loss_pct: float = Field(default=0.0, description="Largest losing trade percentage")
    
    # Risk Metrics
    volatility_pct: float = Field(..., description="Annual volatility percentage")
    var_95_pct: float = Field(..., description="Value at Risk (95%)")
    
    # Duration Statistics
    avg_trade_duration_hours: float = Field(..., description="Average trade duration in hours")
    max_trade_duration_hours: float = Field(..., description="Maximum trade duration in hours")
    
    # Additional Metrics
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_cash: float = Field(..., description="Initial capital")
    final_value: float = Field(..., description="Final portfolio value")
    
    # Trade Details
    trades: List[TradeInfo] = Field(default_factory=list, description="Individual trade details")
    
    # Indicator Data (for frontend charting)
    indicators: Optional[Dict[str, List[float]]] = Field(default=None, description="Calculated indicator values")
    timestamps: Optional[List[str]] = Field(default=None, description="Timestamps for indicator data")

class StrategyConfig(BaseModel):
    """Configuration for trading strategies"""
    strategy_type: str = Field(..., description="Type of strategy (ma_crossover, rsi_oversold, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Strategy-specific parameters")
    
    # Position sizing
    position_size_type: str = Field(default="fixed_amount", description="Position sizing method")
    position_size_value: float = Field(default=1000.0, description="Position size value")
    
    # Risk management
    stop_loss_pct: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, description="Take profit percentage")
    max_positions: int = Field(default=1, description="Maximum concurrent positions")

class ValidationResult(BaseModel):
    """Data validation result"""
    valid: bool
    table_exists: Optional[bool] = None
    record_count: Optional[int] = None
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    error: Optional[str] = None
