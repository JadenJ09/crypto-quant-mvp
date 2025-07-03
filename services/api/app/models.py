# ==============================================================================
# File: services/api/app/models.py
# Description: Centralized Pydantic data models for the API service
# ==============================================================================

from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# =============================================================================
# MARKET DATA MODELS
# =============================================================================

class OHLCVOut(BaseModel):
    """OHLCV data output model"""
    time: datetime
    open: float = Field(..., alias='o')
    high: float = Field(..., alias='h')
    low: float = Field(..., alias='l')
    close: float = Field(..., alias='c')
    volume: float = Field(..., alias='v')

    class Config:
        from_attributes = True
        populate_by_name = True

class CandlestickData(BaseModel):
    """Enhanced candlestick data with indicators"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None

class TimeframeInfo(BaseModel):
    """Available timeframe information"""
    label: str
    value: str
    table: str
    description: str

class SymbolInfo(BaseModel):
    """Trading symbol information"""
    symbol: str
    name: str
    exchange: str
    base_currency: str
    quote_currency: str

# =============================================================================
# BACKTESTING MODELS
# =============================================================================

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

# Frontend-compatible strategy models
class EntryCondition(BaseModel):
    """Entry condition model to match frontend"""
    id: str
    type: str  # 'technical_indicator', 'price_action', 'ml_signal'
    indicator: Optional[str] = None
    operator: str  # 'greater_than', 'less_than', 'crosses_above', etc.
    value: Any  # Can be number or string
    enabled: bool = True

class ExitCondition(BaseModel):
    """Exit condition model to match frontend"""
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
    position_sizing: str = Field(default="percentage")  # 'fixed', 'percentage', 'kelly', 'volatility'
    position_size: float = Field(default=0.1)
    position_direction: str = Field(default="long_only")  # 'long_only', 'short_only', 'both'
    max_positions: int = Field(default=1)
    max_position_strategy: str = Field(default="ignore")  # 'ignore', 'replace_oldest', 'replace_worst'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# Legacy strategy config for backward compatibility
class StrategyConfig(BaseModel):
    """Configuration for trading strategies (legacy)"""
    strategy_type: str = Field(..., description="Type of strategy (ma_crossover, rsi_oversold, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Strategy-specific parameters")
    
    # Position sizing
    position_size_type: str = Field(default="fixed_amount", description="Position sizing method")
    position_size_value: float = Field(default=1000.0, description="Position size value")
    
    # Risk management
    stop_loss_pct: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, description="Take profit percentage")
    max_positions: int = Field(default=1, description="Maximum concurrent positions")

class BacktestRequest(BaseModel):
    """Request model for backtesting - updated to support both legacy and frontend formats"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (5min, 1hour, etc.)")
    start_date: Optional[str] = Field(None, description="Backtest start date")
    end_date: Optional[str] = Field(None, description="Backtest end date")  
    initial_cash: float = Field(default=100000.0, description="Initial capital")
    commission: float = Field(default=0.001, description="Commission rate (0.1%)")
    slippage: float = Field(default=0.0005, description="Slippage rate")
    
    # Use Union to support both strategy formats
    strategy: Union[StrategyConfig, FrontendStrategy] = Field(..., description="Strategy configuration")
    
    @property
    def frontend_strategy(self) -> Optional[FrontendStrategy]:
        """Get frontend strategy if available"""
        return self.strategy if isinstance(self.strategy, FrontendStrategy) else None
    
    @property
    def legacy_strategy(self) -> Optional[StrategyConfig]:
        """Get legacy strategy if available"""
        return self.strategy if isinstance(self.strategy, StrategyConfig) else None

class StrategyInfo(BaseModel):
    """Available strategy information"""
    name: str
    display_name: str
    description: str
    parameters: List[Dict[str, Any]]

# =============================================================================
# VALIDATION MODELS
# =============================================================================

class ValidationResult(BaseModel):
    """Data validation result"""
    valid: bool
    table_exists: Optional[bool] = None
    record_count: Optional[int] = None
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    error: Optional[str] = None

class IndicatorList(BaseModel):
    """Available indicators for a timeframe"""
    timeframe: str
    indicators: List[str]

# =============================================================================
# COMMON RESPONSE MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str = "0.1.0"
    services: Dict[str, str] = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
