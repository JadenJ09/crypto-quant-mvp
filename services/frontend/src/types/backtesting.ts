export interface BacktestParams {
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  initial_cash: number
  commission: number
  slippage: number
}

export interface EntryCondition {
  id: string
  type: 'technical_indicator' | 'price_action' | 'ml_signal'
  indicator?: string
  operator: 'greater_than' | 'less_than' | 'crosses_above' | 'crosses_below' | 'equals'
  value: number | string
  enabled: boolean
}

export interface ExitCondition {
  id: string
  type: 'technical_indicator' | 'price_action' | 'stop_loss' | 'take_profit' | 'time_based'
  indicator?: string
  operator: 'greater_than' | 'less_than' | 'crosses_above' | 'crosses_below' | 'equals'
  value: number | string
  enabled: boolean
}

export interface Strategy {
  id: string
  name: string
  description: string
  entry_conditions: EntryCondition[]
  exit_conditions: ExitCondition[]
  position_sizing: 'fixed' | 'percentage' | 'kelly' | 'volatility'
  position_size: number
  position_direction: 'long_only' | 'short_only' | 'both'
  max_positions: number
  max_position_strategy: 'ignore' | 'replace_oldest' | 'replace_worst'
  stop_loss?: number
  take_profit?: number
}

export interface BacktestResult {
  strategy_id: string
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  initial_cash: number
  final_cash: number
  total_return: number
  total_return_pct: number
  max_drawdown: number
  max_drawdown_pct: number
  sharpe_ratio: number
  win_rate: number
  profit_factor: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  avg_win: number
  avg_loss: number
  largest_win: number
  largest_loss: number
  long_positions: number  // New field for long position count
  short_positions: number // New field for short position count
  trades: Trade[]
  equity_curve: EquityPoint[]
  drawdown_curve: DrawdownPoint[]
  // Indicator data for charting
  indicators?: Record<string, number[]>
  timestamps?: string[]
}

export interface Trade {
  trade_id: number
  entry_time: string
  exit_time: string
  entry_price: number
  exit_price: number
  quantity: number
  side: 'long' | 'short'
  pnl: number
  pnl_pct: number
  duration: number
  entry_reason: string
  exit_reason: string
}

export interface EquityPoint {
  time: string
  equity: number
  cash: number
  holdings_value: number
}

export interface DrawdownPoint {
  time: string
  drawdown: number
  drawdown_pct: number
}

export interface TechnicalIndicator {
  id: string
  name: string
  display_name: string
  category: 'trend' | 'momentum' | 'volatility' | 'volume'
  parameters: IndicatorParameter[]
}

export interface IndicatorParameter {
  name: string
  display_name: string
  type: 'number' | 'string' | 'boolean'
  default_value: string | number | boolean
  min_value?: number
  max_value?: number
  options?: string[]
}

export interface ChartData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  // Technical indicators as optional numeric fields
  sma_20?: number
  sma_50?: number
  sma_100?: number
  rsi?: number
  macd?: number
  bb_upper?: number
  bb_lower?: number
  [key: string]: string | number | undefined
}

export interface BacktestVisualization {
  candlestick_data: ChartData[]
  indicators: { [key: string]: number[] }
  entry_signals: TradeSignal[]
  exit_signals: TradeSignal[]
  equity_curve: EquityPoint[]
  drawdown_curve: DrawdownPoint[]
}

export interface TradeSignal {
  time: string
  price: number
  type: 'entry' | 'exit'
  side: 'long' | 'short'
  reason: string
}
