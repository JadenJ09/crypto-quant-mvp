import React from 'react'
import { BacktestResult } from '../../types/backtesting'
import { TrendingUp, TrendingDown, DollarSign, Target, Calendar, BarChart3, ArrowUpDown } from 'lucide-react'

interface BacktestResultsProps {
  result: BacktestResult | null
}

const BacktestResults: React.FC<BacktestResultsProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="bg-card border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Backtest Results</h3>
        <div className="text-center py-12 text-muted-foreground">
          <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
          <p>No backtest results available</p>
          <p className="text-sm">Run a backtest to see the results here</p>
        </div>
      </div>
    )
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`
  }

  const formatNumber = (value: number) => {
    return value.toFixed(2)
  }

  const getReturnColor = (value: number) => {
    return value >= 0 ? 'text-green-600' : 'text-red-600'
  }

  const getReturnIcon = (value: number) => {
    return value >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />
  }

  return (
    <div className="bg-card border rounded-lg">
      <div className="p-4 border-b">
        <h3 className="text-lg font-semibold">Backtest Results</h3>
        <div className="text-sm text-muted-foreground">
          {result.symbol} • {result.timeframe} • {result.start_date} to {result.end_date}
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Performance Summary */}
        <div className="space-y-3">
          <h4 className="font-medium text-sm text-muted-foreground uppercase tracking-wide">
            Performance Summary
          </h4>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <DollarSign size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Initial Capital</span>
              </div>
              <div className="text-lg font-semibold">
                {formatCurrency(result.initial_cash)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <DollarSign size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Final Capital</span>
              </div>
              <div className="text-lg font-semibold">
                {formatCurrency(result.final_cash)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                {getReturnIcon(result.total_return)}
                <span className="text-xs font-medium text-muted-foreground">Total Return</span>
              </div>
              <div className={`text-lg font-semibold ${getReturnColor(result.total_return)}`}>
                {formatCurrency(result.total_return)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                {getReturnIcon(result.total_return_pct)}
                <span className="text-xs font-medium text-muted-foreground">Return %</span>
              </div>
              <div className={`text-lg font-semibold ${getReturnColor(result.total_return_pct)}`}>
                {formatPercentage(result.total_return_pct)}
              </div>
            </div>
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="space-y-3">
          <h4 className="font-medium text-sm text-muted-foreground uppercase tracking-wide">
            Risk Metrics
          </h4>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingDown size={16} className="text-red-500" />
                <span className="text-xs font-medium text-muted-foreground">Max Drawdown</span>
              </div>
              <div className="text-lg font-semibold text-red-600">
                {formatCurrency(result.max_drawdown)}
              </div>
              <div className="text-sm text-red-500">
                ({formatPercentage(result.max_drawdown_pct)})
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Target size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Sharpe Ratio</span>
              </div>
              <div className="text-lg font-semibold">
                {formatNumber(result.sharpe_ratio)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Target size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Win Rate</span>
              </div>
              <div className="text-lg font-semibold">
                {formatPercentage(result.win_rate * 100)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Target size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Profit Factor</span>
              </div>
              <div className="text-lg font-semibold">
                {formatNumber(result.profit_factor)}
              </div>
            </div>
          </div>
        </div>

        {/* Trade Statistics */}
        <div className="space-y-3">
          <h4 className="font-medium text-sm text-muted-foreground uppercase tracking-wide">
            Trade Statistics
          </h4>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Calendar size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Total Trades</span>
              </div>
              <div className="text-lg font-semibold">
                {result.total_trades}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingUp size={16} className="text-green-500" />
                <span className="text-xs font-medium text-muted-foreground">Winning Trades</span>
              </div>
              <div className="text-lg font-semibold text-green-600">
                {result.winning_trades}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingDown size={16} className="text-red-500" />
                <span className="text-xs font-medium text-muted-foreground">Losing Trades</span>
              </div>
              <div className="text-lg font-semibold text-red-600">
                {result.losing_trades}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingUp size={16} className="text-blue-500" />
                <span className="text-xs font-medium text-muted-foreground">Long Positions</span>
              </div>
              <div className="text-lg font-semibold text-blue-600">
                {result.long_positions || 0}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingDown size={16} className="text-orange-500" />
                <span className="text-xs font-medium text-muted-foreground">Short Positions</span>
              </div>
              <div className="text-lg font-semibold text-orange-600">
                {result.short_positions || 0}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <DollarSign size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Avg Win</span>
              </div>
              <div className="text-lg font-semibold text-green-600">
                {formatCurrency(result.avg_win)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <DollarSign size={16} className="text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">Avg Loss</span>
              </div>
              <div className="text-lg font-semibold text-red-600">
                {formatCurrency(Math.abs(result.avg_loss))}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingUp size={16} className="text-green-500" />
                <span className="text-xs font-medium text-muted-foreground">Largest Win</span>
              </div>
              <div className="text-lg font-semibold text-green-600">
                {formatCurrency(result.largest_win)}
              </div>
            </div>

            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <TrendingDown size={16} className="text-red-500" />
                <span className="text-xs font-medium text-muted-foreground">Largest Loss</span>
              </div>
              <div className="text-lg font-semibold text-red-600">
                {formatCurrency(Math.abs(result.largest_loss))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent Trades */}
        {result.trades.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium text-sm text-muted-foreground uppercase tracking-wide">
              Recent Trades (Last 5)
            </h4>
            
            <div className="space-y-2">
              {result.trades.slice(-5).reverse().map((trade, index) => (
                <div key={trade.trade_id || `trade-${index}`} className="p-3 bg-muted/30 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-muted-foreground">
                          Trade #{trade.trade_id}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded ${
                          trade.side === 'long' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {trade.side.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {new Date(trade.entry_time).toLocaleDateString()} - {trade.exit_time ? new Date(trade.exit_time).toLocaleDateString() : 'N/A'}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Entry: {formatCurrency(trade.entry_price || 0)} → Exit: {formatCurrency(trade.exit_price || 0)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-base font-semibold ${getReturnColor(trade.pnl || 0)}`}>
                        {formatCurrency(trade.pnl || 0)}
                      </div>
                      <div className={`text-sm ${getReturnColor(trade.pnl_pct || 0)}`}>
                        {formatPercentage(trade.pnl_pct || 0)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default BacktestResults
