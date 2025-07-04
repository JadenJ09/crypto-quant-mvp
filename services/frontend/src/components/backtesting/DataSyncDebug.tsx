import React from 'react'
import { Trade, BacktestResult } from '../../types/backtesting'
import { TradeDataPoint } from './charts/types'

interface DataSyncDebugProps {
  chartTradeData: TradeDataPoint[]
  pnlTradeData: Trade[]
  backtestResult: BacktestResult | null
}

const DataSyncDebug: React.FC<DataSyncDebugProps> = ({ chartTradeData, pnlTradeData, backtestResult }) => {
  console.log('🔍 Data Sync Debug:', {
    chartTradeCount: chartTradeData.length,
    pnlTradeCount: pnlTradeData.length,
    areEqual: chartTradeData.length === pnlTradeData.length,
    backtestTotal: backtestResult?.trades?.length || 0,
    chartTrades: chartTradeData.map(t => ({ side: t.side, time: t.time, exit_time: t.exit_time })),
    pnlTrades: pnlTradeData.map(t => ({ side: t.side, time: t.entry_time, exit_time: t.exit_time, pnl: t.pnl }))
  })

  const chartMarkerCount = chartTradeData.length + chartTradeData.filter(t => t.exit_time).length

  return (
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg border border-yellow-300 dark:border-yellow-700">
      <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">📊 Data Sync Debug</h4>
      <div className="text-sm text-yellow-700 dark:text-yellow-300 space-y-2">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="font-medium">Backtest Result:</div>
            <div>Total Trades: {backtestResult?.trades?.length || 0}</div>
            <div>Strategy: {backtestResult?.strategy_id || 'N/A'}</div>
          </div>
          <div>
            <div className="font-medium">P&L History:</div>
            <div>Trade Count: {pnlTradeData.length}</div>
            <div>Sides: {pnlTradeData.map(t => t.side).join(', ')}</div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="font-medium">Chart Data:</div>
            <div>Trade Count: {chartTradeData.length}</div>
            <div>Marker Count: {chartMarkerCount}</div>
            <div>With Exit: {chartTradeData.filter(t => t.exit_time).length}</div>
          </div>
          <div className={`font-medium p-2 rounded ${chartTradeData.length === pnlTradeData.length ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200' : 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200'}`}>
            Sync Status: {chartTradeData.length === pnlTradeData.length ? '✅ In Sync' : '❌ Out of Sync'}
          </div>
        </div>
        {chartTradeData.length !== pnlTradeData.length && (
          <div className="bg-red-100 dark:bg-red-900 p-2 rounded border border-red-300 dark:border-red-700">
            <div className="font-medium text-red-800 dark:text-red-200">⚠️ Sync Issue Detected</div>
            <div className="text-red-700 dark:text-red-300">
              Chart shows {chartTradeData.length} trades, P&L shows {pnlTradeData.length} trades
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DataSyncDebug
