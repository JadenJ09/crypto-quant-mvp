import React from 'react'
import { BacktestParams } from '../../types/backtesting'
import { Calendar, DollarSign, Percent } from 'lucide-react'

interface BacktestParametersPanelProps {
  params: BacktestParams
  onChange: (params: BacktestParams) => void
}

const BacktestParametersPanel: React.FC<BacktestParametersPanelProps> = ({ params, onChange }) => {
  const handleChange = (key: keyof BacktestParams, value: string | number) => {
    onChange({
      ...params,
      [key]: value,
    })
  }

  return (
    <div className="bg-card rounded-lg border p-4 h-full flex flex-col min-h-[500px]">
      <h3 className="text-lg font-semibold mt-2 mb-2 flex items-center">
        <DollarSign size={18} className="mr-2" />
        Backtest Parameters
      </h3>
      <div className="border-b pb-3 mb-2"></div>
      <div className="space-y-4 flex-1">
        {/* Symbol Selection */}
        <div>
          <label className="block text-sm font-medium mt-2 mb-1">Symbol</label>
          <select
            value={params.symbol}
            onChange={(e) => handleChange('symbol', e.target.value)}
            className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
            title="Select trading symbol"
          >
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="ADAUSDT">ADA/USDT</option>
            <option value="SOLUSDT">SOL/USDT</option>
          </select>
        </div>

        {/* Timeframe Selection */}
        <div>
          <label className="block text-sm font-medium mb-1">Timeframe</label>
          <select
            value={params.timeframe}
            onChange={(e) => handleChange('timeframe', e.target.value)}
            className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
            title="Select timeframe"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-sm font-medium mb-1 flex items-center">
              <Calendar size={14} className="mr-1" />
              Start Date
            </label>
            <input
              type="date"
              value={params.start_date}
              onChange={(e) => handleChange('start_date', e.target.value)}
              className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
              title="Select start date for backtesting"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 flex items-center">
              <Calendar size={14} className="mr-1" />
              End Date
            </label>
            <input
              type="date"
              value={params.end_date}
              onChange={(e) => handleChange('end_date', e.target.value)}
              className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
              title="Select end date for backtesting"
            />
          </div>
        </div>

        {/* Initial Capital */}
        <div>
          <label className="block text-sm font-medium mb-1">Initial Capital ($)</label>
          <input
            type="number"
            value={params.initial_cash}
            onChange={(e) => handleChange('initial_cash', parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
            min="100"
            step="100"
            title="Initial capital amount"
            placeholder="Enter initial capital"
          />
        </div>

        {/* Commission */}
        <div>
          <label className="text-sm font-medium mb-1 flex items-center">
            <Percent size={14} className="mr-1" />
            Commission (%)
          </label>
          <input
            type="number"
            value={params.commission * 100}
            onChange={(e) => handleChange('commission', (parseFloat(e.target.value) || 0) / 100)}
            className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
            min="0"
            max="5"
            step="0.01"
            title="Commission percentage"
            placeholder="Enter commission rate"
          />
        </div>

        {/* Slippage */}
        <div>
          <label className="text-sm font-medium mb-1 flex items-center">
            <Percent size={14} className="mr-1" />
            Slippage (%)
          </label>
          <input
            type="number"
            value={params.slippage * 100}
            onChange={(e) => handleChange('slippage', (parseFloat(e.target.value) || 0) / 100)}
            className="w-full px-3 py-2 border rounded-md bg-background text-foreground text-sm"
            min="0"
            max="2"
            step="0.01"
            title="Slippage percentage"
            placeholder="Enter slippage rate"
          />
        </div>
      </div>
    </div>
  )
}

export default BacktestParametersPanel
