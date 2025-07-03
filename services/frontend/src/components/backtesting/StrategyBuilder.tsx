import React from 'react'
import { Strategy, TechnicalIndicator, EntryCondition, ExitCondition } from '../../types/backtesting'
import { Plus, Trash2 } from 'lucide-react'

// Default values configuration based on backend config.py
const INDICATOR_DEFAULTS: Record<string, { operator: string; value: string }> = {
  'macd': { operator: 'crosses_above', value: '0' },
  'rsi': { operator: 'less_than', value: '30' },
  'sma': { operator: 'crosses_above', value: 'close' },
  'ema': { operator: 'crosses_above', value: 'close' },
  'bb': { operator: 'less_than', value: 'bb_lower' },
  'stoch': { operator: 'less_than', value: '20' },
}

const EXIT_DEFAULTS: Record<string, { operator: string; value: string }> = {
  'macd': { operator: 'crosses_below', value: '0' },
  'rsi': { operator: 'greater_than', value: '70' },
  'sma': { operator: 'crosses_below', value: 'close' },
  'ema': { operator: 'crosses_below', value: 'close' },
  'bb': { operator: 'greater_than', value: 'bb_upper' },
  'stoch': { operator: 'greater_than', value: '80' },
}

interface StrategyBuilderProps {
  strategy: Strategy
  onChange: (strategy: Strategy) => void
  availableIndicators: TechnicalIndicator[]
  actionButtons?: React.ReactNode
}

const StrategyBuilder: React.FC<StrategyBuilderProps> = ({
  strategy,
  onChange,
  availableIndicators,
  actionButtons
}) => {
  const addEntryCondition = () => {
    const defaultIndicator = availableIndicators.find(ind => ind.name === 'macd') || availableIndicators[0]
    const defaults = INDICATOR_DEFAULTS[defaultIndicator?.name || 'macd'] || { operator: 'greater_than', value: '0' }
    
    // Add two conditions at once
    const newCondition1: EntryCondition = {
      id: `entry_${Date.now()}_1`,
      type: 'technical_indicator',
      indicator: defaultIndicator?.name || 'macd',
      operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
      value: defaults.value,
      enabled: true,
    }

    const newCondition2: EntryCondition = {
      id: `entry_${Date.now()}_2`,
      type: 'technical_indicator',
      indicator: defaultIndicator?.name || 'macd',
      operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
      value: defaults.value,
      enabled: false, // Second condition is disabled by default
    }

    onChange({
      ...strategy,
      entry_conditions: [...strategy.entry_conditions, newCondition1, newCondition2]
    })
  }

  const addExitCondition = () => {
    const defaultIndicator = availableIndicators.find(ind => ind.name === 'macd') || availableIndicators[0]
    const defaults = EXIT_DEFAULTS[defaultIndicator?.name || 'macd'] || { operator: 'less_than', value: '0' }

    // Add two conditions at once
    const newCondition1: ExitCondition = {
      id: `exit_${Date.now()}_1`,
      type: 'technical_indicator',
      indicator: defaultIndicator?.name || 'macd',
      operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
      value: defaults.value,
      enabled: true,
    }

    const newCondition2: ExitCondition = {
      id: `exit_${Date.now()}_2`,
      type: 'technical_indicator',
      indicator: defaultIndicator?.name || 'macd',
      operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
      value: defaults.value,
      enabled: false, // Second condition is disabled by default
    }

    onChange({
      ...strategy,
      exit_conditions: [...strategy.exit_conditions, newCondition1, newCondition2]
    })
  }

  const updateEntryCondition = (index: number, condition: EntryCondition) => {
    const newConditions = [...strategy.entry_conditions]
    newConditions[index] = condition
    onChange({ ...strategy, entry_conditions: newConditions })
  }

  const updateExitCondition = (index: number, condition: ExitCondition) => {
    const newConditions = [...strategy.exit_conditions]
    newConditions[index] = condition
    onChange({ ...strategy, exit_conditions: newConditions })
  }

  // Handle indicator change with auto-apply defaults
  const handleIndicatorChange = (index: number, newIndicator: string, isEntry: boolean) => {
    const defaults = isEntry 
      ? INDICATOR_DEFAULTS[newIndicator] || { operator: 'greater_than', value: '0' }
      : EXIT_DEFAULTS[newIndicator] || { operator: 'less_than', value: '0' }
    
    if (isEntry) {
      const condition = strategy.entry_conditions[index]
      updateEntryCondition(index, {
        ...condition,
        indicator: newIndicator,
        operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
        value: defaults.value
      })
    } else {
      const condition = strategy.exit_conditions[index]
      updateExitCondition(index, {
        ...condition,
        indicator: newIndicator,
        operator: defaults.operator as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below',
        value: defaults.value
      })
    }
  }

  const removeEntryCondition = (index: number) => {
    const newConditions = strategy.entry_conditions.filter((_, i) => i !== index)
    onChange({ ...strategy, entry_conditions: newConditions })
  }

  const removeExitCondition = (index: number) => {
    const newConditions = strategy.exit_conditions.filter((_, i) => i !== index)
    onChange({ ...strategy, exit_conditions: newConditions })
  }

  const updatePositionSizing = (field: string, value: string | number | undefined) => {
    onChange({ ...strategy, [field]: value })
  }

  const operatorOptions = [
    { value: 'greater_than', label: '>' },
    { value: 'less_than', label: '<' },
    { value: 'equals', label: '=' },
    { value: 'crosses_above', label: 'Crosses Above' },
    { value: 'crosses_below', label: 'Crosses Below' },
  ]

  const renderEntryConditionForm = (condition: EntryCondition, index: number) => {
    return (
      <div key={condition.id} className="border rounded-lg p-3 bg-card">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={condition.enabled}
                onChange={(e) => updateEntryCondition(index, { ...condition, enabled: e.target.checked })}
                className="sr-only"
                aria-label={`Enable entry condition ${index + 1}`}
              />
              <button
                onClick={() => updateEntryCondition(index, { ...condition, enabled: !condition.enabled })}
                className={`w-11 h-6 rounded-full relative transition-colors ${
                  condition.enabled ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                }`}
                aria-label={`Toggle entry condition ${index + 1}`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    condition.enabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
            <span className="text-sm font-medium">Entry {index + 1}</span>
          </div>
          <button
            onClick={() => removeEntryCondition(index)}
            className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
            aria-label={`Remove entry condition ${index + 1}`}
          >
            <Trash2 size={12} />
          </button>
        </div>

        {/* Horizontal layout for compact form */}
        <div className="flex gap-3 items-end">
          <div className="w-56">
            <label className="block text-sm font-medium mb-1">Indicator</label>
            <select
              value={condition.indicator || ''}
              onChange={(e) => handleIndicatorChange(index, e.target.value, true)}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              title="Select technical indicator"
            >
              {availableIndicators.map((indicator) => (
                <option key={indicator.id} value={indicator.name}>
                  {indicator.display_name}
                </option>
              ))}
            </select>
          </div>

          <div className="w-36">
            <label className="block text-sm font-medium mb-1">Operator</label>
            <select
              value={condition.operator}
              onChange={(e) => updateEntryCondition(index, { 
                ...condition, 
                operator: e.target.value as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below' 
              })}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              title="Comparison operator"
            >
              {operatorOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="w-20">
            <label className="block text-sm font-medium mb-1">Value</label>
            <input
              type="text"
              value={condition.value}
              onChange={(e) => updateEntryCondition(index, { ...condition, value: e.target.value })}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              placeholder="Value"
              title="Condition value"
            />
          </div>
        </div>
      </div>
    )
  }

  const renderExitConditionForm = (condition: ExitCondition, index: number) => {
    return (
      <div key={condition.id} className="border rounded-lg p-3 bg-card">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={condition.enabled}
                onChange={(e) => updateExitCondition(index, { ...condition, enabled: e.target.checked })}
                className="sr-only"
                aria-label={`Enable exit condition ${index + 1}`}
              />
              <button
                onClick={() => updateExitCondition(index, { ...condition, enabled: !condition.enabled })}
                className={`w-11 h-6 rounded-full relative transition-colors ${
                  condition.enabled ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                }`}
                aria-label={`Toggle exit condition ${index + 1}`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    condition.enabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
            <span className="text-sm font-medium">Exit {index + 1}</span>
          </div>
          <button
            onClick={() => removeExitCondition(index)}
            className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
            aria-label={`Remove exit condition ${index + 1}`}
          >
            <Trash2 size={12} />
          </button>
        </div>

        {/* Horizontal layout for compact form */}
        <div className="flex gap-3 items-end">
          <div className="w-56">
            <label className="block text-sm font-medium mb-1">Indicator</label>
            <select
              value={condition.indicator || ''}
              onChange={(e) => handleIndicatorChange(index, e.target.value, false)}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              title="Select technical indicator"
            >
              {availableIndicators.map((indicator) => (
                <option key={indicator.id} value={indicator.name}>
                  {indicator.display_name}
                </option>
              ))}
            </select>
          </div>

          <div className="w-36">
            <label className="block text-sm font-medium mb-1">Operator</label>
            <select
              value={condition.operator}
              onChange={(e) => updateExitCondition(index, { 
                ...condition, 
                operator: e.target.value as 'greater_than' | 'less_than' | 'equals' | 'crosses_above' | 'crosses_below' 
              })}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              title="Comparison operator"
            >
              {operatorOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="w-20">
            <label className="block text-sm font-medium mb-1">Value</label>
            <input
              type="text"
              value={condition.value}
              onChange={(e) => updateExitCondition(index, { ...condition, value: e.target.value })}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              placeholder="Value"
              title="Condition value"
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-card border rounded-lg h-full flex flex-col">
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold">Strategy Builder</h3>
            <input
              type="text"
              value={strategy.name}
              onChange={(e) => onChange({ ...strategy, name: e.target.value })}
              className="w-80 px-4 py-2 border rounded-md bg-background text-sm"
              placeholder="Strategy name"
            />
          </div>
          <div className="flex items-center gap-2">
            {actionButtons}
          </div>
        </div>
      </div>

      {/* Horizontal Layout - Position Sizing Above Rules */}
      <div className="flex-1 p-4 space-y-4 overflow-y-auto">
        {/* Position Sizing Panel - Full Width at Top */}
        <div className="bg-muted/20 rounded-lg p-4">
          <h4 className="font-medium text-primary mb-3">Position Sizing</h4>
          
          <div className="flex gap-3 flex-wrap">
            {/* Sizing Method - Wider */}
            <div className="w-56">
              <label className="block text-sm font-medium mb-1">Sizing Method</label>
              <select
                value={strategy.position_sizing}
                onChange={(e) => updatePositionSizing('position_sizing', e.target.value)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                title="Position sizing method"
              >
                <option value="percentage">Percentage of Portfolio</option>
                <option value="fixed">Size (No. of Units)</option>
                <option value="volatility">Volatility-Based (% of portfolio)</option>
                <option value="kelly">Kelly Criterion</option>
              </select>
            </div>

            {/* Size - Narrower width */}
            <div className="w-24">
              <label className="block text-sm font-medium mb-1">
                {strategy.position_sizing === 'percentage' ? 'Size (%)' : 
                 strategy.position_sizing === 'fixed' ? 'Size (Units)' :
                 strategy.position_sizing === 'volatility' ? 'Risk Size (%)' :
                 'Size (% or Units)'}
              </label>
              <input
                type="number"
                value={strategy.position_size}
                onChange={(e) => updatePositionSizing('position_size', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                step={strategy.position_sizing === 'percentage' ? '1' : 
                      strategy.position_sizing === 'fixed' ? '0.01' : '1'}
                min="0"
                max={strategy.position_sizing === 'percentage' ? '100' : 
                     strategy.position_sizing === 'volatility' ? '50' : undefined}
                title={strategy.position_sizing === 'percentage' ? 'Percentage of portfolio (1-100%)' :
                       strategy.position_sizing === 'fixed' ? 'Number of units/shares to trade (e.g., 0.1 BTC)' :
                       strategy.position_sizing === 'volatility' ? 'Maximum risk percentage based on volatility (1-50%)' :
                       'Position size value'}
              />
            </div>

            <div className="w-24">
              <label className="block text-sm font-medium mb-1">Max Position</label>
              <input
                type="number"
                value={strategy.max_positions}
                onChange={(e) => updatePositionSizing('max_positions', parseInt(e.target.value))}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                min="1"
                title="Maximum concurrent positions"
              />
            </div>

            <div className="w-32">
              <label className="block text-sm font-medium mb-1">Stop Loss (%)</label>
              <input
                type="number"
                value={strategy.stop_loss || ''}
                onChange={(e) => updatePositionSizing('stop_loss', e.target.value ? parseFloat(e.target.value) : undefined)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                step="0.1"
                min="0"
                placeholder="Optional"
                title="Stop loss percentage"
              />
            </div>

            <div className="w-36">
              <label className="block text-sm font-medium mb-1">Profit Taking (%)</label>
              <input
                type="number"
                value={strategy.take_profit || ''}
                onChange={(e) => updatePositionSizing('take_profit', e.target.value ? parseFloat(e.target.value) : undefined)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                step="0.1"
                min="0"
                placeholder="Optional"
                title="Take profit percentage"
              />
            </div>

            <div className="w-40">
              <label className="block text-sm font-medium mb-1">Position Direction</label>
              <select
                value={strategy.position_direction}
                onChange={(e) => updatePositionSizing('position_direction', e.target.value)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                title="Position direction preference"
              >
                <option value="long_only">Long Only</option>
                <option value="short_only">Short Only</option>
                <option value="both">Both Long & Short</option>
              </select>
            </div>

            <div className="w-48">
              <label className="block text-sm font-medium mb-1">Max Position Strategy</label>
              <select
                value={strategy.max_position_strategy}
                onChange={(e) => updatePositionSizing('max_position_strategy', e.target.value)}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                title="How to handle max position limit"
              >
                <option value="ignore">Ignore New Signals</option>
                <option value="replace_oldest">Replace Oldest</option>
                <option value="replace_worst">Replace Worst</option>
              </select>
            </div>
          </div>
        </div>

        {/* Entry and Exit Conditions - Horizontal Layout, Wider */}
        <div className="flex gap-4">
          {/* Entry Conditions Panel - Wider by 10% */}
          <div className="bg-muted/20 rounded-lg p-4 flex-[1.1]">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-primary">Entry Rules</h4>
              <button
                onClick={addEntryCondition}
                className="flex items-center space-x-1 px-3 py-1 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90"
              >
                <Plus size={14} />
                <span>Add</span>
              </button>
            </div>
            
            {strategy.entry_conditions.length === 0 ? (
              <div className="text-center py-4 text-muted-foreground">
                <p className="text-sm">No entry conditions</p>
                <p className="text-xs">Click "Add" to get started</p>
              </div>
            ) : (
              <div className="space-y-2">
                {strategy.entry_conditions.map((condition, index) =>
                  renderEntryConditionForm(condition, index)
                )}
              </div>
            )}
          </div>

          {/* Exit Conditions Panel - Wider by 10% */}
          <div className="bg-muted/20 rounded-lg p-4 flex-[1.1]">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-primary">Exit Rules</h4>
              <button
                onClick={addExitCondition}
                className="flex items-center space-x-1 px-3 py-1 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90"
              >
                <Plus size={14} />
                <span>Add</span>
              </button>
            </div>
            
            {strategy.exit_conditions.length === 0 ? (
              <div className="text-center py-4 text-muted-foreground">
                <p className="text-sm">No exit conditions</p>
                <p className="text-xs">Click "Add" to get started</p>
              </div>
            ) : (
              <div className="space-y-2">
                {strategy.exit_conditions.map((condition, index) =>
                  renderExitConditionForm(condition, index)
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default StrategyBuilder
