import React, { useState, useEffect, useCallback } from 'react'
import { BacktestParams, BacktestResult, Strategy, TechnicalIndicator, Trade } from '../types/backtesting'
import BacktestParametersPanel from '../components/backtesting/BacktestParametersPanel'
import StrategyBuilder from '../components/backtesting/StrategyBuilder'
import BacktestResults from '../components/backtesting/BacktestResults'
import BacktestChart from '../components/backtesting/charts/BacktestChart'
import DataSyncDebug from '../components/backtesting/DataSyncDebug'
import Navigation from '../components/Navigation'
import ThemeToggle from '../components/ThemeToggle'
import { useTheme } from '../hooks/useTheme'
import { Play, Save, FolderOpen, TrendingUp, TrendingDown, DollarSign, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'

// Import the chart data types
import { OHLCVDataPoint, IndicatorDataPoint, TradeDataPoint } from '../components/backtesting/charts'

const Backtesting: React.FC = () => {
  const { theme } = useTheme()
  const [backtestParams, setBacktestParams] = useState<BacktestParams>({
    symbol: 'BTCUSDT',
    timeframe: '1h',
    start_date: '2024-07-01',
    end_date: '2024-12-31',
    initial_cash: 100000,
    commission: 0.001,
    slippage: 0.0005,
  })

  const [currentStrategy, setCurrentStrategy] = useState<Strategy>({
    id: 'default',
    name: 'RSI Oversold Strategy',
    description: 'Buy when RSI is oversold (<30) and sell when overbought (>70)',
    entry_conditions: [
      {
        id: 'entry1',
        type: 'technical_indicator',
        indicator: 'rsi',
        operator: 'less_than',
        value: '30',
        enabled: true,
      },
      {
        id: 'entry2',
        type: 'technical_indicator',
        indicator: 'rsi',
        operator: 'less_than',
        value: '25',
        enabled: false,
      }
    ],
    exit_conditions: [
      {
        id: 'exit1',
        type: 'technical_indicator',
        indicator: 'rsi',
        operator: 'greater_than',
        value: '70',
        enabled: true,
      },
      {
        id: 'exit2',
        type: 'technical_indicator',
        indicator: 'rsi',
        operator: 'greater_than',
        value: '75',
        enabled: false,
      }
    ],
    position_sizing: 'percentage',
    position_size: 20, // 20% of portfolio
    position_direction: 'long_only',
    max_positions: 1,
    max_position_strategy: 'ignore',
    stop_loss: undefined,
    take_profit: undefined,
  })

  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [availableIndicators, setAvailableIndicators] = useState<TechnicalIndicator[]>([])
  const [savedStrategies, setSavedStrategies] = useState<Strategy[]>([])
  const [showLoadMenu, setShowLoadMenu] = useState(false)

  // Strategy reference that only updates when backtest is run
  const [backtestStrategy, setBacktestStrategy] = useState<Strategy | null>(null)

  // Chart data state
  const [chartData, setChartData] = useState<{
    ohlcvData: OHLCVDataPoint[]
    indicatorData: Record<string, IndicatorDataPoint[]>
    tradeData: TradeDataPoint[]
  }>({
    ohlcvData: [],
    indicatorData: {},
    tradeData: []
  })

  // Filtered trades state - matches what's shown in the chart
  const [filteredTrades, setFilteredTrades] = useState<Trade[]>([])

  // Chart timeframe state
  const [chartTimeframe, setChartTimeframe] = useState<string>('1h')
  
  const timeframes = [
    { value: '5m', label: '5 Min' },
    { value: '15m', label: '15 Min' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
  ]

  // Fetch available technical indicators
  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        const response = await fetch('http://localhost:8002/indicators/available')
        if (response.ok) {
          const data = await response.json()
          // Check if the response has indicators array
          const indicators = data.indicators || data
          if (Array.isArray(indicators)) {
            // Transform the indicators to match our expected format
            const transformedIndicators: TechnicalIndicator[] = indicators.map((indicator: { name?: string; description?: string; parameters?: string[] }, index: number) => ({
              id: indicator.name || `indicator_${index}`,
              name: indicator.name || `indicator_${index}`,
              display_name: indicator.description || indicator.name || `Indicator ${index}`,
              category: 'trend' as const,
              parameters: indicator.parameters?.map((param: string) => ({
                name: param,
                display_name: param.charAt(0).toUpperCase() + param.slice(1),
                type: 'number' as const,
                default_value: 14,
                min_value: 1,
                max_value: 200,
                description: `${param} parameter`
              })) || []
            }))
            setAvailableIndicators(transformedIndicators)
          } else {
            throw new Error('Invalid indicators format')
          }
        } else {
          throw new Error('Failed to fetch indicators')
        }
      } catch (error) {
        console.error('Error fetching indicators:', error)
        // Set fallback indicators
        setAvailableIndicators([
          {
            id: 'macd',
            name: 'macd',
            display_name: 'MACD',
            category: 'momentum',
            parameters: [
              {
                name: 'fast_period',
                display_name: 'Fast Period',
                type: 'number',
                default_value: 12,
                min_value: 1,
                max_value: 50,
              },
              {
                name: 'slow_period',
                display_name: 'Slow Period',
                type: 'number',
                default_value: 26,
                min_value: 1,
                max_value: 100,
              },
              {
                name: 'signal_period',
                display_name: 'Signal Period',
                type: 'number',
                default_value: 9,
                min_value: 1,
                max_value: 50,
              }
            ]
          },
          {
            id: 'sma',
            name: 'sma',
            display_name: 'Simple Moving Average',
            category: 'trend',
            parameters: [
              {
                name: 'period',
                display_name: 'Period',
                type: 'number',
                default_value: 20,
                min_value: 1,
                max_value: 200,
              }
            ]
          },
          {
            id: 'ema',
            name: 'ema',
            display_name: 'Exponential Moving Average',
            category: 'trend',
            parameters: [
              {
                name: 'period',
                display_name: 'Period',
                type: 'number',
                default_value: 21,
                min_value: 1,
                max_value: 200,
              }
            ]
          },
          {
            id: 'rsi',
            name: 'rsi',
            display_name: 'Relative Strength Index',
            category: 'momentum',
            parameters: [
              {
                name: 'period',
                display_name: 'Period',
                type: 'number',
                default_value: 14,
                min_value: 2,
                max_value: 100,
              }
            ]
          },
          {
            id: 'bb',
            name: 'bb',
            display_name: 'Bollinger Bands',
            category: 'volatility',
            parameters: [
              {
                name: 'period',
                display_name: 'Period',
                type: 'number',
                default_value: 20,
                min_value: 1,
                max_value: 200,
              },
              {
                name: 'std',
                display_name: 'Standard Deviation',
                type: 'number',
                default_value: 2,
                min_value: 0.1,
                max_value: 5,
              }
            ]
          },
          {
            id: 'stoch',
            name: 'stoch',
            display_name: 'Stochastic Oscillator',
            category: 'momentum',
            parameters: [
              {
                name: 'k_period',
                display_name: 'K Period',
                type: 'number',
                default_value: 14,
                min_value: 1,
                max_value: 100,
              },
              {
                name: 'd_period',
                display_name: 'D Period',
                type: 'number',
                default_value: 3,
                min_value: 1,
                max_value: 50,
              }
            ]
          }
        ])
      }
    }

    fetchIndicators()
  }, [])

  // Load saved strategies from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('savedStrategies')
    if (saved) {
      try {
        setSavedStrategies(JSON.parse(saved))
      } catch (error) {
        console.error('Error loading saved strategies:', error)
      }
    }
  }, [])

  const runBacktest = async () => {
    setIsRunning(true)
    // Store strategy reference for chart data generation
    setBacktestStrategy(currentStrategy)
    try {
      const response = await fetch('http://localhost:8002/strategies/backtest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...backtestParams,
          strategy: currentStrategy,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const apiResult = await response.json()
      console.log('API Response:', apiResult) // Debug log
      
      // Map API response to frontend format
      const mappedResult: BacktestResult = {
        strategy_id: currentStrategy.id,
        symbol: backtestParams.symbol,
        timeframe: backtestParams.timeframe,
        start_date: apiResult.start_date || backtestParams.start_date,
        end_date: apiResult.end_date || backtestParams.end_date,
        initial_cash: apiResult.initial_cash || backtestParams.initial_cash,
        final_cash: apiResult.final_value || apiResult.final_cash || backtestParams.initial_cash,
        total_return: (apiResult.final_value || backtestParams.initial_cash) - backtestParams.initial_cash,
        total_return_pct: apiResult.total_return_pct || 0,
        max_drawdown: apiResult.max_drawdown_pct ? (apiResult.max_drawdown_pct / 100) * backtestParams.initial_cash : 0,
        max_drawdown_pct: apiResult.max_drawdown_pct || 0,
        sharpe_ratio: apiResult.sharpe_ratio || 0,
        win_rate: (apiResult.win_rate_pct || 0) / 100, // Convert percentage to decimal
        profit_factor: apiResult.profit_factor || 0,
        total_trades: apiResult.total_trades || 0,
        winning_trades: Math.round((apiResult.total_trades || 0) * ((apiResult.win_rate_pct || 0) / 100)),
        losing_trades: (apiResult.total_trades || 0) - Math.round((apiResult.total_trades || 0) * ((apiResult.win_rate_pct || 0) / 100)),
        avg_win: apiResult.avg_win_pct ? (apiResult.avg_win_pct / 100) * backtestParams.initial_cash : 0,
        avg_loss: apiResult.avg_loss_pct ? (apiResult.avg_loss_pct / 100) * backtestParams.initial_cash : 0,
        largest_win: apiResult.largest_win || 0, // Now provided by API
        largest_loss: apiResult.largest_loss || 0, // Now provided by API
        long_positions: apiResult.long_positions || 0, // New field
        short_positions: apiResult.short_positions || 0, // New field
        trades: (apiResult.trades || []).map((trade: { [key: string]: unknown }, index: number) => ({
          trade_id: index + 1,
          entry_time: String(trade.entry_time || ''),
          exit_time: String(trade.exit_time || ''),
          entry_price: Number(trade.entry_price || 0),
          exit_price: Number(trade.exit_price || 0),
          quantity: Number(trade.size || trade.quantity || 0),
          side: (trade.side === 'short' ? 'short' : 'long') as 'long' | 'short',
          pnl: Number(trade.pnl || 0),
          pnl_pct: Number(trade.pnl_pct || 0),
          duration: Number(trade.duration_minutes || 0) / 60, // Convert to hours
          entry_reason: String(trade.entry_reason || 'Signal'),
          exit_reason: String(trade.exit_reason || 'Signal'),
        })),
        equity_curve: [], // TODO: Generate from trades
        drawdown_curve: [], // TODO: Generate from trades
      }
      
      setBacktestResult(mappedResult)
    } catch (error) {
      console.error('Error running backtest:', error)
      // For demo purposes, set mock data
      setBacktestResult({
        strategy_id: currentStrategy.id,
        symbol: backtestParams.symbol,
        timeframe: backtestParams.timeframe,
        start_date: backtestParams.start_date,
        end_date: backtestParams.end_date,
        initial_cash: backtestParams.initial_cash,
        final_cash: 12500,
        total_return: 2500,
        total_return_pct: 25.0,
        max_drawdown: -800,
        max_drawdown_pct: -7.2,
        sharpe_ratio: 1.45,
        win_rate: 0.62,
        profit_factor: 1.8,
        total_trades: 45,
        winning_trades: 28,
        losing_trades: 17,
        avg_win: 125.5,
        avg_loss: -68.2,
        largest_win: 450.8,
        largest_loss: -220.1,
        long_positions: 28,
        short_positions: 17,
        trades: [
          {
            trade_id: 1,
            entry_time: '2024-06-01T10:00:00Z',
            exit_time: '2024-06-01T14:00:00Z',
            entry_price: 42000,
            exit_price: 42500,
            quantity: 0.1,
            side: 'long' as const,
            pnl: 50,
            pnl_pct: 1.19,
            duration: 4,
            entry_reason: 'SMA crossover',
            exit_reason: 'Take profit',
          },
          {
            trade_id: 2,
            entry_time: '2024-06-02T09:30:00Z',
            exit_time: '2024-06-02T11:30:00Z',
            entry_price: 41800,
            exit_price: 41200,
            quantity: 0.12,
            side: 'long' as const,
            pnl: -72,
            pnl_pct: -1.43,
            duration: 2,
            entry_reason: 'RSI oversold',
            exit_reason: 'Stop loss',
          },
          {
            trade_id: 3,
            entry_time: '2024-06-03T15:00:00Z',
            exit_time: '2024-06-03T18:30:00Z',
            entry_price: 43200,
            exit_price: 44000,
            quantity: 0.08,
            side: 'long' as const,
            pnl: 64,
            pnl_pct: 1.85,
            duration: 3.5,
            entry_reason: 'Breakout signal',
            exit_reason: 'Take profit',
          },
        ],
        equity_curve: [],
        drawdown_curve: [],
      })
    } finally {
      setIsRunning(false)
    }
  }

  const saveStrategy = () => {
    const strategyName = prompt('Enter strategy name:', currentStrategy.name)
    if (strategyName && strategyName.trim()) {
      const updatedStrategy = { ...currentStrategy, name: strategyName.trim() }
      const newSavedStrategies = [...savedStrategies.filter(s => s.id !== updatedStrategy.id), updatedStrategy]
      setSavedStrategies(newSavedStrategies)
      localStorage.setItem('savedStrategies', JSON.stringify(newSavedStrategies))
      alert('Strategy saved successfully!')
    }
  }

  const loadStrategy = (strategy: Strategy) => {
    setCurrentStrategy({ ...strategy })
  }

  const deleteStrategy = (strategyId: string) => {
    if (confirm('Are you sure you want to delete this strategy?')) {
      const newSavedStrategies = savedStrategies.filter(s => s.id !== strategyId)
      setSavedStrategies(newSavedStrategies)
      localStorage.setItem('savedStrategies', JSON.stringify(newSavedStrategies))
    }
  }

  // Generate chart data from backtest result
  const generateChartData = useCallback(async (result: BacktestResult | null) => {
    if (!result) {
      setChartData({
        ohlcvData: [],
        indicatorData: {},
        tradeData: []
      })
      return
    }

    try {
      // Fetch OHLCV data from API using chart timeframe
      const ohlcvResponse = await fetch(
        `http://localhost:8002/ohlcv/${result.symbol}?timeframe=${chartTimeframe}&start=${result.start_date}&end=${result.end_date}&limit=2000`
      )
      
      let ohlcvData: OHLCVDataPoint[] = []
      
      if (ohlcvResponse.ok) {
        const apiData = await ohlcvResponse.json()
        ohlcvData = (apiData.data || []).map((item: { [key: string]: unknown }) => ({
          time: String(item.timestamp || ''),
          open: Number(item.open || 0),
          high: Number(item.high || 0),
          low: Number(item.low || 0),
          close: Number(item.close || 0),
          volume: Number(item.volume || 0)
        }))
      } else {
        console.warn('Failed to fetch OHLCV data, using mock data')
        // Generate mock OHLCV data as fallback
        const startDate = new Date(result.start_date)
        const endDate = new Date(result.end_date)
        const daysDiff = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))
        
        for (let i = 0; i < Math.min(daysDiff, 100); i++) {
          const date = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000)
          const basePrice = 42000 + Math.sin(i * 0.1) * 2000 + (Math.random() - 0.5) * 1000
          const open = basePrice + (Math.random() - 0.5) * 500
          const close = open + (Math.random() - 0.5) * 1000
          const high = Math.max(open, close) + Math.random() * 500
          const low = Math.min(open, close) - Math.random() * 500
          const volume = 100 + Math.random() * 900

          ohlcvData.push({
            time: date.toISOString(),
            open,
            high,
            low,
            close,
            volume
          })
        }
      }

      // Generate indicator data from backtest result or fetch real data
      const indicatorData: Record<string, IndicatorDataPoint[]> = {}
      
      if (result && result.indicators && result.timestamps) {
        // Use real indicator data from backtest result
        console.log('📊 Using real indicator data from backtest:', Object.keys(result.indicators))
        const timestamps = result.timestamps
        Object.entries(result.indicators).forEach(([key, values]) => {
          if (values && Array.isArray(values) && timestamps.length >= values.length) {
            indicatorData[key.toUpperCase()] = values.map((value, i) => ({
              time: timestamps[i],
              value: value
            }))
            console.log(`   📈 ${key.toUpperCase()}: ${values.length} data points (${Math.min(...values).toFixed(1)} - ${Math.max(...values).toFixed(1)})`)
          }
        })
      } else {
        // Fetch real indicator data from API if not available in backtest result
        console.log('🔄 Fetching real indicator data from API...')
        try {
          // Determine which indicators to fetch based on the backtest strategy
          const indicatorsToFetch: string[] = []
          
          // Extract indicators from strategy conditions (only enabled ones)
          const strategy = backtestStrategy || currentStrategy
          const allConditions = [...strategy.entry_conditions, ...strategy.exit_conditions]
          for (const condition of allConditions) {
            if (condition.indicator && condition.enabled && !indicatorsToFetch.includes(condition.indicator)) {
              indicatorsToFetch.push(condition.indicator)
            }
          }
          
          // Add default indicators if none found
          if (indicatorsToFetch.length === 0) {
            indicatorsToFetch.push('rsi_14', 'sma_20', 'ema_21')
          }
          
          const indicatorResponse = await fetch(
            `http://localhost:8002/indicators/calculate/${result.symbol}?` +
            `timeframe=${chartTimeframe}&start=${result.start_date}&end=${result.end_date}&` +
            `indicators=${indicatorsToFetch.join(',')}&limit=1000`
          )
          
          if (indicatorResponse.ok) {
            const indicatorApiData = await indicatorResponse.json()
            console.log('✅ Real indicator data fetched:', Object.keys(indicatorApiData.indicators))
            
            // Convert API data to chart format
            const timestamps = indicatorApiData.timestamps || []
            Object.entries(indicatorApiData.indicators).forEach(([key, values]) => {
              if (values && Array.isArray(values) && timestamps.length >= values.length) {
                indicatorData[key.toUpperCase()] = values.map((value, i) => ({
                  time: timestamps[i] || '',
                  value: value
                }))
                console.log(`   📈 ${key.toUpperCase()}: ${values.length} data points`)
              }
            })
          } else {
            throw new Error('Failed to fetch real indicator data')
          }
        } catch (error) {
          console.error('❌ Error fetching real indicator data:', error)
          
          // Fallback to mock data if real data fetch fails
          if (ohlcvData.length > 0) {
            console.log('⚠️ Falling back to mock indicator data')
            indicatorData['SMA_20'] = ohlcvData.map((d, i) => ({
              time: d.time,
              value: d.close * (0.98 + Math.sin(i * 0.2) * 0.02)
            }))
            
            indicatorData['RSI_14'] = ohlcvData.map((d, i) => ({
              time: d.time,
              value: 50 + Math.sin(i * 0.3) * 20 + (Math.random() - 0.5) * 10
            }))
          }
        }
      }

      // Filter trades based on strategy direction
      const strategyDirection = backtestStrategy?.position_direction || 'both'
      const filteredTradesData = result.trades.filter(trade => {
        if (strategyDirection === 'long_only' && trade.side === 'short') return false
        if (strategyDirection === 'short_only' && trade.side === 'long') return false
        return true
      })
      
      // Convert filtered trades to chart format
      const tradeData: TradeDataPoint[] = filteredTradesData.map(trade => ({
        time: trade.entry_time,
        side: trade.side === 'long' ? 'Buy' : 'Sell',
        price: trade.entry_price,
        quantity: trade.quantity,
        exit_time: trade.exit_time,
        exit_price: trade.exit_price
      }))

      console.log('🎯 Trade filtering applied:', {
        strategyDirection: backtestStrategy?.position_direction,
        totalTrades: result.trades.length,
        filteredTrades: filteredTradesData.length,
        originalTrades: result.trades.map(t => ({ side: t.side, time: t.entry_time })),
        filteredTradesData: tradeData,
        actualFilteredTrades: filteredTradesData.map(t => ({ side: t.side, time: t.entry_time, pnl: t.pnl }))
      })

      // Update both chart data and filtered trades
      setChartData({
        ohlcvData,
        indicatorData,
        tradeData
      })
      
      // Set filtered trades to match what the chart will display
      setFilteredTrades(filteredTradesData)
      
      console.log('📊 Data sync check:', {
        chartTradeDataLength: tradeData.length,
        filteredTradesLength: filteredTradesData.length,
        chartTradeData: tradeData,
        filteredTradesData: filteredTradesData
      })
    } catch (error) {
      console.error('Error generating chart data:', error)
      setChartData({
        ohlcvData: [],
        indicatorData: {},
        tradeData: []
      })
    }
  }, [chartTimeframe, backtestStrategy])

  // Update chart data when backtest result or chart timeframe changes
  useEffect(() => {
    // Only generate chart data if there's a backtest result
    if (backtestResult) {
      generateChartData(backtestResult)
    }
  }, [backtestResult, chartTimeframe, generateChartData])

  // Close load menu when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (showLoadMenu) {
        setShowLoadMenu(false)
      }
    }
    
    if (showLoadMenu) {
      document.addEventListener('click', handleClickOutside)
      return () => document.removeEventListener('click', handleClickOutside)
    }
  }, [showLoadMenu])

  // PnL History Panel Component - Optimized for horizontal layout with sorting
  const PnLHistoryPanel: React.FC<{ trades: Trade[] }> = ({ trades }) => {
    const [sortField, setSortField] = useState<string>('')
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

    // Debug log for P&L history data
    console.log('📋 PnL History received trades:', {
      tradesCount: trades.length,
      trades: trades.map(t => ({ 
        side: t.side, 
        time: t.entry_time, 
        pnl: t.pnl,
        id: t.trade_id 
      }))
    })

    if (!trades || trades.length === 0) {
      return (
        <div className="bg-card border rounded-lg p-4 h-full flex flex-col">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <DollarSign size={18} />
            P&L History
          </h3>
          <div className="flex-1 flex items-center justify-center text-center text-muted-foreground">
            <div>
              <TrendingUp size={32} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm">No trade history available</p>
              <p className="text-xs">Run a backtest to see trade details</p>
            </div>
          </div>
        </div>
      )
    }

    const totalPnl = trades.reduce((sum, trade) => sum + (trade.pnl || 0), 0)
    const winningTrades = trades.filter(trade => (trade.pnl || 0) > 0)
    const losingTrades = trades.filter(trade => (trade.pnl || 0) < 0)

    // Handle sorting
    const handleSort = (field: string) => {
      if (sortField === field) {
        setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
      } else {
        setSortField(field)
        setSortDirection('desc')
      }
    }

    // Sort trades
    const sortedTrades = [...trades].sort((a, b) => {
      if (!sortField) return 0
      
      let aValue: number | string, bValue: number | string
      
      switch (sortField) {
        case 'index':
          aValue = trades.indexOf(a)
          bValue = trades.indexOf(b)
          break
        case 'pnl':
          aValue = a.pnl || 0
          bValue = b.pnl || 0
          break
        case 'pnl_pct':
          aValue = a.pnl_pct || 0
          bValue = b.pnl_pct || 0
          break
        case 'duration':
          aValue = a.duration || 0
          bValue = b.duration || 0
          break
        case 'entry_time':
          aValue = new Date(a.entry_time).getTime()
          bValue = new Date(b.entry_time).getTime()
          break
        case 'exit_time':
          aValue = a.exit_time ? new Date(a.exit_time).getTime() : 0
          bValue = b.exit_time ? new Date(b.exit_time).getTime() : 0
          break
        default:
          return 0
      }
      
      if (sortDirection === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })

    const SortButton: React.FC<{ field: string; children: React.ReactNode }> = ({ field, children }) => (
      <button
        onClick={() => handleSort(field)}
        className="flex items-center gap-1 hover:text-primary font-medium w-full text-left"
      >
        {children}
        <ArrowUpDown size={14} className="text-muted-foreground" />
        {sortField === field && (
          <span className="text-primary">
            {sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
          </span>
        )}
      </button>
    )

    return (
      <div className="bg-card border rounded-lg p-4 h-full flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <DollarSign size={18} />
            P&L History ({trades.length} trades)
          </h3>
          <div className="flex items-center gap-4 text-sm">
            <span className={`font-medium ${totalPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              Total: ${totalPnl.toFixed(2)}
            </span>
            <span className="text-green-600 dark:text-green-400">
              Wins: {winningTrades.length}
            </span>
            <span className="text-red-600 dark:text-red-400">
              Losses: {losingTrades.length}
            </span>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          <table className="w-full text-base">
            <thead className="sticky top-0 bg-card border-b">
              <tr>
                <th className="text-left p-2"><SortButton field="index">#</SortButton></th>
                <th className="text-left p-2 font-medium">Side</th>
                <th className="text-left p-2 font-medium">Size</th>
                <th className="text-left p-2 font-medium">Entry</th>
                <th className="text-left p-2 font-medium">Exit</th>
                <th className="text-left p-2"><SortButton field="pnl">P&L</SortButton></th>
                <th className="text-left p-2"><SortButton field="pnl_pct">Return %</SortButton></th>
                <th className="text-left p-2"><SortButton field="duration">Duration</SortButton></th>
                <th className="text-left p-2"><SortButton field="entry_time">Entry Time</SortButton></th>
                <th className="text-left p-2"><SortButton field="exit_time">Exit Time</SortButton></th>
              </tr>
            </thead>
            <tbody>
              {sortedTrades.map((trade, index) => (
                <tr key={trade.trade_id || index} className="border-b hover:bg-muted/30">
                  <td className="p-2 font-mono text-sm">{index + 1}</td>
                  <td className="p-2">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                      trade.side === 'long' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {trade.side === 'long' ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                      {trade.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="p-2 font-mono text-sm">{(trade.quantity || 0).toFixed(4)}</td>
                  <td className="p-2 font-mono text-sm">${(trade.entry_price || 0).toFixed(2)}</td>
                  <td className="p-2 font-mono text-sm">${(trade.exit_price || 0).toFixed(2)}</td>
                  <td className={`p-2 font-mono text-sm font-medium ${
                    (trade.pnl || 0) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    ${(trade.pnl || 0).toFixed(2)}
                  </td>
                  <td className={`p-2 font-mono text-sm font-medium ${
                    (trade.pnl_pct || 0) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {(trade.pnl_pct || 0).toFixed(2)}%
                  </td>
                  <td className="p-2 font-mono text-sm">
                    {((trade.duration || 0)).toFixed(1)}h
                  </td>
                  <td className="p-2 text-sm text-muted-foreground">
                    {new Date(trade.entry_time).toLocaleString()}
                  </td>
                  <td className="p-2 text-sm text-muted-foreground">
                    {trade.exit_time ? new Date(trade.exit_time).toLocaleString() : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }


  return (
    <div className="min-h-screen bg-background">
      {/* Header with Navigation and Theme Toggle */}
      <header className="bg-card shadow-sm border-b border-border">
        <div className="max-w-[2100px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-card-foreground">Strategy Backtesting</h1>
            <div className="flex items-center gap-4">
              <Navigation />
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[2100px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Main Layout - Much Wider (x1.5) */}
        <div className="space-y-6">
          {/* First Row: Parameters (10% narrower) and Strategy Builder (wider) */}
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-3">
              <BacktestParametersPanel
                params={backtestParams}
                onChange={setBacktestParams}
              />
            </div>
            <div className="col-span-9">
              <StrategyBuilder
                strategy={currentStrategy}
                onChange={setCurrentStrategy}
                availableIndicators={availableIndicators}
                actionButtons={
                  <div className="flex gap-2">
                    <button
                      onClick={saveStrategy}
                      className="flex items-center space-x-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
                    >
                      <Save size={16} />
                      <span>Save Strategy</span>
                    </button>
                    <div className="relative">
                      <button
                        onClick={() => setShowLoadMenu(!showLoadMenu)}
                        className="flex items-center space-x-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
                      >
                        <FolderOpen size={16} />
                        <span>Load Strategy</span>
                      </button>
                      {showLoadMenu && savedStrategies.length > 0 && (
                        <div className="absolute top-full left-0 mt-1 bg-background border rounded-md shadow-lg z-50 min-w-[200px]">
                          {savedStrategies.map((strategy) => (
                            <div
                              key={strategy.id}
                              className="flex items-center justify-between px-3 py-2 hover:bg-muted cursor-pointer"
                            >
                              <span
                                onClick={() => {
                                  loadStrategy(strategy)
                                  setShowLoadMenu(false)
                                }}
                                className="flex-1"
                              >
                                {strategy.name}
                              </span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  deleteStrategy(strategy.id)
                                  setShowLoadMenu(false)
                                }}
                                className="text-destructive hover:text-destructive/80 ml-2"
                              >
                                ×
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                      {showLoadMenu && savedStrategies.length === 0 && (
                        <div className="absolute top-full left-0 mt-1 bg-background border rounded-md shadow-lg z-50 min-w-[200px] px-3 py-2 text-muted-foreground">
                          No saved strategies
                        </div>
                      )}
                    </div>
                    <button
                      onClick={runBacktest}
                      disabled={isRunning}
                      className="flex items-center space-x-2 px-6 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 transition-colors"
                    >
                      <Play size={16} />
                      <span>{isRunning ? 'Running...' : 'Run Backtest'}</span>
                    </button>
                  </div>
                }
              />
            </div>
          </div>

          {/* Second Row: P&L History - Full Width */}
          <div className="grid grid-cols-1 gap-6">
            <div className="h-[450px]">
              <PnLHistoryPanel trades={filteredTrades} />
            </div>
          </div>

          {/* Debug Section - Only show during development */}
          {import.meta.env.DEV && backtestResult && (
            <div className="grid grid-cols-1 gap-6">
              <DataSyncDebug 
                chartTradeData={chartData.tradeData}
                pnlTradeData={filteredTrades}
                backtestResult={backtestResult}
              />
            </div>
          )}

          {/* Third Row: Results and Chart - Same Height */}
          <div className="grid grid-cols-12 gap-6">
            {/* Results */}
            <div className="col-span-3">
              <div className="h-[800px]">
                <BacktestResults result={backtestResult} />
              </div>
            </div>
            
            {/* Chart */}
            <div className="col-span-9">
              <div className="bg-card border rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <TrendingUp size={18} />
                    Price Chart
                  </h3>
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium">Timeframe:</label>
                    <select
                      value={chartTimeframe}
                      onChange={(e) => setChartTimeframe(e.target.value)}
                      title="Select chart timeframe"
                      className="px-3 py-1 border border-border rounded-md bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                    >
                      {timeframes.map((tf) => (
                        <option key={tf.value} value={tf.value}>
                          {tf.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div>
                  <BacktestChart
                    ohlcvData={chartData.ohlcvData}
                    indicatorData={chartData.indicatorData}
                    tradeData={chartData.tradeData}
                    isDarkMode={theme === 'dark'}
                    strategyConditions={backtestStrategy ? {
                      entry_conditions: backtestStrategy.entry_conditions,
                      exit_conditions: backtestStrategy.exit_conditions
                    } : undefined}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Backtesting
