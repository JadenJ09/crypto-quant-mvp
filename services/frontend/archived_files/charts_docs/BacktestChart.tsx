import React, { useEffect, useState, useRef, memo } from 'react'
import { BacktestResult } from '../../types/backtesting'
import { TrendingUp, TrendingDown, Eye, EyeOff } from 'lucide-react'
import { useTheme } from '../../hooks/useTheme'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
  Time,
  SeriesMarker,
  LineData,
  CandlestickData,
  ColorType,
  CandlestickSeries,
  LineSeries,
} from 'lightweight-charts'

interface BacktestChartProps {
  symbol: string
  timeframe: string
  startDate: string
  endDate: string
  result: BacktestResult | null
}

interface ChartDataPoint {
  time: UTCTimestamp
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface CandleData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

// Theme configurations - Match main page colors
const getThemeColors = () => {
  const root = document.documentElement
  const computedStyle = getComputedStyle(root)

  // Helper function to get CSS variable as HSL color
  const getCSSVariable = (property: string) => {
    const value = computedStyle.getPropertyValue(property).trim()
    return `hsl(${value})`
  }

  return {
    background: getCSSVariable('--card'),
    textColor: getCSSVariable('--card-foreground'),
    gridColor: getCSSVariable('--border'),
    upColor: getCSSVariable('--chart-up'),
    downColor: getCSSVariable('--chart-down'),
    wickUpColor: getCSSVariable('--chart-up'),
    wickDownColor: getCSSVariable('--chart-down'),
    buyMarkerColor: '#2196F3',
    sellMarkerColor: '#FF9800',
    equityColor: '#10b981',
    drawdownColor: '#ef4444',
  }
}

const BacktestChart: React.FC<BacktestChartProps> = memo(({
  symbol,
  timeframe,
  startDate,
  endDate,
  result
}) => {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([])
  const [activeChart, setActiveChart] = useState<'price' | 'equity' | 'drawdown'>('price')
  const [showSignals, setShowSignals] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const { theme } = useTheme()

  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const pnlChartRef = useRef<HTMLDivElement>(null)
  const pnlChartInstance = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const pnlSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)

  // Fetch price data
  useEffect(() => {
    const fetchPriceData = async () => {
      setIsLoading(true)
      try {
        const response = await fetch(
          `http://localhost:8002/ohlcv/${symbol}?timeframe=${timeframe}&start=${startDate}&end=${endDate}&limit=1000`
        )

        if (response.ok) {
          const data: CandleData[] = await response.json()

          const processedData: ChartDataPoint[] = data.map(candle => ({
            time: (new Date(candle.timestamp).getTime() / 1000) as UTCTimestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume
          }))

          setChartData(processedData)
        }
      } catch (error) {
        console.error('Error fetching price data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    if (symbol && timeframe && startDate && endDate) {
      fetchPriceData()
    }
  }, [symbol, timeframe, startDate, endDate])

  // Initialize main chart
  useEffect(() => {
    if (!chartContainerRef.current || chartData.length === 0) return

    const colors = getThemeColors()

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: colors.gridColor },
    })

    chartRef.current = chart

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: colors.upColor,
      downColor: colors.downColor,
      borderVisible: false,
      wickUpColor: colors.upColor,
      wickDownColor: colors.downColor,
    })

    candlestickSeriesRef.current = candlestickSeries
    candlestickSeries.setData(chartData.map(d => ({
      time: d.time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    })))

    // Add trade markers if available
    if (result?.trades && showSignals) {
      const markers: SeriesMarker<Time>[] = result.trades.map(trade => ({
        time: (new Date(trade.entry_time).getTime() / 1000) as UTCTimestamp,
        position: trade.side === 'long' ? 'belowBar' : 'aboveBar',
        color: trade.side === 'long' ? colors.buyMarkerColor : colors.sellMarkerColor,
        shape: trade.side === 'long' ? 'arrowUp' : 'arrowDown',
        text: `${trade.side.toUpperCase()} @ ${trade.entry_price.toFixed(2)}`,
      }))

      if ('setMarkers' in candlestickSeries) {
        (candlestickSeries as unknown as { setMarkers: (markers: SeriesMarker<Time>[]) => void }).setMarkers(markers)
      }
    }

    chart.timeScale().fitContent()

    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {
      const { width } = entries[0].contentRect
      chart.resize(width, 400)
    })

    resizeObserver.observe(chartContainerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
      candlestickSeriesRef.current = null
    }
  }, [chartData, theme, result, showSignals])

  // Initialize PnL chart
  useEffect(() => {
    if (!pnlChartRef.current || !result?.trades || activeChart === 'price') return

    const colors = getThemeColors()

    const chart = createChart(pnlChartRef.current, {
      width: pnlChartRef.current.clientWidth,
      height: 200,
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: colors.gridColor },
    })

    pnlChartInstance.current = chart

    const lineSeries = chart.addSeries(LineSeries, {
      color: activeChart === 'equity' ? colors.equityColor : colors.drawdownColor,
      lineWidth: 2,
    })

    pnlSeriesRef.current = lineSeries

    // Calculate equity or drawdown curve
    let cumulativePnl = 0
    let peak = 0
    const pnlData: LineData[] = result.trades.map(trade => {
      cumulativePnl += trade.pnl

      if (activeChart === 'equity') {
        return {
          time: (new Date(trade.entry_time).getTime() / 1000) as UTCTimestamp,
          value: cumulativePnl
        }
      } else {
        // Drawdown calculation
        peak = Math.max(peak, cumulativePnl)
        const drawdown = peak > 0 ? ((cumulativePnl - peak) / peak) * 100 : 0
        return {
          time: (new Date(trade.entry_time).getTime() / 1000) as UTCTimestamp,
          value: drawdown
        }
      }
    })

    lineSeries.setData(pnlData)
    chart.timeScale().fitContent()

    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {
      const { width } = entries[0].contentRect
      chart.resize(width, 200)
    })

    resizeObserver.observe(pnlChartRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      pnlChartInstance.current = null
      pnlSeriesRef.current = null
    }
  }, [result, activeChart, theme])

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center border rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading chart data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Chart Controls */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveChart('price')}
            className={`px-3 py-1 rounded text-sm ${activeChart === 'price'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300'
              }`}
          >
            <TrendingUp className="h-4 w-4 inline mr-1" />
            Price
          </button>
          {result && (
            <>
              <button
                onClick={() => setActiveChart('equity')}
                className={`px-3 py-1 rounded text-sm ${activeChart === 'equity'
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300'
                  }`}
              >
                <TrendingUp className="h-4 w-4 inline mr-1" />
                Equity
              </button>
              <button
                onClick={() => setActiveChart('drawdown')}
                className={`px-3 py-1 rounded text-sm ${activeChart === 'drawdown'
                    ? 'bg-red-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300'
                  }`}
              >
                <TrendingDown className="h-4 w-4 inline mr-1" />
                Drawdown
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-1 ml-4">
          <button
            onClick={() => setShowSignals(!showSignals)}
            className={`px-3 py-1 rounded text-sm ${showSignals
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300'
              }`}
          >
            {showSignals ? <Eye className="h-4 w-4 inline mr-1" /> : <EyeOff className="h-4 w-4 inline mr-1" />}
            Signals
          </button>
        </div>
      </div>

      {/* Main Chart */}
      <div
        ref={chartContainerRef}
        className="w-full h-96 border rounded-lg bg-card"
      />

      {/* PnL Chart */}
      {activeChart !== 'price' && (
        <div
          ref={pnlChartRef}
          className="w-full h-48 border rounded-lg bg-card"
        />
      )}
    </div>
  )
})

BacktestChart.displayName = 'BacktestChart'

export default BacktestChart
