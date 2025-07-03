import React, { useEffect, useRef, useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { 
  createChart, 
  IChartApi, 
  CandlestickData, 
  CandlestickSeries,
  HistogramSeries,
  HistogramData,
  LineSeries,
  LineData,
  Time, 
  ISeriesApi
} from 'lightweight-charts'
import ThemeToggle from './components/ThemeToggle'
import Navigation from './components/Navigation'
import Backtesting from './pages/Backtesting'
import { useTheme } from './hooks/useTheme'
import './App.css'

interface CandlestickApiData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  sma_20?: number
  sma_50?: number
  sma_100?: number
}

interface TimeframeInfo {
  label: string
  value: string
  table: string
  description: string
}

interface SymbolInfo {
  symbol: string
  name: string
  exchange: string
  base_currency: string
  quote_currency: string
}

// Main Dashboard Component (original chart functionality)
const Dashboard: React.FC = () => {
  const { theme } = useTheme()
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const sma20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const sma50SeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const sma100SeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const livePriceSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const refreshIntervalRef = useRef<number | null>(null)
  const legendRef = useRef<HTMLDivElement | null>(null)
  
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])
  const [timeframes, setTimeframes] = useState<TimeframeInfo[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string>(() => {
    return localStorage.getItem('selectedSymbol') || 'BTCUSDT'
  })
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>(() => {
    return localStorage.getItem('selectedTimeframe') || '1h'
  })
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState<Date>(new Date())

  // Get refresh interval based on timeframe
  const getRefreshInterval = (timeframe: string): number => {
    switch (timeframe) {
      case '1m': return 60000 // 1 minute
      case '5m': return 300000 // 5 minutes  
      case '15m': return 900000 // 15 minutes
      case '1h': return 3600000 // 1 hour
      case '4h': return 14400000 // 4 hours
      case '1d': return 86400000 // 1 day
      case '7d': return 604800000 // 7 days
      default: return 300000 // Default 5 minutes
    }
  }

  // Get human-readable refresh interval
  const getRefreshIntervalText = (timeframe: string): string => {
    const interval = getRefreshInterval(timeframe)
    const minutes = interval / 1000 / 60
    if (minutes < 60) {
      return `${minutes}m`
    } else if (minutes < 1440) {
      const hours = minutes / 60
      return `${hours}h`
    } else {
      const days = minutes / 1440
      return `${days}d`
    }
  }

  // Get short timeframe label for buttons
  const getShortTimeframeLabel = (value: string): string => {
    switch (value) {
      case '1m': return '1m'
      case '5m': return '5m'
      case '15m': return '15m'
      case '1h': return '1h'
      case '4h': return '4h'
      case '1d': return '1d'
      case '7d': return '7d'
      default: return value
    }
  }

  // Format current time with timezone in yyyy-mm-dd format (Local time)
  const formatCurrentTime = (date: Date): string => {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    const hours = String(date.getHours()).padStart(2, '0')
    const minutes = String(date.getMinutes()).padStart(2, '0')
    const seconds = String(date.getSeconds()).padStart(2, '0')
    
    const timezoneName = date.toLocaleString('en-US', { timeZoneName: 'short' }).split(' ').pop()
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} ${timezoneName}`
  }

  // Format UTC time in yyyy-mm-dd format
  const formatUTCTime = (date: Date): string => {
    const year = date.getUTCFullYear()
    const month = String(date.getUTCMonth() + 1).padStart(2, '0')
    const day = String(date.getUTCDate()).padStart(2, '0')
    const hours = String(date.getUTCHours()).padStart(2, '0')
    const minutes = String(date.getUTCMinutes()).padStart(2, '0')
    const seconds = String(date.getUTCSeconds()).padStart(2, '0')
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} UTC`
  }

  // Fetch available symbols and timeframes
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [symbolsResponse, timeframesResponse] = await Promise.all([
          fetch('http://localhost:8002/symbols'),
          fetch('http://localhost:8002/timeframes')
        ])

        if (symbolsResponse.ok) {
          const symbolsData = await symbolsResponse.json()
          setSymbols(symbolsData)
        }

        if (timeframesResponse.ok) {
          const timeframesData = await timeframesResponse.json()
          setTimeframes(timeframesData)
        }
      } catch (error) {
        console.error('Error fetching metadata:', error)
        // Set fallback data
        setSymbols([
          { symbol: 'BTCUSDT', name: 'BTC/USDT', exchange: 'Binance', base_currency: 'BTC', quote_currency: 'USDT' }
        ])
        setTimeframes([
          { label: '1 Hour', value: '1h', table: 'ohlcv_1hour', description: '1-hour candlesticks' }
        ])
      }
    }

    fetchMetadata()
  }, [])

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const container = chartContainerRef.current

    // Get computed CSS custom properties for theme colors
    const root = document.documentElement
    const computedStyle = getComputedStyle(root)
    
    // Helper function to get CSS variable as HSL color
    const getCSSVariable = (property: string) => {
      const value = computedStyle.getPropertyValue(property).trim()
      return `hsl(${value})`
    }
    
    const backgroundColor = getCSSVariable('--card')
    const textColor = getCSSVariable('--card-foreground')
    const gridColor = getCSSVariable('--border')
    const borderColor = getCSSVariable('--border')
    
    // Chart up/down colors from CSS variables
    const upColor = getCSSVariable('--chart-up')
    const downColor = getCSSVariable('--chart-down')

    // Create chart
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 500,
      layout: {
        background: { color: backgroundColor },
        textColor: textColor,
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: borderColor,
        rightOffset: 20,
      },
      rightPriceScale: {
        borderColor: borderColor,
        scaleMargins: {
          top: 0.15,
          bottom: 0.4,
        },
      },
    })

    // Create SMA line series only for timeframes that support SMA (not 1m)
    const shouldShowSMA = selectedTimeframe !== '1m'
    
    let sma20Series: ISeriesApi<'Line'> | null = null
    let sma50Series: ISeriesApi<'Line'> | null = null
    let sma100Series: ISeriesApi<'Line'> | null = null
    let livePriceSeries: ISeriesApi<'Line'> | null = null
    
    if (shouldShowSMA) {
      sma20Series = chart.addSeries(LineSeries, {
        color: 'rgba(255, 215, 0, 0.6)',
        lineWidth: 2,
        title: 'SMA 20',
      })

      sma50Series = chart.addSeries(LineSeries, {
        color: 'rgba(0, 206, 209, 0.6)',
        lineWidth: 2,
        title: 'SMA 50',
      })

      sma100Series = chart.addSeries(LineSeries, {
        color: 'rgba(255, 20, 147, 0.6)',
        lineWidth: 2,
        title: 'SMA 100',
      })
    }

    // Create live price line for higher timeframes
    const shouldShowLivePrice = selectedTimeframe !== '1m'
    if (shouldShowLivePrice) {
      livePriceSeries = chart.addSeries(LineSeries, {
        color: 'rgba(255, 255, 0, 0.8)',
        lineWidth: 1,
        lineStyle: 2,
        title: 'Live Price',
        lastValueVisible: true,
        priceLineVisible: true,
      })
    }

    // Create candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: upColor,
      downColor: downColor,
      borderDownColor: downColor,
      borderUpColor: upColor,
      wickDownColor: downColor,
      wickUpColor: upColor,
    })

    candlestickSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.15,
        bottom: 0.4,
      },
    })

    // Create volume histogram series
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(38, 166, 154, 0.6)',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    })

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.7,
        bottom: 0,
      },
    })

    chartRef.current = chart
    seriesRef.current = candlestickSeries
    volumeSeriesRef.current = volumeSeries
    sma20SeriesRef.current = sma20Series
    sma50SeriesRef.current = sma50Series
    sma100SeriesRef.current = sma100Series
    livePriceSeriesRef.current = livePriceSeries

    // Create legend
    const legend = document.createElement('div')
    
    const setLegendTheme = () => {
      const isDark = theme === 'dark'
      const cardColor = computedStyle.getPropertyValue('--card').trim()
      const legendBg = isDark 
        ? `hsla(${cardColor}, 0.75)`
        : 'rgba(255, 255, 255, 0.75)'
      const legendTextColor = isDark ? 'white' : 'black'
      const legendBorder = isDark 
        ? 'rgba(255, 255, 255, 0.08)' 
        : 'rgba(0, 0, 0, 0.08)'
      
      legend.style.cssText = `
        position: absolute; 
        left: 12px; 
        top: 12px; 
        z-index: 100; 
        font-size: 11px; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
        font-weight: 500;
        background: ${legendBg};
        color: ${legendTextColor};
        padding: 6px 10px;
        border-radius: 6px;
        pointer-events: none;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid ${legendBorder};
        backdrop-filter: blur(4px);
      `
    }
    
    setLegendTheme()
    
    container.style.position = 'relative'
    container.appendChild(legend)
    legendRef.current = legend

    const legendContent = shouldShowSMA 
      ? `Price: <span style="color: #ffd700;">--</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">--</span> | SMA20: <span style="color: rgba(255, 215, 0, 0.8);">--</span> | SMA50: <span style="color: rgba(0, 206, 209, 0.8);">--</span> | SMA100: <span style="color: rgba(255, 20, 147, 0.8);">--</span>`
      : `Price: <span style="color: #ffd700;">--</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">--</span>`
    legend.innerHTML = legendContent

    // Subscribe to crosshair move events
    chart.subscribeCrosshairMove(param => {
      let priceFormatted = '--'
      let volumeFormatted = '--'
      let sma20Formatted = '--'
      let sma50Formatted = '--'
      let sma100Formatted = '--'

      if (param.time) {
        const priceData = param.seriesData.get(candlestickSeries)
        if (priceData) {
          const candleData = priceData as CandlestickData
          if (candleData.close !== undefined) {
            priceFormatted = `$${candleData.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
          }
        }

        const volumeData = param.seriesData.get(volumeSeries)
        if (volumeData) {
          const histogramData = volumeData as HistogramData
          if (histogramData.value !== undefined) {
            volumeFormatted = histogramData.value.toLocaleString(undefined, { maximumFractionDigits: 0 })
          }
        }

        if (shouldShowSMA && sma20Series) {
          const sma20Data = param.seriesData.get(sma20Series)
          if (sma20Data) {
            const lineData = sma20Data as LineData
            if (lineData.value !== undefined) {
              sma20Formatted = `$${lineData.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
            }
          }
        }

        if (shouldShowSMA && sma50Series) {
          const sma50Data = param.seriesData.get(sma50Series)
          if (sma50Data) {
            const lineData = sma50Data as LineData
            if (lineData.value !== undefined) {
              sma50Formatted = `$${lineData.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
            }
          }
        }

        if (shouldShowSMA && sma100Series) {
          const sma100Data = param.seriesData.get(sma100Series)
          if (sma100Data) {
            const lineData = sma100Data as LineData
            if (lineData.value !== undefined) {
              sma100Formatted = `$${lineData.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
            }
          }
        }
      }

      const newLegendContent = shouldShowSMA 
        ? `Price: <span style="color: #ffd700;">${priceFormatted}</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">${volumeFormatted}</span> | SMA20: <span style="color: rgba(255, 215, 0, 0.8);">${sma20Formatted}</span> | SMA50: <span style="color: rgba(0, 206, 209, 0.8);">${sma50Formatted}</span> | SMA100: <span style="color: rgba(255, 20, 147, 0.8);">${sma100Formatted}</span>`
        : `Price: <span style="color: #ffd700;">${priceFormatted}</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">${volumeFormatted}</span>`
      legend.innerHTML = newLegendContent
    })

    const handleResize = () => {
      if (container && chart) {
        chart.applyOptions({
          width: container.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (legendRef.current && container.contains(legendRef.current)) {
        container.removeChild(legendRef.current)
        legendRef.current = null
      }
      chart.remove()
    }
  }, [selectedSymbol, symbols, theme, selectedTimeframe])

  // Fetch chart data
  useEffect(() => {
    const fetchChartData = async () => {
      if (!seriesRef.current || !volumeSeriesRef.current) return

      setLoading(true)
      setError(null)

      try {
        const response = await fetch(
          `http://localhost:8002/candlesticks/${selectedSymbol}?timeframe=${selectedTimeframe}`
        )

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: CandlestickApiData[] = await response.json()
        
        const chartData: CandlestickData[] = data.map(item => ({
          time: (new Date(item.time).getTime() / 1000) as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        }))

        const volumeData: HistogramData[] = data.map((item) => {
          const isUp = item.close >= item.open
          return {
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.volume,
            color: isUp ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)',
          }
        })

        chartData.sort((a, b) => (a.time as number) - (b.time as number))
        volumeData.sort((a, b) => (a.time as number) - (b.time as number))

        seriesRef.current.setData(chartData)
        volumeSeriesRef.current.setData(volumeData)

        // Set SMA data if available
        if (sma20SeriesRef.current) {
          const sma20Data: LineData[] = data
            .filter(item => item.sma_20 !== null && item.sma_20 !== undefined)
            .map(item => ({
              time: (new Date(item.time).getTime() / 1000) as Time,
              value: item.sma_20!,
            }))
          sma20Data.sort((a, b) => (a.time as number) - (b.time as number))
          sma20SeriesRef.current.setData(sma20Data)
        }

        if (sma50SeriesRef.current) {
          const sma50Data: LineData[] = data
            .filter(item => item.sma_50 !== null && item.sma_50 !== undefined)
            .map(item => ({
              time: (new Date(item.time).getTime() / 1000) as Time,
              value: item.sma_50!,
            }))
          sma50Data.sort((a, b) => (a.time as number) - (b.time as number))
          sma50SeriesRef.current.setData(sma50Data)
        }

        if (sma100SeriesRef.current) {
          const sma100Data: LineData[] = data
            .filter(item => item.sma_100 !== null && item.sma_100 !== undefined)
            .map(item => ({
              time: (new Date(item.time).getTime() / 1000) as Time,
              value: item.sma_100!,
            }))
          sma100Data.sort((a, b) => (a.time as number) - (b.time as number))
          sma100SeriesRef.current.setData(sma100Data)
        }
        
      } catch (error) {
        console.error('Error fetching chart data:', error)
        setError(`Failed to load chart data for ${selectedSymbol} (${selectedTimeframe}). Please try again.`)
      } finally {
        setLoading(false)
      }
    }

    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
      refreshIntervalRef.current = null
    }

    fetchChartData()

    const refreshInterval = getRefreshInterval(selectedTimeframe)
    refreshIntervalRef.current = window.setInterval(() => {
      fetchChartData()
    }, refreshInterval)

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [selectedSymbol, selectedTimeframe])

  // Update current time
  useEffect(() => {
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timeInterval)
  }, [])

  const scrollToRealtime = () => {
    if (chartRef.current) {
      chartRef.current.timeScale().scrollToRealTime()
    }
  }

  const handleSymbolChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newSymbol = event.target.value
    setSelectedSymbol(newSymbol)
    localStorage.setItem('selectedSymbol', newSymbol)
  }

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
    localStorage.setItem('selectedTimeframe', timeframe)
  }

  const handleManualRefresh = () => {
    setCurrentTime(new Date())
    setTimeout(() => {
      scrollToRealtime()
    }, 150)
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="bg-card shadow-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-card-foreground">
              Crypto Quant MVP
            </h1>
            <div className="flex items-center gap-4">
              <Navigation />
              <div className="text-sm text-muted-foreground">
                Real-time cryptocurrency analytics
              </div>
              <div className="text-sm font-mono text-card-foreground bg-secondary px-3 py-1 rounded">
                {formatCurrentTime(currentTime)}
              </div>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Chart Controls */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <div className="flex items-center gap-2">
                <label htmlFor="symbol-select" className="text-sm font-medium text-card-foreground">
                  Symbol:
                </label>
                <select
                  id="symbol-select"
                  value={selectedSymbol}
                  onChange={handleSymbolChange}
                  className="border border-border rounded px-3 py-1 text-sm bg-background text-foreground focus:ring-2 focus:ring-ring focus:border-ring"
                >
                  {symbols.map((symbol) => (
                    <option key={symbol.symbol} value={symbol.symbol}>
                      {symbol.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-card-foreground">Timeframe:</span>
                <div className="flex gap-1">
                  {timeframes.map((tf) => (
                    <button
                      key={tf.value}
                      onClick={() => handleTimeframeChange(tf.value)}
                      className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                        selectedTimeframe === tf.value
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-secondary text-secondary-foreground hover:bg-accent hover:text-accent-foreground'
                      }`}
                    >
                      {getShortTimeframeLabel(tf.value)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs bg-accent px-2 py-1 rounded">LIVE</span>
              </div>
              <span>{formatUTCTime(currentTime)}</span>
            </div>
          </div>
        </div>

        {/* Chart Container */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-card-foreground">
              {symbols.find(s => s.symbol === selectedSymbol)?.name || selectedSymbol} Price Chart
            </h2>
            <div className="flex items-center gap-4">
              {loading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                  Loading...
                </div>
              )}
              <div className="text-xs text-muted-foreground">
                Auto-refresh: {getRefreshIntervalText(selectedTimeframe)}
              </div>
              <button
                onClick={handleManualRefresh}
                disabled={loading}
                className="px-3 py-1 text-xs bg-secondary text-secondary-foreground hover:bg-accent hover:text-accent-foreground rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                🔄 Refresh
              </button>
              <button
                onClick={scrollToRealtime}
                className="px-3 py-1 text-xs bg-primary text-primary-foreground hover:bg-primary/90 rounded transition-colors"
              >
                📈 Go to Realtime
              </button>
            </div>
          </div>
          
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div 
            ref={chartContainerRef}
            className="w-full border border-border rounded chart-container"
          />
        </div>

        {/* Stats Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">Market Cap</h3>
            <p className="text-3xl font-bold text-chart-up">$1.2T</p>
            <p className="text-sm text-muted-foreground mt-1">+2.3% from yesterday</p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">24h Volume</h3>
            <p className="text-3xl font-bold text-primary">$45.8B</p>
            <p className="text-sm text-muted-foreground mt-1">+15.7% from yesterday</p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">Active Strategies</h3>
            <p className="text-3xl font-bold text-accent">12</p>
            <p className="text-sm text-muted-foreground mt-1">3 running, 9 backtesting</p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">Timeframes</h3>
            <p className="text-3xl font-bold text-secondary-foreground">{timeframes.length}</p>
            <p className="text-sm text-muted-foreground mt-1">1min to 7days coverage</p>
          </div>
        </div>

        {/* Backtesting Section Placeholder */}
        <div className="mt-8 bg-card rounded-lg shadow-sm border border-border p-6">
          <h2 className="text-xl font-semibold text-card-foreground mb-4">Backtesting & Technical Analysis</h2>
          <p className="text-muted-foreground">
            Advanced backtesting features and technical indicators are now available in the{' '}
            <button
              onClick={() => window.location.href = '/backtesting'}
              className="text-primary hover:text-primary/80 underline"
            >
              backtesting page
            </button>
            . This includes strategy building, performance analysis, and comprehensive results visualization.
          </p>
        </div>
      </main>
    </div>
  )
}

// Main App component with routing
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/backtesting" element={<Backtesting />} />
      </Routes>
    </Router>
  )
}

export default App