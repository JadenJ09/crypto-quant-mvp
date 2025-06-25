import React, { useEffect, useRef, useState } from 'react'
import { 
  createChart, 
  IChartApi, 
  CandlestickData, 
  CandlestickSeries,
  Time, 
  ISeriesApi
} from 'lightweight-charts'
import ThemeToggle from './components/ThemeToggle'
import { useTheme } from './hooks/useTheme'
import './App.css'

interface CandlestickApiData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
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

function App() {
  const { theme } = useTheme()
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const refreshIntervalRef = useRef<number | null>(null)
  
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
      return `${minutes}M`
    } else if (minutes < 1440) {
      const hours = minutes / 60
      return `${hours}H`
    } else {
      const days = minutes / 1440
      return `${days}D`
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

  // Function to focus chart on latest 200 candles with some space at the end
  const focusOnLatestCandles = (chartData: CandlestickData[]) => {
    if (!chartRef.current || chartData.length === 0) return

    const dataLength = chartData.length
    const startIndex = Math.max(0, dataLength - 200)
    const endIndex = dataLength - 1
    
    if (startIndex < endIndex) {
      const startTime = chartData[startIndex].time
      const endTime = chartData[endIndex].time
      
      // Add some space (about 10% of the visible range) after the last candle
      const timeRange = (endTime as number) - (startTime as number)
      const extraSpace = timeRange * 0.1
      const adjustedEndTime = (endTime as number) + extraSpace
      
      chartRef.current.timeScale().setVisibleRange({
        from: startTime,
        to: adjustedEndTime as Time
      })
    }
  }

  // Fetch available symbols and timeframes
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [symbolsResponse, timeframesResponse] = await Promise.all([
          fetch('http://localhost:8000/symbols'),
          fetch('http://localhost:8000/timeframes')
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
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
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
      },
      rightPriceScale: {
        borderColor: borderColor,
      },
    })

    // Create candlestick series using the proper API method
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: upColor,
      downColor: downColor,
      borderDownColor: downColor,
      borderUpColor: upColor,
      wickDownColor: downColor,
      wickUpColor: upColor,
    })

    chartRef.current = chart
    seriesRef.current = candlestickSeries

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update chart on theme change
  useEffect(() => {
    if (!chartRef.current || !seriesRef.current) return

    const timer = setTimeout(() => {
      console.log('Updating chart theme to:', theme)
      const root = document.documentElement
      const computedStyle = getComputedStyle(root)
      
      const getCSSVariable = (property: string) => {
        const value = computedStyle.getPropertyValue(property).trim()
        return `hsl(${value})`
      }
      
      const backgroundColor = getCSSVariable('--card')
      const textColor = getCSSVariable('--card-foreground')
      const gridColor = getCSSVariable('--border')
      const borderColor = getCSSVariable('--border')
      const upColor = getCSSVariable('--chart-up')
      const downColor = getCSSVariable('--chart-down')

      chartRef.current?.applyOptions({
        layout: {
          background: { color: backgroundColor },
          textColor: textColor,
        },
        grid: {
          vertLines: { color: gridColor },
          horzLines: { color: gridColor },
        },
        timeScale: {
          borderColor: borderColor,
        },
        rightPriceScale: {
          borderColor: borderColor,
        },
      })

      seriesRef.current?.applyOptions({
        upColor: upColor,
        downColor: downColor,
        borderDownColor: downColor,
        borderUpColor: upColor,
        wickDownColor: downColor,
        wickUpColor: upColor,
      })
    }, 100)

    return () => clearTimeout(timer)
  }, [theme])

  // Fetch chart data when symbol, timeframe changes, or chart is recreated
  useEffect(() => {
    const fetchChartData = async () => {
      if (!seriesRef.current) return

      setLoading(true)
      setError(null)

      try {
        console.log(`Fetching data for ${selectedSymbol} with timeframe ${selectedTimeframe}`)
        const response = await fetch(
          `http://localhost:8000/candlesticks/${selectedSymbol}?timeframe=${selectedTimeframe}`
        )

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: CandlestickApiData[] = await response.json()
        console.log(`Received ${data.length} data points`)
        
        // Convert data to lightweight-charts format (keep as UTC)
        const chartData: CandlestickData[] = data.map(item => ({
          time: (new Date(item.time).getTime() / 1000) as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        }))

        // Sort by time to ensure proper order (data should already be sorted from API)
        chartData.sort((a, b) => (a.time as number) - (b.time as number))

        // Set data using the proper API method
        seriesRef.current.setData(chartData)
        
        // Focus on latest 200 candles with some space at the end
        setTimeout(() => {
          focusOnLatestCandles(chartData)
        }, 100)
        
      } catch (error) {
        console.error('Error fetching chart data:', error)
        setError(`Failed to load chart data for ${selectedSymbol} (${selectedTimeframe}). Please try again.`)
        
        // Fallback sample data
        const sampleData: CandlestickData[] = [
          { time: '2023-12-01' as Time, open: 42000, high: 42500, low: 41800, close: 42200 },
          { time: '2023-12-02' as Time, open: 42200, high: 43000, low: 42000, close: 42800 },
          { time: '2023-12-03' as Time, open: 42800, high: 43200, low: 42600, close: 43000 },
        ]
        seriesRef.current.setData(sampleData)
      } finally {
        setLoading(false)
      }
    }

    // Clear existing refresh interval
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
      refreshIntervalRef.current = null
    }

    // Initial data fetch
    fetchChartData()

    // Set up auto-refresh based on timeframe
    const refreshInterval = getRefreshInterval(selectedTimeframe)
    refreshIntervalRef.current = window.setInterval(() => {
      console.log(`Auto-refreshing chart data for ${selectedSymbol} (${selectedTimeframe})`)
      fetchChartData()
    }, refreshInterval)

    // Cleanup function
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [selectedSymbol, selectedTimeframe])

  const handleSymbolChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newSymbol = event.target.value
    setSelectedSymbol(newSymbol)
    localStorage.setItem('selectedSymbol', newSymbol)
  }

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
    localStorage.setItem('selectedTimeframe', timeframe)
  }

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [])

  // Update current time every second
  useEffect(() => {
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timeInterval)
  }, [])

  // Manual refresh function
  const handleManualRefresh = async () => {
    if (!seriesRef.current) return

    setLoading(true)
    setError(null)

    try {
      console.log(`Manual refresh for ${selectedSymbol} with timeframe ${selectedTimeframe}`)
      const response = await fetch(
        `http://localhost:8000/candlesticks/${selectedSymbol}?timeframe=${selectedTimeframe}`
      )

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: CandlestickApiData[] = await response.json()
      console.log(`Received ${data.length} data points`)
      
      // Convert data to lightweight-charts format (keep as UTC)
      const chartData: CandlestickData[] = data.map(item => ({
        time: (new Date(item.time).getTime() / 1000) as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))

      // Sort by time to ensure proper order
      chartData.sort((a, b) => (a.time as number) - (b.time as number))

      // Set data using the proper API method
      seriesRef.current.setData(chartData)
      
      // Focus on latest 200 candles with some space at the end
      setTimeout(() => {
        focusOnLatestCandles(chartData)
      }, 100)
      
    } catch (error) {
      console.error('Error fetching chart data:', error)
      setError(`Failed to load chart data for ${selectedSymbol} (${selectedTimeframe}). Please try again.`)
    } finally {
      setLoading(false)
    }
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
              {/* Symbol Selector */}
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

              {/* Timeframe Buttons */}
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
            
            {/* UTC Time Display for Chart with Live Indicator */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs bg-accent px-2 py-1 rounded">
                  LIVE
                </span>
              </div>
              <span> {formatUTCTime(currentTime)}</span>
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
            <h3 className="text-lg font-medium text-card-foreground mb-2">
              Market Cap
            </h3>
            <p className="text-3xl font-bold text-chart-up">
              $1.2T
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              +2.3% from yesterday
            </p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">
              24h Volume
            </h3>
            <p className="text-3xl font-bold text-primary">
              $45.8B
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              +15.7% from yesterday
            </p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">
              Active Strategies
            </h3>
            <p className="text-3xl font-bold text-accent">
              12
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              3 running, 9 backtesting
            </p>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-medium text-card-foreground mb-2">
              Timeframes
            </h3>
            <p className="text-3xl font-bold text-secondary-foreground">
              {timeframes.length}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              1min to 7days coverage
            </p>
          </div>
        </div>

        {/* Backtesting Section Placeholder */}
        <div className="mt-8 bg-card rounded-lg shadow-sm border border-border p-6">
          <h2 className="text-xl font-semibold text-card-foreground mb-4">
            Backtesting & Technical Analysis
          </h2>
          <p className="text-muted-foreground">
            Advanced backtesting features and technical indicators will be implemented here.
            This will include RSI, MACD, and other technical analysis tools using the multi-timeframe data.
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
