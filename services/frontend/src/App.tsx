import React, { useEffect, useRef, useState } from 'react'
import { 
  createChart, 
  IChartApi, 
  CandlestickData, 
  CandlestickSeries,
  Time, 
  ISeriesApi
} from 'lightweight-charts'
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
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])
  const [timeframes, setTimeframes] = useState<TimeframeInfo[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT')
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

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

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#d1d4dc',
      },
      rightPriceScale: {
        borderColor: '#d1d4dc',
      },
    })

    // Create candlestick series using the proper API method
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e', // green
      downColor: '#ef4444', // red
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
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

  // Fetch chart data when symbol or timeframe changes
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
        
        // Convert data to lightweight-charts format
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
        
        // Fit content to show all data
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent()
        }
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

    fetchChartData()
  }, [selectedSymbol, selectedTimeframe])

  const handleSymbolChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedSymbol(event.target.value)
  }

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              Crypto Quant MVP
            </h1>
            <div className="text-sm text-gray-500">
              Real-time cryptocurrency analytics
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Chart Controls */}
        <div className="bg-white rounded-lg shadow-sm border p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
            {/* Symbol Selector */}
            <div className="flex items-center gap-2">
              <label htmlFor="symbol-select" className="text-sm font-medium text-gray-700">
                Symbol:
              </label>
              <select
                id="symbol-select"
                value={selectedSymbol}
                onChange={handleSymbolChange}
                className="border border-gray-300 rounded px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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
              <span className="text-sm font-medium text-gray-700">Timeframe:</span>
              <div className="flex gap-1">
                {timeframes.map((tf) => (
                  <button
                    key={tf.value}
                    onClick={() => handleTimeframeChange(tf.value)}
                    className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                      selectedTimeframe === tf.value
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Chart Container */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              {symbols.find(s => s.symbol === selectedSymbol)?.name || selectedSymbol} Price Chart
            </h2>
            {loading && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                Loading...
              </div>
            )}
          </div>
          
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div 
            ref={chartContainerRef}
            className="w-full border rounded chart-container"
          />
        </div>

        {/* Stats Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Market Cap
            </h3>
            <p className="text-3xl font-bold text-green-600">
              $1.2T
            </p>
            <p className="text-sm text-gray-500 mt-1">
              +2.3% from yesterday
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              24h Volume
            </h3>
            <p className="text-3xl font-bold text-blue-600">
              $45.8B
            </p>
            <p className="text-sm text-gray-500 mt-1">
              +15.7% from yesterday
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Active Strategies
            </h3>
            <p className="text-3xl font-bold text-purple-600">
              12
            </p>
            <p className="text-sm text-gray-500 mt-1">
              3 running, 9 backtesting
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Timeframes
            </h3>
            <p className="text-3xl font-bold text-orange-600">
              {timeframes.length}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              1min to 7days coverage
            </p>
          </div>
        </div>

        {/* Backtesting Section Placeholder */}
        <div className="mt-8 bg-white rounded-lg shadow-sm border p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Backtesting & Technical Analysis
          </h2>
          <p className="text-gray-600">
            Advanced backtesting features and technical indicators will be implemented here.
            This will include RSI, MACD, and other technical analysis tools using the multi-timeframe data.
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
