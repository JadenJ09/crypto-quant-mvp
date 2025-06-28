import React, { useEffect, useRef, useState } from 'react'
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

function App() {
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
  const realtimeIntervalRef = useRef<number | null>(null)
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

  // Get real-time update interval based on timeframe
  const getRealtimeInterval = (timeframe: string): number => {
    switch (timeframe) {
      case '1m': return 5000 // Update every 5 seconds for 1-minute charts
      case '5m': return 15000 // Update every 15 seconds for 5-minute charts
      case '15m': return 30000 // Update every 30 seconds for 15-minute charts
      case '1h': return 60000 // Update every minute for hourly charts
      case '4h': return 240000 // Update every 4 minutes for 4-hour charts
      case '1d': return 900000 // Update every 15 minutes for daily charts
      case '7d': return 3600000 // Update every hour for weekly charts
      default: return 30000 // Default 30 seconds
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
        rightOffset: 20, // Add space for labels after the latest candle
      },
      rightPriceScale: {
        borderColor: borderColor,
        scaleMargins: {
          top: 0.15, // leave more space for the legend
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
        color: 'rgba(255, 215, 0, 0.6)', // Yellowish gold with transparency
        lineWidth: 2,
        title: 'SMA 20',
      })

      sma50Series = chart.addSeries(LineSeries, {
        color: 'rgba(0, 206, 209, 0.6)', // Cyanish dark turquoise with transparency
        lineWidth: 2,
        title: 'SMA 50',
      })

      sma100Series = chart.addSeries(LineSeries, {
        color: 'rgba(255, 20, 147, 0.6)', // Magentaish deep pink with transparency
        lineWidth: 2,
        title: 'SMA 100',
      })
    }

    // Create live price line for higher timeframes (not 1m) to show real-time 1m price updates
    const shouldShowLivePrice = selectedTimeframe !== '1m'
    if (shouldShowLivePrice) {
      livePriceSeries = chart.addSeries(LineSeries, {
        color: 'rgba(255, 255, 0, 0.8)', // Bright yellow for live price
        lineWidth: 1,
        lineStyle: 2, // Dotted line style
        title: 'Live Price',
        lastValueVisible: true,
        priceLineVisible: true,
      })
    }

    // Create candlestick series using the proper API method
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: upColor,
      downColor: downColor,
      borderDownColor: downColor,
      borderUpColor: upColor,
      wickDownColor: downColor,
      wickUpColor: upColor,
    })

    // Position the main price series in the top 60% of the chart
    candlestickSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.15, // highest point of the series will be 15% away from the top (leaving space for legend)
        bottom: 0.4, // lowest point will be 40% away from the bottom
      },
    })

    // Create volume histogram series as an overlay
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(38, 166, 154, 0.6)', // Default color for volume bars with less transparency
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // set as an overlay by setting a blank priceScaleId
    })

    // Position the volume series in the bottom 30% of the chart
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.7, // highest point of the series will be 70% away from the top
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
    
    // Set initial legend styles based on current theme
    const setLegendTheme = () => {
      const isDark = theme === 'dark'
      // Get the exact card background color from CSS variables
      const cardColor = computedStyle.getPropertyValue('--card').trim()
      const legendBg = isDark 
        ? `hsla(${cardColor}, 0.75)` // More transparent for dark mode
        : 'rgba(255, 255, 255, 0.75)' // More transparent for light mode
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
    
    // Ensure container is positioned for absolute positioning
    container.style.position = 'relative'
    container.appendChild(legend)
    legendRef.current = legend

    // Initialize legend without symbol name (conditional SMA display)
    const legendContent = shouldShowSMA 
      ? `Price: <span style="color: #ffd700;">--</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">--</span> | SMA20: <span style="color: rgba(255, 215, 0, 0.8);">--</span> | SMA50: <span style="color: rgba(0, 206, 209, 0.8);">--</span> | SMA100: <span style="color: rgba(255, 20, 147, 0.8);">--</span>`
      : `Price: <span style="color: #ffd700;">--</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">--</span>`
    legend.innerHTML = legendContent

    // Subscribe to crosshair move events to update legend
    chart.subscribeCrosshairMove(param => {
      let priceFormatted = '--'
      let volumeFormatted = '--'
      let sma20Formatted = '--'
      let sma50Formatted = '--'
      let sma100Formatted = '--'

      if (param.time) {
        // Get price data
        const priceData = param.seriesData.get(candlestickSeries)
        if (priceData) {
          const candleData = priceData as CandlestickData
          if (candleData.close !== undefined) {
            priceFormatted = `$${candleData.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
          }
        }

        // Get volume data
        const volumeData = param.seriesData.get(volumeSeries)
        if (volumeData) {
          const histogramData = volumeData as HistogramData
          if (histogramData.value !== undefined) {
            volumeFormatted = histogramData.value.toLocaleString(undefined, { maximumFractionDigits: 0 })
          }
        }

        // ...existing code for SMA data...

        // Get SMA data only if SMA series exist
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

      // Update legend content (conditional SMA display)
      const legendContent = shouldShowSMA 
        ? `Price: <span style="color: #ffd700;">${priceFormatted}</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">${volumeFormatted}</span> | SMA20: <span style="color: rgba(255, 215, 0, 0.8);">${sma20Formatted}</span> | SMA50: <span style="color: rgba(0, 206, 209, 0.8);">${sma50Formatted}</span> | SMA100: <span style="color: rgba(255, 20, 147, 0.8);">${sma100Formatted}</span>`
        : `Price: <span style="color: #ffd700;">${priceFormatted}</span> | Vol: <span style="color: rgba(38, 166, 154, 0.8);">${volumeFormatted}</span>`
      legend.innerHTML = legendContent
    })

    // Handle resize
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

  // Update chart on theme change
  useEffect(() => {
    if (!chartRef.current || !seriesRef.current || !volumeSeriesRef.current) return

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

      // Update volume series color based on theme with transparency
      const volumeColor = theme === 'dark' ? 'rgba(38, 166, 154, 0.6)' : 'rgba(33, 150, 243, 0.6)'
      volumeSeriesRef.current?.applyOptions({
        color: volumeColor,
      })

      // Update legend background based on theme
      if (legendRef.current) {
        const computedStyle = getComputedStyle(document.documentElement)
        const cardColor = computedStyle.getPropertyValue('--card').trim()
        const legendBg = theme === 'dark' 
          ? `hsla(${cardColor}, 0.75)` 
          : 'rgba(255, 255, 255, 0.75)'
        const legendTextColor = theme === 'dark' ? 'white' : 'black'
        const legendBorder = theme === 'dark' 
          ? 'rgba(255, 255, 255, 0.08)' 
          : 'rgba(0, 0, 0, 0.08)'
        
        legendRef.current.style.background = legendBg
        legendRef.current.style.color = legendTextColor
        legendRef.current.style.borderColor = legendBorder
        legendRef.current.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)'
        legendRef.current.style.backdropFilter = 'blur(4px)'
      }
    }, 100)

    return () => clearTimeout(timer)
  }, [theme])

  // Fetch chart data when symbol, timeframe changes, or chart is recreated
  useEffect(() => {
    // Manual refresh function (moved inside useEffect to avoid dependency issues)
    const manualRefresh = async () => {
      if (!seriesRef.current || !volumeSeriesRef.current) return

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

        // Prepare volume data with colors based on price movement
        const volumeData: HistogramData[] = data.map((item, index) => {
          const isUp = index === 0 ? true : item.close >= item.open
          return {
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.volume,
            color: isUp ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)', // Green for up, red for down with less transparency
          }
        })

        // Prepare SMA data only if SMA series exist
        const sma20Data: LineData[] = sma20SeriesRef.current ? data
          .filter(item => item.sma_20 !== null && item.sma_20 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_20!,
          })) : []

        const sma50Data: LineData[] = sma50SeriesRef.current ? data
          .filter(item => item.sma_50 !== null && item.sma_50 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_50!,
          })) : []

        const sma100Data: LineData[] = sma100SeriesRef.current ? data
          .filter(item => item.sma_100 !== null && item.sma_100 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_100!,
          })) : []

        // Sort by time to ensure proper order
        chartData.sort((a, b) => (a.time as number) - (b.time as number))
        volumeData.sort((a, b) => (a.time as number) - (b.time as number))
        sma20Data.sort((a, b) => (a.time as number) - (b.time as number))
        sma50Data.sort((a, b) => (a.time as number) - (b.time as number))
        sma100Data.sort((a, b) => (a.time as number) - (b.time as number))

        // Set data using the proper API method
        seriesRef.current.setData(chartData)
        volumeSeriesRef.current.setData(volumeData)
        if (sma20SeriesRef.current) sma20SeriesRef.current.setData(sma20Data)
        if (sma50SeriesRef.current) sma50SeriesRef.current.setData(sma50Data)
        if (sma100SeriesRef.current) sma100SeriesRef.current.setData(sma100Data)
        
        console.log(`Manual refresh complete. Chart data points: ${chartData.length}`)
        
        // Initialize live price data for higher timeframes
        if (selectedTimeframe !== '1m') {
          setTimeout(() => {
            fetchLivePriceUpdate()
          }, 200)
        }
        
        // Simply go to realtime - no back and forth movement
        setTimeout(() => {
          console.log('Auto-scrolling to realtime after manual refresh')
          scrollToRealtime()
        }, 100)
        
      } catch (error) {
        console.error('Error fetching chart data:', error)
        setError(`Failed to load chart data for ${selectedSymbol} (${selectedTimeframe}). Please try again.`)
      } finally {
        setLoading(false)
      }
    }
    // Real-time update function (defined inside useEffect to avoid dependency issues)
    const fetchRealtimeUpdate = async () => {
      if (!seriesRef.current || !volumeSeriesRef.current) return

      try {
        // Only do real-time updates for 1m timeframe to avoid price jumping issues
        if (selectedTimeframe !== '1m') {
          console.log('Skipping real-time update for higher timeframe:', selectedTimeframe)
          return
        }
        
        // Fetch the latest 1m candle for real-time updates
        const response = await fetch(
          `http://localhost:8000/candlesticks/${selectedSymbol}?timeframe=1m&limit=1`
        )

        if (!response.ok) return

        const data: CandlestickApiData[] = await response.json()
        if (data.length === 0) return

        const latestData = data[0]
        
        // Update with latest 1m data
        const chartData: CandlestickData = {
          time: (new Date(latestData.time).getTime() / 1000) as Time,
          open: latestData.open,
          high: latestData.high,
          low: latestData.low,
          close: latestData.close,
        }

        const volumeData: HistogramData = {
          time: (new Date(latestData.time).getTime() / 1000) as Time,
          value: latestData.volume,
          color: latestData.close >= latestData.open ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)',
        }

        seriesRef.current.update(chartData)
        volumeSeriesRef.current.update(volumeData)

        // Update SMA data only if series exist and data is available
        if (sma20SeriesRef.current && latestData.sma_20 !== null && latestData.sma_20 !== undefined) {
          const sma20Data: LineData = {
            time: (new Date(latestData.time).getTime() / 1000) as Time,
            value: latestData.sma_20,
          }
          sma20SeriesRef.current.update(sma20Data)
        }

        if (sma50SeriesRef.current && latestData.sma_50 !== null && latestData.sma_50 !== undefined) {
          const sma50Data: LineData = {
            time: (new Date(latestData.time).getTime() / 1000) as Time,
            value: latestData.sma_50,
          }
          sma50SeriesRef.current.update(sma50Data)
        }

        if (sma100SeriesRef.current && latestData.sma_100 !== null && latestData.sma_100 !== undefined) {
          const sma100Data: LineData = {
            time: (new Date(latestData.time).getTime() / 1000) as Time,
            value: latestData.sma_100,
          }
          sma100SeriesRef.current.update(sma100Data)
        }

      } catch (error) {
        console.error('Error fetching real-time update:', error)
      }
    }

    // Live price update function for higher timeframes using 1m data
    const fetchLivePriceUpdate = async () => {
      if (!livePriceSeriesRef.current || selectedTimeframe === '1m') return

      try {
        // Fetch latest 5 1m candles and get the most recent one
        const response = await fetch(
          `http://localhost:8000/candlesticks/${selectedSymbol}?timeframe=1m&limit=1`
        )

        if (!response.ok) return

        const data: CandlestickApiData[] = await response.json()
        if (data.length === 0) return

        // Get the latest (most recent) candle - should be the last in the array
        const latestCandle = data[data.length - 1]
        const latestPrice = latestCandle.close

        // Get the current chart's visible time range
        if (!chartRef.current) return
        const timeScale = chartRef.current.timeScale()
        const visibleRange = timeScale.getVisibleRange()
        
        if (!visibleRange) return

        // Create a horizontal line across the visible time range using the latest 1m close price
        const livePriceData: LineData[] = [
          { time: visibleRange.from as Time, value: latestPrice },
          { time: visibleRange.to as Time, value: latestPrice }
        ]

        // Set the live price data as a horizontal line
        livePriceSeriesRef.current.setData(livePriceData)

        console.log(`Live price updated: ${latestPrice} from ${latestCandle.time}`)

      } catch (error) {
        console.error('Error fetching live price update:', error)
      }
    }

    const fetchChartData = async () => {
      if (!seriesRef.current || !volumeSeriesRef.current) return

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

        // Prepare volume data with colors based on price movement
        const volumeData: HistogramData[] = data.map((item, index) => {
          const isUp = index === 0 ? true : item.close >= item.open
          return {
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.volume,
            color: isUp ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)', // Green for up, red for down with less transparency
          }
        })

        // Prepare SMA data only if SMA series exist
        const sma20Data: LineData[] = sma20SeriesRef.current ? data
          .filter(item => item.sma_20 !== null && item.sma_20 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_20!,
          })) : []

        const sma50Data: LineData[] = sma50SeriesRef.current ? data
          .filter(item => item.sma_50 !== null && item.sma_50 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_50!,
          })) : []

        const sma100Data: LineData[] = sma100SeriesRef.current ? data
          .filter(item => item.sma_100 !== null && item.sma_100 !== undefined)
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item.sma_100!,
          })) : []

        // Sort by time to ensure proper order (data should already be sorted from API)
        chartData.sort((a, b) => (a.time as number) - (b.time as number))
        volumeData.sort((a, b) => (a.time as number) - (b.time as number))
        sma20Data.sort((a, b) => (a.time as number) - (b.time as number))
        sma50Data.sort((a, b) => (a.time as number) - (b.time as number))
        sma100Data.sort((a, b) => (a.time as number) - (b.time as number))

        // Set data using the proper API method
        seriesRef.current.setData(chartData)
        volumeSeriesRef.current.setData(volumeData)
        if (sma20SeriesRef.current) sma20SeriesRef.current.setData(sma20Data)
        if (sma50SeriesRef.current) sma50SeriesRef.current.setData(sma50Data)
        if (sma100SeriesRef.current) sma100SeriesRef.current.setData(sma100Data)
        
        console.log(`Data set successfully. Chart data points: ${chartData.length}`)
        
        // Automatically trigger refresh button after initial load to ensure proper display
        setTimeout(() => {
          console.log('Auto-triggering manual refresh for proper chart display')
          manualRefresh()
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
        const sampleVolumeData: HistogramData[] = [
          { time: '2023-12-01' as Time, value: 19103293.0, color: '#26a69a' },
          { time: '2023-12-02' as Time, value: 20345000.0, color: '#26a69a' },
          { time: '2023-12-03' as Time, value: 18123456.0, color: '#26a69a' },
        ]
        const sampleSMA20Data: LineData[] = [
          { time: '2023-12-01' as Time, value: 42100 },
          { time: '2023-12-02' as Time, value: 42400 },
          { time: '2023-12-03' as Time, value: 42700 },
        ]
        const sampleSMA50Data: LineData[] = [
          { time: '2023-12-01' as Time, value: 41900 },
          { time: '2023-12-02' as Time, value: 42200 },
          { time: '2023-12-03' as Time, value: 42500 },
        ]
        const sampleSMA100Data: LineData[] = [
          { time: '2023-12-01' as Time, value: 41700 },
          { time: '2023-12-02' as Time, value: 42000 },
          { time: '2023-12-03' as Time, value: 42300 },
        ]
        seriesRef.current.setData(sampleData)
        volumeSeriesRef.current.setData(sampleVolumeData)
        if (sma20SeriesRef.current) sma20SeriesRef.current.setData(sampleSMA20Data)
        if (sma50SeriesRef.current) sma50SeriesRef.current.setData(sampleSMA50Data)
        if (sma100SeriesRef.current) sma100SeriesRef.current.setData(sampleSMA100Data)
      } finally {
        setLoading(false)
      }
    }

    // Clear existing refresh interval
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
      refreshIntervalRef.current = null
    }

    // Clear existing real-time interval
    if (realtimeIntervalRef.current) {
      clearInterval(realtimeIntervalRef.current)
      realtimeIntervalRef.current = null
    }

    // Initial data fetch
    fetchChartData()

    // Set up auto-refresh based on timeframe (for full data reload)
    const refreshInterval = getRefreshInterval(selectedTimeframe)
    refreshIntervalRef.current = window.setInterval(() => {
      console.log(`Auto-refreshing chart data for ${selectedSymbol} (${selectedTimeframe})`)
      fetchChartData()
    }, refreshInterval)

    // Set up real-time updates (only for 1m timeframes to avoid price jumping)
    if (selectedTimeframe === '1m') {
      const realtimeInterval = getRealtimeInterval(selectedTimeframe)
      realtimeIntervalRef.current = window.setInterval(() => {
        console.log(`Real-time update for ${selectedSymbol} (${selectedTimeframe})`)
        fetchRealtimeUpdate()
      }, realtimeInterval)
    } else {
      console.log(`Skipping real-time updates for higher timeframe: ${selectedTimeframe}`)
      
      // Set up live price updates for higher timeframes using 1m data
      const livePriceInterval = 5000 // Update live price every 5 seconds
      realtimeIntervalRef.current = window.setInterval(() => {
        console.log(`Live price update for ${selectedSymbol} (${selectedTimeframe})`)
        fetchLivePriceUpdate()
      }, livePriceInterval)
    }

    // Cleanup function
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
      if (realtimeIntervalRef.current) {
        clearInterval(realtimeIntervalRef.current)
        realtimeIntervalRef.current = null
      }
    }
  }, [selectedSymbol, selectedTimeframe])

  // Function to scroll to real-time (latest 200 candles)
  const scrollToRealtime = () => {
    if (chartRef.current) {
      console.log('Scrolling to latest data')
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

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
      if (realtimeIntervalRef.current) {
        clearInterval(realtimeIntervalRef.current)
        realtimeIntervalRef.current = null
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

  // Manual refresh function (calls the internal one)
  const handleManualRefresh = async () => {
    // This will be handled by the auto-trigger mechanism inside useEffect
    // or users can click the button which triggers a symbol/timeframe change
    // For manual button clicks, we just trigger a re-render by updating state
    setCurrentTime(new Date())
    
    // Also go to realtime when manually refreshing with minimal delay
    setTimeout(() => {
      console.log('Manual refresh: Auto-scrolling to realtime')
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
                {/* {selectedTimeframe === '1m' && ' | Real-time: ON'}
                {selectedTimeframe !== '1m' && ' | Real-time: OFF'} */}
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