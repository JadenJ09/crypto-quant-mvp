import React, { useEffect, useRef, useCallback } from 'react'
import { 
  createChart, 
  IChartApi, 
  CandlestickData, 
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  Time, 
  HistogramData, 
  LineData
} from 'lightweight-charts'
import { CandlestickApiData } from './useChartData'

export const useChart = (
  containerRef: React.RefObject<HTMLDivElement | null>,
  theme: string,
  showVolume: boolean,
  selectedIndicators: string[]
) => {
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<any>(null)
  const volumeSeriesRef = useRef<any>(null)
  const indicatorSeriesMap = useRef<Map<string, any>>(new Map())

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: theme === 'dark' ? '#1a1a1a' : '#ffffff' },
        textColor: theme === 'dark' ? '#ffffff' : '#333333',
      },
      grid: {
        vertLines: { color: theme === 'dark' ? '#2a2a2a' : '#f0f0f0' },
        horzLines: { color: theme === 'dark' ? '#2a2a2a' : '#f0f0f0' },
      },
      rightPriceScale: {
        borderColor: theme === 'dark' ? '#485158' : '#cccccc',
      },
      timeScale: {
        borderColor: theme === 'dark' ? '#485158' : '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Add candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries

    console.log('Chart initialized successfully')

    // Handle resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      console.log('Cleaning up chart')
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
      }
      chartRef.current = null
      candlestickSeriesRef.current = null
      volumeSeriesRef.current = null
      indicatorSeriesMap.current.clear()
    }
  }, [containerRef, theme])

  // Update volume series
  useEffect(() => {
    if (!chartRef.current) return

    // Remove existing volume series
    if (volumeSeriesRef.current) {
      chartRef.current.removeSeries(volumeSeriesRef.current)
      volumeSeriesRef.current = null
    }

    if (showVolume) {
      const volumeSeries = chartRef.current.addSeries(HistogramSeries, {
        color: '#26a69a',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })

      chartRef.current.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })

      volumeSeriesRef.current = volumeSeries
      console.log('Volume series added')
    }
  }, [showVolume])

  // Update indicator series
  useEffect(() => {
    if (!chartRef.current) return

    // Clear existing indicators
    indicatorSeriesMap.current.forEach((series) => {
      chartRef.current?.removeSeries(series)
    })
    indicatorSeriesMap.current.clear()

    // Add selected indicators (except Volume which is handled separately)
    selectedIndicators.forEach((indicator, index) => {
      if (indicator === 'Volume') return

      const colors = ['#2962FF', '#FF6D00', '#00C853', '#AA00FF', '#FF1744']
      const color = colors[index % colors.length]

      const series = chartRef.current!.addSeries(LineSeries, {
        color,
        lineWidth: 2,
      })

      indicatorSeriesMap.current.set(indicator, series)
    })

    console.log(`Added ${indicatorSeriesMap.current.size} indicator series`)
  }, [selectedIndicators])

  // Update chart data
  const updateChartData = useCallback((data: CandlestickApiData[]) => {
    if (!chartRef.current || !candlestickSeriesRef.current || data.length === 0) {
      console.log('Cannot update chart: missing chart or data')
      return
    }

    console.log(`Updating chart with ${data.length} data points`)

    try {
      // Convert and set candlestick data
      const chartData: CandlestickData[] = data.map(item => ({
        time: (new Date(item.time).getTime() / 1000) as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))

      candlestickSeriesRef.current.setData(chartData)

      // Update volume data
      if (volumeSeriesRef.current && showVolume) {
        const volumeData: HistogramData[] = data.map(item => ({
          time: (new Date(item.time).getTime() / 1000) as Time,
          value: item.volume,
          color: item.close >= item.open ? '#26a69a' : '#ef5350',
        }))
        volumeSeriesRef.current.setData(volumeData)
      }

      // Update indicator data
      const indicatorKeyMap: { [key: string]: string } = {
        'RSI': 'rsi_14',
        'MACD': 'macd_line',
        'Bollinger Bands': 'bb_middle',
        'EMA': 'ema_21',
        'SMA': 'sma_20',
        'VWAP': 'vwap',
        'ATR': 'atr_14',
        'Stochastic': 'stoch_k',
        'Williams %R': 'williams_r',
        'Volume SMA': 'volume_sma_20',
        'Volume Ratio': 'volume_ratio',
        'OBV': 'obv',
        'AD Line': 'ad_line',
        'CMF': 'cmf_20'
      }

      indicatorSeriesMap.current.forEach((series, indicatorName) => {
        const key = indicatorKeyMap[indicatorName]
        if (!key) return

        const indicatorData: LineData[] = data
          .filter(item => {
            const value = item[key as keyof CandlestickApiData]
            return typeof value === 'number' && !isNaN(value)
          })
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item[key as keyof CandlestickApiData] as number,
          }))

        if (indicatorData.length > 0) {
          series.setData(indicatorData)
        }
      })

      // Focus on latest candles
      if (chartData.length > 0) {
        const dataLength = chartData.length
        const startIndex = Math.max(0, dataLength - 200)
        const endIndex = dataLength - 1
        
        if (startIndex < endIndex) {
          const startTime = chartData[startIndex].time
          const endTime = chartData[endIndex].time
          const timeRange = (endTime as number) - (startTime as number)
          const extraSpace = timeRange * 0.1
          
          chartRef.current.timeScale().setVisibleRange({
            from: startTime,
            to: (endTime as number + extraSpace) as Time
          })
        }
      }

      console.log('Chart data updated successfully')
    } catch (error) {
      console.error('Failed to update chart data:', error)
    }
  }, [showVolume])

  return { updateChartData }
}
