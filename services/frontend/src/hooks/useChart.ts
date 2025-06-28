import React, { useEffect, useRef, useCallback, useMemo } from 'react'
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
  const seriesRefs = useRef<{
    candlestick: any | null
    volume: any | null
    indicators: Map<string, any>
  }>({
    candlestick: null,
    volume: null,
    indicators: new Map()
  })

  // Map indicator names to API field names
  const indicatorKeyMap = useMemo(() => ({
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
  } as const), [])

  // Get indicator key for API field mapping
  const getIndicatorKey = useCallback((indicator: string): string | null => {
    return (indicatorKeyMap as Record<string, string>)[indicator] || null
  }, [indicatorKeyMap])

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return

    try {
      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height: 500,
        layout: {
          background: {
            color: theme === 'dark' ? '#1a1a1a' : '#ffffff',
          },
          textColor: theme === 'dark' ? '#ffffff' : '#333333',
        },
        grid: {
          vertLines: {
            color: theme === 'dark' ? '#2a2a2a' : '#f0f0f0',
          },
          horzLines: {
            color: theme === 'dark' ? '#2a2a2a' : '#f0f0f0',
          },
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

      // Create main candlestick series
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      })

      chartRef.current = chart
      seriesRefs.current.candlestick = candlestickSeries

      // Handle resize
      const handleResize = () => {
        if (containerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: containerRef.current.clientWidth,
          })
        }
      }

      window.addEventListener('resize', handleResize)

      // Cleanup function
      return () => {
        window.removeEventListener('resize', handleResize)
        if (chartRef.current) {
          chartRef.current.remove()
        }
        chartRef.current = null
        seriesRefs.current = {
          candlestick: null,
          volume: null,
          indicators: new Map()
        }
      }
    } catch (error) {
      console.error('Failed to create chart:', error)
    }
  }, [containerRef, theme])

  // Create volume series
  const createVolumeSeries = useCallback(() => {
    if (!chartRef.current) return

    try {
      // Remove existing volume series if it exists
      if (seriesRefs.current.volume) {
        chartRef.current.removeSeries(seriesRefs.current.volume)
        seriesRefs.current.volume = null
      }

      if (showVolume) {
        const volumeSeries = chartRef.current.addSeries(HistogramSeries, {
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        })

        chartRef.current.priceScale('volume').applyOptions({
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        })

        seriesRefs.current.volume = volumeSeries
      }
    } catch (error) {
      console.error('Failed to create volume series:', error)
    }
  }, [showVolume])

  // Create indicator series
  const createIndicatorSeries = useCallback(() => {
    if (!chartRef.current) return

    try {
      // Clear existing indicator series
      seriesRefs.current.indicators.forEach((series) => {
        chartRef.current?.removeSeries(series)
      })
      seriesRefs.current.indicators.clear()

      // Create series for selected indicators
      selectedIndicators.forEach((indicator, index) => {
        if (indicator === 'Volume') return // Volume is handled separately

        const colors = ['#2962FF', '#FF6D00', '#00C853', '#AA00FF', '#FF1744']
        const colorIndex = index % colors.length

        if (indicator.includes('RSI') || indicator.includes('Stochastic') || indicator.includes('Williams')) {
          // Oscillators go in separate price scale
          const series = chartRef.current!.addSeries(LineSeries, {
            color: colors[colorIndex],
            lineWidth: 2,
            priceScaleId: 'oscillator',
          })
          
          chartRef.current!.priceScale('oscillator').applyOptions({
            scaleMargins: {
              top: 0.1,
              bottom: 0.8,
            },
          })

          seriesRefs.current.indicators.set(indicator, series)
        } else {
          // Moving averages and other overlays go on main price scale
          const series = chartRef.current!.addSeries(LineSeries, {
            color: colors[colorIndex],
            lineWidth: 2,
          })

          seriesRefs.current.indicators.set(indicator, series)
        }
      })
    } catch (error) {
      console.error('Failed to create indicator series:', error)
    }
  }, [selectedIndicators])

  // Update volume and indicator series when dependencies change
  useEffect(() => {
    createVolumeSeries()
  }, [createVolumeSeries])

  useEffect(() => {
    createIndicatorSeries()
  }, [createIndicatorSeries])

  // Update chart data
  const updateChartData = useCallback((data: CandlestickApiData[]) => {
    if (!chartRef.current || !seriesRefs.current.candlestick || data.length === 0) return

    try {
      // Convert API data to chart format
      const chartData: CandlestickData[] = data.map(item => ({
        time: (new Date(item.time).getTime() / 1000) as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))

      seriesRefs.current.candlestick.setData(chartData)

      // Update volume data if series exists
      if (seriesRefs.current.volume) {
        const volumeData: HistogramData[] = data.map(item => ({
          time: (new Date(item.time).getTime() / 1000) as Time,
          value: item.volume,
          color: item.close >= item.open ? '#26a69a' : '#ef5350',
        }))
        seriesRefs.current.volume.setData(volumeData)
      }

      // Update indicator data
      seriesRefs.current.indicators.forEach((series, indicatorName: string) => {
        const key = getIndicatorKey(indicatorName)
        if (!key) return

        const indicatorData: LineData[] = data
          .filter(item => {
            const value = item[key]
            return typeof value === 'number' && !isNaN(value)
          })
          .map(item => ({
            time: (new Date(item.time).getTime() / 1000) as Time,
            value: item[key] as number,
          }))

        series.setData(indicatorData)
      })

      // Focus on latest 200 candles with some space at the end
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
    } catch (error) {
      console.error('Failed to update chart data:', error)
    }
  }, [getIndicatorKey])

  return {
    chartRef,
    updateChartData,
    getIndicatorKey
  }
}
