import { useState, useEffect, useCallback } from 'react'

interface CandlestickApiData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  // Technical Indicators
  rsi_14?: number
  rsi_21?: number
  rsi_30?: number
  macd_line?: number
  macd_signal?: number
  macd_histogram?: number
  bb_upper?: number
  bb_middle?: number
  bb_lower?: number
  ema_9?: number
  ema_21?: number
  ema_50?: number
  ema_100?: number
  ema_200?: number
  sma_20?: number
  sma_50?: number
  sma_100?: number
  sma_200?: number
  vwap?: number
  atr_14?: number
  stoch_k?: number
  stoch_d?: number
  williams_r?: number
  volume_sma_20?: number
  volume_ratio?: number
  obv?: number
  ad_line?: number
  cmf_20?: number
  // Index signature for dynamic access
  [key: string]: number | string | undefined
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

export const useChartData = (selectedSymbol: string, selectedTimeframe: string) => {
  const [data, setData] = useState<CandlestickApiData[]>([])
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])
  const [timeframes, setTimeframes] = useState<TimeframeInfo[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch symbols and timeframes on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        // Fetch symbols
        const symbolsResponse = await fetch('/symbols')
        if (symbolsResponse.ok) {
          const symbolsData = await symbolsResponse.json()
          setSymbols(symbolsData)
        }

        // Fetch timeframes
        const timeframesResponse = await fetch('/timeframes')
        if (timeframesResponse.ok) {
          const timeframesData = await timeframesResponse.json()
          setTimeframes(timeframesData)
        }
      } catch (err) {
        console.error('Failed to fetch metadata:', err)
        setError('Failed to load metadata')
      }
    }

    fetchMetadata()
  }, [])

  // Fetch chart data when symbol or timeframe changes
  const fetchData = useCallback(async () => {
    if (!selectedSymbol || !selectedTimeframe) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`/candlesticks/${selectedSymbol}?timeframe=${selectedTimeframe}&limit=1000`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const result = await response.json()
      console.log(`Received ${result.length} data points`)
      
      if (result && Array.isArray(result)) {
        setData(result)
      } else {
        throw new Error('Invalid data format received')
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch chart data'
      setError(errorMessage)
      console.error('Error fetching chart data:', err)
    } finally {
      setLoading(false)
    }
  }, [selectedSymbol, selectedTimeframe])

  // Effect to fetch data when dependencies change
  useEffect(() => {
    fetchData()
  }, [fetchData])

  return {
    data,
    symbols,
    timeframes,
    loading,
    error,
    refetch: fetchData
  }
}

// Export the interface for use in other components
export type { CandlestickApiData, TimeframeInfo, SymbolInfo }
