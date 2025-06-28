import { useState, useEffect, useCallback } from 'react'

// Simple auto-refresh hook
export const useAutoRefresh = (callback: () => void, interval: number, enabled: boolean = true) => {
  useEffect(() => {
    if (!enabled) return

    const id = setInterval(callback, interval)
    return () => clearInterval(id)
  }, [callback, interval, enabled])
}

// Clock hook for current time updates
export const useClock = () => {
  const [currentTime, setCurrentTime] = useState<Date>(new Date())

  useEffect(() => {
    const updateTime = () => setCurrentTime(new Date())
    const interval = setInterval(updateTime, 1000)
    return () => clearInterval(interval)
  }, [])

  return currentTime
}

// Local storage persistence hook
export const useLocalStorage = <T>(key: string, defaultValue: T) => {
  const [value, setValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : defaultValue
    } catch {
      return defaultValue
    }
  })

  const setStoredValue = useCallback((newValue: T) => {
    try {
      setValue(newValue)
      window.localStorage.setItem(key, JSON.stringify(newValue))
    } catch (error) {
      console.error(`Error saving to localStorage:`, error)
    }
  }, [key])

  return [value, setStoredValue] as const
}

// Utility functions for time formatting
export const formatCurrentTime = (date: Date): string => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')
  
  const timezoneName = date.toLocaleString('en-US', { timeZoneName: 'short' }).split(' ').pop()
  
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} ${timezoneName}`
}

export const formatUTCTime = (date: Date): string => {
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  const hours = String(date.getUTCHours()).padStart(2, '0')
  const minutes = String(date.getUTCMinutes()).padStart(2, '0')
  const seconds = String(date.getUTCSeconds()).padStart(2, '0')
  
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} UTC`
}

// Refresh interval utilities
export const getRefreshInterval = (timeframe: string): number => {
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

export const getRefreshIntervalText = (timeframe: string): string => {
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

export const getShortTimeframeLabel = (value: string): string => {
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
