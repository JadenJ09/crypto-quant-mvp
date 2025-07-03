import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'

interface PriceData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface DashboardStats {
  currentPrice: number
  change24h: number
  change24hPercent: number
  volume24h: number
  high24h: number
  low24h: number
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{
    payload: PriceData
  }>
  label?: string
}

const Dashboard: React.FC = () => {
  const [priceData, setPriceData] = useState<PriceData[]>([])
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        // Fetch recent price data
        const response = await fetch(
          `http://localhost:8002/ohlcv/${selectedSymbol}?timeframe=1h&limit=100`
        )
        
        if (response.ok) {
          const data = await response.json()
          setPriceData(data)
          
          // Calculate stats from the data
          if (data.length > 0) {
            const latest = data[data.length - 1]
            const previous = data[data.length - 2]
            
            setStats({
              currentPrice: latest.close,
              change24h: latest.close - previous.close,
              change24hPercent: ((latest.close - previous.close) / previous.close) * 100,
              volume24h: data.slice(-24).reduce((sum: number, item: PriceData) => sum + item.volume, 0),
              high24h: Math.max(...data.slice(-24).map((item: PriceData) => item.high)),
              low24h: Math.min(...data.slice(-24).map((item: PriceData) => item.low))
            })
          }
        } else {
          // Mock data for demo
          const mockData: PriceData[] = []
          const now = Date.now()
          const basePrice = 45000
          
          for (let i = 99; i >= 0; i--) {
            const time = new Date(now - i * 3600000) // 1 hour intervals
            const price = basePrice + Math.sin(i * 0.1) * 2000 + (Math.random() - 0.5) * 1000
            mockData.push({
              time: time.toISOString(),
              open: price,
              high: price + Math.random() * 500,
              low: price - Math.random() * 500,
              close: price + (Math.random() - 0.5) * 200,
              volume: Math.random() * 1000000
            })
          }
          
          setPriceData(mockData)
          const latest = mockData[mockData.length - 1]
          const previous = mockData[mockData.length - 2]
          
          setStats({
            currentPrice: latest.close,
            change24h: latest.close - previous.close,
            change24hPercent: ((latest.close - previous.close) / previous.close) * 100,
            volume24h: mockData.slice(-24).reduce((sum, item) => sum + item.volume, 0),
            high24h: Math.max(...mockData.slice(-24).map(item => item.high)),
            low24h: Math.min(...mockData.slice(-24).map(item => item.low))
          })
        }
      } catch (error) {
        console.error('Error fetching data:', error)
        // Set mock data on error (same as above)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 60000) // Update every minute
    
    return () => clearInterval(interval)
  }, [selectedSymbol])

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatVolume = (value: number) => {
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`
    return value.toFixed(0)
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const CustomTooltip = ({ active, payload, label }: TooltipProps) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">{label && formatDate(label)}</p>
          <div className="space-y-1 mt-2">
            <p className="text-sm">
              <span className="text-muted-foreground">Price: </span>
              <span className="font-medium">{formatCurrency(data.close)}</span>
            </p>
            <p className="text-sm">
              <span className="text-muted-foreground">Volume: </span>
              <span className="font-medium">{formatVolume(data.volume)}</span>
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-4 py-2 border rounded-md bg-background"
            title="Select trading pair"
          >
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="ADAUSDT">ADA/USDT</option>
            <option value="SOLUSDT">SOL/USDT</option>
          </select>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading dashboard data...</p>
            </div>
          </div>
        ) : (
          <>
            {/* Stats Cards */}
            {stats && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-card border rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Current Price</p>
                      <p className="text-2xl font-bold">{formatCurrency(stats.currentPrice)}</p>
                    </div>
                    <DollarSign className="h-8 w-8 text-muted-foreground" />
                  </div>
                </div>

                <div className="bg-card border rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">24h Change</p>
                      <p className={`text-2xl font-bold ${stats.change24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {stats.change24h >= 0 ? '+' : ''}{formatCurrency(stats.change24h)}
                      </p>
                      <p className={`text-sm ${stats.change24hPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {stats.change24hPercent >= 0 ? '+' : ''}{stats.change24hPercent.toFixed(2)}%
                      </p>
                    </div>
                    {stats.change24h >= 0 ? 
                      <TrendingUp className="h-8 w-8 text-green-600" /> :
                      <TrendingDown className="h-8 w-8 text-red-600" />
                    }
                  </div>
                </div>

                <div className="bg-card border rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">24h Volume</p>
                      <p className="text-2xl font-bold">{formatVolume(stats.volume24h)}</p>
                    </div>
                    <Activity className="h-8 w-8 text-muted-foreground" />
                  </div>
                </div>

                <div className="bg-card border rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">24h Range</p>
                      <p className="text-lg font-bold">{formatCurrency(stats.low24h)}</p>
                      <p className="text-lg font-bold">{formatCurrency(stats.high24h)}</p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-muted-foreground" />
                  </div>
                </div>
              </div>
            )}

            {/* Price Chart */}
            <div className="bg-card border rounded-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">{selectedSymbol} Price Chart</h2>
              </div>
              
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={priceData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={formatDate}
                      className="text-xs"
                    />
                    <YAxis 
                      tickFormatter={(value: number) => formatCurrency(value)}
                      className="text-xs"
                      domain={['dataMin - 100', 'dataMax + 100']}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, fill: '#3b82f6' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default Dashboard
