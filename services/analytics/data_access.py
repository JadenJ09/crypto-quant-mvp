# services/analytics/data_access.py
import pandas as pd
import psycopg
from datetime import datetime, timedelta
import os
from typing import Optional, List

class CryptoDataLoader:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", 
            "postgresql://quant_user:quant_password@localhost:5433/quant_db")
    
    def get_ohlcv_data(self, 
                       symbol: str = None, 
                       start_time: datetime = None, 
                       end_time: datetime = None,
                       limit: int = None) -> pd.DataFrame:
        """
        Load OHLCV data as pandas DataFrame
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT') or None for all
            start_time: Start datetime (default: 24 hours ago)
            end_time: End datetime (default: now)
            limit: Maximum number of records
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        
        # Build query conditions
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
            
        if start_time:
            conditions.append("time >= %(start_time)s")
            params['start_time'] = start_time
            
        if end_time:
            conditions.append("time <= %(end_time)s")
            params['end_time'] = end_time
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT 
            time,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM ohlcv_1min 
        {where_clause}
        ORDER BY time DESC
        {limit_clause}
        """
        
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        # Convert time to datetime index for time series analysis
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')  # Sort ascending for time series
        
        return df
    
    def get_multiple_symbols(self, 
                           symbols: List[str],
                           hours: int = 24) -> pd.DataFrame:
        """Get data for multiple symbols in the last N hours"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        all_data = []
        for symbol in symbols:
            df = self.get_ohlcv_data(symbol, start_time, end_time)
            all_data.append(df)
            
        return pd.concat(all_data, ignore_index=True)
    
    def get_data_for_ml(self, 
                       symbol: str,
                       hours: int = 168) -> pd.DataFrame:  # 1 week default
        """
        Get data formatted for ML/technical analysis
        
        Returns DataFrame with technical indicators added
        """
        df = self.get_ohlcv_data(
            symbol=symbol,
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        if df.empty:
            return df
            
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators for ML features"""
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change features
        df['price_change'] = df['close'].pct_change()
        df['price_change_1h'] = df['close'].pct_change(60)  # 1 hour ago
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # High-Low features
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        return df

# Usage examples
if __name__ == "__main__":
    loader = CryptoDataLoader()
    
    # Example 1: Get recent BTCUSDT data
    btc_data = loader.get_ohlcv_data('BTCUSDT', limit=100)
    print(f"BTCUSDT data shape: {btc_data.shape}")
    print(btc_data.head())
    
    # Example 2: Get data with technical indicators for ML
    ml_data = loader.get_data_for_ml('BTCUSDT', hours=24)
    print(f"ML features: {ml_data.columns.tolist()}")
    
    # Example 3: Get multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    multi_data = loader.get_multiple_symbols(symbols, hours=1)
    print(f"Multi-symbol data: {multi_data.groupby('symbol').count()}")