# services/analytics/spark_data_access.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class SparkDataLoader:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("CryptoAnalytics") \
            .config("spark.jars", "/opt/spark/jars/postgresql-42.7.0.jar") \
            .getOrCreate()
            
        self.db_url = "jdbc:postgresql://timescaledb:5433/quant_db"
        self.db_properties = {
            "user": "quant_user",
            "password": "quant_password",
            "driver": "org.postgresql.Driver"
        }
    
    def load_ohlcv_data(self, symbol=None, hours=24):
        """Load OHLCV data into Spark DataFrame"""
        
        query = """
        (SELECT * FROM ohlcv_1min 
         WHERE time >= NOW() - INTERVAL '%d hours'
         %s
         ORDER BY time) as ohlcv_data
        """ % (hours, f"AND symbol = '{symbol}'" if symbol else "")
        
        df = self.spark.read.jdbc(
            url=self.db_url,
            table=query,
            properties=self.db_properties
        )
        
        return df
    
    def add_spark_features(self, df):
        """Add technical indicators using Spark SQL"""
        
        # Register as temp view for SQL operations
        df.createOrReplaceTempView("ohlcv")
        
        # Add moving averages and technical indicators
        enhanced_df = self.spark.sql("""
        SELECT *,
            AVG(close) OVER (
                PARTITION BY symbol 
                ORDER BY time 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as sma_20,
            
            AVG(close) OVER (
                PARTITION BY symbol 
                ORDER BY time 
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) as sma_5,
            
            (close - LAG(close, 1) OVER (
                PARTITION BY symbol ORDER BY time
            )) / LAG(close, 1) OVER (
                PARTITION BY symbol ORDER BY time
            ) as price_change,
            
            AVG(volume) OVER (
                PARTITION BY symbol 
                ORDER BY time 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as volume_avg_20
            
        FROM ohlcv
        ORDER BY symbol, time
        """)
        
        return enhanced_df

# Usage example
if __name__ == "__main__":
    loader = SparkDataLoader()
    
    # # Load BTCUSDT data
    # df = loader.load_ohlcv_data('BTCUSDT', hours=24)
    # df.show(5)
    
    # # Add technical features
    # enhanced_df = loader.add_spark_features(df)
    # enhanced_df.select("time", "symbol", "close", "sma_20", "price_change").show(5)