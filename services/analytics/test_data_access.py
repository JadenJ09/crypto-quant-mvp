# test_data_access.py
import pandas as pd
import psycopg

# Database connection
db_url = "postgresql://quant_user:quant_password@localhost:5433/quant_db"

try:
    with psycopg.connect(db_url) as conn:
        # Check data summary
        summary_query = """
        SELECT 
            symbol,
            COUNT(*) as records,
            MIN(time) as earliest,
            MAX(time) as latest,
            ROUND(AVG(volume)::numeric, 2) as avg_volume,
            ROUND(AVG(close)::numeric, 2) as avg_price
        FROM ohlcv_1min 
        GROUP BY symbol 
        ORDER BY records DESC;
        """
        
        df_summary = pd.read_sql_query(summary_query, conn)
        print("📊 Data Summary:")
        print(df_summary)
        
        # Get recent BTCUSDT data
        recent_query = """
        SELECT * FROM ohlcv_1min 
        WHERE symbol = 'BTCUSDT' 
        ORDER BY time DESC 
        LIMIT 5;
        """
        
        df_recent = pd.read_sql_query(recent_query, conn)
        print("\n🔍 Recent BTCUSDT Data:")
        print(df_recent)
        
except Exception as e:
    print(f"❌ Connection error: {e}")