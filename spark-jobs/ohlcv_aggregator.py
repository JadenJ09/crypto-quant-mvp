# spark-jobs/ohlcv_aggregator.py
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, first, max, min, last, sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
INPUT_KAFKA_TOPIC = "raw.trades.crypto"
OUTPUT_KAFKA_TOPIC = "agg.ohlcv.1m"
CHECKPOINT_LOCATION = "/opt/bitnami/spark/data/checkpoints/ohlcv_aggregator"
DELTA_LAKE_LOCATION = "/opt/bitnami/spark/data/delta/ohlcv_1min"

# --- Main Spark Logic ---
def main():
    """
    Main function to run the Spark Structured Streaming job.
    """
    logging.info("Starting Spark session...")
    
    # Create Spark session for streaming without Delta Lake
    spark = SparkSession.builder \
        .appName("OHLCVAggregator") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    logging.info("Spark session created successfully.")

    # Define the schema for the incoming JSON trade data from the ingestor.
    # This must match the Binance aggTrade format produced by the ingestor service.
    trade_schema = StructType([
        StructField("e", StringType(), True),   # Event type (aggTrade)
        StructField("E", LongType(), True),     # Event time (milliseconds)
        StructField("a", LongType(), True),     # Aggregate trade ID
        StructField("s", StringType(), True),   # Symbol
        StructField("p", StringType(), True),   # Price
        StructField("q", StringType(), True),   # Quantity
        StructField("f", LongType(), True),     # First trade ID
        StructField("l", LongType(), True),     # Last trade ID
        StructField("T", LongType(), True),     # Trade time (milliseconds)
        StructField("m", BooleanType(), True),  # Is buyer maker
    ])

    # 1. Read from Kafka Source
    raw_trades_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", INPUT_KAFKA_TOPIC) \
        .option("failOnDataLoss", "false") \
        .load()

    # 2. Parse and Transform Data
    # Cast the binary 'value' from Kafka into a JSON string and parse it using our schema.
    parsed_trades_df = raw_trades_df \
        .select(from_json(col("value").cast("string"), trade_schema).alias("trade")) \
        .select(
            (col("trade.T") / 1000).cast("timestamp").alias("event_time"), # Convert to seconds
            col("trade.s").alias("symbol"),
            col("trade.p").cast(DoubleType()).alias("price"),
            col("trade.q").cast(DoubleType()).alias("quantity")
        )

    # 3. Define Windowed Aggregation
    # Watermarking is crucial for handling late-arriving data in streaming systems.
    # It tells Spark to wait for a certain period before finalizing a window.
    # Here, we wait up to 30 seconds for late data.
    ohlcv_df = parsed_trades_df \
        .withWatermark("event_time", "30 seconds") \
        .groupBy(
            window("event_time", "1 minute").alias("window"),
            col("symbol")
        ) \
        .agg(
            first("price", ignorenulls=True).alias("open"),
            max("price").alias("high"),
            min("price").alias("low"),
            last("price", ignorenulls=True).alias("close"),
            sum("quantity").alias("volume")
        ) \
        .select(
            col("window.start").alias("time"),
            "symbol", "open", "high", "low", "close", "volume"
        )

    # 4. Write to Kafka sink using foreachBatch
    # foreachBatch allows us to apply custom logic to the output of each micro-batch.
    def write_to_kafka_sink(batch_df, epoch_id):
        logging.info(f"--- Processing Micro-Batch ID: {epoch_id} ---")
        
        # Cache the batch DataFrame to avoid re-computation
        batch_df.persist()

        # Write to Kafka for real-time consumers
        row_count = batch_df.count()
        logging.info(f"Writing {row_count} rows to Kafka topic: {OUTPUT_KAFKA_TOPIC}")
        
        if row_count > 0:
            batch_df.selectExpr("to_json(struct(*)) AS value") \
                .write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
                .option("topic", OUTPUT_KAFKA_TOPIC) \
                .save()
            logging.info(f"Successfully wrote {row_count} rows to Kafka")
        else:
            logging.info("No data to write in this batch")

        # Release the cached DataFrame
        batch_df.unpersist()

    # Start the streaming query
    query = ohlcv_df.writeStream \
        .outputMode("update") \
        .foreachBatch(write_to_kafka_sink) \
        .option("checkpointLocation", CHECKPOINT_LOCATION) \
        .trigger(processingTime="1 minute") \
        .start()

    logging.info("Streaming query started. Awaiting termination...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
