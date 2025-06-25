-- infra/db/init.sql
-- This script is run automatically by the TimescaleDB container on its first start.
-- It sets up the necessary extensions and tables for our application.

-- The 'IF NOT EXISTS' clause prevents errors if the script is run again.
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the main table for our 1-minute OHLCV data.
CREATE TABLE IF NOT EXISTS ohlcv_1min (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- Create tables for different timeframes
-- 5-minute OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_5min (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- 15-minute OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_15min (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- 1-hour OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_1hour (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- 4-hour OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_4hour (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- 24-hour (1 day) OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_1day (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- 7-day OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_7day (
    "time"      TIMESTAMPTZ       NOT NULL,
    "symbol"    TEXT              NOT NULL,
    "open"      DOUBLE PRECISION  NOT NULL,
    "high"      DOUBLE PRECISION  NOT NULL,
    "low"       DOUBLE PRECISION  NOT NULL,
    "close"     DOUBLE PRECISION  NOT NULL,
    "volume"    DOUBLE PRECISION  NOT NULL
);

-- Turn our regular tables into TimescaleDB hypertables.
-- This is the core command that enables all of TimescaleDB's performance magic.
-- It partitions the data by the 'time' column into chunks.
-- We also add unique constraints on (symbol, time) which is required for UPSERTs.

SELECT create_hypertable('ohlcv_1min', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('ohlcv_5min', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3 days');
SELECT create_hypertable('ohlcv_15min', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');
SELECT create_hypertable('ohlcv_1hour', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 days');
SELECT create_hypertable('ohlcv_4hour', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '90 days');
SELECT create_hypertable('ohlcv_1day', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '365 days');
SELECT create_hypertable('ohlcv_7day', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3650 days');

-- Add unique constraints for all timeframes
ALTER TABLE ohlcv_1min ADD CONSTRAINT unique_ohlcv_1min_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_5min ADD CONSTRAINT unique_ohlcv_5min_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_15min ADD CONSTRAINT unique_ohlcv_15min_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_1hour ADD CONSTRAINT unique_ohlcv_1hour_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_4hour ADD CONSTRAINT unique_ohlcv_4hour_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_1day ADD CONSTRAINT unique_ohlcv_1day_symbol_time UNIQUE (symbol, "time");
ALTER TABLE ohlcv_7day ADD CONSTRAINT unique_ohlcv_7day_symbol_time UNIQUE (symbol, "time");

-- Create indexes to speed up common queries. A composite index on (symbol, time)
-- is essential for efficiently fetching time-series data for a specific symbol.
CREATE INDEX IF NOT EXISTS idx_ohlcv_1min_symbol_time ON ohlcv_1min (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_5min_symbol_time ON ohlcv_5min (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_15min_symbol_time ON ohlcv_15min (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1hour_symbol_time ON ohlcv_1hour (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_4hour_symbol_time ON ohlcv_4hour (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1day_symbol_time ON ohlcv_1day (symbol, "time" DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_7day_symbol_time ON ohlcv_7day (symbol, "time" DESC);

-- Create continuous aggregates for automatic timeframe aggregation
-- These will automatically aggregate 1-minute data into higher timeframes

-- 5-minute aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5min_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('5 minutes', "time"), symbol
WITH NO DATA;

-- 15-minute aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15min_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('15 minutes', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('15 minutes', "time"), symbol
WITH NO DATA;

-- 1-hour aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1hour_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('1 hour', "time"), symbol
WITH NO DATA;

-- 4-hour aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_4hour_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('4 hours', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('4 hours', "time"), symbol
WITH NO DATA;

-- 1-day aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1day_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('1 day', "time"), symbol
WITH NO DATA;

-- 7-day aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_7day_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('7 days', "time") AS "time",
    symbol,
    first("open", "time") AS "open",
    max("high") AS "high",
    min("low") AS "low",
    last("close", "time") AS "close",
    sum("volume") AS "volume"
FROM ohlcv_1min
GROUP BY time_bucket('7 days', "time"), symbol
WITH NO DATA;

-- Set up refresh policies for continuous aggregates
-- These will automatically update the aggregated data at regular intervals
SELECT add_continuous_aggregate_policy('ohlcv_5min_agg',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('ohlcv_15min_agg',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('ohlcv_1hour_agg',
    start_offset => INTERVAL '12 hours',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes');

SELECT add_continuous_aggregate_policy('ohlcv_4hour_agg',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('ohlcv_1day_agg',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours');

SELECT add_continuous_aggregate_policy('ohlcv_7day_agg',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create a function to populate aggregated tables from continuous aggregates
CREATE OR REPLACE FUNCTION populate_timeframe_tables()
RETURNS VOID AS $$
BEGIN
    -- Populate 5min table from aggregate
    INSERT INTO ohlcv_5min (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_5min_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 15min table from aggregate
    INSERT INTO ohlcv_15min (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_15min_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 1hour table from aggregate
    INSERT INTO ohlcv_1hour (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_1hour_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 4hour table from aggregate
    INSERT INTO ohlcv_4hour (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_4hour_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 1day table from aggregate
    INSERT INTO ohlcv_1day (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_1day_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 7day table from aggregate
    INSERT INTO ohlcv_7day (time, symbol, open, high, low, close, volume)
    SELECT time, symbol, open, high, low, close, volume
    FROM ohlcv_7day_agg
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;
END;
$$ LANGUAGE plpgsql;

-- Function to manually populate all timeframe tables from 1-minute data
-- This is used when continuous aggregates are not yet populated or need a full refresh
CREATE OR REPLACE FUNCTION populate_all_timeframes_from_1min()
RETURNS VOID AS $$
BEGIN
    -- First refresh all continuous aggregates to ensure they have the latest data
    CALL refresh_continuous_aggregate('ohlcv_5min_agg', NULL, NULL);
    CALL refresh_continuous_aggregate('ohlcv_15min_agg', NULL, NULL);
    CALL refresh_continuous_aggregate('ohlcv_1hour_agg', NULL, NULL);
    CALL refresh_continuous_aggregate('ohlcv_4hour_agg', NULL, NULL);
    CALL refresh_continuous_aggregate('ohlcv_1day_agg', NULL, NULL);
    CALL refresh_continuous_aggregate('ohlcv_7day_agg', NULL, NULL);

    -- Now populate all timeframe tables from the refreshed aggregates
    PERFORM populate_timeframe_tables();

    RAISE NOTICE 'All timeframe tables have been populated from 1-minute data';
END;
$$ LANGUAGE plpgsql;

-- Function to populate timeframes using direct aggregation from 1-minute data
-- This ensures complete historical data population even if continuous aggregates are empty
CREATE OR REPLACE FUNCTION bootstrap_timeframe_tables()
RETURNS VOID AS $$
BEGIN
    RAISE NOTICE 'Starting bootstrap of all timeframe tables from 1-minute data...';

    -- Populate 5-minute data directly from 1-minute data
    INSERT INTO ohlcv_5min (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes' AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes' ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes') AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes') AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes' ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes') AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes')
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 5) * INTERVAL '5 minutes', time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 15-minute data directly from 1-minute data
    INSERT INTO ohlcv_15min (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes' AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes' ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes') AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes') AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes' ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes') AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes')
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('hour', time) + (EXTRACT(minute FROM time)::int / 15) * INTERVAL '15 minutes', time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 1-hour data directly from 1-minute data
    INSERT INTO ohlcv_1hour (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('hour', time) AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('hour', time) ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('hour', time)) AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('hour', time)) AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('hour', time) ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('hour', time)) AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('hour', time))
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('hour', time), time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 4-hour data directly from 1-minute data
    INSERT INTO ohlcv_4hour (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours' AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours' ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours') AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours') AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours' ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours') AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours')
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('day', time) + (EXTRACT(hour FROM time)::int / 4) * INTERVAL '4 hours', time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 1-day data directly from 1-minute data
    INSERT INTO ohlcv_1day (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('day', time) AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('day', time) ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('day', time)) AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('day', time)) AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('day', time) ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('day', time)) AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('day', time))
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('day', time), time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    -- Populate 7-day data directly from 1-minute data
    INSERT INTO ohlcv_7day (time, symbol, open, high, low, close, volume)
    SELECT 
        date_trunc('week', time) AS time,
        symbol,
        first_value(open) OVER (PARTITION BY symbol, date_trunc('week', time) ORDER BY time) AS open,
        max(high) OVER (PARTITION BY symbol, date_trunc('week', time)) AS high,
        min(low) OVER (PARTITION BY symbol, date_trunc('week', time)) AS low,
        last_value(close) OVER (PARTITION BY symbol, date_trunc('week', time) ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
        sum(volume) OVER (PARTITION BY symbol, date_trunc('week', time)) AS volume
    FROM (
        SELECT DISTINCT ON (symbol, date_trunc('week', time))
            time, symbol, open, high, low, close, volume
        FROM ohlcv_1min
        ORDER BY symbol, date_trunc('week', time), time DESC
    ) sub
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;

    RAISE NOTICE 'Bootstrap complete! All timeframe tables populated from 1-minute data.';
END;
$$ LANGUAGE plpgsql;
