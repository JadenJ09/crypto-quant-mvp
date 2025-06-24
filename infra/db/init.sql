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

-- Turn our regular table into a TimescaleDB hypertable.
-- This is the core command that enables all of TimescaleDB's performance magic.
-- It partitions the data by the 'time' column into chunks of 1 day each.
-- We also add a unique constraint on (symbol, time) which is required for UPSERTs.
SELECT create_hypertable('ohlcv_1min', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
ALTER TABLE ohlcv_1min ADD CONSTRAINT unique_ohlcv_1min_symbol_time UNIQUE (symbol, "time");

-- Create indexes to speed up common queries. A composite index on (symbol, time)
-- is essential for efficiently fetching time-series data for a specific symbol.
CREATE INDEX IF NOT EXISTS idx_ohlcv_1min_symbol_time ON ohlcv_1min (symbol, "time" DESC);
