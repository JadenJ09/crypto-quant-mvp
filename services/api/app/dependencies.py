# ==============================================================================
# File: services/api/app/dependencies.py
# Description: Shared dependencies for the API service
# ==============================================================================

import os
from psycopg_pool import AsyncConnectionPool

# Global variables
db_pool = None

def get_db_pool() -> AsyncConnectionPool:
    """Get the database connection pool"""
    global db_pool
    return db_pool

def set_db_pool(pool: AsyncConnectionPool):
    """Set the database connection pool"""
    global db_pool
    db_pool = pool

def get_vectorbt_service_url() -> str:
    """Get the vectorbt service URL from environment"""
    return os.environ.get("VECTORBT_SERVICE_URL", "http://vectorbt-api:8002")

def get_database_url() -> str:
    """Get the database URL from environment"""
    return os.environ.get("DATABASE_URL", "postgresql://quant_user:quant_password@timescaledb:5433/quant_db")
