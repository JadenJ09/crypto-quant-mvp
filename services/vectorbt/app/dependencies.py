# ==============================================================================
# File: services/vectorbt/app/dependencies.py
# Description: Shared dependencies for VectorBT service
# ==============================================================================

import asyncpg
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Global database pool (will be initialized in lifespan)
db_pool = None

async def get_db_pool():
    """
    Dependency to get the database pool.
    Raises HTTPException if pool is not initialized.
    """
    if not db_pool:
        logger.error("Database pool not initialized")
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    return db_pool

def set_db_pool(pool: asyncpg.Pool):
    """
    Set the global database pool.
    Called during application startup.
    """
    global db_pool
    db_pool = pool
    logger.info("Database pool set successfully")

async def close_db_pool():
    """
    Close the database pool.
    Called during application shutdown.
    """
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None
        logger.info("Database pool closed successfully")
