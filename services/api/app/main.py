# ==============================================================================
# File: services/api/app/main.py
# Description: Lightweight FastAPI application - Web layer only
# ==============================================================================

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool

from .dependencies import set_db_pool, get_db_pool, get_database_url
from .routers import market_data, backtesting
from .models import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logging.info("API starting up... Creating database connection pool.")
    try:
        db_pool = AsyncConnectionPool(
            conninfo=get_database_url(),
            min_size=2,
            max_size=10
        )
        await db_pool.wait()
        set_db_pool(db_pool)
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.critical(f"Failed to create database connection pool: {e}")
    
    yield
    
    # Shutdown
    db_pool = get_db_pool()
    if db_pool:
        logging.info("API shutting down... Closing database connection pool.")
        await db_pool.close()

# Create FastAPI application
app = FastAPI(
    title="Crypto-Quant-MVP API",
    description="Lightweight API service for crypto quantitative trading platform",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(market_data.router)
app.include_router(backtesting.router)

# Health check endpoint
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        services={
            "database": "connected" if get_db_pool() else "disconnected",
            "vectorbt": "external_service"
        }
    )
