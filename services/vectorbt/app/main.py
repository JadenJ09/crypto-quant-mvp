# service/vectorbt/app/main.py
"""
Multi-Purpose VectorBT Service

This service provides:
1. Real-time technical indicators calculation by monitoring TimescaleDB
2. Bulk historical indicators processing 
3. Multi-timeframe OHLCV aggregation
4. Backtesting capabilities (future)
5. ML indicators generation (future)

Architecture:
- Monitors TimescaleDB for new 1m OHLCV data
- Resamples to higher timeframes (5m, 15m, 1h, 4h, 1d, 7d)
- Calculates technical indicators using vectorbt and ta library
- Stores results in TimescaleDB with proper upsert handling
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from app.config import Settings
from app.indicators.processor import TechnicalIndicatorsProcessor
from app.database import DatabaseManager
from app.database_monitor import DatabaseMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorBTService:
    """Main service orchestrator for VectorBT-based analytics"""
    
    def __init__(self):
        self.settings = Settings()
        self.db_manager = DatabaseManager(self.settings.database_url)
        self.indicators_processor = TechnicalIndicatorsProcessor(self.settings)
        self.db_monitor = DatabaseMonitor(self.settings, self.db_manager)
        self._shutdown = False
        
        # Inject database manager into indicators processor
        self.indicators_processor.set_db_manager(self.db_manager)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown = True
        
    async def start(self):
        """Start the service based on configured mode"""
        logger.info(f"🚀 Starting VectorBT Service in {self.settings.service_mode} mode")
        
        try:
            # Initialize database connection
            await self.db_manager.initialize()
            
            if self.settings.service_mode == "bulk":
                await self._run_bulk_processing()
            elif self.settings.service_mode == "indicators":
                await self._run_realtime_processing()
            elif self.settings.service_mode == "orchestrator":
                # Run orchestrator mode - delegate to pipeline_orchestrator
                logger.info("🎯 Running in orchestrator mode - delegating to pipeline_orchestrator")
                from app.pipeline_orchestrator import main as orchestrator_main
                await orchestrator_main()
                return
            else:
                raise ValueError(f"Unknown service mode: {self.settings.service_mode}")
                
        except Exception as e:
            logger.error(f"Service failed to start: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def _run_bulk_processing(self):
        """Process all historical data in bulk - one-shot operation"""
        logger.info("📊 Starting bulk historical indicators processing...")
        
        try:
            # Get all symbols from database
            symbols = await self.db_manager.get_available_symbols()
            logger.info(f"Found {len(symbols)} symbols to process")
            
            for symbol in symbols:
                if self._shutdown:
                    logger.info("Shutdown signal received, stopping bulk processing")
                    break
                    
                logger.info(f"Processing historical data for {symbol}")
                await self.indicators_processor.process_symbol_bulk(symbol)
                
            if not self._shutdown:
                logger.info("✅ Bulk processing completed successfully")
            else:
                logger.info("⚠️ Bulk processing stopped due to shutdown signal")
                
        except Exception as e:
            logger.error(f"Error during bulk processing: {e}")
            raise
    
    async def _run_realtime_processing(self):
        """Process incoming 1m OHLCV data in real-time by monitoring database"""
        logger.info("⚡ Starting real-time indicators processing...")
        
        # Start database monitor
        await self.db_monitor.start(self._handle_ohlcv_message)
        
        # Keep running until shutdown
        while not self._shutdown:
            await asyncio.sleep(1)
            
    async def _handle_ohlcv_message(self, message: Dict[str, Any]):
        """Handle new 1m OHLCV data detected from database polling"""
        try:
            # Extract OHLCV data
            symbol = message.get('symbol')
            timestamp = message.get('time')
            
            if not symbol or not timestamp:
                logger.warning(f"Invalid message format: {message}")
                return
                
            logger.debug(f"Processing 1m OHLCV for {symbol} at {timestamp}")
            
            # Process this new data point
            await self.indicators_processor.process_new_ohlcv(message)
            
        except Exception as e:
            logger.error(f"Error processing OHLCV message: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        if hasattr(self, 'db_monitor'):
            await self.db_monitor.stop()
            
        if hasattr(self, 'db_manager'):
            await self.db_manager.close()
            
        logger.info("✅ Cleanup completed")

async def main():
    """Main entry point"""
    service = VectorBTService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
