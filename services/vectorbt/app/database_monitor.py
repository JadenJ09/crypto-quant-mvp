"""
Database monitor for new 1m OHLCV data
"""

import asyncio
import logging
from typing import Set, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Monitors TimescaleDB for new 1m OHLCV data and triggers processing"""
    
    def __init__(self, settings, db_manager):
        self.settings = settings
        self.db_manager = db_manager
        self._running = False
        self._last_check = {}  # Symbol -> last processed timestamp
        self._last_bulk_check = None  # Last time we checked for bulk processing needs
        
    async def start(self, data_handler):
        """Start monitoring for new data"""
        self._running = True
        self.data_handler = data_handler
        
        logger.info(f"🔍 Database monitor started (polling every {self.settings.polling_interval}s)")
        
        # Initialize last check timestamps
        await self._initialize_last_check()
        
        # Start monitoring loop
        while self._running:
            try:
                await self._check_for_new_data()
                await self._check_for_bulk_processing_needs()
                await asyncio.sleep(self.settings.polling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def stop(self):
        """Stop monitoring"""
        logger.info("🛑 Stopping database monitor...")
        self._running = False
        
    async def _initialize_last_check(self):
        """Initialize last check timestamps for all symbols"""
        try:
            symbols = await self.db_manager.get_available_symbols()
            
            for symbol in symbols:
                # Get the latest timestamp for this symbol
                latest_time = await self.db_manager.get_latest_1m_timestamp(symbol)
                if latest_time:
                    self._last_check[symbol] = latest_time
                    logger.debug(f"Initialized {symbol} last check: {latest_time}")
                    
            logger.info(f"Initialized monitoring for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error initializing last check timestamps: {e}")
            
    async def _check_for_new_data(self):
        """Check for new 1m OHLCV data since last check"""
        
        try:
            symbols = await self.db_manager.get_available_symbols()
            
            for symbol in symbols:
                if not self._running:
                    break
                    
                await self._check_symbol_for_new_data(symbol)
                
        except Exception as e:
            logger.error(f"Error checking for new data: {e}")
            
    async def _check_symbol_for_new_data(self, symbol: str):
        """Check for new data for a specific symbol"""
        
        try:
            # Get last check time for this symbol
            last_check = self._last_check.get(symbol)
            
            if last_check is None:
                # First time checking this symbol, get latest timestamp
                latest_time = await self.db_manager.get_latest_1m_timestamp(symbol)
                if latest_time:
                    self._last_check[symbol] = latest_time
                return
                
            # Look for new data since last check
            new_data = await self.db_manager.get_1m_data_since(symbol, last_check)
            
            if not new_data.empty:
                logger.info(f"📊 Found {len(new_data)} new 1m records for {symbol}")
                
                # Process each new record
                for timestamp, row in new_data.iterrows():
                    await self.data_handler({
                        'symbol': symbol,
                        'time': timestamp,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                    
                # Update last check timestamp
                self._last_check[symbol] = new_data.index.max()
                
        except Exception as e:
            logger.error(f"Error checking new data for {symbol}: {e}")
            
    async def _check_for_bulk_processing_needs(self):
        """Check if we need to run bulk processing for gap-filled data"""
        try:
            # Only check every 5 minutes to avoid excessive processing
            now = datetime.utcnow()
            if (self._last_bulk_check and 
                now - self._last_bulk_check < timedelta(minutes=5)):
                return
                
            self._last_bulk_check = now
            
            # Check if there are gaps between 1m data and higher timeframe data
            symbols = await self.db_manager.get_available_symbols()
            
            for symbol in symbols:
                gaps_found = await self._detect_timeframe_gaps(symbol)
                if gaps_found:
                    logger.info(f"🔧 Detected timeframe gaps for {symbol}, triggering bulk processing...")
                    await self._trigger_bulk_processing_for_symbol(symbol)
                    
        except Exception as e:
            logger.error(f"Error checking for bulk processing needs: {e}")
            
    async def _detect_timeframe_gaps(self, symbol: str) -> bool:
        """Detect if there are gaps between 1m data and higher timeframes"""
        try:
            # Get latest 1m timestamp
            latest_1m = await self.db_manager.get_latest_1m_timestamp(symbol)
            if not latest_1m:
                return False
                
            # Get latest 5m timestamp
            latest_5m = await self.db_manager.get_latest_timeframe_timestamp('ohlcv_5min', symbol)
            
            if not latest_5m:
                # No 5m data exists, definitely need bulk processing
                return True
                
            # Check if 1m data is significantly ahead of 5m data
            time_gap = latest_1m - latest_5m
            
            # If gap is more than 30 minutes, we likely have missing timeframe data
            if time_gap > timedelta(minutes=30):
                logger.info(f"📊 Time gap detected for {symbol}: 1m data up to {latest_1m}, 5m data up to {latest_5m}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting timeframe gaps for {symbol}: {e}")
            return False
            
    async def _trigger_bulk_processing_for_symbol(self, symbol: str):
        """Trigger bulk processing for a specific symbol's recent data"""
        try:
            # Get the time range that needs processing
            latest_5m = await self.db_manager.get_latest_timeframe_timestamp('ohlcv_5min', symbol)
            latest_1m = await self.db_manager.get_latest_1m_timestamp(symbol)
            
            if not latest_1m:
                return
                
            # Set start time for processing (either from latest 5m or 24 hours back)
            start_time = latest_5m if latest_5m else latest_1m - timedelta(hours=24)
            
            logger.info(f"🔄 Processing timeframes for {symbol} from {start_time} to {latest_1m}")
            
            # Process the gap using the data handler
            await self.data_handler.process_symbol_timeframes(
                symbol=symbol,
                start_time=start_time,
                end_time=latest_1m
            )
            
        except Exception as e:
            logger.error(f"Error triggering bulk processing for {symbol}: {e}")
            
    async def force_check_all_symbols(self):
        """Force check all symbols (useful for testing)"""
        logger.info("🔄 Force checking all symbols for new data...")
        await self._check_for_new_data()
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'running': self._running,
            'symbols_monitored': len(self._last_check),
            'last_check_times': {
                symbol: timestamp.isoformat() if timestamp else None 
                for symbol, timestamp in self._last_check.items()
            },
            'polling_interval': self.settings.polling_interval
        }
