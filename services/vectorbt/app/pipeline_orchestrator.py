#!/usr/bin/env python3
"""
Automated Pipeline Orchestrator for Crypto Quant MVP

This orchestrator manages the complete data processing pipeline:
1. Database health check
2. Restore from backup (if needed)
3. Gap filling via data-recovery service
4. Incremental VectorBT processing (smart gap detection)
5. Real-time processing startup

Key Features:
- Intelligent gap detection and processing
- Avoids unnecessary full bulk processing
- Coordinates with Docker services
- Comprehensive logging and monitoring
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import Settings
from app.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the complete automated data processing pipeline"""
    
    def __init__(self):
        self.settings = Settings()
        self.db_manager = DatabaseManager(self.settings.database_url)
        # For Docker environment, the project root is mounted at /workspace
        self.project_root = Path("/workspace") if Path("/workspace").exists() else Path(__file__).parent.parent.parent.parent
        self.backup_dir = self.project_root / "backups"
        self._shutdown = False
        
    async def initialize(self):
        """Initialize the orchestrator"""
        try:
            await self.db_manager.initialize()
            logger.info("✅ Pipeline orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
            
    async def cleanup(self):
        """Cleanup resources"""
        await self.db_manager.close()
        logger.info("✅ Pipeline orchestrator cleaned up")
        
    async def run_automated_pipeline(self) -> bool:
        """
        Run the complete automated pipeline
        Returns True if pipeline completed successfully
        """
        logger.info("🚀 Starting automated data processing pipeline")
        
        try:
            # Step 1: Wait for database to be ready
            logger.info("🔍 Step 1: Waiting for database to be ready...")
            if not await self.db_manager.wait_for_database_ready(timeout=300):
                logger.error("❌ Database failed to become ready")
                return False
                
            # Step 2: Comprehensive health check
            logger.info("🏥 Step 2: Performing comprehensive database health check...")
            health = await self.db_manager.check_database_health()
            await self._log_health_status(health)
            
            # Step 3: Execute processing strategy based on health check
            success = await self._execute_processing_strategy(health)
            
            if success:
                logger.info("🎉 Automated pipeline completed successfully!")
                await self._log_final_status()
                return True
            else:
                logger.error("❌ Automated pipeline failed")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
            
    async def _execute_processing_strategy(self, health: Dict[str, Any]) -> bool:
        """
        Execute the appropriate processing strategy based on health status.
        
        NOTE: This method now only determines the strategy and outputs recommendations.
        The actual Docker orchestration is handled by the shell script on the host.
        """
        
        strategy = health.get('processing_recommendation', 'investigate_manually')
        logger.info(f"📋 Processing strategy determined: {strategy}")
        
        # Output strategy as JSON for shell script to parse
        strategy_output = {
            'strategy': strategy,
            'health_status': self._serialize_health_status(health),
            'next_actions': self._get_strategy_actions(strategy, health),
            'timestamp': datetime.now().isoformat()
        }
        
        # Write strategy to stdout in JSON format for shell script
        print("=== STRATEGY_OUTPUT ===")
        print(json.dumps(strategy_output, indent=2))
        print("=== END_STRATEGY_OUTPUT ===")
        
        # Also write to file for shell script to read
        strategy_file = Path("/workspace/tmp/pipeline_output/pipeline_strategy.json")
        try:
            strategy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(strategy_file, 'w') as f:
                json.dump(strategy_output, f, indent=2)
            logger.info(f"📄 Strategy written to {strategy_file}")
        except Exception as e:
            logger.warning(f"Could not write strategy file: {e}")
    
    def _serialize_health_status(self, health: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize health status for JSON output"""
        serialized = {}
        for key, value in health.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_health_status(value)
            elif isinstance(value, list):
                serialized[key] = [
                    self._serialize_health_status(item) if isinstance(item, dict) else
                    item.isoformat() if isinstance(item, datetime) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized
        
        # Return success - actual orchestration will be done by shell script
        return True
    
    def _get_strategy_actions(self, strategy: str, health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the specific actions required for a strategy"""
        
        actions = []
        
        if strategy == 'restore_from_backup':
            actions = [
                {'action': 'restore_database', 'script': 'restore_database.sh'},
                {'action': 'gap_fill', 'service': 'data-recovery', 'mode': 'gap-fill'},
                {'action': 'incremental_processing', 'service': 'vectorbt', 'mode': 'incremental'},
                {'action': 'start_realtime', 'service': 'vectorbt', 'mode': 'realtime', 'detached': True}
            ]
            
        elif strategy == 'gap_fill_then_incremental':
            actions = [
                {'action': 'gap_fill', 'service': 'data-recovery', 'mode': 'gap-fill'},
                {'action': 'incremental_processing', 'service': 'vectorbt', 'mode': 'incremental'},
                {'action': 'start_realtime', 'service': 'vectorbt', 'mode': 'realtime', 'detached': True}
            ]
            
        elif strategy == 'incremental_processing':
            actions = [
                {'action': 'incremental_processing', 'service': 'vectorbt', 'mode': 'incremental'},
                {'action': 'start_realtime', 'service': 'vectorbt', 'mode': 'realtime', 'detached': True}
            ]
            
        elif strategy == 'full_bulk_processing':
            actions = [
                {'action': 'full_bulk_processing', 'service': 'vectorbt', 'mode': 'bulk'},
                {'action': 'start_realtime', 'service': 'vectorbt', 'mode': 'realtime', 'detached': True}
            ]
            
        elif strategy == 'realtime_only':
            actions = [
                {'action': 'start_realtime', 'service': 'vectorbt', 'mode': 'realtime', 'detached': True},
                {'action': 'start_data_recovery', 'service': 'data-recovery', 'mode': 'continuous', 'detached': True}
            ]
        
        return actions
    
    async def _handle_restore_from_backup(self) -> bool:
        """Handle database restoration from backup"""
        logger.info("📦 Restoring database from latest backup...")
        
        try:
            # Find latest backup
            latest_backup = self._find_latest_backup()
            if not latest_backup:
                logger.error("❌ No backup files found")
                return False
                
            logger.info(f"📂 Using backup: {latest_backup}")
            
            # Restore database
            if not await self._run_script("restore_database.sh", [str(latest_backup)]):
                logger.error("❌ Database restoration failed")
                return False
                
            logger.info("✅ Database restored successfully")
            
            # After restoration, continue with gap filling and processing
            return await self._handle_gap_fill_then_incremental()
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    async def _handle_gap_fill_then_incremental(self) -> bool:
        """Handle gap filling followed by incremental processing"""
        logger.info("🔄 Starting gap filling and incremental processing...")
        
        try:
            # Step 1: Run gap filling
            logger.info("🕳️  Running gap detection and filling...")
            if not await self._run_docker_service("data-recovery", mode="gap-fill"):
                logger.warning("⚠️  Gap filling had issues, but continuing...")
            
            # Step 2: Analyze what needs incremental processing
            logger.info("📊 Analyzing incremental processing needs...")
            processing_periods = await self.db_manager.get_incremental_processing_periods(max_period_hours=24)
            
            if not processing_periods:
                logger.info("✅ No incremental processing needed")
                return await self._handle_realtime_only()
            
            # Step 3: Run incremental processing
            return await self._run_incremental_processing(processing_periods)
            
        except Exception as e:
            logger.error(f"Gap fill and incremental processing failed: {e}")
            return False
    
    async def _handle_incremental_processing(self) -> bool:
        """Handle targeted incremental processing for specific gaps"""
        logger.info("⚡ Running incremental processing for detected gaps...")
        
        try:
            # Get processing periods
            processing_periods = await self.db_manager.get_incremental_processing_periods(max_period_hours=12)
            
            if not processing_periods:
                logger.info("✅ No incremental processing needed")
                return await self._handle_realtime_only()
            
            return await self._run_incremental_processing(processing_periods)
            
        except Exception as e:
            logger.error(f"Incremental processing failed: {e}")
            return False
    
    async def _run_incremental_processing(self, processing_periods: List[Dict[str, Any]]) -> bool:
        """Run incremental processing for specific periods"""
        
        total_periods = len(processing_periods)
        logger.info(f"📋 Processing {total_periods} incremental periods...")
        
        successful_periods = 0
        
        for i, period in enumerate(processing_periods):
            if self._shutdown:
                logger.info("⏹️  Shutdown requested, stopping incremental processing")
                break
                
            logger.info(f"⚡ Processing period {i+1}/{total_periods}: {period['symbol']} "
                       f"from {period['start_time']} to {period['end_time']} "
                       f"({period['estimated_hours']:.1f}h)")
            
            try:
                # Run vectorbt service in incremental mode for this specific period
                success = await self._run_vectorbt_incremental(
                    symbol=period['symbol'],
                    start_time=period['start_time'],
                    end_time=period['end_time']
                )
                
                if success:
                    successful_periods += 1
                    logger.info(f"✅ Completed period {i+1}/{total_periods}")
                else:
                    logger.warning(f"⚠️  Failed period {i+1}/{total_periods}, continuing...")
                    
            except Exception as e:
                logger.error(f"Error processing period {i+1}: {e}")
        
        success_rate = successful_periods / total_periods if total_periods > 0 else 0
        logger.info(f"📊 Incremental processing completed: {successful_periods}/{total_periods} "
                   f"periods successful ({success_rate*100:.1f}%)")
        
        if success_rate >= 0.8:  # 80% success rate threshold
            logger.info("✅ Incremental processing completed successfully")
            return await self._handle_realtime_only()
        else:
            logger.warning("⚠️  Incremental processing had significant failures")
            return await self._handle_realtime_only()  # Still start realtime
    
    async def _handle_full_bulk_processing(self) -> bool:
        """Handle full bulk processing (last resort)"""
        logger.info("🔄 Running full bulk processing (this may take a while)...")
        
        try:
            # Run full bulk processing
            if not await self._run_docker_service("vectorbt-bulk", mode="bulk"):
                logger.error("❌ Full bulk processing failed")
                return False
                
            logger.info("✅ Full bulk processing completed")
            return await self._handle_realtime_only()
            
        except Exception as e:
            logger.error(f"Full bulk processing failed: {e}")
            return False
    
    async def _handle_realtime_only(self) -> bool:
        """Start real-time processing services"""
        logger.info("⚡ Starting real-time processing services...")
        
        try:
            # Start data-recovery in continuous mode
            logger.info("🔄 Starting continuous data-recovery service...")
            if not await self._run_docker_service("data-recovery", mode="continuous", detached=True):
                logger.warning("⚠️  Data-recovery service startup had issues")
            
            # Start vectorbt in real-time mode
            logger.info("📊 Starting real-time vectorbt service...")
            if not await self._run_docker_service("vectorbt", mode="realtime", detached=True):
                logger.error("❌ VectorBT real-time service failed to start")
                return False
            
            logger.info("✅ Real-time services started successfully")
            
            # Monitor for a short while to ensure stability
            await self._monitor_realtime_services(duration_minutes=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Real-time services startup failed: {e}")
            return False
    
    async def _run_vectorbt_incremental(self, symbol: str, start_time: datetime, end_time: datetime) -> bool:
        """Run VectorBT processing for a specific symbol and time range"""
        
        try:
            # Run VectorBT container with specific parameters using docker compose
            cmd = [
                'docker', 'compose', '-f', '/workspace/docker-compose.dev.yml',
                'run', '--rm',
                '-e', f'SERVICE_MODE=incremental',
                '-e', f'TARGET_SYMBOL={symbol}',
                '-e', f'START_TIME={start_time.isoformat()}',
                '-e', f'END_TIME={end_time.isoformat()}',
                'vectorbt-bulk'  # Use the bulk service but with incremental mode
            ]
            
            logger.debug(f"Running VectorBT incremental: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/workspace"
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.debug(f"VectorBT incremental processing output: {stdout.decode()}")
                return True
            else:
                logger.error(f"VectorBT incremental processing failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run incremental vectorbt processing: {e}")
            return False
    
    async def _run_docker_service(self, service_name: str, mode: Optional[str] = None, detached: bool = False) -> bool:
        """Run a Docker service with specified parameters"""
        
        try:
            # For services running inside Docker, we use docker compose commands
            cmd = ['docker', 'compose', '-f', '/workspace/docker-compose.dev.yml']
            
            if service_name in ['data-recovery', 'vectorbt-bulk', 'gap-recovery']:
                if 'bulk' in service_name:
                    cmd.extend(['--profile', 'bulk-indicators'])
                else:
                    cmd.extend(['--profile', 'realtime'])
            
            if detached:
                cmd.extend(['up', '-d', service_name])
            else:
                cmd.extend(['run', '--rm', service_name])
            
            env = os.environ.copy()
            if mode:
                env['SERVICE_MODE'] = mode
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/workspace",
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.debug(f"Service {service_name} output: {stdout.decode()}")
                return True
            else:
                logger.error(f"Service {service_name} failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run docker service {service_name}: {e}")
            return False
    
    async def _run_script(self, script_name: str, args: Optional[List[str]] = None) -> bool:
        """Run a shell script from the scripts directory"""
        
        try:
            script_path = self.project_root / "scripts" / script_name
            if not script_path.exists():
                logger.error(f"Script not found: {script_path}")
                return False
            
            cmd = [str(script_path)]
            if args:
                cmd.extend(args)
            
            logger.debug(f"Running script: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/workspace"
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.debug(f"Script {script_name} output: {stdout.decode()}")
                return True
            else:
                logger.error(f"Script {script_name} failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run script {script_name}: {e}")
            return False
    
    def _find_latest_backup(self) -> Optional[Path]:
        """Find the latest backup file"""
        
        try:
            if not self.backup_dir.exists():
                return None
            
            backup_files = list(self.backup_dir.glob("*.sql.gz"))
            if not backup_files:
                backup_files = list(self.backup_dir.glob("*.sql"))
            
            if not backup_files:
                return None
            
            # Sort by modification time, newest first
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return backup_files[0]
            
        except Exception as e:
            logger.error(f"Failed to find latest backup: {e}")
            return None
    
    async def _monitor_realtime_services(self, duration_minutes: int = 5):
        """Monitor real-time services to ensure they're running properly"""
        
        logger.info(f"👀 Monitoring real-time services for {duration_minutes} minutes...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time and not self._shutdown:
            try:
                # Check service health
                stats = await self.db_manager.get_processing_stats()
                
                # Log current status
                total_1m = stats['timeframes'].get('1min', {}).get('total_records', 0)
                total_5m = stats['timeframes'].get('5min', {}).get('total_records', 0)
                
                logger.info(f"📊 Current status: {total_1m:,} 1m records, {total_5m:,} 5m records")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
                await asyncio.sleep(30)
        
        logger.info("✅ Real-time services monitoring completed")
    
    async def _log_health_status(self, health: Dict[str, Any]):
        """Log detailed health status"""
        
        logger.info("🏥 Database Health Status:")
        logger.info(f"   • Healthy: {health['healthy']}")
        logger.info(f"   • Total 1m records: {health.get('total_1m_records', 0):,}")
        logger.info(f"   • Symbols: {health.get('symbols_count', 0)}")
        logger.info(f"   • Data range: {health.get('earliest_data')} to {health.get('latest_data')}")
        logger.info(f"   • Processing recommendation: {health.get('processing_recommendation', 'unknown')}")
        
        if health.get('gap_periods'):
            logger.info(f"   • Detected gaps: {len(health['gap_periods'])} periods")
            for gap in health['gap_periods'][:3]:  # Show first 3 gaps
                logger.info(f"     - {gap['timeframe']}: {gap['gap_size_hours']:.1f}h "
                           f"({gap['coverage_ratio']*100:.1f}% coverage)")
        
        if health.get('total_gap_hours', 0) > 0:
            logger.info(f"   • Total gap time: {health['total_gap_hours']:.1f} hours")
    
    async def _log_investigation_info(self):
        """Log information for manual investigation"""
        
        logger.info("🔍 Manual investigation recommended:")
        
        try:
            stats = await self.db_manager.get_processing_stats()
            
            logger.info("📊 Current system state:")
            for tf, data in stats['timeframes'].items():
                logger.info(f"   • {tf}: {data.get('total_records', 0):,} records, "
                           f"{data.get('indicator_coverage_pct', 0):.1f}% indicators")
            
            if stats.get('recommendations'):
                logger.info("💡 Recommendations:")
                for rec in stats['recommendations']:
                    logger.info(f"   • {rec['message']} -> {rec['action']}")
                    
        except Exception as e:
            logger.error(f"Failed to gather investigation info: {e}")
    
    async def _log_final_status(self):
        """Log final pipeline status"""
        
        logger.info("🎊 Final Pipeline Status:")
        
        try:
            health = await self.db_manager.check_database_health()
            stats = await self.db_manager.get_processing_stats()
            
            logger.info(f"   • Database health: {'✅ Healthy' if health['healthy'] else '❌ Issues detected'}")
            logger.info(f"   • Total 1m records: {health.get('total_1m_records', 0):,}")
            logger.info(f"   • Active symbols: {len(stats.get('symbols', []))}")
            
            # Show processing coverage
            for tf in ['5min', '15min', '1hour']:
                progress = stats.get('processing_progress', {}).get(tf, {})
                pct = progress.get('progress_pct', 0)
                logger.info(f"   • {tf} processing: {pct:.1f}% complete")
            
            logger.info("🚀 System is ready for real-time processing!")
            
        except Exception as e:
            logger.error(f"Failed to get final status: {e}")

async def main():
    """Main entry point for the pipeline orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline Orchestrator for Crypto Quant MVP')
    parser.add_argument('--mode', choices=['analyze', 'full'], default='analyze',
                       help='Run mode: analyze (health check only) or full (attempt full orchestration)')
    parser.add_argument('--output-strategy', action='store_true',
                       help='Output strategy in JSON format for shell script consumption')
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        if args.mode == 'analyze' or args.output_strategy:
            logger.info("🔍 Running in analysis mode - health check and strategy determination only")
            
            # Wait for database to be ready
            if not await orchestrator.db_manager.wait_for_database_ready(timeout=60):
                logger.error("❌ Database not ready")
                sys.exit(1)
                
            # Perform health check
            health = await orchestrator.db_manager.check_database_health()
            await orchestrator._log_health_status(health)
            
            # Determine strategy
            await orchestrator._execute_processing_strategy(health)
            
            logger.info("✅ Analysis completed")
            
        else:
            # Try to run full pipeline (will likely fail due to DinD issue)
            logger.warning("⚠️  Full orchestration mode has Docker-in-Docker limitations")
            success = await orchestrator.run_automated_pipeline()
            sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
