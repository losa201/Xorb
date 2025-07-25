#!/usr/bin/env python3
"""
Xorb Backup Efficiency Tuning Service
Phase 5.5 - Advanced Backup Optimization with Restic
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("xorb.backup_tuner")

# Phase 5.5 Required Metrics
last_backup_age_seconds = Gauge(
    'last_backup_age_seconds',
    'Seconds since last successful backup',
    ['backup_type', 'target']
)

restore_time_seconds = Histogram(
    'restore_time_seconds', 
    'Time to complete restore operations',
    ['backup_type', 'restore_type']
)

# Additional backup metrics
backup_size_bytes = Gauge(
    'backup_size_bytes',
    'Size of backup in bytes',
    ['backup_type', 'target', 'compression']
)

backup_duration_seconds = Histogram(
    'backup_duration_seconds',
    'Duration of backup operations',
    ['backup_type', 'target', 'compression']
)

backup_operations_total = Counter(
    'backup_operations_total',
    'Total backup operations',
    ['backup_type', 'target', 'result']
)

compression_ratio = Gauge(
    'backup_compression_ratio',
    'Backup compression ratio (original/compressed)',
    ['backup_type', 'compression_method']
)

@dataclass
class BackupConfig:
    """Backup configuration"""
    name: str
    source_path: str
    repository: str
    compression: str  # 'auto', 'max', 'fast', 'off'
    retention_policy: Dict[str, int]  # {'daily': 7, 'weekly': 4, 'monthly': 12}
    schedule_cron: str
    exclude_patterns: List[str]
    pre_backup_script: Optional[str] = None
    post_backup_script: Optional[str] = None

@dataclass
class BackupResult:
    """Backup operation result"""
    config_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    files_processed: int
    error_message: Optional[str] = None
    snapshot_id: Optional[str] = None

class BackupEfficiencyTuner:
    """Advanced backup management with efficiency optimization"""
    
    def __init__(self):
        self.db_pool = None
        
        # Backup configurations
        self.configs = {
            'postgresql': BackupConfig(
                name='postgresql',
                source_path='/var/lib/postgresql/data',
                repository=os.getenv('BACKUP_REPO_POSTGRES', 's3:s3.amazonaws.com/xorb-backups/postgres'),
                compression='max',
                retention_policy={'daily': 7, 'weekly': 4, 'monthly': 12, 'yearly': 2},
                schedule_cron='0 2 * * *',  # Daily at 2 AM
                exclude_patterns=['*.log', 'pg_log/*', 'pg_stat_tmp/*'],
                pre_backup_script='/scripts/pg_backup_prepare.sh',
                post_backup_script='/scripts/pg_backup_cleanup.sh'
            ),
            'redis': BackupConfig(
                name='redis',
                source_path='/var/lib/redis',
                repository=os.getenv('BACKUP_REPO_REDIS', 's3:s3.amazonaws.com/xorb-backups/redis'),
                compression='fast',
                retention_policy={'daily': 7, 'weekly': 4},
                schedule_cron='0 3 * * *',  # Daily at 3 AM
                exclude_patterns=['*.aof.manifest', 'temp-*'],
                pre_backup_script='/scripts/redis_backup_prepare.sh'
            ),
            'application_data': BackupConfig(
                name='application_data',
                source_path='/app/data',
                repository=os.getenv('BACKUP_REPO_APP', 's3:s3.amazonaws.com/xorb-backups/app-data'),
                compression='auto',
                retention_policy={'daily': 14, 'weekly': 8, 'monthly': 6},
                schedule_cron='0 4 * * *',  # Daily at 4 AM
                exclude_patterns=['*.tmp', '*.cache', 'logs/*.log']
            )
        }
        
        # Restic environment
        self.restic_env = {
            'RESTIC_PASSWORD': os.getenv('RESTIC_PASSWORD', 'xorb-backup-key-2024'),
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'RESTIC_CACHE_DIR': '/tmp/restic-cache'
        }
        
        # Alert thresholds (Phase 5.5 requirements)
        self.alert_thresholds = {
            'last_backup_age_seconds': 86400,  # 24 hours
            'restore_time_seconds': 300       # 5 minutes
        }
        
    async def initialize(self):
        """Initialize backup tuner service"""
        logger.info("Initializing Backup Efficiency Tuner...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
        # Create database tables
        await self._create_backup_tables()
        
        # Initialize restic repositories
        await self._initialize_repositories()
        
        # Start background tasks
        asyncio.create_task(self._backup_scheduler())
        asyncio.create_task(self._prune_old_wal_segments())
        asyncio.create_task(self._weekly_restore_tests())
        
        logger.info("Backup Efficiency Tuner initialized successfully")
    
    async def _create_backup_tables(self):
        """Create backup tracking tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_operations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    config_name VARCHAR(100) NOT NULL,
                    operation_type VARCHAR(20) NOT NULL, -- 'backup', 'restore', 'prune'
                    success BOOLEAN NOT NULL,
                    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    duration_seconds FLOAT NOT NULL,
                    original_size_bytes BIGINT DEFAULT 0,
                    compressed_size_bytes BIGINT DEFAULT 0,
                    compression_ratio FLOAT DEFAULT 0,
                    files_processed INTEGER DEFAULT 0,
                    snapshot_id VARCHAR(255),
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_backup_ops_config_time 
                ON backup_operations(config_name, start_time);
                
                CREATE INDEX IF NOT EXISTS idx_backup_ops_type_success 
                ON backup_operations(operation_type, success);
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS restore_tests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    config_name VARCHAR(100) NOT NULL,
                    snapshot_id VARCHAR(255) NOT NULL,
                    test_type VARCHAR(50) NOT NULL, -- 'automated_weekly', 'manual', 'disaster_recovery'
                    success BOOLEAN NOT NULL,
                    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    duration_seconds FLOAT NOT NULL,
                    rto_seconds FLOAT, -- Recovery Time Objective
                    rpo_seconds FLOAT, -- Recovery Point Objective  
                    data_integrity_verified BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_restore_tests_config_time
                ON restore_tests(config_name, start_time);
            """)
    
    async def _initialize_repositories(self):
        """Initialize restic repositories if they don't exist"""
        for config in self.configs.values():
            try:
                # Check if repository exists
                cmd = ['restic', '-r', config.repository, 'snapshots', '--json']
                result = await self._run_restic_command(cmd, check_return=False)
                
                if result.returncode != 0:
                    # Repository doesn't exist, initialize it
                    logger.info(f"Initializing restic repository for {config.name}")
                    cmd = ['restic', '-r', config.repository, 'init']
                    await self._run_restic_command(cmd)
                    logger.info(f"Repository initialized: {config.name}")
                else:
                    logger.info(f"Repository already exists: {config.name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize repository {config.name}", error=str(e))
    
    async def _run_restic_command(self, cmd: List[str], check_return: bool = True) -> subprocess.CompletedProcess:
        """Run restic command with proper environment"""
        env = {**os.environ, **self.restic_env}
        
        logger.debug("Running restic command", cmd=" ".join(cmd))
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
        
        if check_return and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        
        return result
    
    async def perform_backup(self, config_name: str) -> BackupResult:
        """Perform backup with maximum compression"""
        config = self.configs[config_name]
        start_time = datetime.now()
        
        logger.info(f"Starting backup for {config_name}")
        
        try:
            # Run pre-backup script if configured
            if config.pre_backup_script:
                await self._run_script(config.pre_backup_script)
            
            # Build restic backup command with max compression
            cmd = [
                'restic', '-r', config.repository, 'backup',
                config.source_path,
                '--compression', config.compression,
                '--tag', f'automated-{datetime.now().strftime("%Y%m%d")}',
                '--json'
            ]
            
            # Add exclude patterns
            for pattern in config.exclude_patterns:
                cmd.extend(['--exclude', pattern])
            
            with backup_duration_seconds.labels(
                backup_type=config_name, 
                target=config.source_path,
                compression=config.compression
            ).time():
                result = await self._run_restic_command(cmd)
            
            # Parse JSON output for metrics
            output_lines = result.stdout.decode().strip().split('\n')
            backup_summary = None
            snapshot_id = None
            
            for line in output_lines:
                try:
                    data = json.loads(line)
                    if data.get('message_type') == 'summary':
                        backup_summary = data
                    elif data.get('message_type') == 'status' and data.get('snapshot_id'):
                        snapshot_id = data['snapshot_id']
                except json.JSONDecodeError:
                    continue
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract metrics from summary
            original_size = backup_summary.get('total_bytes_processed', 0) if backup_summary else 0
            compressed_size = backup_summary.get('data_added', 0) if backup_summary else 0
            files_processed = backup_summary.get('total_files_processed', 0) if backup_summary else 0
            
            compression_ratio_value = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Update metrics
            backup_size_bytes.labels(
                backup_type=config_name,
                target=config.source_path,
                compression=config.compression
            ).set(compressed_size)
            
            compression_ratio.labels(
                backup_type=config_name,
                compression_method=config.compression
            ).set(compression_ratio_value)
            
            backup_operations_total.labels(
                backup_type=config_name,
                target=config.source_path,
                result='success'
            ).inc()
            
            last_backup_age_seconds.labels(
                backup_type=config_name,
                target=config.source_path
            ).set(0)  # Just completed
            
            # Run post-backup script if configured
            if config.post_backup_script:
                await self._run_script(config.post_backup_script)
            
            backup_result = BackupResult(
                config_name=config_name,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio_value,
                files_processed=files_processed,
                snapshot_id=snapshot_id
            )
            
            # Store result in database
            await self._store_backup_result(backup_result)
            
            logger.info(f"Backup completed successfully for {config_name}",
                       duration=duration,
                       compression_ratio=compression_ratio_value,
                       original_mb=original_size / 1024 / 1024,
                       compressed_mb=compressed_size / 1024 / 1024)
            
            return backup_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            backup_operations_total.labels(
                backup_type=config_name,
                target=config.source_path,
                result='error'
            ).inc()
            
            backup_result = BackupResult(
                config_name=config_name,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                original_size_bytes=0,
                compressed_size_bytes=0,
                compression_ratio=0,
                files_processed=0,
                error_message=str(e)
            )
            
            await self._store_backup_result(backup_result)
            
            logger.error(f"Backup failed for {config_name}", error=str(e))
            return backup_result
    
    async def _run_script(self, script_path: str):
        """Run pre/post backup script"""
        if os.path.exists(script_path):
            process = await asyncio.create_subprocess_exec(
                'bash', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"Script {script_path} failed",
                             returncode=process.returncode,
                             stderr=stderr.decode())
            else:
                logger.debug(f"Script {script_path} completed successfully")
    
    async def _store_backup_result(self, result: BackupResult):
        """Store backup result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO backup_operations 
                    (config_name, operation_type, success, start_time, end_time, 
                     duration_seconds, original_size_bytes, compressed_size_bytes, 
                     compression_ratio, files_processed, snapshot_id, error_message)
                    VALUES ($1, 'backup', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                result.config_name, result.success, result.start_time, result.end_time,
                result.duration_seconds, result.original_size_bytes, result.compressed_size_bytes,
                result.compression_ratio, result.files_processed, result.snapshot_id, 
                result.error_message)
                
        except Exception as e:
            logger.error("Failed to store backup result", error=str(e))
    
    async def _backup_scheduler(self):
        """Background backup scheduler"""
        while True:
            try:
                current_time = datetime.now()
                current_hour = current_time.hour
                
                # Simple scheduling based on configured hours
                for config_name, config in self.configs.items():
                    # Parse cron schedule (simplified - only check hour)
                    # Real implementation would use croniter or similar
                    schedule_hour = int(config.schedule_cron.split()[1])
                    
                    if current_hour == schedule_hour and current_time.minute < 5:
                        # Check if backup already ran today
                        if not await self._backup_ran_today(config_name):
                            logger.info(f"Starting scheduled backup for {config_name}")
                            await self.perform_backup(config_name)
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Backup scheduler error", error=str(e))
                await asyncio.sleep(60)
    
    async def _backup_ran_today(self, config_name: str) -> bool:
        """Check if backup already ran today"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM backup_operations
                    WHERE config_name = $1 AND operation_type = 'backup' 
                    AND success = true AND start_time >= $2
                """, config_name, today_start)
                
                return result['count'] > 0
                
        except Exception as e:
            logger.error("Failed to check if backup ran today", error=str(e))
            return False
    
    async def _prune_old_wal_segments(self):
        """Prune PostgreSQL WAL segments older than 48h"""
        while True:
            try:
                # Run every 6 hours
                await asyncio.sleep(21600)
                
                logger.info("Pruning old PostgreSQL WAL segments")
                
                # Find WAL files older than 48 hours
                wal_dir = "/var/lib/postgresql/data/pg_wal"
                if os.path.exists(wal_dir):
                    cutoff_time = time.time() - (48 * 3600)  # 48 hours ago
                    
                    removed_count = 0
                    for filename in os.listdir(wal_dir):
                        filepath = os.path.join(wal_dir, filename)
                        
                        # Only remove WAL files, not current or special files
                        if (filename.startswith("000000") and 
                            len(filename) == 24 and 
                            os.path.getmtime(filepath) < cutoff_time):
                            
                            try:
                                os.remove(filepath)
                                removed_count += 1
                            except OSError as e:
                                logger.warning(f"Failed to remove WAL file {filename}", error=str(e))
                    
                    logger.info(f"Pruned {removed_count} old WAL segments")
                
            except Exception as e:
                logger.error("WAL segment pruning failed", error=str(e))
    
    async def _weekly_restore_tests(self):
        """Run weekly restore tests"""
        while True:
            try:
                # Run weekly on Sunday at 1 AM
                now = datetime.now()
                if now.weekday() == 6 and now.hour == 1 and now.minute < 5:
                    
                    for config_name in self.configs.keys():
                        try:
                            await self.test_restore(config_name)
                        except Exception as e:
                            logger.error(f"Restore test failed for {config_name}", error=str(e))
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Weekly restore test scheduler error", error=str(e))
                await asyncio.sleep(3600)
    
    async def test_restore(self, config_name: str) -> Dict:
        """Test restore operation and measure RTO/RPO"""
        config = self.configs[config_name]
        start_time = datetime.now()
        
        logger.info(f"Starting restore test for {config_name}")
        
        try:
            # Get latest snapshot
            cmd = ['restic', '-r', config.repository, 'snapshots', '--json', '--latest', '1']
            result = await self._run_restic_command(cmd)
            
            snapshots = json.loads(result.stdout.decode())
            if not snapshots:
                raise Exception("No snapshots found")
            
            latest_snapshot = snapshots[0]
            snapshot_id = latest_snapshot['id']
            
            # Create temporary restore directory
            restore_dir = f"/tmp/restore_test_{config_name}_{int(time.time())}"
            os.makedirs(restore_dir, exist_ok=True)
            
            try:
                # Perform restore
                with restore_time_seconds.labels(
                    backup_type=config_name,
                    restore_type='automated_test'
                ).time():
                    cmd = [
                        'restic', '-r', config.repository, 'restore',
                        snapshot_id, '--target', restore_dir
                    ]
                    await self._run_restic_command(cmd)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Calculate RPO (time between snapshot and restore)
                snapshot_time = datetime.fromisoformat(latest_snapshot['time'].replace('Z', '+00:00'))
                rpo_seconds = (start_time - snapshot_time.replace(tzinfo=None)).total_seconds()
                
                # Basic data integrity check
                data_integrity_verified = await self._verify_restore_integrity(restore_dir, config_name)
                
                # Store test results
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO restore_tests 
                        (config_name, snapshot_id, test_type, success, start_time, end_time,
                         duration_seconds, rto_seconds, rpo_seconds, data_integrity_verified)
                        VALUES ($1, $2, 'automated_weekly', $3, $4, $5, $6, $7, $8, $9)
                    """, 
                    config_name, snapshot_id, True, start_time, end_time,
                    duration, duration, rpo_seconds, data_integrity_verified)
                
                # Update Prometheus metrics
                restore_time_seconds.labels(
                    backup_type=config_name,
                    restore_type='automated_test'
                ).observe(duration)
                
                logger.info(f"Restore test completed for {config_name}",
                           duration=duration,
                           rto_seconds=duration,
                           rpo_seconds=rpo_seconds,
                           data_integrity=data_integrity_verified)
                
                # Check alert thresholds
                if duration > self.alert_thresholds['restore_time_seconds']:
                    logger.warning(f"Restore time exceeded threshold for {config_name}",
                                 duration=duration,
                                 threshold=self.alert_thresholds['restore_time_seconds'])
                
                return {
                    'success': True,
                    'duration_seconds': duration,
                    'rto_seconds': duration,
                    'rpo_seconds': rpo_seconds,
                    'data_integrity_verified': data_integrity_verified
                }
                
            finally:
                # Cleanup restore directory
                import shutil
                shutil.rmtree(restore_dir, ignore_errors=True)
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Store failed test
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO restore_tests 
                    (config_name, snapshot_id, test_type, success, start_time, end_time,
                     duration_seconds, error_message)
                    VALUES ($1, $2, 'automated_weekly', $3, $4, $5, $6, $7)
                """, 
                config_name, '', False, start_time, end_time, duration, str(e))
            
            logger.error(f"Restore test failed for {config_name}", error=str(e))
            
            return {
                'success': False,
                'duration_seconds': duration,
                'error': str(e)
            }
    
    async def _verify_restore_integrity(self, restore_dir: str, config_name: str) -> bool:
        """Basic data integrity verification"""
        try:
            # Simple checks based on backup type
            if config_name == 'postgresql':
                # Check for key PostgreSQL files
                required_files = ['PG_VERSION', 'postgresql.conf']
                return all(os.path.exists(os.path.join(restore_dir, f)) for f in required_files)
            
            elif config_name == 'redis':
                # Check for Redis dump file
                return any(f.endswith('.rdb') for f in os.listdir(restore_dir) if os.path.isfile(os.path.join(restore_dir, f)))
            
            else:
                # General check - directory not empty
                return len(os.listdir(restore_dir)) > 0
                
        except Exception as e:
            logger.error(f"Data integrity verification failed for {config_name}", error=str(e))
            return False
    
    async def get_backup_statistics(self) -> Dict:
        """Get comprehensive backup statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Recent backup status
                recent_backups = await conn.fetch("""
                    SELECT config_name, success, start_time, duration_seconds,
                           original_size_bytes, compressed_size_bytes, compression_ratio
                    FROM backup_operations 
                    WHERE operation_type = 'backup' AND start_time >= NOW() - INTERVAL '7 days'
                    ORDER BY start_time DESC
                    LIMIT 20
                """)
                
                # Restore test results
                restore_tests = await conn.fetch("""
                    SELECT config_name, success, duration_seconds, rpo_seconds,
                           data_integrity_verified, start_time
                    FROM restore_tests
                    WHERE start_time >= NOW() - INTERVAL '30 days'
                    ORDER BY start_time DESC
                    LIMIT 10
                """)
                
                return {
                    'recent_backups': [dict(r) for r in recent_backups],
                    'restore_tests': [dict(r) for r in restore_tests],
                    'alert_thresholds': self.alert_thresholds,
                    'configurations': {name: {
                        'compression': config.compression,
                        'retention_policy': config.retention_policy,
                        'schedule_cron': config.schedule_cron
                    } for name, config in self.configs.items()},
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get backup statistics", error=str(e))
            return {'error': str(e)}

async def main():
    """Main backup tuner service"""
    # Start Prometheus metrics server
    start_http_server(8009)
    
    # Initialize backup tuner
    tuner = BackupEfficiencyTuner()
    await tuner.initialize()
    
    logger.info("🗄️ Xorb Backup Efficiency Tuner started",
               service_version="5.5",
               features=['restic_compression', 'automated_testing', 'wal_pruning'])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down backup tuner")

if __name__ == "__main__":
    asyncio.run(main())