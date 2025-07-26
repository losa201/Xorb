#!/usr/bin/env python3
"""
Xorb Advanced Backup System
Comprehensive backup solution with Restic + Backblaze B2 and lifecycle policies
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile

import asyncpg
import aiohttp
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
backup_operations = Counter('xorb_backup_operations_total', 'Backup operations', ['type', 'status'])
backup_duration = Histogram('xorb_backup_duration_seconds', 'Backup operation duration', ['type'])
backup_size = Gauge('xorb_backup_size_bytes', 'Backup size in bytes', ['type', 'repository'])
backup_retention = Gauge('xorb_backup_retention_days', 'Backup retention in days', ['policy'])

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DATABASE = "database"
    FILES = "files"
    CONFIG = "config"

class BackupStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"

@dataclass
class BackupPolicy:
    """Backup retention policy"""
    name: str
    keep_daily: int
    keep_weekly: int
    keep_monthly: int
    keep_yearly: int
    prune_interval_hours: int = 24

@dataclass
class BackupJob:
    """Backup job configuration"""
    id: str
    name: str
    type: BackupType
    source_paths: List[str]
    repository: str
    schedule_cron: str
    retention_policy: BackupPolicy
    enabled: bool
    encryption_key: str
    pre_backup_script: Optional[str] = None
    post_backup_script: Optional[str] = None

@dataclass
class BackupResult:
    """Backup operation result"""
    job_id: str
    snapshot_id: str
    status: BackupStatus
    start_time: datetime
    end_time: datetime
    files_new: int
    files_changed: int
    files_unmodified: int
    dirs_new: int
    dirs_changed: int
    dirs_unmodified: int
    data_added: int
    total_files_processed: int
    total_bytes_processed: int
    repository: str
    error_message: Optional[str] = None

class AdvancedBackupSystem:
    """Advanced backup system with Restic and B2 integration"""
    
    def __init__(self):
        self.db_pool = None
        self.backup_jobs = {}
        self.active_jobs = {}
        
        # B2 Configuration
        self.b2_account_id = os.getenv("B2_ACCOUNT_ID")
        self.b2_application_key = os.getenv("B2_APPLICATION_KEY")
        self.b2_bucket_name = os.getenv("B2_BUCKET_NAME", "xorb-backup")
        
        # Backup repositories
        self.repositories = {
            "primary": f"b2:{self.b2_bucket_name}:primary",
            "database": f"b2:{self.b2_bucket_name}:database",
            "files": f"b2:{self.b2_bucket_name}:files",
            "config": f"b2:{self.b2_bucket_name}:config"
        }
        
        # Default backup policies
        self.backup_policies = {
            "critical": BackupPolicy(
                name="critical",
                keep_daily=30,
                keep_weekly=12,
                keep_monthly=12,
                keep_yearly=5,
                prune_interval_hours=6
            ),
            "standard": BackupPolicy(
                name="standard",
                keep_daily=14,
                keep_weekly=8,
                keep_monthly=6,
                keep_yearly=2,
                prune_interval_hours=24
            ),
            "archive": BackupPolicy(
                name="archive",
                keep_daily=7,
                keep_weekly=4,
                keep_monthly=12,
                keep_yearly=10,
                prune_interval_hours=48
            )
        }
    
    async def initialize(self):
        """Initialize the backup system"""
        logger.info("Initializing Advanced Backup System...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
        # Create database tables
        await self.create_database_tables()
        
        # Initialize restic repositories
        await self.initialize_repositories()
        
        # Load backup jobs
        await self.load_backup_jobs()
        
        # Start scheduler
        asyncio.create_task(self.backup_scheduler())
        
        logger.info("Advanced Backup System initialized")
    
    async def create_database_tables(self):
        """Create backup system database tables"""
        async with self.db_pool.acquire() as conn:
            # Backup jobs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_jobs (
                    id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    source_paths JSONB NOT NULL,
                    repository VARCHAR(200) NOT NULL,
                    schedule_cron VARCHAR(100) NOT NULL,
                    retention_policy JSONB NOT NULL,
                    enabled BOOLEAN DEFAULT true,
                    encryption_key VARCHAR(200),
                    pre_backup_script TEXT,
                    post_backup_script TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Backup results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_results (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL,
                    snapshot_id VARCHAR(100),
                    status VARCHAR(50) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    files_new INTEGER DEFAULT 0,
                    files_changed INTEGER DEFAULT 0,
                    files_unmodified INTEGER DEFAULT 0,
                    dirs_new INTEGER DEFAULT 0,
                    dirs_changed INTEGER DEFAULT 0,
                    dirs_unmodified INTEGER DEFAULT 0,
                    data_added BIGINT DEFAULT 0,
                    total_files_processed INTEGER DEFAULT 0,
                    total_bytes_processed BIGINT DEFAULT 0,
                    repository VARCHAR(200),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_backup_results_job ON backup_results(job_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_backup_results_status ON backup_results(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_backup_results_time ON backup_results(start_time)")
    
    async def initialize_repositories(self):
        """Initialize Restic repositories"""
        for repo_name, repo_path in self.repositories.items():
            try:
                # Set environment variables for B2
                env = os.environ.copy()
                env.update({
                    'B2_ACCOUNT_ID': self.b2_account_id,
                    'B2_ACCOUNT_KEY': self.b2_application_key,
                    'RESTIC_REPOSITORY': repo_path,
                    'RESTIC_PASSWORD': os.getenv(f"RESTIC_PASSWORD_{repo_name.upper()}", "xorb_backup_2024")
                })
                
                # Check if repository exists
                result = subprocess.run(
                    ['restic', 'snapshots', '--json'],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    # Initialize repository
                    logger.info(f"Initializing repository: {repo_name}")
                    
                    init_result = subprocess.run(
                        ['restic', 'init'],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    if init_result.returncode == 0:
                        logger.info(f"Repository {repo_name} initialized successfully")
                    else:
                        logger.error(f"Failed to initialize repository {repo_name}: {init_result.stderr}")
                else:
                    logger.info(f"Repository {repo_name} already exists")
                    
            except Exception as e:
                logger.error(f"Failed to initialize repository {repo_name}: {e}")
    
    async def load_backup_jobs(self):
        """Load backup jobs from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM backup_jobs WHERE enabled = true")
                
                for row in rows:
                    job = BackupJob(
                        id=row['id'],
                        name=row['name'],
                        type=BackupType(row['type']),
                        source_paths=row['source_paths'],
                        repository=row['repository'],
                        schedule_cron=row['schedule_cron'],
                        retention_policy=BackupPolicy(**row['retention_policy']),
                        enabled=row['enabled'],
                        encryption_key=row['encryption_key'],
                        pre_backup_script=row['pre_backup_script'],
                        post_backup_script=row['post_backup_script']
                    )
                    
                    self.backup_jobs[job.id] = job
                
                # Create default jobs if none exist
                if not self.backup_jobs:
                    await self.create_default_backup_jobs()
                
                logger.info(f"Loaded {len(self.backup_jobs)} backup jobs")
                
        except Exception as e:
            logger.error(f"Failed to load backup jobs: {e}")
    
    async def create_default_backup_jobs(self):
        """Create default backup jobs"""
        
        default_jobs = [
            # Database backup job
            BackupJob(
                id="database_daily",
                name="Daily Database Backup",
                type=BackupType.DATABASE,
                source_paths=["/var/lib/postgresql/data"],
                repository=self.repositories["database"],
                schedule_cron="0 2 * * *",  # Daily at 2 AM
                retention_policy=self.backup_policies["critical"],
                enabled=True,
                encryption_key="database_key",
                pre_backup_script="pg_dumpall > /tmp/db_dump.sql",
                post_backup_script="rm -f /tmp/db_dump.sql"
            ),
            
            # Application files backup
            BackupJob(
                id="files_daily",
                name="Daily Files Backup",
                type=BackupType.FILES,
                source_paths=["/opt/xorb", "/data", "/logs"],
                repository=self.repositories["files"],
                schedule_cron="0 3 * * *",  # Daily at 3 AM
                retention_policy=self.backup_policies["standard"],
                enabled=True,
                encryption_key="files_key"
            ),
            
            # Configuration backup
            BackupJob(
                id="config_weekly",
                name="Weekly Configuration Backup",
                type=BackupType.CONFIG,
                source_paths=["/etc", "/root/.ssh", "/opt/xorb/config"],
                repository=self.repositories["config"],
                schedule_cron="0 4 * * 0",  # Weekly on Sunday at 4 AM
                retention_policy=self.backup_policies["archive"],
                enabled=True,
                encryption_key="config_key"
            )
        ]
        
        for job in default_jobs:
            await self.create_backup_job(job)
    
    async def create_backup_job(self, job: BackupJob):
        """Create a new backup job"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO backup_jobs (
                        id, name, type, source_paths, repository, schedule_cron,
                        retention_policy, enabled, encryption_key, pre_backup_script, post_backup_script
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        type = EXCLUDED.type,
                        source_paths = EXCLUDED.source_paths,
                        repository = EXCLUDED.repository,
                        schedule_cron = EXCLUDED.schedule_cron,
                        retention_policy = EXCLUDED.retention_policy,
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                """,
                job.id, job.name, job.type.value, json.dumps(job.source_paths),
                job.repository, job.schedule_cron, json.dumps(asdict(job.retention_policy)),
                job.enabled, job.encryption_key, job.pre_backup_script, job.post_backup_script)
            
            self.backup_jobs[job.id] = job
            logger.info(f"Created backup job: {job.name}")
            
        except Exception as e:
            logger.error(f"Failed to create backup job {job.id}: {e}")
    
    async def execute_backup_job(self, job_id: str) -> BackupResult:
        """Execute a backup job"""
        
        if job_id not in self.backup_jobs:
            raise ValueError(f"Backup job {job_id} not found")
        
        job = self.backup_jobs[job_id]
        start_time = datetime.now()
        
        # Check if job is already running
        if job_id in self.active_jobs:
            logger.warning(f"Backup job {job_id} is already running")
            return BackupResult(
                job_id=job_id,
                snapshot_id="",
                status=BackupStatus.FAILED,
                start_time=start_time,
                end_time=start_time,
                files_new=0, files_changed=0, files_unmodified=0,
                dirs_new=0, dirs_changed=0, dirs_unmodified=0,
                data_added=0, total_files_processed=0, total_bytes_processed=0,
                repository=job.repository,
                error_message="Job already running"
            )
        
        self.active_jobs[job_id] = True
        
        try:
            logger.info(f"Starting backup job: {job.name}")
            
            with backup_duration.labels(type=job.type.value).time():
                # Execute pre-backup script
                if job.pre_backup_script:
                    await self.execute_script(job.pre_backup_script, "pre-backup")
                
                # Perform backup
                result = await self.perform_restic_backup(job)
                
                # Execute post-backup script
                if job.post_backup_script:
                    await self.execute_script(job.post_backup_script, "post-backup")
                
                # Update metrics
                backup_operations.labels(type=job.type.value, status=result.status.value).inc()
                
                if result.status == BackupStatus.SUCCESS:
                    backup_size.labels(type=job.type.value, repository=job.repository).set(result.data_added)
                
                # Store result in database
                await self.store_backup_result(result)
                
                # Prune old backups if needed
                await self.prune_backups(job)
                
                logger.info(f"Backup job {job.name} completed: {result.status.value}")
                return result
                
        except Exception as e:
            logger.error(f"Backup job {job_id} failed: {e}")
            
            result = BackupResult(
                job_id=job_id,
                snapshot_id="",
                status=BackupStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                files_new=0, files_changed=0, files_unmodified=0,
                dirs_new=0, dirs_changed=0, dirs_unmodified=0,
                data_added=0, total_files_processed=0, total_bytes_processed=0,
                repository=job.repository,
                error_message=str(e)
            )
            
            await self.store_backup_result(result)
            backup_operations.labels(type=job.type.value, status="failed").inc()
            return result
            
        finally:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def perform_restic_backup(self, job: BackupJob) -> BackupResult:
        """Perform the actual restic backup"""
        
        # Prepare environment
        env = os.environ.copy()
        env.update({
            'B2_ACCOUNT_ID': self.b2_account_id,
            'B2_ACCOUNT_KEY': self.b2_application_key,
            'RESTIC_REPOSITORY': job.repository,
            'RESTIC_PASSWORD': os.getenv(f"RESTIC_PASSWORD_{job.encryption_key.upper()}", "xorb_backup_2024")
        })
        
        # Build restic command
        cmd = ['restic', 'backup', '--json'] + job.source_paths
        
        # Add tags
        cmd.extend(['--tag', f'job:{job.id}', '--tag', f'type:{job.type.value}'])
        
        try:
            # Execute backup
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse JSON output
                lines = stdout.decode().strip().split('\n')
                summary_line = lines[-1]  # Last line contains summary
                
                try:
                    summary = json.loads(summary_line)
                    
                    return BackupResult(
                        job_id=job.id,
                        snapshot_id=summary.get('snapshot_id', ''),
                        status=BackupStatus.SUCCESS,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        files_new=summary.get('files_new', 0),
                        files_changed=summary.get('files_changed', 0),
                        files_unmodified=summary.get('files_unmodified', 0),
                        dirs_new=summary.get('dirs_new', 0),
                        dirs_changed=summary.get('dirs_changed', 0),
                        dirs_unmodified=summary.get('dirs_unmodified', 0),
                        data_added=summary.get('data_added', 0),
                        total_files_processed=summary.get('total_files_processed', 0),
                        total_bytes_processed=summary.get('total_bytes_processed', 0),
                        repository=job.repository
                    )
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse restic output: {summary_line}")
                    
                    return BackupResult(
                        job_id=job.id,
                        snapshot_id="unknown",
                        status=BackupStatus.SUCCESS,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        files_new=0, files_changed=0, files_unmodified=0,
                        dirs_new=0, dirs_changed=0, dirs_unmodified=0,
                        data_added=0, total_files_processed=0, total_bytes_processed=0,
                        repository=job.repository
                    )
            else:
                raise Exception(f"Restic backup failed: {stderr.decode()}")
                
        except Exception as e:
            raise Exception(f"Backup execution failed: {e}")
    
    async def execute_script(self, script: str, script_type: str):
        """Execute pre/post backup script"""
        try:
            logger.info(f"Executing {script_type} script")
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(f"#!/bin/bash\nset -e\n{script}\n")
                script_path = f.name
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            # Execute
            process = await asyncio.create_subprocess_exec(
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"{script_type} script failed: {stderr.decode()}")
            else:
                logger.info(f"{script_type} script completed successfully")
            
            # Cleanup
            os.unlink(script_path)
            
        except Exception as e:
            logger.error(f"Failed to execute {script_type} script: {e}")
    
    async def prune_backups(self, job: BackupJob):
        """Prune old backups according to retention policy"""
        
        try:
            policy = job.retention_policy
            
            # Prepare environment
            env = os.environ.copy()
            env.update({
                'B2_ACCOUNT_ID': self.b2_account_id,
                'B2_ACCOUNT_KEY': self.b2_application_key,
                'RESTIC_REPOSITORY': job.repository,
                'RESTIC_PASSWORD': os.getenv(f"RESTIC_PASSWORD_{job.encryption_key.upper()}", "xorb_backup_2024")
            })
            
            # Build forget command
            cmd = [
                'restic', 'forget',
                '--keep-daily', str(policy.keep_daily),
                '--keep-weekly', str(policy.keep_weekly),
                '--keep-monthly', str(policy.keep_monthly),
                '--keep-yearly', str(policy.keep_yearly),
                '--tag', f'job:{job.id}',
                '--prune'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Pruned old backups for job {job.id}")
                
                # Update retention metrics
                backup_retention.labels(policy=policy.name).set(policy.keep_daily)
            else:
                logger.error(f"Failed to prune backups for job {job.id}: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Backup pruning failed for job {job.id}: {e}")
    
    async def store_backup_result(self, result: BackupResult):
        """Store backup result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO backup_results (
                        job_id, snapshot_id, status, start_time, end_time,
                        files_new, files_changed, files_unmodified,
                        dirs_new, dirs_changed, dirs_unmodified,
                        data_added, total_files_processed, total_bytes_processed,
                        repository, error_message
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                result.job_id, result.snapshot_id, result.status.value,
                result.start_time, result.end_time,
                result.files_new, result.files_changed, result.files_unmodified,
                result.dirs_new, result.dirs_changed, result.dirs_unmodified,
                result.data_added, result.total_files_processed, result.total_bytes_processed,
                result.repository, result.error_message)
                
        except Exception as e:
            logger.error(f"Failed to store backup result: {e}")
    
    async def backup_scheduler(self):
        """Backup job scheduler"""
        logger.info("Starting backup scheduler...")
        
        while True:
            try:
                current_time = datetime.now()
                
                for job_id, job in self.backup_jobs.items():
                    if not job.enabled:
                        continue
                    
                    # Simple cron-like scheduling (this would be more sophisticated in production)
                    if await self.should_run_backup(job, current_time):
                        logger.info(f"Scheduling backup job: {job.name}")
                        asyncio.create_task(self.execute_backup_job(job_id))
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def should_run_backup(self, job: BackupJob, current_time: datetime) -> bool:
        """Check if backup job should run based on schedule"""
        
        try:
            # Get last successful backup
            async with self.db_pool.acquire() as conn:
                last_backup = await conn.fetchrow("""
                    SELECT start_time FROM backup_results
                    WHERE job_id = $1 AND status = 'success'
                    ORDER BY start_time DESC LIMIT 1
                """, job.id)
            
            # Simple daily backup check (would implement proper cron parsing in production)
            if job.schedule_cron.startswith("0 2"):  # Daily at 2 AM
                if last_backup:
                    last_time = last_backup['start_time']
                    return (current_time - last_time).days >= 1 and current_time.hour == 2
                else:
                    return current_time.hour == 2
            
            elif job.schedule_cron.startswith("0 3"):  # Daily at 3 AM
                if last_backup:
                    last_time = last_backup['start_time']
                    return (current_time - last_time).days >= 1 and current_time.hour == 3
                else:
                    return current_time.hour == 3
            
            elif job.schedule_cron.startswith("0 4 * * 0"):  # Weekly on Sunday at 4 AM
                if last_backup:
                    last_time = last_backup['start_time']
                    return ((current_time - last_time).days >= 7 and 
                           current_time.weekday() == 6 and current_time.hour == 4)
                else:
                    return current_time.weekday() == 6 and current_time.hour == 4
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking backup schedule for {job.id}: {e}")
            return False
    
    async def get_backup_status(self) -> Dict:
        """Get comprehensive backup system status"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent backup results
                recent_results = await conn.fetch("""
                    SELECT job_id, status, start_time, data_added, error_message
                    FROM backup_results
                    WHERE start_time > NOW() - INTERVAL '7 days'
                    ORDER BY start_time DESC
                    LIMIT 50
                """)
                
                # Get job statistics
                job_stats = await conn.fetch("""
                    SELECT 
                        job_id,
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'success') as successful_runs,
                        MAX(start_time) as last_run,
                        SUM(data_added) as total_data_backed_up
                    FROM backup_results
                    WHERE start_time > NOW() - INTERVAL '30 days'
                    GROUP BY job_id
                """)
            
            return {
                "total_jobs": len(self.backup_jobs),
                "active_jobs": len(self.active_jobs),
                "repositories": list(self.repositories.keys()),
                "recent_results": [dict(r) for r in recent_results],
                "job_statistics": [dict(s) for s in job_stats],
                "backup_policies": {name: asdict(policy) for name, policy in self.backup_policies.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {"error": str(e)}

async def main():
    """Main function for the backup system"""
    backup_system = AdvancedBackupSystem()
    
    try:
        await backup_system.initialize()
        logger.info("🗄️  Advanced Backup System with Restic + B2 running")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Backup system stopped by user")
    except Exception as e:
        logger.error(f"Backup system failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())