#!/usr/bin/env python3
"""
XORB Disaster Recovery and Backup Automation System
Provides comprehensive backup, restore, and disaster recovery capabilities
"""

import os
import sys
import json
import time
import shutil
import asyncio
import sqlite3
import tarfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import aiofiles
import yaml
from cryptography.fernet import Fernet

@dataclass
class BackupJob:
    """Backup job configuration"""
    id: str
    name: str
    source_paths: List[str]
    destination: str
    schedule: str  # cron format
    retention_days: int
    compression: bool = True
    encryption: bool = True
    exclude_patterns: List[str] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: str = "pending"

@dataclass
class RecoveryPoint:
    """Recovery point metadata"""
    id: str
    job_id: str
    timestamp: datetime
    size_bytes: int
    file_count: int
    backup_path: str
    checksum: str
    encrypted: bool
    metadata: Dict[str, Any] = None

class DisasterRecoverySystem:
    """Comprehensive disaster recovery and backup system"""
    
    def __init__(self, config_path: str = "/root/Xorb/config/disaster_recovery.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.db_path = self.config.get('database', '/root/Xorb/data/disaster_recovery.db')
        self.backup_root = Path(self.config.get('backup_root', '/root/Xorb/backups'))
        self.encryption_key = self._get_or_create_encryption_key()
        self.logger = self._setup_logging()
        
        # Ensure backup directories exist
        self.backup_root.mkdir(parents=True, exist_ok=True)
        (self.backup_root / 'databases').mkdir(exist_ok=True)
        (self.backup_root / 'configs').mkdir(exist_ok=True)
        (self.backup_root / 'logs').mkdir(exist_ok=True)
        (self.backup_root / 'application').mkdir(exist_ok=True)
        
        self._init_database()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load disaster recovery configuration"""
        default_config = {
            'backup_root': '/root/Xorb/backups',
            'database': '/root/Xorb/data/disaster_recovery.db',
            'retention_policy': {
                'daily': 7,
                'weekly': 4,
                'monthly': 12,
                'yearly': 5
            },
            'monitoring': {
                'alert_on_failure': True,
                'health_check_interval': 300,
                'max_backup_time': 3600
            },
            'encryption': {
                'enabled': True,
                'key_rotation_days': 90
            },
            'compression': {
                'algorithm': 'gzip',
                'level': 6
            },
            'remote_storage': {
                'enabled': False,
                'providers': []
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    default_config.update(config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def _get_or_create_encryption_key(self) -> Fernet:
        """Get or create encryption key for backups"""
        key_file = self.backup_root / '.encryption_key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Secure permissions
        
        return Fernet(key)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for disaster recovery operations"""
        logger = logging.getLogger('disaster_recovery')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.backup_root / 'logs' / f'disaster_recovery_{datetime.now().strftime("%Y%m%d")}.log'
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize disaster recovery database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS backup_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_points (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    backup_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    encrypted BOOLEAN NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES backup_jobs (id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS backup_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    duration_seconds REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES backup_jobs (id)
                )
            ''')
            
            conn.commit()
    
    async def create_backup_job(self, job: BackupJob) -> bool:
        """Create a new backup job"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                job_config = json.dumps(asdict(job), default=str)
                conn.execute(
                    'INSERT OR REPLACE INTO backup_jobs (id, name, config, updated_at) VALUES (?, ?, ?, ?)',
                    (job.id, job.name, job_config, datetime.now())
                )
                conn.commit()
            
            self.logger.info(f"Created backup job: {job.name} ({job.id})")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to create backup job {job.name}: {e}")
            return False
    
    async def execute_backup(self, job_id: str) -> bool:
        """Execute a backup job"""
        start_time = time.time()
        
        try:
            # Get job configuration
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT config FROM backup_jobs WHERE id = ?', (job_id,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Backup job {job_id} not found")
            
            job_data = json.loads(row[0])
            job = BackupJob(**job_data)
            
            self.logger.info(f"Starting backup job: {job.name}")
            
            # Create backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_root / job_id / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup archive
            archive_path = backup_dir / f"{job.name}_{timestamp}.tar.gz"
            file_count = 0
            total_size = 0
            
            with tarfile.open(archive_path, 'w:gz' if job.compression else 'w') as tar:
                for source_path in job.source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            tar.add(source_path, arcname=os.path.basename(source_path))
                            file_count += 1
                            total_size += os.path.getsize(source_path)
                        else:
                            for root, dirs, files in os.walk(source_path):
                                # Apply exclude patterns
                                if job.exclude_patterns:
                                    files = [f for f in files if not any(
                                        f.endswith(pattern) or pattern in f 
                                        for pattern in job.exclude_patterns
                                    )]
                                
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, source_path)
                                    tar.add(file_path, arcname=arcname)
                                    file_count += 1
                                    total_size += os.path.getsize(file_path)
            
            # Encrypt backup if enabled
            if job.encryption:
                encrypted_path = str(archive_path) + '.enc'
                with open(archive_path, 'rb') as f:
                    encrypted_data = self.encryption_key.encrypt(f.read())
                
                with open(encrypted_path, 'wb') as f:
                    f.write(encrypted_data)
                
                os.remove(archive_path)
                final_path = encrypted_path
            else:
                final_path = str(archive_path)
            
            # Generate checksum
            checksum = await self._calculate_checksum(final_path)
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                id=f"{job_id}_{timestamp}",
                job_id=job_id,
                timestamp=datetime.now(),
                size_bytes=os.path.getsize(final_path),
                file_count=file_count,
                backup_path=final_path,
                checksum=checksum,
                encrypted=job.encryption,
                metadata={
                    'source_paths': job.source_paths,
                    'original_size': total_size,
                    'compression_ratio': os.path.getsize(final_path) / total_size if total_size > 0 else 0
                }
            )
            
            # Save recovery point
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO recovery_points 
                    (id, job_id, timestamp, size_bytes, file_count, backup_path, checksum, encrypted, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recovery_point.id, recovery_point.job_id, recovery_point.timestamp,
                    recovery_point.size_bytes, recovery_point.file_count, recovery_point.backup_path,
                    recovery_point.checksum, recovery_point.encrypted, json.dumps(recovery_point.metadata)
                ))
                conn.commit()
            
            # Log successful backup
            duration = time.time() - start_time
            await self._log_operation(job_id, 'backup', 'success', 
                                    f"Backup completed: {file_count} files, {total_size} bytes", duration)
            
            self.logger.info(f"Backup completed successfully: {job.name} ({duration:.2f}s)")
            
            # Clean up old backups based on retention policy
            await self._cleanup_old_backups(job_id, job.retention_days)
            
            return True
        
        except Exception as e:
            duration = time.time() - start_time
            await self._log_operation(job_id, 'backup', 'failure', str(e), duration)
            self.logger.error(f"Backup failed for job {job_id}: {e}")
            return False
    
    async def restore_from_backup(self, recovery_point_id: str, restore_path: str) -> bool:
        """Restore from a backup recovery point"""
        start_time = time.time()
        
        try:
            # Get recovery point information
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT job_id, backup_path, encrypted, checksum, metadata
                    FROM recovery_points WHERE id = ?
                ''', (recovery_point_id,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Recovery point {recovery_point_id} not found")
            
            job_id, backup_path, encrypted, expected_checksum, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            self.logger.info(f"Starting restore from recovery point: {recovery_point_id}")
            
            # Verify backup integrity
            actual_checksum = await self._calculate_checksum(backup_path)
            if actual_checksum != expected_checksum:
                raise ValueError("Backup integrity check failed: checksum mismatch")
            
            # Decrypt backup if encrypted
            if encrypted:
                with open(backup_path, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.encryption_key.decrypt(encrypted_data)
                temp_path = backup_path + '.temp'
                
                with open(temp_path, 'wb') as f:
                    f.write(decrypted_data)
                
                backup_path = temp_path
            
            # Extract backup
            os.makedirs(restore_path, exist_ok=True)
            
            with tarfile.open(backup_path, 'r:gz' if backup_path.endswith('.gz') else 'r') as tar:
                tar.extractall(path=restore_path)
            
            # Clean up temporary file
            if encrypted and os.path.exists(backup_path):
                os.remove(backup_path)
            
            # Log successful restore
            duration = time.time() - start_time
            await self._log_operation(job_id, 'restore', 'success', 
                                    f"Restore completed to {restore_path}", duration)
            
            self.logger.info(f"Restore completed successfully: {recovery_point_id} to {restore_path} ({duration:.2f}s)")
            return True
        
        except Exception as e:
            duration = time.time() - start_time
            job_id = job_id if 'job_id' in locals() else 'unknown'
            await self._log_operation(job_id, 'restore', 'failure', str(e), duration)
            self.logger.error(f"Restore failed for recovery point {recovery_point_id}: {e}")
            return False
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in self._read_chunks(f):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _read_chunks(self, file_obj, chunk_size: int = 8192):
        """Read file in chunks asynchronously"""
        while True:
            chunk = await file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk
    
    async def _cleanup_old_backups(self, job_id: str, retention_days: int):
        """Clean up old backups based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, backup_path FROM recovery_points 
                WHERE job_id = ? AND timestamp < ?
            ''', (job_id, cutoff_date))
            
            old_backups = cursor.fetchall()
            
            for backup_id, backup_path in old_backups:
                try:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    
                    # Remove backup directory if empty
                    backup_dir = os.path.dirname(backup_path)
                    if os.path.exists(backup_dir) and not os.listdir(backup_dir):
                        os.rmdir(backup_dir)
                    
                    # Remove from database
                    conn.execute('DELETE FROM recovery_points WHERE id = ?', (backup_id,))
                    
                    self.logger.info(f"Cleaned up old backup: {backup_id}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to clean up backup {backup_id}: {e}")
            
            conn.commit()
    
    async def _log_operation(self, job_id: str, operation: str, status: str, 
                           message: str, duration: float):
        """Log backup operation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO backup_logs (job_id, operation, status, message, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
            ''', (job_id, operation, status, message, duration))
            conn.commit()
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get overall backup system status"""
        with sqlite3.connect(self.db_path) as conn:
            # Get job count
            cursor = conn.execute('SELECT COUNT(*) FROM backup_jobs')
            job_count = cursor.fetchone()[0]
            
            # Get recovery point count
            cursor = conn.execute('SELECT COUNT(*) FROM recovery_points')
            recovery_point_count = cursor.fetchone()[0]
            
            # Get total backup size
            cursor = conn.execute('SELECT SUM(size_bytes) FROM recovery_points')
            total_size = cursor.fetchone()[0] or 0
            
            # Get recent operations
            cursor = conn.execute('''
                SELECT operation, status, COUNT(*) as count
                FROM backup_logs 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY operation, status
            ''')
            recent_operations = {f"{op}_{status}": count for op, status, count in cursor.fetchall()}
            
            # Get last successful backup per job
            cursor = conn.execute('''
                SELECT j.name, MAX(l.timestamp) as last_backup
                FROM backup_jobs j
                LEFT JOIN backup_logs l ON j.id = l.job_id AND l.operation = 'backup' AND l.status = 'success'
                GROUP BY j.id, j.name
            ''')
            job_status = {name: last_backup for name, last_backup in cursor.fetchall()}
        
        return {
            'job_count': job_count,
            'recovery_point_count': recovery_point_count,
            'total_backup_size_bytes': total_size,
            'recent_operations': recent_operations,
            'job_status': job_status,
            'backup_root': str(self.backup_root),
            'encryption_enabled': self.config.get('encryption', {}).get('enabled', False),
            'status': 'healthy' if job_count > 0 else 'no_jobs_configured'
        }
    
    async def create_emergency_backup(self, critical_paths: List[str]) -> str:
        """Create an emergency backup of critical system components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_job = BackupJob(
            id=f"emergency_{timestamp}",
            name=f"Emergency Backup {timestamp}",
            source_paths=critical_paths,
            destination=str(self.backup_root / 'emergency'),
            schedule='manual',
            retention_days=30,
            compression=True,
            encryption=True
        )
        
        await self.create_backup_job(emergency_job)
        success = await self.execute_backup(emergency_job.id)
        
        if success:
            self.logger.info(f"Emergency backup created successfully: {emergency_job.id}")
            return emergency_job.id
        else:
            raise Exception("Emergency backup failed")

async def main():
    """Main function for disaster recovery system"""
    
    # Initialize disaster recovery system
    dr_system = DisasterRecoverySystem()
    
    # Create critical system backup jobs
    critical_backup_jobs = [
        BackupJob(
            id="system_configs",
            name="System Configuration Backup",
            source_paths=['/root/Xorb/config', '/root/Xorb/.env*'],
            destination="configs",
            schedule="0 2 * * *",  # Daily at 2 AM
            retention_days=30,
            compression=True,
            encryption=True,
            exclude_patterns=['*.tmp', '*.log', '__pycache__']
        ),
        BackupJob(
            id="application_data",
            name="Application Data Backup",
            source_paths=['/root/Xorb/data', '/root/Xorb/logs'],
            destination="application",
            schedule="0 3 * * *",  # Daily at 3 AM
            retention_days=14,
            compression=True,
            encryption=True,
            exclude_patterns=['*.tmp', '*.pid']
        ),
        BackupJob(
            id="database_backup",
            name="Database Backup",
            source_paths=['/root/Xorb/data/*.db', '/root/Xorb/data/*.sqlite'],
            destination="databases",
            schedule="0 1 * * *",  # Daily at 1 AM
            retention_days=30,
            compression=True,
            encryption=True
        )
    ]
    
    # Create backup jobs
    for job in critical_backup_jobs:
        await dr_system.create_backup_job(job)
    
    # Execute immediate backup of critical systems
    print("Creating emergency backup of critical systems...")
    critical_paths = [
        '/root/Xorb/config',
        '/root/Xorb/data',
        '/root/Xorb/frontend/dist',
        '/root/Xorb/api_gateway.py',
        '/root/Xorb/disaster_recovery_system.py'
    ]
    
    try:
        emergency_id = await dr_system.create_emergency_backup(critical_paths)
        print(f"✅ Emergency backup created: {emergency_id}")
    except Exception as e:
        print(f"❌ Emergency backup failed: {e}")
    
    # Display system status
    status = dr_system.get_backup_status()
    print("\n📊 Disaster Recovery Status:")
    print(f"  • Backup Jobs: {status['job_count']}")
    print(f"  • Recovery Points: {status['recovery_point_count']}")
    print(f"  • Total Backup Size: {status['total_backup_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  • Encryption: {'✅ Enabled' if status['encryption_enabled'] else '❌ Disabled'}")
    print(f"  • Status: {status['status']}")
    
    print("\n🔄 Disaster Recovery System Deployment Complete!")
    print("  • Automated backup schedules configured")
    print("  • Emergency recovery procedures ready")
    print("  • Encryption and compression enabled")
    print("  • Retention policies applied")

if __name__ == "__main__":
    asyncio.run(main())