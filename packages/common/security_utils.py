"""
Centralized security utilities for XORB platform
Provides unified password hashing, backup utilities, and security helpers
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


# Centralized password context - single source of truth for password hashing
if PASSLIB_AVAILABLE:
    try:
        PASSWORD_CONTEXT = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__time_cost=3,       # Higher security - more iterations
            argon2__memory_cost=65536,  # 64MB memory usage
            argon2__parallelism=1,      # Single thread
            argon2__hash_len=32,        # 256-bit hash
            argon2__salt_len=16         # 128-bit salt
        )
    except Exception:
        # Fall back to bcrypt or sha256 if argon2 not available
        try:
            PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
        except Exception:
            PASSWORD_CONTEXT = None
else:
    # Fallback implementation using hashlib
    PASSWORD_CONTEXT = None


def hash_password(password: str) -> str:
    """Hash a password using the centralized context"""
    if PASSWORD_CONTEXT:
        return PASSWORD_CONTEXT.hash(password)
    else:
        # Fallback to SHA-256 with salt if passlib not available
        import secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"sha256${salt}${hashed}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using the centralized context"""
    if PASSWORD_CONTEXT:
        return PASSWORD_CONTEXT.verify(plain_password, hashed_password)
    else:
        # Fallback verification for SHA-256 with salt
        if hashed_password.startswith("sha256$"):
            try:
                _, salt, stored_hash = hashed_password.split("$", 2)
                computed_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            except ValueError:
                return False
        return False


def needs_rehash(hashed_password: str) -> bool:
    """Check if password needs rehashing due to context updates"""
    if PASSWORD_CONTEXT:
        return PASSWORD_CONTEXT.needs_update(hashed_password)
    else:
        # Simple check for fallback hashes
        return not hashed_password.startswith("sha256$")


class BackupResult(Enum):
    """Backup operation results"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BackupOperation:
    """Simplified backup operation for deduplication"""
    operation_id: str
    source_path: str
    destination_path: str
    timestamp: datetime
    result: BackupResult
    file_count: int = 0
    total_size: int = 0
    error_message: Optional[str] = None


class UnifiedBackupManager:
    """
    Unified backup manager to replace multiple backup system implementations
    Consolidates functionality from various backup classes across the codebase
    """
    
    def __init__(self, backup_root: str = "/tmp/xorb_backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.operations: List[BackupOperation] = []
        self.logger = logging.getLogger(f"{__name__}.UnifiedBackupManager")
        
        # Generate encryption key for backup encryption
        key_file = self.backup_root / ".backup_key"
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            os.chmod(key_file, 0o600)  # Restrict permissions
    
    async def create_backup(self, source_path: str, backup_name: str = None) -> BackupOperation:
        """Create a backup of the specified path"""
        operation_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if backup_name is None:
            backup_name = f"{Path(source_path).name}_{operation_id}"
        
        destination_path = self.backup_root / f"{backup_name}.tar.gz"
        
        operation = BackupOperation(
            operation_id=operation_id,
            source_path=source_path,
            destination_path=str(destination_path),
            timestamp=datetime.now(),
            result=BackupResult.FAILED
        )
        
        try:
            if not os.path.exists(source_path):
                operation.error_message = f"Source path does not exist: {source_path}"
                self.logger.error(operation.error_message)
                return operation
            
            # Create compressed backup
            with tarfile.open(destination_path, 'w:gz') as tar:
                if os.path.isfile(source_path):
                    tar.add(source_path, arcname=Path(source_path).name)
                    operation.file_count = 1
                else:
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(Path(source_path).parent)
                            tar.add(file_path, arcname=str(arcname))
                            operation.file_count += 1
            
            # Get backup size
            operation.total_size = destination_path.stat().st_size
            operation.result = BackupResult.SUCCESS
            
            self.logger.info(f"Backup created: {backup_name} ({operation.file_count} files, {operation.total_size} bytes)")
            
        except Exception as e:
            operation.error_message = str(e)
            operation.result = BackupResult.FAILED
            self.logger.error(f"Backup failed: {e}")
        
        self.operations.append(operation)
        return operation
    
    async def restore_backup(self, backup_path: str, restore_path: str) -> bool:
        """Restore a backup to the specified location"""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                backup_file = self.backup_root / backup_path
                if not backup_file.exists():
                    self.logger.error(f"Backup file not found: {backup_path}")
                    return False
            
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(path=restore_dir)
            
            self.logger.info(f"Backup restored from {backup_file} to {restore_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_file in self.backup_root.glob("*.tar.gz"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Error reading backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        
        for backup_file in self.backup_root.glob("*.tar.gz"):
            try:
                file_date = datetime.fromtimestamp(backup_file.stat().st_ctime)
                if file_date < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old backup: {backup_file.name}")
            except Exception as e:
                self.logger.warning(f"Error removing backup {backup_file}: {e}")
        
        return removed_count
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup system statistics"""
        backups = self.list_backups()
        total_size = sum(b["size"] for b in backups)
        
        recent_operations = [op for op in self.operations if op.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "oldest_backup": backups[-1]["created"] if backups else None,
            "newest_backup": backups[0]["created"] if backups else None,
            "recent_operations": len(recent_operations),
            "successful_operations": len([op for op in recent_operations if op.result == BackupResult.SUCCESS]),
            "failed_operations": len([op for op in recent_operations if op.result == BackupResult.FAILED])
        }


# Global instances for easy import
backup_manager = UnifiedBackupManager()


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    import secrets
    return secrets.token_urlsafe(length)


def hash_data(data: str, algorithm: str = "sha256") -> str:
    """Hash data using specified algorithm"""
    if algorithm == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength according to security policies"""
    issues = []
    score = 0
    
    if len(password) >= 8:
        score += 1
    else:
        issues.append("Password must be at least 8 characters long")
    
    if any(c.isupper() for c in password):
        score += 1
    else:
        issues.append("Password must contain at least one uppercase letter")
    
    if any(c.islower() for c in password):
        score += 1
    else:
        issues.append("Password must contain at least one lowercase letter")
    
    if any(c.isdigit() for c in password):
        score += 1
    else:
        issues.append("Password must contain at least one digit")
    
    special_chars = "!@#$%^&*(),.?\":{}|<>"
    if any(c in special_chars for c in password):
        score += 1
    else:
        issues.append("Password must contain at least one special character")
    
    strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
    strength = strength_levels[min(score, 4)]
    
    return {
        "score": score,
        "max_score": 5,
        "strength": strength,
        "is_valid": score >= 4,
        "issues": issues
    }