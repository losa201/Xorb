"""
Unified Backup System for XORB Platform
Redirects to the consolidated backup manager in security_utils.py
"""

# Import unified backup manager from security utils
from .security_utils import (
    backup_manager,
    UnifiedBackupManager,
    BackupResult,
    BackupOperation
)

# Export main classes for backward compatibility
__all__ = [
    'backup_manager',
    'UnifiedBackupManager', 
    'BackupResult',
    'BackupOperation'
]

# Default instance for easy access
BackupManager = UnifiedBackupManager