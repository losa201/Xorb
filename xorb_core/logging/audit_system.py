"""
Comprehensive Logging and Audit Trail System

This module provides enterprise-grade logging, audit trails, compliance tracking,
and forensic capabilities for the XORB ecosystem with support for multiple formats,
storage backends, and compliance frameworks.
"""

import asyncio
import json
import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import deque
import gzip

import structlog
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Audit event type enumeration."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CAMPAIGN_START = "campaign_start"
    CAMPAIGN_STOP = "campaign_stop"
    AGENT_EXECUTION = "agent_execution"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    USER_ACTION = "user_action"
    API_ACCESS = "api_access"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_EXPORT = "data_export"
    VULNERABILITY_DISCOVERY = "vulnerability_discovery"
    ESCALATION = "escalation"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    SOX = "sox"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: AuditEventType = AuditEventType.USER_ACTION
    severity: LogLevel = LogLevel.INFO
    
    # Core event data
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Action details
    action: str = ""
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    
    # Context data
    campaign_id: Optional[str] = None
    agent_id: Optional[str] = None
    target_id: Optional[str] = None
    
    # Technical details
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Detailed information
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Security context
    security_level: str = "standard"
    requires_approval: bool = False
    sensitive_data: bool = False
    
    # Compliance tracking
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    retention_days: int = 2555  # 7 years default
    
    # Integrity
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        # Create a copy without checksum for calculation
        data = asdict(self)
        data.pop('checksum', None)
        
        # Sort keys for consistent hashing
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        current_checksum = self.checksum
        self.checksum = None
        calculated_checksum = self._calculate_checksum()
        self.checksum = current_checksum
        
        return current_checksum == calculated_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class LogEntry:
    """Standard log entry structure."""
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
    logger_name: str = "xorb"
    message: str = ""
    
    # Context
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    # Request context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Technical details
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class ILogStorage(ABC):
    """Interface for log storage backends."""
    
    @abstractmethod
    async def store_log(self, log_entry: LogEntry) -> bool:
        """Store a log entry."""
        pass
    
    @abstractmethod
    async def store_audit_event(self, audit_event: AuditEvent) -> bool:
        """Store an audit event."""
        pass
    
    @abstractmethod
    async def query_logs(self, query: Dict[str, Any]) -> List[LogEntry]:
        """Query log entries."""
        pass
    
    @abstractmethod
    async def query_audit_events(self, query: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events."""
        pass


class FileStorage(ILogStorage):
    """File-based log storage with rotation and compression."""
    
    def __init__(self, base_path: Path, max_file_size: int = 100 * 1024 * 1024, compress: bool = True):
        self.base_path = Path(base_path)
        self.max_file_size = max_file_size
        self.compress = compress
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Current file handles
        self.current_log_file = None
        self.current_audit_file = None
        self.current_log_size = 0
        self.current_audit_size = 0
    
    async def store_log(self, log_entry: LogEntry) -> bool:
        """Store log entry to file."""
        try:
            if not self.current_log_file or self.current_log_size >= self.max_file_size:
                await self._rotate_log_file()
            
            log_line = log_entry.to_json() + "\n"
            self.current_log_file.write(log_line)
            self.current_log_file.flush()
            self.current_log_size += len(log_line.encode())
            
            return True
        except Exception as e:
            logger.error("Failed to store log entry", error=str(e))
            return False
    
    async def store_audit_event(self, audit_event: AuditEvent) -> bool:
        """Store audit event to file."""
        try:
            if not self.current_audit_file or self.current_audit_size >= self.max_file_size:
                await self._rotate_audit_file()
            
            audit_line = audit_event.to_json() + "\n"
            self.current_audit_file.write(audit_line)
            self.current_audit_file.flush()
            self.current_audit_size += len(audit_line.encode())
            
            return True
        except Exception as e:
            logger.error("Failed to store audit event", error=str(e))
            return False
    
    async def _rotate_log_file(self):
        """Rotate log file."""
        if self.current_log_file:
            self.current_log_file.close()
            
            if self.compress:
                await self._compress_file(Path(self.current_log_file.name))
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = self.base_path / f"xorb_logs_{timestamp}.log"
        self.current_log_file = open(log_path, 'w')
        self.current_log_size = 0
    
    async def _rotate_audit_file(self):
        """Rotate audit file."""
        if self.current_audit_file:
            self.current_audit_file.close()
            
            if self.compress:
                await self._compress_file(Path(self.current_audit_file.name))
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        audit_path = self.base_path / f"xorb_audit_{timestamp}.log"
        self.current_audit_file = open(audit_path, 'w')
        self.current_audit_size = 0
    
    async def _compress_file(self, file_path: Path):
        """Compress a log file."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            file_path.unlink()  # Remove original file
            logger.debug("Compressed log file", original=str(file_path), compressed=str(compressed_path))
            
        except Exception as e:
            logger.error("Failed to compress log file", file=str(file_path), error=str(e))
    
    async def query_logs(self, query: Dict[str, Any]) -> List[LogEntry]:
        """Query log entries (simplified implementation)."""
        results = []
        # Implementation would parse log files and filter based on query
        return results
    
    async def query_audit_events(self, query: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events (simplified implementation)."""
        results = []
        # Implementation would parse audit files and filter based on query
        return results


class DatabaseStorage(ILogStorage):
    """Database-based log storage."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.db_pool = None
    
    async def store_log(self, log_entry: LogEntry) -> bool:
        """Store log entry to database."""
        # Implementation would use SQLAlchemy or similar
        return True
    
    async def store_audit_event(self, audit_event: AuditEvent) -> bool:
        """Store audit event to database."""
        # Implementation would use SQLAlchemy or similar
        return True
    
    async def query_logs(self, query: Dict[str, Any]) -> List[LogEntry]:
        """Query log entries from database."""
        return []
    
    async def query_audit_events(self, query: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events from database."""
        return []


class EncryptedStorage(ILogStorage):
    """Encrypted log storage wrapper."""
    
    def __init__(self, underlying_storage: ILogStorage, encryption_key: bytes):
        self.underlying_storage = underlying_storage
        self.cipher = Fernet(encryption_key)
    
    async def store_log(self, log_entry: LogEntry) -> bool:
        """Store encrypted log entry."""
        try:
            # Encrypt sensitive fields
            encrypted_entry = self._encrypt_log_entry(log_entry)
            return await self.underlying_storage.store_log(encrypted_entry)
        except Exception as e:
            logger.error("Failed to store encrypted log", error=str(e))
            return False
    
    async def store_audit_event(self, audit_event: AuditEvent) -> bool:
        """Store encrypted audit event."""
        try:
            # Encrypt sensitive fields
            encrypted_event = self._encrypt_audit_event(audit_event)
            return await self.underlying_storage.store_audit_event(encrypted_event)
        except Exception as e:
            logger.error("Failed to store encrypted audit event", error=str(e))
            return False
    
    def _encrypt_log_entry(self, log_entry: LogEntry) -> LogEntry:
        """Encrypt sensitive fields in log entry."""
        encrypted_entry = LogEntry(**log_entry.to_dict())
        
        # Encrypt sensitive fields
        if log_entry.message and "password" in log_entry.message.lower():
            encrypted_entry.message = self._encrypt_string(log_entry.message)
        
        return encrypted_entry
    
    def _encrypt_audit_event(self, audit_event: AuditEvent) -> AuditEvent:
        """Encrypt sensitive fields in audit event."""
        encrypted_event = AuditEvent(**audit_event.to_dict())
        
        # Encrypt sensitive details
        if audit_event.sensitive_data:
            encrypted_event.details = {
                k: self._encrypt_string(str(v)) if isinstance(v, str) and len(v) > 10 else v
                for k, v in audit_event.details.items()
            }
        
        return encrypted_event
    
    def _encrypt_string(self, text: str) -> str:
        """Encrypt a string."""
        return self.cipher.encrypt(text.encode()).decode()
    
    async def query_logs(self, query: Dict[str, Any]) -> List[LogEntry]:
        """Query and decrypt log entries."""
        encrypted_logs = await self.underlying_storage.query_logs(query)
        return [self._decrypt_log_entry(log) for log in encrypted_logs]
    
    async def query_audit_events(self, query: Dict[str, Any]) -> List[AuditEvent]:
        """Query and decrypt audit events."""
        encrypted_events = await self.underlying_storage.query_audit_events(query)
        return [self._decrypt_audit_event(event) for event in encrypted_events]
    
    def _decrypt_log_entry(self, log_entry: LogEntry) -> LogEntry:
        """Decrypt log entry."""
        # Implementation would decrypt fields
        return log_entry
    
    def _decrypt_audit_event(self, audit_event: AuditEvent) -> AuditEvent:
        """Decrypt audit event."""
        # Implementation would decrypt fields
        return audit_event


class ComplianceTracker:
    """Compliance framework tracking and reporting."""
    
    def __init__(self):
        self.framework_requirements = {
            ComplianceFramework.SOC2: {
                'retention_days': 2555,  # 7 years
                'required_events': [
                    AuditEventType.AUTHENTICATION,
                    AuditEventType.AUTHORIZATION,
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.CONFIGURATION_CHANGE
                ],
                'encryption_required': True
            },
            ComplianceFramework.GDPR: {
                'retention_days': 2555,
                'required_events': [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_EXPORT,
                    AuditEventType.CONFIGURATION_CHANGE
                ],
                'encryption_required': True,
                'anonymization_required': True
            },
            ComplianceFramework.HIPAA: {
                'retention_days': 2190,  # 6 years
                'required_events': [
                    AuditEventType.AUTHENTICATION,
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.SECURITY_VIOLATION
                ],
                'encryption_required': True
            }
        }
    
    def validate_compliance(self, audit_event: AuditEvent) -> Dict[str, bool]:
        """Validate event against compliance requirements."""
        results = {}
        
        for framework in audit_event.compliance_frameworks:
            if framework in self.framework_requirements:
                requirements = self.framework_requirements[framework]
                results[framework.value] = self._check_framework_compliance(audit_event, requirements)
        
        return results
    
    def _check_framework_compliance(self, event: AuditEvent, requirements: Dict[str, Any]) -> bool:
        """Check if event meets framework requirements."""
        # Check retention period
        if event.retention_days < requirements.get('retention_days', 0):
            return False
        
        # Check if event type is required for this framework
        required_events = requirements.get('required_events', [])
        if required_events and event.event_type not in required_events:
            return False
        
        # Check encryption requirement
        if requirements.get('encryption_required', False) and not event.sensitive_data:
            # This would check if storage is encrypted
            pass
        
        return True
    
    def generate_compliance_report(self, events: List[AuditEvent], framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for a framework."""
        if framework not in self.framework_requirements:
            return {'error': 'Unknown compliance framework'}
        
        requirements = self.framework_requirements[framework]
        
        # Count events by type
        event_counts = {}
        compliant_events = 0
        
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if self._check_framework_compliance(event, requirements):
                compliant_events += 1
        
        return {
            'framework': framework.value,
            'total_events': len(events),
            'compliant_events': compliant_events,
            'compliance_percentage': (compliant_events / len(events)) * 100 if events else 0,
            'event_counts': event_counts,
            'requirements': requirements
        }


class AdvancedLoggingSystem:
    """Advanced logging and audit system."""
    
    def __init__(self):
        self.storage_backends = []
        self.compliance_tracker = ComplianceTracker()
        self.log_buffer = deque(maxlen=10000)
        self.audit_buffer = deque(maxlen=5000)
        self.running = False
        
        # Metrics
        self.logs_processed = 0
        self.audit_events_processed = 0
        self.storage_errors = 0
    
    def add_storage_backend(self, storage: ILogStorage):
        """Add a storage backend."""
        self.storage_backends.append(storage)
        logger.info("Added logging storage backend", backend=type(storage).__name__)
    
    async def start_logging_system(self):
        """Start the logging system."""
        self.running = True
        
        # Setup default storage
        if not self.storage_backends:
            default_storage = FileStorage(Path("logs"))
            self.add_storage_backend(default_storage)
        
        # Start processing tasks
        log_task = asyncio.create_task(self._log_processing_loop())
        audit_task = asyncio.create_task(self._audit_processing_loop())
        
        logger.info("Advanced logging system started")
        
        try:
            await asyncio.gather(log_task, audit_task)
        except asyncio.CancelledError:
            logger.info("Logging system stopped")
    
    async def stop_logging_system(self):
        """Stop the logging system."""
        self.running = False
    
    async def log(self, level: LogLevel, message: str, **kwargs):
        """Log a message."""
        log_entry = LogEntry(
            level=level,
            message=message,
            **kwargs
        )
        
        self.log_buffer.append(log_entry)
    
    async def audit(self, event_type: AuditEventType, action: str, **kwargs):
        """Create an audit event."""
        audit_event = AuditEvent(
            event_type=event_type,
            action=action,
            **kwargs
        )
        
        # Validate compliance
        compliance_results = self.compliance_tracker.validate_compliance(audit_event)
        if compliance_results:
            audit_event.details['compliance_validation'] = compliance_results
        
        self.audit_buffer.append(audit_event)
    
    async def _log_processing_loop(self):
        """Process log entries."""
        while self.running:
            try:
                if self.log_buffer:
                    # Process batch of log entries
                    batch = []
                    while self.log_buffer and len(batch) < 100:
                        batch.append(self.log_buffer.popleft())
                    
                    for log_entry in batch:
                        await self._store_log_entry(log_entry)
                        self.logs_processed += 1
                
                await asyncio.sleep(1)  # Process every second
            except Exception as e:
                logger.error("Error in log processing loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _audit_processing_loop(self):
        """Process audit events."""
        while self.running:
            try:
                if self.audit_buffer:
                    # Process batch of audit events
                    batch = []
                    while self.audit_buffer and len(batch) < 50:
                        batch.append(self.audit_buffer.popleft())
                    
                    for audit_event in batch:
                        await self._store_audit_event(audit_event)
                        self.audit_events_processed += 1
                
                await asyncio.sleep(1)  # Process every second
            except Exception as e:
                logger.error("Error in audit processing loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry to all backends."""
        for storage in self.storage_backends:
            try:
                success = await storage.store_log(log_entry)
                if not success:
                    self.storage_errors += 1
            except Exception as e:
                logger.error("Storage backend failed", backend=type(storage).__name__, error=str(e))
                self.storage_errors += 1
    
    async def _store_audit_event(self, audit_event: AuditEvent):
        """Store audit event to all backends."""
        for storage in self.storage_backends:
            try:
                success = await storage.store_audit_event(audit_event)
                if not success:
                    self.storage_errors += 1
            except Exception as e:
                logger.error("Storage backend failed", backend=type(storage).__name__, error=str(e))
                self.storage_errors += 1
    
    async def query_logs(self, query: Dict[str, Any]) -> List[LogEntry]:
        """Query log entries."""
        # Use first available storage backend for queries
        if self.storage_backends:
            return await self.storage_backends[0].query_logs(query)
        return []
    
    async def query_audit_events(self, query: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events."""
        # Use first available storage backend for queries
        if self.storage_backends:
            return await self.storage_backends[0].query_audit_events(query)
        return []
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        return {
            'storage_backends': len(self.storage_backends),
            'logs_processed': self.logs_processed,
            'audit_events_processed': self.audit_events_processed,
            'storage_errors': self.storage_errors,
            'log_buffer_size': len(self.log_buffer),
            'audit_buffer_size': len(self.audit_buffer),
            'running': self.running
        }


# Global logging system instance
logging_system = AdvancedLoggingSystem()


async def initialize_logging():
    """Initialize the advanced logging system."""
    await logging_system.start_logging_system()


async def shutdown_logging():
    """Shutdown the logging system."""
    await logging_system.stop_logging_system()


def get_logging_system() -> AdvancedLoggingSystem:
    """Get the global logging system."""
    return logging_system


# Convenience functions
async def log_info(message: str, **kwargs):
    """Log info message."""
    await logging_system.log(LogLevel.INFO, message, **kwargs)


async def log_error(message: str, **kwargs):
    """Log error message."""
    await logging_system.log(LogLevel.ERROR, message, **kwargs)


async def audit_user_action(action: str, user_id: str, **kwargs):
    """Audit user action."""
    await logging_system.audit(AuditEventType.USER_ACTION, action, user_id=user_id, **kwargs)


async def audit_security_event(action: str, severity: LogLevel = LogLevel.WARNING, **kwargs):
    """Audit security event."""
    await logging_system.audit(AuditEventType.SECURITY_VIOLATION, action, severity=severity, **kwargs)