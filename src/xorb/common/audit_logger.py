#!/usr/bin/env python3
"""
Enterprise Audit Logging System
Comprehensive audit trail and compliance logging for autonomous operations

This module provides enterprise-grade audit logging capabilities including:
- Comprehensive audit trail for all operations
- Compliance-focused event logging
- Real-time security event monitoring
- Tamper-evident log storage
- Automated compliance reporting
- Integration with SIEM systems
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import hmac
from collections import deque
import gzip
import structlog

# Database and storage
try:
    import asyncpg
    import redis.asyncio as redis
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database libraries not available - using file-based logging")

from .security_framework import SecurityLevel

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Audit event types"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    OPERATION_START = "operation_start"
    OPERATION_END = "operation_end"
    TECHNIQUE_EXECUTION = "technique_execution"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"
    EMERGENCY_STOP = "emergency_stop"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_ERROR = "system_error"
    ADMIN_ACTION = "admin_action"


class EventSeverity(Enum):
    """Event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    event_id: str
    event_type: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    target_resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    severity: EventSeverity = EventSeverity.INFO
    compliance_tags: List[str] = None
    related_events: List[str] = None
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.compliance_tags is None:
            self.compliance_tags = []
        if self.related_events is None:
            self.related_events = []
        if self.details is None:
            self.details = {}


@dataclass
class AuditQuery:
    """Audit log query parameters"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    components: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    severity_levels: Optional[List[str]] = None
    security_levels: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class LogIntegrityManager:
    """Manages log integrity and tamper detection"""
    
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.hash_chain: List[str] = []
        
    def calculate_event_hash(self, event: AuditEvent, previous_hash: str = "") -> str:
        """Calculate tamper-evident hash for event"""
        try:
            # Create deterministic string representation
            event_string = json.dumps(asdict(event), sort_keys=True, default=str)
            
            # Include previous hash for chaining
            chain_input = f"{previous_hash}{event_string}"
            
            # Calculate HMAC
            signature = hmac.new(
                self.secret_key,
                chain_input.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error("Failed to calculate event hash", error=str(e))
            return ""
    
    def verify_hash_chain(self, events: List[AuditEvent], hashes: List[str]) -> bool:
        """Verify integrity of hash chain"""
        try:
            if len(events) != len(hashes):
                return False
            
            previous_hash = ""
            for event, expected_hash in zip(events, hashes):
                calculated_hash = self.calculate_event_hash(event, previous_hash)
                if calculated_hash != expected_hash:
                    logger.warning("Hash chain verification failed",
                                 event_id=event.event_id)
                    return False
                previous_hash = calculated_hash
            
            return True
            
        except Exception as e:
            logger.error("Hash chain verification error", error=str(e))
            return False


class DatabaseAuditStore:
    """Database-backed audit log storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            if not DATABASE_AVAILABLE:
                raise RuntimeError("Database libraries not available")
            
            self.connection_pool = await asyncpg.create_pool(
                self.config["database_url"],
                min_size=5,
                max_size=20
            )
            
            # Create audit tables if not exist
            await self._create_audit_tables()
            
            logger.info("Database audit store initialized")
            
        except Exception as e:
            logger.error("Failed to initialize database audit store", error=str(e))
            raise
    
    async def _create_audit_tables(self):
        """Create audit log tables"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id UUID PRIMARY KEY,
            event_type VARCHAR(100) NOT NULL,
            component VARCHAR(100) NOT NULL,
            user_id VARCHAR(100),
            session_id VARCHAR(100),
            source_ip INET,
            target_resource TEXT,
            action VARCHAR(200),
            result VARCHAR(100),
            details JSONB,
            security_level VARCHAR(50),
            severity VARCHAR(50),
            compliance_tags TEXT[],
            related_events UUID[],
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            correlation_id UUID,
            event_hash VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_events(component);
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_correlation ON audit_events(correlation_id);
        CREATE INDEX IF NOT EXISTS idx_audit_security_level ON audit_events(security_level);
        """
        
        async with self.connection_pool.acquire() as conn:
            await conn.execute(create_table_sql)
    
    async def store_event(self, event: AuditEvent, event_hash: str) -> bool:
        """Store audit event in database"""
        try:
            insert_sql = """
            INSERT INTO audit_events (
                event_id, event_type, component, user_id, session_id,
                source_ip, target_resource, action, result, details,
                security_level, severity, compliance_tags, related_events,
                timestamp, correlation_id, event_hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    event.event_id,
                    event.event_type,
                    event.component,
                    event.user_id,
                    event.session_id,
                    event.source_ip,
                    event.target_resource,
                    event.action,
                    event.result,
                    json.dumps(event.details),
                    event.security_level.value,
                    event.severity.value,
                    event.compliance_tags,
                    event.related_events,
                    event.timestamp,
                    event.correlation_id,
                    event_hash
                )
            
            return True
            
        except Exception as e:
            logger.error("Failed to store audit event", 
                        event_id=event.event_id, error=str(e))
            return False
    
    async def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events from database"""
        try:
            # Build dynamic query
            where_conditions = []
            params = []
            param_count = 0
            
            if query.start_time:
                param_count += 1
                where_conditions.append(f"timestamp >= ${param_count}")
                params.append(query.start_time)
            
            if query.end_time:
                param_count += 1
                where_conditions.append(f"timestamp <= ${param_count}")
                params.append(query.end_time)
            
            if query.event_types:
                param_count += 1
                where_conditions.append(f"event_type = ANY(${param_count})")
                params.append(query.event_types)
            
            if query.components:
                param_count += 1
                where_conditions.append(f"component = ANY(${param_count})")
                params.append(query.components)
            
            if query.user_ids:
                param_count += 1
                where_conditions.append(f"user_id = ANY(${param_count})")
                params.append(query.user_ids)
            
            if query.correlation_id:
                param_count += 1
                where_conditions.append(f"correlation_id = ${param_count}")
                params.append(query.correlation_id)
            
            # Construct query
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            select_sql = f"""
            SELECT * FROM audit_events 
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {query.limit} OFFSET {query.offset}
            """
            
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(select_sql, *params)
            
            # Convert rows to AuditEvent objects
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=str(row['event_id']),
                    event_type=row['event_type'],
                    component=row['component'],
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    source_ip=str(row['source_ip']) if row['source_ip'] else None,
                    target_resource=row['target_resource'],
                    action=row['action'],
                    result=row['result'],
                    details=json.loads(row['details']) if row['details'] else {},
                    security_level=SecurityLevel(row['security_level']),
                    severity=EventSeverity(row['severity']),
                    compliance_tags=row['compliance_tags'] or [],
                    related_events=[str(e) for e in row['related_events']] if row['related_events'] else [],
                    timestamp=row['timestamp'],
                    correlation_id=str(row['correlation_id']) if row['correlation_id'] else None
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error("Failed to query audit events", error=str(e))
            return []


class FileAuditStore:
    """File-based audit log storage with rotation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_directory = Path(config.get("log_directory", "logs/audit"))
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.retention_days = config.get("retention_days", 90)
        self.compress_old_logs = config.get("compress_old_logs", True)
        
    async def initialize(self):
        """Initialize file-based storage"""
        try:
            # Create log directory
            self.log_directory.mkdir(parents=True, exist_ok=True)
            
            # Set up log rotation
            await self._setup_log_rotation()
            
            logger.info("File audit store initialized", 
                       log_directory=str(self.log_directory))
            
        except Exception as e:
            logger.error("Failed to initialize file audit store", error=str(e))
            raise
    
    async def store_event(self, event: AuditEvent, event_hash: str) -> bool:
        """Store audit event in file"""
        try:
            # Get current log file
            log_file = await self._get_current_log_file()
            
            # Create log entry
            log_entry = {
                **asdict(event),
                "event_hash": event_hash,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            # Append to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
            
            # Check if rotation is needed
            await self._check_rotation_needed(log_file)
            
            return True
            
        except Exception as e:
            logger.error("Failed to store audit event to file",
                        event_id=event.event_id, error=str(e))
            return False
    
    async def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        today = datetime.now().strftime("%Y%m%d")
        return self.log_directory / f"audit_{today}.jsonl"
    
    async def _check_rotation_needed(self, log_file: Path):
        """Check if log rotation is needed"""
        try:
            if log_file.stat().st_size > self.max_file_size:
                # Compress current file
                if self.compress_old_logs:
                    compressed_file = log_file.with_suffix('.jsonl.gz')
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original
                    log_file.unlink()
                    
                    logger.info("Log file rotated and compressed",
                               compressed_file=str(compressed_file))
        
        except Exception as e:
            logger.error("Log rotation failed", error=str(e))


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger_id = str(uuid.uuid4())
        
        # Initialize integrity manager
        secret_key = self.config.get("integrity_key", b"default_key_change_in_production")
        self.integrity_manager = LogIntegrityManager(secret_key)
        
        # Initialize storage backends
        self.storage_backends = []
        
        # Event queue for asynchronous processing
        self.event_queue = deque()
        self.processing_task = None
        
        # Metrics
        self.metrics = {
            "events_logged": 0,
            "events_failed": 0,
            "storage_errors": 0,
            "integrity_violations": 0
        }
        
    async def initialize(self):
        """Initialize audit logging system"""
        try:
            logger.info("Initializing Audit Logger", logger_id=self.logger_id)
            
            # Initialize storage backends
            if DATABASE_AVAILABLE and self.config.get("use_database", True):
                db_store = DatabaseAuditStore(self.config.get("database", {}))
                await db_store.initialize()
                self.storage_backends.append(db_store)
            
            # Always include file storage as backup
            file_store = FileAuditStore(self.config.get("file_storage", {}))
            await file_store.initialize()
            self.storage_backends.append(file_store)
            
            # Start event processing
            self.processing_task = asyncio.create_task(self._process_events())
            
            logger.info("Audit Logger initialized successfully",
                       backends_count=len(self.storage_backends))
            
        except Exception as e:
            logger.error("Audit Logger initialization failed", error=str(e))
            raise
    
    async def log_event(self, event: AuditEvent) -> bool:
        """Log audit event with integrity protection"""
        try:
            # Calculate integrity hash
            previous_hash = self.integrity_manager.hash_chain[-1] if self.integrity_manager.hash_chain else ""
            event_hash = self.integrity_manager.calculate_event_hash(event, previous_hash)
            
            # Add to hash chain
            self.integrity_manager.hash_chain.append(event_hash)
            
            # Add to processing queue
            self.event_queue.append((event, event_hash))
            
            logger.debug("Audit event queued", 
                        event_id=event.event_id,
                        event_type=event.event_type)
            
            return True
            
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
            self.metrics["events_failed"] += 1
            return False
    
    async def _process_events(self):
        """Process queued events asynchronously"""
        while True:
            try:
                if self.event_queue:
                    event, event_hash = self.event_queue.popleft()
                    
                    # Store in all backends
                    storage_success = True
                    for backend in self.storage_backends:
                        try:
                            success = await backend.store_event(event, event_hash)
                            if not success:
                                storage_success = False
                                self.metrics["storage_errors"] += 1
                        except Exception as e:
                            logger.error("Storage backend failed",
                                        backend=type(backend).__name__,
                                        error=str(e))
                            storage_success = False
                            self.metrics["storage_errors"] += 1
                    
                    if storage_success:
                        self.metrics["events_logged"] += 1
                    else:
                        self.metrics["events_failed"] += 1
                        logger.error("Failed to store event in any backend",
                                   event_id=event.event_id)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Event processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        try:
            # Try database first, fall back to file storage
            for backend in self.storage_backends:
                if hasattr(backend, 'query_events'):
                    try:
                        events = await backend.query_events(query)
                        return events
                    except Exception as e:
                        logger.warning("Query failed on backend",
                                     backend=type(backend).__name__,
                                     error=str(e))
                        continue
            
            return []
            
        except Exception as e:
            logger.error("Event query failed", error=str(e))
            return []
    
    async def generate_compliance_report(self, 
                                       start_time: datetime,
                                       end_time: datetime,
                                       compliance_framework: str) -> Dict[str, Any]:
        """Generate compliance report for specified timeframe"""
        try:
            # Query relevant events
            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            events = await self.query_events(query)
            
            # Analyze events for compliance
            report = {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "compliance_framework": compliance_framework,
                "total_events": len(events),
                "events_by_type": {},
                "security_events": [],
                "compliance_violations": [],
                "recommendations": []
            }
            
            # Categorize events
            for event in events:
                event_type = event.event_type
                if event_type not in report["events_by_type"]:
                    report["events_by_type"][event_type] = 0
                report["events_by_type"][event_type] += 1
                
                # Check for security events
                if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                    report["security_events"].append({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "severity": event.severity.value,
                        "timestamp": event.timestamp.isoformat(),
                        "component": event.component
                    })
                
                # Check for compliance violations
                if "violation" in event.event_type or "denied" in event.event_type:
                    report["compliance_violations"].append({
                        "event_id": event.event_id,
                        "description": event.details.get("description", ""),
                        "timestamp": event.timestamp.isoformat()
                    })
            
            # Generate recommendations
            if report["security_events"]:
                report["recommendations"].append("Review and address security events")
            
            if report["compliance_violations"]:
                report["recommendations"].append("Investigate compliance violations")
            
            return report
            
        except Exception as e:
            logger.error("Compliance report generation failed", error=str(e))
            return {"error": str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get audit logging metrics"""
        return {
            "logger_id": self.logger_id,
            "metrics": self.metrics.copy(),
            "queue_size": len(self.event_queue),
            "hash_chain_length": len(self.integrity_manager.hash_chain),
            "storage_backends": len(self.storage_backends),
            "processing_active": self.processing_task is not None and not self.processing_task.done()
        }
    
    async def shutdown(self):
        """Shutdown audit logger gracefully"""
        try:
            logger.info("Shutting down Audit Logger")
            
            # Process remaining events
            while self.event_queue:
                await asyncio.sleep(0.1)
            
            # Cancel processing task
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Audit Logger shutdown complete")
            
        except Exception as e:
            logger.error("Audit Logger shutdown failed", error=str(e))


# Export main classes
__all__ = [
    "AuditLogger",
    "AuditEvent",
    "AuditQuery",
    "EventType",
    "EventSeverity",
    "LogIntegrityManager"
]