#!/usr/bin/env python3
"""
XORB Enhanced Audit System
Comprehensive audit trails with detailed error context and timestamps
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
import uuid
import hashlib
import threading
from contextlib import contextmanager

# Import error handling framework
from xorb_error_handling_framework import (
    XORBErrorHandler, ErrorCategory, ErrorSeverity, RecoveryStrategy,
    RecoveryAction, ErrorContext, xorb_async_error_handler, get_error_handler
)

class AuditEventType(Enum):
    """Types of audit events"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    ERROR_RESOLVED = "error_resolved"
    RECOVERY_INITIATED = "recovery_initiated"
    RECOVERY_COMPLETED = "recovery_completed"
    CONFIGURATION_CHANGED = "configuration_changed"
    AUTHENTICATION_EVENT = "authentication_event"
    AUTHORIZATION_EVENT = "authorization_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DEPLOYMENT_EVENT = "deployment_event"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_CHECK = "compliance_check"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Compliance standards for audit events"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"

@dataclass
class AuditEvent:
    """Represents a comprehensive audit event"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    service_name: str
    component: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    event_description: str
    event_details: Dict[str, Any]
    error_context: Optional[Dict[str, Any]]
    stack_trace: Optional[str]
    request_id: Optional[str]
    correlation_id: Optional[str]
    duration_ms: Optional[float]
    resource_accessed: Optional[str]
    action_performed: Optional[str]
    result_status: str
    compliance_tags: List[ComplianceStandard]
    sensitive_data_involved: bool
    data_classification: str
    geographic_location: Optional[str]
    additional_metadata: Dict[str, Any]
    checksum: str

    def __post_init__(self):
        """Calculate checksum for integrity verification"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate event checksum for integrity"""
        event_data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "service_name": self.service_name,
            "event_description": self.event_description,
            "result_status": self.result_status
        }

        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'compliance_tags': [tag.value for tag in self.compliance_tags]
        }

    def is_sensitive(self) -> bool:
        """Check if event contains sensitive data"""
        return (
            self.sensitive_data_involved or
            self.data_classification in ["confidential", "restricted", "secret"] or
            self.event_type in [
                AuditEventType.AUTHENTICATION_EVENT,
                AuditEventType.AUTHORIZATION_EVENT,
                AuditEventType.SECURITY_INCIDENT
            ]
        )

@dataclass
class AuditQuery:
    """Represents an audit query for searching events"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severity_levels: Optional[List[AuditSeverity]] = None
    services: Optional[List[str]] = None
    users: Optional[List[str]] = None
    components: Optional[List[str]] = None
    compliance_standards: Optional[List[ComplianceStandard]] = None
    text_search: Optional[str] = None
    error_categories: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0
    sort_by: str = "timestamp"
    sort_order: str = "desc"

class AuditStorage:
    """Handles audit event storage with SQLite backend"""

    def __init__(self, db_path: str = "/tmp/xorb_audit.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for audit events"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    service_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    event_description TEXT NOT NULL,
                    event_details TEXT,
                    error_context TEXT,
                    stack_trace TEXT,
                    request_id TEXT,
                    correlation_id TEXT,
                    duration_ms REAL,
                    resource_accessed TEXT,
                    action_performed TEXT,
                    result_status TEXT NOT NULL,
                    compliance_tags TEXT,
                    sensitive_data_involved INTEGER,
                    data_classification TEXT,
                    geographic_location TEXT,
                    additional_metadata TEXT,
                    checksum TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_service_name ON audit_events(service_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_events(correlation_id)')

            # Create audit log integrity table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    events_count INTEGER NOT NULL,
                    checksum_hash TEXT NOT NULL,
                    verification_status TEXT NOT NULL
                )
            ''')

            conn.commit()
            conn.close()

    @contextmanager
    def get_connection(self):
        """Get database connection with proper locking"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def store_event(self, event: AuditEvent) -> bool:
        """Store audit event in database"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, severity, service_name, component,
                        user_id, session_id, ip_address, user_agent, event_description,
                        event_details, error_context, stack_trace, request_id, correlation_id,
                        duration_ms, resource_accessed, action_performed, result_status,
                        compliance_tags, sensitive_data_involved, data_classification,
                        geographic_location, additional_metadata, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.severity.value,
                    event.service_name,
                    event.component,
                    event.user_id,
                    event.session_id,
                    event.ip_address,
                    event.user_agent,
                    event.event_description,
                    json.dumps(event.event_details) if event.event_details else None,
                    json.dumps(event.error_context) if event.error_context else None,
                    event.stack_trace,
                    event.request_id,
                    event.correlation_id,
                    event.duration_ms,
                    event.resource_accessed,
                    event.action_performed,
                    event.result_status,
                    json.dumps([tag.value for tag in event.compliance_tags]),
                    1 if event.sensitive_data_involved else 0,
                    event.data_classification,
                    event.geographic_location,
                    json.dumps(event.additional_metadata) if event.additional_metadata else None,
                    event.checksum
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Failed to store audit event: {e}")
            return False

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events based on criteria"""
        try:
            with self.get_connection() as conn:
                where_clauses = []
                params = []

                if query.start_time:
                    where_clauses.append("timestamp >= ?")
                    params.append(query.start_time.isoformat())

                if query.end_time:
                    where_clauses.append("timestamp <= ?")
                    params.append(query.end_time.isoformat())

                if query.event_types:
                    placeholders = ','.join(['?' for _ in query.event_types])
                    where_clauses.append(f"event_type IN ({placeholders})")
                    params.extend([et.value for et in query.event_types])

                if query.severity_levels:
                    placeholders = ','.join(['?' for _ in query.severity_levels])
                    where_clauses.append(f"severity IN ({placeholders})")
                    params.extend([sl.value for sl in query.severity_levels])

                if query.services:
                    placeholders = ','.join(['?' for _ in query.services])
                    where_clauses.append(f"service_name IN ({placeholders})")
                    params.extend(query.services)

                if query.users:
                    placeholders = ','.join(['?' for _ in query.users])
                    where_clauses.append(f"user_id IN ({placeholders})")
                    params.extend(query.users)

                if query.text_search:
                    where_clauses.append("(event_description LIKE ? OR event_details LIKE ?)")
                    search_term = f"%{query.text_search}%"
                    params.extend([search_term, search_term])

                where_clause = " AND ".join(where_clauses) if where_clauses else "1"

                sql = f'''
                    SELECT * FROM audit_events
                    WHERE {where_clause}
                    ORDER BY {query.sort_by} {query.sort_order.upper()}
                    LIMIT ? OFFSET ?
                '''
                params.extend([query.limit, query.offset])

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    # Convert row to AuditEvent
                    event = AuditEvent(
                        event_id=row['event_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=AuditEventType(row['event_type']),
                        severity=AuditSeverity(row['severity']),
                        service_name=row['service_name'],
                        component=row['component'],
                        user_id=row['user_id'],
                        session_id=row['session_id'],
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        event_description=row['event_description'],
                        event_details=json.loads(row['event_details']) if row['event_details'] else {},
                        error_context=json.loads(row['error_context']) if row['error_context'] else None,
                        stack_trace=row['stack_trace'],
                        request_id=row['request_id'],
                        correlation_id=row['correlation_id'],
                        duration_ms=row['duration_ms'],
                        resource_accessed=row['resource_accessed'],
                        action_performed=row['action_performed'],
                        result_status=row['result_status'],
                        compliance_tags=[ComplianceStandard(tag) for tag in json.loads(row['compliance_tags'])],
                        sensitive_data_involved=bool(row['sensitive_data_involved']),
                        data_classification=row['data_classification'],
                        geographic_location=row['geographic_location'],
                        additional_metadata=json.loads(row['additional_metadata']) if row['additional_metadata'] else {},
                        checksum=row['checksum']
                    )
                    events.append(event)

                return events

        except Exception as e:
            logging.error(f"Failed to query audit events: {e}")
            return []

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get audit event statistics"""
        try:
            with self.get_connection() as conn:
                # Total events
                total_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

                # Events by type
                type_counts = {}
                cursor = conn.execute("SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type")
                for row in cursor:
                    type_counts[row[0]] = row[1]

                # Events by severity
                severity_counts = {}
                cursor = conn.execute("SELECT severity, COUNT(*) FROM audit_events GROUP BY severity")
                for row in cursor:
                    severity_counts[row[0]] = row[1]

                # Events by service
                service_counts = {}
                cursor = conn.execute("SELECT service_name, COUNT(*) FROM audit_events GROUP BY service_name LIMIT 10")
                for row in cursor:
                    service_counts[row[0]] = row[1]

                # Recent error events
                recent_errors = conn.execute('''
                    SELECT COUNT(*) FROM audit_events
                    WHERE severity IN ('error', 'critical')
                    AND timestamp >= datetime('now', '-1 hour')
                ''').fetchone()[0]

                return {
                    "total_events": total_count,
                    "events_by_type": type_counts,
                    "events_by_severity": severity_counts,
                    "events_by_service": service_counts,
                    "recent_errors": recent_errors,
                    "database_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }

        except Exception as e:
            logging.error(f"Failed to get audit statistics: {e}")
            return {}

class EnhancedAuditSystem:
    """Main enhanced audit system with comprehensive logging"""

    def __init__(self, error_handler: XORBErrorHandler):
        self.error_handler = error_handler
        self.storage = AuditStorage()
        self.correlation_context = {}
        self.audit_filters = []
        self.compliance_rules = {}

        # Initialize compliance rules
        self._initialize_compliance_rules()

        # Register error handler integration
        self._register_error_handler_integration()

    def _initialize_compliance_rules(self):
        """Initialize compliance-specific audit rules"""
        self.compliance_rules = {
            ComplianceStandard.SOX: {
                "required_events": [
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.CONFIGURATION_CHANGED,
                    AuditEventType.AUTHENTICATION_EVENT
                ],
                "retention_days": 2555,  # 7 years
                "encryption_required": True
            },
            ComplianceStandard.GDPR: {
                "required_events": [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.AUTHENTICATION_EVENT
                ],
                "retention_days": 1095,  # 3 years
                "anonymization_required": True
            },
            ComplianceStandard.PCI_DSS: {
                "required_events": [
                    AuditEventType.AUTHENTICATION_EVENT,
                    AuditEventType.AUTHORIZATION_EVENT,
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.SECURITY_INCIDENT
                ],
                "retention_days": 365,  # 1 year
                "access_logging_required": True
            }
        }

    def _register_error_handler_integration(self):
        """Register audit system with error handler for automatic logging"""

        # Register audit recovery action
        audit_recovery = RecoveryAction(
            action_id="audit_error_event",
            name="Audit Error Event",
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            handler=self._audit_error_event,
            conditions={"categories": ["*"]}  # All categories
        )
        self.error_handler.register_recovery_action(audit_recovery)

    async def _audit_error_event(self, error_context: ErrorContext) -> bool:
        """Audit error events automatically"""
        try:
            await self.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                severity=self._map_error_severity(error_context.severity),
                service_name=error_context.service_name,
                component=error_context.function_name,
                event_description=f"Error occurred: {error_context.error_message}",
                error_context={
                    "error_id": error_context.error_id,
                    "error_type": error_context.error_type,
                    "category": error_context.category.value,
                    "severity": error_context.severity.value,
                    "retry_count": error_context.retry_count
                },
                stack_trace=error_context.stack_trace,
                correlation_id=error_context.error_id,
                result_status="error"
            )
            return True
        except Exception:
            return False

    def _map_error_severity(self, error_severity: ErrorSeverity) -> AuditSeverity:
        """Map error severity to audit severity"""
        mapping = {
            ErrorSeverity.LOW: AuditSeverity.WARNING,
            ErrorSeverity.MEDIUM: AuditSeverity.ERROR,
            ErrorSeverity.HIGH: AuditSeverity.ERROR,
            ErrorSeverity.CRITICAL: AuditSeverity.CRITICAL
        }
        return mapping.get(error_severity, AuditSeverity.ERROR)

    @xorb_async_error_handler(
        category=ErrorCategory.SYSTEM_RESOURCE,
        severity=ErrorSeverity.MEDIUM,
        retry_count=2
    )
    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        service_name: str,
        component: str,
        event_description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        event_details: Optional[Dict[str, Any]] = None,
        error_context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        resource_accessed: Optional[str] = None,
        action_performed: Optional[str] = None,
        result_status: str = "success",
        compliance_tags: Optional[List[ComplianceStandard]] = None,
        sensitive_data_involved: bool = False,
        data_classification: str = "public",
        geographic_location: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a comprehensive audit event"""

        event_id = str(uuid.uuid4())

        # Apply compliance rules if tags provided
        if compliance_tags:
            for standard in compliance_tags:
                if standard in self.compliance_rules:
                    rules = self.compliance_rules[standard]
                    if event_type in rules.get("required_events", []):
                        # Apply compliance-specific settings
                        if rules.get("encryption_required"):
                            additional_metadata = additional_metadata or {}
                            additional_metadata["encryption_required"] = True

        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            service_name=service_name,
            component=component,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            event_description=event_description,
            event_details=event_details or {},
            error_context=error_context,
            stack_trace=stack_trace,
            request_id=request_id,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            result_status=result_status,
            compliance_tags=compliance_tags or [],
            sensitive_data_involved=sensitive_data_involved,
            data_classification=data_classification,
            geographic_location=geographic_location,
            additional_metadata=additional_metadata or {},
            checksum=""  # Will be calculated in __post_init__
        )

        # Store event
        success = self.storage.store_event(event)
        if not success:
            raise RuntimeError("Failed to store audit event")

        return event_id

    async def log_system_event(self, event_type: AuditEventType, description: str, details: Dict[str, Any] = None):
        """Log system-level events"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.INFO,
            service_name="xorb_system",
            component="audit_system",
            event_description=description,
            event_details=details,
            result_status="success"
        )

    async def log_authentication_event(
        self,
        user_id: str,
        result_status: str,
        ip_address: str,
        user_agent: str,
        additional_details: Dict[str, Any] = None
    ):
        """Log authentication events with compliance tagging"""
        await self.log_event(
            event_type=AuditEventType.AUTHENTICATION_EVENT,
            severity=AuditSeverity.INFO if result_status == "success" else AuditSeverity.WARNING,
            service_name="auth_service",
            component="authentication",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            event_description=f"User authentication: {result_status}",
            event_details=additional_details,
            result_status=result_status,
            compliance_tags=[ComplianceStandard.SOX, ComplianceStandard.GDPR, ComplianceStandard.PCI_DSS],
            sensitive_data_involved=True,
            data_classification="confidential"
        )

    async def log_data_access_event(
        self,
        user_id: str,
        resource_accessed: str,
        action_performed: str,
        result_status: str,
        session_id: str = None,
        additional_details: Dict[str, Any] = None
    ):
        """Log data access events with compliance tracking"""
        await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            service_name="data_service",
            component="data_access",
            user_id=user_id,
            session_id=session_id,
            event_description=f"Data access: {action_performed} on {resource_accessed}",
            event_details=additional_details,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            result_status=result_status,
            compliance_tags=[ComplianceStandard.GDPR, ComplianceStandard.SOX],
            sensitive_data_involved=True,
            data_classification="restricted"
        )

    async def log_security_incident(
        self,
        incident_type: str,
        severity: AuditSeverity,
        description: str,
        affected_resources: List[str],
        incident_details: Dict[str, Any],
        user_id: str = None,
        ip_address: str = None
    ):
        """Log security incidents with high-priority tracking"""
        await self.log_event(
            event_type=AuditEventType.SECURITY_INCIDENT,
            severity=severity,
            service_name="security_service",
            component="incident_detection",
            user_id=user_id,
            ip_address=ip_address,
            event_description=f"Security incident: {incident_type} - {description}",
            event_details={
                "incident_type": incident_type,
                "affected_resources": affected_resources,
                **incident_details
            },
            result_status="detected",
            compliance_tags=[ComplianceStandard.ISO_27001, ComplianceStandard.NIST],
            sensitive_data_involved=True,
            data_classification="secret"
        )

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        return self.storage.query_events(query)

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        return self.storage.get_event_statistics()

    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specific standard"""

        # Query events for compliance period
        query = AuditQuery(
            start_time=start_date,
            end_time=end_date,
            compliance_standards=[standard]
        )

        events = self.query_events(query)

        # Analyze compliance
        rules = self.compliance_rules.get(standard, {})
        required_events = rules.get("required_events", [])

        compliance_status = {}
        for event_type in required_events:
            type_events = [e for e in events if e.event_type == event_type]
            compliance_status[event_type.value] = {
                "required": True,
                "events_found": len(type_events),
                "compliant": len(type_events) > 0
            }

        # Calculate overall compliance score
        compliant_types = sum(1 for status in compliance_status.values() if status["compliant"])
        total_required = len(required_events)
        compliance_score = (compliant_types / total_required * 100) if total_required > 0 else 100

        return {
            "standard": standard.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_events": len(events),
            "compliance_score": round(compliance_score, 2),
            "compliance_details": compliance_status,
            "sensitive_events": len([e for e in events if e.sensitive_data_involved]),
            "error_events": len([e for e in events if e.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]]),
            "generated_at": datetime.now().isoformat()
        }

    def set_correlation_context(self, key: str, value: Any):
        """Set correlation context for event correlation"""
        self.correlation_context[key] = value

    def get_correlation_context(self, key: str) -> Any:
        """Get correlation context value"""
        return self.correlation_context.get(key)

    def clear_correlation_context(self):
        """Clear correlation context"""
        self.correlation_context.clear()

# Global audit system instance
audit_system = None

def get_audit_system() -> EnhancedAuditSystem:
    """Get or create global audit system instance"""
    global audit_system
    if audit_system is None:
        error_handler = get_error_handler("audit_system")
        audit_system = EnhancedAuditSystem(error_handler)
    return audit_system

# Convenience functions for common audit operations
async def audit_system_start(service_name: str, version: str):
    """Audit system startup"""
    system = get_audit_system()
    await system.log_system_event(
        AuditEventType.SYSTEM_START,
        f"System started: {service_name} v{version}",
        {"service_name": service_name, "version": version}
    )

async def audit_system_stop(service_name: str):
    """Audit system shutdown"""
    system = get_audit_system()
    await system.log_system_event(
        AuditEventType.SYSTEM_STOP,
        f"System stopped: {service_name}"
    )

async def audit_authentication(user_id: str, success: bool, ip_address: str, user_agent: str):
    """Audit authentication event"""
    system = get_audit_system()
    await system.log_authentication_event(
        user_id=user_id,
        result_status="success" if success else "failed",
        ip_address=ip_address,
        user_agent=user_agent
    )

async def audit_data_access(user_id: str, resource: str, action: str, success: bool):
    """Audit data access event"""
    system = get_audit_system()
    await system.log_data_access_event(
        user_id=user_id,
        resource_accessed=resource,
        action_performed=action,
        result_status="success" if success else "failed"
    )

if __name__ == "__main__":
    async def demo_audit_system():
        """Demonstrate enhanced audit system capabilities"""
        print("📋 XORB Enhanced Audit System Demo")

        # Initialize audit system
        system = get_audit_system()

        # Log various types of events
        print("\n📝 Logging audit events...")

        # System startup
        await audit_system_start("xorb_demo", "2.0.0")

        # Authentication events
        await audit_authentication("user123", True, "192.168.1.100", "Mozilla/5.0")
        await audit_authentication("user456", False, "192.168.1.101", "curl/7.68.0")

        # Data access events
        await audit_data_access("user123", "/api/v1/agents", "GET", True)
        await audit_data_access("user123", "/api/v1/data", "POST", True)

        # Security incident
        await system.log_security_incident(
            incident_type="brute_force_attempt",
            severity=AuditSeverity.ERROR,
            description="Multiple failed login attempts detected",
            affected_resources=["/auth/login"],
            incident_details={"attempts": 5, "timeframe": "5 minutes"},
            ip_address="192.168.1.101"
        )

        # Error event (will be automatically logged by error handler integration)
        try:
            raise ValueError("Demo error for audit testing")
        except Exception as e:
            system.error_handler.handle_error(
                e, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM,
                context={"demo": "audit_integration"}
            )

        print("✅ Events logged successfully")

        # Query events
        print("\n🔍 Querying audit events...")
        query = AuditQuery(
            start_time=datetime.now() - timedelta(hours=1),
            limit=10
        )
        events = system.query_events(query)
        print(f"Found {len(events)} events in the last hour")

        # Show event details
        for event in events[:3]:
            print(f"  - {event.timestamp.strftime('%H:%M:%S')} | {event.event_type.value} | {event.severity.value} | {event.event_description}")

        # Generate compliance report
        print("\n📊 Generating compliance report...")
        report = await system.generate_compliance_report(
            ComplianceStandard.GDPR,
            datetime.now() - timedelta(hours=1),
            datetime.now()
        )
        print(f"GDPR Compliance Score: {report['compliance_score']}%")

        # Show statistics
        print("\n📈 Audit Statistics:")
        stats = system.get_audit_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # System shutdown
        await audit_system_stop("xorb_demo")

        print("\n✅ Enhanced audit system demo completed!")

    asyncio.run(demo_audit_system())
