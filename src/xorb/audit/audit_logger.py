"""
Comprehensive Audit Logging System
Enterprise-grade audit logging with compliance support, tamper protection, and real-time monitoring
"""

import asyncio
import json
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import gzip
import base64
from pathlib import Path
import aiofiles
import aioredis
import asyncpg
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"

    # Data access events
    DATA_READ = "data_read"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    BACKUP_CREATE = "backup_create"
    BACKUP_RESTORE = "backup_restore"

    # Security events
    VULNERABILITY_DETECTED = "vulnerability_detected"
    INCIDENT_CREATED = "incident_created"
    INCIDENT_RESOLVED = "incident_resolved"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_SCAN = "security_scan"

    # Administrative events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_SUSPENDED = "user_suspended"
    USER_ACTIVATED = "user_activated"

    # API events
    API_REQUEST = "api_request"
    API_ERROR = "api_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Tenant events
    TENANT_CREATED = "tenant_created"
    TENANT_UPDATED = "tenant_updated"
    TENANT_SUSPENDED = "tenant_suspended"
    TENANT_DELETED = "tenant_deleted"


class AuditSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.API_REQUEST
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Actor information
    user_id: Optional[str] = None
    username: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Event details
    resource: Optional[str] = None
    action: Optional[str] = None
    resource_id: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.INFO
    status: AuditStatus = AuditStatus.SUCCESS

    # Event data
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None

    # Context information
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    source_system: str = "xorb"
    environment: str = "production"

    # Compliance and regulations
    compliance_frameworks: List[str] = field(default_factory=list)
    data_classification: Optional[str] = None
    retention_period_days: int = 2555  # 7 years default for compliance

    # Technical details
    duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Integrity protection
    checksum: Optional[str] = None
    signature: Optional[str] = None

    def calculate_checksum(self, secret_key: str) -> str:
        """Calculate HMAC checksum for integrity verification"""
        data = self._get_signable_data()
        return hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def _get_signable_data(self) -> str:
        """Get canonical string representation for signing"""
        # Create deterministic representation excluding checksum/signature
        signable_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "resource": self.resource,
            "action": self.action,
            "status": self.status.value,
            "description": self.description,
            "details": self.details
        }

        return json.dumps(signable_data, sort_keys=True, separators=(',', ':'))


@dataclass
class AuditConfiguration:
    """Audit logging configuration"""
    # Storage settings
    database_retention_days: int = 90  # Hot storage
    archive_retention_days: int = 2555  # Cold storage (7 years)

    # Logging levels
    minimum_severity: AuditSeverity = AuditSeverity.INFO
    log_successful_reads: bool = False
    log_failed_requests: bool = True
    log_system_events: bool = True

    # Integrity protection
    enable_checksums: bool = True
    enable_signatures: bool = False
    secret_key: Optional[str] = None
    signing_key_path: Optional[str] = None

    # Performance settings
    batch_size: int = 100
    flush_interval_seconds: int = 30
    async_processing: bool = True

    # Storage backends
    enable_database_storage: bool = True
    enable_file_storage: bool = True
    enable_syslog: bool = False
    enable_elasticsearch: bool = False

    # Compliance settings
    enable_gdpr_compliance: bool = True
    enable_sox_compliance: bool = False
    enable_pci_compliance: bool = False
    automatic_anonymization_days: int = 365

    # Alerting
    enable_real_time_alerts: bool = True
    alert_on_failures: bool = True
    alert_on_violations: bool = True


class AuditLogger:
    """Comprehensive audit logging system"""

    def __init__(self, config: AuditConfiguration):
        self.config = config
        self.event_queue: List[AuditEvent] = []
        self.queue_lock = asyncio.Lock()

        # Storage backends
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None

        # Cryptographic keys
        self.signing_key: Optional[rsa.RSAPrivateKey] = None
        self.verification_key: Optional[rsa.RSAPublicKey] = None

        # Background tasks
        self.flush_task: Optional[asyncio.Task] = None
        self.archival_task: Optional[asyncio.Task] = None

        # Metrics
        self.events_logged = 0
        self.events_failed = 0
        self.last_flush_time = datetime.utcnow()

    async def initialize(self):
        """Initialize audit logging system"""
        logger.info("Initializing Audit Logging System...")

        # Initialize database storage
        if self.config.enable_database_storage:
            await self._initialize_database()

        # Initialize Redis for real-time features
        await self._initialize_redis()

        # Load cryptographic keys
        if self.config.enable_signatures:
            await self._load_signing_keys()

        # Start background tasks
        if self.config.async_processing:
            self.flush_task = asyncio.create_task(self._flush_loop())

        self.archival_task = asyncio.create_task(self._archival_loop())

        logger.info("Audit Logging System initialized")

    async def _initialize_database(self):
        """Initialize database connection and tables"""
        db_dsn = "postgresql://localhost/xorb_audit"  # From config

        self.db_pool = await asyncpg.create_pool(
            db_dsn,
            min_size=5,
            max_size=20,
            command_timeout=30
        )

        # Create audit tables
        await self._create_audit_tables()

    async def _create_audit_tables(self):
        """Create audit logging database tables"""
        async with self.db_pool.acquire() as conn:
            # Main audit events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id VARCHAR(36) PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    user_id VARCHAR(36),
                    username VARCHAR(255),
                    tenant_id VARCHAR(36),
                    session_id VARCHAR(255),
                    ip_address INET,
                    user_agent TEXT,
                    resource VARCHAR(255),
                    action VARCHAR(100),
                    resource_id VARCHAR(255),
                    severity VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    description TEXT,
                    details JSONB,
                    before_state JSONB,
                    after_state JSONB,
                    request_id VARCHAR(36),
                    trace_id VARCHAR(36),
                    source_system VARCHAR(50) NOT NULL,
                    environment VARCHAR(20) NOT NULL,
                    compliance_frameworks TEXT[],
                    data_classification VARCHAR(50),
                    retention_period_days INTEGER NOT NULL,
                    duration_ms INTEGER,
                    error_code VARCHAR(50),
                    error_message TEXT,
                    stack_trace TEXT,
                    checksum VARCHAR(64),
                    signature TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Archived events table (for long-term storage)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events_archive (
                    LIKE audit_events INCLUDING ALL,
                    archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Audit statistics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_statistics (
                    date DATE PRIMARY KEY,
                    total_events BIGINT DEFAULT 0,
                    events_by_type JSONB,
                    events_by_severity JSONB,
                    events_by_tenant JSONB,
                    unique_users INTEGER DEFAULT 0,
                    unique_tenants INTEGER DEFAULT 0,
                    failed_events INTEGER DEFAULT 0,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id ON audit_events(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity);
                CREATE INDEX IF NOT EXISTS idx_audit_events_resource ON audit_events(resource);
                CREATE INDEX IF NOT EXISTS idx_audit_events_ip_address ON audit_events(ip_address);
                CREATE INDEX IF NOT EXISTS idx_audit_events_compliance ON audit_events USING GIN(compliance_frameworks);
                CREATE INDEX IF NOT EXISTS idx_audit_events_details ON audit_events USING GIN(details);
            """)

    async def _initialize_redis(self):
        """Initialize Redis for real-time features"""
        redis_url = "redis://localhost:6379/2"  # Use separate DB for audit
        self.redis_client = await aioredis.from_url(redis_url)

    async def _load_signing_keys(self):
        """Load cryptographic signing keys"""
        if self.config.signing_key_path:
            try:
                key_path = Path(self.config.signing_key_path)
                if key_path.exists():
                    async with aiofiles.open(key_path, 'rb') as f:
                        key_data = await f.read()
                        self.signing_key = serialization.load_pem_private_key(
                            key_data, password=None
                        )
                        self.verification_key = self.signing_key.public_key()
                else:
                    # Generate new key pair
                    await self._generate_signing_keys()
            except Exception as e:
                logger.error(f"Failed to load signing keys: {e}")

    async def _generate_signing_keys(self):
        """Generate new RSA signing key pair"""
        self.signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.verification_key = self.signing_key.public_key()

        # Save keys to files
        if self.config.signing_key_path:
            key_dir = Path(self.config.signing_key_path).parent
            key_dir.mkdir(parents=True, exist_ok=True)

            # Save private key
            private_pem = self.signing_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            async with aiofiles.open(self.config.signing_key_path, 'wb') as f:
                await f.write(private_pem)

            # Save public key
            public_pem = self.verification_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            public_key_path = str(self.config.signing_key_path).replace('.pem', '_public.pem')
            async with aiofiles.open(public_key_path, 'wb') as f:
                await f.write(public_pem)

        logger.info("Generated new audit signing key pair")

    async def log_event(self, event: AuditEvent) -> bool:
        """Log audit event"""
        try:
            # Apply minimum severity filter
            if event.severity.value < self.config.minimum_severity.value:
                return True

            # Add integrity protection
            if self.config.enable_checksums and self.config.secret_key:
                event.checksum = event.calculate_checksum(self.config.secret_key)

            if self.config.enable_signatures and self.signing_key:
                event.signature = await self._sign_event(event)

            # Queue for processing
            async with self.queue_lock:
                self.event_queue.append(event)

                # Immediate flush for critical events
                if event.severity == AuditSeverity.CRITICAL:
                    await self._flush_events()
                elif len(self.event_queue) >= self.config.batch_size:
                    await self._flush_events()

            # Real-time alerting
            if self.config.enable_real_time_alerts:
                await self._check_alert_conditions(event)

            self.events_logged += 1
            return True

        except Exception as e:
            logger.error(f"Failed to log audit event {event.event_id}: {e}")
            self.events_failed += 1
            return False

    async def _sign_event(self, event: AuditEvent) -> str:
        """Digitally sign audit event"""
        try:
            data = event._get_signable_data().encode()
            signature = self.signing_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Failed to sign event {event.event_id}: {e}")
            return ""

    async def _flush_events(self):
        """Flush queued events to storage"""
        if not self.event_queue:
            return

        events_to_flush = self.event_queue.copy()
        self.event_queue.clear()

        # Store in database
        if self.config.enable_database_storage and self.db_pool:
            await self._store_events_database(events_to_flush)

        # Store in files
        if self.config.enable_file_storage:
            await self._store_events_files(events_to_flush)

        # Real-time indexing in Redis
        await self._index_events_redis(events_to_flush)

        self.last_flush_time = datetime.utcnow()

    async def _store_events_database(self, events: List[AuditEvent]):
        """Store events in PostgreSQL database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for event in events:
                    values.append((
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.user_id,
                        event.username,
                        event.tenant_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.resource,
                        event.action,
                        event.resource_id,
                        event.severity.value,
                        event.status.value,
                        event.description,
                        json.dumps(event.details) if event.details else None,
                        json.dumps(event.before_state) if event.before_state else None,
                        json.dumps(event.after_state) if event.after_state else None,
                        event.request_id,
                        event.trace_id,
                        event.source_system,
                        event.environment,
                        event.compliance_frameworks,
                        event.data_classification,
                        event.retention_period_days,
                        event.duration_ms,
                        event.error_code,
                        event.error_message,
                        event.stack_trace,
                        event.checksum,
                        event.signature
                    ))

                # Batch insert
                await conn.executemany("""
                    INSERT INTO audit_events (
                        event_id, event_type, timestamp, user_id, username, tenant_id,
                        session_id, ip_address, user_agent, resource, action, resource_id,
                        severity, status, description, details, before_state, after_state,
                        request_id, trace_id, source_system, environment, compliance_frameworks,
                        data_classification, retention_period_days, duration_ms, error_code,
                        error_message, stack_trace, checksum, signature
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                             $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31)
                """, values)

                # Update statistics
                await self._update_audit_statistics(events)

        except Exception as e:
            logger.error(f"Failed to store events in database: {e}")

    async def _store_events_files(self, events: List[AuditEvent]):
        """Store events in log files"""
        try:
            log_dir = Path("./logs/audit")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Group events by date for daily log files
            events_by_date = {}
            for event in events:
                date_key = event.timestamp.strftime("%Y-%m-%d")
                if date_key not in events_by_date:
                    events_by_date[date_key] = []
                events_by_date[date_key].append(event)

            # Write to daily log files
            for date_key, date_events in events_by_date.items():
                log_file = log_dir / f"audit-{date_key}.jsonl"

                async with aiofiles.open(log_file, 'a') as f:
                    for event in date_events:
                        event_json = json.dumps(asdict(event), default=str)
                        await f.write(f"{event_json}\n")

        except Exception as e:
            logger.error(f"Failed to store events in files: {e}")

    async def _index_events_redis(self, events: List[AuditEvent]):
        """Index events in Redis for real-time queries"""
        try:
            pipe = self.redis_client.pipeline()

            for event in events:
                # Store event data
                event_key = f"audit:event:{event.event_id}"
                pipe.setex(event_key, 3600, json.dumps(asdict(event), default=str))

                # Index by user
                if event.user_id:
                    pipe.zadd(f"audit:user:{event.user_id}", {event.event_id: event.timestamp.timestamp()})
                    pipe.expire(f"audit:user:{event.user_id}", 86400)

                # Index by tenant
                if event.tenant_id:
                    pipe.zadd(f"audit:tenant:{event.tenant_id}", {event.event_id: event.timestamp.timestamp()})
                    pipe.expire(f"audit:tenant:{event.tenant_id}", 86400)

                # Index by event type
                pipe.zadd(f"audit:type:{event.event_type.value}", {event.event_id: event.timestamp.timestamp()})
                pipe.expire(f"audit:type:{event.event_type.value}", 86400)

                # Index high-severity events
                if event.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]:
                    pipe.zadd("audit:high_severity", {event.event_id: event.timestamp.timestamp()})
                    pipe.expire("audit:high_severity", 86400)

            await pipe.execute()

        except Exception as e:
            logger.error(f"Failed to index events in Redis: {e}")

    async def _update_audit_statistics(self, events: List[AuditEvent]):
        """Update daily audit statistics"""
        try:
            # Group events by date
            stats_by_date = {}

            for event in events:
                date_key = event.timestamp.date()
                if date_key not in stats_by_date:
                    stats_by_date[date_key] = {
                        'total_events': 0,
                        'events_by_type': {},
                        'events_by_severity': {},
                        'events_by_tenant': {},
                        'unique_users': set(),
                        'unique_tenants': set(),
                        'failed_events': 0
                    }

                stats = stats_by_date[date_key]
                stats['total_events'] += 1

                # Count by type
                event_type = event.event_type.value
                stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1

                # Count by severity
                severity = event.severity.value
                stats['events_by_severity'][severity] = stats['events_by_severity'].get(severity, 0) + 1

                # Count by tenant
                if event.tenant_id:
                    stats['events_by_tenant'][event.tenant_id] = stats['events_by_tenant'].get(event.tenant_id, 0) + 1
                    stats['unique_tenants'].add(event.tenant_id)

                # Count unique users
                if event.user_id:
                    stats['unique_users'].add(event.user_id)

                # Count failures
                if event.status == AuditStatus.FAILURE:
                    stats['failed_events'] += 1

            # Update database statistics
            async with self.db_pool.acquire() as conn:
                for date_key, stats in stats_by_date.items():
                    await conn.execute("""
                        INSERT INTO audit_statistics (
                            date, total_events, events_by_type, events_by_severity,
                            events_by_tenant, unique_users, unique_tenants, failed_events
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (date) DO UPDATE SET
                            total_events = audit_statistics.total_events + EXCLUDED.total_events,
                            events_by_type = EXCLUDED.events_by_type,
                            events_by_severity = EXCLUDED.events_by_severity,
                            events_by_tenant = EXCLUDED.events_by_tenant,
                            unique_users = EXCLUDED.unique_users,
                            unique_tenants = EXCLUDED.unique_tenants,
                            failed_events = audit_statistics.failed_events + EXCLUDED.failed_events,
                            updated_at = NOW()
                    """,
                        date_key,
                        stats['total_events'],
                        json.dumps(stats['events_by_type']),
                        json.dumps(stats['events_by_severity']),
                        json.dumps(stats['events_by_tenant']),
                        len(stats['unique_users']),
                        len(stats['unique_tenants']),
                        stats['failed_events']
                    )

        except Exception as e:
            logger.error(f"Failed to update audit statistics: {e}")

    async def _check_alert_conditions(self, event: AuditEvent):
        """Check if event should trigger alerts"""
        alerts = []

        # Alert on critical events
        if event.severity == AuditSeverity.CRITICAL:
            alerts.append({
                "type": "critical_event",
                "event_id": event.event_id,
                "message": f"Critical audit event: {event.event_type.value}",
                "details": event.details
            })

        # Alert on failures
        if self.config.alert_on_failures and event.status == AuditStatus.FAILURE:
            alerts.append({
                "type": "failure_event",
                "event_id": event.event_id,
                "message": f"System failure: {event.description}",
                "error_code": event.error_code
            })

        # Alert on compliance violations
        if event.event_type == AuditEventType.COMPLIANCE_VIOLATION:
            alerts.append({
                "type": "compliance_violation",
                "event_id": event.event_id,
                "message": f"Compliance violation detected: {event.description}",
                "frameworks": event.compliance_frameworks
            })

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

    async def _send_alert(self, alert: Dict[str, Any]):
        """Send real-time alert"""
        try:
            # Publish to Redis for real-time notifications
            await self.redis_client.publish("audit_alerts", json.dumps(alert))

            # Store in alerts queue
            await self.redis_client.lpush("audit_alert_queue", json.dumps(alert))
            await self.redis_client.ltrim("audit_alert_queue", 0, 999)  # Keep last 1000

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _flush_loop(self):
        """Background task to flush events periodically"""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)

                async with self.queue_lock:
                    if self.event_queue:
                        await self._flush_events()

            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(60)

    async def _archival_loop(self):
        """Background task to archive old events"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily

                if self.config.enable_database_storage and self.db_pool:
                    await self._archive_old_events()

            except Exception as e:
                logger.error(f"Error in archival loop: {e}")
                await asyncio.sleep(3600)

    async def _archive_old_events(self):
        """Archive events older than retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.database_retention_days)

            async with self.db_pool.acquire() as conn:
                # Move old events to archive table
                await conn.execute("""
                    INSERT INTO audit_events_archive
                    SELECT *, NOW() as archived_at
                    FROM audit_events
                    WHERE timestamp < $1
                """, cutoff_date)

                # Delete from main table
                deleted_count = await conn.execute("""
                    DELETE FROM audit_events WHERE timestamp < $1
                """, cutoff_date)

                logger.info(f"Archived {deleted_count} audit events older than {cutoff_date}")

                # Delete very old archived events
                final_cutoff = datetime.utcnow() - timedelta(days=self.config.archive_retention_days)
                final_deleted = await conn.execute("""
                    DELETE FROM audit_events_archive WHERE timestamp < $1
                """, final_cutoff)

                if final_deleted:
                    logger.info(f"Permanently deleted {final_deleted} audit events older than {final_cutoff}")

        except Exception as e:
            logger.error(f"Failed to archive old events: {e}")

    async def query_events(self, filters: Dict[str, Any], limit: int = 100) -> List[AuditEvent]:
        """Query audit events with filters"""
        try:
            if not self.db_pool:
                return []

            # Build query
            where_clauses = []
            params = []
            param_count = 0

            for key, value in filters.items():
                param_count += 1
                if key == "start_date":
                    where_clauses.append(f"timestamp >= ${param_count}")
                    params.append(value)
                elif key == "end_date":
                    where_clauses.append(f"timestamp <= ${param_count}")
                    params.append(value)
                elif key == "user_id":
                    where_clauses.append(f"user_id = ${param_count}")
                    params.append(value)
                elif key == "tenant_id":
                    where_clauses.append(f"tenant_id = ${param_count}")
                    params.append(value)
                elif key == "event_type":
                    where_clauses.append(f"event_type = ${param_count}")
                    params.append(value)
                elif key == "severity":
                    where_clauses.append(f"severity = ${param_count}")
                    params.append(value)
                elif key == "resource":
                    where_clauses.append(f"resource = ${param_count}")
                    params.append(value)

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            query = f"""
                SELECT * FROM audit_events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT {limit}
            """

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                events = []
                for row in rows:
                    event = AuditEvent(
                        event_id=row["event_id"],
                        event_type=AuditEventType(row["event_type"]),
                        timestamp=row["timestamp"],
                        user_id=row["user_id"],
                        username=row["username"],
                        tenant_id=row["tenant_id"],
                        session_id=row["session_id"],
                        ip_address=str(row["ip_address"]) if row["ip_address"] else None,
                        user_agent=row["user_agent"],
                        resource=row["resource"],
                        action=row["action"],
                        resource_id=row["resource_id"],
                        severity=AuditSeverity(row["severity"]),
                        status=AuditStatus(row["status"]),
                        description=row["description"],
                        details=json.loads(row["details"]) if row["details"] else {},
                        before_state=json.loads(row["before_state"]) if row["before_state"] else None,
                        after_state=json.loads(row["after_state"]) if row["after_state"] else None,
                        request_id=row["request_id"],
                        trace_id=row["trace_id"],
                        source_system=row["source_system"],
                        environment=row["environment"],
                        compliance_frameworks=row["compliance_frameworks"] or [],
                        data_classification=row["data_classification"],
                        retention_period_days=row["retention_period_days"],
                        duration_ms=row["duration_ms"],
                        error_code=row["error_code"],
                        error_message=row["error_message"],
                        stack_trace=row["stack_trace"],
                        checksum=row["checksum"],
                        signature=row["signature"]
                    )
                    events.append(event)

                return events

        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []

    async def verify_event_integrity(self, event_id: str) -> bool:
        """Verify audit event integrity"""
        try:
            events = await self.query_events({"event_id": event_id}, 1)
            if not events:
                return False

            event = events[0]

            # Verify checksum
            if event.checksum and self.config.secret_key:
                expected_checksum = event.calculate_checksum(self.config.secret_key)
                if event.checksum != expected_checksum:
                    logger.warning(f"Checksum mismatch for event {event_id}")
                    return False

            # Verify signature
            if event.signature and self.verification_key:
                try:
                    signature_bytes = base64.b64decode(event.signature)
                    data = event._get_signable_data().encode()

                    self.verification_key.verify(
                        signature_bytes,
                        data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                except Exception:
                    logger.warning(f"Signature verification failed for event {event_id}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to verify event integrity for {event_id}: {e}")
            return False

    async def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit statistics"""
        try:
            if not self.db_pool:
                return {}

            start_date = datetime.utcnow().date() - timedelta(days=days)

            async with self.db_pool.acquire() as conn:
                # Get summary statistics
                stats = await conn.fetchrow("""
                    SELECT
                        SUM(total_events) as total_events,
                        SUM(failed_events) as failed_events,
                        AVG(unique_users) as avg_users_per_day,
                        AVG(unique_tenants) as avg_tenants_per_day
                    FROM audit_statistics
                    WHERE date >= $1
                """, start_date)

                # Get trends
                daily_stats = await conn.fetch("""
                    SELECT date, total_events, failed_events, unique_users, unique_tenants
                    FROM audit_statistics
                    WHERE date >= $1
                    ORDER BY date
                """, start_date)

                return {
                    "period_days": days,
                    "total_events": int(stats["total_events"] or 0),
                    "failed_events": int(stats["failed_events"] or 0),
                    "success_rate": 1 - (int(stats["failed_events"] or 0) / max(int(stats["total_events"] or 0), 1)),
                    "avg_users_per_day": float(stats["avg_users_per_day"] or 0),
                    "avg_tenants_per_day": float(stats["avg_tenants_per_day"] or 0),
                    "daily_trends": [
                        {
                            "date": row["date"].isoformat(),
                            "events": row["total_events"],
                            "failures": row["failed_events"],
                            "users": row["unique_users"],
                            "tenants": row["unique_tenants"]
                        }
                        for row in daily_stats
                    ],
                    "system_stats": {
                        "events_logged": self.events_logged,
                        "events_failed": self.events_failed,
                        "last_flush": self.last_flush_time.isoformat(),
                        "queue_size": len(self.event_queue)
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}

    async def shutdown(self):
        """Shutdown audit logging system"""
        # Flush remaining events
        async with self.queue_lock:
            if self.event_queue:
                await self._flush_events()

        # Cancel background tasks
        if self.flush_task:
            self.flush_task.cancel()

        if self.archival_task:
            self.archival_task.cancel()

        # Close connections
        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Audit Logging System shutdown complete")


# Convenience functions for common audit events
async def log_authentication_event(audit_logger: AuditLogger, event_type: AuditEventType,
                                  user_id: str, username: str, ip_address: str,
                                  success: bool, details: Dict[str, Any] = None):
    """Log authentication-related audit event"""
    event = AuditEvent(
        event_type=event_type,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        severity=AuditSeverity.MEDIUM if success else AuditSeverity.HIGH,
        status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
        description=f"User {username} {event_type.value}",
        details=details or {},
        compliance_frameworks=["SOX", "GDPR"]
    )

    await audit_logger.log_event(event)


async def log_data_access_event(audit_logger: AuditLogger, event_type: AuditEventType,
                               user_id: str, resource: str, resource_id: str,
                               before_state: Dict[str, Any] = None,
                               after_state: Dict[str, Any] = None):
    """Log data access audit event"""
    event = AuditEvent(
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        resource_id=resource_id,
        severity=AuditSeverity.LOW if event_type == AuditEventType.DATA_READ else AuditSeverity.MEDIUM,
        status=AuditStatus.SUCCESS,
        description=f"Data {event_type.value} on {resource}",
        before_state=before_state,
        after_state=after_state,
        compliance_frameworks=["GDPR", "PCI_DSS"]
    )

    await audit_logger.log_event(event)


# Factory function
def create_audit_logger(config: Dict[str, Any]) -> AuditLogger:
    """Create and configure audit logger"""
    audit_config = AuditConfiguration(
        database_retention_days=config.get("database_retention_days", 90),
        archive_retention_days=config.get("archive_retention_days", 2555),
        minimum_severity=AuditSeverity(config.get("minimum_severity", "info")),
        enable_checksums=config.get("enable_checksums", True),
        enable_signatures=config.get("enable_signatures", False),
        secret_key=config.get("secret_key"),
        signing_key_path=config.get("signing_key_path"),
        batch_size=config.get("batch_size", 100),
        flush_interval_seconds=config.get("flush_interval_seconds", 30),
        enable_real_time_alerts=config.get("enable_real_time_alerts", True)
    )

    return AuditLogger(audit_config)
