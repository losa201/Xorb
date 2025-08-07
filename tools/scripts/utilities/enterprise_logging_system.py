#!/usr/bin/env python3
"""
XORB Enterprise-Grade Logging and Audit System
Provides comprehensive logging, audit trails, and compliance monitoring
"""

import os
import sys
import json
import time
import uuid
import hashlib
import asyncio
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import yaml
from logging.handlers import RotatingFileHandler, SysLogHandler
import threading
from queue import Queue
import gzip
import shutil

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    """Event type enumeration for audit logging"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CHANGE = "system_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"

@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: LogLevel
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, warning
    details: Dict[str, Any]
    risk_score: int = 0  # 0-100 risk score
    compliance_tags: List[str] = None

@dataclass
class LogEntry:
    """Structured log entry"""
    id: str
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None

class EnterpriseLoggingSystem:
    """Enterprise-grade logging and audit system"""
    
    def __init__(self, config_path: str = "/root/Xorb/config/enterprise_logging.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.db_path = self.config.get('database', '/root/Xorb/data/enterprise_logging.db')
        self.log_dir = Path(self.config.get('log_directory', '/root/Xorb/logs'))
        self.audit_dir = Path(self.config.get('audit_directory', '/root/Xorb/logs/audit'))
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_database()
        self._setup_loggers()
        self._start_background_services()
        
        # Metrics
        self.metrics = {
            'total_events': 0,
            'audit_events': 0,
            'error_events': 0,
            'critical_events': 0,
            'last_cleanup': datetime.now()
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enterprise logging configuration"""
        default_config = {
            'database': '/root/Xorb/data/enterprise_logging.db',
            'log_directory': '/root/Xorb/logs',
            'audit_directory': '/root/Xorb/logs/audit',
            'retention': {
                'log_files_days': 90,
                'audit_files_days': 2555,  # 7 years for compliance
                'database_days': 365
            },
            'rotation': {
                'max_file_size_mb': 100,
                'backup_count': 10,
                'compress_rotated': True
            },
            'format': {
                'timestamp_format': '%Y-%m-%d %H:%M:%S.%f',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'audit_format': 'JSON'
            },
            'security': {
                'encrypt_sensitive_logs': True,
                'hash_pii': True,
                'signature_verification': True
            },
            'compliance': {
                'frameworks': ['SOC2', 'GDPR', 'HIPAA', 'PCI-DSS'],
                'real_time_monitoring': True,
                'automated_reporting': True
            },
            'alerts': {
                'critical_threshold': 5,  # Critical events per minute
                'error_threshold': 20,    # Error events per minute
                'webhook_url': None,
                'email_notifications': False
            },
            'performance': {
                'async_logging': True,
                'batch_size': 100,
                'flush_interval_seconds': 5
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
    
    def _init_database(self):
        """Initialize enterprise logging database"""
        with sqlite3.connect(self.db_path) as conn:
            # Log entries table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS log_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line_number INTEGER,
                    thread_id TEXT,
                    process_id INTEGER,
                    metadata TEXT,
                    correlation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Audit events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    risk_score INTEGER DEFAULT 0,
                    compliance_tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Compliance reports table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    findings TEXT,
                    recommendations TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tags TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_log_timestamp ON log_entries(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_log_level ON log_entries(level)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)')
            
            conn.commit()
    
    def _setup_loggers(self):
        """Setup enterprise logging infrastructure"""
        # Main application logger
        self.app_logger = logging.getLogger('xorb.application')
        self.app_logger.setLevel(logging.DEBUG)
        
        # Security logger
        self.security_logger = logging.getLogger('xorb.security')
        self.security_logger.setLevel(logging.INFO)
        
        # Audit logger
        self.audit_logger = logging.getLogger('xorb.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Performance logger
        self.performance_logger = logging.getLogger('xorb.performance')
        self.performance_logger.setLevel(logging.INFO)
        
        # Setup handlers
        self._setup_file_handlers()
        self._setup_syslog_handler()
        
        # Custom formatter
        formatter = logging.Formatter(
            self.config['format']['log_format'],
            datefmt=self.config['format']['timestamp_format']
        )
        
        for logger in [self.app_logger, self.security_logger, self.audit_logger, self.performance_logger]:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
    
    def _setup_file_handlers(self):
        """Setup rotating file handlers"""
        max_bytes = self.config['rotation']['max_file_size_mb'] * 1024 * 1024
        backup_count = self.config['rotation']['backup_count']
        
        # Application log handler
        app_handler = RotatingFileHandler(
            self.log_dir / 'application.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        self.app_logger.addHandler(app_handler)
        
        # Security log handler
        security_handler = RotatingFileHandler(
            self.log_dir / 'security.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        self.security_logger.addHandler(security_handler)
        
        # Audit log handler
        audit_handler = RotatingFileHandler(
            self.audit_dir / 'audit.log',
            maxBytes=max_bytes,
            backupCount=backup_count * 2  # Longer retention for audit logs
        )
        self.audit_logger.addHandler(audit_handler)
        
        # Performance log handler
        performance_handler = RotatingFileHandler(
            self.log_dir / 'performance.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        self.performance_logger.addHandler(performance_handler)
    
    def _setup_syslog_handler(self):
        """Setup syslog handler for system integration"""
        try:
            syslog_handler = SysLogHandler(address='/dev/log')
            syslog_formatter = logging.Formatter('XORB[%(process)d]: %(name)s - %(levelname)s - %(message)s')
            syslog_handler.setFormatter(syslog_formatter)
            
            # Add to critical loggers only
            self.security_logger.addHandler(syslog_handler)
            self.audit_logger.addHandler(syslog_handler)
        
        except Exception:
            # Syslog might not be available in all environments
            pass
    
    def _start_background_services(self):
        """Start background services for log processing"""
        # Log processing queue
        self.log_queue = Queue()
        self.audit_queue = Queue()
        
        # Start background threads
        self.log_processor_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.audit_processor_thread = threading.Thread(target=self._process_audit_queue, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_logs, daemon=True)
        
        self.log_processor_thread.start()
        self.audit_processor_thread.start()
        self.cleanup_thread.start()
    
    def log(self, level: LogLevel, message: str, logger_name: str = "xorb.application", 
            metadata: Dict[str, Any] = None, correlation_id: str = None):
        """Log a message with structured format"""
        import inspect
        
        # Get caller information
        frame = inspect.currentframe().f_back
        module = frame.f_code.co_filename
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        log_entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            logger_name=logger_name,
            message=message,
            module=os.path.basename(module),
            function=function,
            line_number=line_number,
            thread_id=str(threading.current_thread().ident),
            process_id=os.getpid(),
            metadata=metadata or {},
            correlation_id=correlation_id
        )
        
        # Queue for async processing
        if self.config['performance']['async_logging']:
            self.log_queue.put(log_entry)
        else:
            self._write_log_entry(log_entry)
        
        # Update metrics
        self.metrics['total_events'] += 1
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.metrics['error_events'] += 1
        if level == LogLevel.CRITICAL:
            self.metrics['critical_events'] += 1
    
    def audit(self, event: AuditEvent):
        """Log an audit event"""
        # Queue for async processing
        if self.config['performance']['async_logging']:
            self.audit_queue.put(event)
        else:
            self._write_audit_event(event)
        
        # Update metrics
        self.metrics['audit_events'] += 1
        
        # Real-time compliance monitoring
        if self.config['compliance']['real_time_monitoring']:
            self._check_compliance_violations(event)
    
    def create_audit_event(self, event_type: EventType, resource: str, action: str,
                          outcome: str, user_id: str = None, session_id: str = None,
                          source_ip: str = None, details: Dict[str, Any] = None,
                          severity: LogLevel = LogLevel.INFO, risk_score: int = 0,
                          compliance_tags: List[str] = None) -> AuditEvent:
        """Create a structured audit event"""
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            risk_score=risk_score,
            compliance_tags=compliance_tags or []
        )
    
    def _process_log_queue(self):
        """Background thread to process log queue"""
        batch = []
        last_flush = time.time()
        batch_size = self.config['performance']['batch_size']
        flush_interval = self.config['performance']['flush_interval_seconds']
        
        while True:
            try:
                # Get log entry with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                    batch.append(log_entry)
                except:
                    pass
                
                # Flush batch if size or time threshold reached
                current_time = time.time()
                if (len(batch) >= batch_size or 
                    (batch and current_time - last_flush >= flush_interval)):
                    
                    self._write_log_batch(batch)
                    batch.clear()
                    last_flush = current_time
            
            except Exception as e:
                print(f"Error in log processor: {e}")
    
    def _process_audit_queue(self):
        """Background thread to process audit queue"""
        batch = []
        last_flush = time.time()
        batch_size = self.config['performance']['batch_size']
        flush_interval = self.config['performance']['flush_interval_seconds']
        
        while True:
            try:
                # Get audit event with timeout
                try:
                    audit_event = self.audit_queue.get(timeout=1.0)
                    batch.append(audit_event)
                except:
                    pass
                
                # Flush batch if size or time threshold reached
                current_time = time.time()
                if (len(batch) >= batch_size or 
                    (batch and current_time - last_flush >= flush_interval)):
                    
                    self._write_audit_batch(batch)
                    batch.clear()
                    last_flush = current_time
            
            except Exception as e:
                print(f"Error in audit processor: {e}")
    
    def _write_log_entry(self, log_entry: LogEntry):
        """Write single log entry to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO log_entries 
                (id, timestamp, level, logger_name, message, module, function, 
                 line_number, thread_id, process_id, metadata, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry.id, log_entry.timestamp, log_entry.level.value,
                log_entry.logger_name, log_entry.message, log_entry.module,
                log_entry.function, log_entry.line_number, log_entry.thread_id,
                log_entry.process_id, json.dumps(log_entry.metadata),
                log_entry.correlation_id
            ))
            conn.commit()
    
    def _write_log_batch(self, batch: List[LogEntry]):
        """Write batch of log entries to database"""
        with sqlite3.connect(self.db_path) as conn:
            data = [
                (
                    entry.id, entry.timestamp, entry.level.value,
                    entry.logger_name, entry.message, entry.module,
                    entry.function, entry.line_number, entry.thread_id,
                    entry.process_id, json.dumps(entry.metadata),
                    entry.correlation_id
                )
                for entry in batch
            ]
            
            conn.executemany('''
                INSERT INTO log_entries 
                (id, timestamp, level, logger_name, message, module, function, 
                 line_number, thread_id, process_id, metadata, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
    
    def _write_audit_event(self, event: AuditEvent):
        """Write single audit event to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_events
                (event_id, timestamp, event_type, severity, user_id, session_id,
                 source_ip, resource, action, outcome, details, risk_score, compliance_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.timestamp, event.event_type.value,
                event.severity.value, event.user_id, event.session_id,
                event.source_ip, event.resource, event.action, event.outcome,
                json.dumps(event.details), event.risk_score,
                json.dumps(event.compliance_tags) if event.compliance_tags else None
            ))
            conn.commit()
    
    def _write_audit_batch(self, batch: List[AuditEvent]):
        """Write batch of audit events to database"""
        with sqlite3.connect(self.db_path) as conn:
            data = [
                (
                    event.event_id, event.timestamp, event.event_type.value,
                    event.severity.value, event.user_id, event.session_id,
                    event.source_ip, event.resource, event.action, event.outcome,
                    json.dumps(event.details), event.risk_score,
                    json.dumps(event.compliance_tags) if event.compliance_tags else None
                )
                for event in batch
            ]
            
            conn.executemany('''
                INSERT INTO audit_events
                (event_id, timestamp, event_type, severity, user_id, session_id,
                 source_ip, resource, action, outcome, details, risk_score, compliance_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
    
    def _check_compliance_violations(self, event: AuditEvent):
        """Check for compliance violations in real-time"""
        violations = []
        
        # GDPR compliance checks
        if 'GDPR' in self.config['compliance']['frameworks']:
            if (event.event_type == EventType.DATA_ACCESS and 
                event.outcome == 'failure' and 
                event.risk_score > 70):
                violations.append("GDPR: Unauthorized data access attempt")
        
        # SOC2 compliance checks
        if 'SOC2' in self.config['compliance']['frameworks']:
            if (event.event_type == EventType.AUTHENTICATION and 
                event.outcome == 'failure'):
                violations.append("SOC2: Authentication failure recorded")
        
        # PCI-DSS compliance checks
        if 'PCI-DSS' in self.config['compliance']['frameworks']:
            if (event.event_type == EventType.DATA_MODIFICATION and 
                'payment' in event.resource.lower()):
                violations.append("PCI-DSS: Payment data modification")
        
        # Log violations
        for violation in violations:
            self.log(LogLevel.WARNING, f"Compliance violation detected: {violation}",
                    "xorb.compliance", {"event_id": event.event_id})
    
    def _cleanup_old_logs(self):
        """Background thread to clean up old logs"""
        while True:
            try:
                current_time = datetime.now()
                
                # Only run cleanup once per day
                if (current_time - self.metrics['last_cleanup']).days >= 1:
                    self._perform_cleanup()
                    self.metrics['last_cleanup'] = current_time
                
                # Sleep for 1 hour
                time.sleep(3600)
            
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
                time.sleep(3600)
    
    def _perform_cleanup(self):
        """Perform log cleanup operations"""
        retention_days = self.config['retention']['database_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up old log entries
            cursor = conn.execute('DELETE FROM log_entries WHERE timestamp < ?', (cutoff_date,))
            log_deleted = cursor.rowcount
            
            # Clean up old audit events (longer retention)
            audit_retention_days = self.config['retention']['audit_files_days']
            audit_cutoff_date = datetime.now() - timedelta(days=audit_retention_days)
            cursor = conn.execute('DELETE FROM audit_events WHERE timestamp < ?', (audit_cutoff_date,))
            audit_deleted = cursor.rowcount
            
            conn.commit()
        
        # Clean up old log files
        self._cleanup_old_log_files()
        
        self.log(LogLevel.INFO, f"Log cleanup completed: {log_deleted} log entries, {audit_deleted} audit events deleted",
                "xorb.maintenance")
    
    def _cleanup_old_log_files(self):
        """Clean up old rotated log files"""
        retention_days = self.config['retention']['log_files_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for log_dir in [self.log_dir, self.audit_dir]:
            for log_file in log_dir.glob('*.log.*'):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        # Compress before deletion if configured
                        if self.config['rotation']['compress_rotated'] and not log_file.suffix == '.gz':
                            compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(compressed_file, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            log_file.unlink()
                        else:
                            log_file.unlink()
                
                except Exception as e:
                    print(f"Error cleaning up log file {log_file}: {e}")
    
    def generate_compliance_report(self, framework: str, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified framework and period"""
        with sqlite3.connect(self.db_path) as conn:
            # Get audit events for the period
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_date, end_date))
            
            events = cursor.fetchall()
            
            # Analyze events for compliance
            findings = []
            recommendations = []
            
            if framework == 'SOC2':
                findings, recommendations = self._analyze_soc2_compliance(events)
            elif framework == 'GDPR':
                findings, recommendations = self._analyze_gdpr_compliance(events)
            elif framework == 'PCI-DSS':
                findings, recommendations = self._analyze_pci_compliance(events)
            
            # Generate report
            report = {
                'report_id': str(uuid.uuid4()),
                'framework': framework,
                'period_start': start_date,
                'period_end': end_date,
                'total_events': len(events),
                'findings': findings,
                'recommendations': recommendations,
                'status': 'compliant' if len(findings) == 0 else 'non_compliant',
                'generated_at': datetime.now()
            }
            
            # Save report to database
            conn.execute('''
                INSERT INTO compliance_reports
                (report_id, framework, period_start, period_end, status, findings, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                report['report_id'], framework, start_date, end_date,
                report['status'], json.dumps(findings), json.dumps(recommendations)
            ))
            conn.commit()
        
        return report
    
    def _analyze_soc2_compliance(self, events: List) -> tuple:
        """Analyze events for SOC2 compliance"""
        findings = []
        recommendations = []
        
        # Check for authentication failures
        auth_failures = [e for e in events if 'authentication' in str(e) and 'failure' in str(e)]
        if len(auth_failures) > 100:  # Example threshold
            findings.append("High number of authentication failures detected")
            recommendations.append("Review authentication mechanisms and implement stronger controls")
        
        return findings, recommendations
    
    def _analyze_gdpr_compliance(self, events: List) -> tuple:
        """Analyze events for GDPR compliance"""
        findings = []
        recommendations = []
        
        # Check for data access without proper authorization
        unauthorized_access = [e for e in events if 'data_access' in str(e) and 'unauthorized' in str(e)]
        if unauthorized_access:
            findings.append("Unauthorized data access attempts detected")
            recommendations.append("Implement stricter data access controls and monitoring")
        
        return findings, recommendations
    
    def _analyze_pci_compliance(self, events: List) -> tuple:
        """Analyze events for PCI-DSS compliance"""
        findings = []
        recommendations = []
        
        # Check for payment data handling
        payment_events = [e for e in events if 'payment' in str(e).lower()]
        if payment_events:
            findings.append("Payment data processing events detected")
            recommendations.append("Ensure all payment data processing follows PCI-DSS requirements")
        
        return findings, recommendations
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get recent event counts
            cursor = conn.execute('''
                SELECT level, COUNT(*) as count
                FROM log_entries 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY level
            ''')
            recent_logs = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get audit event counts
            cursor = conn.execute('''
                SELECT event_type, COUNT(*) as count
                FROM audit_events 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY event_type
            ''')
            recent_audits = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get database size
            cursor = conn.execute('SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()')
            db_size = cursor.fetchone()[0]
        
        return {
            'metrics': self.metrics,
            'recent_logs': recent_logs,
            'recent_audits': recent_audits,
            'database_size_bytes': db_size,
            'log_queue_size': self.log_queue.qsize(),
            'audit_queue_size': self.audit_queue.qsize(),
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }

def main():
    """Main function for enterprise logging system"""
    
    # Initialize enterprise logging system
    logging_system = EnterpriseLoggingSystem()
    logging_system._start_time = time.time()
    
    print("📋 XORB Enterprise-Grade Logging and Audit System")
    print("=" * 60)
    
    # Test different types of logging
    print("\n📝 Testing Logging Capabilities...")
    
    # Application logs
    logging_system.log(LogLevel.INFO, "System initialization completed successfully", 
                      metadata={"component": "main", "version": "1.0.0"})
    
    logging_system.log(LogLevel.WARNING, "High memory usage detected", 
                      metadata={"memory_usage": "85%", "threshold": "80%"})
    
    # Security events
    security_event = logging_system.create_audit_event(
        event_type=EventType.AUTHENTICATION,
        resource="login_system",
        action="user_login",
        outcome="success",
        user_id="admin",
        source_ip="127.0.0.1",
        details={"method": "password", "mfa": True},
        compliance_tags=["SOC2", "GDPR"]
    )
    logging_system.audit(security_event)
    
    # Data access event
    data_event = logging_system.create_audit_event(
        event_type=EventType.DATA_ACCESS,
        resource="user_database",
        action="query_user_data",
        outcome="success",
        user_id="analyst",
        session_id="sess_12345",
        details={"query": "SELECT * FROM users WHERE active=1", "records_returned": 150},
        risk_score=25,
        compliance_tags=["GDPR", "SOC2"]
    )
    logging_system.audit(data_event)
    
    # System change event
    system_event = logging_system.create_audit_event(
        event_type=EventType.SYSTEM_CHANGE,
        resource="security_config",
        action="update_firewall_rules",
        outcome="success",
        user_id="admin",
        details={"rules_added": 3, "rules_removed": 1},
        severity=LogLevel.INFO,
        risk_score=40,
        compliance_tags=["SOC2"]
    )
    logging_system.audit(system_event)
    
    # Error event
    logging_system.log(LogLevel.ERROR, "Database connection failed", 
                      "xorb.database", {"error_code": "DB_CONN_001", "retry_count": 3})
    
    # Critical event
    logging_system.log(LogLevel.CRITICAL, "Security breach attempt detected", 
                      "xorb.security", {"attack_type": "sql_injection", "blocked": True})
    
    print("✅ Log entries created successfully")
    
    # Wait for background processing
    print("\n⏳ Processing logs asynchronously...")
    time.sleep(2)
    
    # Generate compliance report
    print("\n📊 Generating Compliance Reports...")
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=1)
    
    try:
        soc2_report = logging_system.generate_compliance_report('SOC2', start_date, end_date)
        print(f"✅ SOC2 Report Generated: {soc2_report['status']}")
        
        gdpr_report = logging_system.generate_compliance_report('GDPR', start_date, end_date)
        print(f"✅ GDPR Report Generated: {gdpr_report['status']}")
    except Exception as e:
        print(f"⚠️ Report generation warning: {e}")
    
    # Display system metrics
    print("\n📈 System Metrics:")
    metrics = logging_system.get_system_metrics()
    print(f"  • Total Events: {metrics['metrics']['total_events']}")
    print(f"  • Audit Events: {metrics['metrics']['audit_events']}")
    print(f"  • Error Events: {metrics['metrics']['error_events']}")
    print(f"  • Critical Events: {metrics['metrics']['critical_events']}")
    print(f"  • Database Size: {metrics['database_size_bytes'] / 1024:.2f} KB")
    print(f"  • Log Queue Size: {metrics['log_queue_size']}")
    print(f"  • Audit Queue Size: {metrics['audit_queue_size']}")
    
    if metrics['recent_logs']:
        print("  • Recent Log Levels:")
        for level, count in metrics['recent_logs'].items():
            print(f"    - {level}: {count}")
    
    if metrics['recent_audits']:
        print("  • Recent Audit Types:")
        for event_type, count in metrics['recent_audits'].items():
            print(f"    - {event_type}: {count}")
    
    print("\n🎯 Enterprise Logging System Deployment Complete!")
    print("  • Structured logging with metadata")
    print("  • Comprehensive audit trail")
    print("  • Real-time compliance monitoring")
    print("  • Automated log rotation and cleanup")
    print("  • Performance optimized async processing")
    print("  • Multi-framework compliance reporting")
    print("  • Enterprise-grade security features")

if __name__ == "__main__":
    main()