#!/usr/bin/env python3
"""
XORB Centralized Error Aggregation & Alerting System
Real-time error monitoring, pattern detection, and intelligent alerting
"""

import asyncio
import json
import logging
import smtplib
import ssl
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
import uuid
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import sqlite3
import queue

# Import error handling framework
from xorb_error_handling_framework import (
    ErrorContext, ErrorCategory, ErrorSeverity, get_all_error_handlers
)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: str  # Python expression
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 10
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    custom_message: Optional[str] = None

@dataclass
class ErrorPattern:
    """Detected error pattern"""
    pattern_id: str
    pattern_hash: str
    error_signature: str
    services_affected: Set[str]
    first_occurrence: datetime
    last_occurrence: datetime
    frequency: int
    severity_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    trend: str  # "increasing", "decreasing", "stable"
    confidence_score: float
    resolution_suggestions: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str
    error_contexts: List[str]  # Error IDs
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0
    notification_sent: Dict[AlertChannel, bool] = field(default_factory=dict)

class ErrorAggregator:
    """Centralized error aggregation and analysis engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Error storage and tracking
        self.error_buffer = deque(maxlen=10000)  # Recent errors
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Pattern detection
        self.pattern_window_size = 100  # Errors to analyze for patterns
        self.pattern_threshold = 5  # Minimum occurrences to form pattern
        
        # Rate limiting and cooldowns
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_counters: Dict[str, List[datetime]] = defaultdict(list)
        
        # Background processing
        self.processing_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xorb-error-agg")
        
        # Database for persistence
        self._init_database()
        
        # Setup logging
        self.logger = logging.getLogger("xorb.error_aggregator")
        
        # Load default alert rules
        self._load_default_alert_rules()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("XORB Error Aggregator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "pattern_detection_enabled": True,
            "pattern_analysis_interval": 60,  # seconds
            "alert_processing_interval": 30,  # seconds
            "database_path": "xorb_errors.db",
            "max_errors_in_memory": 10000,
            "pattern_confidence_threshold": 0.7,
            "notification_channels": {
                "email": {
                    "enabled": True,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": ["admin@xorb.local"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#alerts"
                },
                "webhook": {
                    "enabled": True,
                    "urls": ["http://localhost:9093/api/v1/alerts"]  # AlertManager
                }
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            self.db_path = self.config.get("database_path", "xorb_errors.db")
            conn = sqlite3.connect(self.db_path)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    error_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    service_name TEXT,
                    category TEXT,
                    severity TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    pattern_hash TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    context TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_hash TEXT UNIQUE,
                    error_signature TEXT,
                    first_occurrence TEXT,
                    last_occurrence TEXT,
                    frequency INTEGER,
                    confidence_score REAL,
                    services_affected TEXT,
                    resolution_suggestions TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    severity TEXT,
                    title TEXT,
                    message TEXT,
                    triggered_at TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    error_contexts TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise e
    
    def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Triggers when error rate exceeds threshold",
                condition="error_rate_per_minute > 10",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="critical_service_error",
                name="Critical Service Error",
                description="Critical errors in core services",
                condition="severity == 'critical' and service_name in ['neural_orchestrator', 'learning_service']",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="circuit_breaker_trip",
                name="Circuit Breaker Tripped",
                description="Circuit breaker has been tripped",
                condition="'circuit breaker' in error_message.lower() and 'open' in error_message.lower()",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="database_connection_failure",
                name="Database Connection Failure",
                description="Database connectivity issues",
                condition="category == 'database' and 'connection' in error_message.lower()",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="recurring_error_pattern",
                name="Recurring Error Pattern",
                description="Same error pattern occurring frequently",
                condition="pattern_frequency > 5 and pattern_confidence > 0.8",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        self.logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Pattern detection task
        self.executor.submit(self._pattern_detection_loop)
        
        # Alert processing task
        self.executor.submit(self._alert_processing_loop)
        
        # Error collection task
        self.executor.submit(self._error_collection_loop)
        
        # Database persistence task
        self.executor.submit(self._database_persistence_loop)
    
    def ingest_error(self, error_context: ErrorContext):
        """Ingest error for aggregation and analysis"""
        try:
            # Add to processing queue
            self.processing_queue.put(error_context)
            
            # Add to buffer for immediate analysis
            self.error_buffer.append(error_context)
            
            # Update real-time metrics
            self._update_real_time_metrics(error_context)
            
            self.logger.debug(f"Error ingested: {error_context.error_id}")
            
        except Exception as e:
            self.logger.error(f"Error ingestion failed: {e}")
    
    def _update_real_time_metrics(self, error_context: ErrorContext):
        """Update real-time error metrics"""
        try:
            # Update pattern detection
            pattern_hash = self._calculate_pattern_hash(error_context)
            
            if pattern_hash in self.error_patterns:
                pattern = self.error_patterns[pattern_hash]
                pattern.frequency += 1
                pattern.last_occurrence = error_context.timestamp
                pattern.services_affected.add(error_context.service_name)
                
                # Update distributions
                severity = error_context.severity.value
                pattern.severity_distribution[severity] = pattern.severity_distribution.get(severity, 0) + 1
                
                category = error_context.category.value
                pattern.category_distribution[category] = pattern.category_distribution.get(category, 0) + 1
                
                # Update trend analysis
                self._update_pattern_trend(pattern)
                
            else:
                # Create new pattern
                pattern = ErrorPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_hash=pattern_hash,
                    error_signature=f"{error_context.error_type}:{error_context.category.value}",
                    services_affected={error_context.service_name},
                    first_occurrence=error_context.timestamp,
                    last_occurrence=error_context.timestamp,
                    frequency=1,
                    severity_distribution={error_context.severity.value: 1},
                    category_distribution={error_context.category.value: 1},
                    trend="new",
                    confidence_score=0.1
                )
                self.error_patterns[pattern_hash] = pattern
            
            # Check for immediate alert conditions
            self._check_immediate_alerts(error_context)
            
        except Exception as e:
            self.logger.error(f"Real-time metrics update failed: {e}")
    
    def _calculate_pattern_hash(self, error_context: ErrorContext) -> str:
        """Calculate hash for error pattern identification"""
        signature_components = [
            error_context.error_type,
            error_context.category.value,
            error_context.function_name,
            # Normalize error message (remove dynamic parts)
            self._normalize_error_message(error_context.error_message)
        ]
        
        signature = "|".join(signature_components)
        return hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message by removing dynamic parts"""
        import re
        
        # Remove UUIDs
        message = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', message)
        
        # Remove timestamps
        message = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', message)
        
        # Remove file paths
        message = re.sub(r'/[^\s]+\.py', 'FILE_PATH', message)
        
        # Remove numbers that might be dynamic
        message = re.sub(r'\b\d+\b', 'NUMBER', message)
        
        return message
    
    def _update_pattern_trend(self, pattern: ErrorPattern):
        """Update pattern trend analysis"""
        try:
            # Simple trend analysis based on recent frequency
            recent_errors = [
                error for error in self.error_buffer
                if (datetime.now() - error.timestamp).total_seconds() < 3600  # Last hour
                and self._calculate_pattern_hash(error) == pattern.pattern_hash
            ]
            
            if len(recent_errors) > pattern.frequency * 0.5:
                pattern.trend = "increasing"
                pattern.confidence_score = min(1.0, pattern.confidence_score + 0.1)
            elif len(recent_errors) < pattern.frequency * 0.2:
                pattern.trend = "decreasing"
                pattern.confidence_score = max(0.1, pattern.confidence_score - 0.05)
            else:
                pattern.trend = "stable"
            
            # Update resolution suggestions based on pattern
            self._update_resolution_suggestions(pattern)
            
        except Exception as e:
            self.logger.error(f"Pattern trend update failed: {e}")
    
    def _update_resolution_suggestions(self, pattern: ErrorPattern):
        """Update resolution suggestions for a pattern"""
        suggestions = []
        
        # Database-related suggestions
        if "database" in pattern.category_distribution:
            suggestions.extend([
                "Check database connection pool settings",
                "Verify database server health",
                "Review recent database schema changes",
                "Check for long-running queries"
            ])
        
        # Network-related suggestions
        if "network" in pattern.category_distribution:
            suggestions.extend([
                "Check network connectivity",
                "Review firewall rules",
                "Verify DNS resolution",
                "Check load balancer configuration"
            ])
        
        # High frequency patterns
        if pattern.frequency > 10:
            suggestions.extend([
                "Consider implementing circuit breaker",
                "Review error handling in affected code",
                "Implement exponential backoff",
                "Add more detailed logging"
            ])
        
        pattern.resolution_suggestions = list(set(suggestions))  # Remove duplicates
    
    def _check_immediate_alerts(self, error_context: ErrorContext):
        """Check for immediate alert conditions"""
        try:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule.rule_id in self.alert_cooldowns:
                    cooldown_end = self.alert_cooldowns[rule.rule_id] + timedelta(minutes=rule.cooldown_minutes)
                    if datetime.now() < cooldown_end:
                        continue
                
                # Check rate limiting
                now = datetime.now()
                hour_ago = now - timedelta(hours=1)
                recent_alerts = self.alert_counters[rule.rule_id]
                recent_alerts = [alert_time for alert_time in recent_alerts if alert_time > hour_ago]
                self.alert_counters[rule.rule_id] = recent_alerts
                
                if len(recent_alerts) >= rule.max_alerts_per_hour:
                    continue
                
                # Evaluate rule condition
                if self._evaluate_alert_condition(rule, error_context):
                    self._trigger_alert(rule, [error_context])
        
        except Exception as e:
            self.logger.error(f"Immediate alert check failed: {e}")
    
    def _evaluate_alert_condition(self, rule: AlertRule, error_context: ErrorContext) -> bool:
        """Evaluate alert rule condition"""
        try:
            # Create evaluation context
            eval_context = {
                "error_id": error_context.error_id,
                "service_name": error_context.service_name,
                "function_name": error_context.function_name,
                "error_type": error_context.error_type,
                "error_message": error_context.error_message,
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "timestamp": error_context.timestamp,
                "user_id": error_context.user_id,
                "request_id": error_context.request_id,
                
                # Aggregate metrics
                "error_rate_per_minute": self._calculate_error_rate_per_minute(),
                "pattern_frequency": self._get_pattern_frequency(error_context),
                "pattern_confidence": self._get_pattern_confidence(error_context)
            }
            
            # Safely evaluate condition
            return eval(rule.condition, {"__builtins__": {}}, eval_context)
            
        except Exception as e:
            self.logger.error(f"Alert condition evaluation failed for rule {rule.rule_id}: {e}")
            return False
    
    def _calculate_error_rate_per_minute(self) -> float:
        """Calculate current error rate per minute"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        recent_errors = [
            error for error in self.error_buffer
            if error.timestamp > minute_ago
        ]
        
        return len(recent_errors)
    
    def _get_pattern_frequency(self, error_context: ErrorContext) -> int:
        """Get frequency of pattern for this error"""
        pattern_hash = self._calculate_pattern_hash(error_context)
        pattern = self.error_patterns.get(pattern_hash)
        return pattern.frequency if pattern else 1
    
    def _get_pattern_confidence(self, error_context: ErrorContext) -> float:
        """Get confidence score of pattern for this error"""
        pattern_hash = self._calculate_pattern_hash(error_context)
        pattern = self.error_patterns.get(pattern_hash)
        return pattern.confidence_score if pattern else 0.1
    
    def _trigger_alert(self, rule: AlertRule, error_contexts: List[ErrorContext]):
        """Trigger an alert"""
        try:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                severity=rule.severity,
                title=rule.name,
                message=self._generate_alert_message(rule, error_contexts),
                error_contexts=[ctx.error_id for ctx in error_contexts],
                triggered_at=datetime.now()
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update cooldown and rate limiting
            self.alert_cooldowns[rule.rule_id] = datetime.now()
            self.alert_counters[rule.rule_id].append(datetime.now())
            
            # Send notifications
            self._send_alert_notifications(alert, rule)
            
            self.logger.warning(f"Alert triggered: {alert.title} (ID: {alert.alert_id})")
            
        except Exception as e:
            self.logger.error(f"Alert triggering failed: {e}")
    
    def _generate_alert_message(self, rule: AlertRule, error_contexts: List[ErrorContext]) -> str:
        """Generate alert message"""
        if rule.custom_message:
            return rule.custom_message
        
        if len(error_contexts) == 1:
            ctx = error_contexts[0]
            return f"""
Alert: {rule.name}

Service: {ctx.service_name}
Function: {ctx.function_name}
Error: {ctx.error_type}: {ctx.error_message}
Severity: {ctx.severity.value}
Category: {ctx.category.value}
Timestamp: {ctx.timestamp.isoformat()}

Error ID: {ctx.error_id}
Request ID: {ctx.request_id or 'N/A'}
            """.strip()
        else:
            services = set(ctx.service_name for ctx in error_contexts)
            error_types = set(ctx.error_type for ctx in error_contexts)
            
            return f"""
Alert: {rule.name}

Multiple errors detected:
- Services affected: {', '.join(services)}
- Error types: {', '.join(error_types)}
- Total errors: {len(error_contexts)}
- Time range: {min(ctx.timestamp for ctx in error_contexts).isoformat()} - {max(ctx.timestamp for ctx in error_contexts).isoformat()}

Error IDs: {', '.join(ctx.error_id for ctx in error_contexts)}
            """.strip()
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications through configured channels"""
        for channel in rule.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email_notification(alert, rule)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook_notification(alert, rule)
                elif channel == AlertChannel.SLACK:
                    self._send_slack_notification(alert, rule)
                
                alert.notification_sent[channel] = True
                
            except Exception as e:
                self.logger.error(f"Failed to send {channel.value} notification: {e}")
                alert.notification_sent[channel] = False
    
    def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Send email notification"""
        email_config = self.config["notification_channels"]["email"]
        if not email_config.get("enabled", False):
            return
        
        msg = MIMEMultipart()
        msg['From'] = email_config.get("from", "xorb-alerts@localhost")
        msg['To'] = ", ".join(email_config.get("recipients", []))
        msg['Subject'] = f"[XORB Alert] {alert.severity.value.upper()}: {alert.title}"
        
        body = f"""
XORB Platform Alert

Alert ID: {alert.alert_id}
Severity: {alert.severity.value.upper()}
Rule: {rule.name}
Triggered: {alert.triggered_at.isoformat()}

Message:
{alert.message}

Error Contexts: {len(alert.error_contexts)} errors
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email (simplified - would need proper SMTP configuration)
        self.logger.info(f"Email notification prepared for alert {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Send webhook notification (e.g., to AlertManager)"""
        webhook_config = self.config["notification_channels"]["webhook"]
        if not webhook_config.get("enabled", False):
            return
        
        payload = {
            "alerts": [{
                "alertname": rule.name,
                "status": "firing",
                "labels": {
                    "alertname": rule.name,
                    "severity": alert.severity.value,
                    "service": "xorb-platform",
                    "rule_id": rule.rule_id,
                    "alert_id": alert.alert_id
                },
                "annotations": {
                    "summary": alert.title,
                    "description": alert.message,
                    "runbook_url": "https://docs.xorb.local/runbooks/alerts"
                },
                "startsAt": alert.triggered_at.isoformat(),
                "generatorURL": f"http://localhost:8000/alerts/{alert.alert_id}"
            }]
        }
        
        for url in webhook_config.get("urls", []):
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                self.logger.info(f"Webhook notification sent to {url}")
            except requests.RequestException as e:
                self.logger.error(f"Webhook notification failed for {url}: {e}")
    
    def _send_slack_notification(self, alert: Alert, rule: AlertRule):
        """Send Slack notification"""
        slack_config = self.config["notification_channels"]["slack"]
        if not slack_config.get("enabled", False):
            return
        
        webhook_url = slack_config.get("webhook_url")
        if not webhook_url:
            return
        
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }
        
        payload = {
            "channel": slack_config.get("channel", "#alerts"),
            "username": "XORB Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": f"{alert.severity.value.upper()}: {alert.title}",
                "text": alert.message,
                "fields": [
                    {"title": "Alert ID", "value": alert.alert_id, "short": True},
                    {"title": "Rule", "value": rule.name, "short": True},
                    {"title": "Errors", "value": str(len(alert.error_contexts)), "short": True},
                    {"title": "Triggered", "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                ],
                "footer": "XORB Platform",
                "ts": int(alert.triggered_at.timestamp())
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Slack notification sent")
        except requests.RequestException as e:
            self.logger.error(f"Slack notification failed: {e}")
    
    def _error_collection_loop(self):
        """Background task to collect errors from all services"""
        while True:
            try:
                time.sleep(5)  # Collect every 5 seconds
                
                # Get all error handlers
                error_handlers = get_all_error_handlers()
                
                for service_name, handler in error_handlers.items():
                    # Get recent errors from each handler
                    for error_context in handler.error_history[-10:]:  # Last 10 errors
                        # Check if we've already processed this error
                        if not any(e.error_id == error_context.error_id for e in self.error_buffer):
                            self.ingest_error(error_context)
                
            except Exception as e:
                self.logger.error(f"Error collection loop failed: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _pattern_detection_loop(self):
        """Background task for pattern detection and analysis"""
        while True:
            try:
                interval = self.config.get("pattern_analysis_interval", 60)
                time.sleep(interval)
                
                if not self.config.get("pattern_detection_enabled", True):
                    continue
                
                self._analyze_patterns()
                self._detect_anomalies()
                self._update_pattern_confidence()
                
            except Exception as e:
                self.logger.error(f"Pattern detection loop failed: {e}")
                time.sleep(60)
    
    def _alert_processing_loop(self):
        """Background task for alert processing"""
        while True:
            try:
                interval = self.config.get("alert_processing_interval", 30)
                time.sleep(interval)
                
                self._process_pattern_alerts()
                self._escalate_alerts()
                self._cleanup_old_alerts()
                
            except Exception as e:
                self.logger.error(f"Alert processing loop failed: {e}")
                time.sleep(60)
    
    def _database_persistence_loop(self):
        """Background task for database persistence"""
        while True:
            try:
                time.sleep(300)  # Persist every 5 minutes
                self._persist_to_database()
                
            except Exception as e:
                self.logger.error(f"Database persistence loop failed: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _analyze_patterns(self):
        """Analyze error patterns for trends and insights"""
        try:
            for pattern in self.error_patterns.values():
                # Update confidence based on consistency
                if pattern.frequency > 10:
                    pattern.confidence_score = min(1.0, pattern.confidence_score + 0.05)
                
                # Generate insights
                if pattern.confidence_score > 0.8 and pattern.frequency > 15:
                    self.logger.warning(f"High-confidence error pattern detected: {pattern.error_signature} ({pattern.frequency} occurrences)")
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
    
    def _detect_anomalies(self):
        """Detect anomalies in error patterns"""
        try:
            # Detect sudden spikes in error rates
            recent_errors = [
                e for e in self.error_buffer
                if (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if len(recent_errors) > 50:  # Anomaly threshold
                self.logger.warning(f"Error rate anomaly detected: {len(recent_errors)} errors in last 5 minutes")
                
                # Could trigger special alert here
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
    
    def _update_pattern_confidence(self):
        """Update confidence scores for all patterns"""
        try:
            for pattern in self.error_patterns.values():
                # Decay confidence for old patterns
                age_hours = (datetime.now() - pattern.last_occurrence).total_seconds() / 3600
                if age_hours > 24:
                    pattern.confidence_score *= 0.95  # Slight decay
                
                # Boost confidence for recurring patterns
                if pattern.frequency > 20:
                    pattern.confidence_score = min(1.0, pattern.confidence_score * 1.05)
            
        except Exception as e:
            self.logger.error(f"Pattern confidence update failed: {e}")
    
    def _process_pattern_alerts(self):
        """Process alerts based on error patterns"""
        try:
            for pattern in self.error_patterns.values():
                # Check if pattern should trigger an alert
                if (pattern.confidence_score > self.config.get("pattern_confidence_threshold", 0.7) and
                    pattern.frequency > 10 and
                    pattern.trend == "increasing"):
                    
                    # Find matching alert rule
                    for rule in self.alert_rules.values():
                        if "pattern" in rule.name.lower():
                            # Create mock error context for pattern alert
                            pattern_error = type('obj', (object,), {
                                'pattern_frequency': pattern.frequency,
                                'pattern_confidence': pattern.confidence_score,
                                'error_message': f"Pattern alert: {pattern.error_signature}",
                                'category': ErrorCategory.UNKNOWN,
                                'severity': ErrorSeverity.WARNING
                            })()
                            
                            if self._evaluate_alert_condition(rule, pattern_error):
                                # Get recent errors for this pattern
                                pattern_errors = [
                                    e for e in self.error_buffer
                                    if self._calculate_pattern_hash(e) == pattern.pattern_hash
                                ][-5:]  # Last 5 errors
                                
                                self._trigger_alert(rule, pattern_errors)
                                break
            
        except Exception as e:
            self.logger.error(f"Pattern alert processing failed: {e}")
    
    def _escalate_alerts(self):
        """Escalate unacknowledged critical alerts"""
        try:
            for alert in self.active_alerts.values():
                if (not alert.acknowledged and 
                    alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] and
                    (datetime.now() - alert.triggered_at).total_seconds() > 1800):  # 30 minutes
                    
                    if not alert.escalated:
                        alert.escalated = True
                        alert.escalation_level += 1
                        self.logger.critical(f"Escalating unacknowledged alert: {alert.alert_id}")
                        
                        # Could send to different channels or contacts
            
        except Exception as e:
            self.logger.error(f"Alert escalation failed: {e}")
    
    def _cleanup_old_alerts(self):
        """Cleanup old resolved alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # Move old alerts to history and remove from active
            old_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.active_alerts[alert_id]
            
            self.logger.debug(f"Cleaned up {len(old_alerts)} old alerts")
            
        except Exception as e:
            self.logger.error(f"Alert cleanup failed: {e}")
    
    def _persist_to_database(self):
        """Persist data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Persist recent errors
            for error in list(self.error_buffer)[-100:]:  # Last 100 errors
                pattern_hash = self._calculate_pattern_hash(error)
                
                conn.execute('''
                    INSERT OR REPLACE INTO errors 
                    (error_id, timestamp, service_name, category, severity, 
                     error_type, error_message, pattern_hash, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error.error_id,
                    error.timestamp.isoformat(),
                    error.service_name,
                    error.category.value,
                    error.severity.value,
                    error.error_type,
                    error.error_message,
                    pattern_hash,
                    json.dumps(error.business_context)
                ))
            
            # Persist patterns
            for pattern in self.error_patterns.values():
                conn.execute('''
                    INSERT OR REPLACE INTO patterns
                    (pattern_id, pattern_hash, error_signature, first_occurrence,
                     last_occurrence, frequency, confidence_score, services_affected,
                     resolution_suggestions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_hash,
                    pattern.error_signature,
                    pattern.first_occurrence.isoformat(),
                    pattern.last_occurrence.isoformat(),
                    pattern.frequency,
                    pattern.confidence_score,
                    json.dumps(list(pattern.services_affected)),
                    json.dumps(pattern.resolution_suggestions)
                ))
            
            # Persist alerts
            for alert in self.alert_history[-50:]:  # Last 50 alerts
                conn.execute('''
                    INSERT OR REPLACE INTO alerts
                    (alert_id, rule_id, severity, title, message, triggered_at,
                     acknowledged, resolved, error_contexts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.rule_id,
                    alert.severity.value,
                    alert.title,
                    alert.message,
                    alert.triggered_at.isoformat(),
                    alert.acknowledged,
                    alert.resolved,
                    json.dumps(alert.error_contexts)
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug("Data persisted to database")
            
        except Exception as e:
            self.logger.error(f"Database persistence failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Alert resolution failed: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring"""
        try:
            current_time = datetime.now()
            
            # Recent error statistics
            recent_errors = [
                e for e in self.error_buffer
                if (current_time - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            # Error rate by service
            service_errors = defaultdict(int)
            for error in recent_errors:
                service_errors[error.service_name] += 1
            
            # Top error patterns
            top_patterns = sorted(
                self.error_patterns.values(),
                key=lambda p: p.frequency,
                reverse=True
            )[:10]
            
            # Active alerts by severity
            alert_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                alert_counts[alert.severity.value] += 1
            
            return {
                "summary": {
                    "total_errors_last_hour": len(recent_errors),
                    "active_alerts": len(self.active_alerts),
                    "error_patterns": len(self.error_patterns),
                    "services_affected": len(service_errors)
                },
                "error_rate_by_service": dict(service_errors),
                "top_error_patterns": [
                    {
                        "signature": p.error_signature,
                        "frequency": p.frequency,
                        "confidence": round(p.confidence_score, 2),
                        "trend": p.trend,
                        "services": list(p.services_affected)
                    }
                    for p in top_patterns
                ],
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "acknowledged": alert.acknowledged,
                        "error_count": len(alert.error_contexts)
                    }
                    for alert in list(self.active_alerts.values())[:20]
                ],
                "alert_counts_by_severity": dict(alert_counts),
                "recent_error_timeline": [
                    {
                        "timestamp": error.timestamp.isoformat(),
                        "service": error.service_name,
                        "severity": error.severity.value,
                        "category": error.category.value
                    }
                    for error in recent_errors[-50:]  # Last 50 errors
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {"error": "Failed to generate dashboard data"}

# Global error aggregator instance
_global_aggregator: Optional[ErrorAggregator] = None

def get_error_aggregator(config: Optional[Dict[str, Any]] = None) -> ErrorAggregator:
    """Get or create global error aggregator"""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = ErrorAggregator(config)
    return _global_aggregator

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Starting XORB Error Aggregation System...")
    
    # Create error aggregator
    aggregator = ErrorAggregator()
    
    # Simulate some errors for testing
    from xorb_error_handling_framework import ErrorContext, ErrorCategory, ErrorSeverity
    
    # Test error patterns
    test_errors = [
        ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            service_name="neural_orchestrator",
            function_name="orchestrate_agents",
            error_type="ConnectionError",
            error_message="Database connection failed",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            stack_trace="Test stack trace",
            system_context={"cpu_percent": 45},
            business_context={"user_operation": "agent_creation"}
        ),
        ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            service_name="learning_service",
            function_name="process_learning",
            error_type="ValidationError",
            error_message="Invalid learning data format",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            stack_trace="Test stack trace",
            system_context={"memory_percent": 67},
            business_context={"session_id": "test_session"}
        )
    ]
    
    # Ingest test errors
    for error in test_errors:
        aggregator.ingest_error(error)
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Get dashboard data
    dashboard_data = aggregator.get_dashboard_data()
    print(f"📊 Dashboard Data: {json.dumps(dashboard_data, indent=2, default=str)}")
    
    print("✅ Error Aggregation System test completed")