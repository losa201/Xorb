"""
Comprehensive Security Monitoring Service
Real-time security monitoring, alerting, and incident response automation
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque

from ..infrastructure.redis_compatibility import get_redis_client
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

# ML imports with graceful fallbacks
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..infrastructure.database import get_async_session
from ..infrastructure.observability import get_metrics_collector, add_trace_context
from .base_service import XORBService, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    NETWORK_SECURITY = "network_security"
    MALWARE = "malware"
    ANOMALY = "anomaly"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_SECURITY = "system_security"
    COMPLIANCE = "compliance"

class IncidentStatus(Enum):
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    source: str
    event_type: str
    severity: AlertSeverity
    category: AlertCategory
    description: str
    details: Dict[str, Any]
    affected_resources: List[str]
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    tags: List[str] = None
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class SecurityAlert:
    """Security alert with enrichment and context"""
    alert_id: str
    event_ids: List[str]
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    confidence: float
    risk_score: float
    created_at: datetime
    updated_at: datetime
    status: str
    assigned_to: Optional[str] = None
    indicators: List[str] = None
    mitre_tactics: List[str] = None
    mitre_techniques: List[str] = None
    remediation_steps: List[str] = None
    false_positive_likelihood: float = 0.0
    correlation_count: int = 1

@dataclass
class SecurityIncident:
    """Security incident management"""
    incident_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: IncidentStatus
    category: AlertCategory
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    alert_ids: List[str] = None
    affected_systems: List[str] = None
    timeline: List[Dict[str, Any]] = None
    containment_actions: List[str] = None
    recovery_actions: List[str] = None
    lessons_learned: Optional[str] = None

@dataclass
class ThreatDetectionRule:
    """Rule for threat detection"""
    rule_id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool
    false_positive_rate: float
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class SecurityMonitoringService(XORBService):
    """Comprehensive security monitoring and incident response service"""
    
    def __init__(self):
        super().__init__()
        
        # Storage
        self.events: deque = deque(maxlen=100000)  # Recent events
        self.alerts: Dict[str, SecurityAlert] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        self.detection_rules: Dict[str, ThreatDetectionRule] = {}
        
        # ML Models for anomaly detection
        self.anomaly_models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        
        # Configuration
        self.correlation_window = 300  # 5 minutes
        self.alert_threshold = 0.7
        self.incident_threshold = 0.8
        self.max_events_memory = 100000
        self.cleanup_interval = 3600  # 1 hour
        
        # Redis for real-time processing
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Background tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Metrics
        self.metrics = get_metrics_collector()
        
        # Event processors
        self.event_processors = {
            "authentication": self._process_auth_event,
            "network": self._process_network_event,
            "file_access": self._process_file_event,
            "system": self._process_system_event,
            "application": self._process_app_event
        }
    
    async def initialize(self) -> bool:
        """Initialize the security monitoring service"""
        try:
            logger.info("Initializing Security Monitoring Service...")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load detection rules
            await self._load_detection_rules()
            
            # Start background workers
            await self._start_background_workers()
            
            self.running = True
            logger.info("Security Monitoring Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Monitoring Service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the security monitoring service"""
        try:
            self.running = False
            
            # Cancel background tasks
            for task in self.worker_tasks:
                task.cancel()
            
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Security Monitoring Service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown Security Monitoring Service: {e}")
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connection for real-time processing"""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("Connected to Redis for security monitoring")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory-only mode: {e}")
            self.redis_client = None
    
    async def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection"""
        try:
            if HAS_SKLEARN:
                # Authentication anomaly detection
                self.anomaly_models['auth_anomaly'] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                
                # Network traffic anomaly detection
                self.anomaly_models['network_anomaly'] = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )
                
                # File access anomaly detection
                self.anomaly_models['file_anomaly'] = IsolationForest(
                    contamination=0.08,
                    random_state=42
                )
                
                # Feature scalers
                self.feature_scalers['standard'] = StandardScaler()
                
                logger.info("ML models initialized for security monitoring")
            else:
                logger.warning("scikit-learn not available, using rule-based detection only")
                
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _load_detection_rules(self):
        """Load threat detection rules"""
        try:
            # Default detection rules
            default_rules = [
                ThreatDetectionRule(
                    rule_id="multiple_failed_logins",
                    name="Multiple Failed Login Attempts",
                    description="Detect multiple failed login attempts from same source",
                    category=AlertCategory.AUTHENTICATION,
                    severity=AlertSeverity.MEDIUM,
                    conditions={
                        "event_type": "login_failed",
                        "count_threshold": 5,
                        "time_window": 300,
                        "group_by": "source_ip"
                    },
                    actions=["create_alert", "notify_security", "temp_block_ip"],
                    enabled=True,
                    false_positive_rate=0.1
                ),
                ThreatDetectionRule(
                    rule_id="privilege_escalation",
                    name="Privilege Escalation Attempt",
                    description="Detect potential privilege escalation activities",
                    category=AlertCategory.AUTHORIZATION,
                    severity=AlertSeverity.HIGH,
                    conditions={
                        "event_type": "permission_change",
                        "permission_level": "admin",
                        "unusual_time": True
                    },
                    actions=["create_alert", "notify_security", "require_approval"],
                    enabled=True,
                    false_positive_rate=0.05
                ),
                ThreatDetectionRule(
                    rule_id="suspicious_data_access",
                    name="Suspicious Data Access Pattern",
                    description="Detect unusual data access patterns",
                    category=AlertCategory.DATA_ACCESS,
                    severity=AlertSeverity.HIGH,
                    conditions={
                        "event_type": "file_access",
                        "sensitive_data": True,
                        "volume_threshold": 100,
                        "time_window": 600
                    },
                    actions=["create_alert", "notify_security", "audit_access"],
                    enabled=True,
                    false_positive_rate=0.15
                ),
                ThreatDetectionRule(
                    rule_id="malware_detected",
                    name="Malware Detection",
                    description="Malware detected on system",
                    category=AlertCategory.MALWARE,
                    severity=AlertSeverity.CRITICAL,
                    conditions={
                        "event_type": "malware_scan",
                        "result": "detected"
                    },
                    actions=["create_incident", "isolate_system", "notify_security"],
                    enabled=True,
                    false_positive_rate=0.02
                ),
                ThreatDetectionRule(
                    rule_id="unusual_network_traffic",
                    name="Unusual Network Traffic",
                    description="Detect anomalous network traffic patterns",
                    category=AlertCategory.NETWORK_SECURITY,
                    severity=AlertSeverity.MEDIUM,
                    conditions={
                        "event_type": "network_flow",
                        "anomaly_score": 0.8,
                        "external_destination": True
                    },
                    actions=["create_alert", "analyze_traffic"],
                    enabled=True,
                    false_positive_rate=0.2
                )
            ]
            
            for rule in default_rules:
                self.detection_rules[rule.rule_id] = rule
            
            logger.info(f"Loaded {len(self.detection_rules)} detection rules")
            
        except Exception as e:
            logger.error(f"Failed to load detection rules: {e}")
    
    async def _start_background_workers(self):
        """Start background processing workers"""
        try:
            # Event correlation worker
            correlation_task = asyncio.create_task(self._correlation_worker())
            self.worker_tasks.append(correlation_task)
            
            # Alert processing worker
            alert_task = asyncio.create_task(self._alert_processing_worker())
            self.worker_tasks.append(alert_task)
            
            # Incident management worker
            incident_task = asyncio.create_task(self._incident_management_worker())
            self.worker_tasks.append(incident_task)
            
            # Cleanup worker
            cleanup_task = asyncio.create_task(self._cleanup_worker())
            self.worker_tasks.append(cleanup_task)
            
            # ML training worker
            if HAS_SKLEARN:
                ml_task = asyncio.create_task(self._ml_training_worker())
                self.worker_tasks.append(ml_task)
            
            logger.info(f"Started {len(self.worker_tasks)} background workers")
            
        except Exception as e:
            logger.error(f"Failed to start background workers: {e}")
    
    async def ingest_security_event(self, event_data: Dict[str, Any]) -> str:
        """Ingest a security event for processing"""
        try:
            # Create security event
            event = SecurityEvent(
                event_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                source=event_data.get("source", "unknown"),
                event_type=event_data.get("event_type", "generic"),
                severity=AlertSeverity(event_data.get("severity", "info")),
                category=AlertCategory(event_data.get("category", "system_security")),
                description=event_data.get("description", ""),
                details=event_data.get("details", {}),
                affected_resources=event_data.get("affected_resources", []),
                source_ip=event_data.get("source_ip"),
                user_id=event_data.get("user_id"),
                tenant_id=event_data.get("tenant_id"),
                tags=event_data.get("tags", []),
                raw_data=event_data
            )
            
            # Store event
            self.events.append(event)
            
            # Real-time processing
            await self._process_event_real_time(event)
            
            # Store in Redis for real-time queries
            if self.redis_client:
                await self._store_event_in_redis(event)
            
            # Record metrics
            self.metrics.record_api_request(f"security_event_{event.category.value}", 1)
            
            logger.debug(f"Ingested security event: {event.event_id}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to ingest security event: {e}")
            raise
    
    async def _process_event_real_time(self, event: SecurityEvent):
        """Process event in real-time for immediate threats"""
        try:
            # Apply detection rules
            await self._apply_detection_rules(event)
            
            # Check for immediate anomalies
            if event.category in self.anomaly_models and HAS_SKLEARN:
                anomaly_score = await self._detect_anomaly(event)
                if anomaly_score > self.alert_threshold:
                    await self._create_anomaly_alert(event, anomaly_score)
            
            # Process by event type
            processor = self.event_processors.get(event.event_type)
            if processor:
                await processor(event)
            
        except Exception as e:
            logger.error(f"Failed to process event real-time: {e}")
    
    async def _apply_detection_rules(self, event: SecurityEvent):
        """Apply detection rules to the event"""
        try:
            for rule in self.detection_rules.values():
                if not rule.enabled:
                    continue
                
                if await self._rule_matches_event(rule, event):
                    await self._trigger_rule_actions(rule, event)
                    
                    rule.last_triggered = datetime.utcnow()
                    rule.trigger_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to apply detection rules: {e}")
    
    async def _rule_matches_event(self, rule: ThreatDetectionRule, event: SecurityEvent) -> bool:
        """Check if a rule matches the given event"""
        try:
            conditions = rule.conditions
            
            # Check event type
            if "event_type" in conditions and event.event_type != conditions["event_type"]:
                return False
            
            # Check category
            if "category" in conditions and event.category.value != conditions["category"]:
                return False
            
            # Check severity
            if "min_severity" in conditions:
                min_severity = AlertSeverity(conditions["min_severity"])
                if self._severity_weight(event.severity) < self._severity_weight(min_severity):
                    return False
            
            # Check count-based conditions
            if "count_threshold" in conditions:
                count = await self._get_event_count(rule, event)
                if count < conditions["count_threshold"]:
                    return False
            
            # Check specific field conditions
            for field, expected_value in conditions.items():
                if field in ["event_type", "category", "min_severity", "count_threshold", "time_window", "group_by"]:
                    continue
                
                actual_value = event.details.get(field) or getattr(event, field, None)
                if actual_value != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check rule match: {e}")
            return False
    
    def _severity_weight(self, severity: AlertSeverity) -> int:
        """Get numeric weight for severity"""
        weights = {
            AlertSeverity.INFO: 1,
            AlertSeverity.LOW: 2,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.HIGH: 4,
            AlertSeverity.CRITICAL: 5
        }
        return weights.get(severity, 1)
    
    async def _get_event_count(self, rule: ThreatDetectionRule, event: SecurityEvent) -> int:
        """Get count of similar events for rule evaluation"""
        try:
            time_window = rule.conditions.get("time_window", 300)
            group_by = rule.conditions.get("group_by", "source_ip")
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            
            # Get grouping value
            if group_by == "source_ip":
                group_value = event.source_ip
            elif group_by == "user_id":
                group_value = event.user_id
            else:
                group_value = getattr(event, group_by, None)
            
            if not group_value:
                return 0
            
            # Count matching events
            count = 0
            for stored_event in self.events:
                if stored_event.timestamp < cutoff_time:
                    continue
                
                if stored_event.event_type != event.event_type:
                    continue
                
                stored_group_value = getattr(stored_event, group_by, None)
                if stored_group_value == group_value:
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to get event count: {e}")
            return 0
    
    async def _trigger_rule_actions(self, rule: ThreatDetectionRule, event: SecurityEvent):
        """Trigger actions for a matched rule"""
        try:
            for action in rule.actions:
                if action == "create_alert":
                    await self._create_rule_alert(rule, event)
                elif action == "create_incident":
                    await self._create_rule_incident(rule, event)
                elif action == "notify_security":
                    await self._notify_security_team(rule, event)
                elif action == "temp_block_ip":
                    await self._temp_block_ip(event)
                elif action == "isolate_system":
                    await self._isolate_system(event)
                elif action == "require_approval":
                    await self._require_approval(event)
                else:
                    logger.warning(f"Unknown rule action: {action}")
                    
        except Exception as e:
            logger.error(f"Failed to trigger rule actions: {e}")
    
    async def _create_rule_alert(self, rule: ThreatDetectionRule, event: SecurityEvent):
        """Create alert from rule trigger"""
        try:
            alert = SecurityAlert(
                alert_id=str(uuid4()),
                event_ids=[event.event_id],
                title=f"Security Rule Triggered: {rule.name}",
                description=f"{rule.description}\n\nTriggered by: {event.description}",
                severity=rule.severity,
                category=rule.category,
                confidence=1.0 - rule.false_positive_rate,
                risk_score=self._calculate_risk_score(rule.severity, 1.0 - rule.false_positive_rate),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="new",
                indicators=[event.source_ip] if event.source_ip else [],
                remediation_steps=await self._generate_remediation_steps(rule, event),
                false_positive_likelihood=rule.false_positive_rate
            )
            
            self.alerts[alert.alert_id] = alert
            
            # Store in database
            await self._store_alert_in_database(alert)
            
            logger.info(f"Created alert from rule {rule.name}: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to create rule alert: {e}")
    
    async def _create_rule_incident(self, rule: ThreatDetectionRule, event: SecurityEvent):
        """Create incident from critical rule trigger"""
        try:
            incident = SecurityIncident(
                incident_id=str(uuid4()),
                title=f"Security Incident: {rule.name}",
                description=f"Critical security rule triggered: {rule.description}",
                severity=rule.severity,
                status=IncidentStatus.NEW,
                category=rule.category,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                affected_systems=[event.source] if event.source else [],
                timeline=[{
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "Incident created",
                    "description": f"Rule {rule.name} triggered",
                    "event_id": event.event_id
                }],
                containment_actions=await self._generate_containment_actions(rule, event)
            )
            
            self.incidents[incident.incident_id] = incident
            
            # Store in database
            await self._store_incident_in_database(incident)
            
            logger.warning(f"Created security incident: {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to create rule incident: {e}")
    
    async def _generate_remediation_steps(self, rule: ThreatDetectionRule, event: SecurityEvent) -> List[str]:
        """Generate remediation steps for an alert"""
        steps = []
        
        if rule.category == AlertCategory.AUTHENTICATION:
            steps.extend([
                "Verify user identity and recent login activity",
                "Check for account compromise indicators",
                "Consider implementing additional authentication factors",
                "Review authentication logs for patterns"
            ])
        elif rule.category == AlertCategory.MALWARE:
            steps.extend([
                "Isolate affected system immediately",
                "Run comprehensive malware scan",
                "Check for lateral movement indicators",
                "Review system integrity"
            ])
        elif rule.category == AlertCategory.DATA_ACCESS:
            steps.extend([
                "Review data access permissions",
                "Audit recent file access activity",
                "Check for data exfiltration indicators",
                "Verify business justification for access"
            ])
        elif rule.category == AlertCategory.NETWORK_SECURITY:
            steps.extend([
                "Analyze network traffic patterns",
                "Check firewall and IDS logs",
                "Verify network segmentation",
                "Review external connections"
            ])
        
        # Generic steps
        steps.extend([
            "Document all findings and actions taken",
            "Correlate with other security events",
            "Update threat intelligence if applicable",
            "Review and update detection rules as needed"
        ])
        
        return steps
    
    async def _generate_containment_actions(self, rule: ThreatDetectionRule, event: SecurityEvent) -> List[str]:
        """Generate containment actions for an incident"""
        actions = []
        
        if rule.category == AlertCategory.MALWARE:
            actions.extend([
                "Isolate infected systems from network",
                "Disable compromised user accounts",
                "Block malicious network communications",
                "Preserve evidence for forensic analysis"
            ])
        elif rule.category == AlertCategory.DATA_ACCESS:
            actions.extend([
                "Revoke unnecessary data access permissions",
                "Monitor data exfiltration channels",
                "Enable enhanced data loss prevention",
                "Notify data owners of potential breach"
            ])
        elif rule.category == AlertCategory.NETWORK_SECURITY:
            actions.extend([
                "Block suspicious network traffic",
                "Isolate affected network segments",
                "Monitor for lateral movement",
                "Update firewall rules"
            ])
        
        return actions
    
    # Background workers
    async def _correlation_worker(self):
        """Background worker for event correlation"""
        while self.running:
            try:
                await self._correlate_recent_events()
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Correlation worker error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processing_worker(self):
        """Background worker for alert processing"""
        while self.running:
            try:
                await self._process_pending_alerts()
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Alert processing worker error: {e}")
                await asyncio.sleep(60)
    
    async def _incident_management_worker(self):
        """Background worker for incident management"""
        while self.running:
            try:
                await self._update_incident_statuses()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Incident management worker error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_worker(self):
        """Background worker for data cleanup"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(60)
    
    async def _ml_training_worker(self):
        """Background worker for ML model training"""
        while self.running:
            try:
                await self._retrain_anomaly_models()
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"ML training worker error: {e}")
                await asyncio.sleep(60)
    
    async def _correlate_recent_events(self):
        """Correlate recent events to identify attack patterns"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.correlation_window)
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            
            if len(recent_events) < 3:
                return
            
            # Group events by various criteria
            correlations = await self._find_event_correlations(recent_events)
            
            # Create alerts for significant correlations
            for correlation in correlations:
                if correlation["confidence"] > self.alert_threshold:
                    await self._create_correlation_alert(correlation)
                    
        except Exception as e:
            logger.error(f"Event correlation failed: {e}")
    
    async def _find_event_correlations(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Find correlations between events"""
        correlations = []
        
        try:
            # IP-based correlation
            ip_groups = defaultdict(list)
            for event in events:
                if event.source_ip:
                    ip_groups[event.source_ip].append(event)
            
            for ip, ip_events in ip_groups.items():
                if len(ip_events) >= 3:
                    correlation = {
                        "type": "ip_based",
                        "key": ip,
                        "events": [e.event_id for e in ip_events],
                        "confidence": min(1.0, len(ip_events) / 10.0),
                        "description": f"Multiple security events from IP {ip}",
                        "severity": max(ip_events, key=lambda e: self._severity_weight(e.severity)).severity
                    }
                    correlations.append(correlation)
            
            # User-based correlation
            user_groups = defaultdict(list)
            for event in events:
                if event.user_id:
                    user_groups[event.user_id].append(event)
            
            for user_id, user_events in user_groups.items():
                if len(user_events) >= 3:
                    correlation = {
                        "type": "user_based",
                        "key": user_id,
                        "events": [e.event_id for e in user_events],
                        "confidence": min(1.0, len(user_events) / 8.0),
                        "description": f"Multiple security events for user {user_id}",
                        "severity": max(user_events, key=lambda e: self._severity_weight(e.severity)).severity
                    }
                    correlations.append(correlation)
            
            # Time-based correlation (rapid succession)
            time_sorted = sorted(events, key=lambda e: e.timestamp)
            for i in range(len(time_sorted) - 2):
                window_events = []
                for j in range(i, len(time_sorted)):
                    if (time_sorted[j].timestamp - time_sorted[i].timestamp).total_seconds() <= 60:
                        window_events.append(time_sorted[j])
                    else:
                        break
                
                if len(window_events) >= 3:
                    correlation = {
                        "type": "temporal",
                        "key": f"rapid_{i}",
                        "events": [e.event_id for e in window_events],
                        "confidence": min(1.0, len(window_events) / 5.0),
                        "description": f"Rapid succession of {len(window_events)} security events",
                        "severity": max(window_events, key=lambda e: self._severity_weight(e.severity)).severity
                    }
                    correlations.append(correlation)
                    break
                    
        except Exception as e:
            logger.error(f"Failed to find event correlations: {e}")
        
        return correlations
    
    async def _create_correlation_alert(self, correlation: Dict[str, Any]):
        """Create alert from event correlation"""
        try:
            alert = SecurityAlert(
                alert_id=str(uuid4()),
                event_ids=correlation["events"],
                title=f"Correlated Security Events: {correlation['type']}",
                description=correlation["description"],
                severity=correlation["severity"],
                category=AlertCategory.ANOMALY,
                confidence=correlation["confidence"],
                risk_score=self._calculate_risk_score(correlation["severity"], correlation["confidence"]),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="new",
                correlation_count=len(correlation["events"]),
                remediation_steps=await self._generate_correlation_remediation(correlation)
            )
            
            self.alerts[alert.alert_id] = alert
            await self._store_alert_in_database(alert)
            
            logger.info(f"Created correlation alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to create correlation alert: {e}")
    
    async def _generate_correlation_remediation(self, correlation: Dict[str, Any]) -> List[str]:
        """Generate remediation steps for correlated events"""
        steps = [
            f"Investigate the {correlation['type']} correlation pattern",
            "Review all related events for attack indicators",
            "Check for signs of coordinated attack",
            "Verify the legitimacy of the correlated activities"
        ]
        
        if correlation["type"] == "ip_based":
            steps.extend([
                "Consider blocking the source IP temporarily",
                "Check threat intelligence for IP reputation",
                "Review firewall logs for this IP"
            ])
        elif correlation["type"] == "user_based":
            steps.extend([
                "Verify user identity and current activities",
                "Check for account compromise indicators",
                "Review user's recent access patterns"
            ])
        elif correlation["type"] == "temporal":
            steps.extend([
                "Analyze the rapid event sequence",
                "Look for automated attack patterns",
                "Check for script or bot activity"
            ])
        
        return steps
    
    def _calculate_risk_score(self, severity: AlertSeverity, confidence: float) -> float:
        """Calculate risk score based on severity and confidence"""
        severity_score = self._severity_weight(severity) / 5.0
        return min(1.0, severity_score * confidence)
    
    # Utility methods for action implementation
    async def _notify_security_team(self, rule: ThreatDetectionRule, event: SecurityEvent):
        """Notify security team about the event"""
        try:
            # In production, this would send notifications via email, Slack, etc.
            logger.warning(f"SECURITY ALERT: {rule.name} - {event.description}")
            
        except Exception as e:
            logger.error(f"Failed to notify security team: {e}")
    
    async def _temp_block_ip(self, event: SecurityEvent):
        """Temporarily block an IP address"""
        try:
            if event.source_ip:
                # In production, this would update firewall rules
                logger.warning(f"Temporarily blocking IP: {event.source_ip}")
                
        except Exception as e:
            logger.error(f"Failed to block IP: {e}")
    
    async def _isolate_system(self, event: SecurityEvent):
        """Isolate a compromised system"""
        try:
            # In production, this would trigger network isolation
            logger.critical(f"Isolating system: {event.source}")
            
        except Exception as e:
            logger.error(f"Failed to isolate system: {e}")
    
    async def _require_approval(self, event: SecurityEvent):
        """Require approval for high-risk activities"""
        try:
            # In production, this would trigger approval workflow
            logger.warning(f"Requiring approval for: {event.description}")
            
        except Exception as e:
            logger.error(f"Failed to require approval: {e}")
    
    # Database operations
    async def _store_alert_in_database(self, alert: SecurityAlert):
        """Store alert in database"""
        try:
            # In production, this would store in database
            logger.debug(f"Storing alert in database: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")
    
    async def _store_incident_in_database(self, incident: SecurityIncident):
        """Store incident in database"""
        try:
            # In production, this would store in database
            logger.debug(f"Storing incident in database: {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to store incident in database: {e}")
    
    async def _store_event_in_redis(self, event: SecurityEvent):
        """Store event in Redis for real-time queries"""
        try:
            if self.redis_client:
                event_data = asdict(event)
                # Convert datetime to string for JSON serialization
                event_data['timestamp'] = event.timestamp.isoformat()
                
                await self.redis_client.lpush(
                    f"security_events:{event.tenant_id or 'global'}", 
                    json.dumps(event_data, default=str)
                )
                
                # Expire after 24 hours
                await self.redis_client.expire(
                    f"security_events:{event.tenant_id or 'global'}", 
                    86400
                )
                
        except Exception as e:
            logger.error(f"Failed to store event in Redis: {e}")
    
    async def get_security_dashboard_data(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get security dashboard data"""
        try:
            # Filter events by tenant if specified
            filtered_events = [e for e in self.events if not tenant_id or e.tenant_id == tenant_id]
            filtered_alerts = [a for a in self.alerts.values() if not tenant_id or any(
                e.tenant_id == tenant_id for e in self.events if e.event_id in a.event_ids
            )]
            filtered_incidents = [i for i in self.incidents.values() if not tenant_id]
            
            # Calculate time periods
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Event statistics
            recent_events = [e for e in filtered_events if e.timestamp > last_24h]
            weekly_events = [e for e in filtered_events if e.timestamp > last_7d]
            
            # Alert statistics
            recent_alerts = [a for a in filtered_alerts if a.created_at > last_24h]
            weekly_alerts = [a for a in filtered_alerts if a.created_at > last_7d]
            
            # Incident statistics
            recent_incidents = [i for i in filtered_incidents if i.created_at > last_24h]
            open_incidents = [i for i in filtered_incidents if i.status != IncidentStatus.CLOSED]
            
            # Category breakdown
            category_counts = defaultdict(int)
            for event in recent_events:
                category_counts[event.category.value] += 1
            
            # Severity breakdown
            severity_counts = defaultdict(int)
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1
            
            dashboard_data = {
                "summary": {
                    "events_24h": len(recent_events),
                    "events_7d": len(weekly_events),
                    "alerts_24h": len(recent_alerts),
                    "alerts_7d": len(weekly_alerts),
                    "incidents_24h": len(recent_incidents),
                    "open_incidents": len(open_incidents),
                    "total_events": len(filtered_events),
                    "total_alerts": len(filtered_alerts),
                    "total_incidents": len(filtered_incidents)
                },
                "category_breakdown": dict(category_counts),
                "severity_breakdown": dict(severity_counts),
                "recent_alerts": [
                    {
                        "alert_id": a.alert_id,
                        "title": a.title,
                        "severity": a.severity.value,
                        "created_at": a.created_at.isoformat(),
                        "status": a.status
                    }
                    for a in sorted(recent_alerts, key=lambda x: x.created_at, reverse=True)[:10]
                ],
                "active_incidents": [
                    {
                        "incident_id": i.incident_id,
                        "title": i.title,
                        "severity": i.severity.value,
                        "status": i.status.value,
                        "created_at": i.created_at.isoformat()
                    }
                    for i in sorted(open_incidents, key=lambda x: x.created_at, reverse=True)[:5]
                ]
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {
                "events_in_memory": len(self.events),
                "active_alerts": len([a for a in self.alerts.values() if a.status != "closed"]),
                "open_incidents": len([i for i in self.incidents.values() if i.status != IncidentStatus.CLOSED]),
                "detection_rules": len(self.detection_rules),
                "background_tasks": len([t for t in self.worker_tasks if not t.done()]),
                "redis_connected": self.redis_client is not None
            }
            
            status = ServiceStatus.HEALTHY
            message = "Security Monitoring Service is operational"
            
            # Check for issues
            if checks["background_tasks"] < len(self.worker_tasks) * 0.8:
                status = ServiceStatus.DEGRADED
                message = "Some background tasks are not running"
            elif len(self.events) >= self.max_events_memory * 0.9:
                status = ServiceStatus.DEGRADED
                message = "High memory usage for events"
            
            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )

# Global service instance
_security_monitoring_service: Optional[SecurityMonitoringService] = None

async def get_security_monitoring_service() -> SecurityMonitoringService:
    """Get global security monitoring service instance"""
    global _security_monitoring_service
    
    if _security_monitoring_service is None:
        _security_monitoring_service = SecurityMonitoringService()
        await _security_monitoring_service.initialize()
    
    return _security_monitoring_service