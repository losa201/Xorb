"""
Real-time event correlation engine for SIEM
Detects patterns and relationships between security events
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import time

from ..ingestion.event_normalizer import NormalizedEvent, EventCategory, EventSeverity, EventAction

logger = logging.getLogger(__name__)


class CorrelationRuleType(Enum):
    """Types of correlation rules"""
    SEQUENCE = "sequence"           # Events in specific order
    FREQUENCY = "frequency"         # Event frequency threshold
    STATISTICAL = "statistical"    # Statistical anomaly
    PATTERN = "pattern"            # Pattern matching
    TEMPORAL = "temporal"          # Time-based correlation
    GEOLOCATION = "geolocation"    # Geographic correlation


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CorrelationRule:
    """Correlation rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: CorrelationRuleType
    severity: AlertSeverity
    conditions: Dict[str, Any]
    time_window: int = 300  # seconds
    threshold: int = 1
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tags:
            self.tags = []


@dataclass
class CorrelationAlert:
    """Generated correlation alert"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    description: str
    events: List[NormalizedEvent]
    first_seen: datetime
    last_seen: datetime
    count: int
    confidence: float
    additional_context: Dict[str, Any] = field(default_factory=dict)
    mitre_techniques: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EventBuffer:
    """Time-based event buffer for correlation"""

    def __init__(self, max_size: int = 10000, retention_minutes: int = 60):
        self.max_size = max_size
        self.retention_seconds = retention_minutes * 60
        self.events = deque(maxlen=max_size)
        self.index_by_ip = defaultdict(list)
        self.index_by_user = defaultdict(list)
        self.index_by_category = defaultdict(list)
        self.lock = threading.RLock()

    def add_event(self, event: NormalizedEvent):
        """Add event to buffer with indexing"""
        with self.lock:
            self.events.append(event)

            # Index by source IP
            if event.source_ip:
                self.index_by_ip[event.source_ip].append(event)

            # Index by user
            if event.user:
                self.index_by_user[event.user].append(event)

            # Index by category
            self.index_by_category[event.category].append(event)

            # Clean old events
            self._cleanup_old_events()

    def get_events_by_ip(self, ip: str, time_window: int = 300) -> List[NormalizedEvent]:
        """Get events for specific IP within time window"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            return [e for e in self.index_by_ip.get(ip, [])
                   if e.timestamp > cutoff_time]

    def get_events_by_user(self, user: str, time_window: int = 300) -> List[NormalizedEvent]:
        """Get events for specific user within time window"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            return [e for e in self.index_by_user.get(user, [])
                   if e.timestamp > cutoff_time]

    def get_events_by_category(self, category: EventCategory, time_window: int = 300) -> List[NormalizedEvent]:
        """Get events for specific category within time window"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            return [e for e in self.index_by_category.get(category, [])
                   if e.timestamp > cutoff_time]

    def get_recent_events(self, time_window: int = 300) -> List[NormalizedEvent]:
        """Get all recent events within time window"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            return [e for e in self.events if e.timestamp > cutoff_time]

    def _cleanup_old_events(self):
        """Remove events older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.retention_seconds)

        # Clean main event buffer (deque automatically handles max size)
        while self.events and self.events[0].timestamp < cutoff_time:
            old_event = self.events.popleft()

            # Clean indexes
            if old_event.source_ip and old_event in self.index_by_ip[old_event.source_ip]:
                self.index_by_ip[old_event.source_ip].remove(old_event)
                if not self.index_by_ip[old_event.source_ip]:
                    del self.index_by_ip[old_event.source_ip]

            if old_event.user and old_event in self.index_by_user[old_event.user]:
                self.index_by_user[old_event.user].remove(old_event)
                if not self.index_by_user[old_event.user]:
                    del self.index_by_user[old_event.user]

            if old_event in self.index_by_category[old_event.category]:
                self.index_by_category[old_event.category].remove(old_event)
                if not self.index_by_category[old_event.category]:
                    del self.index_by_category[old_event.category]


class CorrelationEngine:
    """Main correlation engine"""

    def __init__(self):
        self.rules: Dict[str, CorrelationRule] = {}
        self.event_buffer = EventBuffer()
        self.alert_cache: Dict[str, CorrelationAlert] = {}
        self.statistics = defaultdict(int)
        self.lock = threading.RLock()

        # Load default rules
        self._load_default_rules()

    def add_rule(self, rule: CorrelationRule):
        """Add correlation rule"""
        with self.lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added correlation rule: {rule.name}")

    def remove_rule(self, rule_id: str):
        """Remove correlation rule"""
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed correlation rule: {rule_id}")

    def process_event(self, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process event through correlation rules"""
        alerts = []

        with self.lock:
            # Add event to buffer
            self.event_buffer.add_event(event)

            # Update statistics
            self.statistics['events_processed'] += 1

            # Process each rule
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                try:
                    rule_alerts = self._process_rule(rule, event)
                    alerts.extend(rule_alerts)
                except Exception as e:
                    logger.error(f"Error processing rule {rule.rule_id}: {e}")
                    self.statistics['rule_errors'] += 1

        return alerts

    def _process_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process single correlation rule"""
        if rule.rule_type == CorrelationRuleType.FREQUENCY:
            return self._process_frequency_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.SEQUENCE:
            return self._process_sequence_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.PATTERN:
            return self._process_pattern_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.STATISTICAL:
            return self._process_statistical_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.TEMPORAL:
            return self._process_temporal_rule(rule, event)
        else:
            return []

    def _process_frequency_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process frequency-based correlation rule"""
        alerts = []
        conditions = rule.conditions

        # Check if event matches rule conditions
        if not self._event_matches_conditions(event, conditions):
            return alerts

        # Get correlation key (what to count by)
        correlation_key = conditions.get('correlation_key', 'source_ip')
        key_value = getattr(event, correlation_key, None)

        if not key_value:
            return alerts

        # Get recent events for the same key
        if correlation_key == 'source_ip':
            recent_events = self.event_buffer.get_events_by_ip(key_value, rule.time_window)
        elif correlation_key == 'user':
            recent_events = self.event_buffer.get_events_by_user(key_value, rule.time_window)
        else:
            recent_events = self.event_buffer.get_recent_events(rule.time_window)
            recent_events = [e for e in recent_events if getattr(e, correlation_key, None) == key_value]

        # Filter events that match conditions
        matching_events = [e for e in recent_events if self._event_matches_conditions(e, conditions)]

        # Check threshold
        if len(matching_events) >= rule.threshold:
            alert_key = f"{rule.rule_id}:{key_value}"

            # Check if alert already exists (avoid duplicates)
            if alert_key not in self.alert_cache:
                alert = self._create_alert(rule, matching_events, {
                    'correlation_key': correlation_key,
                    'key_value': key_value,
                    'event_count': len(matching_events)
                })

                self.alert_cache[alert_key] = alert
                alerts.append(alert)
                self.statistics['alerts_generated'] += 1

        return alerts

    def _process_sequence_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process sequence-based correlation rule"""
        # Implementation for sequence correlation
        # This would track event sequences and detect patterns
        return []

    def _process_pattern_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process pattern-based correlation rule"""
        # Implementation for pattern matching correlation
        return []

    def _process_statistical_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process statistical anomaly correlation rule"""
        # Implementation for statistical anomaly detection
        return []

    def _process_temporal_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process temporal correlation rule"""
        # Implementation for time-based correlation
        return []

    def _event_matches_conditions(self, event: NormalizedEvent, conditions: Dict[str, Any]) -> bool:
        """Check if event matches rule conditions"""
        for field, expected_value in conditions.items():
            if field in ['correlation_key', 'threshold', 'time_window']:
                continue  # Skip rule configuration fields

            event_value = getattr(event, field, None)

            if isinstance(expected_value, list):
                if event_value not in expected_value:
                    return False
            elif isinstance(expected_value, str):
                if str(event_value).lower() != expected_value.lower():
                    return False
            else:
                if event_value != expected_value:
                    return False

        return True

    def _create_alert(self, rule: CorrelationRule, events: List[NormalizedEvent],
                     context: Dict[str, Any]) -> CorrelationAlert:
        """Create correlation alert"""
        import uuid

        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)

        alert = CorrelationAlert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            description=rule.description,
            events=events,
            first_seen=events[0].timestamp,
            last_seen=events[-1].timestamp,
            count=len(events),
            confidence=self._calculate_confidence(rule, events),
            additional_context=context,
            mitre_techniques=self._get_mitre_techniques(rule, events),
            recommendations=self._get_recommendations(rule, events)
        )

        return alert

    def _calculate_confidence(self, rule: CorrelationRule, events: List[NormalizedEvent]) -> float:
        """Calculate confidence score for alert"""
        base_confidence = 0.5

        # Higher confidence for more events
        event_factor = min(len(events) / 10.0, 0.4)

        # Higher confidence for higher severity events
        severity_factor = sum(1.0 if e.severity.value == 'critical' else
                            0.8 if e.severity.value == 'high' else
                            0.6 if e.severity.value == 'medium' else 0.4
                            for e in events) / len(events) * 0.3

        # Higher confidence for diverse event types
        unique_categories = len(set(e.category for e in events))
        diversity_factor = min(unique_categories / 5.0, 0.3)

        confidence = base_confidence + event_factor + severity_factor + diversity_factor
        return min(confidence, 1.0)

    def _get_mitre_techniques(self, rule: CorrelationRule, events: List[NormalizedEvent]) -> List[str]:
        """Get MITRE ATT&CK techniques for alert"""
        techniques = []

        # Map event categories to MITRE techniques
        category_techniques = {
            EventCategory.AUTHENTICATION: ['T1078', 'T1110', 'T1021'],
            EventCategory.NETWORK_TRAFFIC: ['T1071', 'T1090', 'T1095'],
            EventCategory.MALWARE_DETECTION: ['T1055', 'T1059', 'T1064'],
            EventCategory.INTRUSION_DETECTION: ['T1027', 'T1036', 'T1055']
        }

        for event in events:
            if event.category in category_techniques:
                techniques.extend(category_techniques[event.category])

        return list(set(techniques))  # Remove duplicates

    def _get_recommendations(self, rule: CorrelationRule, events: List[NormalizedEvent]) -> List[str]:
        """Get recommendations for alert"""
        recommendations = []

        # Generic recommendations based on rule type
        if rule.rule_type == CorrelationRuleType.FREQUENCY:
            recommendations.extend([
                "Review logs for additional suspicious activity",
                "Consider implementing rate limiting",
                "Investigate source IP/user for compromise"
            ])

        # Category-specific recommendations
        categories = set(e.category for e in events)

        if EventCategory.AUTHENTICATION in categories:
            recommendations.extend([
                "Verify user credentials and access patterns",
                "Check for password spraying or brute force attacks",
                "Consider implementing multi-factor authentication"
            ])

        if EventCategory.NETWORK_TRAFFIC in categories:
            recommendations.extend([
                "Analyze network traffic patterns",
                "Check firewall rules and network segmentation",
                "Monitor for data exfiltration indicators"
            ])

        return recommendations

    def _load_default_rules(self):
        """Load default correlation rules"""

        # Failed login attempts
        self.add_rule(CorrelationRule(
            rule_id="failed_login_frequency",
            name="Multiple Failed Login Attempts",
            description="Detects multiple failed login attempts from same IP",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=AlertSeverity.MEDIUM,
            conditions={
                'action': EventAction.FAILED_LOGIN,
                'correlation_key': 'source_ip'
            },
            time_window=300,  # 5 minutes
            threshold=5,
            tags=['authentication', 'brute_force']
        ))

        # High privilege access
        self.add_rule(CorrelationRule(
            rule_id="admin_access_frequency",
            name="Frequent Administrative Access",
            description="Detects frequent administrative access from same user",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=AlertSeverity.LOW,
            conditions={
                'category': EventCategory.AUTHORIZATION,
                'correlation_key': 'user'
            },
            time_window=600,  # 10 minutes
            threshold=10,
            tags=['privilege_escalation', 'insider_threat']
        ))

        # Network scanning
        self.add_rule(CorrelationRule(
            rule_id="port_scan_detection",
            name="Potential Port Scanning Activity",
            description="Detects connections to multiple ports from same IP",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=AlertSeverity.MEDIUM,
            conditions={
                'category': EventCategory.NETWORK_TRAFFIC,
                'correlation_key': 'source_ip'
            },
            time_window=120,  # 2 minutes
            threshold=20,
            tags=['reconnaissance', 'scanning']
        ))

        # Malware detection
        self.add_rule(CorrelationRule(
            rule_id="malware_activity",
            name="Malware Detection Alert",
            description="Malware or suspicious activity detected",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=AlertSeverity.CRITICAL,
            conditions={
                'category': EventCategory.MALWARE_DETECTION
            },
            time_window=60,
            threshold=1,
            tags=['malware', 'threat']
        ))

    def get_statistics(self) -> Dict[str, Any]:
        """Get correlation engine statistics"""
        with self.lock:
            return {
                'events_processed': self.statistics['events_processed'],
                'alerts_generated': self.statistics['alerts_generated'],
                'rule_errors': self.statistics['rule_errors'],
                'active_rules': len([r for r in self.rules.values() if r.enabled]),
                'total_rules': len(self.rules),
                'buffer_size': len(self.event_buffer.events),
                'cached_alerts': len(self.alert_cache)
            }

    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old cached alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        with self.lock:
            old_alerts = [
                alert_key for alert_key, alert in self.alert_cache.items()
                if alert.last_seen < cutoff_time
            ]

            for alert_key in old_alerts:
                del self.alert_cache[alert_key]

            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
