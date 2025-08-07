"""
Security event correlation engine
Analyzes normalized events for threat patterns, relationships, and attack sequences
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..ingestion.event_normalizer import NormalizedEvent, EventCategory, EventSeverity


class CorrelationRuleType(Enum):
    """Types of correlation rules"""
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"
    STATISTICAL = "statistical"
    GEOGRAPHICAL = "geographical"
    BEHAVIORAL = "behavioral"
    THREAT_HUNTING = "threat_hunting"


class CorrelationSeverity(Enum):
    """Correlation alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CorrelationRule:
    """Correlation rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: CorrelationRuleType
    severity: CorrelationSeverity
    enabled: bool = True
    
    # Rule conditions
    event_filters: Dict[str, Any] = field(default_factory=dict)
    time_window: timedelta = field(default=timedelta(minutes=5))
    threshold_count: int = 1
    group_by_fields: List[str] = field(default_factory=list)
    
    # Advanced conditions
    sequence_pattern: List[Dict[str, Any]] = field(default_factory=list)
    statistical_baseline: Dict[str, float] = field(default_factory=dict)
    geographic_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    author: str = "system"
    tags: List[str] = field(default_factory=list)


@dataclass
class CorrelationAlert:
    """Correlation alert generated from event analysis"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    
    # Alert details
    title: str = ""
    description: str = ""
    severity: CorrelationSeverity = CorrelationSeverity.INFO
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Event information
    triggering_events: List[NormalizedEvent] = field(default_factory=list)
    event_count: int = 0
    time_span: timedelta = field(default=timedelta())
    
    # Context information
    affected_assets: Set[str] = field(default_factory=set)
    source_ips: Set[str] = field(default_factory=set)
    dest_ips: Set[str] = field(default_factory=set)
    users: Set[str] = field(default_factory=set)
    
    # Threat intelligence
    iocs: List[str] = field(default_factory=list)  # Indicators of Compromise
    attack_techniques: List[str] = field(default_factory=list)  # MITRE ATT&CK
    kill_chain_phase: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    assigned_to: Optional[str] = None
    status: str = "open"  # open, investigating, resolved, false_positive
    
    def get_asset_summary(self) -> Dict[str, int]:
        """Get summary of affected assets"""
        return {
            "hosts": len(self.affected_assets),
            "source_ips": len(self.source_ips),
            "dest_ips": len(self.dest_ips),
            "users": len(self.users)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "event_count": self.event_count,
            "time_span_seconds": self.time_span.total_seconds(),
            "affected_assets": list(self.affected_assets),
            "source_ips": list(self.source_ips),
            "dest_ips": list(self.dest_ips),
            "users": list(self.users),
            "iocs": self.iocs,
            "attack_techniques": self.attack_techniques,
            "kill_chain_phase": self.kill_chain_phase,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "assigned_to": self.assigned_to,
            "status": self.status
        }


class EventBuffer:
    """Buffer for storing events with time-based expiration"""
    
    def __init__(self, max_size: int = 10000, max_age: timedelta = timedelta(hours=24)):
        self.max_size = max_size
        self.max_age = max_age
        self.events: deque = deque(maxlen=max_size)
        self.event_index: Dict[str, List[NormalizedEvent]] = defaultdict(list)
    
    def add_event(self, event: NormalizedEvent):
        """Add event to buffer with indexing"""
        self.events.append(event)
        
        # Index by various fields for fast lookup
        self.event_index[f"source_ip:{event.source_ip}"].append(event)
        self.event_index[f"dest_ip:{event.dest_ip}"].append(event)
        self.event_index[f"user:{event.source_user}"].append(event)
        self.event_index[f"host:{event.source_host}"].append(event)
        self.event_index[f"category:{event.category.value}"].append(event)
        
        # Clean old events
        self._cleanup_old_events()
    
    def get_events_by_criteria(self, 
                              time_window: timedelta,
                              filters: Optional[Dict[str, Any]] = None) -> List[NormalizedEvent]:
        """Get events matching criteria within time window"""
        cutoff_time = datetime.utcnow() - time_window
        matching_events = []
        
        for event in self.events:
            if event.timestamp < cutoff_time:
                continue
                
            if self._matches_filters(event, filters):
                matching_events.append(event)
        
        return matching_events
    
    def get_events_by_index(self, index_key: str, time_window: timedelta) -> List[NormalizedEvent]:
        """Get events by index key within time window"""
        cutoff_time = datetime.utcnow() - time_window
        events = self.event_index.get(index_key, [])
        
        return [event for event in events if event.timestamp >= cutoff_time]
    
    def _matches_filters(self, event: NormalizedEvent, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if event matches filter criteria"""
        if not filters:
            return True
        
        for field, value in filters.items():
            event_value = getattr(event, field, None)
            if event_value != value:
                return False
        
        return True
    
    def _cleanup_old_events(self):
        """Remove events older than max_age from index"""
        cutoff_time = datetime.utcnow() - self.max_age
        
        for key, events in self.event_index.items():
            self.event_index[key] = [
                event for event in events 
                if event.timestamp >= cutoff_time
            ]


class CorrelationEngine:
    """Main correlation engine for security event analysis"""
    
    def __init__(self):
        self.rules: Dict[str, CorrelationRule] = {}
        self.event_buffer = EventBuffer()
        self.active_correlations: Dict[str, Dict] = {}
        self.generated_alerts: List[CorrelationAlert] = []
        
        # Initialize built-in rules
        self._initialize_builtin_rules()
    
    def add_rule(self, rule: CorrelationRule):
        """Add correlation rule"""
        self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """Remove correlation rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def process_event(self, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Process incoming event and generate alerts"""
        self.event_buffer.add_event(event)
        alerts = []
        
        # Process against all enabled rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                rule_alerts = self._evaluate_rule(rule, event)
                alerts.extend(rule_alerts)
            except Exception as e:
                # Log error but continue processing other rules
                print(f"Error evaluating rule {rule.rule_id}: {e}")
        
        # Store generated alerts
        self.generated_alerts.extend(alerts)
        
        return alerts
    
    def _evaluate_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate specific rule against event"""
        if rule.rule_type == CorrelationRuleType.FREQUENCY:
            return self._evaluate_frequency_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.SEQUENCE:
            return self._evaluate_sequence_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.STATISTICAL:
            return self._evaluate_statistical_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.GEOGRAPHICAL:
            return self._evaluate_geographical_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.BEHAVIORAL:
            return self._evaluate_behavioral_rule(rule, event)
        elif rule.rule_type == CorrelationRuleType.THREAT_HUNTING:
            return self._evaluate_threat_hunting_rule(rule, event)
        
        return []
    
    def _evaluate_frequency_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate frequency-based correlation rule"""
        # Get events matching filter criteria within time window
        matching_events = self.event_buffer.get_events_by_criteria(
            rule.time_window, rule.event_filters
        )
        
        if len(matching_events) < rule.threshold_count:
            return []
        
        # Group by specified fields
        if rule.group_by_fields:
            groups = self._group_events(matching_events, rule.group_by_fields)
            alerts = []
            
            for group_key, group_events in groups.items():
                if len(group_events) >= rule.threshold_count:
                    alert = self._create_frequency_alert(rule, group_events, group_key)
                    alerts.append(alert)
            
            return alerts
        else:
            # Simple threshold check
            alert = self._create_frequency_alert(rule, matching_events)
            return [alert]
    
    def _evaluate_sequence_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate sequence-based correlation rule"""
        if not rule.sequence_pattern:
            return []
        
        # Get events within time window
        events = self.event_buffer.get_events_by_criteria(rule.time_window)
        
        # Look for sequence pattern
        sequence_matches = self._find_sequence_patterns(events, rule.sequence_pattern)
        
        alerts = []
        for match in sequence_matches:
            alert = self._create_sequence_alert(rule, match)
            alerts.append(alert)
        
        return alerts
    
    def _evaluate_statistical_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate statistical anomaly rule"""
        # Implement statistical analysis (e.g., standard deviation from baseline)
        baseline = rule.statistical_baseline
        
        # Get recent events for statistical analysis
        recent_events = self.event_buffer.get_events_by_criteria(
            timedelta(hours=1), rule.event_filters
        )
        
        # Calculate current metrics
        current_rate = len(recent_events) / 60  # events per minute
        baseline_rate = baseline.get('events_per_minute', 0)
        threshold_multiplier = baseline.get('threshold_multiplier', 3.0)
        
        if current_rate > baseline_rate * threshold_multiplier:
            alert = self._create_statistical_alert(rule, recent_events, current_rate, baseline_rate)
            return [alert]
        
        return []
    
    def _evaluate_geographical_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate geographical correlation rule"""
        geo_constraints = rule.geographic_constraints
        
        # Check for impossible travel (same user from different locations)
        if event.source_user and event.geo_location:
            user_events = self.event_buffer.get_events_by_index(
                f"user:{event.source_user}", rule.time_window
            )
            
            # Look for events from different locations within short time
            for prev_event in user_events:
                if prev_event.geo_location and prev_event != event:
                    time_diff = abs((event.timestamp - prev_event.timestamp).total_seconds())
                    if time_diff < geo_constraints.get('min_travel_time', 3600):  # 1 hour default
                        alert = self._create_geographical_alert(rule, [prev_event, event])
                        return [alert]
        
        return []
    
    def _evaluate_behavioral_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate behavioral anomaly rule"""
        # Analyze user behavior patterns
        if event.source_user:
            user_events = self.event_buffer.get_events_by_index(
                f"user:{event.source_user}", timedelta(days=7)
            )
            
            # Analyze for unusual patterns
            if self._is_unusual_behavior(event, user_events):
                alert = self._create_behavioral_alert(rule, [event], user_events)
                return [alert]
        
        return []
    
    def _evaluate_threat_hunting_rule(self, rule: CorrelationRule, event: NormalizedEvent) -> List[CorrelationAlert]:
        """Evaluate threat hunting rule"""
        # Look for specific threat indicators
        threat_indicators = rule.event_filters.get('threat_indicators', [])
        
        for indicator in threat_indicators:
            if self._matches_threat_indicator(event, indicator):
                alert = self._create_threat_hunting_alert(rule, [event], indicator)
                return [alert]
        
        return []
    
    def _group_events(self, events: List[NormalizedEvent], group_fields: List[str]) -> Dict[str, List[NormalizedEvent]]:
        """Group events by specified fields"""
        groups = defaultdict(list)
        
        for event in events:
            group_key = "|".join([str(getattr(event, field, "")) for field in group_fields])
            groups[group_key].append(event)
        
        return dict(groups)
    
    def _find_sequence_patterns(self, events: List[NormalizedEvent], pattern: List[Dict[str, Any]]) -> List[List[NormalizedEvent]]:
        """Find sequence patterns in events"""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        matches = []
        
        # Simple sequence matching implementation
        for i in range(len(sorted_events) - len(pattern) + 1):
            potential_match = []
            pattern_index = 0
            
            for j in range(i, len(sorted_events)):
                event = sorted_events[j]
                
                if pattern_index < len(pattern):
                    pattern_step = pattern[pattern_index]
                    
                    if self._matches_filters(event, pattern_step):
                        potential_match.append(event)
                        pattern_index += 1
                        
                        if pattern_index == len(pattern):
                            matches.append(potential_match)
                            break
        
        return matches
    
    def _is_unusual_behavior(self, event: NormalizedEvent, historical_events: List[NormalizedEvent]) -> bool:
        """Determine if event represents unusual behavior"""
        # Simple behavioral analysis
        if not historical_events:
            return True  # First time seeing this user
        
        # Check for unusual time patterns
        usual_hours = set()
        for hist_event in historical_events[-100:]:  # Recent history
            usual_hours.add(hist_event.timestamp.hour)
        
        current_hour = event.timestamp.hour
        if current_hour not in usual_hours:
            return True
        
        # Check for unusual access patterns
        usual_destinations = set()
        for hist_event in historical_events[-100:]:
            if hist_event.dest_ip:
                usual_destinations.add(hist_event.dest_ip)
        
        if event.dest_ip and event.dest_ip not in usual_destinations:
            return True
        
        return False
    
    def _matches_threat_indicator(self, event: NormalizedEvent, indicator: Dict[str, Any]) -> bool:
        """Check if event matches threat indicator"""
        indicator_type = indicator.get('type')
        indicator_value = indicator.get('value')
        
        if indicator_type == 'ip':
            return event.source_ip == indicator_value or event.dest_ip == indicator_value
        elif indicator_type == 'domain':
            return indicator_value in event.message or indicator_value in event.description
        elif indicator_type == 'hash':
            return event.file_hash == indicator_value
        elif indicator_type == 'process':
            return event.process_name == indicator_value
        
        return False
    
    def _create_frequency_alert(self, rule: CorrelationRule, events: List[NormalizedEvent], group_key: str = "") -> CorrelationAlert:
        """Create frequency-based correlation alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"High frequency: {rule.name}",
            description=f"Detected {len(events)} events matching pattern within {rule.time_window}",
            severity=rule.severity,
            confidence=min(0.9, len(events) / (rule.threshold_count * 2)),
            triggering_events=events,
            event_count=len(events)
        )
        
        # Extract context information
        self._enrich_alert_context(alert, events)
        
        if events:
            alert.time_span = events[-1].timestamp - events[0].timestamp
        
        return alert
    
    def _create_sequence_alert(self, rule: CorrelationRule, events: List[NormalizedEvent]) -> CorrelationAlert:
        """Create sequence-based correlation alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"Attack sequence: {rule.name}",
            description=f"Detected attack sequence with {len(events)} steps",
            severity=rule.severity,
            confidence=0.95,  # High confidence for sequence detection
            triggering_events=events,
            event_count=len(events)
        )
        
        self._enrich_alert_context(alert, events)
        
        if events:
            alert.time_span = events[-1].timestamp - events[0].timestamp
        
        return alert
    
    def _create_statistical_alert(self, rule: CorrelationRule, events: List[NormalizedEvent], 
                                current_rate: float, baseline_rate: float) -> CorrelationAlert:
        """Create statistical anomaly alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"Statistical anomaly: {rule.name}",
            description=f"Event rate {current_rate:.2f}/min exceeds baseline {baseline_rate:.2f}/min",
            severity=rule.severity,
            confidence=min(0.9, current_rate / baseline_rate / 10),
            triggering_events=events,
            event_count=len(events)
        )
        
        self._enrich_alert_context(alert, events)
        return alert
    
    def _create_geographical_alert(self, rule: CorrelationRule, events: List[NormalizedEvent]) -> CorrelationAlert:
        """Create geographical correlation alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"Impossible travel: {rule.name}",
            description="User appeared in multiple locations within impossible timeframe",
            severity=rule.severity,
            confidence=0.85,
            triggering_events=events,
            event_count=len(events)
        )
        
        self._enrich_alert_context(alert, events)
        return alert
    
    def _create_behavioral_alert(self, rule: CorrelationRule, events: List[NormalizedEvent], 
                               baseline_events: List[NormalizedEvent]) -> CorrelationAlert:
        """Create behavioral anomaly alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"Behavioral anomaly: {rule.name}",
            description="Detected unusual user behavior pattern",
            severity=rule.severity,
            confidence=0.7,
            triggering_events=events,
            event_count=len(events)
        )
        
        self._enrich_alert_context(alert, events)
        return alert
    
    def _create_threat_hunting_alert(self, rule: CorrelationRule, events: List[NormalizedEvent], 
                                   indicator: Dict[str, Any]) -> CorrelationAlert:
        """Create threat hunting alert"""
        alert = CorrelationAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            title=f"Threat indicator: {rule.name}",
            description=f"Detected threat indicator: {indicator}",
            severity=rule.severity,
            confidence=0.9,
            triggering_events=events,
            event_count=len(events)
        )
        
        self._enrich_alert_context(alert, events)
        alert.iocs.append(indicator.get('value', ''))
        return alert
    
    def _enrich_alert_context(self, alert: CorrelationAlert, events: List[NormalizedEvent]):
        """Enrich alert with context information from events"""
        for event in events:
            if event.source_host:
                alert.affected_assets.add(event.source_host)
            if event.dest_host:
                alert.affected_assets.add(event.dest_host)
            if event.source_ip:
                alert.source_ips.add(event.source_ip)
            if event.dest_ip:
                alert.dest_ips.add(event.dest_ip)
            if event.source_user:
                alert.users.add(event.source_user)
            if event.dest_user:
                alert.users.add(event.dest_user)
    
    def _initialize_builtin_rules(self):
        """Initialize built-in correlation rules"""
        
        # Brute force authentication rule
        brute_force_rule = CorrelationRule(
            rule_id="auth_brute_force",
            name="Authentication Brute Force",
            description="Multiple failed authentication attempts",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=CorrelationSeverity.HIGH,
            event_filters={'category': EventCategory.AUTHENTICATION, 'outcome': 'failure'},
            time_window=timedelta(minutes=5),
            threshold_count=5,
            group_by_fields=['source_ip', 'dest_user']
        )
        self.add_rule(brute_force_rule)
        
        # Port scan detection
        port_scan_rule = CorrelationRule(
            rule_id="network_port_scan",
            name="Network Port Scan",
            description="Multiple connection attempts to different ports",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=CorrelationSeverity.MEDIUM,
            event_filters={'category': EventCategory.NETWORK_TRAFFIC},
            time_window=timedelta(minutes=2),
            threshold_count=10,
            group_by_fields=['source_ip', 'dest_ip']
        )
        self.add_rule(port_scan_rule)
        
        # Privilege escalation sequence
        privesc_rule = CorrelationRule(
            rule_id="privilege_escalation",
            name="Privilege Escalation",
            description="Authentication followed by privileged access",
            rule_type=CorrelationRuleType.SEQUENCE,
            severity=CorrelationSeverity.HIGH,
            sequence_pattern=[
                {'category': EventCategory.AUTHENTICATION, 'outcome': 'success'},
                {'category': EventCategory.AUTHORIZATION, 'message': 'admin'}
            ],
            time_window=timedelta(minutes=10)
        )
        self.add_rule(privesc_rule)
        
        # Data exfiltration
        data_exfil_rule = CorrelationRule(
            rule_id="data_exfiltration",
            name="Potential Data Exfiltration",
            description="Large volume of data access followed by network activity",
            rule_type=CorrelationRuleType.FREQUENCY,
            severity=CorrelationSeverity.CRITICAL,
            event_filters={'category': EventCategory.DATA_ACCESS},
            time_window=timedelta(minutes=15),
            threshold_count=20,
            group_by_fields=['source_user']
        )
        self.add_rule(data_exfil_rule)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get correlation engine statistics"""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "events_in_buffer": len(self.event_buffer.events),
            "total_alerts": len(self.generated_alerts),
            "alerts_by_severity": {
                severity.value: len([a for a in self.generated_alerts if a.severity == severity])
                for severity in CorrelationSeverity
            }
        }