"""
Production Security Monitor
Advanced real-time security monitoring, threat detection, and incident response
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import ipaddress

# Machine Learning for anomaly detection
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Network and security libraries
try:
    import psutil
    import geoip2.database
    import yara
    SECURITY_LIBS_AVAILABLE = True
except ImportError:
    SECURITY_LIBS_AVAILABLE = False

from ..services.interfaces import SecurityMonitoringService
from ..infrastructure.observability import add_trace_context, get_metrics_collector


class SecurityEventType(Enum):
    """Security event classification"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_LOGIN = "suspicious_login"
    BRUTE_FORCE_ATTEMPT = "brute_force"
    ACCOUNT_LOCKOUT = "account_lockout"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS_VIOLATION = "data_access_violation"
    MALWARE_DETECTION = "malware_detection"
    NETWORK_INTRUSION = "network_intrusion"
    DDoS_ATTACK = "ddos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    FILE_INTEGRITY_VIOLATION = "file_integrity"
    POLICY_VIOLATION = "policy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SYSTEM_COMPROMISE = "system_compromise"
    DATA_EXFILTRATION = "data_exfiltration"
    INSIDER_THREAT = "insider_threat"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RiskLevel(Enum):
    """Risk assessment levels"""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    BLOCK_USER = "block_user"
    QUARANTINE_SYSTEM = "quarantine_system"
    ISOLATE_NETWORK = "isolate_network"
    FORCE_PASSWORD_RESET = "force_password_reset"
    DISABLE_ACCOUNT = "disable_account"
    ESCALATE_TO_SOC = "escalate_to_soc"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class SecurityEvent:
    """Security event with comprehensive metadata"""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source_ip: str
    source_country: Optional[str]
    user_id: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    risk_level: RiskLevel
    confidence_score: float
    raw_data: Dict[str, Any]
    context: Dict[str, Any]
    tags: List[str]
    correlation_id: Optional[str]


@dataclass
class ThreatIndicator:
    """Threat indicator with scoring"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, pattern
    value: str
    threat_level: RiskLevel
    confidence: float
    first_seen: datetime
    last_seen: datetime
    count: int
    sources: List[str]
    context: Dict[str, Any]


@dataclass
class SecurityIncident:
    """Security incident aggregation"""
    incident_id: str
    title: str
    description: str
    incident_type: SecurityEventType
    risk_level: RiskLevel
    status: str  # open, investigating, contained, resolved, closed
    created_at: datetime
    updated_at: datetime
    events: List[str]  # Event IDs
    affected_systems: List[str]
    affected_users: List[str]
    indicators: List[str]  # Indicator IDs
    response_actions: List[ResponseAction]
    timeline: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BehaviorAnalyzer:
    """ML-powered behavior analysis"""
    
    def __init__(self):
        self.user_profiles = {}
        self.system_baseline = {}
        self.anomaly_detector = None
        self.behavior_classifier = None
        self.scaler = StandardScaler()
        
        if ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.behavior_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    
    def create_user_profile(self, user_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create baseline behavior profile for user"""
        profile = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'login_patterns': {
                'typical_hours': [],
                'common_locations': [],
                'usual_devices': [],
                'average_session_duration': 0
            },
            'access_patterns': {
                'frequent_resources': [],
                'typical_actions': [],
                'data_access_volume': 0
            },
            'behavioral_metrics': {
                'keystroke_dynamics': {},
                'mouse_patterns': {},
                'navigation_patterns': {}
            },
            'risk_factors': {
                'failed_login_attempts': 0,
                'privilege_requests': 0,
                'policy_violations': 0
            },
            'anomaly_score': 0.0,
            'last_updated': datetime.utcnow()
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def update_profile(self, user_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile with new event data"""
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id, event_data)
        
        profile = self.user_profiles[user_id]
        
        # Update login patterns
        if event_data.get('event_type') == 'login':
            login_hour = datetime.utcnow().hour
            profile['login_patterns']['typical_hours'].append(login_hour)
            
            if 'location' in event_data:
                profile['login_patterns']['common_locations'].append(event_data['location'])
            
            if 'device' in event_data:
                profile['login_patterns']['usual_devices'].append(event_data['device'])
        
        # Update access patterns
        if 'resource' in event_data:
            profile['access_patterns']['frequent_resources'].append(event_data['resource'])
        
        if 'action' in event_data:
            profile['access_patterns']['typical_actions'].append(event_data['action'])
        
        # Calculate anomaly score
        profile['anomaly_score'] = self._calculate_anomaly_score(profile, event_data)
        profile['last_updated'] = datetime.utcnow()
        
        return profile
    
    def _calculate_anomaly_score(self, profile: Dict[str, Any], event_data: Dict[str, Any]) -> float:
        """Calculate anomaly score for current event"""
        if not ML_AVAILABLE or not self.anomaly_detector:
            return 0.0
        
        try:
            # Extract features for anomaly detection
            features = self._extract_behavioral_features(profile, event_data)
            
            if len(features) == 0:
                return 0.0
            
            # Reshape for sklearn
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Predict anomaly score
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Normalize to 0-1 range
            normalized_score = max(0.0, min(1.0, (anomaly_score + 0.5) / 1.0))
            
            return normalized_score
            
        except Exception as e:
            logging.error(f"Anomaly score calculation failed: {e}")
            return 0.0
    
    def _extract_behavioral_features(self, profile: Dict[str, Any], event_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features for ML analysis"""
        features = []
        
        try:
            # Time-based features
            current_hour = datetime.utcnow().hour
            typical_hours = profile['login_patterns']['typical_hours']
            
            if typical_hours:
                hour_deviation = min([abs(current_hour - h) for h in typical_hours[-10:]])  # Last 10 logins
                features.append(hour_deviation)
            else:
                features.append(12)  # Maximum possible deviation
            
            # Location-based features
            current_location = event_data.get('location', 'unknown')
            common_locations = profile['login_patterns']['common_locations']
            
            location_familiarity = 1.0 if current_location in common_locations[-5:] else 0.0
            features.append(location_familiarity)
            
            # Device-based features
            current_device = event_data.get('device', 'unknown')
            usual_devices = profile['login_patterns']['usual_devices']
            
            device_familiarity = 1.0 if current_device in usual_devices[-3:] else 0.0
            features.append(device_familiarity)
            
            # Access pattern features
            resource_access_frequency = profile['access_patterns']['frequent_resources'].count(
                event_data.get('resource', '')
            )
            features.append(min(resource_access_frequency, 10) / 10.0)  # Normalize
            
            # Risk factor features
            features.append(profile['risk_factors']['failed_login_attempts'] / 10.0)
            features.append(profile['risk_factors']['privilege_requests'] / 5.0)
            features.append(profile['risk_factors']['policy_violations'] / 3.0)
            
            # Pad to fixed size
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return [0.0] * 20


class ThreatDetectionEngine:
    """Advanced threat detection with ML and rules"""
    
    def __init__(self):
        self.detection_rules = {}
        self.threat_patterns = {}
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_indicators = {}
        self.attack_patterns = {}
        
        # Initialize detection rules
        self._initialize_detection_rules()
    
    def _initialize_detection_rules(self):
        """Initialize built-in detection rules"""
        self.detection_rules.update({
            'brute_force_login': {
                'condition': 'failed_login_count > 5 in 5_minutes',
                'risk_level': RiskLevel.HIGH,
                'response': [ResponseAction.BLOCK_IP, ResponseAction.ALERT]
            },
            'suspicious_login_location': {
                'condition': 'login_location not in user_common_locations and distance > 1000km',
                'risk_level': RiskLevel.MEDIUM,
                'response': [ResponseAction.ALERT, ResponseAction.FORCE_PASSWORD_RESET]
            },
            'privilege_escalation_attempt': {
                'condition': 'failed_auth_count > 3 and target_resource = admin',
                'risk_level': RiskLevel.CRITICAL,
                'response': [ResponseAction.BLOCK_USER, ResponseAction.ESCALATE_TO_SOC]
            },
            'data_exfiltration_pattern': {
                'condition': 'data_download_volume > 1GB in 1_hour',
                'risk_level': RiskLevel.HIGH,
                'response': [ResponseAction.ALERT, ResponseAction.QUARANTINE_SYSTEM]
            },
            'sql_injection_attempt': {
                'condition': 'request_contains_sql_patterns and response_error',
                'risk_level': RiskLevel.HIGH,
                'response': [ResponseAction.BLOCK_IP, ResponseAction.ALERT]
            },
            'anomalous_behavior': {
                'condition': 'user_anomaly_score > 0.8',
                'risk_level': RiskLevel.MEDIUM,
                'response': [ResponseAction.ALERT]
            }
        })
    
    def analyze_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze security event for threats"""
        analysis_result = {
            'event_id': event.event_id,
            'threat_detected': False,
            'matched_rules': [],
            'risk_assessment': event.risk_level,
            'confidence_score': event.confidence_score,
            'recommendations': [],
            'response_actions': []
        }
        
        try:
            # Rule-based detection
            matched_rules = self._apply_detection_rules(event)
            analysis_result['matched_rules'] = matched_rules
            
            # Behavioral analysis
            if event.user_id:
                behavior_analysis = self._analyze_behavior(event)
                analysis_result['behavior_analysis'] = behavior_analysis
            
            # Pattern matching
            pattern_matches = self._match_attack_patterns(event)
            analysis_result['pattern_matches'] = pattern_matches
            
            # Determine overall threat level
            if matched_rules or pattern_matches:
                analysis_result['threat_detected'] = True
                
                # Calculate combined risk level
                max_risk = max([
                    *[rule['risk_level'] for rule in matched_rules],
                    *[pattern['risk_level'] for pattern in pattern_matches],
                    event.risk_level
                ], key=lambda x: self._risk_level_weight(x))
                
                analysis_result['risk_assessment'] = max_risk
            
            # Generate recommendations
            analysis_result['recommendations'] = self._generate_recommendations(analysis_result)
            
            # Determine response actions
            analysis_result['response_actions'] = self._determine_response_actions(analysis_result)
            
        except Exception as e:
            logging.error(f"Threat analysis failed: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _apply_detection_rules(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Apply detection rules to event"""
        matched_rules = []
        
        for rule_name, rule_config in self.detection_rules.items():
            try:
                if self._evaluate_rule_condition(rule_config['condition'], event):
                    matched_rules.append({
                        'rule_name': rule_name,
                        'risk_level': rule_config['risk_level'],
                        'response': rule_config['response'],
                        'confidence': 0.9
                    })
            except Exception as e:
                logging.warning(f"Rule evaluation failed for {rule_name}: {e}")
        
        return matched_rules
    
    def _evaluate_rule_condition(self, condition: str, event: SecurityEvent) -> bool:
        """Evaluate rule condition against event"""
        # Simplified rule evaluation - in production would use a proper rule engine
        
        if 'failed_login_count > 5' in condition:
            # Check recent failed logins from same IP
            return self._check_failed_login_threshold(event.source_ip, 5, 300)
        
        elif 'login_location not in user_common_locations' in condition:
            return self._check_suspicious_location(event)
        
        elif 'failed_auth_count > 3 and target_resource = admin' in condition:
            return self._check_privilege_escalation(event)
        
        elif 'data_download_volume > 1GB' in condition:
            return self._check_data_exfiltration(event)
        
        elif 'request_contains_sql_patterns' in condition:
            return self._check_sql_injection(event)
        
        elif 'user_anomaly_score > 0.8' in condition:
            return self._check_anomaly_score(event)
        
        return False
    
    def _check_failed_login_threshold(self, source_ip: str, threshold: int, time_window: int) -> bool:
        """Check if failed login threshold exceeded"""
        # In production, this would query the event store
        # For now, return a simple simulation
        return False  # Would implement actual check
    
    def _check_suspicious_location(self, event: SecurityEvent) -> bool:
        """Check for suspicious login location"""
        # Would implement geolocation checking
        return event.source_country not in ['US', 'CA', 'GB']  # Example
    
    def _check_privilege_escalation(self, event: SecurityEvent) -> bool:
        """Check for privilege escalation attempt"""
        return (event.event_type == SecurityEventType.AUTHORIZATION_FAILURE and 
                'admin' in event.resource.lower())
    
    def _check_data_exfiltration(self, event: SecurityEvent) -> bool:
        """Check for data exfiltration patterns"""
        return (event.raw_data.get('response_size', 0) > 1024 * 1024 * 100)  # 100MB
    
    def _check_sql_injection(self, event: SecurityEvent) -> bool:
        """Check for SQL injection patterns"""
        request_data = event.raw_data.get('request_data', '')
        sql_patterns = ['union select', 'drop table', '--', '/*', 'xp_cmdshell']
        return any(pattern in request_data.lower() for pattern in sql_patterns)
    
    def _check_anomaly_score(self, event: SecurityEvent) -> bool:
        """Check user anomaly score"""
        if event.user_id:
            profile = self.behavior_analyzer.user_profiles.get(event.user_id)
            return profile and profile.get('anomaly_score', 0) > 0.8
        return False
    
    def _analyze_behavior(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze user behavior for anomalies"""
        if not event.user_id:
            return {'analysis': 'no_user_context'}
        
        # Update user profile
        event_data = {
            'event_type': event.event_type.value,
            'timestamp': event.timestamp,
            'location': event.source_country,
            'resource': event.resource,
            'action': event.action,
            'device': event.user_agent
        }
        
        profile = self.behavior_analyzer.update_profile(event.user_id, event_data)
        
        return {
            'anomaly_score': profile['anomaly_score'],
            'profile_age': (datetime.utcnow() - profile['created_at']).days,
            'recent_activity': len(profile['login_patterns']['typical_hours'][-10:]),
            'risk_factors': profile['risk_factors']
        }
    
    def _match_attack_patterns(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Match event against known attack patterns"""
        matches = []
        
        # MITRE ATT&CK pattern matching
        attack_patterns = {
            'T1078': {  # Valid Accounts
                'indicators': ['multiple_failed_auth', 'unusual_login_time'],
                'risk_level': RiskLevel.MEDIUM
            },
            'T1110': {  # Brute Force
                'indicators': ['rapid_auth_attempts', 'multiple_user_targets'],
                'risk_level': RiskLevel.HIGH
            },
            'T1566': {  # Phishing
                'indicators': ['suspicious_email_link', 'credential_harvest'],
                'risk_level': RiskLevel.HIGH
            }
        }
        
        for technique_id, pattern in attack_patterns.items():
            if self._check_pattern_indicators(event, pattern['indicators']):
                matches.append({
                    'technique_id': technique_id,
                    'risk_level': pattern['risk_level'],
                    'confidence': 0.7
                })
        
        return matches
    
    def _check_pattern_indicators(self, event: SecurityEvent, indicators: List[str]) -> bool:
        """Check if event matches attack pattern indicators"""
        # Simplified pattern matching
        event_indicators = []
        
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            event_indicators.append('multiple_failed_auth')
        
        if datetime.utcnow().hour < 6 or datetime.utcnow().hour > 22:
            event_indicators.append('unusual_login_time')
        
        # Check if any indicators match
        return bool(set(indicators) & set(event_indicators))
    
    def _risk_level_weight(self, risk_level: RiskLevel) -> int:
        """Get numeric weight for risk level"""
        weights = {
            RiskLevel.INFORMATIONAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
            RiskLevel.EMERGENCY: 5
        }
        return weights.get(risk_level, 0)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if analysis['threat_detected']:
            recommendations.append("Immediate investigation required")
            
            if analysis['risk_assessment'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append("Consider blocking source IP address")
                recommendations.append("Review related user accounts")
            
            for rule in analysis['matched_rules']:
                if 'brute_force' in rule['rule_name']:
                    recommendations.append("Implement rate limiting")
                    recommendations.append("Enable account lockout policies")
                
                elif 'privilege_escalation' in rule['rule_name']:
                    recommendations.append("Review user permissions")
                    recommendations.append("Enable privileged access monitoring")
        
        return recommendations
    
    def _determine_response_actions(self, analysis: Dict[str, Any]) -> List[ResponseAction]:
        """Determine automated response actions"""
        actions = []
        
        if not analysis['threat_detected']:
            return [ResponseAction.LOG_ONLY]
        
        # Collect actions from matched rules
        for rule in analysis['matched_rules']:
            actions.extend(rule.get('response', []))
        
        # Add risk-based actions
        risk_level = analysis['risk_assessment']
        
        if risk_level == RiskLevel.CRITICAL:
            actions.extend([ResponseAction.ESCALATE_TO_SOC, ResponseAction.ALERT])
        elif risk_level == RiskLevel.HIGH:
            actions.extend([ResponseAction.ALERT, ResponseAction.BLOCK_IP])
        elif risk_level == RiskLevel.MEDIUM:
            actions.append(ResponseAction.ALERT)
        
        return list(set(actions))  # Remove duplicates


class ProductionSecurityMonitor(SecurityMonitoringService):
    """Production security monitoring service"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.metrics = get_metrics_collector()
        
        # Core components
        self.threat_detection = ThreatDetectionEngine()
        self.event_store = deque(maxlen=100000)  # In-memory event store
        self.incidents = {}
        self.threat_indicators = {}
        
        # Monitoring state
        self.active_threats = {}
        self.blocked_ips = set()
        self.blocked_users = set()
        self.alert_suppression = {}
        
        # Performance metrics
        self.metrics_data = {
            'events_processed': 0,
            'threats_detected': 0,
            'incidents_created': 0,
            'false_positives': 0,
            'response_actions_taken': 0,
            'average_detection_time': 0.0
        }
        
        # Initialize components
        asyncio.create_task(self.initialize())
    
    async def initialize(self):
        """Initialize security monitoring components"""
        try:
            # Load threat intelligence feeds
            await self._load_threat_intelligence()
            
            # Initialize detection rules
            await self._initialize_custom_rules()
            
            # Start monitoring services
            asyncio.create_task(self._incident_correlator())
            asyncio.create_task(self._threat_intelligence_updater())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._cleanup_old_data())
            
            self.logger.info("Security Monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Security Monitor: {e}")
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence feeds"""
        try:
            # In production, this would load from external feeds
            sample_indicators = [
                ThreatIndicator(
                    indicator_id=str(uuid4()),
                    indicator_type="ip",
                    value="192.0.2.1",
                    threat_level=RiskLevel.HIGH,
                    confidence=0.9,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    count=1,
                    sources=["internal_analysis"],
                    context={"category": "malware_c2"}
                ),
                ThreatIndicator(
                    indicator_id=str(uuid4()),
                    indicator_type="domain",
                    value="evil.example.com",
                    threat_level=RiskLevel.CRITICAL,
                    confidence=0.95,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    count=1,
                    sources=["threat_feed"],
                    context={"category": "phishing"}
                )
            ]
            
            for indicator in sample_indicators:
                self.threat_indicators[indicator.indicator_id] = indicator
            
            self.logger.info(f"Loaded {len(sample_indicators)} threat indicators")
            
        except Exception as e:
            self.logger.error(f"Failed to load threat intelligence: {e}")
    
    async def _initialize_custom_rules(self):
        """Initialize custom detection rules"""
        # Add organization-specific rules
        custom_rules = {
            'admin_access_after_hours': {
                'condition': 'admin_access and (hour < 6 or hour > 20)',
                'risk_level': RiskLevel.MEDIUM,
                'response': [ResponseAction.ALERT]
            },
            'mass_data_download': {
                'condition': 'download_count > 100 in 10_minutes',
                'risk_level': RiskLevel.HIGH,
                'response': [ResponseAction.ALERT, ResponseAction.QUARANTINE_SYSTEM]
            }
        }
        
        self.threat_detection.detection_rules.update(custom_rules)
        self.logger.info(f"Loaded {len(custom_rules)} custom detection rules")
    
    @add_trace_context
    async def process_security_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming security event"""
        start_time = time.time()
        
        try:
            self.metrics.counter('security_events_received').inc()
            
            # Create security event
            event = self._create_security_event(event_data)
            
            # Store event
            self.event_store.append(event)
            
            # Analyze for threats
            analysis = self.threat_detection.analyze_event(event)
            
            # Handle detected threats
            if analysis['threat_detected']:
                await self._handle_threat_detection(event, analysis)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.histogram('security_event_processing_time').observe(processing_time)
            self.metrics_data['events_processed'] += 1
            
            return {
                'event_id': event.event_id,
                'processed': True,
                'threat_detected': analysis['threat_detected'],
                'risk_level': analysis['risk_assessment'].value,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.metrics.counter('security_event_processing_errors').inc()
            self.logger.error(f"Failed to process security event: {e}")
            return {
                'processed': False,
                'error': str(e)
            }
    
    def _create_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Create SecurityEvent from raw data"""
        # Extract and normalize event data
        event_id = event_data.get('event_id', str(uuid4()))
        
        # Determine event type
        event_type = self._classify_event_type(event_data)
        
        # Extract geolocation
        source_country = self._get_country_from_ip(event_data.get('source_ip', ''))
        
        # Calculate initial risk level
        risk_level = self._calculate_initial_risk(event_data, event_type)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(event_data, event_type)
        
        return SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            source_ip=event_data.get('source_ip', ''),
            source_country=source_country,
            user_id=event_data.get('user_id'),
            user_agent=event_data.get('user_agent'),
            resource=event_data.get('resource', ''),
            action=event_data.get('action', ''),
            risk_level=risk_level,
            confidence_score=confidence_score,
            raw_data=event_data,
            context=self._extract_context(event_data),
            tags=self._generate_tags(event_data),
            correlation_id=event_data.get('correlation_id')
        )
    
    def _classify_event_type(self, event_data: Dict[str, Any]) -> SecurityEventType:
        """Classify event type from raw data"""
        event_type = event_data.get('event_type', '').lower()
        
        # Map common event types
        type_mapping = {
            'login_failed': SecurityEventType.AUTHENTICATION_FAILURE,
            'login_success': SecurityEventType.AUTHENTICATION_SUCCESS,
            'auth_failed': SecurityEventType.AUTHENTICATION_FAILURE,
            'access_denied': SecurityEventType.AUTHORIZATION_FAILURE,
            'admin_access': SecurityEventType.PRIVILEGE_ESCALATION,
            'sql_error': SecurityEventType.SQL_INJECTION,
            'xss_detected': SecurityEventType.XSS_ATTEMPT,
            'file_modified': SecurityEventType.FILE_INTEGRITY_VIOLATION,
            'policy_violation': SecurityEventType.POLICY_VIOLATION
        }
        
        return type_mapping.get(event_type, SecurityEventType.ANOMALOUS_BEHAVIOR)
    
    def _get_country_from_ip(self, ip_address: str) -> Optional[str]:
        """Get country from IP address using GeoIP"""
        try:
            if SECURITY_LIBS_AVAILABLE:
                # In production, would use actual GeoIP database
                # For now, return mock data
                if ip_address.startswith('192.168.') or ip_address.startswith('10.'):
                    return 'US'  # Internal IP
                else:
                    return 'Unknown'
            return None
        except Exception:
            return None
    
    def _calculate_initial_risk(self, event_data: Dict[str, Any], event_type: SecurityEventType) -> RiskLevel:
        """Calculate initial risk level for event"""
        # Base risk by event type
        risk_mapping = {
            SecurityEventType.AUTHENTICATION_FAILURE: RiskLevel.LOW,
            SecurityEventType.AUTHORIZATION_FAILURE: RiskLevel.MEDIUM,
            SecurityEventType.PRIVILEGE_ESCALATION: RiskLevel.HIGH,
            SecurityEventType.SQL_INJECTION: RiskLevel.HIGH,
            SecurityEventType.XSS_ATTEMPT: RiskLevel.MEDIUM,
            SecurityEventType.MALWARE_DETECTION: RiskLevel.CRITICAL,
            SecurityEventType.DATA_EXFILTRATION: RiskLevel.CRITICAL,
            SecurityEventType.SYSTEM_COMPROMISE: RiskLevel.EMERGENCY
        }
        
        base_risk = risk_mapping.get(event_type, RiskLevel.LOW)
        
        # Adjust based on context
        if event_data.get('admin_user'):
            base_risk = RiskLevel(min(base_risk.value, RiskLevel.HIGH.value))
        
        if event_data.get('external_ip'):
            # Increase risk for external access
            pass
        
        return base_risk
    
    def _calculate_confidence(self, event_data: Dict[str, Any], event_type: SecurityEventType) -> float:
        """Calculate confidence score for event classification"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data quality
        if event_data.get('source_ip'):
            confidence += 0.1
        if event_data.get('user_id'):
            confidence += 0.1
        if event_data.get('timestamp'):
            confidence += 0.1
        if event_data.get('user_agent'):
            confidence += 0.1
        
        # Adjust based on event type certainty
        if event_type in [SecurityEventType.AUTHENTICATION_FAILURE, SecurityEventType.AUTHENTICATION_SUCCESS]:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_context(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual information from event"""
        context = {}
        
        # System context
        if 'system_load' in event_data:
            context['system_load'] = event_data['system_load']
        
        # Request context
        if 'request_size' in event_data:
            context['request_size'] = event_data['request_size']
        
        # Session context
        if 'session_id' in event_data:
            context['session_id'] = event_data['session_id']
        
        return context
    
    def _generate_tags(self, event_data: Dict[str, Any]) -> List[str]:
        """Generate tags for event categorization"""
        tags = []
        
        # Source tags
        if event_data.get('source_ip', '').startswith('192.168.'):
            tags.append('internal')
        else:
            tags.append('external')
        
        # User tags
        if event_data.get('admin_user'):
            tags.append('admin_user')
        
        # Time tags
        hour = datetime.utcnow().hour
        if hour < 6 or hour > 22:
            tags.append('after_hours')
        
        return tags
    
    async def _handle_threat_detection(self, event: SecurityEvent, analysis: Dict[str, Any]):
        """Handle detected threat"""
        try:
            self.metrics_data['threats_detected'] += 1
            
            # Execute response actions
            for action in analysis['response_actions']:
                await self._execute_response_action(action, event, analysis)
            
            # Create or update incident
            incident_id = await self._create_or_update_incident(event, analysis)
            
            # Send alerts
            await self._send_security_alert(event, analysis, incident_id)
            
            self.logger.warning(f"Threat detected: {event.event_id} - {analysis['risk_assessment'].value}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle threat detection: {e}")
    
    async def _execute_response_action(self, action: ResponseAction, event: SecurityEvent, analysis: Dict[str, Any]):
        """Execute automated response action"""
        try:
            self.metrics_data['response_actions_taken'] += 1
            
            if action == ResponseAction.BLOCK_IP:
                self.blocked_ips.add(event.source_ip)
                self.logger.info(f"Blocked IP address: {event.source_ip}")
            
            elif action == ResponseAction.BLOCK_USER and event.user_id:
                self.blocked_users.add(event.user_id)
                self.logger.info(f"Blocked user: {event.user_id}")
            
            elif action == ResponseAction.ALERT:
                # Alert handled separately
                pass
            
            elif action == ResponseAction.ESCALATE_TO_SOC:
                await self._escalate_to_soc(event, analysis)
            
            elif action == ResponseAction.QUARANTINE_SYSTEM:
                await self._quarantine_system(event)
            
            # Log action
            self.metrics.counter('security_response_actions').labels(action=action.value).inc()
            
        except Exception as e:
            self.logger.error(f"Failed to execute response action {action}: {e}")
    
    async def _create_or_update_incident(self, event: SecurityEvent, analysis: Dict[str, Any]) -> str:
        """Create new incident or update existing one"""
        # Check for related incidents
        related_incident = await self._find_related_incident(event)
        
        if related_incident:
            # Update existing incident
            incident = self.incidents[related_incident]
            incident.events.append(event.event_id)
            incident.updated_at = datetime.utcnow()
            
            # Update risk level if higher
            if self.threat_detection._risk_level_weight(analysis['risk_assessment']) > self.threat_detection._risk_level_weight(incident.risk_level):
                incident.risk_level = analysis['risk_assessment']
            
            return related_incident
        else:
            # Create new incident
            incident_id = str(uuid4())
            
            incident = SecurityIncident(
                incident_id=incident_id,
                title=f"{event.event_type.value.replace('_', ' ').title()} - {event.source_ip}",
                description=f"Security incident detected from {event.source_ip}",
                incident_type=event.event_type,
                risk_level=analysis['risk_assessment'],
                status='open',
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                events=[event.event_id],
                affected_systems=[event.source_ip],
                affected_users=[event.user_id] if event.user_id else [],
                indicators=[],
                response_actions=analysis['response_actions'],
                timeline=[{
                    'timestamp': datetime.utcnow().isoformat(),
                    'action': 'incident_created',
                    'details': 'Incident created from threat detection'
                }],
                metadata={'analysis': analysis}
            )
            
            self.incidents[incident_id] = incident
            self.metrics_data['incidents_created'] += 1
            
            return incident_id
    
    async def _find_related_incident(self, event: SecurityEvent) -> Optional[str]:
        """Find related open incident"""
        for incident_id, incident in self.incidents.items():
            if incident.status in ['open', 'investigating']:
                # Check if same source IP within last hour
                if (event.source_ip in incident.affected_systems and
                    (datetime.utcnow() - incident.updated_at) < timedelta(hours=1)):
                    return incident_id
                
                # Check if same user within last hour
                if (event.user_id and event.user_id in incident.affected_users and
                    (datetime.utcnow() - incident.updated_at) < timedelta(hours=1)):
                    return incident_id
        
        return None
    
    async def _send_security_alert(self, event: SecurityEvent, analysis: Dict[str, Any], incident_id: str):
        """Send security alert"""
        # Check alert suppression
        alert_key = f"{event.source_ip}_{event.event_type.value}"
        
        if alert_key in self.alert_suppression:
            last_alert = self.alert_suppression[alert_key]
            if (datetime.utcnow() - last_alert) < timedelta(minutes=5):
                return  # Suppress duplicate alerts
        
        alert_data = {
            'alert_id': str(uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_id': event.event_id,
            'incident_id': incident_id,
            'severity': analysis['risk_assessment'].value,
            'title': f"Security Threat Detected: {event.event_type.value}",
            'description': f"Threat detected from {event.source_ip}",
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'recommendations': analysis['recommendations']
        }
        
        # In production, would send to SIEM, email, Slack, etc.
        self.logger.warning(f"SECURITY ALERT: {alert_data}")
        
        # Update suppression
        self.alert_suppression[alert_key] = datetime.utcnow()
    
    async def _escalate_to_soc(self, event: SecurityEvent, analysis: Dict[str, Any]):
        """Escalate to Security Operations Center"""
        escalation_data = {
            'escalation_id': str(uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_id': event.event_id,
            'priority': 'high' if analysis['risk_assessment'] == RiskLevel.CRITICAL else 'medium',
            'details': analysis,
            'event_data': asdict(event)
        }
        
        # In production, would integrate with SOC ticketing system
        self.logger.critical(f"SOC ESCALATION: {escalation_data}")
    
    async def _quarantine_system(self, event: SecurityEvent):
        """Quarantine affected system"""
        quarantine_data = {
            'quarantine_id': str(uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'system': event.source_ip,
            'reason': f"Security threat detected: {event.event_type.value}"
        }
        
        # In production, would trigger network isolation
        self.logger.warning(f"SYSTEM QUARANTINE: {quarantine_data}")
    
    async def _incident_correlator(self):
        """Correlate events into incidents"""
        while True:
            try:
                # Look for patterns in recent events
                recent_events = [e for e in self.event_store if (datetime.utcnow() - e.timestamp).seconds < 3600]
                
                # Group events by source IP
                ip_groups = defaultdict(list)
                for event in recent_events:
                    ip_groups[event.source_ip].append(event)
                
                # Look for suspicious patterns
                for ip, events in ip_groups.items():
                    if len(events) > 10:  # Many events from same IP
                        self.logger.info(f"High activity from IP {ip}: {len(events)} events")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Incident correlation error: {e}")
                await asyncio.sleep(600)
    
    async def _threat_intelligence_updater(self):
        """Update threat intelligence feeds"""
        while True:
            try:
                # In production, would fetch from external feeds
                self.logger.info("Updating threat intelligence feeds")
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                self.logger.error(f"Threat intelligence update error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while True:
            try:
                # Update performance metrics
                self.metrics.gauge('security_events_in_store').set(len(self.event_store))
                self.metrics.gauge('active_incidents').set(len([i for i in self.incidents.values() if i.status == 'open']))
                self.metrics.gauge('blocked_ips').set(len(self.blocked_ips))
                self.metrics.gauge('blocked_users').set(len(self.blocked_users))
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_old_data(self):
        """Clean up old data"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                # Clean up old incidents
                to_remove = []
                for incident_id, incident in self.incidents.items():
                    if incident.status == 'closed' and incident.updated_at < cutoff_time:
                        to_remove.append(incident_id)
                
                for incident_id in to_remove:
                    del self.incidents[incident_id]
                
                # Clean up alert suppression
                self.alert_suppression = {
                    k: v for k, v in self.alert_suppression.items()
                    if (datetime.utcnow() - v) < timedelta(hours=1)
                }
                
                if to_remove:
                    self.logger.info(f"Cleaned up {len(to_remove)} old incidents")
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data"""
        try:
            # Active threats summary
            active_incidents = [i for i in self.incidents.values() if i.status in ['open', 'investigating']]
            
            # Recent events summary
            recent_events = [e for e in self.event_store if (datetime.utcnow() - e.timestamp).seconds < 3600]
            
            # Threat level distribution
            threat_levels = defaultdict(int)
            for incident in active_incidents:
                threat_levels[incident.risk_level.value] += 1
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'summary': {
                    'active_incidents': len(active_incidents),
                    'events_last_hour': len(recent_events),
                    'blocked_ips': len(self.blocked_ips),
                    'blocked_users': len(self.blocked_users),
                    'threat_indicators': len(self.threat_indicators)
                },
                'threat_levels': dict(threat_levels),
                'recent_incidents': [
                    {
                        'incident_id': i.incident_id,
                        'title': i.title,
                        'risk_level': i.risk_level.value,
                        'created_at': i.created_at.isoformat(),
                        'status': i.status
                    }
                    for i in sorted(active_incidents, key=lambda x: x.created_at, reverse=True)[:10]
                ],
                'performance_metrics': self.metrics_data,
                'system_status': 'healthy'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard: {e}")
            return {
                'error': 'Dashboard generation failed',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check security monitor health"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'threat_detection': 'operational',
                'event_processing': 'operational',
                'incident_management': 'operational',
                'alerting': 'operational'
            },
            'metrics': self.metrics_data,
            'event_store_size': len(self.event_store),
            'active_incidents': len([i for i in self.incidents.values() if i.status == 'open'])
        }