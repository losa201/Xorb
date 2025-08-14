"""
Advanced Threat Detection Engine with ML-based Anomaly Detection

This module provides enterprise-grade threat detection with:
- Real-time behavioral analytics and anomaly detection
- ML-powered attack pattern recognition
- Automated threat scoring and classification
- Multi-vector threat correlation
- Explainable AI for security decisions
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import uuid4
import logging
from collections import defaultdict, deque

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = structlog.get_logger("advanced_threat_detection")


class ThreatCategory(Enum):
    """Categories of security threats"""
    BRUTE_FORCE = "brute_force"
    ACCOUNT_TAKEOVER = "account_takeover"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_AUTOMATION = "malicious_automation"
    INSIDER_THREAT = "insider_threat"
    ADVANCED_PERSISTENT_THREAT = "apt"
    DENIAL_OF_SERVICE = "dos"
    INJECTION_ATTACK = "injection"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY_EXPLOIT = "zero_day"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    INFO = "info"           # Score: 0-25
    LOW = "low"             # Score: 26-40
    MEDIUM = "medium"       # Score: 41-60
    HIGH = "high"           # Score: 61-80
    CRITICAL = "critical"   # Score: 81-95
    EMERGENCY = "emergency" # Score: 96-100


class ConfidenceLevel(Enum):
    """ML model confidence levels"""
    VERY_LOW = "very_low"       # < 30%
    LOW = "low"                 # 30-50%
    MEDIUM = "medium"           # 50-70%
    HIGH = "high"               # 70-85%
    VERY_HIGH = "very_high"     # 85-95%
    CERTAIN = "certain"         # > 95%


@dataclass
class ThreatIndicator:
    """Individual threat indicator with scoring"""
    indicator_id: str
    category: ThreatCategory
    severity: ThreatSeverity
    confidence: ConfidenceLevel
    score: float  # 0-100
    description: str
    evidence: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    # Attribution
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Technical details
    attack_vector: Optional[str] = None
    payload: Optional[str] = None
    affected_resources: List[str] = field(default_factory=list)
    
    # ML model details
    model_name: Optional[str] = None
    feature_weights: Dict[str, float] = field(default_factory=dict)
    anomaly_score: Optional[float] = None


@dataclass
class BehavioralProfile:
    """User behavioral profile for anomaly detection"""
    user_id: str
    tenant_id: Optional[str]
    
    # Temporal patterns
    typical_login_hours: Set[int] = field(default_factory=set)
    typical_login_days: Set[int] = field(default_factory=set)
    average_session_duration: float = 0.0
    
    # Access patterns
    typical_endpoints: Set[str] = field(default_factory=set)
    typical_user_agents: Set[str] = field(default_factory=set)
    typical_ip_ranges: Set[str] = field(default_factory=set)
    
    # Activity patterns
    average_request_rate: float = 0.0
    typical_request_sizes: Tuple[float, float] = (0.0, 0.0)  # mean, std
    typical_response_times: Tuple[float, float] = (0.0, 0.0)
    
    # Data access patterns
    typical_data_volumes: Tuple[float, float] = (0.0, 0.0)
    sensitive_data_access_frequency: float = 0.0
    
    # Risk factors
    failed_login_rate: float = 0.0
    privilege_escalation_attempts: int = 0
    
    # Learning metrics
    samples_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0  # 0-1


@dataclass
class ThreatEvent:
    """Aggregated threat event with multiple indicators"""
    event_id: str
    category: ThreatCategory
    severity: ThreatSeverity
    overall_score: float
    confidence: ConfidenceLevel
    
    title: str
    description: str
    
    # Attribution
    source_ips: Set[str] = field(default_factory=set)
    user_ids: Set[str] = field(default_factory=set)
    tenant_ids: Set[str] = field(default_factory=set)
    
    # Timing
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    event_count: int = 1
    
    # Indicators
    indicators: List[ThreatIndicator] = field(default_factory=list)
    
    # Context
    attack_chain: List[str] = field(default_factory=list)
    affected_resources: Set[str] = field(default_factory=set)
    geographical_anomaly: bool = False
    temporal_anomaly: bool = False
    
    # Response
    auto_blocked: bool = False
    human_reviewed: bool = False
    false_positive: bool = False
    
    # Explainability
    explanation: str = ""
    evidence_summary: Dict[str, Any] = field(default_factory=dict)


class MLAnomalyDetector:
    """Machine learning-based anomaly detector"""
    
    def __init__(self, model_type: str = "isolation_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.last_training = None
        
        # Initialize model
        if model_type == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=200,
                max_samples=0.8,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def extract_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from security events"""
        features = []
        
        for event in events:
            feature_vector = [
                # Temporal features
                datetime.fromisoformat(event.get('timestamp', '2024-01-01')).hour,
                datetime.fromisoformat(event.get('timestamp', '2024-01-01')).weekday(),
                
                # Request features
                len(event.get('endpoint', '')),
                len(event.get('user_agent', '')),
                event.get('request_size', 0),
                event.get('response_time', 0),
                event.get('status_code', 200),
                
                # Authentication features
                1 if event.get('is_authenticated', False) else 0,
                len(event.get('user_roles', [])),
                
                # Rate limiting features
                event.get('requests_in_window', 0),
                event.get('rate_limit_remaining', 100),
                
                # Geographical features (simplified)
                hash(event.get('country_code', 'XX')) % 1000,
                hash(event.get('asn', '0')) % 1000,
                
                # Behavioral features
                len(event.get('unique_endpoints_accessed', [])),
                event.get('session_duration', 0),
                event.get('data_volume_accessed', 0),
                
                # Security features
                1 if event.get('failed_login', False) else 0,
                1 if event.get('privilege_escalation_attempt', False) else 0,
                1 if event.get('sensitive_data_access', False) else 0,
                
                # Pattern features
                event.get('entropy_score', 0.0),
                event.get('similarity_to_known_attacks', 0.0)
            ]
            
            features.append(feature_vector)
        
        self.feature_names = [
            'hour', 'weekday', 'endpoint_length', 'user_agent_length',
            'request_size', 'response_time', 'status_code', 'is_authenticated',
            'role_count', 'requests_in_window', 'rate_limit_remaining',
            'country_hash', 'asn_hash', 'endpoints_accessed', 'session_duration',
            'data_volume', 'failed_login', 'privilege_escalation', 'sensitive_access',
            'entropy_score', 'attack_similarity'
        ]
        
        return np.array(features)
    
    def train(self, training_events: List[Dict[str, Any]]) -> bool:
        """Train the anomaly detection model"""
        try:
            if len(training_events) < 100:
                logger.warning("Insufficient training data", sample_count=len(training_events))
                return False
            
            # Extract features
            features = self.extract_features(training_events)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            if self.model_type == "isolation_forest":
                self.model.fit(features_scaled)
            elif self.model_type == "dbscan":
                self.model.fit(features_scaled)
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            
            logger.info("ML anomaly detector trained",
                       model_type=self.model_type,
                       training_samples=len(training_events),
                       feature_count=len(self.feature_names))
            
            return True
        
        except Exception as e:
            logger.error("Failed to train anomaly detector", error=str(e))
            return False
    
    def predict_anomaly(self, event: Dict[str, Any]) -> Tuple[bool, float, Dict[str, float]]:
        """Predict if an event is anomalous"""
        if not self.is_trained:
            return False, 0.0, {}
        
        try:
            # Extract features
            features = self.extract_features([event])
            features_scaled = self.scaler.transform(features)
            
            # Predict
            if self.model_type == "isolation_forest":
                anomaly_score = self.model.decision_function(features_scaled)[0]
                is_anomaly = self.model.predict(features_scaled)[0] == -1
                
                # Convert to 0-1 probability
                probability = max(0, min(1, (0.5 - anomaly_score) * 2))
            else:
                # For DBSCAN, use distance to nearest cluster
                is_anomaly = True  # Simplified
                probability = 0.5
                anomaly_score = probability
            
            # Feature importance (simplified)
            feature_weights = {}
            if len(self.feature_names) == len(features[0]):
                for i, name in enumerate(self.feature_names):
                    feature_weights[name] = abs(features[0][i])
            
            return is_anomaly, probability, feature_weights
        
        except Exception as e:
            logger.error("Anomaly prediction failed", error=str(e))
            return False, 0.0, {}


class AttackPatternRecognizer:
    """Recognizes known attack patterns using rule-based and ML approaches"""
    
    def __init__(self):
        self.attack_patterns = self._load_attack_patterns()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_cache = {}
        self.is_trained = False
        
    def _load_attack_patterns(self) -> Dict[ThreatCategory, List[Dict[str, Any]]]:
        """Load known attack patterns"""
        return {
            ThreatCategory.BRUTE_FORCE: [
                {
                    "name": "Failed Login Spike",
                    "conditions": [
                        {"field": "failed_logins_per_minute", "operator": ">", "value": 10},
                        {"field": "unique_usernames_attempted", "operator": ">", "value": 5}
                    ],
                    "score": 70,
                    "confidence": 0.85
                },
                {
                    "name": "Credential Stuffing",
                    "conditions": [
                        {"field": "failed_logins_per_hour", "operator": ">", "value": 100},
                        {"field": "user_agent_diversity", "operator": "<", "value": 3},
                        {"field": "source_ip_diversity", "operator": ">", "value": 10}
                    ],
                    "score": 85,
                    "confidence": 0.90
                }
            ],
            ThreatCategory.DATA_EXFILTRATION: [
                {
                    "name": "Unusual Data Volume",
                    "conditions": [
                        {"field": "data_downloaded_gb", "operator": ">", "value": 1.0},
                        {"field": "time_of_day", "operator": "outside_business_hours", "value": True}
                    ],
                    "score": 75,
                    "confidence": 0.70
                },
                {
                    "name": "Sensitive Data Access Spike",
                    "conditions": [
                        {"field": "sensitive_files_accessed", "operator": ">", "value": 50},
                        {"field": "access_pattern_unusual", "operator": "=", "value": True}
                    ],
                    "score": 90,
                    "confidence": 0.85
                }
            ],
            ThreatCategory.INJECTION_ATTACK: [
                {
                    "name": "SQL Injection Attempt",
                    "conditions": [
                        {"field": "sql_keywords_in_params", "operator": ">", "value": 2},
                        {"field": "special_chars_ratio", "operator": ">", "value": 0.3}
                    ],
                    "score": 95,
                    "confidence": 0.95
                },
                {
                    "name": "XSS Attempt",
                    "conditions": [
                        {"field": "script_tags_in_input", "operator": ">", "value": 0},
                        {"field": "javascript_keywords", "operator": ">", "value": 1}
                    ],
                    "score": 80,
                    "confidence": 0.85
                }
            ],
            ThreatCategory.LATERAL_MOVEMENT: [
                {
                    "name": "Unusual Cross-Tenant Access",
                    "conditions": [
                        {"field": "tenants_accessed", "operator": ">", "value": 3},
                        {"field": "user_typical_tenant_count", "operator": "<", "value": 2}
                    ],
                    "score": 85,
                    "confidence": 0.75
                }
            ]
        }
    
    def recognize_pattern(self, events: List[Dict[str, Any]]) -> List[ThreatIndicator]:
        """Recognize attack patterns in event sequence"""
        indicators = []
        
        # Aggregate events for pattern matching
        aggregated = self._aggregate_events(events)
        
        # Check each pattern category
        for category, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if self._matches_pattern(aggregated, pattern):
                    indicator = ThreatIndicator(
                        indicator_id=str(uuid4()),
                        category=category,
                        severity=self._score_to_severity(pattern["score"]),
                        confidence=self._score_to_confidence(pattern["confidence"]),
                        score=pattern["score"],
                        description=f"Detected {pattern['name']} attack pattern",
                        evidence={"pattern": pattern["name"], "matched_conditions": pattern["conditions"]},
                        created_at=datetime.utcnow(),
                        model_name="rule_based_pattern_recognizer"
                    )
                    indicators.append(indicator)
        
        return indicators
    
    def _aggregate_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate events for pattern analysis"""
        if not events:
            return {}
        
        aggregated = {
            "event_count": len(events),
            "time_span_minutes": 0,
            "failed_logins_per_minute": 0,
            "failed_logins_per_hour": 0,
            "unique_usernames_attempted": 0,
            "unique_source_ips": 0,
            "user_agent_diversity": 0,
            "data_downloaded_gb": 0,
            "sensitive_files_accessed": 0,
            "sql_keywords_in_params": 0,
            "special_chars_ratio": 0.0,
            "script_tags_in_input": 0,
            "javascript_keywords": 0,
            "tenants_accessed": 0
        }
        
        # Calculate aggregations
        unique_usernames = set()
        unique_ips = set()
        unique_user_agents = set()
        unique_tenants = set()
        
        failed_logins = 0
        total_data_volume = 0
        sensitive_access_count = 0
        sql_injection_indicators = 0
        xss_indicators = 0
        
        first_event_time = None
        last_event_time = None
        
        for event in events:
            timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
            if first_event_time is None:
                first_event_time = timestamp
            last_event_time = timestamp
            
            # Collect unique values
            if event.get('username'):
                unique_usernames.add(event['username'])
            if event.get('source_ip'):
                unique_ips.add(event['source_ip'])
            if event.get('user_agent'):
                unique_user_agents.add(event['user_agent'])
            if event.get('tenant_id'):
                unique_tenants.add(event['tenant_id'])
            
            # Count events
            if event.get('failed_login'):
                failed_logins += 1
            
            total_data_volume += event.get('response_size', 0)
            
            if event.get('sensitive_data_access'):
                sensitive_access_count += 1
            
            # Check for injection patterns
            params = event.get('request_params', '')
            if any(keyword in params.lower() for keyword in ['select', 'union', 'drop', 'insert']):
                sql_injection_indicators += 1
            
            special_chars = sum(1 for c in params if c in "'\"();--/*")
            if len(params) > 0:
                aggregated["special_chars_ratio"] = max(
                    aggregated["special_chars_ratio"],
                    special_chars / len(params)
                )
            
            if '<script' in params.lower() or 'javascript:' in params.lower():
                xss_indicators += 1
        
        # Calculate final aggregated values
        if first_event_time and last_event_time:
            time_span = (last_event_time - first_event_time).total_seconds() / 60
            aggregated["time_span_minutes"] = time_span
            
            if time_span > 0:
                aggregated["failed_logins_per_minute"] = failed_logins / time_span
                aggregated["failed_logins_per_hour"] = failed_logins / (time_span / 60)
        
        aggregated.update({
            "unique_usernames_attempted": len(unique_usernames),
            "unique_source_ips": len(unique_ips),
            "user_agent_diversity": len(unique_user_agents),
            "source_ip_diversity": len(unique_ips),
            "data_downloaded_gb": total_data_volume / (1024**3),
            "sensitive_files_accessed": sensitive_access_count,
            "sql_keywords_in_params": sql_injection_indicators,
            "script_tags_in_input": xss_indicators,
            "javascript_keywords": xss_indicators,
            "tenants_accessed": len(unique_tenants)
        })
        
        return aggregated
    
    def _matches_pattern(self, aggregated: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if aggregated data matches attack pattern"""
        for condition in pattern["conditions"]:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]
            
            if field not in aggregated:
                continue
            
            actual_value = aggregated[field]
            
            if operator == ">" and actual_value <= value:
                return False
            elif operator == "<" and actual_value >= value:
                return False
            elif operator == "=" and actual_value != value:
                return False
            elif operator == "outside_business_hours" and value:
                # Check if current time is outside business hours
                current_hour = datetime.utcnow().hour
                if 9 <= current_hour <= 17:  # Business hours
                    return False
        
        return True
    
    def _score_to_severity(self, score: float) -> ThreatSeverity:
        """Convert numerical score to severity enum"""
        if score >= 96:
            return ThreatSeverity.EMERGENCY
        elif score >= 81:
            return ThreatSeverity.CRITICAL
        elif score >= 61:
            return ThreatSeverity.HIGH
        elif score >= 41:
            return ThreatSeverity.MEDIUM
        elif score >= 26:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO
    
    def _score_to_confidence(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence enum"""
        if confidence >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif confidence >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.30:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class BehavioralAnalyzer:
    """Analyzes user behavior patterns and detects anomalies"""
    
    def __init__(self):
        self.user_profiles: Dict[str, BehavioralProfile] = {}
        self.learning_window_days = 30
        self.minimum_samples = 50
        
    def update_profile(self, user_id: str, event: Dict[str, Any]) -> BehavioralProfile:
        """Update user behavioral profile with new event"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = BehavioralProfile(
                user_id=user_id,
                tenant_id=event.get('tenant_id')
            )
        
        profile = self.user_profiles[user_id]
        
        # Update temporal patterns
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        profile.typical_login_hours.add(timestamp.hour)
        profile.typical_login_days.add(timestamp.weekday())
        
        # Update access patterns
        if event.get('endpoint'):
            profile.typical_endpoints.add(event['endpoint'])
        if event.get('user_agent'):
            profile.typical_user_agents.add(event['user_agent'])
        if event.get('source_ip'):
            # Store IP ranges instead of exact IPs for privacy
            ip_range = '.'.join(event['source_ip'].split('.')[:3]) + '.0/24'
            profile.typical_ip_ranges.add(ip_range)
        
        # Update activity patterns (running average)
        if profile.samples_count > 0:
            alpha = 1.0 / min(profile.samples_count + 1, 100)  # Decay factor
            
            session_duration = event.get('session_duration', 0)
            profile.average_session_duration = (
                (1 - alpha) * profile.average_session_duration + 
                alpha * session_duration
            )
            
            request_rate = event.get('requests_per_minute', 0)
            profile.average_request_rate = (
                (1 - alpha) * profile.average_request_rate +
                alpha * request_rate
            )
        else:
            profile.average_session_duration = event.get('session_duration', 0)
            profile.average_request_rate = event.get('requests_per_minute', 0)
        
        # Update risk factors
        if event.get('failed_login'):
            profile.failed_login_rate = min(1.0, profile.failed_login_rate + 0.1)
        else:
            profile.failed_login_rate = max(0.0, profile.failed_login_rate - 0.01)
        
        if event.get('privilege_escalation_attempt'):
            profile.privilege_escalation_attempts += 1
        
        # Update learning metrics
        profile.samples_count += 1
        profile.last_updated = datetime.utcnow()
        profile.confidence_score = min(1.0, profile.samples_count / self.minimum_samples)
        
        return profile
    
    def detect_behavioral_anomalies(self, user_id: str, event: Dict[str, Any]) -> List[ThreatIndicator]:
        """Detect behavioral anomalies for a user"""
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        if profile.confidence_score < 0.3:  # Not enough learning
            return []
        
        indicators = []
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        
        # Temporal anomalies
        if timestamp.hour not in profile.typical_login_hours and len(profile.typical_login_hours) > 3:
            indicators.append(ThreatIndicator(
                indicator_id=str(uuid4()),
                category=ThreatCategory.INSIDER_THREAT,
                severity=ThreatSeverity.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
                score=60,
                description="Unusual login time detected",
                evidence={"unusual_hour": timestamp.hour, "typical_hours": list(profile.typical_login_hours)},
                created_at=datetime.utcnow(),
                user_id=user_id,
                model_name="behavioral_temporal_analyzer"
            ))
        
        # Geographical anomalies
        if event.get('source_ip'):
            ip_range = '.'.join(event['source_ip'].split('.')[:3]) + '.0/24'
            if (ip_range not in profile.typical_ip_ranges and 
                len(profile.typical_ip_ranges) > 2):
                indicators.append(ThreatIndicator(
                    indicator_id=str(uuid4()),
                    category=ThreatCategory.ACCOUNT_TAKEOVER,
                    severity=ThreatSeverity.HIGH,
                    confidence=ConfidenceLevel.HIGH,
                    score=75,
                    description="Login from unusual IP range",
                    evidence={"unusual_ip_range": ip_range, "typical_ranges": list(profile.typical_ip_ranges)},
                    created_at=datetime.utcnow(),
                    user_id=user_id,
                    source_ip=event['source_ip'],
                    model_name="behavioral_geo_analyzer"
                ))
        
        # Access pattern anomalies
        if event.get('endpoint'):
            if (event['endpoint'] not in profile.typical_endpoints and 
                len(profile.typical_endpoints) > 10):
                indicators.append(ThreatIndicator(
                    indicator_id=str(uuid4()),
                    category=ThreatCategory.LATERAL_MOVEMENT,
                    severity=ThreatSeverity.MEDIUM,
                    confidence=ConfidenceLevel.MEDIUM,
                    score=55,
                    description="Access to unusual endpoint",
                    evidence={"unusual_endpoint": event['endpoint']},
                    created_at=datetime.utcnow(),
                    user_id=user_id,
                    affected_resources=[event['endpoint']],
                    model_name="behavioral_access_analyzer"
                ))
        
        # Activity pattern anomalies
        request_rate = event.get('requests_per_minute', 0)
        if (profile.average_request_rate > 0 and 
            request_rate > profile.average_request_rate * 5):
            indicators.append(ThreatIndicator(
                indicator_id=str(uuid4()),
                category=ThreatCategory.MALICIOUS_AUTOMATION,
                severity=ThreatSeverity.HIGH,
                confidence=ConfidenceLevel.HIGH,
                score=80,
                description="Unusual request rate spike detected",
                evidence={
                    "current_rate": request_rate,
                    "typical_rate": profile.average_request_rate
                },
                created_at=datetime.utcnow(),
                user_id=user_id,
                model_name="behavioral_activity_analyzer"
            ))
        
        return indicators


class ThreatCorrelationEngine:
    """Correlates multiple threat indicators into threat events"""
    
    def __init__(self):
        self.correlation_window_minutes = 15
        self.active_events: Dict[str, ThreatEvent] = {}
        self.correlation_rules = self._load_correlation_rules()
        
    def _load_correlation_rules(self) -> List[Dict[str, Any]]:
        """Load threat correlation rules"""
        return [
            {
                "name": "Coordinated Attack",
                "indicators": [ThreatCategory.BRUTE_FORCE, ThreatCategory.LATERAL_MOVEMENT],
                "time_window_minutes": 30,
                "score_multiplier": 1.5,
                "category": ThreatCategory.ADVANCED_PERSISTENT_THREAT
            },
            {
                "name": "Account Compromise Chain",
                "indicators": [ThreatCategory.ACCOUNT_TAKEOVER, ThreatCategory.PRIVILEGE_ESCALATION],
                "time_window_minutes": 60,
                "score_multiplier": 2.0,
                "category": ThreatCategory.INSIDER_THREAT
            },
            {
                "name": "Data Theft Pattern",
                "indicators": [ThreatCategory.LATERAL_MOVEMENT, ThreatCategory.DATA_EXFILTRATION],
                "time_window_minutes": 120,
                "score_multiplier": 2.5,
                "category": ThreatCategory.DATA_EXFILTRATION
            }
        ]
    
    def correlate_indicators(self, indicators: List[ThreatIndicator]) -> List[ThreatEvent]:
        """Correlate threat indicators into events"""
        events = []
        current_time = datetime.utcnow()
        
        # Group indicators by attribution
        attribution_groups = defaultdict(list)
        for indicator in indicators:
            # Group by source IP, user ID, or tenant ID
            key = (indicator.source_ip, indicator.user_id, indicator.tenant_id)
            attribution_groups[key].append(indicator)
        
        # Process each attribution group
        for attribution, group_indicators in attribution_groups.items():
            # Check for existing events to update
            existing_event = self._find_existing_event(attribution, group_indicators)
            
            if existing_event:
                # Update existing event
                self._update_event(existing_event, group_indicators)
                events.append(existing_event)
            else:
                # Create new event
                new_event = self._create_new_event(attribution, group_indicators)
                if new_event:
                    self.active_events[new_event.event_id] = new_event
                    events.append(new_event)
        
        # Check correlation rules
        for event in events:
            self._apply_correlation_rules(event)
        
        # Cleanup old events
        self._cleanup_old_events(current_time)
        
        return events
    
    def _find_existing_event(self, attribution: Tuple, indicators: List[ThreatIndicator]) -> Optional[ThreatEvent]:
        """Find existing event that matches attribution and timing"""
        source_ip, user_id, tenant_id = attribution
        current_time = datetime.utcnow()
        
        for event in self.active_events.values():
            # Check attribution match
            if (source_ip and source_ip in event.source_ips) or \
               (user_id and user_id in event.user_ids) or \
               (tenant_id and tenant_id in event.tenant_ids):
                
                # Check timing window
                time_diff = (current_time - event.last_seen).total_seconds() / 60
                if time_diff <= self.correlation_window_minutes:
                    return event
        
        return None
    
    def _create_new_event(self, attribution: Tuple, indicators: List[ThreatIndicator]) -> Optional[ThreatEvent]:
        """Create new threat event from indicators"""
        if not indicators:
            return None
        
        source_ip, user_id, tenant_id = attribution
        
        # Calculate overall score and severity
        max_score = max(indicator.score for indicator in indicators)
        avg_score = sum(indicator.score for indicator in indicators) / len(indicators)
        overall_score = (max_score + avg_score) / 2
        
        # Determine primary category
        category_counts = defaultdict(int)
        for indicator in indicators:
            category_counts[indicator.category] += 1
        primary_category = max(category_counts, key=category_counts.get)
        
        # Calculate confidence
        confidences = [indicator.confidence.value for indicator in indicators]
        avg_confidence = self._calculate_average_confidence(confidences)
        
        event = ThreatEvent(
            event_id=str(uuid4()),
            category=primary_category,
            severity=self._score_to_severity(overall_score),
            overall_score=overall_score,
            confidence=avg_confidence,
            title=f"{primary_category.value.replace('_', ' ').title()} Activity Detected",
            description=f"Multiple threat indicators detected: {', '.join([i.description for i in indicators[:3]])}",
            indicators=indicators.copy()
        )
        
        # Set attribution
        if source_ip:
            event.source_ips.add(source_ip)
        if user_id:
            event.user_ids.add(user_id)
        if tenant_id:
            event.tenant_ids.add(tenant_id)
        
        # Collect affected resources
        for indicator in indicators:
            event.affected_resources.update(indicator.affected_resources)
        
        return event
    
    def _update_event(self, event: ThreatEvent, new_indicators: List[ThreatIndicator]):
        """Update existing event with new indicators"""
        event.indicators.extend(new_indicators)
        event.last_seen = datetime.utcnow()
        event.event_count += len(new_indicators)
        
        # Recalculate score
        all_scores = [indicator.score for indicator in event.indicators]
        event.overall_score = max(all_scores)
        event.severity = self._score_to_severity(event.overall_score)
        
        # Update attribution
        for indicator in new_indicators:
            if indicator.source_ip:
                event.source_ips.add(indicator.source_ip)
            if indicator.user_id:
                event.user_ids.add(indicator.user_id)
            if indicator.tenant_id:
                event.tenant_ids.add(indicator.tenant_id)
            
            event.affected_resources.update(indicator.affected_resources)
    
    def _apply_correlation_rules(self, event: ThreatEvent):
        """Apply correlation rules to enhance event scoring"""
        indicator_categories = set(indicator.category for indicator in event.indicators)
        
        for rule in self.correlation_rules:
            required_categories = set(rule["indicators"])
            if required_categories.issubset(indicator_categories):
                # Rule matches - enhance the event
                event.overall_score *= rule["score_multiplier"]
                event.overall_score = min(100, event.overall_score)  # Cap at 100
                event.severity = self._score_to_severity(event.overall_score)
                event.category = rule["category"]
                event.attack_chain.append(rule["name"])
                
                logger.info("Correlation rule applied",
                           rule=rule["name"],
                           event_id=event.event_id,
                           new_score=event.overall_score)
    
    def _cleanup_old_events(self, current_time: datetime):
        """Remove old events outside correlation window"""
        cutoff_time = current_time - timedelta(hours=24)  # Keep events for 24 hours
        
        expired_events = [
            event_id for event_id, event in self.active_events.items()
            if event.last_seen < cutoff_time
        ]
        
        for event_id in expired_events:
            del self.active_events[event_id]
    
    def _calculate_average_confidence(self, confidences: List[str]) -> ConfidenceLevel:
        """Calculate average confidence level"""
        confidence_values = {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "very_high": 0.9,
            "certain": 0.98
        }
        
        avg_value = sum(confidence_values.get(c, 0.5) for c in confidences) / len(confidences)
        
        if avg_value >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif avg_value >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif avg_value >= 0.70:
            return ConfidenceLevel.HIGH
        elif avg_value >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif avg_value >= 0.30:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _score_to_severity(self, score: float) -> ThreatSeverity:
        """Convert numerical score to severity enum"""
        if score >= 96:
            return ThreatSeverity.EMERGENCY
        elif score >= 81:
            return ThreatSeverity.CRITICAL
        elif score >= 61:
            return ThreatSeverity.HIGH
        elif score >= 41:
            return ThreatSeverity.MEDIUM
        elif score >= 26:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO


class AdvancedThreatDetectionEngine:
    """
    Main threat detection engine that orchestrates all detection components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.ml_detector = MLAnomalyDetector()
        self.pattern_recognizer = AttackPatternRecognizer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.correlation_engine = ThreatCorrelationEngine()
        
        # Event processing
        self.event_buffer = deque(maxlen=10000)
        self.processing_enabled = True
        self.last_training_time = None
        
        # Metrics
        self.processed_events = 0
        self.detected_threats = 0
        self.false_positives = 0
        
        logger.info("Advanced threat detection engine initialized")
    
    async def process_security_event(self, event: Dict[str, Any]) -> List[ThreatEvent]:
        """Process a single security event and return threat events"""
        if not self.processing_enabled:
            return []
        
        try:
            # Add to buffer for training
            self.event_buffer.append(event)
            self.processed_events += 1
            
            # Collect all indicators
            all_indicators = []
            
            # 1. ML-based anomaly detection
            if self.ml_detector.is_trained:
                is_anomaly, probability, feature_weights = self.ml_detector.predict_anomaly(event)
                if is_anomaly and probability > 0.6:
                    indicator = ThreatIndicator(
                        indicator_id=str(uuid4()),
                        category=ThreatCategory.ADVANCED_PERSISTENT_THREAT,
                        severity=self._probability_to_severity(probability),
                        confidence=self._probability_to_confidence(probability),
                        score=probability * 100,
                        description=f"ML anomaly detected (probability: {probability:.2f})",
                        evidence={"probability": probability, "feature_weights": feature_weights},
                        created_at=datetime.utcnow(),
                        source_ip=event.get('source_ip'),
                        user_id=event.get('user_id'),
                        tenant_id=event.get('tenant_id'),
                        model_name="ml_anomaly_detector",
                        anomaly_score=probability,
                        feature_weights=feature_weights
                    )
                    all_indicators.append(indicator)
            
            # 2. Pattern recognition
            pattern_indicators = self.pattern_recognizer.recognize_pattern([event])
            all_indicators.extend(pattern_indicators)
            
            # 3. Behavioral analysis
            user_id = event.get('user_id')
            if user_id:
                # Update profile
                self.behavioral_analyzer.update_profile(user_id, event)
                
                # Detect behavioral anomalies
                behavioral_indicators = self.behavioral_analyzer.detect_behavioral_anomalies(user_id, event)
                all_indicators.extend(behavioral_indicators)
            
            # 4. Correlate indicators into events
            threat_events = []
            if all_indicators:
                threat_events = self.correlation_engine.correlate_indicators(all_indicators)
                self.detected_threats += len(threat_events)
            
            # 5. Retrain models periodically
            await self._periodic_training()
            
            return threat_events
        
        except Exception as e:
            logger.error("Failed to process security event", error=str(e), event_id=event.get('event_id'))
            return []
    
    async def process_security_events_batch(self, events: List[Dict[str, Any]]) -> List[ThreatEvent]:
        """Process multiple security events in batch"""
        all_threat_events = []
        
        # Process events individually first
        for event in events:
            threat_events = await self.process_security_event(event)
            all_threat_events.extend(threat_events)
        
        # Additional pattern recognition on batch
        batch_patterns = self.pattern_recognizer.recognize_pattern(events)
        if batch_patterns:
            batch_threat_events = self.correlation_engine.correlate_indicators(batch_patterns)
            all_threat_events.extend(batch_threat_events)
        
        return all_threat_events
    
    async def train_models(self, training_events: List[Dict[str, Any]]) -> bool:
        """Train ML models with historical data"""
        try:
            # Train anomaly detector
            if len(training_events) >= 100:
                success = self.ml_detector.train(training_events)
                if success:
                    self.last_training_time = datetime.utcnow()
                    logger.info("ML models trained successfully",
                               training_samples=len(training_events))
                return success
            else:
                logger.warning("Insufficient training data", sample_count=len(training_events))
                return False
        
        except Exception as e:
            logger.error("Failed to train models", error=str(e))
            return False
    
    async def _periodic_training(self):
        """Perform periodic model retraining"""
        if not self.last_training_time:
            return
        
        # Retrain every 24 hours
        if (datetime.utcnow() - self.last_training_time).total_seconds() > 86400:
            if len(self.event_buffer) >= 100:
                await self.train_models(list(self.event_buffer))
    
    def _probability_to_severity(self, probability: float) -> ThreatSeverity:
        """Convert probability to severity"""
        if probability >= 0.95:
            return ThreatSeverity.EMERGENCY
        elif probability >= 0.85:
            return ThreatSeverity.CRITICAL
        elif probability >= 0.75:
            return ThreatSeverity.HIGH
        elif probability >= 0.60:
            return ThreatSeverity.MEDIUM
        elif probability >= 0.40:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO
    
    def _probability_to_confidence(self, probability: float) -> ConfidenceLevel:
        """Convert probability to confidence"""
        if probability >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif probability >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif probability >= 0.70:
            return ConfidenceLevel.HIGH
        elif probability >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif probability >= 0.30:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "processed_events": self.processed_events,
            "detected_threats": self.detected_threats,
            "false_positives": self.false_positives,
            "accuracy": 1.0 - (self.false_positives / max(1, self.detected_threats)),
            "ml_model_trained": self.ml_detector.is_trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "active_threat_events": len(self.correlation_engine.active_events),
            "user_profiles": len(self.behavioral_analyzer.user_profiles),
            "buffer_size": len(self.event_buffer)
        }
    
    async def shutdown(self):
        """Shutdown the threat detection engine"""
        self.processing_enabled = False
        logger.info("Advanced threat detection engine shutdown")


# Global instance
_threat_detection_engine: Optional[AdvancedThreatDetectionEngine] = None


def get_threat_detection_engine() -> AdvancedThreatDetectionEngine:
    """Get global threat detection engine instance"""
    global _threat_detection_engine
    if _threat_detection_engine is None:
        _threat_detection_engine = AdvancedThreatDetectionEngine()
    return _threat_detection_engine


async def initialize_threat_detection_engine(config: Optional[Dict[str, Any]] = None) -> AdvancedThreatDetectionEngine:
    """Initialize global threat detection engine"""
    global _threat_detection_engine
    _threat_detection_engine = AdvancedThreatDetectionEngine(config)
    return _threat_detection_engine