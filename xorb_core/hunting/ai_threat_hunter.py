"""
Advanced AI-Powered Threat Hunting Engine

This module provides comprehensive AI-driven threat hunting capabilities including
behavioral analysis, anomaly detection, pattern recognition, hypothesis generation,
and automated threat hunting workflows for the XORB ecosystem.
"""

import asyncio
import json
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from collections import defaultdict, deque
import re

import structlog
from prometheus_client import Counter, Gauge, Histogram

# Import related XORB modules
from xorb_core.intelligence.threat_intelligence_engine import (
    IoC, IoC_Type, ThreatLevel, threat_intel_engine
)
from xorb_core.vulnerabilities import (
    Vulnerability, VulnerabilitySeverity, vulnerability_manager
)

# Metrics
HUNT_SESSIONS_TOTAL = Counter('xorb_hunt_sessions_total', 'Hunt sessions started', ['hunt_type', 'trigger'])
THREATS_DETECTED = Counter('xorb_threats_detected_total', 'Threats detected by hunting', ['threat_type', 'confidence'])
ANOMALIES_DETECTED = Counter('xorb_anomalies_detected_total', 'Anomalies detected', ['anomaly_type'])
HUNT_DURATION = Histogram('xorb_hunt_duration_seconds', 'Hunt session duration')
ACTIVE_HUNTS = Gauge('xorb_active_hunts', 'Active hunt sessions')

logger = structlog.get_logger(__name__)


class HuntTrigger(Enum):
    """Hunt session trigger types."""
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    IOC_DETECTED = "ioc_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    VULNERABILITY_DISCOVERED = "vulnerability_discovered"
    INTELLIGENCE_UPDATE = "intelligence_update"
    ESCALATION = "escalation"


class ThreatType(Enum):
    """Detected threat types."""
    APT = "apt"
    MALWARE = "malware"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    COMMAND_CONTROL = "command_control"
    RECONNAISSANCE = "reconnaissance"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class ConfidenceLevel(Enum):
    """Confidence levels for threat detection."""
    VERY_LOW = "very_low"      # 0.0-0.2
    LOW = "low"                # 0.2-0.4
    MEDIUM = "medium"          # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # 0.8-1.0


class AnomalyType(Enum):
    """Types of anomalies detected."""
    BEHAVIORAL = "behavioral"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    NETWORK = "network"
    ACCESS_PATTERN = "access_pattern"
    DATA_FLOW = "data_flow"
    AUTHENTICATION = "authentication"
    PRIVILEGE_USAGE = "privilege_usage"


@dataclass
class DataPoint:
    """Data point for analysis."""
    timestamp: float
    source: str
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    anomaly_type: AnomalyType = AnomalyType.BEHAVIORAL
    severity: float = 0.5  # 0.0-1.0
    confidence: float = 0.5  # 0.0-1.0
    
    # Description
    title: str = ""
    description: str = ""
    
    # Source data
    source_events: List[DataPoint] = field(default_factory=list)
    baseline_data: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis
    features: Dict[str, float] = field(default_factory=dict)
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Context
    affected_entities: List[str] = field(default_factory=list)
    related_iocs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThreatHypothesis:
    """AI-generated threat hypothesis."""
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Hypothesis details
    threat_type: ThreatType = ThreatType.SUSPICIOUS_ACTIVITY
    title: str = ""
    description: str = ""
    confidence: float = 0.5
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    iocs: List[IoC] = field(default_factory=list)
    
    # Investigation
    investigation_queries: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Scoring
    risk_score: float = 0.0
    impact_score: float = 0.0
    likelihood_score: float = 0.0
    
    def calculate_risk_score(self):
        """Calculate overall risk score."""
        self.risk_score = (self.impact_score * self.likelihood_score * self.confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HuntSession:
    """Active threat hunting session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Session details
    trigger: HuntTrigger = HuntTrigger.MANUAL
    hunter_name: str = "ai_hunter"
    target_entities: List[str] = field(default_factory=list)
    
    # Hunt parameters
    time_window_hours: int = 24
    focus_areas: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    
    # Results
    anomalies_found: List[Anomaly] = field(default_factory=list)
    hypotheses_generated: List[ThreatHypothesis] = field(default_factory=list)
    iocs_discovered: List[IoC] = field(default_factory=list)
    
    # Statistics
    data_points_analyzed: int = 0
    processing_time: float = 0.0
    
    def get_duration(self) -> float:
        """Get session duration."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IAnomalyDetector(ABC):
    """Interface for anomaly detection algorithms."""
    
    @abstractmethod
    async def detect_anomalies(self, data_points: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies in data points."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get detector name."""
        pass


class StatisticalAnomalyDetector(IAnomalyDetector):
    """Statistical anomaly detector using Z-score and IQR methods."""
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
    
    async def detect_anomalies(self, data_points: List[DataPoint]) -> List[Anomaly]:
        """Detect statistical anomalies."""
        anomalies = []
        
        # Group data points by event type
        grouped_data = defaultdict(list)
        for dp in data_points:
            grouped_data[dp.event_type].append(dp)
        
        for event_type, points in grouped_data.items():
            if len(points) < 10:  # Need sufficient data
                continue
            
            # Extract numeric features
            numeric_features = self._extract_numeric_features(points)
            
            for feature_name, values in numeric_features.items():
                if len(values) < 10:
                    continue
                
                # Z-score analysis
                z_anomalies = self._detect_zscore_anomalies(
                    values, points, feature_name, event_type
                )
                anomalies.extend(z_anomalies)
                
                # IQR analysis
                iqr_anomalies = self._detect_iqr_anomalies(
                    values, points, feature_name, event_type
                )
                anomalies.extend(iqr_anomalies)
        
        return anomalies
    
    def _extract_numeric_features(self, data_points: List[DataPoint]) -> Dict[str, List[float]]:
        """Extract numeric features from data points."""
        features = defaultdict(list)
        
        for dp in data_points:
            # Extract common numeric features
            if 'size' in dp.data:
                features['size'].append(float(dp.data['size']))
            if 'duration' in dp.data:
                features['duration'].append(float(dp.data['duration']))
            if 'count' in dp.data:
                features['count'].append(float(dp.data['count']))
            if 'response_time' in dp.data:
                features['response_time'].append(float(dp.data['response_time']))
        
        return dict(features)
    
    def _detect_zscore_anomalies(self, values: List[float], points: List[DataPoint], 
                                feature_name: str, event_type: str) -> List[Anomaly]:
        """Detect anomalies using Z-score method."""
        anomalies = []
        
        if len(values) < 3:
            return anomalies
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return anomalies
        
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_dev
            
            if z_score > self.z_threshold:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    title=f"Statistical Anomaly in {feature_name}",
                    description=f"Z-score of {z_score:.2f} exceeds threshold {self.z_threshold}",
                    severity=min(z_score / 10.0, 1.0),
                    confidence=min(z_score / 5.0, 1.0),
                    source_events=[points[i]],
                    features={feature_name: value, 'z_score': z_score},
                    statistical_metrics={
                        'mean': mean_val,
                        'std_dev': std_dev,
                        'z_score': z_score
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(self, values: List[float], points: List[DataPoint],
                             feature_name: str, event_type: str) -> List[Anomaly]:
        """Detect anomalies using IQR method."""
        anomalies = []
        
        if len(values) < 4:
            return anomalies
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        if iqr == 0:
            return anomalies
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                distance = min(abs(value - lower_bound), abs(value - upper_bound))
                severity = min(distance / iqr, 1.0)
                
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    title=f"IQR Outlier in {feature_name}",
                    description=f"Value {value} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    severity=severity,
                    confidence=0.7,
                    source_events=[points[i]],
                    features={feature_name: value},
                    statistical_metrics={
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def get_detector_name(self) -> str:
        return "statistical_detector"


class BehavioralAnomalyDetector(IAnomalyDetector):
    """Behavioral anomaly detector using pattern analysis."""
    
    def __init__(self, pattern_threshold: float = 0.3):
        self.pattern_threshold = pattern_threshold
        self.user_baselines = {}
        self.system_baselines = {}
    
    async def detect_anomalies(self, data_points: List[DataPoint]) -> List[Anomaly]:
        """Detect behavioral anomalies."""
        anomalies = []
        
        # Group by user/entity
        user_activities = defaultdict(list)
        for dp in data_points:
            user_id = dp.data.get('user_id') or dp.data.get('source_ip', 'unknown')
            user_activities[user_id].append(dp)
        
        for user_id, activities in user_activities.items():
            user_anomalies = await self._analyze_user_behavior(user_id, activities)
            anomalies.extend(user_anomalies)
        
        return anomalies
    
    async def _analyze_user_behavior(self, user_id: str, activities: List[DataPoint]) -> List[Anomaly]:
        """Analyze user behavior patterns."""
        anomalies = []
        
        # Time-based analysis
        time_anomalies = self._detect_time_anomalies(user_id, activities)
        anomalies.extend(time_anomalies)
        
        # Access pattern analysis
        access_anomalies = self._detect_access_anomalies(user_id, activities)
        anomalies.extend(access_anomalies)
        
        # Volume analysis
        volume_anomalies = self._detect_volume_anomalies(user_id, activities)
        anomalies.extend(volume_anomalies)
        
        return anomalies
    
    def _detect_time_anomalies(self, user_id: str, activities: List[DataPoint]) -> List[Anomaly]:
        """Detect unusual time-based activities."""
        anomalies = []
        
        # Extract hour of day for each activity
        hours = []
        for activity in activities:
            dt = datetime.fromtimestamp(activity.timestamp)
            hours.append(dt.hour)
        
        if len(hours) < 5:
            return anomalies
        
        # Simple heuristic: activities outside normal business hours (9-17)
        after_hours = [h for h in hours if h < 9 or h > 17]
        
        if len(after_hours) > len(hours) * 0.3:  # More than 30% after hours
            anomaly = Anomaly(
                anomaly_type=AnomalyType.TEMPORAL,
                title="Unusual Activity Hours",
                description=f"User {user_id} active outside normal hours ({len(after_hours)}/{len(hours)} activities)",
                severity=0.6,
                confidence=0.7,
                source_events=activities,
                affected_entities=[user_id],
                features={'after_hours_ratio': len(after_hours) / len(hours)}
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_access_anomalies(self, user_id: str, activities: List[DataPoint]) -> List[Anomaly]:
        """Detect unusual access patterns."""
        anomalies = []
        
        # Extract accessed resources
        resources = set()
        for activity in activities:
            if 'resource' in activity.data:
                resources.add(activity.data['resource'])
            if 'url' in activity.data:
                resources.add(activity.data['url'])
        
        # Simple heuristic: accessing too many different resources
        if len(resources) > 20:  # Threshold for suspicious access
            anomaly = Anomaly(
                anomaly_type=AnomalyType.ACCESS_PATTERN,
                title="Excessive Resource Access",
                description=f"User {user_id} accessed {len(resources)} different resources",
                severity=0.5,
                confidence=0.6,
                source_events=activities,
                affected_entities=[user_id],
                features={'unique_resources': len(resources)}
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_volume_anomalies(self, user_id: str, activities: List[DataPoint]) -> List[Anomaly]:
        """Detect unusual activity volumes."""
        anomalies = []
        
        # Calculate activity rate
        if len(activities) < 2:
            return anomalies
        
        time_span = max(a.timestamp for a in activities) - min(a.timestamp for a in activities)
        if time_span == 0:
            return anomalies
        
        activity_rate = len(activities) / time_span  # activities per second
        
        # Heuristic: very high activity rate
        if activity_rate > 0.1:  # More than 6 activities per minute
            anomaly = Anomaly(
                anomaly_type=AnomalyType.BEHAVIORAL,
                title="High Activity Rate",
                description=f"User {user_id} has unusually high activity rate: {activity_rate:.3f}/sec",
                severity=0.7,
                confidence=0.8,
                source_events=activities,
                affected_entities=[user_id],
                features={'activity_rate': activity_rate}
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def get_detector_name(self) -> str:
        return "behavioral_detector"


class AIHypothesisGenerator:
    """AI-powered threat hypothesis generator."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.mitre_tactics = self._load_mitre_tactics()
    
    def _load_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load threat patterns for hypothesis generation."""
        return {
            "apt_pattern": {
                "threat_type": ThreatType.APT,
                "indicators": ["lateral_movement", "persistence", "data_staging"],
                "techniques": ["T1021", "T1053", "T1002"],
                "typical_sequence": ["reconnaissance", "initial_access", "persistence", "lateral_movement", "exfiltration"]
            },
            "insider_threat": {
                "threat_type": ThreatType.INSIDER_THREAT,
                "indicators": ["after_hours_access", "excessive_downloads", "privilege_escalation"],
                "techniques": ["T1078", "T1005", "T1083"],
                "typical_sequence": ["privilege_abuse", "data_collection", "exfiltration"]
            },
            "malware_infection": {
                "threat_type": ThreatType.MALWARE,
                "indicators": ["command_control", "process_injection", "file_modification"],
                "techniques": ["T1055", "T1105", "T1082"],
                "typical_sequence": ["initial_infection", "persistence", "command_control", "impact"]
            }
        }
    
    def _load_mitre_tactics(self) -> Dict[str, List[str]]:
        """Load MITRE ATT&CK tactics."""
        return {
            "TA0001": ["Initial Access"],
            "TA0002": ["Execution"],
            "TA0003": ["Persistence"],
            "TA0004": ["Privilege Escalation"],
            "TA0005": ["Defense Evasion"],
            "TA0006": ["Credential Access"],
            "TA0007": ["Discovery"],
            "TA0008": ["Lateral Movement"],
            "TA0009": ["Collection"],
            "TA0010": ["Exfiltration"],
            "TA0011": ["Command and Control"],
            "TA0040": ["Impact"]
        }
    
    async def generate_hypotheses(self, anomalies: List[Anomaly], 
                                 iocs: List[IoC], 
                                 context: Dict[str, Any] = None) -> List[ThreatHypothesis]:
        """Generate threat hypotheses from anomalies and IoCs."""
        hypotheses = []
        
        # Pattern-based hypothesis generation
        pattern_hypotheses = await self._generate_pattern_hypotheses(anomalies, iocs)
        hypotheses.extend(pattern_hypotheses)
        
        # Correlation-based hypothesis generation
        correlation_hypotheses = await self._generate_correlation_hypotheses(anomalies, iocs)
        hypotheses.extend(correlation_hypotheses)
        
        # Timeline-based hypothesis generation
        timeline_hypotheses = await self._generate_timeline_hypotheses(anomalies, iocs)
        hypotheses.extend(timeline_hypotheses)
        
        # Score and rank hypotheses
        for hypothesis in hypotheses:
            await self._score_hypothesis(hypothesis, anomalies, iocs)
        
        # Sort by risk score
        hypotheses.sort(key=lambda h: h.risk_score, reverse=True)
        
        return hypotheses
    
    async def _generate_pattern_hypotheses(self, anomalies: List[Anomaly], 
                                         iocs: List[IoC]) -> List[ThreatHypothesis]:
        """Generate hypotheses based on known threat patterns."""
        hypotheses = []
        
        # Analyze anomaly patterns
        anomaly_types = set(a.anomaly_type for a in anomalies)
        high_severity_anomalies = [a for a in anomalies if a.severity > 0.6]
        
        # Check for APT patterns
        if (AnomalyType.BEHAVIORAL in anomaly_types and 
            AnomalyType.ACCESS_PATTERN in anomaly_types and
            len(high_severity_anomalies) >= 2):
            
            hypothesis = ThreatHypothesis(
                threat_type=ThreatType.APT,
                title="Potential Advanced Persistent Threat Activity",
                description="Multiple behavioral and access pattern anomalies suggest coordinated threat actor",
                confidence=0.7,
                anomalies=high_severity_anomalies,
                iocs=iocs,
                supporting_evidence=[
                    f"Detected {len(anomaly_types)} different anomaly types",
                    f"Found {len(high_severity_anomalies)} high-severity anomalies",
                    f"Behavioral patterns consistent with APT tactics"
                ],
                investigation_queries=[
                    "Look for lateral movement indicators",
                    "Check for persistence mechanisms",
                    "Analyze command and control communications"
                ],
                recommended_actions=[
                    "Isolate affected systems",
                    "Preserve forensic evidence",
                    "Initiate incident response procedures"
                ]
            )
            hypotheses.append(hypothesis)
        
        # Check for insider threat patterns
        insider_indicators = 0
        for anomaly in anomalies:
            if anomaly.anomaly_type == AnomalyType.TEMPORAL:
                insider_indicators += 1
            if "excessive" in anomaly.title.lower():
                insider_indicators += 1
            if "unusual" in anomaly.description.lower():
                insider_indicators += 1
        
        if insider_indicators >= 2:
            hypothesis = ThreatHypothesis(
                threat_type=ThreatType.INSIDER_THREAT,
                title="Potential Insider Threat Activity",
                description="Anomalous user behavior suggests possible insider threat",
                confidence=0.6,
                anomalies=[a for a in anomalies if a.anomaly_type in [AnomalyType.TEMPORAL, AnomalyType.ACCESS_PATTERN]],
                supporting_evidence=[
                    f"Found {insider_indicators} insider threat indicators",
                    "Unusual access patterns detected",
                    "After-hours activity anomalies"
                ],
                investigation_queries=[
                    "Review user access logs",
                    "Check data download volumes",
                    "Analyze privilege usage patterns"
                ],
                recommended_actions=[
                    "Review user privileges",
                    "Monitor data access closely",
                    "Consider HR consultation"
                ]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_correlation_hypotheses(self, anomalies: List[Anomaly], 
                                             iocs: List[IoC]) -> List[ThreatHypothesis]:
        """Generate hypotheses based on correlations."""
        hypotheses = []
        
        # Correlate anomalies with IoCs
        if anomalies and iocs:
            high_threat_iocs = [ioc for ioc in iocs if ioc.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
            
            if high_threat_iocs:
                hypothesis = ThreatHypothesis(
                    threat_type=ThreatType.MALWARE,
                    title="Malware Activity Correlation",
                    description="Behavioral anomalies correlate with known threat indicators",
                    confidence=0.8,
                    anomalies=anomalies,
                    iocs=high_threat_iocs,
                    supporting_evidence=[
                        f"Detected {len(high_threat_iocs)} high-threat IoCs",
                        f"Found {len(anomalies)} behavioral anomalies",
                        "Temporal correlation between anomalies and IoCs"
                    ],
                    investigation_queries=[
                        "Analyze malware signatures",
                        "Check for command and control communication",
                        "Look for file system modifications"
                    ],
                    recommended_actions=[
                        "Run antimalware scans",
                        "Block malicious IPs/domains",
                        "Quarantine affected systems"
                    ]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_timeline_hypotheses(self, anomalies: List[Anomaly], 
                                          iocs: List[IoC]) -> List[ThreatHypothesis]:
        """Generate hypotheses based on timeline analysis."""
        hypotheses = []
        
        if not anomalies:
            return hypotheses
        
        # Analyze temporal clustering
        timestamps = [a.timestamp for a in anomalies]
        time_span = max(timestamps) - min(timestamps)
        
        if time_span < 3600 and len(anomalies) >= 3:  # Multiple anomalies within 1 hour
            hypothesis = ThreatHypothesis(
                threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                title="Coordinated Attack Pattern",
                description="Multiple anomalies detected within short timeframe suggest coordinated attack",
                confidence=0.6,
                anomalies=anomalies,
                supporting_evidence=[
                    f"Found {len(anomalies)} anomalies within {time_span/60:.1f} minutes",
                    "Temporal clustering suggests coordinated activity",
                    "Multiple attack vectors identified"
                ],
                investigation_queries=[
                    "Analyze attack timeline",
                    "Look for common attack vectors",
                    "Check for automated attack tools"
                ],
                recommended_actions=[
                    "Implement emergency controls",
                    "Monitor for continued activity",
                    "Prepare incident response"
                ]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _score_hypothesis(self, hypothesis: ThreatHypothesis, 
                               anomalies: List[Anomaly], iocs: List[IoC]):
        """Score hypothesis based on various factors."""
        # Impact score (0-1)
        if hypothesis.threat_type in [ThreatType.APT, ThreatType.DATA_EXFILTRATION]:
            hypothesis.impact_score = 0.9
        elif hypothesis.threat_type in [ThreatType.MALWARE, ThreatType.INSIDER_THREAT]:
            hypothesis.impact_score = 0.7
        else:
            hypothesis.impact_score = 0.5
        
        # Likelihood score based on evidence strength
        evidence_strength = len(hypothesis.supporting_evidence) / 10.0
        anomaly_strength = sum(a.confidence * a.severity for a in hypothesis.anomalies) / max(len(hypothesis.anomalies), 1)
        ioc_strength = sum(1.0 if ioc.threat_level == ThreatLevel.CRITICAL else 0.7 if ioc.threat_level == ThreatLevel.HIGH else 0.5 
                          for ioc in hypothesis.iocs) / max(len(hypothesis.iocs), 1) if hypothesis.iocs else 0.3
        
        hypothesis.likelihood_score = min((evidence_strength + anomaly_strength + ioc_strength) / 3.0, 1.0)
        
        # Calculate overall risk score
        hypothesis.calculate_risk_score()


class AIThreatHunter:
    """Main AI-powered threat hunting engine."""
    
    def __init__(self):
        self.anomaly_detectors: List[IAnomalyDetector] = []
        self.hypothesis_generator = AIHypothesisGenerator()
        self.active_sessions: Dict[str, HuntSession] = {}
        self.historical_hunts: List[HuntSession] = []
        self.running = False
        
        # Data storage
        self.data_buffer = deque(maxlen=100000)  # Store recent data points
        
        # Initialize default detectors
        self.add_anomaly_detector(StatisticalAnomalyDetector())
        self.add_anomaly_detector(BehavioralAnomalyDetector())
        
        # Statistics
        self.stats = {
            "hunts_executed": 0,
            "threats_detected": 0,
            "anomalies_found": 0,
            "hypotheses_generated": 0
        }
    
    def add_anomaly_detector(self, detector: IAnomalyDetector):
        """Add an anomaly detector."""
        self.anomaly_detectors.append(detector)
        logger.info("Added anomaly detector", detector=detector.get_detector_name())
    
    async def start_threat_hunting(self):
        """Start the AI threat hunting system."""
        self.running = True
        
        # Start background tasks
        continuous_hunt_task = asyncio.create_task(self._continuous_hunting_loop())
        data_ingestion_task = asyncio.create_task(self._data_ingestion_loop())
        
        logger.info("AI threat hunting engine started")
        
        try:
            await asyncio.gather(continuous_hunt_task, data_ingestion_task)
        except asyncio.CancelledError:
            logger.info("AI threat hunting engine stopped")
    
    async def stop_threat_hunting(self):
        """Stop the threat hunting system."""
        self.running = False
        
        # Complete active sessions
        for session in self.active_sessions.values():
            await self._complete_hunt_session(session)
    
    async def _continuous_hunting_loop(self):
        """Continuous threat hunting loop."""
        while self.running:
            try:
                # Check for triggers
                await self._check_hunt_triggers()
                
                # Process active sessions
                await self._process_active_sessions()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error("Continuous hunting loop failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _data_ingestion_loop(self):
        """Data ingestion and preprocessing loop."""
        while self.running:
            try:
                # Simulate data ingestion (in real implementation, this would connect to SIEM, logs, etc.)
                await self._ingest_sample_data()
                await asyncio.sleep(60)  # Ingest data every minute
            except Exception as e:
                logger.error("Data ingestion loop failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _ingest_sample_data(self):
        """Ingest sample data for demonstration."""
        # Generate sample security events
        sample_events = [
            {
                "event_type": "login",
                "user_id": f"user_{random.randint(1, 50)}",
                "source_ip": f"192.168.1.{random.randint(1, 254)}",
                "success": random.choice([True, True, True, False]),  # Mostly successful
                "timestamp": time.time(),
                "duration": random.uniform(0.1, 5.0)
            },
            {
                "event_type": "file_access",
                "user_id": f"user_{random.randint(1, 50)}",
                "resource": f"/data/file_{random.randint(1, 1000)}.txt",
                "action": random.choice(["read", "write", "delete"]),
                "size": random.randint(100, 1000000),
                "timestamp": time.time()
            },
            {
                "event_type": "network_connection",
                "source_ip": f"192.168.1.{random.randint(1, 254)}",
                "dest_ip": f"10.0.0.{random.randint(1, 254)}",
                "port": random.choice([80, 443, 22, 3389, 1433]),
                "bytes_transferred": random.randint(100, 100000),
                "timestamp": time.time()
            }
        ]
        
        for event_data in sample_events:
            data_point = DataPoint(
                timestamp=event_data["timestamp"],
                source="sample_generator",
                event_type=event_data["event_type"],
                data=event_data
            )
            self.data_buffer.append(data_point)
    
    async def _check_hunt_triggers(self):
        """Check for conditions that should trigger new hunt sessions."""
        # Check for new high-severity vulnerabilities
        high_vulns = vulnerability_manager.get_vulnerabilities_by_severity(VulnerabilitySeverity.CRITICAL)
        recent_high_vulns = [v for v in high_vulns if v.discovered_at > time.time() - 3600]  # Last hour
        
        if recent_high_vulns:
            await self.start_hunt_session(
                trigger=HuntTrigger.VULNERABILITY_DISCOVERED,
                focus_areas=["vulnerability_exploitation", "lateral_movement"],
                time_window_hours=6
            )
        
        # Check for new high-threat IoCs
        threat_intel_stats = threat_intel_engine.get_cache_statistics()
        if threat_intel_stats["total_iocs"] > 0:
            # In a real implementation, we'd check for new high-threat IoCs
            # For demo, randomly trigger intel-based hunts
            if random.random() < 0.1:  # 10% chance every check
                await self.start_hunt_session(
                    trigger=HuntTrigger.INTELLIGENCE_UPDATE,
                    focus_areas=["ioc_activity", "command_control"],
                    time_window_hours=12
                )
    
    async def _process_active_sessions(self):
        """Process all active hunt sessions."""
        completed_sessions = []
        
        for session_id, session in self.active_sessions.items():
            try:
                # Check if session should be completed
                if session.get_duration() > session.time_window_hours * 3600:
                    completed_sessions.append(session_id)
                    continue
                
                # Continue processing session
                await self._process_hunt_session(session)
                
            except Exception as e:
                logger.error("Failed to process hunt session", 
                           session_id=session_id, error=str(e))
                completed_sessions.append(session_id)
        
        # Complete finished sessions
        for session_id in completed_sessions:
            session = self.active_sessions.pop(session_id, None)
            if session:
                await self._complete_hunt_session(session)
    
    @HUNT_DURATION.time()
    async def start_hunt_session(self, trigger: HuntTrigger = HuntTrigger.MANUAL,
                                hunter_name: str = "ai_hunter",
                                target_entities: List[str] = None,
                                time_window_hours: int = 24,
                                focus_areas: List[str] = None,
                                techniques: List[str] = None) -> str:
        """Start a new threat hunting session."""
        session = HuntSession(
            trigger=trigger,
            hunter_name=hunter_name,
            target_entities=target_entities or [],
            time_window_hours=time_window_hours,
            focus_areas=focus_areas or [],
            techniques=techniques or []
        )
        
        self.active_sessions[session.session_id] = session
        self.stats["hunts_executed"] += 1
        
        # Record metrics
        HUNT_SESSIONS_TOTAL.labels(
            hunt_type="ai_automated",
            trigger=trigger.value
        ).inc()
        ACTIVE_HUNTS.set(len(self.active_sessions))
        
        logger.info("Started threat hunting session",
                   session_id=session.session_id,
                   trigger=trigger.value,
                   time_window=time_window_hours)
        
        # Start initial processing
        await self._process_hunt_session(session)
        
        return session.session_id
    
    async def _process_hunt_session(self, session: HuntSession):
        """Process a hunt session."""
        start_time = time.time()
        
        # Get relevant data for the time window
        cutoff_time = session.start_time - (session.time_window_hours * 3600)
        relevant_data = [
            dp for dp in self.data_buffer 
            if dp.timestamp >= cutoff_time
        ]
        
        session.data_points_analyzed = len(relevant_data)
        
        if not relevant_data:
            return
        
        # Run anomaly detection
        all_anomalies = []
        for detector in self.anomaly_detectors:
            try:
                anomalies = await detector.detect_anomalies(relevant_data)
                all_anomalies.extend(anomalies)
                
                # Record metrics
                for anomaly in anomalies:
                    ANOMALIES_DETECTED.labels(
                        anomaly_type=anomaly.anomaly_type.value
                    ).inc()
                
            except Exception as e:
                logger.error("Anomaly detector failed",
                           detector=detector.get_detector_name(),
                           error=str(e))
        
        session.anomalies_found.extend(all_anomalies)
        self.stats["anomalies_found"] += len(all_anomalies)
        
        # Get relevant IoCs from threat intelligence
        relevant_iocs = []
        for ioc in threat_intel_engine.ioc_cache.values():
            if ioc.last_seen >= cutoff_time:
                relevant_iocs.append(ioc)
        
        session.iocs_discovered.extend(relevant_iocs)
        
        # Generate threat hypotheses
        if all_anomalies or relevant_iocs:
            hypotheses = await self.hypothesis_generator.generate_hypotheses(
                all_anomalies, relevant_iocs
            )
            session.hypotheses_generated.extend(hypotheses)
            self.stats["hypotheses_generated"] += len(hypotheses)
            
            # Record threat detections
            for hypothesis in hypotheses:
                if hypothesis.confidence > 0.6:
                    self.stats["threats_detected"] += 1
                    THREATS_DETECTED.labels(
                        threat_type=hypothesis.threat_type.value,
                        confidence="high" if hypothesis.confidence > 0.8 else "medium"
                    ).inc()
        
        session.processing_time += time.time() - start_time
        
        logger.debug("Processed hunt session",
                    session_id=session.session_id,
                    anomalies=len(all_anomalies),
                    hypotheses=len(session.hypotheses_generated))
    
    async def _complete_hunt_session(self, session: HuntSession):
        """Complete a hunt session."""
        session.end_time = time.time()
        self.historical_hunts.append(session)
        
        # Update metrics
        ACTIVE_HUNTS.set(len(self.active_sessions))
        HUNT_DURATION.observe(session.get_duration())
        
        logger.info("Completed hunt session",
                   session_id=session.session_id,
                   duration=session.get_duration(),
                   anomalies=len(session.anomalies_found),
                   hypotheses=len(session.hypotheses_generated))
    
    def get_active_sessions(self) -> List[HuntSession]:
        """Get all active hunt sessions."""
        return list(self.active_sessions.values())
    
    def get_session(self, session_id: str) -> Optional[HuntSession]:
        """Get hunt session by ID."""
        return self.active_sessions.get(session_id) or next(
            (s for s in self.historical_hunts if s.session_id == session_id), None
        )
    
    def get_recent_threats(self, hours: int = 24) -> List[ThreatHypothesis]:
        """Get recent threat hypotheses."""
        cutoff_time = time.time() - (hours * 3600)
        threats = []
        
        for session in self.historical_hunts + list(self.active_sessions.values()):
            for hypothesis in session.hypotheses_generated:
                if hypothesis.timestamp >= cutoff_time:
                    threats.append(hypothesis)
        
        return sorted(threats, key=lambda t: t.risk_score, reverse=True)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get hunting system statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "historical_hunts": len(self.historical_hunts),
            "anomaly_detectors": len(self.anomaly_detectors),
            "data_points_buffered": len(self.data_buffer),
            **self.stats
        }


# Global AI threat hunter instance
ai_threat_hunter = AIThreatHunter()


async def initialize_ai_threat_hunting():
    """Initialize the AI threat hunting system."""
    await ai_threat_hunter.start_threat_hunting()


async def shutdown_ai_threat_hunting():
    """Shutdown the AI threat hunting system."""
    await ai_threat_hunter.stop_threat_hunting()


def get_ai_threat_hunter() -> AIThreatHunter:
    """Get the global AI threat hunter."""
    return ai_threat_hunter