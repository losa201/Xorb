"""
Advanced threat detection engine
Implements ML-based anomaly detection and threat intelligence integration
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from ..ingestion.event_normalizer import NormalizedEvent, EventCategory, EventSeverity


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ThreatType(Enum):
    """Types of detected threats"""
    MALWARE = "malware"
    BOTNET = "botnet"
    APT = "apt"
    INSIDER_THREAT = "insider_threat"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    RECONNAISSANCE = "reconnaissance"
    DENIAL_OF_SERVICE = "denial_of_service"
    PHISHING = "phishing"
    CRYPTOMINING = "cryptomining"
    UNKNOWN = "unknown"


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    indicator_id: str
    value: str
    type: str  # ip, domain, hash, url, email
    threat_type: ThreatType
    severity: ThreatLevel
    confidence: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str = ""
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """Detected threat result"""
    detection_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.utcnow()).encode()).hexdigest())
    threat_type: ThreatType = ThreatType.UNKNOWN
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    confidence: float = 0.0
    
    # Detection details
    title: str = ""
    description: str = ""
    evidence: List[NormalizedEvent] = field(default_factory=list)
    indicators: List[ThreatIndicator] = field(default_factory=list)
    
    # Attack context
    attack_vector: str = ""
    kill_chain_phase: str = ""
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    
    # Asset context
    affected_assets: Set[str] = field(default_factory=set)
    compromised_accounts: Set[str] = field(default_factory=set)
    network_indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Timeline
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    detection_time: datetime = field(default_factory=datetime.utcnow)
    
    # Response
    recommended_actions: List[str] = field(default_factory=list)
    containment_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "detection_id": self.detection_id,
            "threat_type": self.threat_type.value,
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "attack_vector": self.attack_vector,
            "kill_chain_phase": self.kill_chain_phase,
            "mitre_tactics": self.mitre_tactics,
            "mitre_techniques": self.mitre_techniques,
            "affected_assets": list(self.affected_assets),
            "compromised_accounts": list(self.compromised_accounts),
            "network_indicators": self.network_indicators,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "detection_time": self.detection_time.isoformat(),
            "recommended_actions": self.recommended_actions,
            "containment_suggestions": self.containment_suggestions,
            "evidence_count": len(self.evidence),
            "indicator_count": len(self.indicators)
        }


class BehavioralProfile:
    """User/Asset behavioral profile for anomaly detection"""
    
    def __init__(self, entity_id: str, entity_type: str = "user"):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.baseline_established = False
        
        # Activity patterns
        self.login_times: List[int] = []  # Hours of day
        self.active_days: Set[int] = set()  # Days of week
        self.source_ips: Set[str] = set()
        self.accessed_resources: Set[str] = set()
        self.typical_actions: Dict[str, int] = defaultdict(int)
        
        # Statistical baselines
        self.avg_session_duration: float = 0.0
        self.avg_data_volume: float = 0.0
        self.typical_error_rate: float = 0.0
        
        # Learning parameters
        self.observation_window = timedelta(days=30)
        self.min_observations = 50
        self.observations = 0
        
        # Last updated
        self.last_updated = datetime.utcnow()
    
    def update_profile(self, event: NormalizedEvent):
        """Update profile with new event"""
        self.observations += 1
        self.last_updated = datetime.utcnow()
        
        # Update patterns
        self.login_times.append(event.timestamp.hour)
        self.active_days.add(event.timestamp.weekday())
        
        if event.source_ip:
            self.source_ips.add(event.source_ip)
        
        if event.action:
            self.typical_actions[event.action] += 1
        
        # Maintain reasonable memory usage
        if len(self.login_times) > 1000:
            self.login_times = self.login_times[-500:]
        
        # Check if baseline is established
        if self.observations >= self.min_observations:
            self.baseline_established = True
    
    def calculate_anomaly_score(self, event: NormalizedEvent) -> float:
        """Calculate anomaly score for event (0.0 = normal, 1.0 = highly anomalous)"""
        if not self.baseline_established:
            return 0.0
        
        anomaly_factors = []
        
        # Time-based anomaly
        if self.login_times:
            typical_hours = set(self.login_times[-100:])  # Recent patterns
            if event.timestamp.hour not in typical_hours:
                anomaly_factors.append(0.6)
        
        # Day-based anomaly
        if event.timestamp.weekday() not in self.active_days:
            anomaly_factors.append(0.4)
        
        # IP-based anomaly
        if event.source_ip and event.source_ip not in self.source_ips:
            anomaly_factors.append(0.8)
        
        # Action-based anomaly
        if event.action and event.action not in self.typical_actions:
            anomaly_factors.append(0.5)
        
        # Calculate weighted average
        if anomaly_factors:
            return min(1.0, sum(anomaly_factors) / len(anomaly_factors))
        
        return 0.0


class ThreatIntelligence:
    """Threat intelligence management and lookup"""
    
    def __init__(self):
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.reputation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=24)
    
    def add_indicator(self, indicator: ThreatIndicator):
        """Add threat indicator"""
        self.indicators[indicator.value] = indicator
    
    def lookup_indicator(self, value: str, indicator_type: str = None) -> Optional[ThreatIndicator]:
        """Lookup threat indicator"""
        indicator = self.indicators.get(value)
        if indicator and (not indicator_type or indicator.type == indicator_type):
            indicator.last_seen = datetime.utcnow()
            return indicator
        return None
    
    def get_reputation(self, value: str, indicator_type: str) -> Dict[str, Any]:
        """Get reputation information for indicator"""
        cache_key = f"{indicator_type}:{value}"
        
        # Check cache
        if cache_key in self.reputation_cache:
            cached_result = self.reputation_cache[cache_key]
            if datetime.utcnow() - cached_result['timestamp'] < self.cache_ttl:
                return cached_result['data']
        
        # Mock reputation lookup (in production, integrate with threat intel feeds)
        reputation = {
            "malicious": False,
            "suspicious": False,
            "reputation_score": 0,  # -100 to 100
            "categories": [],
            "last_analysis": datetime.utcnow().isoformat(),
            "sources": []
        }
        
        # Cache result
        self.reputation_cache[cache_key] = {
            "data": reputation,
            "timestamp": datetime.utcnow()
        }
        
        return reputation
    
    def enrich_event(self, event: NormalizedEvent) -> Dict[str, Any]:
        """Enrich event with threat intelligence"""
        enrichment = {
            "threat_indicators": [],
            "reputation_data": {},
            "risk_score": 0
        }
        
        # Check various fields for indicators
        fields_to_check = [
            (event.source_ip, "ip"),
            (event.dest_ip, "ip"),
            (event.file_hash, "hash"),
            (event.process_name, "process")
        ]
        
        for value, indicator_type in fields_to_check:
            if value:
                indicator = self.lookup_indicator(value, indicator_type)
                if indicator:
                    enrichment["threat_indicators"].append({
                        "value": value,
                        "type": indicator_type,
                        "threat_type": indicator.threat_type.value,
                        "severity": indicator.severity.value,
                        "confidence": indicator.confidence
                    })
                    enrichment["risk_score"] += indicator.confidence * 10
                
                # Get reputation data
                if indicator_type in ["ip", "hash"]:
                    reputation = self.get_reputation(value, indicator_type)
                    enrichment["reputation_data"][value] = reputation
                    
                    if reputation["malicious"]:
                        enrichment["risk_score"] += 50
                    elif reputation["suspicious"]:
                        enrichment["risk_score"] += 25
        
        return enrichment


class ThreatDetector:
    """Advanced threat detection engine"""
    
    def __init__(self):
        self.behavioral_profiles: Dict[str, BehavioralProfile] = {}
        self.threat_intelligence = ThreatIntelligence()
        self.detection_rules = {}
        self.recent_detections: deque = deque(maxlen=1000)
        
        # Initialize detection rules
        self._initialize_detection_rules()
        
        # Load threat intelligence
        self._load_threat_intelligence()
    
    def analyze_event(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """Analyze event for threats"""
        detections = []
        
        # Enrich event with threat intelligence
        enrichment = self.threat_intelligence.enrich_event(event)
        event.threat_intel = enrichment
        
        # Update behavioral profiles
        self._update_behavioral_profiles(event)
        
        # Apply detection rules
        for rule_name, rule_func in self.detection_rules.items():
            try:
                detection = rule_func(event)
                if detection:
                    detections.append(detection)
            except Exception as e:
                print(f"Error in detection rule {rule_name}: {e}")
        
        # Store detections
        self.recent_detections.extend(detections)
        
        return detections
    
    def _update_behavioral_profiles(self, event: NormalizedEvent):
        """Update behavioral profiles for users and assets"""
        entities_to_profile = []
        
        if event.source_user:
            entities_to_profile.append((event.source_user, "user"))
        if event.source_host:
            entities_to_profile.append((event.source_host, "host"))
        
        for entity_id, entity_type in entities_to_profile:
            if entity_id not in self.behavioral_profiles:
                self.behavioral_profiles[entity_id] = BehavioralProfile(entity_id, entity_type)
            
            self.behavioral_profiles[entity_id].update_profile(event)
    
    def _initialize_detection_rules(self):
        """Initialize built-in detection rules"""
        self.detection_rules = {
            "malware_detection": self._detect_malware,
            "insider_threat": self._detect_insider_threat,
            "brute_force": self._detect_brute_force,
            "data_exfiltration": self._detect_data_exfiltration,
            "lateral_movement": self._detect_lateral_movement,
            "privilege_escalation": self._detect_privilege_escalation,
            "reconnaissance": self._detect_reconnaissance,
            "anomalous_behavior": self._detect_anomalous_behavior
        }
    
    def _detect_malware(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect malware-related activity"""
        threat_indicators = event.threat_intel.get("threat_indicators", [])
        
        for indicator in threat_indicators:
            if indicator["threat_type"] == "malware":
                return ThreatDetection(
                    threat_type=ThreatType.MALWARE,
                    threat_level=ThreatLevel.HIGH,
                    confidence=indicator["confidence"],
                    title="Malware Detection",
                    description=f"Detected malware indicator: {indicator['value']}",
                    evidence=[event],
                    attack_vector="malware",
                    kill_chain_phase="installation",
                    mitre_tactics=["TA0002"],  # Execution
                    mitre_techniques=["T1204"],  # User Execution
                    recommended_actions=[
                        "Isolate affected system",
                        "Run full antimalware scan",
                        "Check for persistence mechanisms",
                        "Analyze network connections"
                    ]
                )
        
        # Check for malware-like process behavior
        if (event.process_name and 
            event.category == EventCategory.PROCESS_ACTIVITY and
            any(suspicious in event.process_name.lower() for suspicious in 
                ['powershell', 'cmd', 'wscript', 'cscript', 'regsvr32'])):
            
            if event.command_line and any(indicator in event.command_line.lower() for indicator in
                                        ['downloadstring', 'invoke-expression', 'base64', 'encoded']):
                return ThreatDetection(
                    threat_type=ThreatType.MALWARE,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.7,
                    title="Suspicious Process Execution",
                    description="Detected potentially malicious process execution",
                    evidence=[event],
                    attack_vector="process_execution",
                    kill_chain_phase="execution",
                    mitre_tactics=["TA0002"],
                    mitre_techniques=["T1059"]
                )
        
        return None
    
    def _detect_insider_threat(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect insider threat indicators"""
        if not event.source_user:
            return None
        
        # Check behavioral anomaly
        profile = self.behavioral_profiles.get(event.source_user)
        if profile:
            anomaly_score = profile.calculate_anomaly_score(event)
            
            if anomaly_score > 0.8:
                return ThreatDetection(
                    threat_type=ThreatType.INSIDER_THREAT,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=anomaly_score,
                    title="Insider Threat - Anomalous Behavior",
                    description=f"User {event.source_user} exhibiting anomalous behavior",
                    evidence=[event],
                    attack_vector="insider",
                    kill_chain_phase="persistence",
                    mitre_tactics=["TA0003"],  # Persistence
                    recommended_actions=[
                        "Review user access permissions",
                        "Monitor user activity closely",
                        "Check for data access patterns",
                        "Validate business justification"
                    ]
                )
        
        # Check for after-hours access to sensitive data
        if (event.category == EventCategory.DATA_ACCESS and
            (event.timestamp.hour < 6 or event.timestamp.hour > 22)):
            return ThreatDetection(
                threat_type=ThreatType.INSIDER_THREAT,
                threat_level=ThreatLevel.LOW,
                confidence=0.6,
                title="After-hours Data Access",
                description="Sensitive data accessed outside normal business hours",
                evidence=[event]
            )
        
        return None
    
    def _detect_brute_force(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect brute force attacks"""
        if (event.category == EventCategory.AUTHENTICATION and 
            event.outcome.value == "failure"):
            
            # This would typically check frequency across events
            # For now, flag high-severity authentication failures
            return ThreatDetection(
                threat_type=ThreatType.BRUTE_FORCE,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.8,
                title="Potential Brute Force Attack",
                description=f"Authentication failure from {event.source_ip}",
                evidence=[event],
                attack_vector="credential_access",
                kill_chain_phase="credential_access",
                mitre_tactics=["TA0006"],  # Credential Access
                mitre_techniques=["T1110"],  # Brute Force
                recommended_actions=[
                    "Block source IP",
                    "Enable account lockout",
                    "Implement rate limiting",
                    "Review authentication logs"
                ]
            )
        
        return None
    
    def _detect_data_exfiltration(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect data exfiltration attempts"""
        if event.category == EventCategory.DATA_ACCESS:
            # Check for large data transfers
            if event.bytes_out and event.bytes_out > 1000000:  # 1MB
                return ThreatDetection(
                    threat_type=ThreatType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.7,
                    title="Potential Data Exfiltration",
                    description=f"Large data transfer detected: {event.bytes_out} bytes",
                    evidence=[event],
                    attack_vector="data_transfer",
                    kill_chain_phase="exfiltration",
                    mitre_tactics=["TA0010"],  # Exfiltration
                    mitre_techniques=["T1041"],  # Exfiltration Over C2 Channel
                    recommended_actions=[
                        "Monitor network traffic",
                        "Review data access logs",
                        "Check for unauthorized transfers",
                        "Validate business justification"
                    ]
                )
        
        return None
    
    def _detect_lateral_movement(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect lateral movement attempts"""
        if (event.category == EventCategory.NETWORK_TRAFFIC and
            event.source_ip != event.dest_ip):
            
            # Check for internal-to-internal connections on admin ports
            admin_ports = [22, 23, 135, 139, 445, 3389, 5985, 5986]
            if event.dest_port in admin_ports:
                return ThreatDetection(
                    threat_type=ThreatType.LATERAL_MOVEMENT,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.6,
                    title="Potential Lateral Movement",
                    description=f"Connection to admin port {event.dest_port}",
                    evidence=[event],
                    attack_vector="network",
                    kill_chain_phase="lateral_movement",
                    mitre_tactics=["TA0008"],  # Lateral Movement
                    mitre_techniques=["T1021"],  # Remote Services
                    recommended_actions=[
                        "Monitor network segments",
                        "Review admin account usage",
                        "Check for privilege escalation",
                        "Validate connection necessity"
                    ]
                )
        
        return None
    
    def _detect_privilege_escalation(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect privilege escalation attempts"""
        if event.category == EventCategory.AUTHORIZATION:
            if "admin" in event.message.lower() or "root" in event.message.lower():
                return ThreatDetection(
                    threat_type=ThreatType.PRIVILEGE_ESCALATION,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.8,
                    title="Privilege Escalation Detected",
                    description="Elevated privileges obtained",
                    evidence=[event],
                    attack_vector="privilege_escalation",
                    kill_chain_phase="privilege_escalation",
                    mitre_tactics=["TA0004"],  # Privilege Escalation
                    mitre_techniques=["T1078"],  # Valid Accounts
                    recommended_actions=[
                        "Review privilege changes",
                        "Validate authorization",
                        "Check for persistence",
                        "Monitor privileged operations"
                    ]
                )
        
        return None
    
    def _detect_reconnaissance(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect reconnaissance activity"""
        if event.category == EventCategory.NETWORK_TRAFFIC:
            # Check for port scanning patterns
            common_scan_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995]
            if event.dest_port in common_scan_ports:
                return ThreatDetection(
                    threat_type=ThreatType.RECONNAISSANCE,
                    threat_level=ThreatLevel.LOW,
                    confidence=0.5,
                    title="Potential Reconnaissance",
                    description=f"Connection to common service port {event.dest_port}",
                    evidence=[event],
                    attack_vector="network_scanning",
                    kill_chain_phase="reconnaissance",
                    mitre_tactics=["TA0007"],  # Discovery
                    mitre_techniques=["T1046"],  # Network Service Scanning
                    recommended_actions=[
                        "Monitor for scanning patterns",
                        "Check firewall logs",
                        "Review network topology",
                        "Implement network segmentation"
                    ]
                )
        
        return None
    
    def _detect_anomalous_behavior(self, event: NormalizedEvent) -> Optional[ThreatDetection]:
        """Detect general anomalous behavior"""
        # Check threat intelligence risk score
        risk_score = event.threat_intel.get("risk_score", 0)
        
        if risk_score > 75:
            return ThreatDetection(
                threat_type=ThreatType.UNKNOWN,
                threat_level=ThreatLevel.MEDIUM,
                confidence=min(1.0, risk_score / 100.0),
                title="High-Risk Activity Detected",
                description=f"Activity with high threat intelligence risk score: {risk_score}",
                evidence=[event],
                recommended_actions=[
                    "Investigate activity context",
                    "Review threat intelligence",
                    "Monitor for related events",
                    "Consider containment measures"
                ]
            )
        
        return None
    
    def _load_threat_intelligence(self):
        """Load threat intelligence indicators"""
        # Sample threat indicators (in production, load from feeds)
        sample_indicators = [
            {
                "value": "192.168.1.100",
                "type": "ip",
                "threat_type": "botnet",
                "severity": "high",
                "confidence": 0.9,
                "source": "internal_honeypot"
            },
            {
                "value": "malware.exe",
                "type": "process",
                "threat_type": "malware",
                "severity": "critical",
                "confidence": 0.95,
                "source": "signature_detection"
            }
        ]
        
        for indicator_data in sample_indicators:
            indicator = ThreatIndicator(
                indicator_id=hashlib.md5(indicator_data["value"].encode()).hexdigest(),
                value=indicator_data["value"],
                type=indicator_data["type"],
                threat_type=ThreatType(indicator_data["threat_type"]),
                severity=ThreatLevel(indicator_data["severity"]),
                confidence=indicator_data["confidence"],
                source=indicator_data["source"],
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
            self.threat_intelligence.add_indicator(indicator)
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        total_detections = len(self.recent_detections)
        
        detections_by_type = defaultdict(int)
        detections_by_level = defaultdict(int)
        
        for detection in self.recent_detections:
            detections_by_type[detection.threat_type.value] += 1
            detections_by_level[detection.threat_level.value] += 1
        
        return {
            "total_detections": total_detections,
            "detections_by_type": dict(detections_by_type),
            "detections_by_level": dict(detections_by_level),
            "behavioral_profiles": len(self.behavioral_profiles),
            "threat_indicators": len(self.threat_intelligence.indicators)
        }