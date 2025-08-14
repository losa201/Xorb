"""
Advanced threat detection engine
Uses machine learning and rule-based detection for threat identification
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import threading
from collections import defaultdict

from ..ingestion.event_normalizer import NormalizedEvent, EventCategory, EventSeverity, EventAction

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of threats detected"""
    MALWARE = "malware"
    BOTNET = "botnet"
    APT = "apt"
    INSIDER_THREAT = "insider_threat"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_INJECTION = "command_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PHISHING = "phishing"
    RECONNAISSANCE = "reconnaissance"
    UNKNOWN = "unknown"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ThreatDetection:
    """Threat detection result"""
    detection_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float
    description: str
    event: NormalizedEvent
    indicators: List[str]
    mitre_techniques: List[str] = field(default_factory=list)
    iocs: List[str] = field(default_factory=list)  # Indicators of Compromise
    recommendations: List[str] = field(default_factory=list)
    additional_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ThreatSignature:
    """Threat detection signature"""

    def __init__(self, signature_id: str, name: str, threat_type: ThreatType,
                 severity: ThreatSeverity, patterns: List[str],
                 mitre_techniques: List[str] = None):
        self.signature_id = signature_id
        self.name = name
        self.threat_type = threat_type
        self.severity = severity
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.mitre_techniques = mitre_techniques or []
        self.enabled = True
        self.hit_count = 0
        self.last_hit = None


class IPReputation:
    """IP reputation tracking"""

    def __init__(self):
        self.known_bad_ips: Set[str] = set()
        self.reputation_scores: Dict[str, float] = {}  # 0.0 = bad, 1.0 = good
        self.lock = threading.RLock()

        # Load some known bad IP ranges (example)
        self._load_known_bad_ips()

    def check_ip(self, ip: str) -> Tuple[bool, float]:
        """Check IP reputation. Returns (is_malicious, confidence)"""
        with self.lock:
            if ip in self.known_bad_ips:
                return True, 1.0

            reputation = self.reputation_scores.get(ip, 0.5)  # Default neutral
            is_malicious = reputation < 0.3
            confidence = abs(reputation - 0.5) * 2  # Convert to 0-1 confidence

            return is_malicious, confidence

    def add_bad_ip(self, ip: str, confidence: float = 1.0):
        """Add IP to bad reputation list"""
        with self.lock:
            self.known_bad_ips.add(ip)
            self.reputation_scores[ip] = max(0.0, 1.0 - confidence)

    def _load_known_bad_ips(self):
        """Load known bad IP addresses"""
        # In production, this would load from threat intelligence feeds
        known_bad = [
            "127.0.0.1",  # Example - don't use localhost in production
            "0.0.0.0",
            "255.255.255.255"
        ]

        for ip in known_bad:
            self.known_bad_ips.add(ip)
            self.reputation_scores[ip] = 0.0


class ThreatDetector:
    """Advanced threat detection engine"""

    def __init__(self):
        self.signatures: Dict[str, ThreatSignature] = {}
        self.ip_reputation = IPReputation()
        self.user_behavior_baselines: Dict[str, Dict[str, Any]] = {}
        self.detection_statistics = defaultdict(int)
        self.lock = threading.RLock()

        # Load detection signatures
        self._load_threat_signatures()

        # Behavioral analysis parameters
        self.baseline_learning_period = 7  # days
        self.anomaly_threshold = 2.0  # standard deviations

    def analyze_event(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """Analyze event for threats"""
        detections = []

        with self.lock:
            self.detection_statistics['events_analyzed'] += 1

            try:
                # Signature-based detection
                signature_detections = self._signature_detection(event)
                detections.extend(signature_detections)

                # IP reputation check
                ip_detections = self._ip_reputation_check(event)
                detections.extend(ip_detections)

                # Behavioral analysis
                behavioral_detections = self._behavioral_analysis(event)
                detections.extend(behavioral_detections)

                # Pattern analysis
                pattern_detections = self._pattern_analysis(event)
                detections.extend(pattern_detections)

                # Update statistics
                if detections:
                    self.detection_statistics['threats_detected'] += len(detections)
                    for detection in detections:
                        self.detection_statistics[f'threat_type_{detection.threat_type.value}'] += 1

            except Exception as e:
                logger.error(f"Error analyzing event for threats: {e}")
                self.detection_statistics['analysis_errors'] += 1

        return detections

    def _signature_detection(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """Signature-based threat detection"""
        detections = []

        # Check message against all signatures
        message = event.message.lower() if event.message else ""

        for signature in self.signatures.values():
            if not signature.enabled:
                continue

            for pattern in signature.patterns:
                if pattern.search(message):
                    detection = self._create_detection(
                        signature.threat_type,
                        signature.severity,
                        1.0,  # High confidence for signature match
                        f"Signature match: {signature.name}",
                        event,
                        indicators=[f"Pattern: {pattern.pattern}"],
                        mitre_techniques=signature.mitre_techniques
                    )
                    detections.append(detection)

                    # Update signature statistics
                    signature.hit_count += 1
                    signature.last_hit = datetime.utcnow()
                    break  # One match per signature is enough

        return detections

    def _ip_reputation_check(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """IP reputation-based detection"""
        detections = []

        # Check source IP
        if event.source_ip:
            is_malicious, confidence = self.ip_reputation.check_ip(event.source_ip)
            if is_malicious:
                detection = self._create_detection(
                    ThreatType.MALWARE,  # Could be more specific
                    ThreatSeverity.HIGH,
                    confidence,
                    f"Malicious IP detected: {event.source_ip}",
                    event,
                    indicators=[f"Source IP: {event.source_ip}"],
                    iocs=[event.source_ip]
                )
                detections.append(detection)

        # Check destination IP
        if event.destination_ip:
            is_malicious, confidence = self.ip_reputation.check_ip(event.destination_ip)
            if is_malicious:
                detection = self._create_detection(
                    ThreatType.DATA_EXFILTRATION,
                    ThreatSeverity.MEDIUM,
                    confidence,
                    f"Communication with malicious IP: {event.destination_ip}",
                    event,
                    indicators=[f"Destination IP: {event.destination_ip}"],
                    iocs=[event.destination_ip]
                )
                detections.append(detection)

        return detections

    def _behavioral_analysis(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """Behavioral anomaly detection"""
        detections = []

        if not event.user:
            return detections

        # Get user behavior baseline
        baseline = self.user_behavior_baselines.get(event.user, {})

        # Check for time-based anomalies
        if 'login_hours' in baseline:
            current_hour = event.timestamp.hour
            normal_hours = baseline['login_hours']

            if current_hour not in normal_hours:
                detection = self._create_detection(
                    ThreatType.INSIDER_THREAT,
                    ThreatSeverity.MEDIUM,
                    0.7,
                    f"User {event.user} accessing system outside normal hours",
                    event,
                    indicators=[f"Login at hour {current_hour}, normal hours: {normal_hours}"],
                    mitre_techniques=['T1078']
                )
                detections.append(detection)

        # Check for geographic anomalies (if geolocation data available)
        if event.source_geolocation and 'normal_locations' in baseline:
            # Implementation would check if current location is anomalous
            pass

        # Update baseline (simplified - in production would be more sophisticated)
        self._update_user_baseline(event.user, event)

        return detections

    def _pattern_analysis(self, event: NormalizedEvent) -> List[ThreatDetection]:
        """Pattern-based threat detection"""
        detections = []

        # SQL Injection detection
        if event.http_method and event.url:
            sql_patterns = [
                r"union\s+select", r"or\s+1\s*=\s*1", r"drop\s+table",
                r"exec\s*\(", r"script>", r"javascript:"
            ]

            combined_text = f"{event.url} {event.message}".lower()

            for pattern in sql_patterns:
                if re.search(pattern, combined_text):
                    threat_type = ThreatType.SQL_INJECTION if "union" in pattern or "drop" in pattern else ThreatType.XSS

                    detection = self._create_detection(
                        threat_type,
                        ThreatSeverity.HIGH,
                        0.8,
                        f"Potential {threat_type.value} attack detected",
                        event,
                        indicators=[f"Pattern: {pattern}"],
                        mitre_techniques=['T1190']
                    )
                    detections.append(detection)
                    break

        # Command injection detection
        if event.message:
            command_patterns = [
                r";\s*(cat|ls|ps|id|whoami|uname)", r"\|\s*(nc|netcat|wget|curl)",
                r"&&\s*(rm|mv|cp)\s", r"`.*`", r"\$\(.*\)"
            ]

            for pattern in command_patterns:
                if re.search(pattern, event.message, re.IGNORECASE):
                    detection = self._create_detection(
                        ThreatType.COMMAND_INJECTION,
                        ThreatSeverity.HIGH,
                        0.9,
                        "Potential command injection detected",
                        event,
                        indicators=[f"Pattern: {pattern}"],
                        mitre_techniques=['T1059']
                    )
                    detections.append(detection)
                    break

        # Reconnaissance detection
        if event.category == EventCategory.NETWORK_TRAFFIC:
            # Check for port scanning patterns
            if event.destination_port and event.destination_port in [22, 23, 135, 139, 445, 3389]:
                detection = self._create_detection(
                    ThreatType.RECONNAISSANCE,
                    ThreatSeverity.LOW,
                    0.6,
                    f"Potential reconnaissance on sensitive port {event.destination_port}",
                    event,
                    indicators=[f"Port: {event.destination_port}"],
                    mitre_techniques=['T1046']
                )
                detections.append(detection)

        return detections

    def _create_detection(self, threat_type: ThreatType, severity: ThreatSeverity,
                         confidence: float, description: str, event: NormalizedEvent,
                         indicators: List[str], mitre_techniques: List[str] = None,
                         iocs: List[str] = None) -> ThreatDetection:
        """Create threat detection object"""

        detection_id = hashlib.md5(
            f"{event.event_id}{threat_type.value}{description}".encode()
        ).hexdigest()[:16]

        recommendations = self._get_recommendations(threat_type)

        return ThreatDetection(
            detection_id=detection_id,
            threat_type=threat_type,
            severity=severity,
            confidence=confidence,
            description=description,
            event=event,
            indicators=indicators,
            mitre_techniques=mitre_techniques or [],
            iocs=iocs or [],
            recommendations=recommendations
        )

    def _get_recommendations(self, threat_type: ThreatType) -> List[str]:
        """Get recommendations based on threat type"""
        recommendations_map = {
            ThreatType.MALWARE: [
                "Isolate affected systems immediately",
                "Run full antivirus scan",
                "Check for lateral movement",
                "Review network traffic for C2 communication"
            ],
            ThreatType.BRUTE_FORCE: [
                "Implement account lockout policies",
                "Enable multi-factor authentication",
                "Monitor for credential stuffing attacks",
                "Block source IP temporarily"
            ],
            ThreatType.SQL_INJECTION: [
                "Validate all user inputs",
                "Use parameterized queries",
                "Apply principle of least privilege",
                "Update web application firewall rules"
            ],
            ThreatType.COMMAND_INJECTION: [
                "Sanitize all user inputs",
                "Implement input validation",
                "Use safe API calls instead of system commands",
                "Apply sandboxing for user inputs"
            ],
            ThreatType.RECONNAISSANCE: [
                "Monitor for additional scanning activity",
                "Check firewall logs for patterns",
                "Consider implementing port knocking",
                "Block suspicious source IPs"
            ],
            ThreatType.INSIDER_THREAT: [
                "Review user access privileges",
                "Monitor data access patterns",
                "Implement user activity monitoring",
                "Conduct security awareness training"
            ]
        }

        return recommendations_map.get(threat_type, [
            "Monitor for additional suspicious activity",
            "Review security logs",
            "Consider threat containment measures"
        ])

    def _update_user_baseline(self, user: str, event: NormalizedEvent):
        """Update user behavioral baseline"""
        if user not in self.user_behavior_baselines:
            self.user_behavior_baselines[user] = {
                'login_hours': set(),
                'source_ips': set(),
                'first_seen': event.timestamp,
                'last_seen': event.timestamp,
                'event_count': 0
            }

        baseline = self.user_behavior_baselines[user]
        baseline['login_hours'].add(event.timestamp.hour)
        if event.source_ip:
            baseline['source_ips'].add(event.source_ip)
        baseline['last_seen'] = event.timestamp
        baseline['event_count'] += 1

    def _load_threat_signatures(self):
        """Load threat detection signatures"""

        # Malware signatures
        self.signatures['malware_generic'] = ThreatSignature(
            "malware_generic",
            "Generic Malware Detection",
            ThreatType.MALWARE,
            ThreatSeverity.HIGH,
            [
                r"virus.*detected", r"malware.*found", r"trojan.*identified",
                r"backdoor.*installed", r"rootkit.*detected"
            ],
            ['T1055', 'T1027']
        )

        # Brute force signatures
        self.signatures['brute_force_ssh'] = ThreatSignature(
            "brute_force_ssh",
            "SSH Brute Force Attack",
            ThreatType.BRUTE_FORCE,
            ThreatSeverity.MEDIUM,
            [
                r"failed.*ssh.*login", r"authentication.*failure.*ssh",
                r"invalid.*user.*ssh", r"failed.*password.*ssh"
            ],
            ['T1110', 'T1021.004']
        )

        # Web attack signatures
        self.signatures['web_attack'] = ThreatSignature(
            "web_attack",
            "Web Application Attack",
            ThreatType.SQL_INJECTION,
            ThreatSeverity.HIGH,
            [
                r"union.*select", r"or.*1.*=.*1", r"<script>",
                r"javascript:", r"eval\(", r"exec\("
            ],
            ['T1190']
        )

        # Privilege escalation
        self.signatures['privilege_escalation'] = ThreatSignature(
            "privilege_escalation",
            "Privilege Escalation Attempt",
            ThreatType.PRIVILEGE_ESCALATION,
            ThreatSeverity.HIGH,
            [
                r"sudo.*su", r"privilege.*escalation", r"admin.*access",
                r"root.*access", r"setuid.*exploit"
            ],
            ['T1548', 'T1134']
        )

        # Data exfiltration
        self.signatures['data_exfiltration'] = ThreatSignature(
            "data_exfiltration",
            "Data Exfiltration Activity",
            ThreatType.DATA_EXFILTRATION,
            ThreatSeverity.CRITICAL,
            [
                r"large.*file.*transfer", r"database.*dump", r"export.*data",
                r"ftp.*upload", r"scp.*transfer"
            ],
            ['T1041', 'T1020']
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        with self.lock:
            stats = dict(self.detection_statistics)
            stats.update({
                'active_signatures': len([s for s in self.signatures.values() if s.enabled]),
                'total_signatures': len(self.signatures),
                'known_bad_ips': len(self.ip_reputation.known_bad_ips),
                'user_baselines': len(self.user_behavior_baselines)
            })
            return stats

    def add_signature(self, signature: ThreatSignature):
        """Add custom threat signature"""
        with self.lock:
            self.signatures[signature.signature_id] = signature
            logger.info(f"Added threat signature: {signature.name}")

    def remove_signature(self, signature_id: str):
        """Remove threat signature"""
        with self.lock:
            if signature_id in self.signatures:
                del self.signatures[signature_id]
                logger.info(f"Removed threat signature: {signature_id}")

    def update_ip_reputation(self, ip: str, is_malicious: bool, confidence: float = 1.0):
        """Update IP reputation"""
        if is_malicious:
            self.ip_reputation.add_bad_ip(ip, confidence)
        else:
            with self.ip_reputation.lock:
                # Remove from bad IPs if present
                self.ip_reputation.known_bad_ips.discard(ip)
                # Set good reputation
                self.ip_reputation.reputation_scores[ip] = confidence
