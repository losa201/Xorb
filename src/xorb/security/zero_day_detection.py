"""
Advanced Zero-Day Detection Engine
Utilizes ML models and behavioral analysis to detect unknown vulnerabilities
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    ZERO_DAY = "zero_day"


class AttackVector(Enum):
    """Known attack vectors"""
    BUFFER_OVERFLOW = "buffer_overflow"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    RCE = "remote_code_execution"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNKNOWN = "unknown"


@dataclass
class AnomalyPattern:
    """Represents an anomalous behavior pattern"""
    pattern_id: str
    signature: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    frequency: int = 1
    attack_vector: AttackVector = AttackVector.UNKNOWN
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZeroDayCandidate:
    """Potential zero-day vulnerability candidate"""
    candidate_id: str
    pattern: AnomalyPattern
    exploitability_score: float
    impact_score: float
    novelty_score: float
    verification_status: str
    discovered_at: datetime
    affected_systems: List[str] = field(default_factory=list)
    proof_of_concept: Optional[str] = None
    mitigation_suggestions: List[str] = field(default_factory=list)


class BehavioralAnalysisEngine:
    """Advanced behavioral analysis for zero-day detection"""

    def __init__(self):
        self.baseline_profiles = {}
        self.anomaly_threshold = 0.85
        self.learning_window_days = 30
        self.pattern_database = {}

    async def initialize(self):
        """Initialize the behavioral analysis engine"""
        logger.info("Initializing Behavioral Analysis Engine for zero-day detection")
        await self._load_baseline_profiles()
        await self._load_known_patterns()

    async def _load_baseline_profiles(self):
        """Load baseline behavioral profiles"""
        # Simulate loading baseline profiles for different system types
        self.baseline_profiles = {
            "web_server": {
                "normal_request_patterns": ["GET /", "POST /api/*", "HEAD /*"],
                "typical_response_codes": [200, 301, 302, 404],
                "average_response_time": 150,
                "peak_traffic_hours": [9, 10, 11, 14, 15, 16]
            },
            "database": {
                "normal_query_patterns": ["SELECT", "INSERT", "UPDATE"],
                "connection_patterns": {"avg_connections": 25, "max_connections": 100},
                "query_complexity_baseline": 0.3
            },
            "network": {
                "normal_protocols": ["HTTP", "HTTPS", "DNS", "SMTP"],
                "traffic_patterns": {"inbound_ratio": 0.6, "outbound_ratio": 0.4},
                "port_usage_baseline": [80, 443, 22, 25, 53]
            }
        }

    async def _load_known_patterns(self):
        """Load known attack patterns for comparison"""
        self.pattern_database = {
            "sql_injection": {
                "signatures": ["'", "OR 1=1", "UNION SELECT", "DROP TABLE"],
                "payload_characteristics": ["unusual_quotes", "sql_keywords", "comment_strings"]
            },
            "xss": {
                "signatures": ["<script>", "javascript:", "onload=", "onerror="],
                "payload_characteristics": ["html_tags", "event_handlers", "javascript_code"]
            },
            "buffer_overflow": {
                "signatures": ["\\x41\\x41\\x41", "AAAA", "\\x90\\x90\\x90"],
                "payload_characteristics": ["repeated_characters", "shellcode_patterns", "nop_sleds"]
            }
        }

    async def analyze_traffic_anomalies(self, traffic_data: Dict[str, Any]) -> List[AnomalyPattern]:
        """Analyze network traffic for anomalous patterns"""
        anomalies = []

        try:
            # Analyze request patterns
            request_anomalies = await self._analyze_request_patterns(traffic_data.get("requests", []))
            anomalies.extend(request_anomalies)

            # Analyze payload characteristics
            payload_anomalies = await self._analyze_payload_patterns(traffic_data.get("payloads", []))
            anomalies.extend(payload_anomalies)

            # Analyze timing patterns
            timing_anomalies = await self._analyze_timing_patterns(traffic_data.get("timing", {}))
            anomalies.extend(timing_anomalies)

            # Analyze protocol violations
            protocol_anomalies = await self._analyze_protocol_violations(traffic_data.get("protocols", []))
            anomalies.extend(protocol_anomalies)

            return anomalies

        except Exception as e:
            logger.error(f"Traffic anomaly analysis failed: {e}")
            return []

    async def _analyze_request_patterns(self, requests: List[Dict]) -> List[AnomalyPattern]:
        """Analyze HTTP request patterns for anomalies"""
        anomalies = []

        for request in requests:
            try:
                # Check for unusual request structure
                if self._is_unusual_request_structure(request):
                    pattern = AnomalyPattern(
                        pattern_id=f"req_struct_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=self._generate_request_signature(request),
                        confidence=0.8,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.MEDIUM,
                        indicators=["unusual_request_structure"],
                        metadata={"request": request}
                    )
                    anomalies.append(pattern)

                # Check for encoding anomalies
                if self._has_encoding_anomalies(request):
                    pattern = AnomalyPattern(
                        pattern_id=f"encoding_anom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=f"encoding_anomaly_{hashlib.md5(str(request).encode()).hexdigest()[:8]}",
                        confidence=0.7,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.MEDIUM,
                        indicators=["encoding_anomalies"],
                        metadata={"request": request}
                    )
                    anomalies.append(pattern)

            except Exception as e:
                logger.debug(f"Request pattern analysis error: {e}")

        return anomalies

    async def _analyze_payload_patterns(self, payloads: List[str]) -> List[AnomalyPattern]:
        """Analyze payload patterns for novel attack techniques"""
        anomalies = []

        for payload in payloads:
            try:
                # Calculate payload entropy
                entropy = self._calculate_entropy(payload)

                # Check for high entropy (potential obfuscation)
                if entropy > 7.5:  # High entropy threshold
                    pattern = AnomalyPattern(
                        pattern_id=f"high_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=f"high_entropy_{hashlib.md5(payload.encode()).hexdigest()[:8]}",
                        confidence=0.9,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.HIGH,
                        indicators=["high_entropy_payload", f"entropy_{entropy:.2f}"],
                        metadata={"payload_length": len(payload), "entropy": entropy}
                    )
                    anomalies.append(pattern)

                # Check for novel encoding techniques
                if self._has_novel_encoding(payload):
                    pattern = AnomalyPattern(
                        pattern_id=f"novel_encoding_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=f"novel_encoding_{hashlib.md5(payload.encode()).hexdigest()[:8]}",
                        confidence=0.8,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.HIGH,
                        indicators=["novel_encoding_technique"],
                        metadata={"payload_sample": payload[:100]}
                    )
                    anomalies.append(pattern)

            except Exception as e:
                logger.debug(f"Payload analysis error: {e}")

        return anomalies

    async def _analyze_timing_patterns(self, timing_data: Dict) -> List[AnomalyPattern]:
        """Analyze timing patterns for timing-based attacks"""
        anomalies = []

        try:
            response_times = timing_data.get("response_times", [])
            if not response_times:
                return anomalies

            # Calculate timing statistics
            avg_response_time = np.mean(response_times)
            std_response_time = np.std(response_times)

            # Check for timing attack patterns
            for i, response_time in enumerate(response_times):
                # Check for unusually long response times (potential timing attacks)
                if response_time > avg_response_time + (3 * std_response_time):
                    pattern = AnomalyPattern(
                        pattern_id=f"timing_attack_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=f"timing_anomaly_{response_time}ms",
                        confidence=0.7,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.MEDIUM,
                        indicators=["timing_attack_pattern"],
                        metadata={"response_time": response_time, "baseline": avg_response_time}
                    )
                    anomalies.append(pattern)

        except Exception as e:
            logger.error(f"Timing pattern analysis failed: {e}")

        return anomalies

    async def _analyze_protocol_violations(self, protocols: List[Dict]) -> List[AnomalyPattern]:
        """Analyze for protocol-level violations and novel techniques"""
        anomalies = []

        for protocol_data in protocols:
            try:
                protocol_type = protocol_data.get("type", "unknown")

                # Check for protocol violations
                if self._has_protocol_violations(protocol_data):
                    pattern = AnomalyPattern(
                        pattern_id=f"protocol_violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        signature=f"protocol_violation_{protocol_type}",
                        confidence=0.85,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        attack_vector=AttackVector.UNKNOWN,
                        threat_level=ThreatLevel.HIGH,
                        indicators=["protocol_violation"],
                        metadata={"protocol": protocol_type, "violation_details": protocol_data}
                    )
                    anomalies.append(pattern)

            except Exception as e:
                logger.debug(f"Protocol analysis error: {e}")

        return anomalies

    def _is_unusual_request_structure(self, request: Dict) -> bool:
        """Check if request has unusual structure"""
        try:
            # Check for unusual header combinations
            headers = request.get("headers", {})
            unusual_headers = ["X-Original-URL", "X-Rewrite-URL", "X-Forwarded-Host"]

            for header in unusual_headers:
                if header in headers:
                    return True

            # Check for unusual parameter patterns
            params = request.get("params", {})
            if len(str(params)) > 10000:  # Unusually large parameters
                return True

            # Check for unusual method combinations
            method = request.get("method", "")
            if method in ["TRACE", "CONNECT", "OPTIONS"] and "admin" in request.get("path", ""):
                return True

            return False

        except Exception:
            return False

    def _has_encoding_anomalies(self, request: Dict) -> bool:
        """Check for encoding-based evasion techniques"""
        try:
            path = request.get("path", "")
            params = str(request.get("params", ""))

            # Check for multiple encoding layers
            encoding_indicators = ["%25", "\\u", "\\x", "&amp;", "&lt;", "&gt;"]
            encoding_count = sum(1 for indicator in encoding_indicators if indicator in path + params)

            return encoding_count >= 3

        except Exception:
            return False

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        try:
            if not data:
                return 0.0

            # Calculate frequency of each character
            freq = {}
            for char in data:
                freq[char] = freq.get(char, 0) + 1

            # Calculate entropy
            entropy = 0.0
            for count in freq.values():
                p = count / len(data)
                if p > 0:
                    entropy -= p * np.log2(p)

            return entropy

        except Exception:
            return 0.0

    def _has_novel_encoding(self, payload: str) -> bool:
        """Check for novel encoding techniques"""
        try:
            # Check for unusual character combinations
            unusual_patterns = [
                r"[\x00-\x1f]",  # Control characters
                r"[\x80-\xff]",  # High-bit characters
                r"\\[0-7]{3}",   # Octal encoding
                r"\\x[0-9a-fA-F]{2}",  # Hex encoding
                r"&\#[0-9]+;",   # Numeric entities
            ]

            import re
            for pattern in unusual_patterns:
                if re.search(pattern, payload):
                    return True

            return False

        except Exception:
            return False

    def _has_protocol_violations(self, protocol_data: Dict) -> bool:
        """Check for protocol-level violations"""
        try:
            protocol_type = protocol_data.get("type", "")

            if protocol_type.upper() == "HTTP":
                # Check for HTTP protocol violations
                version = protocol_data.get("version", "")
                if version not in ["1.0", "1.1", "2.0"]:
                    return True

                # Check for malformed headers
                headers = protocol_data.get("headers", {})
                if "Content-Length" in headers and "Transfer-Encoding" in headers:
                    return True  # HTTP smuggling attempt

            return False

        except Exception:
            return False

    def _generate_request_signature(self, request: Dict) -> str:
        """Generate a signature for the request"""
        try:
            method = request.get("method", "")
            path = request.get("path", "")
            params = str(request.get("params", ""))

            signature_data = f"{method}|{path}|{hashlib.md5(params.encode()).hexdigest()}"
            return hashlib.sha256(signature_data.encode()).hexdigest()[:16]

        except Exception:
            return "unknown_signature"


class ZeroDayDetectionEngine:
    """Main zero-day detection engine"""

    def __init__(self):
        self.behavioral_engine = BehavioralAnalysisEngine()
        self.pattern_correlator = PatternCorrelationEngine()
        self.vulnerability_assessor = VulnerabilityAssessmentEngine()
        self.candidates_database = {}
        self.detection_threshold = 0.8

    async def initialize(self):
        """Initialize the zero-day detection engine"""
        logger.info("Initializing Zero-Day Detection Engine")
        await self.behavioral_engine.initialize()
        await self.pattern_correlator.initialize()
        await self.vulnerability_assessor.initialize()

    async def analyze_for_zero_days(self, scan_data: Dict[str, Any]) -> List[ZeroDayCandidate]:
        """Main analysis function for zero-day detection"""
        try:
            logger.info("Starting zero-day analysis")

            # Step 1: Behavioral analysis
            anomalies = await self.behavioral_engine.analyze_traffic_anomalies(scan_data)
            logger.info(f"Found {len(anomalies)} behavioral anomalies")

            # Step 2: Pattern correlation
            correlated_patterns = await self.pattern_correlator.correlate_patterns(anomalies)
            logger.info(f"Correlated {len(correlated_patterns)} pattern groups")

            # Step 3: Vulnerability assessment
            candidates = []
            for pattern_group in correlated_patterns:
                candidate = await self.vulnerability_assessor.assess_zero_day_potential(pattern_group)
                if candidate and candidate.novelty_score > self.detection_threshold:
                    candidates.append(candidate)

            # Step 4: Rank and prioritize candidates
            ranked_candidates = await self._rank_candidates(candidates)

            logger.info(f"Identified {len(ranked_candidates)} zero-day candidates")
            return ranked_candidates

        except Exception as e:
            logger.error(f"Zero-day analysis failed: {e}")
            return []

    async def _rank_candidates(self, candidates: List[ZeroDayCandidate]) -> List[ZeroDayCandidate]:
        """Rank zero-day candidates by threat level and exploitability"""
        try:
            # Calculate composite score for each candidate
            for candidate in candidates:
                composite_score = (
                    candidate.exploitability_score * 0.4 +
                    candidate.impact_score * 0.3 +
                    candidate.novelty_score * 0.3
                )
                candidate.metadata["composite_score"] = composite_score

            # Sort by composite score (highest first)
            return sorted(candidates, key=lambda c: c.metadata.get("composite_score", 0), reverse=True)

        except Exception as e:
            logger.error(f"Candidate ranking failed: {e}")
            return candidates


class PatternCorrelationEngine:
    """Correlates anomaly patterns to identify potential zero-days"""

    def __init__(self):
        self.correlation_threshold = 0.7
        self.temporal_window_minutes = 60

    async def initialize(self):
        """Initialize pattern correlation engine"""
        logger.info("Initializing Pattern Correlation Engine")

    async def correlate_patterns(self, anomalies: List[AnomalyPattern]) -> List[List[AnomalyPattern]]:
        """Correlate anomaly patterns into related groups"""
        try:
            correlated_groups = []
            processed_patterns = set()

            for i, pattern in enumerate(anomalies):
                if pattern.pattern_id in processed_patterns:
                    continue

                # Find correlated patterns
                group = [pattern]
                processed_patterns.add(pattern.pattern_id)

                for j, other_pattern in enumerate(anomalies[i+1:], i+1):
                    if other_pattern.pattern_id in processed_patterns:
                        continue

                    correlation_score = await self._calculate_correlation(pattern, other_pattern)
                    if correlation_score > self.correlation_threshold:
                        group.append(other_pattern)
                        processed_patterns.add(other_pattern.pattern_id)

                if len(group) > 1:  # Only include groups with multiple patterns
                    correlated_groups.append(group)

            return correlated_groups

        except Exception as e:
            logger.error(f"Pattern correlation failed: {e}")
            return []

    async def _calculate_correlation(self, pattern1: AnomalyPattern, pattern2: AnomalyPattern) -> float:
        """Calculate correlation score between two patterns"""
        try:
            score = 0.0

            # Temporal correlation
            time_diff = abs((pattern1.first_seen - pattern2.first_seen).total_seconds())
            if time_diff < self.temporal_window_minutes * 60:
                score += 0.3

            # Signature similarity
            if self._calculate_signature_similarity(pattern1.signature, pattern2.signature) > 0.5:
                score += 0.3

            # Indicator overlap
            common_indicators = set(pattern1.indicators) & set(pattern2.indicators)
            if common_indicators:
                score += 0.2 * (len(common_indicators) / max(len(pattern1.indicators), len(pattern2.indicators)))

            # Attack vector correlation
            if pattern1.attack_vector == pattern2.attack_vector and pattern1.attack_vector != AttackVector.UNKNOWN:
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return 0.0

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures"""
        try:
            # Simple Jaccard similarity on character bigrams
            bigrams1 = set(sig1[i:i+2] for i in range(len(sig1)-1))
            bigrams2 = set(sig2[i:i+2] for i in range(len(sig2)-1))

            if not bigrams1 and not bigrams2:
                return 1.0
            if not bigrams1 or not bigrams2:
                return 0.0

            intersection = len(bigrams1 & bigrams2)
            union = len(bigrams1 | bigrams2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0


class VulnerabilityAssessmentEngine:
    """Assesses the vulnerability potential of correlated patterns"""

    def __init__(self):
        self.known_cves = {}
        self.exploit_patterns = {}

    async def initialize(self):
        """Initialize vulnerability assessment engine"""
        logger.info("Initializing Vulnerability Assessment Engine")
        await self._load_cve_database()
        await self._load_exploit_patterns()

    async def _load_cve_database(self):
        """Load known CVE database for comparison"""
        # Simulate loading CVE database
        self.known_cves = {
            "CVE-2023-1234": {
                "description": "Buffer overflow in web server",
                "cvss_score": 9.8,
                "attack_patterns": ["buffer_overflow", "remote_execution"]
            },
            "CVE-2023-5678": {
                "description": "SQL injection in database driver",
                "cvss_score": 8.5,
                "attack_patterns": ["sql_injection", "data_exfiltration"]
            }
        }

    async def _load_exploit_patterns(self):
        """Load known exploit patterns"""
        self.exploit_patterns = {
            "buffer_overflow": {
                "indicators": ["repeated_chars", "shellcode", "nop_sled"],
                "exploitability": 0.8,
                "typical_impact": 0.9
            },
            "sql_injection": {
                "indicators": ["sql_keywords", "union_select", "comment_strings"],
                "exploitability": 0.7,
                "typical_impact": 0.6
            }
        }

    async def assess_zero_day_potential(self, pattern_group: List[AnomalyPattern]) -> Optional[ZeroDayCandidate]:
        """Assess if a pattern group represents a potential zero-day"""
        try:
            if not pattern_group:
                return None

            # Calculate novelty score
            novelty_score = await self._calculate_novelty_score(pattern_group)
            if novelty_score < 0.6:  # Not novel enough
                return None

            # Calculate exploitability score
            exploitability_score = await self._calculate_exploitability_score(pattern_group)

            # Calculate impact score
            impact_score = await self._calculate_impact_score(pattern_group)

            # Generate candidate
            candidate_id = f"zd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(pattern_group).encode()).hexdigest()[:8]}"

            candidate = ZeroDayCandidate(
                candidate_id=candidate_id,
                pattern=pattern_group[0],  # Primary pattern
                exploitability_score=exploitability_score,
                impact_score=impact_score,
                novelty_score=novelty_score,
                verification_status="pending",
                discovered_at=datetime.now(),
                affected_systems=await self._identify_affected_systems(pattern_group),
                mitigation_suggestions=await self._generate_mitigation_suggestions(pattern_group)
            )

            return candidate

        except Exception as e:
            logger.error(f"Zero-day assessment failed: {e}")
            return None

    async def _calculate_novelty_score(self, pattern_group: List[AnomalyPattern]) -> float:
        """Calculate how novel the pattern group is"""
        try:
            total_novelty = 0.0

            for pattern in pattern_group:
                # Check against known attack patterns
                pattern_novelty = 1.0

                for known_pattern, details in self.exploit_patterns.items():
                    overlap = len(set(pattern.indicators) & set(details["indicators"]))
                    if overlap > 0:
                        similarity = overlap / len(details["indicators"])
                        pattern_novelty = min(pattern_novelty, 1.0 - similarity)

                total_novelty += pattern_novelty

            return total_novelty / len(pattern_group)

        except Exception as e:
            logger.error(f"Novelty score calculation failed: {e}")
            return 0.0

    async def _calculate_exploitability_score(self, pattern_group: List[AnomalyPattern]) -> float:
        """Calculate exploitability score"""
        try:
            max_exploitability = 0.0

            for pattern in pattern_group:
                # Base exploitability on pattern characteristics
                exploitability = 0.5  # Base score

                # High entropy suggests obfuscation (more sophisticated)
                if "high_entropy_payload" in pattern.indicators:
                    exploitability += 0.2

                # Protocol violations suggest deep technical knowledge
                if "protocol_violation" in pattern.indicators:
                    exploitability += 0.3

                # Novel encoding suggests evasion capabilities
                if "novel_encoding_technique" in pattern.indicators:
                    exploitability += 0.2

                max_exploitability = max(max_exploitability, min(exploitability, 1.0))

            return max_exploitability

        except Exception as e:
            logger.error(f"Exploitability score calculation failed: {e}")
            return 0.5

    async def _calculate_impact_score(self, pattern_group: List[AnomalyPattern]) -> float:
        """Calculate potential impact score"""
        try:
            max_impact = 0.0

            for pattern in pattern_group:
                impact = 0.5  # Base impact

                # High threat level indicates higher impact
                if pattern.threat_level == ThreatLevel.CRITICAL:
                    impact = 0.9
                elif pattern.threat_level == ThreatLevel.HIGH:
                    impact = 0.7
                elif pattern.threat_level == ThreatLevel.MEDIUM:
                    impact = 0.5

                max_impact = max(max_impact, impact)

            return max_impact

        except Exception as e:
            logger.error(f"Impact score calculation failed: {e}")
            return 0.5

    async def _identify_affected_systems(self, pattern_group: List[AnomalyPattern]) -> List[str]:
        """Identify systems potentially affected by the zero-day"""
        affected_systems = set()

        for pattern in pattern_group:
            # Extract system information from metadata
            metadata = pattern.metadata
            if "target_host" in metadata:
                affected_systems.add(metadata["target_host"])
            if "service_type" in metadata:
                affected_systems.add(f"service_{metadata['service_type']}")

        return list(affected_systems)

    async def _generate_mitigation_suggestions(self, pattern_group: List[AnomalyPattern]) -> List[str]:
        """Generate mitigation suggestions based on patterns"""
        suggestions = []

        # Generic suggestions
        suggestions.extend([
            "Implement comprehensive monitoring for unusual traffic patterns",
            "Deploy advanced intrusion detection systems",
            "Conduct immediate security assessment of affected systems",
            "Review and update security policies and procedures"
        ])

        # Pattern-specific suggestions
        for pattern in pattern_group:
            if "high_entropy_payload" in pattern.indicators:
                suggestions.append("Implement payload entropy analysis and blocking")
            if "protocol_violation" in pattern.indicators:
                suggestions.append("Enforce strict protocol compliance validation")
            if "novel_encoding_technique" in pattern.indicators:
                suggestions.append("Deploy multi-layer encoding detection and normalization")

        return list(set(suggestions))  # Remove duplicates


# Global instance
_zero_day_engine: Optional[ZeroDayDetectionEngine] = None

async def get_zero_day_engine() -> ZeroDayDetectionEngine:
    """Get global zero-day detection engine instance"""
    global _zero_day_engine

    if _zero_day_engine is None:
        _zero_day_engine = ZeroDayDetectionEngine()
        await _zero_day_engine.initialize()

    return _zero_day_engine
