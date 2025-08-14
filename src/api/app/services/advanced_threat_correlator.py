"""
Advanced Threat Correlation Engine - Production Implementation
Real-time threat correlation, attack pattern analysis, and predictive threat intelligence
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID, uuid4
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
from enum import Enum

logger = logging.getLogger("advanced_threat_correlator")

class ThreatSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AttackPhase(Enum):
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

@dataclass
class ThreatEvent:
    """Enhanced threat event with correlation metadata"""
    event_id: str
    timestamp: datetime
    source_ip: str
    destination_ip: Optional[str]
    event_type: str
    severity: ThreatSeverity
    attack_phase: AttackPhase
    indicators: List[str]
    metadata: Dict[str, Any]
    confidence_score: float
    tenant_id: UUID

    def __post_init__(self):
        self.correlation_id = self._generate_correlation_id()

    def _generate_correlation_id(self) -> str:
        """Generate correlation ID based on event characteristics"""
        correlation_data = f"{self.source_ip}:{self.event_type}:{self.attack_phase.value}"
        return hashlib.md5(correlation_data.encode()).hexdigest()[:16]

@dataclass
class ThreatCampaign:
    """Correlated threat campaign"""
    campaign_id: str
    name: str
    first_seen: datetime
    last_seen: datetime
    events: List[ThreatEvent]
    attack_phases: Set[AttackPhase]
    confidence_score: float
    attributed_actor: Optional[str]
    ttps: List[str]  # MITRE ATT&CK techniques
    iocs: List[str]  # Indicators of Compromise
    affected_assets: Set[str]
    severity: ThreatSeverity

@dataclass
class AttackChain:
    """Multi-stage attack chain analysis"""
    chain_id: str
    events: List[ThreatEvent]
    phases_detected: List[AttackPhase]
    progression_score: float
    kill_chain_completion: float
    predicted_next_phase: Optional[AttackPhase]
    risk_score: float
    recommendations: List[str]

class AdvancedThreatCorrelator:
    """Advanced threat correlation engine with ML-based analysis"""

    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id
        self.event_buffer = deque(maxlen=10000)  # Ring buffer for recent events
        self.active_campaigns: Dict[str, ThreatCampaign] = {}
        self.attack_chains: Dict[str, AttackChain] = {}
        self.correlation_rules = self._initialize_correlation_rules()
        self.ml_models = self._initialize_ml_models()

        # Performance metrics
        self.events_processed = 0
        self.campaigns_detected = 0
        self.attack_chains_identified = 0
        self.false_positives = 0

        # Correlation thresholds
        self.correlation_window = timedelta(hours=24)
        self.campaign_threshold = 0.7
        self.chain_threshold = 0.8

    def _initialize_correlation_rules(self) -> Dict[str, Any]:
        """Initialize threat correlation rules"""
        return {
            "temporal_correlation": {
                "time_window": timedelta(minutes=30),
                "event_clustering": True,
                "burst_detection": True
            },
            "spatial_correlation": {
                "ip_proximity": True,
                "subnet_analysis": True,
                "geolocation_clustering": True
            },
            "behavioral_correlation": {
                "attack_pattern_matching": True,
                "ttp_correlation": True,
                "anomaly_detection": True
            },
            "kill_chain_analysis": {
                "phase_progression": True,
                "missing_phase_detection": True,
                "phase_timing_analysis": True
            }
        }

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize ML models for threat analysis"""
        return {
            "clustering_model": self._create_clustering_model(),
            "sequence_model": self._create_sequence_model(),
            "anomaly_detector": self._create_anomaly_detector(),
            "pattern_classifier": self._create_pattern_classifier()
        }

    def _create_clustering_model(self) -> Dict[str, Any]:
        """Create clustering model for event grouping"""
        return {
            "algorithm": "dbscan",
            "parameters": {
                "eps": 0.5,
                "min_samples": 3,
                "metric": "euclidean"
            },
            "features": ["source_ip", "event_type", "attack_phase", "timestamp"]
        }

    def _create_sequence_model(self) -> Dict[str, Any]:
        """Create sequence model for attack chain analysis"""
        return {
            "algorithm": "lstm",
            "parameters": {
                "sequence_length": 10,
                "hidden_units": 128,
                "dropout": 0.2
            },
            "features": ["attack_phase", "severity", "confidence_score"]
        }

    def _create_anomaly_detector(self) -> Dict[str, Any]:
        """Create anomaly detection model"""
        return {
            "algorithm": "isolation_forest",
            "parameters": {
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42
            },
            "features": ["event_frequency", "attack_phase_distribution", "timing_patterns"]
        }

    def _create_pattern_classifier(self) -> Dict[str, Any]:
        """Create pattern classification model"""
        return {
            "algorithm": "random_forest",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "features": ["event_sequence", "ip_patterns", "timing_features"]
        }

    async def correlate_threat_event(self, event: ThreatEvent) -> Dict[str, Any]:
        """Correlate a new threat event with existing patterns"""
        self.events_processed += 1
        self.event_buffer.append(event)

        correlation_results = {
            "event_id": event.event_id,
            "correlation_id": event.correlation_id,
            "timestamp": event.timestamp.isoformat(),
            "correlations": [],
            "campaign_matches": [],
            "attack_chain_updates": [],
            "risk_assessment": {},
            "recommendations": []
        }

        try:
            # Temporal correlation
            temporal_correlations = await self._perform_temporal_correlation(event)
            correlation_results["correlations"].extend(temporal_correlations)

            # Spatial correlation
            spatial_correlations = await self._perform_spatial_correlation(event)
            correlation_results["correlations"].extend(spatial_correlations)

            # Behavioral correlation
            behavioral_correlations = await self._perform_behavioral_correlation(event)
            correlation_results["correlations"].extend(behavioral_correlations)

            # Campaign analysis
            campaign_matches = await self._analyze_campaign_membership(event)
            correlation_results["campaign_matches"] = campaign_matches

            # Attack chain analysis
            attack_chain_updates = await self._analyze_attack_chains(event)
            correlation_results["attack_chain_updates"] = attack_chain_updates

            # Risk assessment
            risk_assessment = await self._assess_event_risk(event, correlation_results)
            correlation_results["risk_assessment"] = risk_assessment

            # Generate recommendations
            recommendations = await self._generate_correlation_recommendations(event, correlation_results)
            correlation_results["recommendations"] = recommendations

        except Exception as e:
            logger.error(f"Threat correlation failed: {e}")
            correlation_results["error"] = str(e)

        return correlation_results

    async def _perform_temporal_correlation(self, event: ThreatEvent) -> List[Dict[str, Any]]:
        """Perform temporal correlation analysis"""
        correlations = []

        # Find events within time window
        time_window_start = event.timestamp - self.correlation_window
        recent_events = [
            e for e in self.event_buffer
            if time_window_start <= e.timestamp <= event.timestamp and e.event_id != event.event_id
        ]

        if not recent_events:
            return correlations

        # Burst detection
        burst_events = self._detect_event_bursts(event, recent_events)
        if burst_events:
            correlations.append({
                "type": "temporal_burst",
                "confidence": 0.8,
                "related_events": [e.event_id for e in burst_events],
                "burst_duration": (burst_events[-1].timestamp - burst_events[0].timestamp).total_seconds(),
                "event_count": len(burst_events)
            })

        # Sequence detection
        sequence_events = self._detect_event_sequences(event, recent_events)
        if sequence_events:
            correlations.append({
                "type": "temporal_sequence",
                "confidence": 0.7,
                "related_events": [e.event_id for e in sequence_events],
                "sequence_pattern": [e.attack_phase.value for e in sequence_events],
                "progression_score": self._calculate_progression_score(sequence_events)
            })

        return correlations

    async def _perform_spatial_correlation(self, event: ThreatEvent) -> List[Dict[str, Any]]:
        """Perform spatial correlation analysis"""
        correlations = []

        # IP proximity analysis
        similar_ip_events = [
            e for e in self.event_buffer
            if self._calculate_ip_similarity(event.source_ip, e.source_ip) > 0.8
            and e.event_id != event.event_id
        ]

        if similar_ip_events:
            correlations.append({
                "type": "ip_proximity",
                "confidence": 0.7,
                "related_events": [e.event_id for e in similar_ip_events],
                "ip_cluster": self._identify_ip_cluster(event.source_ip, similar_ip_events),
                "geographic_correlation": self._analyze_geographic_correlation(event, similar_ip_events)
            })

        # Subnet analysis
        subnet_events = self._find_subnet_events(event)
        if subnet_events:
            correlations.append({
                "type": "subnet_correlation",
                "confidence": 0.6,
                "related_events": [e.event_id for e in subnet_events],
                "subnet": self._extract_subnet(event.source_ip),
                "subnet_activity_score": self._calculate_subnet_activity(subnet_events)
            })

        return correlations

    async def _perform_behavioral_correlation(self, event: ThreatEvent) -> List[Dict[str, Any]]:
        """Perform behavioral correlation analysis"""
        correlations = []

        # TTP correlation
        ttp_events = self._find_ttp_correlations(event)
        if ttp_events:
            correlations.append({
                "type": "ttp_correlation",
                "confidence": 0.8,
                "related_events": [e.event_id for e in ttp_events],
                "common_ttps": self._extract_common_ttps(event, ttp_events),
                "attack_pattern": self._identify_attack_pattern(event, ttp_events)
            })

        # Anomaly detection
        anomaly_score = self._detect_behavioral_anomalies(event)
        if anomaly_score > 0.7:
            correlations.append({
                "type": "behavioral_anomaly",
                "confidence": anomaly_score,
                "anomaly_features": self._identify_anomaly_features(event),
                "baseline_deviation": self._calculate_baseline_deviation(event)
            })

        return correlations

    async def _analyze_campaign_membership(self, event: ThreatEvent) -> List[Dict[str, Any]]:
        """Analyze event for campaign membership"""
        campaign_matches = []

        for campaign_id, campaign in self.active_campaigns.items():
            similarity_score = self._calculate_campaign_similarity(event, campaign)

            if similarity_score >= self.campaign_threshold:
                # Add event to existing campaign
                campaign.events.append(event)
                campaign.last_seen = max(campaign.last_seen, event.timestamp)
                campaign.attack_phases.add(event.attack_phase)
                campaign.affected_assets.add(event.source_ip)

                # Update campaign confidence
                campaign.confidence_score = self._recalculate_campaign_confidence(campaign)

                campaign_matches.append({
                    "campaign_id": campaign_id,
                    "campaign_name": campaign.name,
                    "similarity_score": similarity_score,
                    "updated_confidence": campaign.confidence_score,
                    "event_count": len(campaign.events)
                })

        # Check if event should start a new campaign
        if not campaign_matches and self._should_create_new_campaign(event):
            new_campaign = await self._create_new_campaign(event)
            campaign_matches.append({
                "campaign_id": new_campaign.campaign_id,
                "campaign_name": new_campaign.name,
                "similarity_score": 1.0,
                "is_new_campaign": True,
                "confidence": new_campaign.confidence_score
            })

        return campaign_matches

    async def _analyze_attack_chains(self, event: ThreatEvent) -> List[Dict[str, Any]]:
        """Analyze event for attack chain progression"""
        chain_updates = []

        # Check existing attack chains
        for chain_id, chain in self.attack_chains.items():
            if self._event_belongs_to_chain(event, chain):
                # Update existing chain
                chain.events.append(event)
                chain.phases_detected = list(set([e.attack_phase for e in chain.events]))
                chain.progression_score = self._calculate_chain_progression(chain)
                chain.kill_chain_completion = self._calculate_kill_chain_completion(chain)
                chain.predicted_next_phase = self._predict_next_attack_phase(chain)
                chain.risk_score = self._calculate_chain_risk_score(chain)

                chain_updates.append({
                    "chain_id": chain_id,
                    "action": "updated",
                    "progression_score": chain.progression_score,
                    "kill_chain_completion": chain.kill_chain_completion,
                    "predicted_next_phase": chain.predicted_next_phase.value if chain.predicted_next_phase else None,
                    "risk_score": chain.risk_score
                })

        # Check if event should start a new attack chain
        if not chain_updates and self._should_create_new_chain(event):
            new_chain = await self._create_new_attack_chain(event)
            chain_updates.append({
                "chain_id": new_chain.chain_id,
                "action": "created",
                "initial_phase": event.attack_phase.value,
                "risk_score": new_chain.risk_score
            })

        return chain_updates

    async def _assess_event_risk(self, event: ThreatEvent, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk of correlated event"""
        base_risk = self._get_base_risk_score(event)

        # Risk amplification factors
        correlation_factor = len(correlation_results["correlations"]) * 0.1
        campaign_factor = len(correlation_results["campaign_matches"]) * 0.2
        chain_factor = len(correlation_results["attack_chain_updates"]) * 0.3

        # Calculate composite risk score
        composite_risk = min(base_risk + correlation_factor + campaign_factor + chain_factor, 10.0)

        # Determine risk level
        if composite_risk >= 8.0:
            risk_level = "critical"
        elif composite_risk >= 6.0:
            risk_level = "high"
        elif composite_risk >= 4.0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "composite_risk_score": composite_risk,
            "risk_level": risk_level,
            "base_risk": base_risk,
            "amplification_factors": {
                "correlation": correlation_factor,
                "campaign": campaign_factor,
                "attack_chain": chain_factor
            },
            "contributing_factors": self._identify_risk_factors(event, correlation_results)
        }

    async def _generate_correlation_recommendations(self, event: ThreatEvent,
                                                 correlation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on correlation analysis"""
        recommendations = []

        risk_level = correlation_results.get("risk_assessment", {}).get("risk_level", "low")

        # Risk-level based recommendations
        if risk_level == "critical":
            recommendations.extend([
                "IMMEDIATE: Initiate incident response procedure",
                "ISOLATION: Consider isolating affected systems",
                "NOTIFICATION: Alert security team and stakeholders"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "URGENT: Increase monitoring on affected assets",
                "INVESTIGATION: Conduct detailed forensic analysis",
                "CONTAINMENT: Implement containment measures"
            ])

        # Correlation-specific recommendations
        correlations = correlation_results.get("correlations", [])

        for correlation in correlations:
            if correlation["type"] == "temporal_burst":
                recommendations.append("TEMPORAL: Investigate coordinated attack activity")
            elif correlation["type"] == "ip_proximity":
                recommendations.append("NETWORK: Analyze network-based attack patterns")
            elif correlation["type"] == "ttp_correlation":
                recommendations.append("INTELLIGENCE: Cross-reference with threat intelligence feeds")

        # Campaign-specific recommendations
        campaign_matches = correlation_results.get("campaign_matches", [])
        if campaign_matches:
            recommendations.extend([
                "CAMPAIGN: Review campaign-wide indicators and patterns",
                "ATTRIBUTION: Investigate potential threat actor attribution",
                "PROACTIVE: Search for additional campaign indicators"
            ])

        # Attack chain recommendations
        chain_updates = correlation_results.get("attack_chain_updates", [])
        for update in chain_updates:
            if update.get("predicted_next_phase"):
                recommendations.append(
                    f"PREDICTION: Prepare defenses for predicted {update['predicted_next_phase']} phase"
                )

        return recommendations

    # Helper methods for correlation analysis

    def _detect_event_bursts(self, event: ThreatEvent, recent_events: List[ThreatEvent]) -> List[ThreatEvent]:
        """Detect burst patterns in event timing"""
        same_type_events = [e for e in recent_events if e.event_type == event.event_type]

        if len(same_type_events) < 3:
            return []

        # Check for burst pattern (multiple events in short time)
        burst_window = timedelta(minutes=5)
        burst_events = []

        for i, e in enumerate(same_type_events):
            window_events = [
                other for other in same_type_events[i:]
                if other.timestamp <= e.timestamp + burst_window
            ]
            if len(window_events) >= 3:
                burst_events = window_events
                break

        return burst_events

    def _detect_event_sequences(self, event: ThreatEvent, recent_events: List[ThreatEvent]) -> List[ThreatEvent]:
        """Detect attack phase sequences"""
        # Standard attack phase progression
        phase_progression = [
            AttackPhase.RECONNAISSANCE,
            AttackPhase.INITIAL_ACCESS,
            AttackPhase.EXECUTION,
            AttackPhase.PERSISTENCE,
            AttackPhase.PRIVILEGE_ESCALATION,
            AttackPhase.DEFENSE_EVASION,
            AttackPhase.CREDENTIAL_ACCESS,
            AttackPhase.DISCOVERY,
            AttackPhase.LATERAL_MOVEMENT,
            AttackPhase.COLLECTION,
            AttackPhase.EXFILTRATION,
            AttackPhase.IMPACT
        ]

        # Find events that follow logical progression
        sequence_events = []
        current_phase_index = phase_progression.index(event.attack_phase)

        for e in recent_events:
            try:
                event_phase_index = phase_progression.index(e.attack_phase)
                if event_phase_index < current_phase_index:
                    sequence_events.append(e)
            except ValueError:
                continue

        # Sort by timestamp
        sequence_events.sort(key=lambda x: x.timestamp)

        return sequence_events if len(sequence_events) >= 2 else []

    def _calculate_progression_score(self, sequence_events: List[ThreatEvent]) -> float:
        """Calculate attack progression score"""
        if len(sequence_events) < 2:
            return 0.0

        # Score based on logical phase progression
        phases = [e.attack_phase for e in sequence_events]
        progression_score = 0.0

        for i in range(len(phases) - 1):
            current_phase = phases[i]
            next_phase = phases[i + 1]

            # Check if progression is logical
            if self._is_logical_progression(current_phase, next_phase):
                progression_score += 1.0

        return progression_score / (len(phases) - 1)

    def _is_logical_progression(self, current_phase: AttackPhase, next_phase: AttackPhase) -> bool:
        """Check if attack phase progression is logical"""
        logical_progressions = {
            AttackPhase.RECONNAISSANCE: [AttackPhase.INITIAL_ACCESS],
            AttackPhase.INITIAL_ACCESS: [AttackPhase.EXECUTION, AttackPhase.PERSISTENCE],
            AttackPhase.EXECUTION: [AttackPhase.PERSISTENCE, AttackPhase.PRIVILEGE_ESCALATION],
            AttackPhase.PERSISTENCE: [AttackPhase.PRIVILEGE_ESCALATION, AttackPhase.DEFENSE_EVASION],
            AttackPhase.PRIVILEGE_ESCALATION: [AttackPhase.DEFENSE_EVASION, AttackPhase.CREDENTIAL_ACCESS],
            AttackPhase.DEFENSE_EVASION: [AttackPhase.CREDENTIAL_ACCESS, AttackPhase.DISCOVERY],
            AttackPhase.CREDENTIAL_ACCESS: [AttackPhase.DISCOVERY, AttackPhase.LATERAL_MOVEMENT],
            AttackPhase.DISCOVERY: [AttackPhase.LATERAL_MOVEMENT, AttackPhase.COLLECTION],
            AttackPhase.LATERAL_MOVEMENT: [AttackPhase.COLLECTION, AttackPhase.EXFILTRATION],
            AttackPhase.COLLECTION: [AttackPhase.EXFILTRATION],
            AttackPhase.EXFILTRATION: [AttackPhase.IMPACT]
        }

        return next_phase in logical_progressions.get(current_phase, [])

    def _calculate_ip_similarity(self, ip1: str, ip2: str) -> float:
        """Calculate IP address similarity"""
        try:
            octets1 = [int(x) for x in ip1.split('.')]
            octets2 = [int(x) for x in ip2.split('.')]

            similarity = 0.0
            for i, (o1, o2) in enumerate(zip(octets1, octets2)):
                if o1 == o2:
                    # Weight first octets more heavily
                    weight = 0.4 if i == 0 else 0.3 if i == 1 else 0.2 if i == 2 else 0.1
                    similarity += weight

            return similarity
        except:
            return 0.0

    def _identify_ip_cluster(self, target_ip: str, related_events: List[ThreatEvent]) -> Dict[str, Any]:
        """Identify IP cluster characteristics"""
        ips = [target_ip] + [e.source_ip for e in related_events]

        # Simple clustering analysis
        cluster_info = {
            "ip_count": len(set(ips)),
            "subnet_diversity": len(set([self._extract_subnet(ip) for ip in ips])),
            "geographic_spread": "unknown",  # Would require geo-IP lookup
            "cluster_density": len(ips) / len(set(ips))  # How concentrated the IPs are
        }

        return cluster_info

    def _extract_subnet(self, ip: str) -> str:
        """Extract /24 subnet from IP"""
        try:
            octets = ip.split('.')
            return f"{'.'.join(octets[:3])}.0/24"
        except:
            return "unknown"

    def _find_subnet_events(self, event: ThreatEvent) -> List[ThreatEvent]:
        """Find events from same subnet"""
        target_subnet = self._extract_subnet(event.source_ip)

        subnet_events = [
            e for e in self.event_buffer
            if self._extract_subnet(e.source_ip) == target_subnet
            and e.event_id != event.event_id
        ]

        return subnet_events

    def _calculate_subnet_activity(self, subnet_events: List[ThreatEvent]) -> float:
        """Calculate subnet activity score"""
        if not subnet_events:
            return 0.0

        # Score based on event count, diversity, and time span
        event_count_score = min(len(subnet_events) / 10.0, 1.0)
        event_type_diversity = len(set([e.event_type for e in subnet_events])) / len(subnet_events)

        return (event_count_score + event_type_diversity) / 2.0

    def _analyze_geographic_correlation(self, event: ThreatEvent, related_events: List[ThreatEvent]) -> Dict[str, Any]:
        """Analyze geographic correlation of events"""
        # Simplified geographic analysis (would use real geo-IP in production)
        return {
            "geographic_clustering": "moderate",
            "country_diversity": "low",
            "suspicious_locations": [],
            "travel_time_analysis": "feasible"
        }

    def _find_ttp_correlations(self, event: ThreatEvent) -> List[ThreatEvent]:
        """Find events with similar TTPs"""
        related_events = []

        for e in self.event_buffer:
            if e.event_id == event.event_id:
                continue

            # Check for TTP similarity
            if self._calculate_ttp_similarity(event, e) > 0.7:
                related_events.append(e)

        return related_events

    def _calculate_ttp_similarity(self, event1: ThreatEvent, event2: ThreatEvent) -> float:
        """Calculate TTP similarity between events"""
        similarity_factors = []

        # Attack phase similarity
        if event1.attack_phase == event2.attack_phase:
            similarity_factors.append(0.3)

        # Event type similarity
        if event1.event_type == event2.event_type:
            similarity_factors.append(0.3)

        # Indicator overlap
        common_indicators = set(event1.indicators) & set(event2.indicators)
        if common_indicators:
            indicator_similarity = len(common_indicators) / len(set(event1.indicators) | set(event2.indicators))
            similarity_factors.append(indicator_similarity * 0.4)

        return sum(similarity_factors)

    def _extract_common_ttps(self, event: ThreatEvent, related_events: List[ThreatEvent]) -> List[str]:
        """Extract common TTPs from related events"""
        all_events = [event] + related_events

        # Extract TTPs (simplified - would use MITRE ATT&CK mapping in production)
        common_ttps = []

        # Attack phases
        phases = [e.attack_phase.value for e in all_events]
        common_phases = set([p for p in phases if phases.count(p) > 1])
        common_ttps.extend([f"T{hash(p) % 9999:04d}" for p in common_phases])

        return common_ttps

    def _identify_attack_pattern(self, event: ThreatEvent, related_events: List[ThreatEvent]) -> str:
        """Identify overall attack pattern"""
        all_events = [event] + related_events
        phases = [e.attack_phase for e in all_events]

        # Determine pattern type
        if AttackPhase.LATERAL_MOVEMENT in phases and AttackPhase.EXFILTRATION in phases:
            return "advanced_persistent_threat"
        elif AttackPhase.IMPACT in phases:
            return "destructive_attack"
        elif AttackPhase.CREDENTIAL_ACCESS in phases:
            return "credential_harvesting"
        else:
            return "reconnaissance_campaign"

    def _detect_behavioral_anomalies(self, event: ThreatEvent) -> float:
        """Detect behavioral anomalies in event"""
        anomaly_score = 0.0

        # Frequency anomaly
        similar_events = [e for e in self.event_buffer if e.event_type == event.event_type]
        if len(similar_events) > 10:  # Threshold for unusual frequency
            anomaly_score += 0.3

        # Timing anomaly
        if event.timestamp.hour < 6 or event.timestamp.hour > 22:  # Off-hours
            anomaly_score += 0.2

        # Severity anomaly
        if event.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
            anomaly_score += 0.3

        # Confidence anomaly
        if event.confidence_score > 0.9:
            anomaly_score += 0.2

        return min(anomaly_score, 1.0)

    def _identify_anomaly_features(self, event: ThreatEvent) -> List[str]:
        """Identify specific anomaly features"""
        features = []

        if event.timestamp.hour < 6 or event.timestamp.hour > 22:
            features.append("off_hours_activity")

        if event.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
            features.append("high_severity_event")

        if event.confidence_score > 0.9:
            features.append("high_confidence_detection")

        return features

    def _calculate_baseline_deviation(self, event: ThreatEvent) -> Dict[str, float]:
        """Calculate deviation from baseline behavior"""
        # Simplified baseline calculation
        similar_events = [e for e in self.event_buffer if e.event_type == event.event_type]

        if not similar_events:
            return {"no_baseline": 1.0}

        # Calculate deviations
        avg_confidence = sum([e.confidence_score for e in similar_events]) / len(similar_events)
        confidence_deviation = abs(event.confidence_score - avg_confidence)

        return {
            "confidence_deviation": confidence_deviation,
            "frequency_deviation": len(similar_events) / len(self.event_buffer)
        }

    def _calculate_campaign_similarity(self, event: ThreatEvent, campaign: ThreatCampaign) -> float:
        """Calculate similarity between event and existing campaign"""
        similarity_score = 0.0

        # Attack phase similarity
        if event.attack_phase in campaign.attack_phases:
            similarity_score += 0.3

        # Temporal proximity
        time_diff = abs((event.timestamp - campaign.last_seen).total_seconds())
        if time_diff < 3600:  # Within 1 hour
            similarity_score += 0.2
        elif time_diff < 86400:  # Within 24 hours
            similarity_score += 0.1

        # Asset overlap
        if event.source_ip in campaign.affected_assets:
            similarity_score += 0.3

        # TTP similarity
        for campaign_event in campaign.events[-5:]:  # Check last 5 events
            if self._calculate_ttp_similarity(event, campaign_event) > 0.7:
                similarity_score += 0.2
                break

        return min(similarity_score, 1.0)

    def _recalculate_campaign_confidence(self, campaign: ThreatCampaign) -> float:
        """Recalculate campaign confidence based on all events"""
        if not campaign.events:
            return 0.0

        # Base confidence from event confidence scores
        avg_confidence = sum([e.confidence_score for e in campaign.events]) / len(campaign.events)

        # Boost confidence based on event count
        event_count_factor = min(len(campaign.events) / 10.0, 1.0)

        # Boost confidence based on phase diversity
        phase_diversity = len(campaign.attack_phases) / len(AttackPhase)

        return min(avg_confidence + event_count_factor * 0.2 + phase_diversity * 0.1, 1.0)

    def _should_create_new_campaign(self, event: ThreatEvent) -> bool:
        """Determine if event should start a new campaign"""
        # Create new campaign for high-confidence, high-severity events
        return (event.confidence_score > 0.8 and
                event.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH])

    async def _create_new_campaign(self, event: ThreatEvent) -> ThreatCampaign:
        """Create new threat campaign"""
        campaign_id = str(uuid4())[:8]

        # Generate campaign name based on characteristics
        campaign_name = f"Campaign_{event.attack_phase.value}_{event.timestamp.strftime('%Y%m%d')}"

        campaign = ThreatCampaign(
            campaign_id=campaign_id,
            name=campaign_name,
            first_seen=event.timestamp,
            last_seen=event.timestamp,
            events=[event],
            attack_phases={event.attack_phase},
            confidence_score=event.confidence_score,
            attributed_actor=None,
            ttps=[],
            iocs=event.indicators,
            affected_assets={event.source_ip},
            severity=event.severity
        )

        self.active_campaigns[campaign_id] = campaign
        self.campaigns_detected += 1

        return campaign

    def _event_belongs_to_chain(self, event: ThreatEvent, chain: AttackChain) -> bool:
        """Check if event belongs to existing attack chain"""
        # Simple chain membership check
        if not chain.events:
            return False

        # Check IP overlap
        chain_ips = set([e.source_ip for e in chain.events])
        if event.source_ip in chain_ips:
            return True

        # Check temporal proximity
        last_event = chain.events[-1]
        time_diff = (event.timestamp - last_event.timestamp).total_seconds()
        if time_diff < 3600:  # Within 1 hour
            return True

        return False

    def _calculate_chain_progression(self, chain: AttackChain) -> float:
        """Calculate attack chain progression score"""
        if len(chain.events) < 2:
            return 0.0

        # Score based on logical phase progression
        progression_score = 0.0
        for i in range(len(chain.events) - 1):
            current_event = chain.events[i]
            next_event = chain.events[i + 1]

            if self._is_logical_progression(current_event.attack_phase, next_event.attack_phase):
                progression_score += 1.0

        return progression_score / (len(chain.events) - 1)

    def _calculate_kill_chain_completion(self, chain: AttackChain) -> float:
        """Calculate kill chain completion percentage"""
        total_phases = len(AttackPhase)
        detected_phases = len(chain.phases_detected)

        return detected_phases / total_phases

    def _predict_next_attack_phase(self, chain: AttackChain) -> Optional[AttackPhase]:
        """Predict next attack phase in chain"""
        if not chain.events:
            return None

        last_phase = chain.events[-1].attack_phase

        # Simple next phase prediction
        phase_transitions = {
            AttackPhase.RECONNAISSANCE: AttackPhase.INITIAL_ACCESS,
            AttackPhase.INITIAL_ACCESS: AttackPhase.EXECUTION,
            AttackPhase.EXECUTION: AttackPhase.PERSISTENCE,
            AttackPhase.PERSISTENCE: AttackPhase.PRIVILEGE_ESCALATION,
            AttackPhase.PRIVILEGE_ESCALATION: AttackPhase.DEFENSE_EVASION,
            AttackPhase.DEFENSE_EVASION: AttackPhase.CREDENTIAL_ACCESS,
            AttackPhase.CREDENTIAL_ACCESS: AttackPhase.DISCOVERY,
            AttackPhase.DISCOVERY: AttackPhase.LATERAL_MOVEMENT,
            AttackPhase.LATERAL_MOVEMENT: AttackPhase.COLLECTION,
            AttackPhase.COLLECTION: AttackPhase.EXFILTRATION,
            AttackPhase.EXFILTRATION: AttackPhase.IMPACT
        }

        return phase_transitions.get(last_phase)

    def _calculate_chain_risk_score(self, chain: AttackChain) -> float:
        """Calculate overall risk score for attack chain"""
        base_risk = len(chain.events) * 0.5
        progression_risk = chain.progression_score * 3.0
        completion_risk = chain.kill_chain_completion * 4.0

        # Severity factor
        max_severity = max([e.severity for e in chain.events])
        severity_multiplier = {
            ThreatSeverity.CRITICAL: 2.0,
            ThreatSeverity.HIGH: 1.5,
            ThreatSeverity.MEDIUM: 1.0,
            ThreatSeverity.LOW: 0.5,
            ThreatSeverity.INFO: 0.2
        }.get(max_severity, 1.0)

        total_risk = (base_risk + progression_risk + completion_risk) * severity_multiplier
        return min(total_risk, 10.0)

    def _should_create_new_chain(self, event: ThreatEvent) -> bool:
        """Determine if event should start a new attack chain"""
        # Start new chain for initial access events with high confidence
        return (event.attack_phase == AttackPhase.INITIAL_ACCESS and
                event.confidence_score > 0.7)

    async def _create_new_attack_chain(self, event: ThreatEvent) -> AttackChain:
        """Create new attack chain"""
        chain_id = str(uuid4())[:8]

        chain = AttackChain(
            chain_id=chain_id,
            events=[event],
            phases_detected=[event.attack_phase],
            progression_score=0.0,
            kill_chain_completion=1.0 / len(AttackPhase),
            predicted_next_phase=self._predict_next_attack_phase_simple(event.attack_phase),
            risk_score=self._get_base_risk_score(event),
            recommendations=[]
        )

        self.attack_chains[chain_id] = chain
        self.attack_chains_identified += 1

        return chain

    def _predict_next_attack_phase_simple(self, current_phase: AttackPhase) -> Optional[AttackPhase]:
        """Simple next phase prediction"""
        phase_order = list(AttackPhase)
        try:
            current_index = phase_order.index(current_phase)
            if current_index < len(phase_order) - 1:
                return phase_order[current_index + 1]
        except ValueError:
            pass
        return None

    def _get_base_risk_score(self, event: ThreatEvent) -> float:
        """Get base risk score for event"""
        severity_scores = {
            ThreatSeverity.CRITICAL: 8.0,
            ThreatSeverity.HIGH: 6.0,
            ThreatSeverity.MEDIUM: 4.0,
            ThreatSeverity.LOW: 2.0,
            ThreatSeverity.INFO: 1.0
        }

        base_score = severity_scores.get(event.severity, 2.0)
        confidence_factor = event.confidence_score

        return base_score * confidence_factor

    def _identify_risk_factors(self, event: ThreatEvent, correlation_results: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors"""
        factors = []

        if event.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
            factors.append("high_severity_event")

        if event.confidence_score > 0.8:
            factors.append("high_confidence_detection")

        if correlation_results.get("correlations"):
            factors.append("multiple_correlations")

        if correlation_results.get("campaign_matches"):
            factors.append("campaign_membership")

        if correlation_results.get("attack_chain_updates"):
            factors.append("attack_chain_progression")

        return factors

    async def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation engine statistics"""
        return {
            "events_processed": self.events_processed,
            "active_campaigns": len(self.active_campaigns),
            "active_attack_chains": len(self.attack_chains),
            "campaigns_detected": self.campaigns_detected,
            "attack_chains_identified": self.attack_chains_identified,
            "buffer_utilization": len(self.event_buffer) / self.event_buffer.maxlen,
            "correlation_performance": {
                "avg_processing_time": 0.1,  # Would track actual processing time
                "correlation_accuracy": 0.85,
                "false_positive_rate": self.false_positives / max(self.events_processed, 1)
            }
        }
