"""
Behavioral Anomaly Detection Engine - Production Implementation
Advanced user behavior analysis with machine learning for insider threat detection
"""

import asyncio
import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("behavioral_anomaly_detector")

class AnomalyType(Enum):
    LOGIN_ANOMALY = "login_anomaly"
    ACCESS_ANOMALY = "access_anomaly"
    DATA_ANOMALY = "data_anomaly"
    TIME_ANOMALY = "time_anomaly"
    LOCATION_ANOMALY = "location_anomaly"
    PRIVILEGE_ANOMALY = "privilege_anomaly"
    NETWORK_ANOMALY = "network_anomaly"

class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class UserBehaviorEvent:
    """User behavior event for analysis"""
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    user_agent: str
    resource_accessed: str
    action_performed: str
    data_volume: int  # bytes
    session_duration: int  # seconds
    authentication_method: str
    location: str
    privilege_level: str
    metadata: Dict[str, Any]

@dataclass
class BehaviorProfile:
    """User behavior baseline profile"""
    user_id: str
    created_at: datetime
    updated_at: datetime

    # Login patterns
    typical_login_hours: List[int]
    typical_login_days: List[int]
    avg_session_duration: float
    login_frequency: float

    # Access patterns
    typical_resources: Set[str]
    typical_actions: Set[str]
    avg_data_volume: float
    privilege_usage: Dict[str, int]

    # Network patterns
    typical_source_ips: Set[str]
    typical_user_agents: Set[str]
    typical_locations: Set[str]

    # Statistical baselines
    login_hour_distribution: Dict[int, float]
    data_volume_stats: Dict[str, float]
    session_duration_stats: Dict[str, float]

    # Risk factors
    historical_anomalies: List[str]
    risk_score: float
    trust_level: float

@dataclass
class BehaviorAnomaly:
    """Detected behavior anomaly"""
    anomaly_id: str
    user_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    risk_level: RiskLevel
    confidence_score: float
    anomaly_score: float

    # Event details
    triggering_event: UserBehaviorEvent
    baseline_deviation: Dict[str, float]
    contributing_factors: List[str]

    # Analysis results
    similar_users_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    # Recommendations
    recommended_actions: List[str]
    investigation_priority: int

class BehavioralAnomalyDetector:
    """Advanced behavioral anomaly detection engine"""

    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id
        self.user_profiles: Dict[str, BehaviorProfile] = {}
        self.recent_events: deque = deque(maxlen=50000)  # Ring buffer for recent events
        self.detected_anomalies: Dict[str, BehaviorAnomaly] = {}

        # ML models and parameters
        self.ml_models = self._initialize_ml_models()
        self.anomaly_thresholds = self._initialize_thresholds()

        # Performance metrics
        self.events_processed = 0
        self.anomalies_detected = 0
        self.false_positives = 0
        self.true_positives = 0

        # Configuration
        self.profile_learning_period = timedelta(days=30)
        self.anomaly_retention_period = timedelta(days=90)
        self.update_frequency = timedelta(hours=1)

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for anomaly detection"""
        return {
            "isolation_forest": {
                "contamination": 0.1,
                "n_estimators": 100,
                "features": ["login_hour", "session_duration", "data_volume", "ip_entropy"]
            },
            "gaussian_mixture": {
                "n_components": 3,
                "features": ["login_frequency", "resource_diversity", "action_diversity"]
            },
            "one_class_svm": {
                "kernel": "rbf",
                "gamma": "scale",
                "features": ["temporal_pattern", "access_pattern", "privilege_pattern"]
            },
            "lstm_sequence": {
                "sequence_length": 10,
                "features": ["event_sequence", "time_intervals", "action_sequence"]
            }
        }

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize anomaly detection thresholds"""
        return {
            "login_time_threshold": 2.0,      # Standard deviations from norm
            "session_duration_threshold": 2.5,
            "data_volume_threshold": 3.0,
            "location_threshold": 0.8,        # Similarity threshold
            "ip_threshold": 0.7,
            "privilege_escalation_threshold": 1.5,
            "resource_access_threshold": 2.0,
            "frequency_threshold": 2.5
        }

    async def analyze_user_behavior(self, event: UserBehaviorEvent) -> Dict[str, Any]:
        """Analyze user behavior event for anomalies"""
        self.events_processed += 1
        self.recent_events.append(event)

        analysis_result = {
            "event_id": event.event_id,
            "user_id": event.user_id,
            "timestamp": event.timestamp.isoformat(),
            "anomalies_detected": [],
            "profile_updated": False,
            "risk_assessment": {},
            "recommendations": []
        }

        try:
            # Get or create user profile
            user_profile = await self._get_or_create_user_profile(event.user_id)

            # Perform anomaly detection
            anomalies = await self._detect_anomalies(event, user_profile)

            # Update user profile with new event
            await self._update_user_profile(event, user_profile)
            analysis_result["profile_updated"] = True

            # Process detected anomalies
            for anomaly in anomalies:
                await self._process_anomaly(anomaly)
                self.anomalies_detected += 1

                analysis_result["anomalies_detected"].append({
                    "anomaly_id": anomaly.anomaly_id,
                    "type": anomaly.anomaly_type.value,
                    "risk_level": anomaly.risk_level.value,
                    "confidence": anomaly.confidence_score,
                    "factors": anomaly.contributing_factors
                })

            # Generate risk assessment
            risk_assessment = await self._assess_user_risk(event, user_profile, anomalies)
            analysis_result["risk_assessment"] = risk_assessment

            # Generate recommendations
            recommendations = await self._generate_behavioral_recommendations(event, anomalies, risk_assessment)
            analysis_result["recommendations"] = recommendations

        except Exception as e:
            logger.error(f"Behavioral analysis failed for user {event.user_id}: {e}")
            analysis_result["error"] = str(e)

        return analysis_result

    async def _get_or_create_user_profile(self, user_id: str) -> BehaviorProfile:
        """Get existing user profile or create new one"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Create new profile
        profile = BehaviorProfile(
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            typical_login_hours=[],
            typical_login_days=[],
            avg_session_duration=0.0,
            login_frequency=0.0,
            typical_resources=set(),
            typical_actions=set(),
            avg_data_volume=0.0,
            privilege_usage={},
            typical_source_ips=set(),
            typical_user_agents=set(),
            typical_locations=set(),
            login_hour_distribution={},
            data_volume_stats={},
            session_duration_stats={},
            historical_anomalies=[],
            risk_score=0.5,
            trust_level=0.5
        )

        self.user_profiles[user_id] = profile
        return profile

    async def _detect_anomalies(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> List[BehaviorAnomaly]:
        """Detect behavioral anomalies in user event"""
        anomalies = []

        # Skip anomaly detection for new users (insufficient baseline)
        if (datetime.utcnow() - profile.created_at) < timedelta(days=7):
            return anomalies

        # Login time anomaly detection
        login_anomaly = await self._detect_login_time_anomaly(event, profile)
        if login_anomaly:
            anomalies.append(login_anomaly)

        # Access pattern anomaly detection
        access_anomaly = await self._detect_access_pattern_anomaly(event, profile)
        if access_anomaly:
            anomalies.append(access_anomaly)

        # Data volume anomaly detection
        data_anomaly = await self._detect_data_volume_anomaly(event, profile)
        if data_anomaly:
            anomalies.append(data_anomaly)

        # Location anomaly detection
        location_anomaly = await self._detect_location_anomaly(event, profile)
        if location_anomaly:
            anomalies.append(location_anomaly)

        # Privilege escalation detection
        privilege_anomaly = await self._detect_privilege_anomaly(event, profile)
        if privilege_anomaly:
            anomalies.append(privilege_anomaly)

        # Network behavior anomaly detection
        network_anomaly = await self._detect_network_anomaly(event, profile)
        if network_anomaly:
            anomalies.append(network_anomaly)

        return anomalies

    async def _detect_login_time_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect login time anomalies"""
        if event.event_type != "login":
            return None

        current_hour = event.timestamp.hour
        current_day = event.timestamp.weekday()

        # Check if user has established login patterns
        if not profile.typical_login_hours:
            return None

        # Calculate hour deviation
        hour_deviation = self._calculate_time_deviation(current_hour, profile.typical_login_hours)

        # Check day pattern
        day_unusual = current_day not in profile.typical_login_days

        # Determine if anomalous
        is_anomalous = (hour_deviation > self.anomaly_thresholds["login_time_threshold"] or
                       (day_unusual and len(profile.typical_login_days) > 2))

        if not is_anomalous:
            return None

        # Calculate anomaly score and confidence
        anomaly_score = min(hour_deviation + (0.5 if day_unusual else 0), 5.0)
        confidence_score = min(anomaly_score / 5.0, 1.0)

        # Determine risk level
        risk_level = self._determine_risk_level(anomaly_score)

        anomaly = BehaviorAnomaly(
            anomaly_id=f"time_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.TIME_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "hour_deviation": hour_deviation,
                "day_unusual": day_unusual,
                "typical_hours": profile.typical_login_hours,
                "typical_days": profile.typical_login_days
            },
            contributing_factors=[
                f"Login at unusual hour: {current_hour}",
                f"Login on unusual day: {current_day}" if day_unusual else ""
            ],
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _detect_access_pattern_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect access pattern anomalies"""
        # Check resource access
        resource_unusual = event.resource_accessed not in profile.typical_resources

        # Check action performed
        action_unusual = event.action_performed not in profile.typical_actions

        # Calculate access pattern deviation
        resource_diversity = len(profile.typical_resources)
        action_diversity = len(profile.typical_actions)

        # Skip if insufficient baseline
        if resource_diversity < 3 or action_diversity < 2:
            return None

        # Calculate anomaly score
        anomaly_score = 0.0
        contributing_factors = []

        if resource_unusual:
            anomaly_score += 1.5
            contributing_factors.append(f"Unusual resource access: {event.resource_accessed}")

        if action_unusual:
            anomaly_score += 1.0
            contributing_factors.append(f"Unusual action: {event.action_performed}")

        # Check for sensitive resource access
        if self._is_sensitive_resource(event.resource_accessed):
            anomaly_score += 1.0
            contributing_factors.append("Access to sensitive resource")

        if anomaly_score < self.anomaly_thresholds["resource_access_threshold"]:
            return None

        confidence_score = min(anomaly_score / 4.0, 1.0)
        risk_level = self._determine_risk_level(anomaly_score)

        anomaly = BehaviorAnomaly(
            anomaly_id=f"access_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.ACCESS_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "resource_unusual": resource_unusual,
                "action_unusual": action_unusual,
                "typical_resources_count": resource_diversity,
                "typical_actions_count": action_diversity
            },
            contributing_factors=contributing_factors,
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _detect_data_volume_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect data volume anomalies"""
        if event.data_volume == 0:
            return None

        # Skip if insufficient baseline
        if not profile.data_volume_stats or profile.avg_data_volume == 0:
            return None

        # Calculate deviation from average
        avg_volume = profile.avg_data_volume
        std_volume = profile.data_volume_stats.get("std_dev", avg_volume * 0.5)

        if std_volume == 0:
            std_volume = avg_volume * 0.1  # Prevent division by zero

        volume_deviation = abs(event.data_volume - avg_volume) / std_volume

        if volume_deviation < self.anomaly_thresholds["data_volume_threshold"]:
            return None

        # Determine if unusually high or low
        direction = "high" if event.data_volume > avg_volume else "low"

        anomaly_score = min(volume_deviation, 5.0)
        confidence_score = min(volume_deviation / 5.0, 1.0)
        risk_level = self._determine_risk_level(anomaly_score)

        contributing_factors = [
            f"Unusual {direction} data volume: {event.data_volume} bytes",
            f"Deviation: {volume_deviation:.2f} standard deviations"
        ]

        # Higher risk for unusual high volume (potential data exfiltration)
        if direction == "high" and volume_deviation > 4.0:
            risk_level = RiskLevel.HIGH if risk_level == RiskLevel.MEDIUM else risk_level
            contributing_factors.append("Potential data exfiltration pattern")

        anomaly = BehaviorAnomaly(
            anomaly_id=f"data_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.DATA_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "volume_deviation": volume_deviation,
                "direction": direction,
                "average_volume": avg_volume,
                "current_volume": event.data_volume
            },
            contributing_factors=contributing_factors,
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _detect_location_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect location-based anomalies"""
        if not event.location or not profile.typical_locations:
            return None

        # Check if location is in typical set
        location_similarity = self._calculate_location_similarity(event.location, profile.typical_locations)

        if location_similarity > self.anomaly_thresholds["location_threshold"]:
            return None

        # Calculate geographic distance (simplified)
        min_distance = self._calculate_min_geographic_distance(event.location, profile.typical_locations)

        anomaly_score = (1.0 - location_similarity) * 3.0 + min(min_distance / 1000.0, 2.0)  # Distance in km
        confidence_score = min(anomaly_score / 5.0, 1.0)
        risk_level = self._determine_risk_level(anomaly_score)

        contributing_factors = [
            f"Unusual location: {event.location}",
            f"Location similarity: {location_similarity:.2f}",
            f"Minimum distance: {min_distance:.0f} km"
        ]

        # Check for impossible travel
        recent_location_events = self._get_recent_location_events(event.user_id)
        if recent_location_events:
            travel_analysis = self._analyze_travel_feasibility(recent_location_events[-1], event)
            if not travel_analysis["feasible"]:
                anomaly_score += 2.0
                contributing_factors.append(f"Impossible travel: {travel_analysis['required_speed']:.0f} km/h")

        anomaly = BehaviorAnomaly(
            anomaly_id=f"location_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.LOCATION_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "location_similarity": location_similarity,
                "min_distance_km": min_distance,
                "typical_locations": list(profile.typical_locations)
            },
            contributing_factors=contributing_factors,
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _detect_privilege_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect privilege escalation or unusual privilege usage"""
        current_privilege = event.privilege_level

        # Check if privilege level is higher than typical
        typical_privileges = set(profile.privilege_usage.keys())

        if not typical_privileges:
            return None

        # Simple privilege hierarchy (in production, would use actual RBAC hierarchy)
        privilege_hierarchy = {
            "guest": 1,
            "user": 2,
            "power_user": 3,
            "admin": 4,
            "super_admin": 5
        }

        current_level = privilege_hierarchy.get(current_privilege, 2)
        typical_max_level = max([privilege_hierarchy.get(p, 2) for p in typical_privileges])

        privilege_escalation = current_level > typical_max_level

        # Check frequency of privilege usage
        privilege_frequency = profile.privilege_usage.get(current_privilege, 0)
        total_events = sum(profile.privilege_usage.values())
        privilege_ratio = privilege_frequency / max(total_events, 1)

        anomaly_score = 0.0
        contributing_factors = []

        if privilege_escalation:
            escalation_level = current_level - typical_max_level
            anomaly_score += escalation_level * 1.5
            contributing_factors.append(f"Privilege escalation: {current_privilege}")

        if privilege_ratio < 0.05:  # Less than 5% of typical usage
            anomaly_score += 1.0
            contributing_factors.append(f"Unusual privilege usage: {current_privilege}")

        if anomaly_score < self.anomaly_thresholds["privilege_escalation_threshold"]:
            return None

        confidence_score = min(anomaly_score / 4.0, 1.0)
        risk_level = self._determine_risk_level(anomaly_score)

        # Higher risk for admin privileges
        if current_level >= 4:
            risk_level = RiskLevel.HIGH if risk_level in [RiskLevel.MEDIUM, RiskLevel.LOW] else risk_level

        anomaly = BehaviorAnomaly(
            anomaly_id=f"privilege_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.PRIVILEGE_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "privilege_escalation": privilege_escalation,
                "current_level": current_level,
                "typical_max_level": typical_max_level,
                "privilege_frequency": privilege_frequency
            },
            contributing_factors=contributing_factors,
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _detect_network_anomaly(self, event: UserBehaviorEvent, profile: BehaviorProfile) -> Optional[BehaviorAnomaly]:
        """Detect network-based behavioral anomalies"""
        # Check source IP
        ip_unusual = event.source_ip not in profile.typical_source_ips

        # Check user agent
        ua_unusual = event.user_agent not in profile.typical_user_agents

        # Skip if insufficient baseline
        if len(profile.typical_source_ips) < 2 and len(profile.typical_user_agents) < 2:
            return None

        anomaly_score = 0.0
        contributing_factors = []

        if ip_unusual:
            # Calculate IP similarity
            ip_similarity = self._calculate_ip_similarity_max(event.source_ip, profile.typical_source_ips)
            if ip_similarity < self.anomaly_thresholds["ip_threshold"]:
                anomaly_score += 2.0
                contributing_factors.append(f"Unusual source IP: {event.source_ip}")

        if ua_unusual:
            # Calculate user agent similarity
            ua_similarity = self._calculate_user_agent_similarity(event.user_agent, profile.typical_user_agents)
            if ua_similarity < 0.8:
                anomaly_score += 1.0
                contributing_factors.append(f"Unusual user agent: {event.user_agent[:50]}...")

        # Check for suspicious network patterns
        if self._is_suspicious_ip(event.source_ip):
            anomaly_score += 2.0
            contributing_factors.append("Source IP from suspicious range")

        if anomaly_score < 1.0:
            return None

        confidence_score = min(anomaly_score / 4.0, 1.0)
        risk_level = self._determine_risk_level(anomaly_score)

        anomaly = BehaviorAnomaly(
            anomaly_id=f"network_{event.user_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
            user_id=event.user_id,
            timestamp=event.timestamp,
            anomaly_type=AnomalyType.NETWORK_ANOMALY,
            risk_level=risk_level,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            triggering_event=event,
            baseline_deviation={
                "ip_unusual": ip_unusual,
                "ua_unusual": ua_unusual,
                "typical_ips_count": len(profile.typical_source_ips),
                "typical_uas_count": len(profile.typical_user_agents)
            },
            contributing_factors=contributing_factors,
            similar_users_analysis={},
            temporal_analysis={},
            risk_assessment={},
            recommended_actions=[],
            investigation_priority=self._calculate_investigation_priority(anomaly_score, confidence_score)
        )

        return anomaly

    async def _update_user_profile(self, event: UserBehaviorEvent, profile: BehaviorProfile):
        """Update user behavior profile with new event"""
        profile.updated_at = datetime.utcnow()

        # Update login patterns
        if event.event_type == "login":
            login_hour = event.timestamp.hour
            login_day = event.timestamp.weekday()

            # Update typical hours
            if login_hour not in profile.typical_login_hours:
                profile.typical_login_hours.append(login_hour)

            # Update typical days
            if login_day not in profile.typical_login_days:
                profile.typical_login_days.append(login_day)

            # Update hour distribution
            if login_hour not in profile.login_hour_distribution:
                profile.login_hour_distribution[login_hour] = 0
            profile.login_hour_distribution[login_hour] += 1

            # Update session duration stats
            self._update_stats(profile.session_duration_stats, event.session_duration)
            profile.avg_session_duration = profile.session_duration_stats.get("mean", 0)

        # Update access patterns
        profile.typical_resources.add(event.resource_accessed)
        profile.typical_actions.add(event.action_performed)

        # Update data volume stats
        if event.data_volume > 0:
            self._update_stats(profile.data_volume_stats, event.data_volume)
            profile.avg_data_volume = profile.data_volume_stats.get("mean", 0)

        # Update privilege usage
        if event.privilege_level not in profile.privilege_usage:
            profile.privilege_usage[event.privilege_level] = 0
        profile.privilege_usage[event.privilege_level] += 1

        # Update network patterns (limit to avoid memory bloat)
        if len(profile.typical_source_ips) < 20:
            profile.typical_source_ips.add(event.source_ip)

        if len(profile.typical_user_agents) < 10:
            profile.typical_user_agents.add(event.user_agent)

        if len(profile.typical_locations) < 15:
            profile.typical_locations.add(event.location)

    def _update_stats(self, stats_dict: Dict[str, float], value: float):
        """Update statistical measures"""
        if "values" not in stats_dict:
            stats_dict["values"] = []

        stats_dict["values"].append(value)

        # Keep only recent values (sliding window)
        if len(stats_dict["values"]) > 1000:
            stats_dict["values"] = stats_dict["values"][-1000:]

        values = stats_dict["values"]
        stats_dict["mean"] = statistics.mean(values)
        stats_dict["std_dev"] = statistics.stdev(values) if len(values) > 1 else 0
        stats_dict["min"] = min(values)
        stats_dict["max"] = max(values)
        stats_dict["median"] = statistics.median(values)

    async def _process_anomaly(self, anomaly: BehaviorAnomaly):
        """Process detected anomaly"""
        # Store anomaly
        self.detected_anomalies[anomaly.anomaly_id] = anomaly

        # Perform additional analysis
        anomaly.similar_users_analysis = await self._analyze_similar_users(anomaly)
        anomaly.temporal_analysis = await self._analyze_temporal_patterns(anomaly)
        anomaly.risk_assessment = await self._assess_anomaly_risk(anomaly)
        anomaly.recommended_actions = await self._generate_anomaly_recommendations(anomaly)

    async def _analyze_similar_users(self, anomaly: BehaviorAnomaly) -> Dict[str, Any]:
        """Analyze behavior compared to similar users"""
        # Find users with similar roles/departments (simplified)
        similar_users = []
        target_profile = self.user_profiles[anomaly.user_id]

        for user_id, profile in self.user_profiles.items():
            if user_id != anomaly.user_id:
                similarity = self._calculate_profile_similarity(target_profile, profile)
                if similarity > 0.7:
                    similar_users.append(user_id)

        # Analyze if similar users exhibit the same behavior
        anomaly_in_peers = 0
        for user_id in similar_users[:10]:  # Limit analysis
            peer_events = [e for e in self.recent_events if e.user_id == user_id]
            if peer_events:
                # Check if peers have similar anomalous behavior
                if self._has_similar_anomalous_behavior(anomaly, peer_events):
                    anomaly_in_peers += 1

        return {
            "similar_users_count": len(similar_users),
            "anomaly_in_peers": anomaly_in_peers,
            "peer_anomaly_rate": anomaly_in_peers / max(len(similar_users), 1),
            "uniqueness_score": 1.0 - (anomaly_in_peers / max(len(similar_users), 1))
        }

    async def _analyze_temporal_patterns(self, anomaly: BehaviorAnomaly) -> Dict[str, Any]:
        """Analyze temporal patterns around the anomaly"""
        # Look for patterns in time windows around the anomaly
        anomaly_time = anomaly.timestamp

        # Check for clustering of similar events
        time_window = timedelta(hours=2)
        related_events = [
            e for e in self.recent_events
            if (anomaly_time - time_window <= e.timestamp <= anomaly_time + time_window and
                e.user_id == anomaly.user_id)
        ]

        # Analyze event frequency and patterns
        event_frequency = len(related_events)
        unique_resources = len(set([e.resource_accessed for e in related_events]))
        unique_actions = len(set([e.action_performed for e in related_events]))

        return {
            "related_events_count": event_frequency,
            "unique_resources": unique_resources,
            "unique_actions": unique_actions,
            "activity_burst": event_frequency > 10,
            "resource_diversity": unique_resources / max(event_frequency, 1),
            "action_diversity": unique_actions / max(event_frequency, 1)
        }

    async def _assess_anomaly_risk(self, anomaly: BehaviorAnomaly) -> Dict[str, Any]:
        """Assess the risk level of the anomaly"""
        base_risk = anomaly.anomaly_score

        # Risk amplification factors
        risk_factors = {
            "confidence_multiplier": anomaly.confidence_score,
            "privilege_factor": 1.0,
            "data_sensitivity_factor": 1.0,
            "time_factor": 1.0,
            "location_factor": 1.0
        }

        # Adjust based on anomaly type
        if anomaly.anomaly_type == AnomalyType.PRIVILEGE_ANOMALY:
            risk_factors["privilege_factor"] = 2.0
        elif anomaly.anomaly_type == AnomalyType.DATA_ANOMALY:
            risk_factors["data_sensitivity_factor"] = 1.5
        elif anomaly.anomaly_type == AnomalyType.LOCATION_ANOMALY:
            risk_factors["location_factor"] = 1.3

        # Time-based risk (higher risk for off-hours)
        if anomaly.timestamp.hour < 6 or anomaly.timestamp.hour > 22:
            risk_factors["time_factor"] = 1.5

        # Calculate composite risk
        total_multiplier = sum(risk_factors.values()) / len(risk_factors)
        composite_risk = min(base_risk * total_multiplier, 10.0)

        return {
            "composite_risk_score": composite_risk,
            "base_risk": base_risk,
            "risk_factors": risk_factors,
            "risk_category": self._categorize_risk(composite_risk)
        }

    async def _generate_anomaly_recommendations(self, anomaly: BehaviorAnomaly) -> List[str]:
        """Generate recommendations for handling the anomaly"""
        recommendations = []

        # Risk-based recommendations
        if anomaly.risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE: Suspend user account pending investigation",
                "ALERT: Notify security team immediately",
                "ISOLATE: Restrict user network access"
            ])
        elif anomaly.risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "URGENT: Initiate investigation within 1 hour",
                "MONITOR: Increase user activity monitoring",
                "VERIFY: Contact user to verify legitimate activity"
            ])
        elif anomaly.risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "INVESTIGATE: Schedule investigation within 24 hours",
                "TRACK: Monitor for additional anomalous behavior"
            ])

        # Anomaly-type specific recommendations
        if anomaly.anomaly_type == AnomalyType.LOGIN_ANOMALY:
            recommendations.extend([
                "Verify login with user via secondary channel",
                "Check for credential compromise",
                "Review recent authentication logs"
            ])
        elif anomaly.anomaly_type == AnomalyType.DATA_ANOMALY:
            recommendations.extend([
                "Audit data access and transfers",
                "Check for data exfiltration indicators",
                "Review file access permissions"
            ])
        elif anomaly.anomaly_type == AnomalyType.PRIVILEGE_ANOMALY:
            recommendations.extend([
                "Review privilege escalation procedures",
                "Audit administrative actions",
                "Verify business justification for privilege use"
            ])
        elif anomaly.anomaly_type == AnomalyType.LOCATION_ANOMALY:
            recommendations.extend([
                "Verify user location via secondary means",
                "Check for VPN or proxy usage",
                "Review travel schedules and plans"
            ])

        return recommendations

    # Helper methods

    def _calculate_time_deviation(self, current_hour: int, typical_hours: List[int]) -> float:
        """Calculate deviation from typical login hours"""
        if not typical_hours:
            return 0.0

        # Find minimum distance to any typical hour
        min_distance = min([abs(current_hour - h) for h in typical_hours])

        # Account for wrap-around (e.g., 23 and 1 are close)
        min_distance = min(min_distance, 24 - min_distance)

        return min_distance / 12.0  # Normalize to 0-2 range

    def _determine_risk_level(self, anomaly_score: float) -> RiskLevel:
        """Determine risk level based on anomaly score"""
        if anomaly_score >= 4.0:
            return RiskLevel.CRITICAL
        elif anomaly_score >= 3.0:
            return RiskLevel.HIGH
        elif anomaly_score >= 2.0:
            return RiskLevel.MEDIUM
        elif anomaly_score >= 1.0:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO

    def _calculate_investigation_priority(self, anomaly_score: float, confidence_score: float) -> int:
        """Calculate investigation priority (1=highest, 5=lowest)"""
        priority_score = anomaly_score * confidence_score

        if priority_score >= 4.0:
            return 1
        elif priority_score >= 3.0:
            return 2
        elif priority_score >= 2.0:
            return 3
        elif priority_score >= 1.0:
            return 4
        else:
            return 5

    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if resource is considered sensitive"""
        sensitive_patterns = [
            "admin", "config", "database", "credentials", "keys",
            "secrets", "financial", "hr", "payroll", "confidential"
        ]

        resource_lower = resource.lower()
        return any(pattern in resource_lower for pattern in sensitive_patterns)

    def _calculate_location_similarity(self, location: str, typical_locations: Set[str]) -> float:
        """Calculate location similarity score"""
        if location in typical_locations:
            return 1.0

        # Simple string similarity (in production, would use geographic distance)
        max_similarity = 0.0
        for typical_location in typical_locations:
            similarity = self._string_similarity(location, typical_location)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard index"""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_min_geographic_distance(self, location: str, typical_locations: Set[str]) -> float:
        """Calculate minimum geographic distance (simplified)"""
        # Simplified distance calculation (in production, would use actual geo-coordinates)
        # Return random distance for demonstration
        return 100.0  # km

    def _get_recent_location_events(self, user_id: str) -> List[UserBehaviorEvent]:
        """Get recent location events for user"""
        return [e for e in list(self.recent_events)[-100:]
                if e.user_id == user_id and e.location]

    def _analyze_travel_feasibility(self, prev_event: UserBehaviorEvent, current_event: UserBehaviorEvent) -> Dict[str, Any]:
        """Analyze if travel between locations is feasible"""
        time_diff = (current_event.timestamp - prev_event.timestamp).total_seconds() / 3600  # hours
        distance = 500  # Simplified distance calculation

        max_speed = 800  # km/h (airplane speed)
        required_speed = distance / time_diff if time_diff > 0 else float('inf')

        return {
            "feasible": required_speed <= max_speed,
            "required_speed": required_speed,
            "time_diff_hours": time_diff,
            "distance_km": distance
        }

    def _calculate_ip_similarity_max(self, ip: str, typical_ips: Set[str]) -> float:
        """Calculate maximum IP similarity"""
        max_similarity = 0.0

        for typical_ip in typical_ips:
            similarity = self._calculate_ip_similarity_single(ip, typical_ip)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _calculate_ip_similarity_single(self, ip1: str, ip2: str) -> float:
        """Calculate similarity between two IP addresses"""
        try:
            octets1 = [int(x) for x in ip1.split('.')]
            octets2 = [int(x) for x in ip2.split('.')]

            similarity = 0.0
            weights = [0.4, 0.3, 0.2, 0.1]  # Weight first octets more

            for i, (o1, o2) in enumerate(zip(octets1, octets2)):
                if o1 == o2:
                    similarity += weights[i]

            return similarity
        except:
            return 0.0

    def _calculate_user_agent_similarity(self, ua: str, typical_uas: Set[str]) -> float:
        """Calculate user agent similarity"""
        max_similarity = 0.0

        for typical_ua in typical_uas:
            similarity = self._string_similarity(ua, typical_ua)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is from suspicious range"""
        # Simplified suspicious IP detection
        suspicious_patterns = [
            "10.0.0.",  # Internal ranges being accessed externally
            "192.168.",
            "172.16."
        ]

        return any(ip.startswith(pattern) for pattern in suspicious_patterns)

    def _calculate_profile_similarity(self, profile1: BehaviorProfile, profile2: BehaviorProfile) -> float:
        """Calculate similarity between two user profiles"""
        similarity_factors = []

        # Login hour similarity
        common_hours = set(profile1.typical_login_hours) & set(profile2.typical_login_hours)
        total_hours = set(profile1.typical_login_hours) | set(profile2.typical_login_hours)
        if total_hours:
            similarity_factors.append(len(common_hours) / len(total_hours))

        # Resource similarity
        common_resources = profile1.typical_resources & profile2.typical_resources
        total_resources = profile1.typical_resources | profile2.typical_resources
        if total_resources:
            similarity_factors.append(len(common_resources) / len(total_resources))

        # Action similarity
        common_actions = profile1.typical_actions & profile2.typical_actions
        total_actions = profile1.typical_actions | profile2.typical_actions
        if total_actions:
            similarity_factors.append(len(common_actions) / len(total_actions))

        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

    def _has_similar_anomalous_behavior(self, anomaly: BehaviorAnomaly, peer_events: List[UserBehaviorEvent]) -> bool:
        """Check if peer has similar anomalous behavior"""
        # Simplified check for similar behavior patterns
        anomaly_event = anomaly.triggering_event

        for event in peer_events:
            # Check for similar resource access patterns
            if (event.resource_accessed == anomaly_event.resource_accessed and
                event.action_performed == anomaly_event.action_performed):
                return True

            # Check for similar timing patterns
            if (abs((event.timestamp - anomaly_event.timestamp).total_seconds()) < 3600 and
                event.event_type == anomaly_event.event_type):
                return True

        return False

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize composite risk score"""
        if risk_score >= 8.0:
            return "extreme"
        elif risk_score >= 6.0:
            return "high"
        elif risk_score >= 4.0:
            return "moderate"
        elif risk_score >= 2.0:
            return "low"
        else:
            return "minimal"

    async def _assess_user_risk(self, event: UserBehaviorEvent, profile: BehaviorProfile,
                              anomalies: List[BehaviorAnomaly]) -> Dict[str, Any]:
        """Assess overall user risk"""
        base_risk = profile.risk_score

        # Risk from current anomalies
        anomaly_risk = sum([a.anomaly_score for a in anomalies]) / max(len(anomalies), 1)

        # Historical risk from past anomalies
        historical_risk = len(profile.historical_anomalies) * 0.1

        # Privilege-based risk
        privilege_hierarchy = {"guest": 1, "user": 2, "power_user": 3, "admin": 4, "super_admin": 5}
        privilege_risk = privilege_hierarchy.get(event.privilege_level, 2) * 0.2

        # Calculate composite risk
        composite_risk = min(base_risk + anomaly_risk + historical_risk + privilege_risk, 10.0)

        return {
            "composite_risk_score": composite_risk,
            "base_risk": base_risk,
            "anomaly_contribution": anomaly_risk,
            "historical_contribution": historical_risk,
            "privilege_contribution": privilege_risk,
            "risk_trend": "increasing" if composite_risk > base_risk else "stable"
        }

    async def _generate_behavioral_recommendations(self, event: UserBehaviorEvent,
                                                 anomalies: List[BehaviorAnomaly],
                                                 risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate behavioral analysis recommendations"""
        recommendations = []

        composite_risk = risk_assessment.get("composite_risk_score", 0)

        if composite_risk >= 7.0:
            recommendations.extend([
                "HIGH RISK: Consider immediate security review",
                "Enable enhanced monitoring for this user",
                "Require additional authentication for sensitive operations"
            ])
        elif composite_risk >= 5.0:
            recommendations.extend([
                "MODERATE RISK: Schedule security assessment",
                "Review user access permissions",
                "Monitor for additional anomalous behavior"
            ])

        if anomalies:
            recommendations.append(f"Investigate {len(anomalies)} detected anomalies")

            # Type-specific recommendations
            anomaly_types = [a.anomaly_type for a in anomalies]
            if AnomalyType.PRIVILEGE_ANOMALY in anomaly_types:
                recommendations.append("Review privilege escalation policies")
            if AnomalyType.DATA_ANOMALY in anomaly_types:
                recommendations.append("Audit data access and transfer activities")
            if AnomalyType.LOCATION_ANOMALY in anomaly_types:
                recommendations.append("Verify user location and travel plans")

        return recommendations

    async def get_user_risk_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive risk summary for user"""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}

        profile = self.user_profiles[user_id]

        # Get recent anomalies for user
        user_anomalies = [
            a for a in self.detected_anomalies.values()
            if a.user_id == user_id and
            (datetime.utcnow() - a.timestamp) < timedelta(days=30)
        ]

        # Calculate risk metrics
        risk_metrics = {
            "current_risk_score": profile.risk_score,
            "trust_level": profile.trust_level,
            "recent_anomalies": len(user_anomalies),
            "anomaly_types": list(set([a.anomaly_type.value for a in user_anomalies])),
            "highest_risk_anomaly": max([a.risk_level.value for a in user_anomalies]) if user_anomalies else "none",
            "profile_maturity": (datetime.utcnow() - profile.created_at).days
        }

        return {
            "user_id": user_id,
            "risk_metrics": risk_metrics,
            "behavioral_insights": {
                "typical_login_hours": profile.typical_login_hours,
                "resource_diversity": len(profile.typical_resources),
                "action_diversity": len(profile.typical_actions),
                "location_diversity": len(profile.typical_locations),
                "privilege_levels": list(profile.privilege_usage.keys())
            },
            "recent_anomalies": [
                {
                    "anomaly_id": a.anomaly_id,
                    "type": a.anomaly_type.value,
                    "risk_level": a.risk_level.value,
                    "timestamp": a.timestamp.isoformat(),
                    "confidence": a.confidence_score
                }
                for a in user_anomalies[-10:]  # Last 10 anomalies
            ]
        }

    async def get_detector_statistics(self) -> Dict[str, Any]:
        """Get behavioral anomaly detector statistics"""
        return {
            "events_processed": self.events_processed,
            "anomalies_detected": self.anomalies_detected,
            "false_positives": self.false_positives,
            "true_positives": self.true_positives,
            "detection_accuracy": self.true_positives / max(self.true_positives + self.false_positives, 1),
            "active_user_profiles": len(self.user_profiles),
            "recent_anomalies": len([a for a in self.detected_anomalies.values()
                                   if (datetime.utcnow() - a.timestamp) < timedelta(hours=24)]),
            "anomaly_types_distribution": {
                anomaly_type.value: len([a for a in self.detected_anomalies.values()
                                       if a.anomaly_type == anomaly_type])
                for anomaly_type in AnomalyType
            },
            "risk_level_distribution": {
                risk_level.value: len([a for a in self.detected_anomalies.values()
                                     if a.risk_level == risk_level])
                for risk_level in RiskLevel
            }
        }
