"""
Advanced Behavioral Analytics Engine
ML-powered behavioral analysis with anomaly detection, user profiling,
and predictive threat modeling capabilities.
"""

import asyncio
import json
import logging
import time
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import uuid4, UUID
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import hashlib

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Types of behavioral patterns"""
    USER_AUTHENTICATION = "user_authentication"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    SYSTEM_COMMANDS = "system_commands"
    APPLICATION_USAGE = "application_usage"
    DATA_TRANSFER = "data_transfer"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    API_USAGE = "api_usage"


class AnomalyType(Enum):
    """Types of behavioral anomalies"""
    TEMPORAL_ANOMALY = "temporal_anomaly"
    LOCATION_ANOMALY = "location_anomaly" 
    VOLUME_ANOMALY = "volume_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    VELOCITY_ANOMALY = "velocity_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    PEER_GROUP_ANOMALY = "peer_group_anomaly"


class RiskLevel(Enum):
    """Risk levels for behavioral anomalies"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class BehaviorEvent:
    """Individual behavior event"""
    id: str
    user_id: str
    entity_id: str
    behavior_type: BehaviorType
    timestamp: datetime
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    resource_accessed: Optional[str] = None
    action_performed: Optional[str] = None
    data_volume: Optional[int] = None
    duration: Optional[int] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_indicators: List[str] = field(default_factory=list)


@dataclass
class BehaviorBaseline:
    """Baseline behavioral profile for entity"""
    entity_id: str
    entity_type: str  # user, service, device
    behavior_type: BehaviorType
    created_at: datetime
    last_updated: datetime
    sample_count: int
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    location_patterns: Dict[str, Any] = field(default_factory=dict)
    volume_patterns: Dict[str, Any] = field(default_factory=dict)
    frequency_patterns: Dict[str, Any] = field(default_factory=dict)
    sequence_patterns: List[str] = field(default_factory=list)
    peer_group: Optional[str] = None
    statistical_features: Dict[str, float] = field(default_factory=dict)
    ml_model_features: Optional[List[float]] = None
    confidence_score: float = 0.0


@dataclass
class BehaviorAnomaly:
    """Detected behavioral anomaly"""
    id: str
    entity_id: str
    behavior_type: BehaviorType
    anomaly_type: AnomalyType
    risk_level: RiskLevel
    anomaly_score: float
    confidence: float
    detected_at: datetime
    triggering_events: List[str] = field(default_factory=list)
    baseline_deviation: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    prediction_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    false_positive_probability: float = 0.0
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatPrediction:
    """ML-powered threat prediction"""
    id: str
    entity_id: str
    prediction_type: str
    probability: float
    confidence: float
    time_horizon: str  # 1h, 24h, 7d, 30d
    predicted_at: datetime
    contributing_factors: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    recommended_mitigations: List[str] = field(default_factory=list)
    model_version: str = "1.0"
    feature_importance: Dict[str, float] = field(default_factory=dict)


class AdvancedBehavioralAnalyticsEngine:
    """
    Advanced Behavioral Analytics Engine with:
    - Real-time behavioral profiling and baseline establishment
    - ML-powered anomaly detection using multiple algorithms
    - Peer group analysis and outlier detection
    - Temporal, spatial, and volumetric pattern analysis
    - Predictive threat modeling
    - Adaptive learning and model updating
    - Context-aware risk scoring
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Behavioral data storage
        self.behavior_events: deque = deque(maxlen=1000000)  # Keep last 1M events
        self.baselines: Dict[str, BehaviorBaseline] = {}
        self.anomalies: Dict[str, BehaviorAnomaly] = {}
        self.predictions: Dict[str, ThreatPrediction] = {}
        
        # ML Models for different anomaly types
        self.anomaly_models = {
            "isolation_forest": {},
            "dbscan_clustering": {},
            "statistical_outlier": {},
            "sequence_analysis": {},
            "peer_group_analysis": {}
        }
        
        # Feature extractors and scalers
        self.feature_extractors = {}
        self.scalers = {}
        
        # Peer group clustering
        self.peer_groups: Dict[str, Set[str]] = {}
        self.group_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Real-time processing
        self.event_buffer: deque = deque(maxlen=10000)
        self.processing_queue = asyncio.Queue()
        
        # Learning and adaptation
        self.learning_schedule = {
            "baseline_update": timedelta(hours=1),
            "model_retrain": timedelta(days=1),
            "peer_group_refresh": timedelta(days=7)
        }
        
        # Performance metrics
        self.analytics_metrics = {
            "events_processed": 0,
            "anomalies_detected": 0,
            "predictions_generated": 0,
            "false_positive_rate": 0.0,
            "detection_accuracy": 0.0,
            "processing_latency": 0.0
        }
        
        # Configuration parameters
        self.baseline_learning_period = timedelta(days=30)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.7)
        self.min_baseline_samples = self.config.get("min_baseline_samples", 100)
        self.max_processing_latency = self.config.get("max_processing_latency", 1.0)  # seconds

    async def initialize(self) -> bool:
        """Initialize the behavioral analytics engine"""
        try:
            self.logger.info("Initializing Advanced Behavioral Analytics Engine...")
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load existing baselines and models
            await self._load_persisted_data()
            
            # Start processing workers
            for i in range(3):
                asyncio.create_task(self._behavior_processing_worker(f"worker_{i}"))
            
            # Start background tasks
            asyncio.create_task(self._baseline_learning_task())
            asyncio.create_task(self._model_training_task())
            asyncio.create_task(self._peer_group_analysis_task())
            asyncio.create_task(self._threat_prediction_task())
            asyncio.create_task(self._performance_monitoring_task())
            
            self.logger.info("Behavioral Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize behavioral analytics engine: {e}")
            return False

    async def analyze_behavior_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single behavior event in real-time"""
        try:
            start_time = time.time()
            
            # Create behavior event
            event = BehaviorEvent(
                id=str(uuid4()),
                user_id=event_data.get("user_id", "unknown"),
                entity_id=event_data.get("entity_id", event_data.get("user_id", "unknown")),
                behavior_type=BehaviorType(event_data.get("behavior_type", "user_authentication")),
                timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
                source_ip=event_data.get("source_ip"),
                user_agent=event_data.get("user_agent"),
                location=event_data.get("location"),
                session_id=event_data.get("session_id"),
                resource_accessed=event_data.get("resource_accessed"),
                action_performed=event_data.get("action_performed"),
                data_volume=event_data.get("data_volume"),
                duration=event_data.get("duration"),
                success=event_data.get("success", True),
                metadata=event_data.get("metadata", {}),
                risk_indicators=event_data.get("risk_indicators", [])
            )
            
            # Store event
            self.behavior_events.append(event)
            self.event_buffer.append(event)
            
            # Queue for asynchronous processing
            await self.processing_queue.put(event)
            
            # Perform real-time analysis
            analysis_result = await self._real_time_analysis(event)
            
            # Update metrics
            self.analytics_metrics["events_processed"] += 1
            self.analytics_metrics["processing_latency"] = time.time() - start_time
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Behavior event analysis failed: {e}")
            raise

    async def detect_anomalies_for_entity(
        self, 
        entity_id: str, 
        time_range: Dict[str, Any] = None
    ) -> List[BehaviorAnomaly]:
        """Detect behavioral anomalies for specific entity"""
        try:
            # Filter events for entity and time range
            if time_range:
                start_time = datetime.fromisoformat(time_range["start"])
                end_time = datetime.fromisoformat(time_range["end"])
                events = [
                    e for e in self.behavior_events
                    if e.entity_id == entity_id and start_time <= e.timestamp <= end_time
                ]
            else:
                # Default to last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(days=1)
                events = [
                    e for e in self.behavior_events
                    if e.entity_id == entity_id and e.timestamp >= cutoff_time
                ]
            
            if not events:
                return []
            
            # Perform comprehensive anomaly detection
            anomalies = []
            
            # Temporal anomaly detection
            temporal_anomalies = await self._detect_temporal_anomalies(entity_id, events)
            anomalies.extend(temporal_anomalies)
            
            # Volume/frequency anomaly detection
            volume_anomalies = await self._detect_volume_anomalies(entity_id, events)
            anomalies.extend(volume_anomalies)
            
            # Pattern anomaly detection
            pattern_anomalies = await self._detect_pattern_anomalies(entity_id, events)
            anomalies.extend(pattern_anomalies)
            
            # Peer group anomaly detection
            peer_anomalies = await self._detect_peer_group_anomalies(entity_id, events)
            anomalies.extend(peer_anomalies)
            
            # ML-based anomaly detection
            ml_anomalies = await self._detect_ml_anomalies(entity_id, events)
            anomalies.extend(ml_anomalies)
            
            # Store detected anomalies
            for anomaly in anomalies:
                self.anomalies[anomaly.id] = anomaly
            
            self.analytics_metrics["anomalies_detected"] += len(anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for entity {entity_id}: {e}")
            return []

    async def generate_threat_predictions(
        self, 
        entity_id: str, 
        prediction_horizon: str = "24h"
    ) -> List[ThreatPrediction]:
        """Generate ML-powered threat predictions for entity"""
        try:
            # Get entity baseline and recent events
            baseline = await self._get_or_create_baseline(entity_id, BehaviorType.USER_AUTHENTICATION)
            recent_events = await self._get_recent_events(entity_id, timedelta(days=7))
            
            if not baseline or len(recent_events) < 10:
                return []
            
            # Extract features for prediction
            features = await self._extract_prediction_features(entity_id, recent_events, baseline)
            
            # Generate predictions for different threat types
            predictions = []
            
            # Account compromise prediction
            account_compromise_prob = await self._predict_account_compromise(features)
            if account_compromise_prob > 0.3:
                predictions.append(ThreatPrediction(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    prediction_type="account_compromise",
                    probability=account_compromise_prob,
                    confidence=0.75,
                    time_horizon=prediction_horizon,
                    predicted_at=datetime.utcnow(),
                    contributing_factors=await self._get_compromise_factors(features),
                    recommended_mitigations=await self._get_compromise_mitigations(account_compromise_prob)
                ))
            
            # Insider threat prediction
            insider_threat_prob = await self._predict_insider_threat(features)
            if insider_threat_prob > 0.2:
                predictions.append(ThreatPrediction(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    prediction_type="insider_threat",
                    probability=insider_threat_prob,
                    confidence=0.65,
                    time_horizon=prediction_horizon,
                    predicted_at=datetime.utcnow(),
                    contributing_factors=await self._get_insider_threat_factors(features),
                    recommended_mitigations=await self._get_insider_threat_mitigations(insider_threat_prob)
                ))
            
            # Data exfiltration prediction
            exfiltration_prob = await self._predict_data_exfiltration(features)
            if exfiltration_prob > 0.25:
                predictions.append(ThreatPrediction(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    prediction_type="data_exfiltration",
                    probability=exfiltration_prob,
                    confidence=0.70,
                    time_horizon=prediction_horizon,
                    predicted_at=datetime.utcnow(),
                    contributing_factors=await self._get_exfiltration_factors(features),
                    recommended_mitigations=await self._get_exfiltration_mitigations(exfiltration_prob)
                ))
            
            # Store predictions
            for prediction in predictions:
                self.predictions[prediction.id] = prediction
            
            self.analytics_metrics["predictions_generated"] += len(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Threat prediction failed for entity {entity_id}: {e}")
            return []

    async def update_entity_baseline(self, entity_id: str, force_update: bool = False) -> bool:
        """Update behavioral baseline for entity"""
        try:
            # Get recent events for baseline calculation
            cutoff_time = datetime.utcnow() - self.baseline_learning_period
            events = [
                e for e in self.behavior_events
                if e.entity_id == entity_id and e.timestamp >= cutoff_time
            ]
            
            if len(events) < self.min_baseline_samples and not force_update:
                self.logger.debug(f"Insufficient events for baseline update: {len(events)}")
                return False
            
            # Group events by behavior type
            events_by_type = defaultdict(list)
            for event in events:
                events_by_type[event.behavior_type].append(event)
            
            # Update baselines for each behavior type
            for behavior_type, type_events in events_by_type.items():
                baseline = await self._calculate_baseline(entity_id, behavior_type, type_events)
                baseline_key = f"{entity_id}_{behavior_type.value}"
                self.baselines[baseline_key] = baseline
            
            self.logger.info(f"Updated baseline for entity {entity_id} with {len(events)} events")
            return True
            
        except Exception as e:
            self.logger.error(f"Baseline update failed for entity {entity_id}: {e}")
            return False

    async def get_entity_risk_profile(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive risk profile for entity"""
        try:
            # Get recent anomalies
            recent_anomalies = [
                a for a in self.anomalies.values()
                if a.entity_id == entity_id and 
                a.detected_at >= datetime.utcnow() - timedelta(days=30)
            ]
            
            # Get recent predictions
            recent_predictions = [
                p for p in self.predictions.values()
                if p.entity_id == entity_id and
                p.predicted_at >= datetime.utcnow() - timedelta(days=7)
            ]
            
            # Calculate risk scores
            anomaly_risk_score = self._calculate_anomaly_risk_score(recent_anomalies)
            prediction_risk_score = self._calculate_prediction_risk_score(recent_predictions)
            overall_risk_score = (anomaly_risk_score + prediction_risk_score) / 2
            
            # Determine risk level
            if overall_risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif overall_risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
            elif overall_risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            elif overall_risk_score >= 0.2:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.INFORMATIONAL
            
            # Get peer group information
            peer_group = await self._get_entity_peer_group(entity_id)
            
            # Generate risk factors and recommendations
            risk_factors = await self._identify_risk_factors(entity_id, recent_anomalies, recent_predictions)
            recommendations = await self._generate_risk_mitigation_recommendations(risk_factors, risk_level)
            
            return {
                "entity_id": entity_id,
                "overall_risk_score": overall_risk_score,
                "risk_level": risk_level.value,
                "anomaly_risk_score": anomaly_risk_score,
                "prediction_risk_score": prediction_risk_score,
                "recent_anomalies_count": len(recent_anomalies),
                "active_predictions_count": len(recent_predictions),
                "peer_group": peer_group,
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk profile generation failed for entity {entity_id}: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    async def _real_time_analysis(self, event: BehaviorEvent) -> Dict[str, Any]:
        """Perform real-time analysis of behavior event"""
        analysis_result = {
            "event_id": event.id,
            "entity_id": event.entity_id,
            "immediate_risk_score": 0.0,
            "anomaly_indicators": [],
            "baseline_deviations": {},
            "peer_comparison": {},
            "recommended_actions": []
        }
        
        try:
            # Get baseline for comparison
            baseline = await self._get_baseline(event.entity_id, event.behavior_type)
            
            if baseline:
                # Check for immediate anomalies
                analysis_result["baseline_deviations"] = await self._compare_to_baseline(event, baseline)
                
                # Calculate immediate risk score
                analysis_result["immediate_risk_score"] = self._calculate_immediate_risk_score(
                    event, analysis_result["baseline_deviations"]
                )
                
                # Check peer group comparison
                analysis_result["peer_comparison"] = await self._compare_to_peer_group(event)
                
                # Identify anomaly indicators
                if analysis_result["immediate_risk_score"] > self.anomaly_threshold:
                    analysis_result["anomaly_indicators"] = self._identify_anomaly_indicators(
                        event, analysis_result["baseline_deviations"]
                    )
                    
                    # Generate immediate recommendations
                    analysis_result["recommended_actions"] = self._generate_immediate_recommendations(
                        analysis_result["immediate_risk_score"], 
                        analysis_result["anomaly_indicators"]
                    )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Real-time analysis failed: {e}")
            analysis_result["error"] = str(e)
            return analysis_result

    async def _detect_temporal_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> List[BehaviorAnomaly]:
        """Detect temporal pattern anomalies"""
        anomalies = []
        
        try:
            # Get baseline temporal patterns
            baseline = await self._get_baseline(entity_id, events[0].behavior_type)
            if not baseline or not baseline.temporal_patterns:
                return anomalies
            
            # Analyze temporal patterns in events
            hour_counts = defaultdict(int)
            day_counts = defaultdict(int)
            
            for event in events:
                hour_counts[event.timestamp.hour] += 1
                day_counts[event.timestamp.weekday()] += 1
            
            # Check for unusual hours
            baseline_hours = baseline.temporal_patterns.get("active_hours", set())
            unusual_hours = []
            for hour, count in hour_counts.items():
                if hour not in baseline_hours and count > 1:  # More than 1 event in unusual hour
                    unusual_hours.append(hour)
            
            if unusual_hours:
                anomaly = BehaviorAnomaly(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    behavior_type=events[0].behavior_type,
                    anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    anomaly_score=0.7,
                    confidence=0.8,
                    detected_at=datetime.utcnow(),
                    triggering_events=[e.id for e in events if e.timestamp.hour in unusual_hours],
                    baseline_deviation={"unusual_hours": unusual_hours},
                    context={"hour_counts": dict(hour_counts)},
                    recommended_actions=["investigate_off_hours_activity", "verify_user_identity"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Temporal anomaly detection failed: {e}")
        
        return anomalies

    async def _detect_volume_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> List[BehaviorAnomaly]:
        """Detect volume/frequency anomalies"""
        anomalies = []
        
        try:
            # Get baseline volume patterns
            baseline = await self._get_baseline(entity_id, events[0].behavior_type)
            if not baseline or not baseline.volume_patterns:
                return anomalies
            
            # Calculate current volume metrics
            total_events = len(events)
            time_span_hours = (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600
            events_per_hour = total_events / max(time_span_hours, 1)
            
            # Compare to baseline
            baseline_rate = baseline.volume_patterns.get("events_per_hour", 0)
            if baseline_rate > 0:
                volume_deviation = (events_per_hour - baseline_rate) / baseline_rate
                
                if volume_deviation > 2.0:  # 200% increase
                    anomaly = BehaviorAnomaly(
                        id=str(uuid4()),
                        entity_id=entity_id,
                        behavior_type=events[0].behavior_type,
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        risk_level=RiskLevel.HIGH if volume_deviation > 5.0 else RiskLevel.MEDIUM,
                        anomaly_score=min(volume_deviation / 10.0, 1.0),
                        confidence=0.85,
                        detected_at=datetime.utcnow(),
                        triggering_events=[e.id for e in events],
                        baseline_deviation={"volume_increase_ratio": volume_deviation},
                        context={"current_rate": events_per_hour, "baseline_rate": baseline_rate},
                        recommended_actions=["investigate_bulk_activity", "check_automation", "verify_legitimate_use"]
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Volume anomaly detection failed: {e}")
        
        return anomalies

    async def _detect_pattern_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> List[BehaviorAnomaly]:
        """Detect behavioral pattern anomalies"""
        anomalies = []
        
        try:
            # Get baseline patterns
            baseline = await self._get_baseline(entity_id, events[0].behavior_type)
            if not baseline:
                return anomalies
            
            # Analyze action sequences
            action_sequences = []
            for i in range(len(events) - 2):
                sequence = tuple([
                    events[i].action_performed,
                    events[i+1].action_performed,
                    events[i+2].action_performed
                ])
                action_sequences.append(sequence)
            
            # Check against baseline sequences
            baseline_sequences = set(baseline.sequence_patterns)
            unusual_sequences = [seq for seq in action_sequences if seq not in baseline_sequences]
            
            if unusual_sequences and len(unusual_sequences) / len(action_sequences) > 0.3:
                anomaly = BehaviorAnomaly(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    behavior_type=events[0].behavior_type,
                    anomaly_type=AnomalyType.PATTERN_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    anomaly_score=len(unusual_sequences) / len(action_sequences),
                    confidence=0.75,
                    detected_at=datetime.utcnow(),
                    triggering_events=[e.id for e in events],
                    baseline_deviation={"unusual_sequence_ratio": len(unusual_sequences) / len(action_sequences)},
                    context={"unusual_sequences": unusual_sequences[:10]},  # Limit for storage
                    recommended_actions=["analyze_behavior_change", "verify_user_training"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies

    async def _detect_peer_group_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> List[BehaviorAnomaly]:
        """Detect anomalies compared to peer group"""
        anomalies = []
        
        try:
            # Get entity's peer group
            peer_group = await self._get_entity_peer_group(entity_id)
            if not peer_group:
                return anomalies
            
            # Get peer group profile
            group_profile = self.group_profiles.get(peer_group)
            if not group_profile:
                return anomalies
            
            # Calculate entity metrics
            entity_metrics = await self._calculate_entity_metrics(events)
            
            # Compare to peer group averages
            deviations = {}
            for metric, value in entity_metrics.items():
                if metric in group_profile:
                    group_avg = group_profile[metric]["average"]
                    group_std = group_profile[metric]["std_dev"]
                    
                    if group_std > 0:
                        z_score = abs(value - group_avg) / group_std
                        if z_score > 3.0:  # 3 standard deviations
                            deviations[metric] = z_score
            
            if deviations:
                max_deviation = max(deviations.values())
                anomaly = BehaviorAnomaly(
                    id=str(uuid4()),
                    entity_id=entity_id,
                    behavior_type=events[0].behavior_type,
                    anomaly_type=AnomalyType.PEER_GROUP_ANOMALY,
                    risk_level=RiskLevel.HIGH if max_deviation > 5.0 else RiskLevel.MEDIUM,
                    anomaly_score=min(max_deviation / 10.0, 1.0),
                    confidence=0.80,
                    detected_at=datetime.utcnow(),
                    triggering_events=[e.id for e in events],
                    baseline_deviation=deviations,
                    context={"peer_group": peer_group, "entity_metrics": entity_metrics},
                    recommended_actions=["compare_with_peers", "investigate_outlier_behavior"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Peer group anomaly detection failed: {e}")
        
        return anomalies

    async def _detect_ml_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> List[BehaviorAnomaly]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        try:
            # Extract features for ML models
            features = await self._extract_ml_features(events)
            if not features:
                return anomalies
            
            # Prepare feature vector
            feature_vector = np.array(features).reshape(1, -1)
            
            # Use Isolation Forest for anomaly detection
            if "isolation_forest" in self.anomaly_models:
                model = self.anomaly_models["isolation_forest"].get(entity_id)
                if model:
                    anomaly_score = model.decision_function(feature_vector)[0]
                    is_anomaly = model.predict(feature_vector)[0] == -1
                    
                    if is_anomaly:
                        anomaly = BehaviorAnomaly(
                            id=str(uuid4()),
                            entity_id=entity_id,
                            behavior_type=events[0].behavior_type,
                            anomaly_type=AnomalyType.PATTERN_ANOMALY,
                            risk_level=RiskLevel.HIGH if anomaly_score < -0.5 else RiskLevel.MEDIUM,
                            anomaly_score=abs(anomaly_score),
                            confidence=0.85,
                            detected_at=datetime.utcnow(),
                            triggering_events=[e.id for e in events],
                            baseline_deviation={"ml_anomaly_score": anomaly_score},
                            context={"model_type": "isolation_forest", "features": features[:10]},
                            recommended_actions=["detailed_investigation", "ml_model_analysis"]
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"ML anomaly detection failed: {e}")
        
        return anomalies

    # Additional helper methods would continue here for model training, baseline calculation, etc.
    # Due to length constraints, I'll include key methods

    async def _calculate_baseline(
        self, 
        entity_id: str, 
        behavior_type: BehaviorType, 
        events: List[BehaviorEvent]
    ) -> BehaviorBaseline:
        """Calculate behavioral baseline from events"""
        
        baseline = BehaviorBaseline(
            entity_id=entity_id,
            entity_type="user",  # Could be determined from entity_id
            behavior_type=behavior_type,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            sample_count=len(events)
        )
        
        # Calculate temporal patterns
        hours = [e.timestamp.hour for e in events]
        days = [e.timestamp.weekday() for e in events]
        baseline.temporal_patterns = {
            "active_hours": list(set(hours)),
            "active_days": list(set(days)),
            "peak_hour": max(set(hours), key=hours.count) if hours else None,
            "peak_day": max(set(days), key=days.count) if days else None
        }
        
        # Calculate volume patterns
        if len(events) > 1:
            time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600
            baseline.volume_patterns = {
                "events_per_hour": len(events) / max(time_span, 1),
                "avg_session_length": statistics.mean([e.duration for e in events if e.duration]),
                "avg_data_volume": statistics.mean([e.data_volume for e in events if e.data_volume])
            }
        
        # Calculate statistical features for ML
        numerical_features = []
        for event in events:
            features = [
                event.timestamp.hour,
                event.timestamp.weekday(),
                event.duration or 0,
                event.data_volume or 0,
                1 if event.success else 0
            ]
            numerical_features.append(features)
        
        if numerical_features:
            features_array = np.array(numerical_features)
            baseline.statistical_features = {
                "feature_means": features_array.mean(axis=0).tolist(),
                "feature_stds": features_array.std(axis=0).tolist(),
                "feature_mins": features_array.min(axis=0).tolist(),
                "feature_maxs": features_array.max(axis=0).tolist()
            }
        
        baseline.confidence_score = min(len(events) / self.min_baseline_samples, 1.0)
        
        return baseline

    async def _get_baseline(self, entity_id: str, behavior_type: BehaviorType) -> Optional[BehaviorBaseline]:
        """Get baseline for entity and behavior type"""
        baseline_key = f"{entity_id}_{behavior_type.value}"
        return self.baselines.get(baseline_key)

    async def _get_or_create_baseline(self, entity_id: str, behavior_type: BehaviorType) -> Optional[BehaviorBaseline]:
        """Get existing baseline or create new one"""
        baseline = await self._get_baseline(entity_id, behavior_type)
        if not baseline:
            # Try to create baseline from recent events
            await self.update_entity_baseline(entity_id)
            baseline = await self._get_baseline(entity_id, behavior_type)
        return baseline


# Global instance
_behavioral_analytics_engine: Optional[AdvancedBehavioralAnalyticsEngine] = None

async def get_behavioral_analytics_engine(config: Dict[str, Any] = None) -> AdvancedBehavioralAnalyticsEngine:
    """Get global behavioral analytics engine instance"""
    global _behavioral_analytics_engine
    
    if _behavioral_analytics_engine is None:
        _behavioral_analytics_engine = AdvancedBehavioralAnalyticsEngine(config)
        await _behavioral_analytics_engine.initialize()
    
    return _behavioral_analytics_engine