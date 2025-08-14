#!/usr/bin/env python3
"""
XORB Advanced Behavioral Analytics Engine
Real-world ML-powered user and entity behavior analysis with production-grade implementations
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import pickle
import os
from pathlib import Path
from collections import defaultdict, deque
import math
import statistics

# Machine Learning Dependencies with graceful fallbacks
try:
    import sklearn
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import DBSCAN, KMeans, OPTICS
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.covariance import EllipticEnvelope
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create comprehensive compatibility stubs for TensorFlow
    class TensorFlowStub:
        """Compatibility stub for TensorFlow functionality"""
        def __init__(self):
            self.keras = self.KerasStub()
            self.nn = self.NNStub()

        class KerasStub:
            def __init__(self):
                self.models = self.ModelsStub()
                self.layers = self.LayersStub()
                self.optimizers = self.OptimizersStub()

            class ModelsStub:
                def Sequential(self, *args, **kwargs):
                    logging.warning("TensorFlow not available - Sequential model placeholder")
                    return None

            class LayersStub:
                def Dense(self, *args, **kwargs):
                    logging.warning("TensorFlow not available - Dense layer placeholder")
                    return None
                def LSTM(self, *args, **kwargs):
                    logging.warning("TensorFlow not available - LSTM layer placeholder")
                    return None

            class OptimizersStub:
                def Adam(self, *args, **kwargs):
                    logging.warning("TensorFlow not available - Adam optimizer placeholder")
                    return None

        class NNStub:
            def relu(self, x):
                return max(0, x) if isinstance(x, (int, float)) else x

    tf = TensorFlowStub()

try:
    from scipy import stats
    from scipy.spatial.distance import euclidean, cosine
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Types of behavioral patterns"""
    USER_ACTIVITY = "user_activity"
    NETWORK_ACCESS = "network_access"
    DATA_ACCESS = "data_access"
    SYSTEM_INTERACTION = "system_interaction"
    APPLICATION_USAGE = "application_usage"
    AUTHENTICATION = "authentication"
    PRIVILEGE_USAGE = "privilege_usage"
    FILE_OPERATIONS = "file_operations"
    EMAIL_COMMUNICATION = "email_communication"
    WEB_BROWSING = "web_browsing"

class AnomalyType(Enum):
    """Types of behavioral anomalies"""
    TEMPORAL_ANOMALY = "temporal_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    LOCATION_ANOMALY = "location_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    CORRELATION_ANOMALY = "correlation_anomaly"
    STATISTICAL_ANOMALY = "statistical_anomaly"

class RiskLevel(Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ConfidenceLevel(Enum):
    """Confidence levels for behavioral analysis"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 60-74%
    LOW = "low"            # 40-59%
    VERY_LOW = "very_low"  # 0-39%

@dataclass
class BehaviorEvent:
    """Individual behavior event"""
    event_id: str
    user_id: str
    entity_id: str
    event_type: BehaviorType
    timestamp: datetime
    metadata: Dict[str, Any]
    risk_score: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorBaseline:
    """Behavioral baseline for an entity"""
    entity_id: str
    behavior_type: BehaviorType
    baseline_metrics: Dict[str, float]
    temporal_patterns: Dict[str, Any]
    frequency_patterns: Dict[str, Any]
    typical_volumes: Dict[str, float]
    learned_patterns: List[Dict[str, Any]]
    last_updated: datetime
    confidence: float
    sample_size: int

@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly"""
    anomaly_id: str
    entity_id: str
    behavior_type: BehaviorType
    anomaly_type: AnomalyType
    severity: RiskLevel
    confidence: ConfidenceLevel
    confidence_score: float
    anomaly_score: float
    description: str
    evidence: Dict[str, Any]
    baseline_deviation: Dict[str, float]
    temporal_context: Dict[str, Any]
    mitigation_recommendations: List[str]
    detection_timestamp: datetime
    affected_events: List[str]

@dataclass
class EntityProfile:
    """Comprehensive entity behavioral profile"""
    entity_id: str
    entity_type: str  # user, service_account, system, etc.
    baselines: Dict[BehaviorType, BehaviorBaseline]
    risk_score: float
    anomaly_history: List[BehavioralAnomaly]
    peer_groups: List[str]
    behavioral_traits: Dict[str, Any]
    last_analysis: datetime
    profile_created: datetime

class TimeSeriesAnalyzer:
    """Advanced time series analysis for behavioral patterns"""

    def __init__(self):
        self.seasonal_patterns = {}
        self.trend_models = {}

    async def analyze_temporal_patterns(self, events: List[BehaviorEvent], entity_id: str) -> Dict[str, Any]:
        """Analyze temporal patterns in behavior"""
        if not events:
            return {"error": "No events provided"}

        # Convert events to time series
        timestamps = [event.timestamp for event in events]
        event_counts = self._create_hourly_buckets(timestamps)

        # Analyze patterns
        patterns = {
            "hourly_distribution": await self._analyze_hourly_distribution(event_counts),
            "daily_patterns": await self._analyze_daily_patterns(timestamps),
            "weekly_patterns": await self._analyze_weekly_patterns(timestamps),
            "seasonal_trends": await self._analyze_seasonal_trends(timestamps),
            "frequency_analysis": await self._analyze_frequency_patterns(timestamps),
            "burst_detection": await self._detect_activity_bursts(timestamps),
            "periodicity": await self._detect_periodicity(event_counts)
        }

        return patterns

    def _create_hourly_buckets(self, timestamps: List[datetime]) -> Dict[int, int]:
        """Create hourly activity buckets"""
        hourly_counts = defaultdict(int)
        for timestamp in timestamps:
            hour = timestamp.hour
            hourly_counts[hour] += 1
        return dict(hourly_counts)

    async def _analyze_hourly_distribution(self, event_counts: Dict[int, int]) -> Dict[str, Any]:
        """Analyze hourly distribution patterns"""
        if not event_counts:
            return {"pattern": "no_activity"}

        # Calculate statistics
        hours = list(range(24))
        counts = [event_counts.get(hour, 0) for hour in hours]

        analysis = {
            "peak_hours": [h for h, c in event_counts.items() if c == max(event_counts.values())],
            "quiet_hours": [h for h in hours if event_counts.get(h, 0) == 0],
            "activity_variance": statistics.variance(counts) if len(counts) > 1 else 0,
            "business_hours_ratio": sum(event_counts.get(h, 0) for h in range(9, 17)) / sum(counts) if sum(counts) > 0 else 0,
            "after_hours_ratio": sum(event_counts.get(h, 0) for h in list(range(0, 9)) + list(range(17, 24))) / sum(counts) if sum(counts) > 0 else 0
        }

        # Classify pattern
        if analysis["business_hours_ratio"] > 0.8:
            analysis["pattern"] = "business_hours_focused"
        elif analysis["after_hours_ratio"] > 0.6:
            analysis["pattern"] = "after_hours_activity"
        elif analysis["activity_variance"] < 5:
            analysis["pattern"] = "uniform_distribution"
        else:
            analysis["pattern"] = "irregular_pattern"

        return analysis

    async def _analyze_daily_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze daily activity patterns"""
        daily_counts = defaultdict(int)
        for timestamp in timestamps:
            day_key = timestamp.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

        if not daily_counts:
            return {"pattern": "no_activity"}

        counts = list(daily_counts.values())
        return {
            "avg_daily_activity": statistics.mean(counts),
            "daily_variance": statistics.variance(counts) if len(counts) > 1 else 0,
            "max_daily_activity": max(counts),
            "min_daily_activity": min(counts),
            "active_days": len([c for c in counts if c > 0]),
            "consistency_score": 1.0 - (statistics.stdev(counts) / statistics.mean(counts)) if statistics.mean(counts) > 0 and len(counts) > 1 else 1.0
        }

    async def _analyze_weekly_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze weekly activity patterns"""
        weekday_counts = defaultdict(int)
        for timestamp in timestamps:
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            weekday_counts[weekday] += 1

        if not weekday_counts:
            return {"pattern": "no_activity"}

        total_events = sum(weekday_counts.values())
        weekday_distribution = {day: count/total_events for day, count in weekday_counts.items()}

        # Calculate workday vs weekend ratio
        workday_activity = sum(weekday_counts.get(day, 0) for day in range(5))  # Mon-Fri
        weekend_activity = sum(weekday_counts.get(day, 0) for day in [5, 6])    # Sat-Sun

        return {
            "weekday_distribution": weekday_distribution,
            "workday_weekend_ratio": workday_activity / max(weekend_activity, 1),
            "most_active_day": max(weekday_counts, key=weekday_counts.get) if weekday_counts else None,
            "least_active_day": min(weekday_counts, key=weekday_counts.get) if weekday_counts else None,
            "weekend_activity_ratio": weekend_activity / total_events if total_events > 0 else 0
        }

    async def _analyze_seasonal_trends(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze seasonal trends and long-term patterns"""
        if len(timestamps) < 7:  # Need at least a week of data
            return {"trend": "insufficient_data"}

        # Group by week
        weekly_counts = defaultdict(int)
        for timestamp in timestamps:
            week_key = timestamp.strftime("%Y-W%W")
            weekly_counts[week_key] += 1

        if len(weekly_counts) < 3:
            return {"trend": "insufficient_timespan"}

        # Analyze trend
        weeks = sorted(weekly_counts.keys())
        counts = [weekly_counts[week] for week in weeks]

        # Simple linear trend analysis
        n = len(counts)
        x = list(range(n))
        trend_slope = (n * sum(x[i] * counts[i] for i in range(n)) - sum(x) * sum(counts)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2) if n > 1 else 0

        return {
            "trend_slope": trend_slope,
            "trend_direction": "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable",
            "weekly_variance": statistics.variance(counts) if len(counts) > 1 else 0,
            "data_points": len(weekly_counts)
        }

    async def _analyze_frequency_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze frequency patterns and intervals"""
        if len(timestamps) < 2:
            return {"pattern": "insufficient_data"}

        # Sort timestamps
        sorted_timestamps = sorted(timestamps)

        # Calculate intervals between events
        intervals = []
        for i in range(1, len(sorted_timestamps)):
            interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return {"pattern": "single_event"}

        return {
            "avg_interval_seconds": statistics.mean(intervals),
            "median_interval_seconds": statistics.median(intervals),
            "interval_variance": statistics.variance(intervals) if len(intervals) > 1 else 0,
            "min_interval_seconds": min(intervals),
            "max_interval_seconds": max(intervals),
            "regularity_score": 1.0 / (1.0 + statistics.stdev(intervals)/statistics.mean(intervals)) if statistics.mean(intervals) > 0 and len(intervals) > 1 else 1.0
        }

    async def _detect_activity_bursts(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect unusual bursts of activity"""
        if len(timestamps) < 10:
            return {"bursts_detected": 0}

        # Create 5-minute buckets
        bucket_size = 300  # 5 minutes in seconds
        buckets = defaultdict(int)

        base_time = min(timestamps)
        for timestamp in timestamps:
            bucket_key = int((timestamp - base_time).total_seconds() // bucket_size)
            buckets[bucket_key] += 1

        counts = list(buckets.values())
        if len(counts) < 3:
            return {"bursts_detected": 0}

        # Calculate burst threshold (mean + 2 standard deviations)
        mean_activity = statistics.mean(counts)
        std_activity = statistics.stdev(counts) if len(counts) > 1 else 0
        burst_threshold = mean_activity + (2 * std_activity)

        bursts = [count for count in counts if count > burst_threshold]

        return {
            "bursts_detected": len(bursts),
            "burst_threshold": burst_threshold,
            "max_burst_intensity": max(bursts) if bursts else 0,
            "burst_ratio": len(bursts) / len(counts) if counts else 0
        }

    async def _detect_periodicity(self, event_counts: Dict[int, int]) -> Dict[str, Any]:
        """Detect periodic patterns in activity"""
        if len(event_counts) < 5:
            return {"periodicity": "insufficient_data"}

        # Simple periodicity detection
        hours = sorted(event_counts.keys())
        counts = [event_counts[hour] for hour in hours]

        # Calculate autocorrelation for common periods
        periods_to_check = [6, 8, 12, 24]  # 6h, 8h, 12h, 24h periods
        autocorrelations = {}

        for period in periods_to_check:
            if len(counts) > period:
                correlation = self._calculate_autocorrelation(counts, period)
                autocorrelations[f"{period}h"] = correlation

        # Find strongest periodic pattern
        strongest_period = max(autocorrelations, key=autocorrelations.get) if autocorrelations else None
        strongest_correlation = autocorrelations.get(strongest_period, 0) if strongest_period else 0

        return {
            "strongest_period": strongest_period,
            "correlation_strength": strongest_correlation,
            "periodic_patterns": autocorrelations,
            "is_periodic": strongest_correlation > 0.7
        }

    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0.0

        n = len(data) - lag
        if n <= 0:
            return 0.0

        mean_data = statistics.mean(data)

        numerator = sum((data[i] - mean_data) * (data[i + lag] - mean_data) for i in range(n))
        denominator = sum((x - mean_data) ** 2 for x in data)

        return numerator / denominator if denominator > 0 else 0.0

class MLAnomalyDetector:
    """Advanced ML-based anomaly detection using ensemble methods"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_importance = {}

    async def initialize(self):
        """Initialize ML models for anomaly detection"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using simplified anomaly detection")
            return

        # Initialize ensemble of anomaly detection models
        self.models = {
            "isolation_forest": IsolationForest(
                contamination=0.1,
                n_estimators=200,
                max_samples='auto',
                random_state=42
            ),
            "local_outlier_factor": LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                novelty=True
            ),
            "one_class_svm": OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            ),
            "elliptic_envelope": EllipticEnvelope(
                contamination=0.1,
                random_state=42
            )
        }

        self.scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler()
        }

        # Train with synthetic behavioral data
        await self._train_with_synthetic_data()
        logger.info("ML anomaly detection models initialized")

    async def _train_with_synthetic_data(self):
        """Train models with synthetic behavioral data"""
        if not SKLEARN_AVAILABLE:
            return

        # Generate synthetic training data representing normal behavior
        np.random.seed(42)

        # Normal behavioral patterns
        normal_data = []
        for _ in range(1000):
            # Simulate normal user behavior features
            features = [
                np.random.normal(0.3, 0.1),    # login_frequency
                np.random.normal(0.5, 0.15),   # data_access_volume
                np.random.normal(0.4, 0.1),    # application_usage
                np.random.normal(0.2, 0.05),   # privilege_usage
                np.random.normal(0.6, 0.2),    # network_activity
                np.random.normal(0.4, 0.1),    # file_operations
                np.random.normal(0.3, 0.08),   # email_activity
                np.random.normal(0.5, 0.12),   # web_browsing
                np.random.normal(0.2, 0.05),   # after_hours_activity
                np.random.normal(0.8, 0.1),    # business_hours_activity
                np.random.normal(0.1, 0.03),   # failed_logins
                np.random.normal(0.05, 0.02),  # admin_operations
                np.random.normal(0.7, 0.15),   # predictable_patterns
                np.random.normal(0.4, 0.1),    # geographical_consistency
                np.random.normal(0.3, 0.08)    # device_consistency
            ]
            normal_data.append(features)

        # Anomalous behavioral patterns
        anomaly_data = []
        for _ in range(100):
            # Simulate anomalous behavior
            features = [
                np.random.normal(0.8, 0.2),    # high login frequency
                np.random.normal(0.9, 0.1),    # high data access
                np.random.normal(0.2, 0.1),    # low app usage
                np.random.normal(0.7, 0.15),   # high privilege usage
                np.random.normal(0.9, 0.1),    # high network activity
                np.random.normal(0.8, 0.15),   # high file operations
                np.random.normal(0.1, 0.05),   # low email activity
                np.random.normal(0.2, 0.1),    # low web browsing
                np.random.normal(0.8, 0.2),    # high after hours
                np.random.normal(0.2, 0.1),    # low business hours
                np.random.normal(0.5, 0.2),    # high failed logins
                np.random.normal(0.6, 0.2),    # high admin ops
                np.random.normal(0.1, 0.05),   # unpredictable
                np.random.normal(0.1, 0.05),   # geographical inconsistency
                np.random.normal(0.1, 0.05)    # device inconsistency
            ]
            anomaly_data.append(features)

        # Combine normal and anomalous data
        X_train = np.vstack([normal_data, anomaly_data])
        y_train = np.hstack([np.ones(1000), np.full(100, -1)])  # 1 for normal, -1 for anomaly

        # Scale data
        X_scaled = self.scalers["standard"].fit_transform(X_train)

        # Train models
        self.models["isolation_forest"].fit(X_scaled)
        self.models["local_outlier_factor"].fit(X_scaled[y_train == 1])  # Train only on normal data
        self.models["one_class_svm"].fit(X_scaled[y_train == 1])
        self.models["elliptic_envelope"].fit(X_scaled[y_train == 1])

        self.is_trained = True
        logger.info("ML anomaly detection models trained successfully")

    async def detect_anomalies(self, behavior_features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using ensemble ML models"""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return await self._fallback_anomaly_detection(behavior_features)

        # Scale features
        scaled_features = self.scalers["standard"].transform(behavior_features.reshape(1, -1))

        # Run through ensemble of models
        results = {}

        # Isolation Forest
        iso_score = self.models["isolation_forest"].decision_function(scaled_features)[0]
        iso_anomaly = self.models["isolation_forest"].predict(scaled_features)[0] == -1
        results["isolation_forest"] = {
            "anomaly_score": float(iso_score),
            "is_anomaly": bool(iso_anomaly),
            "confidence": min(abs(iso_score) / 0.5, 1.0)
        }

        # Local Outlier Factor
        lof_score = self.models["local_outlier_factor"].decision_function(scaled_features)[0]
        lof_anomaly = self.models["local_outlier_factor"].predict(scaled_features)[0] == -1
        results["local_outlier_factor"] = {
            "anomaly_score": float(lof_score),
            "is_anomaly": bool(lof_anomaly),
            "confidence": min(abs(lof_score) / 2.0, 1.0)
        }

        # One-Class SVM
        svm_score = self.models["one_class_svm"].decision_function(scaled_features)[0]
        svm_anomaly = self.models["one_class_svm"].predict(scaled_features)[0] == -1
        results["one_class_svm"] = {
            "anomaly_score": float(svm_score),
            "is_anomaly": bool(svm_anomaly),
            "confidence": min(abs(svm_score) / 1.0, 1.0)
        }

        # Elliptic Envelope
        ee_score = self.models["elliptic_envelope"].decision_function(scaled_features)[0]
        ee_anomaly = self.models["elliptic_envelope"].predict(scaled_features)[0] == -1
        results["elliptic_envelope"] = {
            "anomaly_score": float(ee_score),
            "is_anomaly": bool(ee_anomaly),
            "confidence": min(abs(ee_score) / 5.0, 1.0)
        }

        # Ensemble decision
        anomaly_scores = [results[model]["anomaly_score"] for model in results]
        anomaly_votes = sum(1 for model in results if results[model]["is_anomaly"])
        confidence_scores = [results[model]["confidence"] for model in results]

        ensemble_score = np.mean(anomaly_scores)
        ensemble_confidence = np.mean(confidence_scores)
        ensemble_anomaly = anomaly_votes >= 2  # Majority vote

        return {
            "ensemble_anomaly_score": float(ensemble_score),
            "is_anomaly": bool(ensemble_anomaly),
            "confidence": float(ensemble_confidence),
            "anomaly_votes": int(anomaly_votes),
            "total_models": len(results),
            "individual_models": results,
            "feature_importance": await self._calculate_feature_importance(scaled_features[0])
        }

    async def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for anomaly detection"""
        feature_names = [
            "login_frequency", "data_access_volume", "application_usage",
            "privilege_usage", "network_activity", "file_operations",
            "email_activity", "web_browsing", "after_hours_activity",
            "business_hours_activity", "failed_logins", "admin_operations",
            "predictable_patterns", "geographical_consistency", "device_consistency"
        ]

        # Calculate relative importance based on feature values
        importance = {}
        max_value = np.max(np.abs(features))

        for i, name in enumerate(feature_names):
            importance[name] = float(abs(features[i]) / max_value) if max_value > 0 else 0.0

        return importance

    async def _fallback_anomaly_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Fallback anomaly detection when ML libraries unavailable"""
        # Simple statistical anomaly detection
        anomaly_score = 0.0
        anomaly_indicators = []

        feature_names = [
            "login_frequency", "data_access_volume", "application_usage",
            "privilege_usage", "network_activity", "file_operations",
            "email_activity", "web_browsing", "after_hours_activity",
            "business_hours_activity", "failed_logins", "admin_operations",
            "predictable_patterns", "geographical_consistency", "device_consistency"
        ]

        # Check individual feature thresholds
        for i, (feature_value, feature_name) in enumerate(zip(features, feature_names)):
            if feature_value > 0.8:  # High activity threshold
                anomaly_score += 0.2
                anomaly_indicators.append(f"High {feature_name}: {feature_value:.2f}")
            elif feature_value < 0.1:  # Low activity threshold (suspicious for some features)
                if feature_name in ["predictable_patterns", "geographical_consistency", "device_consistency"]:
                    anomaly_score += 0.15
                    anomaly_indicators.append(f"Low {feature_name}: {feature_value:.2f}")

        anomaly_score = min(anomaly_score, 1.0)

        return {
            "ensemble_anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > 0.5,
            "confidence": 0.7,
            "anomaly_indicators": anomaly_indicators,
            "method": "statistical_fallback"
        }

class PeerGroupAnalyzer:
    """Analyze peer group behavior for comparative analysis"""

    def __init__(self):
        self.peer_groups = {}
        self.clustering_model = None

    async def analyze_peer_groups(self, entity_profiles: List[EntityProfile]) -> Dict[str, Any]:
        """Analyze and create peer groups based on behavioral similarity"""
        if not entity_profiles:
            return {"peer_groups": []}

        if SKLEARN_AVAILABLE:
            return await self._ml_peer_grouping(entity_profiles)
        else:
            return await self._simple_peer_grouping(entity_profiles)

    async def _ml_peer_grouping(self, entity_profiles: List[EntityProfile]) -> Dict[str, Any]:
        """ML-based peer group analysis using clustering"""
        # Extract behavioral features for clustering
        features = []
        entity_ids = []

        for profile in entity_profiles:
            if profile.baselines:
                # Extract representative features from baselines
                feature_vector = await self._extract_clustering_features(profile)
                features.append(feature_vector)
                entity_ids.append(profile.entity_id)

        if len(features) < 3:
            return {"peer_groups": [], "method": "insufficient_data"}

        features_array = np.array(features)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_array)

        # Apply clustering
        optimal_clusters = min(max(len(features) // 10, 2), 8)  # 2-8 clusters

        # Try multiple clustering algorithms
        clustering_results = {}

        # K-Means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_features)
        clustering_results["kmeans"] = {
            "labels": kmeans_labels,
            "silhouette_score": silhouette_score(scaled_features, kmeans_labels) if len(set(kmeans_labels)) > 1 else -1
        }

        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(scaled_features)
        clustering_results["dbscan"] = {
            "labels": dbscan_labels,
            "silhouette_score": silhouette_score(scaled_features, dbscan_labels) if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels else -1
        }

        # Choose best clustering method
        best_method = max(clustering_results.keys(),
                         key=lambda k: clustering_results[k]["silhouette_score"])
        best_labels = clustering_results[best_method]["labels"]

        # Create peer groups
        peer_groups = defaultdict(list)
        for entity_id, label in zip(entity_ids, best_labels):
            if label != -1:  # Ignore noise points in DBSCAN
                peer_groups[f"group_{label}"].append(entity_id)

        # Calculate group characteristics
        group_analysis = {}
        for group_name, members in peer_groups.items():
            member_profiles = [p for p in entity_profiles if p.entity_id in members]
            group_analysis[group_name] = await self._analyze_group_characteristics(member_profiles)

        return {
            "peer_groups": dict(peer_groups),
            "group_analysis": group_analysis,
            "clustering_method": best_method,
            "silhouette_score": clustering_results[best_method]["silhouette_score"],
            "total_groups": len(peer_groups)
        }

    async def _extract_clustering_features(self, profile: EntityProfile) -> List[float]:
        """Extract features for clustering analysis"""
        features = []

        # Average risk score
        features.append(profile.risk_score)

        # Baseline metrics aggregation
        baseline_features = []
        for behavior_type, baseline in profile.baselines.items():
            baseline_features.extend([
                baseline.confidence,
                len(baseline.learned_patterns),
                baseline.sample_size / 1000.0,  # Normalize
                sum(baseline.baseline_metrics.values()) / len(baseline.baseline_metrics) if baseline.baseline_metrics else 0
            ])

        # Pad or truncate to fixed size
        while len(baseline_features) < 20:
            baseline_features.append(0.0)
        features.extend(baseline_features[:20])

        # Behavioral traits
        traits = profile.behavioral_traits
        features.extend([
            traits.get("activity_level", 0.5),
            traits.get("predictability", 0.5),
            traits.get("risk_propensity", 0.5),
            traits.get("temporal_consistency", 0.5)
        ])

        # Anomaly history features
        recent_anomalies = [a for a in profile.anomaly_history
                          if (datetime.utcnow() - a.detection_timestamp).days <= 30]
        features.extend([
            len(recent_anomalies) / 10.0,  # Normalize
            len([a for a in recent_anomalies if a.severity == RiskLevel.HIGH]) / max(len(recent_anomalies), 1),
            np.mean([a.confidence_score for a in recent_anomalies]) if recent_anomalies else 0.5
        ])

        return features

    async def _analyze_group_characteristics(self, member_profiles: List[EntityProfile]) -> Dict[str, Any]:
        """Analyze characteristics of a peer group"""
        if not member_profiles:
            return {}

        # Calculate group statistics
        risk_scores = [p.risk_score for p in member_profiles]

        characteristics = {
            "member_count": len(member_profiles),
            "avg_risk_score": statistics.mean(risk_scores),
            "risk_variance": statistics.variance(risk_scores) if len(risk_scores) > 1 else 0,
            "high_risk_members": len([r for r in risk_scores if r > 0.7]),
            "behavioral_consistency": await self._calculate_group_consistency(member_profiles),
            "common_behaviors": await self._identify_common_behaviors(member_profiles),
            "group_anomaly_rate": await self._calculate_group_anomaly_rate(member_profiles)
        }

        return characteristics

    async def _calculate_group_consistency(self, profiles: List[EntityProfile]) -> float:
        """Calculate behavioral consistency within the group"""
        if len(profiles) < 2:
            return 1.0

        # Calculate pairwise behavioral similarity
        similarities = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                similarity = await self._calculate_profile_similarity(profiles[i], profiles[j])
                similarities.append(similarity)

        return statistics.mean(similarities) if similarities else 0.0

    async def _calculate_profile_similarity(self, profile1: EntityProfile, profile2: EntityProfile) -> float:
        """Calculate similarity between two behavioral profiles"""
        # Compare risk scores
        risk_similarity = 1.0 - abs(profile1.risk_score - profile2.risk_score)

        # Compare common behavior types
        common_behaviors = set(profile1.baselines.keys()) & set(profile2.baselines.keys())
        if not common_behaviors:
            return risk_similarity * 0.5

        # Calculate baseline similarity for common behaviors
        baseline_similarities = []
        for behavior_type in common_behaviors:
            baseline1 = profile1.baselines[behavior_type]
            baseline2 = profile2.baselines[behavior_type]

            # Compare baseline metrics
            metrics1 = baseline1.baseline_metrics
            metrics2 = baseline2.baseline_metrics

            if metrics1 and metrics2:
                common_metrics = set(metrics1.keys()) & set(metrics2.keys())
                if common_metrics:
                    metric_similarities = [
                        1.0 - abs(metrics1[metric] - metrics2[metric])
                        for metric in common_metrics
                    ]
                    baseline_similarities.append(statistics.mean(metric_similarities))

        baseline_similarity = statistics.mean(baseline_similarities) if baseline_similarities else 0.5

        # Combine similarities
        return (risk_similarity + baseline_similarity) / 2.0

    async def _identify_common_behaviors(self, profiles: List[EntityProfile]) -> List[str]:
        """Identify common behavioral patterns in the group"""
        behavior_counts = defaultdict(int)

        for profile in profiles:
            for behavior_type in profile.baselines.keys():
                behavior_counts[behavior_type.value] += 1

        # Return behaviors present in at least 50% of group members
        threshold = len(profiles) * 0.5
        return [behavior for behavior, count in behavior_counts.items() if count >= threshold]

    async def _calculate_group_anomaly_rate(self, profiles: List[EntityProfile]) -> float:
        """Calculate the group's anomaly detection rate"""
        total_anomalies = sum(len(p.anomaly_history) for p in profiles)
        total_members = len(profiles)

        return total_anomalies / max(total_members, 1)

    async def _simple_peer_grouping(self, entity_profiles: List[EntityProfile]) -> Dict[str, Any]:
        """Simple peer grouping when ML libraries unavailable"""
        # Group by risk level
        peer_groups = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }

        for profile in entity_profiles:
            if profile.risk_score > 0.7:
                peer_groups["high_risk"].append(profile.entity_id)
            elif profile.risk_score > 0.3:
                peer_groups["medium_risk"].append(profile.entity_id)
            else:
                peer_groups["low_risk"].append(profile.entity_id)

        return {
            "peer_groups": peer_groups,
            "method": "risk_based_grouping",
            "total_groups": len([g for g in peer_groups.values() if g])
        }

class BehavioralAnalyticsEngine:
    """Main behavioral analytics engine combining all analysis components"""

    def __init__(self):
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.ml_detector = MLAnomalyDetector()
        self.peer_analyzer = PeerGroupAnalyzer()

        # Storage for profiles and baselines
        self.entity_profiles: Dict[str, EntityProfile] = {}
        self.behavioral_baselines: Dict[str, Dict[BehaviorType, BehaviorBaseline]] = {}

        # Analytics configuration
        self.config = {
            "baseline_learning_period": 30,  # days
            "minimum_events_for_baseline": 100,
            "anomaly_detection_threshold": 0.7,
            "peer_group_update_interval": 24,  # hours
            "feature_importance_threshold": 0.1
        }

    async def initialize(self):
        """Initialize the behavioral analytics engine"""
        logger.info("Initializing Behavioral Analytics Engine...")

        # Initialize ML components
        await self.ml_detector.initialize()

        logger.info("Behavioral Analytics Engine initialization complete")

    async def process_behavior_events(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Process a batch of behavior events for analysis"""
        if not events:
            return {"processed_events": 0}

        results = {
            "processed_events": len(events),
            "entity_updates": {},
            "anomalies_detected": [],
            "baseline_updates": [],
            "peer_group_changes": []
        }

        # Group events by entity
        entity_events = defaultdict(list)
        for event in events:
            entity_events[event.entity_id].append(event)

        # Process each entity's events
        for entity_id, entity_event_list in entity_events.items():
            entity_results = await self._process_entity_events(entity_id, entity_event_list)
            results["entity_updates"][entity_id] = entity_results

            # Collect anomalies
            if entity_results.get("anomalies"):
                results["anomalies_detected"].extend(entity_results["anomalies"])

            # Track baseline updates
            if entity_results.get("baseline_updated"):
                results["baseline_updates"].append(entity_id)

        # Update peer groups if needed
        if len(results["entity_updates"]) > 0:
            peer_group_results = await self._update_peer_groups()
            results["peer_group_changes"] = peer_group_results.get("changes", [])

        return results

    async def _process_entity_events(self, entity_id: str, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Process events for a single entity"""
        entity_results = {
            "events_processed": len(events),
            "anomalies": [],
            "baseline_updated": False,
            "risk_score_change": 0.0,
            "new_patterns_learned": 0
        }

        # Get or create entity profile
        if entity_id not in self.entity_profiles:
            self.entity_profiles[entity_id] = await self._create_entity_profile(entity_id, events)
            entity_results["profile_created"] = True

        profile = self.entity_profiles[entity_id]

        # Update baselines with new events
        baseline_results = await self._update_entity_baselines(entity_id, events)
        entity_results["baseline_updated"] = baseline_results.get("updated", False)
        entity_results["new_patterns_learned"] = baseline_results.get("new_patterns", 0)

        # Detect anomalies
        anomaly_results = await self._detect_entity_anomalies(entity_id, events)
        entity_results["anomalies"] = anomaly_results.get("anomalies", [])

        # Update risk score
        old_risk_score = profile.risk_score
        new_risk_score = await self._calculate_entity_risk_score(entity_id, events, anomaly_results)
        profile.risk_score = new_risk_score
        entity_results["risk_score_change"] = new_risk_score - old_risk_score

        # Update profile timestamp
        profile.last_analysis = datetime.utcnow()

        return entity_results

    async def _create_entity_profile(self, entity_id: str, events: List[BehaviorEvent]) -> EntityProfile:
        """Create new entity profile from initial events"""
        profile = EntityProfile(
            entity_id=entity_id,
            entity_type="user",  # Default type
            baselines={},
            risk_score=0.5,  # Default risk score
            anomaly_history=[],
            peer_groups=[],
            behavioral_traits={},
            last_analysis=datetime.utcnow(),
            profile_created=datetime.utcnow()
        )

        # Initial behavioral traits analysis
        profile.behavioral_traits = await self._analyze_initial_behavioral_traits(events)

        return profile

    async def _analyze_initial_behavioral_traits(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyze initial behavioral traits from events"""
        if not events:
            return {}

        # Calculate basic traits
        event_count_by_type = defaultdict(int)
        for event in events:
            event_count_by_type[event.event_type] += 1

        # Activity level
        total_events = len(events)
        time_span = (max(event.timestamp for event in events) -
                    min(event.timestamp for event in events)).total_seconds() / 3600  # hours

        activity_level = min(total_events / max(time_span, 1), 1.0)

        # Behavioral diversity
        diversity = len(event_count_by_type) / len(BehaviorType)

        # Time consistency
        timestamps = [event.timestamp for event in events]
        temporal_patterns = await self.time_series_analyzer.analyze_temporal_patterns(events, entity_id="temp")
        consistency = temporal_patterns.get("daily_patterns", {}).get("consistency_score", 0.5)

        return {
            "activity_level": activity_level,
            "behavioral_diversity": diversity,
            "temporal_consistency": consistency,
            "primary_behaviors": [bt.value for bt, count in
                                sorted(event_count_by_type.items(), key=lambda x: x[1], reverse=True)[:3]]
        }

    async def _update_entity_baselines(self, entity_id: str, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Update behavioral baselines for an entity"""
        update_results = {"updated": False, "new_patterns": 0}

        if entity_id not in self.behavioral_baselines:
            self.behavioral_baselines[entity_id] = {}

        # Group events by behavior type
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.behavior_type].append(event)

        # Update baseline for each behavior type
        for behavior_type, behavior_events in events_by_type.items():
            baseline_result = await self._update_behavior_baseline(
                entity_id, behavior_type, behavior_events
            )

            if baseline_result.get("updated"):
                update_results["updated"] = True
                update_results["new_patterns"] += baseline_result.get("new_patterns", 0)

        return update_results

    async def _update_behavior_baseline(
        self,
        entity_id: str,
        behavior_type: BehaviorType,
        events: List[BehaviorEvent]
    ) -> Dict[str, Any]:
        """Update baseline for specific behavior type"""

        # Get existing baseline or create new one
        baselines = self.behavioral_baselines[entity_id]
        if behavior_type not in baselines:
            baselines[behavior_type] = BehaviorBaseline(
                entity_id=entity_id,
                behavior_type=behavior_type,
                baseline_metrics={},
                temporal_patterns={},
                frequency_patterns={},
                typical_volumes={},
                learned_patterns=[],
                last_updated=datetime.utcnow(),
                confidence=0.0,
                sample_size=0
            )

        baseline = baselines[behavior_type]

        # Analyze temporal patterns
        temporal_analysis = await self.time_series_analyzer.analyze_temporal_patterns(events, entity_id)

        # Update baseline metrics
        new_metrics = await self._extract_baseline_metrics(events, temporal_analysis)
        baseline.baseline_metrics.update(new_metrics)

        # Update patterns
        baseline.temporal_patterns = temporal_analysis
        baseline.frequency_patterns = temporal_analysis.get("frequency_analysis", {})

        # Update volumes
        baseline.typical_volumes = {
            "events_per_hour": len(events) / max((events[-1].timestamp - events[0].timestamp).total_seconds() / 3600, 1),
            "events_per_day": len(events) / max((events[-1].timestamp - events[0].timestamp).days + 1, 1),
            "peak_volume": temporal_analysis.get("burst_detection", {}).get("max_burst_intensity", 0)
        }

        # Learn new patterns
        new_patterns = await self._learn_behavioral_patterns(events)
        patterns_added = len(new_patterns)
        baseline.learned_patterns.extend(new_patterns)

        # Keep only recent patterns (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        baseline.learned_patterns = [
            pattern for pattern in baseline.learned_patterns
            if pattern.get("discovered_date", datetime.utcnow()) > cutoff_date
        ]

        # Update metadata
        baseline.sample_size += len(events)
        baseline.last_updated = datetime.utcnow()
        baseline.confidence = min(baseline.sample_size / self.config["minimum_events_for_baseline"], 1.0)

        # Update entity profile
        if entity_id in self.entity_profiles:
            self.entity_profiles[entity_id].baselines[behavior_type] = baseline

        return {
            "updated": True,
            "new_patterns": patterns_added,
            "total_patterns": len(baseline.learned_patterns),
            "confidence": baseline.confidence
        }

    async def _extract_baseline_metrics(self, events: List[BehaviorEvent], temporal_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract baseline metrics from events and temporal analysis"""
        metrics = {}

        if not events:
            return metrics

        # Basic volume metrics
        metrics["event_count"] = len(events)
        metrics["unique_days"] = len(set(event.timestamp.date() for event in events))

        # Temporal metrics
        hourly_dist = temporal_analysis.get("hourly_distribution", {})
        metrics["business_hours_ratio"] = hourly_dist.get("business_hours_ratio", 0.5)
        metrics["after_hours_ratio"] = hourly_dist.get("after_hours_ratio", 0.5)
        metrics["activity_variance"] = hourly_dist.get("activity_variance", 0.0)

        # Frequency metrics
        frequency_analysis = temporal_analysis.get("frequency_analysis", {})
        metrics["avg_interval_hours"] = frequency_analysis.get("avg_interval_seconds", 3600) / 3600
        metrics["regularity_score"] = frequency_analysis.get("regularity_score", 0.5)

        # Risk metrics
        risk_scores = [event.risk_score for event in events if event.risk_score > 0]
        metrics["avg_risk_score"] = statistics.mean(risk_scores) if risk_scores else 0.0
        metrics["max_risk_score"] = max(risk_scores) if risk_scores else 0.0
        metrics["high_risk_ratio"] = len([r for r in risk_scores if r > 0.7]) / len(events)

        return metrics

    async def _learn_behavioral_patterns(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Learn new behavioral patterns from events"""
        patterns = []

        if len(events) < 5:  # Need minimum events to learn patterns
            return patterns

        # Sequential pattern learning
        sequential_patterns = await self._learn_sequential_patterns(events)
        patterns.extend(sequential_patterns)

        # Contextual pattern learning
        contextual_patterns = await self._learn_contextual_patterns(events)
        patterns.extend(contextual_patterns)

        # Metadata pattern learning
        metadata_patterns = await self._learn_metadata_patterns(events)
        patterns.extend(metadata_patterns)

        return patterns

    async def _learn_sequential_patterns(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Learn sequential behavioral patterns"""
        patterns = []

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Look for common sequences
        for window_size in [2, 3, 4]:
            if len(sorted_events) < window_size:
                continue

            sequences = defaultdict(int)
            for i in range(len(sorted_events) - window_size + 1):
                window = sorted_events[i:i + window_size]
                sequence_key = tuple(event.event_type.value for event in window)
                sequences[sequence_key] += 1

            # Identify frequent sequences
            total_windows = len(sorted_events) - window_size + 1
            for sequence, count in sequences.items():
                frequency = count / total_windows
                if frequency > 0.3:  # Appears in >30% of windows
                    patterns.append({
                        "type": "sequential_pattern",
                        "pattern": sequence,
                        "frequency": frequency,
                        "window_size": window_size,
                        "discovered_date": datetime.utcnow(),
                        "confidence": frequency
                    })

        return patterns

    async def _learn_contextual_patterns(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Learn contextual behavioral patterns"""
        patterns = []

        # Group events by context attributes
        context_groups = defaultdict(list)
        for event in events:
            for key, value in event.context.items():
                context_groups[f"{key}:{value}"].append(event)

        # Analyze patterns within contexts
        for context_key, context_events in context_groups.items():
            if len(context_events) >= 3:  # Minimum for pattern
                # Analyze behavior within this context
                behavior_types = [event.event_type.value for event in context_events]
                most_common = max(set(behavior_types), key=behavior_types.count)
                frequency = behavior_types.count(most_common) / len(behavior_types)

                if frequency > 0.6:  # Strong pattern
                    patterns.append({
                        "type": "contextual_pattern",
                        "context": context_key,
                        "primary_behavior": most_common,
                        "frequency": frequency,
                        "event_count": len(context_events),
                        "discovered_date": datetime.utcnow(),
                        "confidence": frequency
                    })

        return patterns

    async def _learn_metadata_patterns(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Learn patterns from event metadata"""
        patterns = []

        # Analyze metadata patterns
        metadata_values = defaultdict(list)
        for event in events:
            for key, value in event.metadata.items():
                if isinstance(value, (str, int, float)):
                    metadata_values[key].append(value)

        for metadata_key, values in metadata_values.items():
            if len(values) >= 5:  # Minimum for analysis
                if isinstance(values[0], (int, float)):
                    # Numerical analysis
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0

                    patterns.append({
                        "type": "metadata_numerical_pattern",
                        "metadata_key": metadata_key,
                        "mean_value": mean_val,
                        "std_deviation": std_val,
                        "sample_size": len(values),
                        "discovered_date": datetime.utcnow(),
                        "confidence": min(len(values) / 20, 1.0)  # More samples = higher confidence
                    })
                else:
                    # Categorical analysis
                    value_counts = defaultdict(int)
                    for value in values:
                        value_counts[str(value)] += 1

                    most_common_value = max(value_counts, key=value_counts.get)
                    frequency = value_counts[most_common_value] / len(values)

                    if frequency > 0.5:  # Dominant pattern
                        patterns.append({
                            "type": "metadata_categorical_pattern",
                            "metadata_key": metadata_key,
                            "dominant_value": most_common_value,
                            "frequency": frequency,
                            "sample_size": len(values),
                            "discovered_date": datetime.utcnow(),
                            "confidence": frequency
                        })

        return patterns

    async def _detect_entity_anomalies(self, entity_id: str, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Detect behavioral anomalies for an entity"""
        anomaly_results = {"anomalies": [], "analysis_summary": {}}

        if not events:
            return anomaly_results

        # Get entity baseline for comparison
        baselines = self.behavioral_baselines.get(entity_id, {})

        # Extract behavioral features for ML analysis
        behavior_features = await self._extract_behavioral_features(entity_id, events, baselines)

        # ML-based anomaly detection
        ml_results = await self.ml_detector.detect_anomalies(behavior_features)

        if ml_results.get("is_anomaly"):
            # Create anomaly record
            anomaly = BehavioralAnomaly(
                anomaly_id=f"anomaly_{entity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                entity_id=entity_id,
                behavior_type=BehaviorType.USER_ACTIVITY,  # Primary type
                anomaly_type=AnomalyType.PATTERN_ANOMALY,
                severity=self._calculate_anomaly_severity(ml_results["confidence"]),
                confidence=self._map_confidence_level(ml_results["confidence"]),
                confidence_score=ml_results["confidence"],
                anomaly_score=ml_results["ensemble_anomaly_score"],
                description=f"ML-detected behavioral anomaly with {ml_results['confidence']:.2f} confidence",
                evidence=ml_results,
                baseline_deviation=await self._calculate_baseline_deviation(entity_id, events, baselines),
                temporal_context=await self._extract_temporal_context(events),
                mitigation_recommendations=await self._generate_anomaly_recommendations(ml_results),
                detection_timestamp=datetime.utcnow(),
                affected_events=[event.event_id for event in events]
            )

            anomaly_results["anomalies"].append(anomaly)

            # Add to entity profile anomaly history
            if entity_id in self.entity_profiles:
                self.entity_profiles[entity_id].anomaly_history.append(anomaly)

        # Rule-based anomaly detection
        rule_anomalies = await self._rule_based_anomaly_detection(entity_id, events, baselines)
        anomaly_results["anomalies"].extend(rule_anomalies)

        # Statistical anomaly detection
        stat_anomalies = await self._statistical_anomaly_detection(entity_id, events, baselines)
        anomaly_results["anomalies"].extend(stat_anomalies)

        anomaly_results["analysis_summary"] = {
            "total_anomalies": len(anomaly_results["anomalies"]),
            "ml_anomalies": 1 if ml_results.get("is_anomaly") else 0,
            "rule_anomalies": len(rule_anomalies),
            "statistical_anomalies": len(stat_anomalies),
            "highest_severity": max([a.severity for a in anomaly_results["anomalies"]], default=RiskLevel.LOW),
            "highest_confidence": max([a.confidence_score for a in anomaly_results["anomalies"]], default=0.0)
        }

        return anomaly_results

    async def _extract_behavioral_features(
        self,
        entity_id: str,
        events: List[BehaviorEvent],
        baselines: Dict[BehaviorType, BehaviorBaseline]
    ) -> np.ndarray:
        """Extract behavioral features for ML analysis"""
        features = []

        # Temporal features
        temporal_analysis = await self.time_series_analyzer.analyze_temporal_patterns(events, entity_id)

        hourly_dist = temporal_analysis.get("hourly_distribution", {})
        features.extend([
            hourly_dist.get("business_hours_ratio", 0.5),
            hourly_dist.get("after_hours_ratio", 0.5),
            hourly_dist.get("activity_variance", 0.0)
        ])

        frequency_analysis = temporal_analysis.get("frequency_analysis", {})
        features.extend([
            min(frequency_analysis.get("avg_interval_seconds", 3600) / 3600, 24),  # Normalize to 24h max
            frequency_analysis.get("regularity_score", 0.5)
        ])

        # Volume features
        features.extend([
            len(events) / 100,  # Normalize event count
            len(set(event.event_type for event in events)) / len(BehaviorType),  # Behavior diversity
            len(set(event.timestamp.date() for event in events))  # Active days
        ])

        # Risk features
        risk_scores = [event.risk_score for event in events if event.risk_score > 0]
        features.extend([
            statistics.mean(risk_scores) if risk_scores else 0.0,
            max(risk_scores) if risk_scores else 0.0,
            len([r for r in risk_scores if r > 0.7]) / len(events),  # High risk ratio
        ])

        # Baseline deviation features
        baseline_deviations = []
        for behavior_type, baseline in baselines.items():
            # Calculate deviation from baseline metrics
            baseline_metrics = baseline.baseline_metrics
            current_metrics = await self._calculate_current_metrics(events, behavior_type)

            for metric_name, baseline_value in baseline_metrics.items():
                current_value = current_metrics.get(metric_name, 0.0)
                deviation = abs(current_value - baseline_value) / max(baseline_value, 0.1)
                baseline_deviations.append(min(deviation, 5.0))  # Cap at 5x deviation

        # Pad or average baseline deviations to fixed size
        while len(baseline_deviations) < 5:
            baseline_deviations.append(0.0)
        if len(baseline_deviations) > 5:
            baseline_deviations = baseline_deviations[:5]

        features.extend(baseline_deviations)

        # Ensure we have exactly 15 features for ML model
        while len(features) < 15:
            features.append(0.0)

        return np.array(features[:15])

    async def _calculate_current_metrics(self, events: List[BehaviorEvent], behavior_type: BehaviorType) -> Dict[str, float]:
        """Calculate current metrics for specific behavior type"""
        behavior_events = [e for e in events if e.behavior_type == behavior_type]

        if not behavior_events:
            return {}

        # Calculate basic metrics
        metrics = {
            "event_count": len(behavior_events),
            "avg_risk_score": statistics.mean([e.risk_score for e in behavior_events if e.risk_score > 0]) if any(e.risk_score > 0 for e in behavior_events) else 0.0,
            "high_risk_ratio": len([e for e in behavior_events if e.risk_score > 0.7]) / len(behavior_events)
        }

        return metrics

    def _calculate_anomaly_severity(self, confidence: float) -> RiskLevel:
        """Calculate anomaly severity based on confidence"""
        if confidence >= 0.9:
            return RiskLevel.CRITICAL
        elif confidence >= 0.75:
            return RiskLevel.HIGH
        elif confidence >= 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _map_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map numerical confidence to confidence level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _calculate_baseline_deviation(
        self,
        entity_id: str,
        events: List[BehaviorEvent],
        baselines: Dict[BehaviorType, BehaviorBaseline]
    ) -> Dict[str, float]:
        """Calculate deviation from behavioral baselines"""
        deviations = {}

        for behavior_type, baseline in baselines.items():
            behavior_events = [e for e in events if e.behavior_type == behavior_type]
            if not behavior_events:
                continue

            current_metrics = await self._calculate_current_metrics(behavior_events, behavior_type)
            baseline_metrics = baseline.baseline_metrics

            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name in current_metrics:
                    current_value = current_metrics[metric_name]
                    deviation = (current_value - baseline_value) / max(baseline_value, 0.1)
                    deviations[f"{behavior_type.value}_{metric_name}"] = deviation

        return deviations

    async def _extract_temporal_context(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Extract temporal context for anomaly analysis"""
        if not events:
            return {}

        timestamps = [event.timestamp for event in events]

        return {
            "time_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
            },
            "time_distribution": {
                "business_hours_events": len([t for t in timestamps if 9 <= t.hour <= 17]),
                "after_hours_events": len([t for t in timestamps if t.hour < 9 or t.hour > 17]),
                "weekend_events": len([t for t in timestamps if t.weekday() >= 5])
            },
            "activity_pattern": {
                "peak_hour": max(set(t.hour for t in timestamps), key=lambda h: sum(1 for t in timestamps if t.hour == h)),
                "active_days": len(set(t.date() for t in timestamps)),
                "events_per_hour": len(events) / max((max(timestamps) - min(timestamps)).total_seconds() / 3600, 1)
            }
        }

    async def _generate_anomaly_recommendations(self, ml_results: Dict[str, Any]) -> List[str]:
        """Generate mitigation recommendations for detected anomalies"""
        recommendations = []

        confidence = ml_results.get("confidence", 0.0)
        feature_importance = ml_results.get("feature_importance", {})

        # High confidence anomalies
        if confidence > 0.8:
            recommendations.append("Immediate investigation required - high confidence anomaly detected")
            recommendations.append("Review user activity logs and system access patterns")
            recommendations.append("Consider temporary access restriction pending investigation")

        # Feature-specific recommendations
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]

        for feature_name, importance in top_features:
            if importance > 0.3:  # Significant feature
                if "login" in feature_name.lower():
                    recommendations.append("Monitor authentication patterns and failed login attempts")
                elif "privilege" in feature_name.lower():
                    recommendations.append("Review privileged access usage and administrative actions")
                elif "data_access" in feature_name.lower():
                    recommendations.append("Audit data access patterns and sensitive file operations")
                elif "network" in feature_name.lower():
                    recommendations.append("Analyze network communication patterns and external connections")

        # General recommendations
        recommendations.extend([
            "Enable enhanced monitoring for this entity",
            "Update behavioral baselines after investigation",
            "Consider peer group analysis for context"
        ])

        return recommendations

    async def _rule_based_anomaly_detection(
        self,
        entity_id: str,
        events: List[BehaviorEvent],
        baselines: Dict[BehaviorType, BehaviorBaseline]
    ) -> List[BehavioralAnomaly]:
        """Rule-based anomaly detection"""
        anomalies = []

        # High volume anomaly
        if len(events) > 100:  # Threshold for high volume
            anomaly = BehavioralAnomaly(
                anomaly_id=f"rule_volume_{entity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                entity_id=entity_id,
                behavior_type=BehaviorType.USER_ACTIVITY,
                anomaly_type=AnomalyType.VOLUME_ANOMALY,
                severity=RiskLevel.MEDIUM,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                anomaly_score=0.8,
                description=f"High volume of activity detected: {len(events)} events",
                evidence={"event_count": len(events), "threshold": 100},
                baseline_deviation={},
                temporal_context=await self._extract_temporal_context(events),
                mitigation_recommendations=["Review activity patterns", "Verify legitimate business need"],
                detection_timestamp=datetime.utcnow(),
                affected_events=[event.event_id for event in events]
            )
            anomalies.append(anomaly)

        # After-hours activity anomaly
        after_hours_events = [e for e in events if e.timestamp.hour < 9 or e.timestamp.hour > 17]
        if len(after_hours_events) > len(events) * 0.5:  # >50% after hours
            anomaly = BehavioralAnomaly(
                anomaly_id=f"rule_timing_{entity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                entity_id=entity_id,
                behavior_type=BehaviorType.USER_ACTIVITY,
                anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                severity=RiskLevel.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                anomaly_score=0.7,
                description=f"Unusual after-hours activity: {len(after_hours_events)} events",
                evidence={"after_hours_events": len(after_hours_events), "total_events": len(events)},
                baseline_deviation={},
                temporal_context=await self._extract_temporal_context(events),
                mitigation_recommendations=["Verify legitimate after-hours access", "Review access justification"],
                detection_timestamp=datetime.utcnow(),
                affected_events=[event.event_id for event in after_hours_events]
            )
            anomalies.append(anomaly)

        return anomalies

    async def _statistical_anomaly_detection(
        self,
        entity_id: str,
        events: List[BehaviorEvent],
        baselines: Dict[BehaviorType, BehaviorBaseline]
    ) -> List[BehavioralAnomaly]:
        """Statistical anomaly detection using baseline comparison"""
        anomalies = []

        if not baselines or len(events) < 5:
            return anomalies

        # Compare current behavior to baselines
        for behavior_type, baseline in baselines.items():
            behavior_events = [e for e in events if e.behavior_type == behavior_type]
            if not behavior_events:
                continue

            # Statistical comparison
            current_count = len(behavior_events)
            typical_volumes = baseline.typical_volumes
            expected_count = typical_volumes.get("events_per_day", 10) * \
                           len(set(event.timestamp.date() for event in behavior_events))

            # Z-score calculation (simplified)
            if expected_count > 0:
                z_score = abs(current_count - expected_count) / max(expected_count * 0.3, 1)  # Assume 30% std dev

                if z_score > 2.5:  # Significant deviation
                    anomaly = BehavioralAnomaly(
                        anomaly_id=f"stat_{behavior_type.value}_{entity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        entity_id=entity_id,
                        behavior_type=behavior_type,
                        anomaly_type=AnomalyType.STATISTICAL_ANOMALY,
                        severity=RiskLevel.HIGH if z_score > 4 else RiskLevel.MEDIUM,
                        confidence=ConfidenceLevel.HIGH if z_score > 4 else ConfidenceLevel.MEDIUM,
                        confidence_score=min(z_score / 5.0, 1.0),
                        anomaly_score=min(z_score / 5.0, 1.0),
                        description=f"Statistical anomaly in {behavior_type.value}: z-score {z_score:.2f}",
                        evidence={
                            "current_count": current_count,
                            "expected_count": expected_count,
                            "z_score": z_score,
                            "baseline_confidence": baseline.confidence
                        },
                        baseline_deviation={"volume_deviation": (current_count - expected_count) / expected_count},
                        temporal_context=await self._extract_temporal_context(behavior_events),
                        mitigation_recommendations=[
                            f"Investigate unusual {behavior_type.value} volume",
                            "Compare with peer group behavior",
                            "Update baseline if behavior change is legitimate"
                        ],
                        detection_timestamp=datetime.utcnow(),
                        affected_events=[event.event_id for event in behavior_events]
                    )
                    anomalies.append(anomaly)

        return anomalies

    async def _calculate_entity_risk_score(
        self,
        entity_id: str,
        events: List[BehaviorEvent],
        anomaly_results: Dict[str, Any]
    ) -> float:
        """Calculate updated risk score for entity"""
        base_risk = 0.3  # Base risk score

        # Factor in anomalies
        anomalies = anomaly_results.get("anomalies", [])
        anomaly_impact = 0.0

        for anomaly in anomalies:
            severity_weights = {
                RiskLevel.CRITICAL: 0.4,
                RiskLevel.HIGH: 0.3,
                RiskLevel.MEDIUM: 0.2,
                RiskLevel.LOW: 0.1
            }
            anomaly_impact += severity_weights.get(anomaly.severity, 0.1) * anomaly.confidence_score

        anomaly_impact = min(anomaly_impact, 0.6)  # Cap anomaly impact

        # Factor in event risk scores
        event_risk = 0.0
        if events:
            risk_scores = [event.risk_score for event in events if event.risk_score > 0]
            if risk_scores:
                event_risk = statistics.mean(risk_scores) * 0.3

        # Factor in historical behavior (if profile exists)
        historical_factor = 0.0
        if entity_id in self.entity_profiles:
            profile = self.entity_profiles[entity_id]
            # Consider recent anomaly history
            recent_anomalies = [a for a in profile.anomaly_history
                             if (datetime.utcnow() - a.detection_timestamp).days <= 7]
            if recent_anomalies:
                historical_factor = min(len(recent_anomalies) * 0.1, 0.2)

        # Calculate final risk score
        total_risk = base_risk + anomaly_impact + event_risk + historical_factor
        return min(total_risk, 1.0)

    async def _update_peer_groups(self) -> Dict[str, Any]:
        """Update peer group analysis"""
        if len(self.entity_profiles) < 3:
            return {"changes": [], "total_groups": 0}

        # Analyze peer groups
        peer_results = await self.peer_analyzer.analyze_peer_groups(list(self.entity_profiles.values()))

        # Update entity profiles with peer group information
        peer_groups = peer_results.get("peer_groups", {})
        changes = []

        for group_name, members in peer_groups.items():
            for member_id in members:
                if member_id in self.entity_profiles:
                    profile = self.entity_profiles[member_id]
                    old_groups = set(profile.peer_groups)
                    profile.peer_groups = [group_name]

                    if set(profile.peer_groups) != old_groups:
                        changes.append({
                            "entity_id": member_id,
                            "old_groups": list(old_groups),
                            "new_groups": profile.peer_groups
                        })

        return {
            "changes": changes,
            "total_groups": len(peer_groups),
            "peer_analysis": peer_results
        }

    async def get_entity_analysis(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis for an entity"""
        if entity_id not in self.entity_profiles:
            return {"error": "Entity profile not found"}

        profile = self.entity_profiles[entity_id]

        analysis = {
            "entity_id": entity_id,
            "profile_summary": {
                "risk_score": profile.risk_score,
                "entity_type": profile.entity_type,
                "profile_age_days": (datetime.utcnow() - profile.profile_created).days,
                "last_analysis": profile.last_analysis.isoformat(),
                "behavioral_traits": profile.behavioral_traits
            },
            "baseline_summary": {},
            "recent_anomalies": [],
            "peer_groups": profile.peer_groups,
            "recommendations": []
        }

        # Baseline summary
        for behavior_type, baseline in profile.baselines.items():
            analysis["baseline_summary"][behavior_type.value] = {
                "confidence": baseline.confidence,
                "sample_size": baseline.sample_size,
                "patterns_learned": len(baseline.learned_patterns),
                "last_updated": baseline.last_updated.isoformat()
            }

        # Recent anomalies (last 7 days)
        recent_anomalies = [
            a for a in profile.anomaly_history
            if (datetime.utcnow() - a.detection_timestamp).days <= 7
        ]

        for anomaly in recent_anomalies:
            analysis["recent_anomalies"].append({
                "anomaly_id": anomaly.anomaly_id,
                "type": anomaly.anomaly_type.value,
                "severity": anomaly.severity.value,
                "confidence": anomaly.confidence.value,
                "description": anomaly.description,
                "detection_time": anomaly.detection_timestamp.isoformat()
            })

        # Generate recommendations
        analysis["recommendations"] = await self._generate_entity_recommendations(profile)

        return analysis

    async def _generate_entity_recommendations(self, profile: EntityProfile) -> List[str]:
        """Generate recommendations for an entity based on profile analysis"""
        recommendations = []

        # Risk-based recommendations
        if profile.risk_score > 0.8:
            recommendations.append("High risk entity - enable enhanced monitoring")
            recommendations.append("Review recent activity patterns for anomalies")
            recommendations.append("Consider additional authentication requirements")
        elif profile.risk_score > 0.6:
            recommendations.append("Medium risk entity - periodic review recommended")
            recommendations.append("Monitor for unusual activity patterns")

        # Baseline-based recommendations
        low_confidence_baselines = [
            bt for bt, baseline in profile.baselines.items()
            if baseline.confidence < 0.7
        ]

        if low_confidence_baselines:
            recommendations.append(f"Improve baseline confidence for: {', '.join([bt.value for bt in low_confidence_baselines])}")
            recommendations.append("Collect more behavioral data for accurate baseline establishment")

        # Anomaly-based recommendations
        recent_anomalies = [
            a for a in profile.anomaly_history
            if (datetime.utcnow() - a.detection_timestamp).days <= 7
        ]

        if len(recent_anomalies) > 3:
            recommendations.append("Multiple recent anomalies detected - investigation recommended")
            recommendations.append("Review user training and security awareness")

        # Peer group recommendations
        if not profile.peer_groups:
            recommendations.append("No peer group assigned - run peer group analysis")

        return recommendations

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        total_entities = len(self.entity_profiles)

        if total_entities == 0:
            return {"total_entities": 0, "status": "no_data"}

        # Risk distribution
        risk_distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for profile in self.entity_profiles.values():
            if profile.risk_score > 0.9:
                risk_distribution["critical"] += 1
            elif profile.risk_score > 0.7:
                risk_distribution["high"] += 1
            elif profile.risk_score > 0.4:
                risk_distribution["medium"] += 1
            else:
                risk_distribution["low"] += 1

        # Anomaly statistics
        total_anomalies = sum(len(p.anomaly_history) for p in self.entity_profiles.values())
        recent_anomalies = sum(
            len([a for a in p.anomaly_history if (datetime.utcnow() - a.detection_timestamp).days <= 7])
            for p in self.entity_profiles.values()
        )

        # Baseline statistics
        baseline_stats = {
            "entities_with_baselines": len([p for p in self.entity_profiles.values() if p.baselines]),
            "total_baselines": sum(len(p.baselines) for p in self.entity_profiles.values()),
            "avg_baseline_confidence": statistics.mean([
                baseline.confidence
                for profile in self.entity_profiles.values()
                for baseline in profile.baselines.values()
            ]) if any(p.baselines for p in self.entity_profiles.values()) else 0.0
        }

        # ML model status
        ml_status = {
            "models_available": SKLEARN_AVAILABLE and self.ml_detector.is_trained,
            "sklearn_available": SKLEARN_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE
        }

        return {
            "total_entities": total_entities,
            "risk_distribution": risk_distribution,
            "anomaly_statistics": {
                "total_anomalies": total_anomalies,
                "recent_anomalies": recent_anomalies,
                "anomaly_rate": total_anomalies / total_entities if total_entities > 0 else 0
            },
            "baseline_statistics": baseline_stats,
            "ml_model_status": ml_status,
            "engine_status": "operational",
            "last_updated": datetime.utcnow().isoformat()
        }

# Global instance for enterprise usage
_behavioral_analytics_engine: Optional[BehavioralAnalyticsEngine] = None

async def get_behavioral_analytics_engine() -> BehavioralAnalyticsEngine:
    """Get global behavioral analytics engine instance"""
    global _behavioral_analytics_engine

    if _behavioral_analytics_engine is None:
        _behavioral_analytics_engine = BehavioralAnalyticsEngine()
        await _behavioral_analytics_engine.initialize()

    return _behavioral_analytics_engine

# Helper functions for creating behavioral events
def create_behavior_event(
    user_id: str,
    event_type: BehaviorType,
    metadata: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    risk_score: float = 0.0
) -> BehaviorEvent:
    """Create a behavioral event"""
    return BehaviorEvent(
        event_id=f"event_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(metadata)) % 10000}",
        user_id=user_id,
        entity_id=user_id,
        event_type=event_type,
        timestamp=datetime.utcnow(),
        metadata=metadata,
        risk_score=risk_score,
        context=context or {}
    )

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize the engine
        engine = await get_behavioral_analytics_engine()

        # Create sample behavioral events
        events = []
        for i in range(50):
            event = create_behavior_event(
                user_id="user_001",
                event_type=BehaviorType.USER_ACTIVITY,
                metadata={
                    "action": "file_access",
                    "file_path": f"/documents/file_{i}.txt",
                    "bytes_accessed": np.random.randint(1000, 50000)
                },
                context={
                    "department": "finance",
                    "location": "office"
                },
                risk_score=np.random.uniform(0.0, 0.8)
            )
            # Vary timestamps
            event.timestamp = datetime.utcnow() - timedelta(
                hours=np.random.randint(0, 24 * 7),
                minutes=np.random.randint(0, 60)
            )
            events.append(event)

        # Process events
        results = await engine.process_behavior_events(events)

        print(f"Behavioral Analytics Results:")
        print(f"Events Processed: {results['processed_events']}")
        print(f"Anomalies Detected: {len(results['anomalies_detected'])}")
        print(f"Entities Updated: {len(results['entity_updates'])}")

        # Get entity analysis
        entity_analysis = await engine.get_entity_analysis("user_001")
        if "error" not in entity_analysis:
            print(f"\nEntity Analysis for user_001:")
            print(f"Risk Score: {entity_analysis['profile_summary']['risk_score']:.3f}")
            print(f"Baselines: {len(entity_analysis['baseline_summary'])}")
            print(f"Recent Anomalies: {len(entity_analysis['recent_anomalies'])}")

        # Get analytics summary
        summary = await engine.get_analytics_summary()
        print(f"\nAnalytics Summary:")
        print(f"Total Entities: {summary['total_entities']}")
        print(f"ML Models Available: {summary['ml_model_status']['models_available']}")
        print(f"Engine Status: {summary['engine_status']}")

    # Run if executed directly
    asyncio.run(main())
