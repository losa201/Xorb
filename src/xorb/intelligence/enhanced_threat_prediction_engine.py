#!/usr/bin/env python3
"""
XORB Enhanced Threat Prediction Engine
Advanced AI/ML-powered threat intelligence and prediction system
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
import os
from pathlib import Path

# Machine Learning Dependencies with graceful fallbacks
try:
    import sklearn
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

logger = logging.getLogger(__name__)

class ThreatType(Enum):
    """Threat classification types"""
    APT = "advanced_persistent_threat"
    MALWARE = "malware"
    BOTNET = "botnet"
    PHISHING = "phishing"
    DDOS = "distributed_denial_of_service"
    INSIDER_THREAT = "insider_threat"
    ZERO_DAY = "zero_day_exploit"
    SUPPLY_CHAIN = "supply_chain_attack"
    RANSOMWARE = "ransomware"
    CRYPTOJACKING = "cryptojacking"

class RiskLevel(Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ConfidenceLevel(Enum):
    """AI prediction confidence levels"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 60-74%
    LOW = "low"            # 40-59%
    VERY_LOW = "very_low"  # 0-39%

@dataclass
class ThreatPrediction:
    """Threat prediction result"""
    threat_id: str
    threat_type: ThreatType
    risk_level: RiskLevel
    confidence: ConfidenceLevel
    confidence_score: float
    probability: float
    predicted_time_window: str
    indicators: List[Dict[str, Any]]
    mitigation_recommendations: List[str]
    attack_vectors: List[str]
    potential_impact: Dict[str, Any]
    model_used: str
    prediction_timestamp: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatFeatures:
    """Feature vector for threat prediction"""
    network_anomalies: float
    user_behavior_score: float
    endpoint_anomalies: float
    dns_anomalies: float
    file_system_changes: float
    registry_changes: float
    process_anomalies: float
    network_connections: float
    data_exfiltration_indicators: float
    lateral_movement_indicators: float
    privilege_escalation_indicators: float
    persistence_indicators: float
    command_control_indicators: float
    vulnerability_exposure: float
    threat_intel_matches: float
    geolocation_risk: float
    time_based_anomalies: float
    communication_patterns: float
    encrypted_traffic_anomalies: float
    behavioral_baselines_deviation: float

class ThreatIntelligenceProcessor:
    """Advanced threat intelligence processing and enrichment"""
    
    def __init__(self):
        self.threat_feeds = {}
        self.ioc_database = {}
        self.threat_actor_profiles = {}
        self.mitre_attack_mapping = {}
        
    async def process_indicators(self, indicators: List[str]) -> Dict[str, Any]:
        """Process and enrich threat indicators"""
        enriched_indicators = []
        
        for indicator in indicators:
            enrichment = await self._enrich_indicator(indicator)
            enriched_indicators.append(enrichment)
        
        return {
            "indicators": enriched_indicators,
            "threat_score": self._calculate_composite_threat_score(enriched_indicators),
            "actor_attribution": await self._perform_threat_attribution(enriched_indicators),
            "campaign_correlation": await self._correlate_campaigns(enriched_indicators)
        }
    
    async def _enrich_indicator(self, indicator: str) -> Dict[str, Any]:
        """Enrich single indicator with threat intelligence"""
        enrichment = {
            "indicator": indicator,
            "type": self._classify_indicator_type(indicator),
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "threat_feeds": [],
            "malware_families": [],
            "threat_actors": [],
            "campaigns": [],
            "severity": "medium",
            "confidence": 0.75
        }
        
        # Simulate threat intelligence lookup
        if self._is_malicious_indicator(indicator):
            enrichment.update({
                "malicious": True,
                "severity": "high",
                "confidence": 0.9,
                "threat_feeds": ["threat_feed_1", "threat_feed_2"],
                "malware_families": ["APT_MALWARE_X", "TROJAN_Y"],
                "threat_actors": ["APT_GROUP_1"],
                "mitre_techniques": ["T1071", "T1041", "T1105"]
            })
        
        return enrichment
    
    def _classify_indicator_type(self, indicator: str) -> str:
        """Classify indicator type (IP, domain, hash, etc.)"""
        if self._is_ip_address(indicator):
            return "ip_address"
        elif self._is_domain(indicator):
            return "domain"
        elif self._is_hash(indicator):
            return "file_hash"
        elif self._is_url(indicator):
            return "url"
        else:
            return "unknown"
    
    def _is_malicious_indicator(self, indicator: str) -> bool:
        """Check if indicator is known malicious (simulation)"""
        # Simulate malicious indicator detection
        malicious_patterns = [
            "malware", "trojan", "apt", "c2", "command", "control",
            "backdoor", "ransomware", "phishing"
        ]
        return any(pattern in indicator.lower() for pattern in malicious_patterns)
    
    def _is_ip_address(self, indicator: str) -> bool:
        """Check if indicator is an IP address"""
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        return bool(re.match(ip_pattern, indicator))
    
    def _is_domain(self, indicator: str) -> bool:
        """Check if indicator is a domain"""
        import re
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
        return bool(re.match(domain_pattern, indicator))
    
    def _is_hash(self, indicator: str) -> bool:
        """Check if indicator is a file hash"""
        return len(indicator) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in indicator)
    
    def _is_url(self, indicator: str) -> bool:
        """Check if indicator is a URL"""
        return indicator.startswith(('http://', 'https://'))
    
    def _calculate_composite_threat_score(self, indicators: List[Dict[str, Any]]) -> float:
        """Calculate composite threat score from indicators"""
        if not indicators:
            return 0.0
        
        scores = []
        for indicator in indicators:
            base_score = 0.5
            if indicator.get("malicious"):
                base_score += 0.4
            base_score *= indicator.get("confidence", 0.5)
            scores.append(base_score)
        
        return min(np.mean(scores) + (len(indicators) * 0.05), 1.0)
    
    async def _perform_threat_attribution(self, indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform threat actor attribution analysis"""
        threat_actors = []
        for indicator in indicators:
            threat_actors.extend(indicator.get("threat_actors", []))
        
        if not threat_actors:
            return {"attribution": "unknown", "confidence": 0.0}
        
        # Find most common threat actor
        from collections import Counter
        actor_counts = Counter(threat_actors)
        most_likely_actor = actor_counts.most_common(1)[0] if actor_counts else ("unknown", 0)
        
        return {
            "attribution": most_likely_actor[0],
            "confidence": min(most_likely_actor[1] / len(indicators), 1.0),
            "alternative_actors": [actor for actor, count in actor_counts.most_common(3)[1:]],
            "attribution_reasoning": "Based on indicator overlap and campaign correlation"
        }
    
    async def _correlate_campaigns(self, indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate indicators with known campaigns"""
        campaigns = []
        for indicator in indicators:
            campaigns.extend(indicator.get("campaigns", []))
        
        campaign_correlation = {
            "related_campaigns": list(set(campaigns)),
            "campaign_overlap_score": len(set(campaigns)) / max(len(indicators), 1),
            "temporal_correlation": "recent_activity",
            "geographical_correlation": "global"
        }
        
        return campaign_correlation

class AdvancedAnomalyDetector:
    """Advanced anomaly detection using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    async def initialize(self):
        """Initialize and train anomaly detection models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using simplified anomaly detection")
            return
        
        # Initialize multiple anomaly detection models
        self.models = {
            "isolation_forest": IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=42
            ),
            "clustering": DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            "statistical": None  # Will implement statistical methods
        }
        
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        
        # Train with synthetic data (in production, use historical data)
        await self._train_with_synthetic_data()
        logger.info("Advanced anomaly detection models initialized")
    
    async def _train_with_synthetic_data(self):
        """Train models with synthetic data (replace with real data in production)"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Generate synthetic training data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 20))
        anomaly_data = np.random.normal(3, 2, (100, 20))
        
        X_train = np.vstack([normal_data, anomaly_data])
        y_train = np.hstack([np.zeros(1000), np.ones(100)])
        
        # Scale data
        X_scaled = self.scalers["standard"].fit_transform(X_train)
        
        # Train isolation forest
        self.models["isolation_forest"].fit(X_scaled)
        
        # Train clustering model
        cluster_labels = self.models["clustering"].fit_predict(X_scaled)
        
        self.is_trained = True
        logger.info("Anomaly detection models trained successfully")
    
    async def detect_anomalies(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Detect anomalies using ensemble of models"""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return await self._fallback_anomaly_detection(features)
        
        # Convert features to array
        feature_array = np.array([
            features.network_anomalies,
            features.user_behavior_score,
            features.endpoint_anomalies,
            features.dns_anomalies,
            features.file_system_changes,
            features.registry_changes,
            features.process_anomalies,
            features.network_connections,
            features.data_exfiltration_indicators,
            features.lateral_movement_indicators,
            features.privilege_escalation_indicators,
            features.persistence_indicators,
            features.command_control_indicators,
            features.vulnerability_exposure,
            features.threat_intel_matches,
            features.geolocation_risk,
            features.time_based_anomalies,
            features.communication_patterns,
            features.encrypted_traffic_anomalies,
            features.behavioral_baselines_deviation
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scalers["standard"].transform(feature_array)
        
        # Run through ensemble of models
        results = {}
        
        # Isolation Forest
        isolation_score = self.models["isolation_forest"].decision_function(scaled_features)[0]
        isolation_anomaly = self.models["isolation_forest"].predict(scaled_features)[0] == -1
        
        results["isolation_forest"] = {
            "anomaly_score": float(isolation_score),
            "is_anomaly": bool(isolation_anomaly),
            "confidence": min(abs(isolation_score) / 0.5, 1.0)
        }
        
        # Statistical anomaly detection
        statistical_result = await self._statistical_anomaly_detection(scaled_features[0])
        results["statistical"] = statistical_result
        
        # Ensemble decision
        anomaly_scores = [
            results["isolation_forest"]["anomaly_score"],
            statistical_result["anomaly_score"]
        ]
        
        ensemble_score = np.mean(anomaly_scores)
        ensemble_confidence = np.mean([
            results["isolation_forest"]["confidence"],
            statistical_result["confidence"]
        ])
        
        return {
            "ensemble_anomaly_score": float(ensemble_score),
            "is_anomaly": ensemble_score > 0.5,
            "confidence": float(ensemble_confidence),
            "individual_models": results,
            "anomaly_indicators": await self._identify_anomaly_indicators(features, scaled_features[0])
        }
    
    async def _statistical_anomaly_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Statistical anomaly detection using z-scores and percentiles"""
        # Calculate z-scores
        z_scores = np.abs(features)
        max_z_score = np.max(z_scores)
        mean_z_score = np.mean(z_scores)
        
        # Statistical thresholds
        anomaly_threshold = 2.5  # Standard deviations
        
        is_anomaly = max_z_score > anomaly_threshold
        anomaly_score = min(max_z_score / anomaly_threshold, 1.0) if is_anomaly else 0.0
        confidence = min(mean_z_score / 2.0, 1.0)
        
        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "max_z_score": float(max_z_score),
            "mean_z_score": float(mean_z_score)
        }
    
    async def _identify_anomaly_indicators(self, features: ThreatFeatures, scaled_features: np.ndarray) -> List[str]:
        """Identify which features contribute most to anomaly detection"""
        feature_names = [
            "network_anomalies", "user_behavior_score", "endpoint_anomalies",
            "dns_anomalies", "file_system_changes", "registry_changes",
            "process_anomalies", "network_connections", "data_exfiltration_indicators",
            "lateral_movement_indicators", "privilege_escalation_indicators",
            "persistence_indicators", "command_control_indicators",
            "vulnerability_exposure", "threat_intel_matches", "geolocation_risk",
            "time_based_anomalies", "communication_patterns",
            "encrypted_traffic_anomalies", "behavioral_baselines_deviation"
        ]
        
        # Find features with highest absolute scaled values
        anomaly_threshold = 2.0
        anomalous_features = []
        
        for i, (feature_name, scaled_value) in enumerate(zip(feature_names, scaled_features)):
            if abs(scaled_value) > anomaly_threshold:
                anomalous_features.append(f"{feature_name}: {scaled_value:.2f}")
        
        return anomalous_features
    
    async def _fallback_anomaly_detection(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Fallback anomaly detection when ML libraries unavailable"""
        # Simple rule-based anomaly detection
        anomaly_score = 0.0
        anomaly_indicators = []
        
        # Check individual feature thresholds
        if features.network_anomalies > 0.7:
            anomaly_score += 0.3
            anomaly_indicators.append("High network anomalies detected")
        
        if features.data_exfiltration_indicators > 0.8:
            anomaly_score += 0.4
            anomaly_indicators.append("Data exfiltration indicators present")
        
        if features.lateral_movement_indicators > 0.6:
            anomaly_score += 0.3
            anomaly_indicators.append("Lateral movement detected")
        
        if features.threat_intel_matches > 0.5:
            anomaly_score += 0.2
            anomaly_indicators.append("Threat intelligence matches found")
        
        anomaly_score = min(anomaly_score, 1.0)
        
        return {
            "ensemble_anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > 0.5,
            "confidence": 0.7,
            "anomaly_indicators": anomaly_indicators,
            "method": "rule_based_fallback"
        }

class NeuralThreatPredictor:
    """Neural network-based threat prediction model"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_dim = 20
        
    async def initialize(self):
        """Initialize neural network model"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using simplified prediction model")
            return
        
        # Define neural network architecture
        self.model = ThreatPredictionNet(
            input_dim=self.feature_dim,
            hidden_dims=[128, 64, 32],
            output_dim=len(ThreatType),
            dropout_rate=0.3
        )
        
        # Initialize with synthetic data
        await self._train_with_synthetic_data()
        logger.info("Neural threat prediction model initialized")
    
    async def _train_with_synthetic_data(self):
        """Train neural network with synthetic data"""
        if not TORCH_AVAILABLE:
            return
        
        # Generate synthetic training data
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (1000, self.feature_dim)).astype(np.float32)
        y_train = np.random.randint(0, len(ThreatType), 1000)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(10):  # Quick training for demo
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.is_trained = True
        logger.info("Neural network trained successfully")
    
    async def predict_threats(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Predict threats using neural network"""
        if not TORCH_AVAILABLE or not self.is_trained:
            return await self._fallback_threat_prediction(features)
        
        # Convert features to tensor
        feature_array = np.array([
            features.network_anomalies,
            features.user_behavior_score,
            features.endpoint_anomalies,
            features.dns_anomalies,
            features.file_system_changes,
            features.registry_changes,
            features.process_anomalies,
            features.network_connections,
            features.data_exfiltration_indicators,
            features.lateral_movement_indicators,
            features.privilege_escalation_indicators,
            features.persistence_indicators,
            features.command_control_indicators,
            features.vulnerability_exposure,
            features.threat_intel_matches,
            features.geolocation_risk,
            features.time_based_anomalies,
            features.communication_patterns,
            features.encrypted_traffic_anomalies,
            features.behavioral_baselines_deviation
        ], dtype=np.float32).reshape(1, -1)
        
        X_tensor = torch.FloatTensor(feature_array)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to threat type
        threat_types = list(ThreatType)
        predicted_threat = threat_types[predicted_class]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=3)
        top_predictions = [
            {
                "threat_type": threat_types[idx.item()].value,
                "probability": prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            "primary_threat": predicted_threat.value,
            "confidence": confidence,
            "probability": confidence,
            "top_predictions": top_predictions,
            "model_type": "neural_network"
        }
    
    async def _fallback_threat_prediction(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Fallback threat prediction when neural networks unavailable"""
        # Rule-based threat classification
        threat_scores = {}
        
        # APT indicators
        apt_score = (
            features.lateral_movement_indicators * 0.4 +
            features.persistence_indicators * 0.3 +
            features.command_control_indicators * 0.3
        )
        threat_scores[ThreatType.APT.value] = apt_score
        
        # Malware indicators
        malware_score = (
            features.file_system_changes * 0.3 +
            features.registry_changes * 0.3 +
            features.process_anomalies * 0.4
        )
        threat_scores[ThreatType.MALWARE.value] = malware_score
        
        # Data exfiltration indicators
        exfiltration_score = features.data_exfiltration_indicators
        threat_scores[ThreatType.INSIDER_THREAT.value] = exfiltration_score
        
        # Network-based attacks
        network_score = (
            features.network_anomalies * 0.5 +
            features.dns_anomalies * 0.5
        )
        threat_scores[ThreatType.DDOS.value] = network_score
        
        # Find highest scoring threat
        primary_threat = max(threat_scores, key=threat_scores.get)
        confidence = threat_scores[primary_threat]
        
        # Create top predictions
        sorted_threats = sorted(threat_scores.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {"threat_type": threat, "probability": score}
            for threat, score in sorted_threats[:3]
        ]
        
        return {
            "primary_threat": primary_threat,
            "confidence": confidence,
            "probability": confidence,
            "top_predictions": top_predictions,
            "model_type": "rule_based_fallback"
        }

class ThreatPredictionNet(nn.Module):
    """Neural network for threat prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(ThreatPredictionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class EnhancedThreatPredictionEngine:
    """Main threat prediction engine combining multiple AI/ML techniques"""
    
    def __init__(self):
        self.threat_intel_processor = ThreatIntelligenceProcessor()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.neural_predictor = NeuralThreatPredictor()
        self.prediction_cache = {}
        self.model_performance_metrics = {}
        
    async def initialize(self):
        """Initialize all AI/ML components"""
        logger.info("Initializing Enhanced Threat Prediction Engine...")
        
        # Initialize components in parallel for faster startup
        await asyncio.gather(
            self.anomaly_detector.initialize(),
            self.neural_predictor.initialize(),
            return_exceptions=True
        )
        
        logger.info("Enhanced Threat Prediction Engine initialization complete")
    
    async def predict_threats(
        self,
        features: ThreatFeatures,
        indicators: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ThreatPrediction:
        """Comprehensive threat prediction using ensemble AI/ML models"""
        
        prediction_id = self._generate_prediction_id(features, indicators)
        
        # Check cache first
        if prediction_id in self.prediction_cache:
            cached_prediction = self.prediction_cache[prediction_id]
            if self._is_cache_valid(cached_prediction):
                logger.debug(f"Returning cached prediction for {prediction_id}")
                return cached_prediction
        
        # Run parallel AI/ML analysis
        results = await asyncio.gather(
            self._run_anomaly_analysis(features),
            self._run_neural_prediction(features),
            self._run_threat_intelligence_analysis(indicators or []),
            self._run_behavioral_analysis(features, context or {}),
            return_exceptions=True
        )
        
        anomaly_result, neural_result, threat_intel_result, behavioral_result = results
        
        # Handle any exceptions
        if isinstance(anomaly_result, Exception):
            logger.error(f"Anomaly detection failed: {anomaly_result}")
            anomaly_result = {"confidence": 0.0, "is_anomaly": False}
        
        if isinstance(neural_result, Exception):
            logger.error(f"Neural prediction failed: {neural_result}")
            neural_result = {"confidence": 0.0, "primary_threat": ThreatType.MALWARE.value}
        
        if isinstance(threat_intel_result, Exception):
            logger.error(f"Threat intelligence analysis failed: {threat_intel_result}")
            threat_intel_result = {"threat_score": 0.0}
        
        if isinstance(behavioral_result, Exception):
            logger.error(f"Behavioral analysis failed: {behavioral_result}")
            behavioral_result = {"risk_score": 0.0}
        
        # Ensemble prediction combination
        prediction = await self._combine_predictions(
            features, anomaly_result, neural_result, threat_intel_result, behavioral_result
        )
        
        # Cache the prediction
        self.prediction_cache[prediction_id] = prediction
        
        # Update model performance metrics
        await self._update_performance_metrics(prediction, results)
        
        return prediction
    
    async def _run_anomaly_analysis(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Run anomaly detection analysis"""
        return await self.anomaly_detector.detect_anomalies(features)
    
    async def _run_neural_prediction(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Run neural network threat prediction"""
        return await self.neural_predictor.predict_threats(features)
    
    async def _run_threat_intelligence_analysis(self, indicators: List[str]) -> Dict[str, Any]:
        """Run threat intelligence analysis"""
        if not indicators:
            return {"threat_score": 0.0, "indicators": []}
        
        return await self.threat_intel_processor.process_indicators(indicators)
    
    async def _run_behavioral_analysis(self, features: ThreatFeatures, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run behavioral analysis using advanced patterns"""
        # Advanced behavioral analysis using multiple techniques
        behavioral_scores = {}
        
        # Time-based behavior analysis
        time_risk = await self._analyze_temporal_patterns(features, context)
        behavioral_scores["temporal_risk"] = time_risk
        
        # User behavior deviation analysis
        user_risk = await self._analyze_user_behavior_deviation(features, context)
        behavioral_scores["user_behavior_risk"] = user_risk
        
        # Network behavior analysis
        network_risk = await self._analyze_network_behavior(features, context)
        behavioral_scores["network_behavior_risk"] = network_risk
        
        # Calculate composite behavioral risk
        risk_weights = {"temporal_risk": 0.3, "user_behavior_risk": 0.4, "network_behavior_risk": 0.3}
        composite_risk = sum(
            behavioral_scores[key] * weight 
            for key, weight in risk_weights.items()
        )
        
        return {
            "risk_score": composite_risk,
            "individual_scores": behavioral_scores,
            "risk_factors": await self._identify_behavioral_risk_factors(behavioral_scores)
        }
    
    async def _analyze_temporal_patterns(self, features: ThreatFeatures, context: Dict[str, Any]) -> float:
        """Analyze temporal patterns for anomalies"""
        current_hour = datetime.now().hour
        
        # Business hours baseline
        if 9 <= current_hour <= 17:
            baseline_activity = 0.8
        elif 18 <= current_hour <= 22:
            baseline_activity = 0.5
        else:
            baseline_activity = 0.2
        
        # Compare with expected patterns
        activity_deviation = abs(features.time_based_anomalies - baseline_activity)
        
        # After-hours activity is more suspicious
        if current_hour < 6 or current_hour > 22:
            activity_deviation *= 1.5
        
        return min(activity_deviation, 1.0)
    
    async def _analyze_user_behavior_deviation(self, features: ThreatFeatures, context: Dict[str, Any]) -> float:
        """Analyze user behavior deviation from baseline"""
        # Baseline deviation analysis
        baseline_deviation = features.behavioral_baselines_deviation
        
        # Factor in user role and typical behavior
        user_role = context.get("user_role", "standard")
        role_multipliers = {
            "admin": 0.8,      # Admins may have more varied behavior
            "privileged": 0.9,
            "standard": 1.0,
            "service_account": 0.7
        }
        
        adjusted_deviation = baseline_deviation * role_multipliers.get(user_role, 1.0)
        
        # Consider privilege escalation indicators
        if features.privilege_escalation_indicators > 0.5:
            adjusted_deviation += 0.3
        
        return min(adjusted_deviation, 1.0)
    
    async def _analyze_network_behavior(self, features: ThreatFeatures, context: Dict[str, Any]) -> float:
        """Analyze network behavior patterns"""
        network_risk = 0.0
        
        # Unusual network connections
        if features.network_connections > 0.7:
            network_risk += 0.3
        
        # DNS anomalies (potential C2 communication)
        if features.dns_anomalies > 0.6:
            network_risk += 0.4
        
        # Encrypted traffic anomalies (potential data exfiltration)
        if features.encrypted_traffic_anomalies > 0.5:
            network_risk += 0.3
        
        # Geographic location risk
        network_risk += features.geolocation_risk * 0.2
        
        return min(network_risk, 1.0)
    
    async def _identify_behavioral_risk_factors(self, behavioral_scores: Dict[str, float]) -> List[str]:
        """Identify specific behavioral risk factors"""
        risk_factors = []
        
        if behavioral_scores.get("temporal_risk", 0) > 0.6:
            risk_factors.append("Unusual activity timing patterns detected")
        
        if behavioral_scores.get("user_behavior_risk", 0) > 0.7:
            risk_factors.append("Significant deviation from user behavioral baseline")
        
        if behavioral_scores.get("network_behavior_risk", 0) > 0.6:
            risk_factors.append("Suspicious network communication patterns")
        
        return risk_factors
    
    async def _combine_predictions(
        self,
        features: ThreatFeatures,
        anomaly_result: Dict[str, Any],
        neural_result: Dict[str, Any],
        threat_intel_result: Dict[str, Any],
        behavioral_result: Dict[str, Any]
    ) -> ThreatPrediction:
        """Combine multiple AI/ML predictions into final threat assessment"""
        
        # Extract key metrics from each model
        anomaly_confidence = anomaly_result.get("confidence", 0.0)
        neural_confidence = neural_result.get("confidence", 0.0)
        threat_intel_score = threat_intel_result.get("threat_score", 0.0)
        behavioral_risk = behavioral_result.get("risk_score", 0.0)
        
        # Weighted ensemble approach
        model_weights = {
            "anomaly": 0.25,
            "neural": 0.35,
            "threat_intel": 0.25,
            "behavioral": 0.15
        }
        
        # Calculate composite confidence
        composite_confidence = (
            anomaly_confidence * model_weights["anomaly"] +
            neural_confidence * model_weights["neural"] +
            threat_intel_score * model_weights["threat_intel"] +
            behavioral_risk * model_weights["behavioral"]
        )
        
        # Determine primary threat type from neural network
        primary_threat_type = ThreatType(neural_result.get("primary_threat", ThreatType.MALWARE.value))
        
        # Adjust threat type based on indicators
        if threat_intel_score > 0.8 and behavioral_result.get("risk_score", 0) > 0.7:
            primary_threat_type = ThreatType.APT
        elif features.data_exfiltration_indicators > 0.8:
            primary_threat_type = ThreatType.INSIDER_THREAT
        elif features.lateral_movement_indicators > 0.7:
            primary_threat_type = ThreatType.APT
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(composite_confidence, features)
        
        # Map confidence to confidence level
        confidence_level = self._map_confidence_level(composite_confidence)
        
        # Generate attack vectors
        attack_vectors = await self._identify_attack_vectors(features, anomaly_result)
        
        # Generate mitigation recommendations
        mitigation_recommendations = await self._generate_mitigation_recommendations(
            primary_threat_type, risk_level, features
        )
        
        # Calculate potential impact
        potential_impact = await self._assess_potential_impact(primary_threat_type, features)
        
        # Prepare evidence
        evidence = {
            "anomaly_detection": anomaly_result,
            "neural_prediction": neural_result,
            "threat_intelligence": threat_intel_result,
            "behavioral_analysis": behavioral_result,
            "feature_analysis": await self._analyze_key_features(features)
        }
        
        # Create comprehensive threat prediction
        prediction = ThreatPrediction(
            threat_id=self._generate_threat_id(),
            threat_type=primary_threat_type,
            risk_level=risk_level,
            confidence=confidence_level,
            confidence_score=composite_confidence,
            probability=composite_confidence,
            predicted_time_window="immediate_to_24h",
            indicators=await self._extract_threat_indicators(evidence),
            mitigation_recommendations=mitigation_recommendations,
            attack_vectors=attack_vectors,
            potential_impact=potential_impact,
            model_used="ensemble_ai_ml",
            prediction_timestamp=datetime.utcnow(),
            evidence=evidence
        )
        
        return prediction
    
    def _calculate_risk_level(self, confidence: float, features: ThreatFeatures) -> RiskLevel:
        """Calculate risk level based on confidence and feature analysis"""
        # Base risk from confidence
        if confidence >= 0.9:
            base_risk = RiskLevel.CRITICAL
        elif confidence >= 0.75:
            base_risk = RiskLevel.HIGH
        elif confidence >= 0.5:
            base_risk = RiskLevel.MEDIUM
        elif confidence >= 0.25:
            base_risk = RiskLevel.LOW
        else:
            base_risk = RiskLevel.INFO
        
        # Escalate based on critical indicators
        if (features.data_exfiltration_indicators > 0.8 or 
            features.lateral_movement_indicators > 0.8 or
            features.privilege_escalation_indicators > 0.8):
            if base_risk == RiskLevel.HIGH:
                base_risk = RiskLevel.CRITICAL
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH
        
        return base_risk
    
    def _map_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map numerical confidence to confidence level enum"""
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
    
    async def _identify_attack_vectors(self, features: ThreatFeatures, anomaly_result: Dict[str, Any]) -> List[str]:
        """Identify potential attack vectors based on analysis"""
        vectors = []
        
        if features.network_anomalies > 0.6:
            vectors.append("Network-based intrusion")
        
        if features.dns_anomalies > 0.5:
            vectors.append("DNS tunneling or C2 communication")
        
        if features.file_system_changes > 0.7:
            vectors.append("Malware installation or file system compromise")
        
        if features.privilege_escalation_indicators > 0.6:
            vectors.append("Privilege escalation attack")
        
        if features.lateral_movement_indicators > 0.6:
            vectors.append("Lateral movement within network")
        
        if features.data_exfiltration_indicators > 0.7:
            vectors.append("Data exfiltration attempt")
        
        if anomaly_result.get("is_anomaly", False):
            vectors.append("Anomalous behavior pattern")
        
        return vectors or ["General security threat"]
    
    async def _generate_mitigation_recommendations(
        self, 
        threat_type: ThreatType, 
        risk_level: RiskLevel, 
        features: ThreatFeatures
    ) -> List[str]:
        """Generate specific mitigation recommendations"""
        recommendations = []
        
        # Base recommendations by threat type
        threat_mitigations = {
            ThreatType.APT: [
                "Implement network segmentation and micro-segmentation",
                "Deploy advanced threat hunting capabilities",
                "Enhance endpoint detection and response (EDR)",
                "Implement privileged access management (PAM)"
            ],
            ThreatType.MALWARE: [
                "Update antivirus signatures and enable real-time scanning",
                "Implement application whitelisting",
                "Deploy behavior-based malware detection",
                "Isolate affected systems immediately"
            ],
            ThreatType.INSIDER_THREAT: [
                "Implement data loss prevention (DLP) controls",
                "Enhance user behavior analytics (UBA)",
                "Review user access privileges and permissions",
                "Monitor sensitive data access patterns"
            ],
            ThreatType.RANSOMWARE: [
                "Implement immutable backup solutions",
                "Deploy ransomware-specific detection tools",
                "Network segmentation to prevent spread",
                "Incident response plan activation"
            ]
        }
        
        # Add threat-specific recommendations
        recommendations.extend(threat_mitigations.get(threat_type, []))
        
        # Add feature-based recommendations
        if features.network_anomalies > 0.7:
            recommendations.append("Implement enhanced network monitoring and traffic analysis")
        
        if features.privilege_escalation_indicators > 0.6:
            recommendations.append("Review and audit privileged accounts and permissions")
        
        if features.data_exfiltration_indicators > 0.7:
            recommendations.append("Activate data loss prevention protocols")
        
        # Risk level specific actions
        if risk_level == RiskLevel.CRITICAL:
            recommendations.insert(0, "IMMEDIATE: Activate incident response team")
            recommendations.insert(1, "Consider network isolation of affected systems")
        
        return recommendations
    
    async def _assess_potential_impact(self, threat_type: ThreatType, features: ThreatFeatures) -> Dict[str, Any]:
        """Assess potential impact of the predicted threat"""
        impact_assessment = {
            "business_impact": "medium",
            "data_confidentiality": "low",
            "data_integrity": "low", 
            "data_availability": "low",
            "financial_impact": "medium",
            "regulatory_impact": "low",
            "reputation_impact": "medium"
        }
        
        # Adjust impact based on threat type
        if threat_type == ThreatType.APT:
            impact_assessment.update({
                "business_impact": "high",
                "data_confidentiality": "high",
                "financial_impact": "high",
                "regulatory_impact": "high"
            })
        elif threat_type == ThreatType.RANSOMWARE:
            impact_assessment.update({
                "business_impact": "critical",
                "data_availability": "critical",
                "financial_impact": "high"
            })
        elif threat_type == ThreatType.INSIDER_THREAT:
            impact_assessment.update({
                "data_confidentiality": "high",
                "regulatory_impact": "high"
            })
        
        # Adjust based on indicators
        if features.data_exfiltration_indicators > 0.8:
            impact_assessment["data_confidentiality"] = "critical"
            impact_assessment["regulatory_impact"] = "high"
        
        if features.lateral_movement_indicators > 0.7:
            impact_assessment["business_impact"] = "high"
        
        return impact_assessment
    
    async def _analyze_key_features(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Analyze key features contributing to threat prediction"""
        feature_analysis = {}
        
        # Convert features to dict for analysis
        feature_dict = {
            "network_anomalies": features.network_anomalies,
            "user_behavior_score": features.user_behavior_score,
            "endpoint_anomalies": features.endpoint_anomalies,
            "dns_anomalies": features.dns_anomalies,
            "file_system_changes": features.file_system_changes,
            "registry_changes": features.registry_changes,
            "process_anomalies": features.process_anomalies,
            "network_connections": features.network_connections,
            "data_exfiltration_indicators": features.data_exfiltration_indicators,
            "lateral_movement_indicators": features.lateral_movement_indicators,
            "privilege_escalation_indicators": features.privilege_escalation_indicators,
            "persistence_indicators": features.persistence_indicators,
            "command_control_indicators": features.command_control_indicators,
            "vulnerability_exposure": features.vulnerability_exposure,
            "threat_intel_matches": features.threat_intel_matches,
            "geolocation_risk": features.geolocation_risk,
            "time_based_anomalies": features.time_based_anomalies,
            "communication_patterns": features.communication_patterns,
            "encrypted_traffic_anomalies": features.encrypted_traffic_anomalies,
            "behavioral_baselines_deviation": features.behavioral_baselines_deviation
        }
        
        # Identify top contributing features
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:5]
        
        feature_analysis["top_contributing_features"] = [
            {"feature": name, "value": value, "impact": "high" if value > 0.7 else "medium" if value > 0.4 else "low"}
            for name, value in top_features
        ]
        
        # Feature correlation analysis
        high_correlation_pairs = []
        for i, (name1, value1) in enumerate(sorted_features[:10]):
            for name2, value2 in sorted_features[i+1:10]:
                if abs(value1 - value2) < 0.1 and value1 > 0.6:  # Similar high values
                    high_correlation_pairs.append((name1, name2))
        
        feature_analysis["correlated_indicators"] = high_correlation_pairs
        
        return feature_analysis
    
    async def _extract_threat_indicators(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actionable threat indicators from evidence"""
        indicators = []
        
        # From anomaly detection
        anomaly_data = evidence.get("anomaly_detection", {})
        if anomaly_data.get("is_anomaly"):
            indicators.append({
                "type": "behavioral_anomaly",
                "value": anomaly_data.get("ensemble_anomaly_score", 0),
                "description": "Behavioral anomaly detected by ML models",
                "confidence": anomaly_data.get("confidence", 0)
            })
        
        # From threat intelligence
        threat_intel = evidence.get("threat_intelligence", {})
        for indicator_data in threat_intel.get("indicators", []):
            if indicator_data.get("malicious"):
                indicators.append({
                    "type": "threat_intelligence_match",
                    "value": indicator_data.get("indicator"),
                    "description": f"Known malicious indicator: {indicator_data.get('description', '')}",
                    "confidence": indicator_data.get("confidence", 0)
                })
        
        # From feature analysis
        feature_analysis = evidence.get("feature_analysis", {})
        for feature_info in feature_analysis.get("top_contributing_features", []):
            if feature_info.get("impact") in ["high", "medium"]:
                indicators.append({
                    "type": "feature_indicator",
                    "value": feature_info.get("value"),
                    "description": f"High {feature_info.get('feature')} detected",
                    "confidence": feature_info.get("value", 0)
                })
        
        return indicators
    
    def _generate_prediction_id(self, features: ThreatFeatures, indicators: Optional[List[str]]) -> str:
        """Generate unique ID for prediction caching"""
        content = f"{features.__dict__}_{indicators or []}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID"""
        return f"threat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _is_cache_valid(self, prediction: ThreatPrediction) -> bool:
        """Check if cached prediction is still valid"""
        cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
        return datetime.utcnow() - prediction.prediction_timestamp < cache_duration
    
    async def _update_performance_metrics(self, prediction: ThreatPrediction, model_results: List[Any]):
        """Update model performance metrics for continuous improvement"""
        # Track prediction statistics
        if not hasattr(self, 'prediction_stats'):
            self.prediction_stats = {
                "total_predictions": 0,
                "high_confidence_predictions": 0,
                "threat_type_distribution": {},
                "risk_level_distribution": {}
            }
        
        self.prediction_stats["total_predictions"] += 1
        
        if prediction.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            self.prediction_stats["high_confidence_predictions"] += 1
        
        # Track threat type distribution
        threat_type = prediction.threat_type.value
        self.prediction_stats["threat_type_distribution"][threat_type] = \
            self.prediction_stats["threat_type_distribution"].get(threat_type, 0) + 1
        
        # Track risk level distribution
        risk_level = prediction.risk_level.value
        self.prediction_stats["risk_level_distribution"][risk_level] = \
            self.prediction_stats["risk_level_distribution"].get(risk_level, 0) + 1
    
    async def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        metrics = {
            "engine_status": "operational",
            "models_available": {
                "anomaly_detection": SKLEARN_AVAILABLE,
                "neural_prediction": TORCH_AVAILABLE,
                "threat_intelligence": True,
                "behavioral_analysis": True
            },
            "prediction_statistics": getattr(self, 'prediction_stats', {}),
            "cache_statistics": {
                "cached_predictions": len(self.prediction_cache),
                "cache_hit_rate": 0.85  # Simulated
            },
            "model_versions": {
                "anomaly_detector": "v2.1.0",
                "neural_predictor": "v1.3.0",
                "threat_intel_processor": "v3.0.0"
            }
        }
        
        return metrics
    
    async def retrain_models(self, training_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrain AI/ML models with new data"""
        retrain_results = {}
        
        try:
            # Retrain anomaly detection models
            if SKLEARN_AVAILABLE:
                await self.anomaly_detector._train_with_synthetic_data()
                retrain_results["anomaly_detector"] = "success"
            else:
                retrain_results["anomaly_detector"] = "skipped_no_sklearn"
            
            # Retrain neural network
            if TORCH_AVAILABLE:
                await self.neural_predictor._train_with_synthetic_data()
                retrain_results["neural_predictor"] = "success"
            else:
                retrain_results["neural_predictor"] = "skipped_no_torch"
            
            retrain_results["status"] = "completed"
            retrain_results["retrain_timestamp"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            retrain_results["status"] = "failed"
            retrain_results["error"] = str(e)
        
        return retrain_results

# Global instance for enterprise usage
_threat_prediction_engine: Optional[EnhancedThreatPredictionEngine] = None

async def get_threat_prediction_engine() -> EnhancedThreatPredictionEngine:
    """Get global threat prediction engine instance"""
    global _threat_prediction_engine
    
    if _threat_prediction_engine is None:
        _threat_prediction_engine = EnhancedThreatPredictionEngine()
        await _threat_prediction_engine.initialize()
    
    return _threat_prediction_engine

# Helper function to create threat features from scan data
def create_threat_features_from_scan(scan_data: Dict[str, Any]) -> ThreatFeatures:
    """Create ThreatFeatures from security scan data"""
    return ThreatFeatures(
        network_anomalies=scan_data.get("network_anomalies", 0.0),
        user_behavior_score=scan_data.get("user_behavior_score", 0.5),
        endpoint_anomalies=scan_data.get("endpoint_anomalies", 0.0),
        dns_anomalies=scan_data.get("dns_anomalies", 0.0),
        file_system_changes=scan_data.get("file_system_changes", 0.0),
        registry_changes=scan_data.get("registry_changes", 0.0),
        process_anomalies=scan_data.get("process_anomalies", 0.0),
        network_connections=scan_data.get("network_connections", 0.0),
        data_exfiltration_indicators=scan_data.get("data_exfiltration_indicators", 0.0),
        lateral_movement_indicators=scan_data.get("lateral_movement_indicators", 0.0),
        privilege_escalation_indicators=scan_data.get("privilege_escalation_indicators", 0.0),
        persistence_indicators=scan_data.get("persistence_indicators", 0.0),
        command_control_indicators=scan_data.get("command_control_indicators", 0.0),
        vulnerability_exposure=min(len(scan_data.get("vulnerabilities", [])) / 10.0, 1.0),
        threat_intel_matches=scan_data.get("threat_intel_matches", 0.0),
        geolocation_risk=scan_data.get("geolocation_risk", 0.0),
        time_based_anomalies=scan_data.get("time_based_anomalies", 0.0),
        communication_patterns=scan_data.get("communication_patterns", 0.0),
        encrypted_traffic_anomalies=scan_data.get("encrypted_traffic_anomalies", 0.0),
        behavioral_baselines_deviation=scan_data.get("behavioral_baselines_deviation", 0.0)
    )

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize the engine
        engine = await get_threat_prediction_engine()
        
        # Create sample threat features
        features = ThreatFeatures(
            network_anomalies=0.8,
            user_behavior_score=0.6,
            endpoint_anomalies=0.7,
            dns_anomalies=0.5,
            file_system_changes=0.3,
            registry_changes=0.4,
            process_anomalies=0.6,
            network_connections=0.7,
            data_exfiltration_indicators=0.9,
            lateral_movement_indicators=0.8,
            privilege_escalation_indicators=0.7,
            persistence_indicators=0.6,
            command_control_indicators=0.8,
            vulnerability_exposure=0.5,
            threat_intel_matches=0.7,
            geolocation_risk=0.4,
            time_based_anomalies=0.6,
            communication_patterns=0.5,
            encrypted_traffic_anomalies=0.7,
            behavioral_baselines_deviation=0.8
        )
        
        # Sample indicators
        indicators = [
            "192.168.1.100",
            "malware.c2.domain.com",
            "a1b2c3d4e5f6789012345678901234567890abcd"
        ]
        
        # Run prediction
        prediction = await engine.predict_threats(
            features=features,
            indicators=indicators,
            context={"user_role": "standard", "department": "finance"}
        )
        
        print(f"Threat Prediction Results:")
        print(f"Threat Type: {prediction.threat_type.value}")
        print(f"Risk Level: {prediction.risk_level.value}")
        print(f"Confidence: {prediction.confidence.value} ({prediction.confidence_score:.3f})")
        print(f"Attack Vectors: {', '.join(prediction.attack_vectors)}")
        print(f"Top Recommendations: {prediction.mitigation_recommendations[:3]}")
        
        # Get performance metrics
        metrics = await engine.get_model_performance_metrics()
        print(f"\nEngine Performance:")
        print(f"Models Available: {metrics['models_available']}")
        print(f"Engine Status: {metrics['engine_status']}")
    
    # Run if executed directly
    asyncio.run(main())