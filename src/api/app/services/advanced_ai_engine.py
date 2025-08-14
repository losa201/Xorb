"""
Advanced AI Engine for XORB Platform
Enterprise-grade AI/ML capabilities for threat detection, behavioral analysis, and predictive security
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from uuid import uuid4
import hashlib
import pickle
import base64
from pathlib import Path

# ML/AI imports with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base_service import XORBService, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)


@dataclass
class ThreatPrediction:
    """Threat prediction result"""
    threat_id: str
    threat_type: str
    confidence: float
    probability: float
    risk_score: int
    predicted_timeframe: str
    indicators: List[str]
    recommended_actions: List[str]
    model_version: str


@dataclass
class BehavioralProfile:
    """User/Entity behavioral profile"""
    entity_id: str
    entity_type: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    anomaly_score: float
    risk_level: str
    behavioral_patterns: List[str]
    temporal_analysis: Dict[str, Any]


@dataclass
class MLModelMetrics:
    """Machine learning model performance metrics"""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime
    training_samples: int
    validation_score: float


class AdvancedAIEngine(XORBService):
    """Advanced AI/ML engine for cybersecurity applications"""

    def __init__(self, **kwargs):
        super().__init__(
            service_id="advanced_ai_engine",
            dependencies=["database", "cache", "vector_store"],
            **kwargs
        )

        # Model configurations
        self.models = {}
        self.feature_extractors = {}
        self.behavioral_baselines = {}

        # AI capabilities flags
        self.capabilities = {
            "torch_models": TORCH_AVAILABLE,
            "sklearn_models": SKLEARN_AVAILABLE,
            "deep_learning": TORCH_AVAILABLE,
            "traditional_ml": SKLEARN_AVAILABLE,
            "behavioral_analysis": True,
            "threat_prediction": True,
            "anomaly_detection": SKLEARN_AVAILABLE
        }

        # Model cache
        self.model_cache = {}
        self.prediction_cache = {}

        logger.info(f"AI Engine initialized with capabilities: {self.capabilities}")

    async def initialize(self) -> bool:
        """Initialize AI engine and models"""
        try:
            logger.info("Initializing Advanced AI Engine...")

            # Initialize base models
            await self._initialize_threat_detection_models()
            await self._initialize_behavioral_models()
            await self._initialize_feature_extractors()

            # Load pre-trained models if available
            await self._load_pretrained_models()

            # Warm up models
            await self._warmup_models()

            logger.info("Advanced AI Engine initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            return False

    async def predict_threats(
        self,
        environmental_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        prediction_horizon: str = "24h"
    ) -> List[ThreatPrediction]:
        """Predict potential threats using AI/ML models"""

        try:
            # Extract features from environmental data
            features = await self._extract_threat_features(environmental_data, historical_data)

            # Run ensemble prediction
            predictions = await self._run_threat_prediction_ensemble(features, prediction_horizon)

            # Post-process and rank predictions
            ranked_predictions = self._rank_and_filter_predictions(predictions)

            # Generate actionable insights
            actionable_predictions = []
            for pred in ranked_predictions:
                actionable_pred = await self._enhance_prediction_with_context(pred, environmental_data)
                actionable_predictions.append(actionable_pred)

            logger.info(f"Generated {len(actionable_predictions)} threat predictions")
            return actionable_predictions

        except Exception as e:
            logger.error(f"Threat prediction failed: {e}")
            return []

    async def analyze_behavioral_anomalies(
        self,
        entity_data: Dict[str, Any],
        baseline_period: str = "30d",
        sensitivity: float = 0.8
    ) -> BehavioralProfile:
        """Analyze behavioral anomalies using advanced ML"""

        entity_id = entity_data.get("entity_id")
        entity_type = entity_data.get("entity_type", "user")

        try:
            # Get or create baseline profile
            baseline = await self._get_behavioral_baseline(entity_id, baseline_period)

            # Extract current behavioral features
            current_features = self._extract_behavioral_features(entity_data)

            # Anomaly detection
            anomaly_score = await self._detect_behavioral_anomalies(
                current_features, baseline, sensitivity
            )

            # Risk assessment
            risk_level = self._assess_behavioral_risk(anomaly_score, current_features, baseline)

            # Pattern analysis
            patterns = await self._analyze_behavioral_patterns(entity_data, baseline)

            # Temporal analysis
            temporal_analysis = await self._perform_temporal_analysis(entity_data, baseline)

            profile = BehavioralProfile(
                entity_id=entity_id,
                entity_type=entity_type,
                baseline_metrics=baseline,
                current_metrics=current_features,
                anomaly_score=anomaly_score,
                risk_level=risk_level,
                behavioral_patterns=patterns,
                temporal_analysis=temporal_analysis
            )

            # Update baseline if within normal parameters
            if anomaly_score < 0.5:
                await self._update_behavioral_baseline(entity_id, current_features)

            logger.info(f"Behavioral analysis complete for {entity_id}: risk={risk_level}, anomaly={anomaly_score:.3f}")
            return profile

        except Exception as e:
            logger.error(f"Behavioral analysis failed for {entity_id}: {e}")
            raise

    async def detect_advanced_threats(
        self,
        network_data: Dict[str, Any],
        endpoint_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect advanced threats using multi-modal AI analysis"""

        detection_id = str(uuid4())

        try:
            # Multi-modal feature extraction
            network_features = self._extract_network_features(network_data)
            endpoint_features = self._extract_endpoint_features(endpoint_data)
            contextual_features = self._extract_contextual_features(context)

            # Fusion of multi-modal features
            fused_features = self._fuse_multimodal_features(
                network_features, endpoint_features, contextual_features
            )

            # Advanced threat detection models
            detection_results = await self._run_advanced_detection_models(fused_features)

            # Threat classification and attribution
            threat_classification = await self._classify_threat_type(detection_results, fused_features)

            # Confidence assessment
            confidence_metrics = self._calculate_detection_confidence(detection_results)

            # Generate detailed analysis
            analysis = {
                "detection_id": detection_id,
                "threat_detected": detection_results["threat_present"],
                "threat_classification": threat_classification,
                "confidence_score": confidence_metrics["overall_confidence"],
                "risk_score": detection_results["risk_score"],
                "attack_stages": detection_results.get("attack_stages", []),
                "ttps_identified": detection_results.get("ttps", []),
                "indicators_of_compromise": detection_results.get("iocs", []),
                "recommended_response": self._generate_response_recommendations(detection_results),
                "model_explanations": detection_results.get("explanations", {}),
                "detection_timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"Advanced threat detection complete: {detection_id}")
            return analysis

        except Exception as e:
            logger.error(f"Advanced threat detection failed: {e}")
            raise

    async def train_adaptive_model(
        self,
        training_data: List[Dict[str, Any]],
        model_type: str,
        validation_split: float = 0.2
    ) -> MLModelMetrics:
        """Train adaptive ML model with new data"""

        model_id = f"{model_type}_{int(datetime.utcnow().timestamp())}"

        try:
            logger.info(f"Training adaptive model: {model_id}")

            # Prepare training data
            X, y = self._prepare_training_data(training_data, model_type)

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Select and train model
            model = await self._train_model_by_type(model_type, X_train, y_train)

            # Validate model
            metrics = await self._validate_model(model, X_val, y_val, model_type)

            # Store model if performance is acceptable
            if metrics.accuracy > 0.7:  # Minimum threshold
                self.models[model_id] = model
                await self._persist_model(model_id, model)
                logger.info(f"Model {model_id} trained successfully with accuracy: {metrics.accuracy:.3f}")
            else:
                logger.warning(f"Model {model_id} performance below threshold: {metrics.accuracy:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    async def generate_security_insights(
        self,
        scan_results: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive security insights using AI analysis"""

        insight_id = str(uuid4())

        try:
            # Multi-source analysis
            vulnerability_insights = await self._analyze_vulnerability_patterns(scan_results)
            threat_insights = await self._analyze_threat_landscape(threat_intelligence)
            behavioral_insights = await self._analyze_behavioral_trends(behavioral_data)

            # Cross-correlation analysis
            correlations = await self._cross_correlate_security_data(
                vulnerability_insights, threat_insights, behavioral_insights
            )

            # Predictive analysis
            predictions = await self._predict_security_trends(correlations)

            # Risk prioritization
            risk_priorities = self._prioritize_security_risks(correlations, predictions)

            # Actionable recommendations
            recommendations = await self._generate_actionable_recommendations(
                correlations, predictions, risk_priorities
            )

            insights = {
                "insight_id": insight_id,
                "analysis_scope": {
                    "vulnerabilities_analyzed": len(scan_results.get("vulnerabilities", [])),
                    "threat_indicators": len(threat_intelligence.get("indicators", [])),
                    "behavioral_entities": len(behavioral_data.get("entities", []))
                },
                "key_findings": {
                    "vulnerability_insights": vulnerability_insights,
                    "threat_insights": threat_insights,
                    "behavioral_insights": behavioral_insights,
                    "cross_correlations": correlations
                },
                "predictive_analysis": predictions,
                "risk_prioritization": risk_priorities,
                "actionable_recommendations": recommendations,
                "confidence_metrics": self._calculate_insight_confidence(correlations),
                "generated_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Generated comprehensive security insights: {insight_id}")
            return insights

        except Exception as e:
            logger.error(f"Security insight generation failed: {e}")
            raise

    # Private helper methods for AI/ML operations
    async def _initialize_threat_detection_models(self):
        """Initialize threat detection models"""

        if SKLEARN_AVAILABLE:
            # Anomaly detection model
            self.models["anomaly_detector"] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )

            # Threat classifier
            self.models["threat_classifier"] = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=10
            )

            logger.info("Initialized sklearn-based threat detection models")

        if TORCH_AVAILABLE:
            # Deep learning threat detection model
            self.models["deep_threat_detector"] = self._create_deep_threat_model()
            logger.info("Initialized PyTorch-based deep learning models")

    async def _initialize_behavioral_models(self):
        """Initialize behavioral analysis models"""

        if SKLEARN_AVAILABLE:
            # Behavioral clustering
            self.models["behavioral_clustering"] = DBSCAN(
                eps=0.5,
                min_samples=5
            )

            # Feature scaling
            self.feature_extractors["scaler"] = StandardScaler()

            # Dimensionality reduction
            self.feature_extractors["pca"] = PCA(n_components=10)

            logger.info("Initialized behavioral analysis models")

    async def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""

        # Network feature extractor
        self.feature_extractors["network"] = NetworkFeatureExtractor()

        # Endpoint feature extractor
        self.feature_extractors["endpoint"] = EndpointFeatureExtractor()

        # Behavioral feature extractor
        self.feature_extractors["behavioral"] = BehavioralFeatureExtractor()

        logger.info("Initialized feature extractors")

    async def _load_pretrained_models(self):
        """Load pre-trained models if available"""

        model_dir = Path("models/pretrained")
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                try:
                    model_name = model_file.stem
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded pre-trained model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {e}")

    async def _warmup_models(self):
        """Warm up models with dummy data"""

        try:
            # Generate dummy data for warmup
            dummy_network_features = np.random.random((10, 20))
            dummy_behavioral_features = np.random.random((10, 15))

            # Warmup sklearn models
            if SKLEARN_AVAILABLE and "anomaly_detector" in self.models:
                self.models["anomaly_detector"].fit(dummy_network_features)

            logger.info("Model warmup complete")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _create_deep_threat_model(self):
        """Create deep learning threat detection model"""

        if not TORCH_AVAILABLE:
            return None

        class DeepThreatDetector(nn.Module):
            def __init__(self, input_size=100, hidden_sizes=[64, 32, 16], num_classes=5):
                super().__init__()
                layers = []
                prev_size = input_size

                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_size = hidden_size

                layers.append(nn.Linear(prev_size, num_classes))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        return DeepThreatDetector()

    async def _extract_threat_features(
        self,
        environmental_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract features for threat prediction"""

        features = []

        # Environmental features
        env_features = [
            len(environmental_data.get("open_ports", [])),
            len(environmental_data.get("services", [])),
            len(environmental_data.get("vulnerabilities", [])),
            environmental_data.get("security_score", 0),
            environmental_data.get("patch_level", 0)
        ]
        features.extend(env_features)

        # Historical trend features
        if historical_data:
            trend_features = self._calculate_trend_features(historical_data)
            features.extend(trend_features)
        else:
            features.extend([0] * 10)  # Placeholder for missing historical data

        # Temporal features
        now = datetime.utcnow()
        temporal_features = [
            now.hour / 24.0,  # Hour of day normalized
            now.weekday() / 6.0,  # Day of week normalized
            (now.day - 1) / 30.0,  # Day of month normalized
        ]
        features.extend(temporal_features)

        return np.array(features, dtype=np.float32)

    def _calculate_trend_features(self, historical_data: List[Dict[str, Any]]) -> List[float]:
        """Calculate trend features from historical data"""

        if len(historical_data) < 2:
            return [0] * 10

        # Extract time series of key metrics
        timestamps = [datetime.fromisoformat(d.get("timestamp", datetime.utcnow().isoformat())) for d in historical_data]
        vulnerabilities = [len(d.get("vulnerabilities", [])) for d in historical_data]
        security_scores = [d.get("security_score", 0) for d in historical_data]

        # Calculate trends
        vuln_trend = (vulnerabilities[-1] - vulnerabilities[0]) / len(vulnerabilities)
        score_trend = (security_scores[-1] - security_scores[0]) / len(security_scores)

        # Calculate volatility
        vuln_volatility = np.std(vulnerabilities) if vulnerabilities else 0
        score_volatility = np.std(security_scores) if security_scores else 0

        # Calculate moving averages
        recent_avg_vulns = np.mean(vulnerabilities[-5:]) if len(vulnerabilities) >= 5 else np.mean(vulnerabilities)
        recent_avg_score = np.mean(security_scores[-5:]) if len(security_scores) >= 5 else np.mean(security_scores)

        return [
            vuln_trend, score_trend, vuln_volatility, score_volatility,
            recent_avg_vulns, recent_avg_score,
            len(historical_data), max(vulnerabilities) if vulnerabilities else 0,
            min(security_scores) if security_scores else 0, np.mean(vulnerabilities) if vulnerabilities else 0
        ]

    async def _run_threat_prediction_ensemble(
        self,
        features: np.ndarray,
        prediction_horizon: str
    ) -> List[Dict[str, Any]]:
        """Run ensemble of threat prediction models"""

        predictions = []

        # Simple rule-based predictions (always available)
        rule_based_pred = self._rule_based_threat_prediction(features, prediction_horizon)
        predictions.append(rule_based_pred)

        # ML-based predictions (if available)
        if SKLEARN_AVAILABLE and "threat_classifier" in self.models:
            ml_pred = await self._ml_threat_prediction(features, prediction_horizon)
            predictions.append(ml_pred)

        # Deep learning predictions (if available)
        if TORCH_AVAILABLE and "deep_threat_detector" in self.models:
            dl_pred = await self._deep_learning_threat_prediction(features, prediction_horizon)
            predictions.append(dl_pred)

        return predictions

    def _rule_based_threat_prediction(self, features: np.ndarray, horizon: str) -> Dict[str, Any]:
        """Rule-based threat prediction as fallback"""

        # Simple rules based on feature values
        vuln_count = features[2] if len(features) > 2 else 0
        security_score = features[3] if len(features) > 3 else 0

        # Determine threat probability
        if vuln_count > 10 or security_score < 30:
            threat_prob = 0.8
            threat_type = "vulnerability_exploitation"
        elif vuln_count > 5 or security_score < 50:
            threat_prob = 0.5
            threat_type = "opportunistic_attack"
        else:
            threat_prob = 0.2
            threat_type = "low_risk_probing"

        return {
            "model": "rule_based",
            "threat_type": threat_type,
            "probability": threat_prob,
            "confidence": 0.6,
            "indicators": [f"vulnerability_count_{vuln_count}", f"security_score_{security_score}"]
        }

    async def _ml_threat_prediction(self, features: np.ndarray, horizon: str) -> Dict[str, Any]:
        """ML-based threat prediction"""

        try:
            # Ensure features are in correct shape
            features_2d = features.reshape(1, -1)

            # Predict using trained model (or train if not available)
            model = self.models.get("threat_classifier")
            if model is None:
                # Train a simple model with dummy data for demonstration
                model = self._train_dummy_threat_model()
                self.models["threat_classifier"] = model

            # Make prediction
            prediction_prob = model.predict_proba(features_2d)[0]
            predicted_class = model.predict(features_2d)[0]

            threat_types = ["benign", "malware", "phishing", "dos", "intrusion"]
            threat_type = threat_types[predicted_class] if predicted_class < len(threat_types) else "unknown"

            return {
                "model": "sklearn_ml",
                "threat_type": threat_type,
                "probability": float(np.max(prediction_prob)),
                "confidence": 0.75,
                "indicators": [f"ml_feature_{i}" for i in range(min(3, len(features)))]
            }

        except Exception as e:
            logger.warning(f"ML threat prediction failed: {e}")
            return self._rule_based_threat_prediction(features, horizon)

    def _train_dummy_threat_model(self):
        """Train a dummy threat model for demonstration"""

        if not SKLEARN_AVAILABLE:
            return None

        # Generate dummy training data
        np.random.seed(42)
        X_dummy = np.random.random((100, 18))  # Match feature dimension
        y_dummy = np.random.randint(0, 5, 100)  # 5 threat classes

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_dummy, y_dummy)

        return model

    async def _deep_learning_threat_prediction(self, features: np.ndarray, horizon: str) -> Dict[str, Any]:
        """Deep learning-based threat prediction"""

        if not TORCH_AVAILABLE:
            return self._rule_based_threat_prediction(features, horizon)

        try:
            model = self.models.get("deep_threat_detector")
            if model is None:
                return self._rule_based_threat_prediction(features, horizon)

            # Prepare features for PyTorch
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Pad or truncate features to match model input size
            if features_tensor.size(1) < 100:
                padding = torch.zeros(1, 100 - features_tensor.size(1))
                features_tensor = torch.cat([features_tensor, padding], dim=1)
            elif features_tensor.size(1) > 100:
                features_tensor = features_tensor[:, :100]

            # Make prediction
            with torch.no_grad():
                output = model(features_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

            threat_types = ["benign", "malware", "phishing", "dos", "intrusion"]
            threat_type = threat_types[predicted_class] if predicted_class < len(threat_types) else "unknown"

            return {
                "model": "deep_learning",
                "threat_type": threat_type,
                "probability": float(probabilities[predicted_class]),
                "confidence": 0.85,
                "indicators": [f"dl_feature_{i}" for i in range(min(5, len(features)))]
            }

        except Exception as e:
            logger.warning(f"Deep learning prediction failed: {e}")
            return self._rule_based_threat_prediction(features, horizon)

    def _rank_and_filter_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank and filter threat predictions"""

        # Filter out low-confidence predictions
        filtered = [p for p in predictions if p.get("confidence", 0) > 0.5]

        # Sort by probability * confidence
        ranked = sorted(
            filtered,
            key=lambda x: x.get("probability", 0) * x.get("confidence", 0),
            reverse=True
        )

        return ranked[:5]  # Return top 5 predictions

    async def _enhance_prediction_with_context(
        self,
        prediction: Dict[str, Any],
        environmental_data: Dict[str, Any]
    ) -> ThreatPrediction:
        """Enhance prediction with contextual information"""

        threat_id = str(uuid4())

        # Calculate risk score
        risk_score = int(prediction.get("probability", 0) * prediction.get("confidence", 0) * 100)

        # Determine timeframe based on threat type
        timeframe_mapping = {
            "malware": "1-3 days",
            "phishing": "immediate",
            "dos": "1-2 hours",
            "intrusion": "12-24 hours",
            "vulnerability_exploitation": "2-7 days"
        }

        timeframe = timeframe_mapping.get(prediction.get("threat_type", ""), "24-48 hours")

        # Generate recommendations
        recommendations = self._generate_threat_recommendations(prediction, environmental_data)

        return ThreatPrediction(
            threat_id=threat_id,
            threat_type=prediction.get("threat_type", "unknown"),
            confidence=prediction.get("confidence", 0),
            probability=prediction.get("probability", 0),
            risk_score=risk_score,
            predicted_timeframe=timeframe,
            indicators=prediction.get("indicators", []),
            recommended_actions=recommendations,
            model_version=prediction.get("model", "ensemble")
        )

    def _generate_threat_recommendations(
        self,
        prediction: Dict[str, Any],
        environmental_data: Dict[str, Any]
    ) -> List[str]:
        """Generate threat-specific recommendations"""

        threat_type = prediction.get("threat_type", "")
        recommendations = []

        if "malware" in threat_type:
            recommendations.extend([
                "Update endpoint protection signatures",
                "Scan all systems for indicators of compromise",
                "Review email security settings"
            ])

        if "phishing" in threat_type:
            recommendations.extend([
                "Enhance email filtering rules",
                "Conduct user awareness training",
                "Monitor for suspicious email patterns"
            ])

        if "intrusion" in threat_type:
            recommendations.extend([
                "Review access logs for anomalies",
                "Strengthen network segmentation",
                "Enable additional monitoring"
            ])

        if "vulnerability" in threat_type:
            recommendations.extend([
                "Prioritize vulnerability patching",
                "Implement compensating controls",
                "Conduct impact assessment"
            ])

        # Add general recommendations
        if prediction.get("probability", 0) > 0.7:
            recommendations.append("Consider increasing security monitoring level")

        return recommendations[:5]  # Limit to top 5 recommendations

    async def _get_behavioral_baseline(self, entity_id: str, period: str) -> Dict[str, float]:
        """Get or create behavioral baseline for entity"""

        if entity_id in self.behavioral_baselines:
            return self.behavioral_baselines[entity_id]

        # Create default baseline
        baseline = {
            "login_frequency": 8.0,
            "session_duration": 4.5,
            "data_access_rate": 15.0,
            "location_variance": 0.2,
            "time_variance": 0.3,
            "privilege_usage": 0.1,
            "network_activity": 25.0,
            "file_operations": 12.0,
            "system_commands": 3.0,
            "error_rate": 0.05
        }

        self.behavioral_baselines[entity_id] = baseline
        return baseline

    def _extract_behavioral_features(self, entity_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract behavioral features from entity data"""

        return {
            "login_frequency": entity_data.get("logins_per_day", 8.0),
            "session_duration": entity_data.get("avg_session_hours", 4.5),
            "data_access_rate": entity_data.get("files_accessed_per_hour", 15.0),
            "location_variance": entity_data.get("location_entropy", 0.2),
            "time_variance": entity_data.get("time_pattern_variance", 0.3),
            "privilege_usage": entity_data.get("admin_actions_ratio", 0.1),
            "network_activity": entity_data.get("network_requests_per_hour", 25.0),
            "file_operations": entity_data.get("file_ops_per_hour", 12.0),
            "system_commands": entity_data.get("commands_per_hour", 3.0),
            "error_rate": entity_data.get("error_ratio", 0.05)
        }

    async def _detect_behavioral_anomalies(
        self,
        current_features: Dict[str, float],
        baseline: Dict[str, float],
        sensitivity: float
    ) -> float:
        """Detect behavioral anomalies using statistical analysis"""

        anomaly_scores = []

        for feature, current_value in current_features.items():
            baseline_value = baseline.get(feature, current_value)

            if baseline_value == 0:
                baseline_value = 0.01  # Avoid division by zero

            # Calculate relative difference
            relative_diff = abs(current_value - baseline_value) / baseline_value

            # Apply sensitivity scaling
            anomaly_score = min(relative_diff * sensitivity, 1.0)
            anomaly_scores.append(anomaly_score)

        # Return weighted average anomaly score
        return np.mean(anomaly_scores)

    def _assess_behavioral_risk(
        self,
        anomaly_score: float,
        current_features: Dict[str, float],
        baseline: Dict[str, float]
    ) -> str:
        """Assess behavioral risk level"""

        if anomaly_score > 0.8:
            return "critical"
        elif anomaly_score > 0.6:
            return "high"
        elif anomaly_score > 0.4:
            return "medium"
        elif anomaly_score > 0.2:
            return "low"
        else:
            return "normal"

    async def _analyze_behavioral_patterns(
        self,
        entity_data: Dict[str, Any],
        baseline: Dict[str, float]
    ) -> List[str]:
        """Analyze behavioral patterns"""

        patterns = []

        # Analyze login patterns
        login_freq = entity_data.get("logins_per_day", 0)
        if login_freq > baseline.get("login_frequency", 0) * 2:
            patterns.append("excessive_login_activity")
        elif login_freq < baseline.get("login_frequency", 0) * 0.5:
            patterns.append("reduced_login_activity")

        # Analyze time patterns
        time_variance = entity_data.get("time_pattern_variance", 0)
        if time_variance > baseline.get("time_variance", 0) * 1.5:
            patterns.append("unusual_time_patterns")

        # Analyze location patterns
        location_variance = entity_data.get("location_entropy", 0)
        if location_variance > baseline.get("location_variance", 0) * 2:
            patterns.append("unusual_location_access")

        # Analyze privilege usage
        privilege_usage = entity_data.get("admin_actions_ratio", 0)
        if privilege_usage > baseline.get("privilege_usage", 0) * 3:
            patterns.append("elevated_privilege_usage")

        return patterns

    async def _perform_temporal_analysis(
        self,
        entity_data: Dict[str, Any],
        baseline: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform temporal analysis of behavioral data"""

        return {
            "peak_activity_hours": entity_data.get("peak_hours", [9, 10, 11, 14, 15]),
            "weekend_activity_ratio": entity_data.get("weekend_ratio", 0.1),
            "late_night_activity": entity_data.get("after_hours_ratio", 0.05),
            "pattern_consistency": entity_data.get("pattern_consistency", 0.8),
            "trend_direction": "stable"  # Would be calculated from historical data
        }

    async def _update_behavioral_baseline(self, entity_id: str, current_features: Dict[str, float]):
        """Update behavioral baseline with new data"""

        if entity_id not in self.behavioral_baselines:
            self.behavioral_baselines[entity_id] = current_features
            return

        # Exponential moving average update
        alpha = 0.1  # Learning rate
        baseline = self.behavioral_baselines[entity_id]

        for feature, value in current_features.items():
            if feature in baseline:
                baseline[feature] = alpha * value + (1 - alpha) * baseline[feature]
            else:
                baseline[feature] = value

    def _extract_network_features(self, network_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from network data"""

        features = [
            network_data.get("packet_count", 0),
            network_data.get("byte_count", 0),
            network_data.get("connection_count", 0),
            network_data.get("unique_ips", 0),
            network_data.get("port_diversity", 0),
            network_data.get("protocol_diversity", 0),
            network_data.get("dns_queries", 0),
            network_data.get("failed_connections", 0),
            network_data.get("avg_packet_size", 0),
            network_data.get("traffic_variance", 0)
        ]

        return np.array(features, dtype=np.float32)

    def _extract_endpoint_features(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from endpoint data"""

        features = [
            endpoint_data.get("process_count", 0),
            endpoint_data.get("cpu_usage", 0),
            endpoint_data.get("memory_usage", 0),
            endpoint_data.get("disk_operations", 0),
            endpoint_data.get("network_connections", 0),
            endpoint_data.get("registry_modifications", 0),
            endpoint_data.get("file_modifications", 0),
            endpoint_data.get("new_processes", 0),
            endpoint_data.get("terminated_processes", 0),
            endpoint_data.get("privilege_escalations", 0)
        ]

        return np.array(features, dtype=np.float32)

    def _extract_contextual_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract contextual features"""

        features = [
            context.get("time_of_day", 12) / 24.0,  # Normalized hour
            context.get("day_of_week", 3) / 7.0,    # Normalized day
            context.get("security_level", 1),        # Security posture
            context.get("user_count", 1),            # Active users
            context.get("system_load", 0.5),         # System load
        ]

        return np.array(features, dtype=np.float32)

    def _fuse_multimodal_features(
        self,
        network_features: np.ndarray,
        endpoint_features: np.ndarray,
        contextual_features: np.ndarray
    ) -> np.ndarray:
        """Fuse multi-modal features"""

        # Simple concatenation fusion
        return np.concatenate([network_features, endpoint_features, contextual_features])

    async def _run_advanced_detection_models(self, features: np.ndarray) -> Dict[str, Any]:
        """Run advanced detection models on fused features"""

        results = {
            "threat_present": False,
            "risk_score": 0,
            "confidence": 0.0,
            "attack_stages": [],
            "ttps": [],
            "iocs": [],
            "explanations": {}
        }

        # Rule-based detection
        rule_result = self._rule_based_detection(features)
        results.update(rule_result)

        # ML-based detection (if available)
        if SKLEARN_AVAILABLE and "anomaly_detector" in self.models:
            ml_result = await self._ml_based_detection(features)
            results = self._merge_detection_results(results, ml_result)

        return results

    def _rule_based_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Rule-based threat detection"""

        # Simple threshold-based rules
        threat_score = 0

        # Check for suspicious network activity
        if len(features) > 0 and features[0] > 1000:  # High packet count
            threat_score += 30

        # Check for suspicious endpoint activity
        if len(features) > 10 and features[10] > 80:  # High CPU usage
            threat_score += 20

        # Check for unusual process activity
        if len(features) > 15 and features[15] > 10:  # High new processes
            threat_score += 25

        threat_present = threat_score > 50

        return {
            "threat_present": threat_present,
            "risk_score": min(threat_score, 100),
            "confidence": 0.6,
            "attack_stages": ["reconnaissance"] if threat_present else [],
            "explanations": {"rule_based": f"Threat score: {threat_score}"}
        }

    async def _ml_based_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """ML-based threat detection"""

        try:
            # Ensure features are in correct shape
            features_2d = features.reshape(1, -1)

            # Anomaly detection
            anomaly_detector = self.models.get("anomaly_detector")
            if anomaly_detector is not None:
                anomaly_score = anomaly_detector.decision_function(features_2d)[0]
                is_anomaly = anomaly_detector.predict(features_2d)[0] == -1

                return {
                    "threat_present": is_anomaly,
                    "risk_score": int(abs(anomaly_score) * 100),
                    "confidence": 0.8,
                    "attack_stages": ["lateral_movement"] if is_anomaly else [],
                    "explanations": {"ml_based": f"Anomaly score: {anomaly_score:.3f}"}
                }

        except Exception as e:
            logger.warning(f"ML detection failed: {e}")

        return {"threat_present": False, "risk_score": 0, "confidence": 0.0}

    def _merge_detection_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge detection results from multiple models"""

        return {
            "threat_present": result1.get("threat_present", False) or result2.get("threat_present", False),
            "risk_score": max(result1.get("risk_score", 0), result2.get("risk_score", 0)),
            "confidence": (result1.get("confidence", 0) + result2.get("confidence", 0)) / 2,
            "attack_stages": list(set(result1.get("attack_stages", []) + result2.get("attack_stages", []))),
            "ttps": list(set(result1.get("ttps", []) + result2.get("ttps", []))),
            "iocs": list(set(result1.get("iocs", []) + result2.get("iocs", []))),
            "explanations": {**result1.get("explanations", {}), **result2.get("explanations", {})}
        }

    async def _classify_threat_type(self, detection_results: Dict[str, Any], features: np.ndarray) -> Dict[str, Any]:
        """Classify the type of detected threat"""

        if not detection_results.get("threat_present", False):
            return {"type": "none", "confidence": 1.0}

        # Simple classification based on features and detection results
        risk_score = detection_results.get("risk_score", 0)

        if risk_score > 80:
            threat_type = "advanced_persistent_threat"
        elif risk_score > 60:
            threat_type = "malware_infection"
        elif risk_score > 40:
            threat_type = "suspicious_activity"
        else:
            threat_type = "anomalous_behavior"

        return {
            "type": threat_type,
            "confidence": detection_results.get("confidence", 0.5),
            "subcategory": "unknown",
            "family": "unknown"
        }

    def _calculate_detection_confidence(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detection confidence metrics"""

        base_confidence = detection_results.get("confidence", 0.5)
        risk_score = detection_results.get("risk_score", 0)

        # Adjust confidence based on risk score
        risk_factor = min(risk_score / 100.0, 1.0)
        overall_confidence = (base_confidence + risk_factor) / 2

        return {
            "overall_confidence": overall_confidence,
            "detection_confidence": base_confidence,
            "risk_confidence": risk_factor,
            "model_agreement": 0.7  # Simulated model agreement score
        }

    def _generate_response_recommendations(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate response recommendations based on detection"""

        recommendations = []
        risk_score = detection_results.get("risk_score", 0)

        if risk_score > 70:
            recommendations.extend([
                {"action": "isolate_affected_systems", "priority": "immediate", "effort": "low"},
                {"action": "collect_forensic_evidence", "priority": "high", "effort": "medium"},
                {"action": "notify_security_team", "priority": "immediate", "effort": "low"}
            ])
        elif risk_score > 40:
            recommendations.extend([
                {"action": "increase_monitoring", "priority": "high", "effort": "low"},
                {"action": "validate_findings", "priority": "medium", "effort": "medium"},
                {"action": "review_access_logs", "priority": "medium", "effort": "high"}
            ])
        else:
            recommendations.extend([
                {"action": "continue_monitoring", "priority": "low", "effort": "low"},
                {"action": "update_baselines", "priority": "low", "effort": "medium"}
            ])

        return recommendations

    def _prepare_training_data(self, training_data: List[Dict[str, Any]], model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""

        X = []
        y = []

        for sample in training_data:
            features = sample.get("features", [])
            label = sample.get("label", 0)

            if features:
                X.append(features)
                y.append(label)

        return np.array(X, dtype=np.float32), np.array(y)

    async def _train_model_by_type(self, model_type: str, X: np.ndarray, y: np.ndarray):
        """Train model based on type"""

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available for model training")

        if model_type == "anomaly_detection":
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)
        elif model_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    async def _validate_model(self, model, X_val: np.ndarray, y_val: np.ndarray, model_type: str) -> MLModelMetrics:
        """Validate trained model"""

        if model_type == "anomaly_detection":
            # For anomaly detection, we'll use a simple accuracy measure
            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == -1) if len(predictions) > 0 else 0.0

            return MLModelMetrics(
                model_id=f"anomaly_{int(datetime.utcnow().timestamp())}",
                accuracy=accuracy,
                precision=accuracy,
                recall=accuracy,
                f1_score=accuracy,
                last_trained=datetime.utcnow(),
                training_samples=len(X_val),
                validation_score=accuracy
            )

        elif model_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            predictions = model.predict(X_val)

            return MLModelMetrics(
                model_id=f"classifier_{int(datetime.utcnow().timestamp())}",
                accuracy=accuracy_score(y_val, predictions),
                precision=precision_score(y_val, predictions, average='weighted', zero_division=0),
                recall=recall_score(y_val, predictions, average='weighted', zero_division=0),
                f1_score=f1_score(y_val, predictions, average='weighted', zero_division=0),
                last_trained=datetime.utcnow(),
                training_samples=len(X_val),
                validation_score=accuracy_score(y_val, predictions)
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def _persist_model(self, model_id: str, model):
        """Persist trained model to storage"""

        try:
            model_dir = Path("models/trained")
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Model persisted: {model_path}")

        except Exception as e:
            logger.error(f"Failed to persist model {model_id}: {e}")

    async def _analyze_vulnerability_patterns(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vulnerability patterns in scan results"""

        vulnerabilities = scan_results.get("vulnerabilities", [])

        if not vulnerabilities:
            return {"pattern": "no_vulnerabilities", "insights": []}

        # Analyze vulnerability categories
        categories = {}
        severities = {}

        for vuln in vulnerabilities:
            category = vuln.get("category", "unknown")
            severity = vuln.get("severity", "unknown")

            categories[category] = categories.get(category, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1

        # Generate insights
        insights = []

        # Most common category
        top_category = max(categories.items(), key=lambda x: x[1]) if categories else ("none", 0)
        if top_category[1] > len(vulnerabilities) * 0.3:
            insights.append(f"High concentration of {top_category[0]} vulnerabilities")

        # Severity distribution
        critical_count = severities.get("critical", 0)
        if critical_count > 0:
            insights.append(f"{critical_count} critical vulnerabilities require immediate attention")

        return {
            "total_vulnerabilities": len(vulnerabilities),
            "categories": categories,
            "severities": severities,
            "top_category": top_category[0],
            "insights": insights,
            "pattern": "vulnerability_cluster" if len(insights) > 1 else "normal_distribution"
        }

    async def _analyze_threat_landscape(self, threat_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat landscape from intelligence data"""

        indicators = threat_intelligence.get("indicators", [])

        return {
            "active_campaigns": threat_intelligence.get("active_campaigns", 0),
            "threat_actors": threat_intelligence.get("threat_actors", []),
            "trending_ttps": threat_intelligence.get("trending_ttps", []),
            "geographic_hotspots": threat_intelligence.get("geographic_hotspots", []),
            "industry_targeting": threat_intelligence.get("industry_targeting", []),
            "threat_level": self._assess_threat_level(indicators),
            "prediction_confidence": 0.75
        }

    def _assess_threat_level(self, indicators: List[str]) -> str:
        """Assess overall threat level"""

        if len(indicators) > 50:
            return "critical"
        elif len(indicators) > 20:
            return "high"
        elif len(indicators) > 5:
            return "medium"
        else:
            return "low"

    async def _analyze_behavioral_trends(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral trends"""

        entities = behavioral_data.get("entities", [])

        high_risk_count = sum(1 for e in entities if e.get("risk_level") in ["high", "critical"])
        anomaly_count = sum(1 for e in entities if e.get("anomaly_score", 0) > 0.5)

        return {
            "total_entities": len(entities),
            "high_risk_entities": high_risk_count,
            "anomalous_entities": anomaly_count,
            "risk_trend": "increasing" if high_risk_count > len(entities) * 0.2 else "stable",
            "behavioral_insights": [
                f"{high_risk_count} entities showing high-risk behavior",
                f"{anomaly_count} entities with behavioral anomalies"
            ]
        }

    async def _cross_correlate_security_data(
        self,
        vulnerability_insights: Dict[str, Any],
        threat_insights: Dict[str, Any],
        behavioral_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cross-correlate security data from multiple sources"""

        correlations = {
            "vulnerability_threat_correlation": 0.0,
            "behavioral_threat_correlation": 0.0,
            "vulnerability_behavioral_correlation": 0.0,
            "overall_correlation_strength": 0.0
        }

        # Calculate vulnerability-threat correlation
        vuln_severity = vulnerability_insights.get("severities", {}).get("critical", 0)
        threat_level = threat_insights.get("threat_level", "low")

        if vuln_severity > 0 and threat_level in ["high", "critical"]:
            correlations["vulnerability_threat_correlation"] = 0.8

        # Calculate behavioral-threat correlation
        high_risk_entities = behavioral_insights.get("high_risk_entities", 0)
        active_campaigns = threat_insights.get("active_campaigns", 0)

        if high_risk_entities > 0 and active_campaigns > 0:
            correlations["behavioral_threat_correlation"] = 0.7

        # Calculate overall correlation
        correlations["overall_correlation_strength"] = np.mean(list(correlations.values()))

        return correlations

    async def _predict_security_trends(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Predict security trends based on correlations"""

        overall_strength = correlations.get("overall_correlation_strength", 0)

        if overall_strength > 0.6:
            trend = "deteriorating"
            prediction = "Increased threat activity expected"
        elif overall_strength > 0.3:
            trend = "stable_with_concerns"
            prediction = "Moderate threat activity expected"
        else:
            trend = "improving"
            prediction = "Normal threat activity expected"

        return {
            "trend_direction": trend,
            "prediction": prediction,
            "confidence": overall_strength,
            "timeframe": "7-14 days",
            "key_factors": [
                "Vulnerability-threat correlation",
                "Behavioral anomaly patterns",
                "Threat intelligence indicators"
            ]
        }

    def _prioritize_security_risks(
        self,
        correlations: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize security risks based on analysis"""

        risks = []

        # High correlation between vulnerabilities and threats
        if correlations.get("vulnerability_threat_correlation", 0) > 0.5:
            risks.append({
                "risk": "Vulnerability exploitation",
                "priority": "high",
                "likelihood": "high",
                "impact": "high"
            })

        # Behavioral anomalies with threat activity
        if correlations.get("behavioral_threat_correlation", 0) > 0.5:
            risks.append({
                "risk": "Insider threat activity",
                "priority": "medium",
                "likelihood": "medium",
                "impact": "high"
            })

        # Trend-based risks
        if predictions.get("trend_direction") == "deteriorating":
            risks.append({
                "risk": "Escalating threat landscape",
                "priority": "high",
                "likelihood": "high",
                "impact": "medium"
            })

        return sorted(risks, key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}.get(x["priority"], 0)
        ), reverse=True)

    async def _generate_actionable_recommendations(
        self,
        correlations: Dict[str, Any],
        predictions: Dict[str, Any],
        risk_priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""

        recommendations = []

        # Address high-priority risks
        for risk in risk_priorities[:3]:  # Top 3 risks
            if "vulnerability" in risk["risk"].lower():
                recommendations.append({
                    "action": "Accelerate vulnerability patching",
                    "priority": "immediate",
                    "effort": "medium",
                    "impact": "high",
                    "timeframe": "1-3 days"
                })

            if "insider" in risk["risk"].lower():
                recommendations.append({
                    "action": "Enhance user behavior monitoring",
                    "priority": "high",
                    "effort": "low",
                    "impact": "medium",
                    "timeframe": "immediate"
                })

            if "threat landscape" in risk["risk"].lower():
                recommendations.append({
                    "action": "Increase security monitoring level",
                    "priority": "high",
                    "effort": "low",
                    "impact": "medium",
                    "timeframe": "immediate"
                })

        # General improvements based on correlations
        if correlations.get("overall_correlation_strength", 0) > 0.4:
            recommendations.append({
                "action": "Implement threat hunting program",
                "priority": "medium",
                "effort": "high",
                "impact": "high",
                "timeframe": "1-2 weeks"
            })

        return recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_insight_confidence(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for insights"""

        overall_correlation = correlations.get("overall_correlation_strength", 0)

        return {
            "overall_confidence": overall_correlation,
            "data_quality": 0.8,  # Simulated data quality score
            "analysis_depth": 0.9,  # Comprehensive analysis indicator
            "prediction_reliability": min(overall_correlation + 0.2, 1.0)
        }


# Feature extractor classes
class NetworkFeatureExtractor:
    """Extract features from network data"""

    def extract(self, data: Dict[str, Any]) -> np.ndarray:
        return np.random.random(10)  # Placeholder


class EndpointFeatureExtractor:
    """Extract features from endpoint data"""

    def extract(self, data: Dict[str, Any]) -> np.ndarray:
        return np.random.random(10)  # Placeholder


class BehavioralFeatureExtractor:
    """Extract features from behavioral data"""

    def extract(self, data: Dict[str, Any]) -> np.ndarray:
        return np.random.random(10)  # Placeholder
