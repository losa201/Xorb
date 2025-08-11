#!/usr/bin/env python3
"""
Advanced ML-Based Anomaly Detection Service
Real-time behavioral analysis and threat detection for XORB Platform

This service implements multiple machine learning algorithms for detecting:
- User behavior anomalies
- Network traffic anomalies
- API usage pattern anomalies
- Authentication anomalies
- System performance anomalies
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
import redis
from pathlib import Path

# ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Deep learning for advanced patterns
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - using scikit-learn only")

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected"""
    USER_BEHAVIOR = "user_behavior"
    NETWORK_TRAFFIC = "network_traffic"
    API_USAGE = "api_usage"
    AUTHENTICATION = "authentication"
    SYSTEM_PERFORMANCE = "system_performance"
    DATA_ACCESS = "data_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    # Model parameters
    contamination_rate: float = 0.1  # Expected percentage of anomalies
    sensitivity: float = 0.8  # Detection sensitivity (0.0-1.0)
    
    # Time windows
    training_window_days: int = 30
    detection_window_minutes: int = 15
    
    # Feature engineering
    max_features: int = 100
    min_samples: int = 50
    
    # Model persistence
    model_cache_ttl: int = 3600  # 1 hour
    retrain_interval_hours: int = 24
    
    # Thresholds
    anomaly_threshold: float = -0.5
    critical_threshold: float = -0.8
    
    # Real-time processing
    batch_size: int = 1000
    max_memory_mb: int = 512


@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    event_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    risk_score: float
    description: str
    features: Dict[str, Any]
    source_data: Dict[str, Any]
    model_version: str
    mitigation_suggestions: List[str]


class FeatureExtractor:
    """Advanced feature extraction for anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
    
    def extract_user_behavior_features(self, user_events: List[Dict]) -> np.ndarray:
        """Extract features from user behavior data"""
        if not user_events:
            return np.array([])
        
        df = pd.DataFrame(user_events)
        features = []
        
        # Temporal patterns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Session patterns
        features.extend([
            df['hour'].nunique(),  # Active hours diversity
            df['day_of_week'].nunique(),  # Active days diversity
            df['is_weekend'].mean(),  # Weekend activity ratio
            len(df),  # Total events
        ])
        
        # API endpoint patterns
        if 'endpoint' in df.columns:
            endpoint_counts = df['endpoint'].value_counts()
            features.extend([
                len(endpoint_counts),  # Unique endpoints
                endpoint_counts.max(),  # Max endpoint usage
                endpoint_counts.std(),  # Endpoint usage variance
            ])
        
        # Geographic patterns
        if 'source_ip' in df.columns:
            ip_counts = df['source_ip'].value_counts()
            features.extend([
                len(ip_counts),  # Unique IPs
                ip_counts.max(),  # Max IP usage
            ])
        
        # Response patterns
        if 'response_time' in df.columns:
            features.extend([
                df['response_time'].mean(),
                df['response_time'].std(),
                df['response_time'].max(),
            ])
        
        # Status code patterns
        if 'status_code' in df.columns:
            error_rate = (df['status_code'] >= 400).mean()
            features.append(error_rate)
        
        # Padding to ensure consistent feature count
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def extract_network_traffic_features(self, network_data: List[Dict]) -> np.ndarray:
        """Extract features from network traffic data"""
        if not network_data:
            return np.array([])
        
        df = pd.DataFrame(network_data)
        features = []
        
        # Traffic volume patterns
        if 'bytes_sent' in df.columns:
            features.extend([
                df['bytes_sent'].sum(),
                df['bytes_sent'].mean(),
                df['bytes_sent'].std(),
                df['bytes_sent'].max(),
            ])
        
        if 'bytes_received' in df.columns:
            features.extend([
                df['bytes_received'].sum(),
                df['bytes_received'].mean(),
                df['bytes_received'].std(),
                df['bytes_received'].max(),
            ])
        
        # Connection patterns
        if 'destination_port' in df.columns:
            port_diversity = df['destination_port'].nunique()
            common_ports = df['destination_port'].isin([80, 443, 22, 25]).mean()
            features.extend([port_diversity, common_ports])
        
        # Protocol patterns
        if 'protocol' in df.columns:
            protocol_counts = df['protocol'].value_counts()
            features.extend([
                len(protocol_counts),
                protocol_counts.max() if len(protocol_counts) > 0 else 0,
            ])
        
        # Temporal patterns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diff = df['timestamp'].diff().dt.total_seconds()
        features.extend([
            time_diff.mean() if len(time_diff) > 1 else 0,
            time_diff.std() if len(time_diff) > 1 else 0,
        ])
        
        # Padding
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])
    
    def extract_api_usage_features(self, api_events: List[Dict]) -> np.ndarray:
        """Extract features from API usage patterns"""
        if not api_events:
            return np.array([])
        
        df = pd.DataFrame(api_events)
        features = []
        
        # Request rate patterns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        
        features.extend([
            len(df),  # Total requests
            time_diffs.mean() if len(time_diffs) > 1 else 0,  # Avg time between requests
            time_diffs.std() if len(time_diffs) > 1 else 0,   # Request timing variance
        ])
        
        # Endpoint diversity
        if 'endpoint' in df.columns:
            endpoint_entropy = -sum(p * np.log(p) for p in df['endpoint'].value_counts(normalize=True))
            features.append(endpoint_entropy)
        
        # HTTP method patterns
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            features.extend([
                len(method_counts),
                method_counts.get('GET', 0) / len(df),
                method_counts.get('POST', 0) / len(df),
                method_counts.get('PUT', 0) / len(df),
                method_counts.get('DELETE', 0) / len(df),
            ])
        
        # Response patterns
        if 'status_code' in df.columns:
            status_counts = df['status_code'].value_counts()
            features.extend([
                status_counts.get(200, 0) / len(df),  # Success rate
                status_counts.get(401, 0) / len(df),  # Auth failure rate
                status_counts.get(403, 0) / len(df),  # Forbidden rate
                status_counts.get(500, 0) / len(df),  # Server error rate
            ])
        
        # Payload size patterns
        if 'payload_size' in df.columns:
            features.extend([
                df['payload_size'].mean(),
                df['payload_size'].std(),
                df['payload_size'].max(),
            ])
        
        # Padding
        while len(features) < 18:
            features.append(0.0)
        
        return np.array(features[:18])
    
    def extract_authentication_features(self, auth_events: List[Dict]) -> np.ndarray:
        """Extract features from authentication patterns"""
        if not auth_events:
            return np.array([])
        
        df = pd.DataFrame(auth_events)
        features = []
        
        # Success/failure patterns
        if 'status' in df.columns:
            success_rate = (df['status'] == 'success').mean()
            failure_rate = (df['status'] == 'failure').mean()
            features.extend([success_rate, failure_rate])
        
        # Temporal patterns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Off-hours activity
        off_hours = df['hour'].isin(list(range(0, 6)) + list(range(22, 24))).mean()
        features.append(off_hours)
        
        # Geographic patterns
        if 'source_ip' in df.columns:
            unique_ips = df['source_ip'].nunique()
            ip_entropy = -sum(p * np.log(p) for p in df['source_ip'].value_counts(normalize=True))
            features.extend([unique_ips, ip_entropy])
        
        # Device patterns
        if 'user_agent' in df.columns:
            unique_agents = df['user_agent'].nunique()
            features.append(unique_agents)
        
        # Session patterns
        if 'session_duration' in df.columns:
            features.extend([
                df['session_duration'].mean(),
                df['session_duration'].std(),
                df['session_duration'].max(),
            ])
        
        # Authentication method patterns
        if 'auth_method' in df.columns:
            method_counts = df['auth_method'].value_counts()
            mfa_rate = method_counts.get('mfa', 0) / len(df)
            features.append(mfa_rate)
        
        # Padding
        while len(features) < 12:
            features.append(0.0)
        
        return np.array(features[:12])


class AnomalyDetectionModel:
    """Advanced ML model for anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig, anomaly_type: AnomalyType):
        self.config = config
        self.anomaly_type = anomaly_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.model_version = "1.0"
        self.last_training = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # Isolation Forest for unsupervised anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.config.contamination_rate,
            random_state=42,
            n_estimators=100
        )
        
        # DBSCAN for density-based clustering
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=self.config.min_samples
        )
        
        # Random Forest for supervised learning (when labels available)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # LSTM for time series anomalies (if TensorFlow available)
        if TF_AVAILABLE:
            self.models['lstm'] = self._create_lstm_model()
        
        # Standard scaler for feature normalization
        self.scalers['standard'] = StandardScaler()
    
    def _create_lstm_model(self):
        """Create LSTM model for time series anomaly detection"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(10, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    async def train(self, training_data: List[Dict], labels: Optional[List[int]] = None):
        """Train anomaly detection models"""
        try:
            logger.info(f"Training {self.anomaly_type.value} anomaly detection model")
            
            if len(training_data) < self.config.min_samples:
                logger.warning(f"Insufficient training data: {len(training_data)} < {self.config.min_samples}")
                return False
            
            # Extract features
            features = self._extract_features(training_data)
            if features.size == 0:
                logger.warning("No features extracted from training data")
                return False
            
            # Scale features
            features_scaled = self.scalers['standard'].fit_transform(features)
            
            # Train Isolation Forest (unsupervised)
            self.models['isolation_forest'].fit(features_scaled)
            
            # Train DBSCAN (unsupervised clustering)
            cluster_labels = self.models['dbscan'].fit_predict(features_scaled)
            
            # Train Random Forest if labels provided
            if labels is not None and len(labels) == len(features_scaled):
                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, labels, test_size=0.2, random_state=42
                )
                self.models['random_forest'].fit(X_train, y_train)
                
                # Evaluate model
                y_pred = self.models['random_forest'].predict(X_test)
                logger.info(f"Random Forest accuracy: {self.models['random_forest'].score(X_test, y_test):.3f}")
            
            # Train LSTM for time series
            if TF_AVAILABLE and self.models['lstm'] is not None:
                self._train_lstm(features_scaled, labels)
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            
            logger.info(f"Model training completed for {self.anomaly_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed for {self.anomaly_type.value}: {str(e)}")
            return False
    
    def _train_lstm(self, features: np.ndarray, labels: Optional[List[int]]):
        """Train LSTM model for time series anomaly detection"""
        try:
            # Prepare time series data
            sequence_length = 10
            X, y = self._prepare_lstm_data(features, labels, sequence_length)
            
            if len(X) < 10:  # Minimum sequences needed
                return
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.models['lstm'].fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            logger.info("LSTM model training completed")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
    
    def _prepare_lstm_data(self, features: np.ndarray, labels: Optional[List[int]], sequence_length: int):
        """Prepare data for LSTM training"""
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            if labels is not None:
                y.append(labels[i + sequence_length])
            else:
                # Use isolation forest predictions as pseudo-labels
                y.append(0)  # Normal by default
        
        return np.array(X), np.array(y)
    
    async def detect_anomalies(self, data: List[Dict]) -> List[AnomalyEvent]:
        """Detect anomalies in new data"""
        if not self.is_trained:
            logger.warning(f"Model not trained for {self.anomaly_type.value}")
            return []
        
        try:
            # Extract features
            features = self._extract_features(data)
            if features.size == 0:
                return []
            
            # Scale features
            features_scaled = self.scalers['standard'].transform(features)
            
            # Get predictions from multiple models
            anomalies = []
            
            # Isolation Forest predictions
            if_scores = self.models['isolation_forest'].decision_function(features_scaled)
            if_predictions = self.models['isolation_forest'].predict(features_scaled)
            
            # Random Forest predictions (if trained)
            rf_predictions = None
            rf_probabilities = None
            if hasattr(self.models['random_forest'], 'classes_'):
                rf_predictions = self.models['random_forest'].predict(features_scaled)
                rf_probabilities = self.models['random_forest'].predict_proba(features_scaled)
            
            # Process each prediction
            for i, (score, prediction) in enumerate(zip(if_scores, if_predictions)):
                if prediction == -1:  # Anomaly detected
                    confidence = self._calculate_confidence(score, rf_probabilities[i] if rf_probabilities is not None else None)
                    severity = self._calculate_severity(score, confidence)
                    risk_score = self._calculate_risk_score(score, confidence, features_scaled[i])
                    
                    anomaly = AnomalyEvent(
                        event_id=f"{self.anomaly_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i}",
                        timestamp=datetime.utcnow(),
                        anomaly_type=self.anomaly_type,
                        severity=severity,
                        confidence=confidence,
                        risk_score=risk_score,
                        description=self._generate_description(features_scaled[i], score),
                        features=self._features_to_dict(features_scaled[i]),
                        source_data=data[i] if i < len(data) else {},
                        model_version=self.model_version,
                        mitigation_suggestions=self._generate_mitigation_suggestions(features_scaled[i])
                    )
                    
                    anomalies.append(anomaly)
            
            logger.info(f"Detected {len(anomalies)} anomalies for {self.anomaly_type.value}")
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {self.anomaly_type.value}: {str(e)}")
            return []
    
    def _extract_features(self, data: List[Dict]) -> np.ndarray:
        """Extract features based on anomaly type"""
        feature_extractor = FeatureExtractor(self.config)
        
        if self.anomaly_type == AnomalyType.USER_BEHAVIOR:
            return feature_extractor.extract_user_behavior_features(data)
        elif self.anomaly_type == AnomalyType.NETWORK_TRAFFIC:
            return feature_extractor.extract_network_traffic_features(data)
        elif self.anomaly_type == AnomalyType.API_USAGE:
            return feature_extractor.extract_api_usage_features(data)
        elif self.anomaly_type == AnomalyType.AUTHENTICATION:
            return feature_extractor.extract_authentication_features(data)
        else:
            # Generic feature extraction
            return np.array([[0.0] * 10] * len(data))
    
    def _calculate_confidence(self, if_score: float, rf_prob: Optional[np.ndarray]) -> float:
        """Calculate confidence score"""
        # Isolation Forest confidence (lower score = higher anomaly)
        if_confidence = max(0.0, min(1.0, (-if_score + 0.5) * 2))
        
        # Random Forest confidence
        rf_confidence = 0.5
        if rf_prob is not None and len(rf_prob) > 1:
            rf_confidence = max(rf_prob)
        
        # Combined confidence
        return (if_confidence + rf_confidence) / 2
    
    def _calculate_severity(self, score: float, confidence: float) -> AnomalySeverity:
        """Calculate anomaly severity"""
        if score <= self.config.critical_threshold or confidence >= 0.9:
            return AnomalySeverity.CRITICAL
        elif score <= self.config.anomaly_threshold or confidence >= 0.7:
            return AnomalySeverity.HIGH
        elif confidence >= 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _calculate_risk_score(self, score: float, confidence: float, features: np.ndarray) -> float:
        """Calculate risk score (0-100)"""
        base_risk = max(0, (-score + 0.5) * 100)
        confidence_factor = confidence
        feature_variance = np.std(features) * 10
        
        risk_score = min(100, base_risk * confidence_factor + feature_variance)
        return round(risk_score, 2)
    
    def _generate_description(self, features: np.ndarray, score: float) -> str:
        """Generate human-readable description"""
        descriptions = {
            AnomalyType.USER_BEHAVIOR: f"Unusual user behavior pattern detected (anomaly score: {score:.3f})",
            AnomalyType.NETWORK_TRAFFIC: f"Abnormal network traffic pattern identified (anomaly score: {score:.3f})",
            AnomalyType.API_USAGE: f"Irregular API usage pattern detected (anomaly score: {score:.3f})",
            AnomalyType.AUTHENTICATION: f"Suspicious authentication pattern identified (anomaly score: {score:.3f})",
            AnomalyType.SYSTEM_PERFORMANCE: f"System performance anomaly detected (anomaly score: {score:.3f})"
        }
        
        return descriptions.get(self.anomaly_type, f"Anomaly detected (score: {score:.3f})")
    
    def _features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Convert feature array to dictionary"""
        feature_names = {
            AnomalyType.USER_BEHAVIOR: [
                "active_hours", "active_days", "weekend_ratio", "total_events",
                "unique_endpoints", "max_endpoint_usage", "endpoint_variance",
                "unique_ips", "max_ip_usage", "avg_response_time", "response_variance",
                "max_response_time", "error_rate"
            ],
            AnomalyType.API_USAGE: [
                "total_requests", "avg_time_between", "timing_variance", "endpoint_entropy",
                "method_count", "get_ratio", "post_ratio", "put_ratio", "delete_ratio",
                "success_rate", "auth_failure_rate", "forbidden_rate", "error_rate",
                "avg_payload_size", "payload_variance", "max_payload_size"
            ]
        }
        
        names = feature_names.get(self.anomaly_type, [f"feature_{i}" for i in range(len(features))])
        return {name: float(value) for name, value in zip(names, features)}
    
    def _generate_mitigation_suggestions(self, features: np.ndarray) -> List[str]:
        """Generate mitigation suggestions based on anomaly type"""
        suggestions = {
            AnomalyType.USER_BEHAVIOR: [
                "Review user access permissions",
                "Check for compromised accounts",
                "Implement additional authentication factors",
                "Monitor user session activity"
            ],
            AnomalyType.NETWORK_TRAFFIC: [
                "Analyze network flow patterns",
                "Check for data exfiltration",
                "Review firewall rules",
                "Implement network segmentation"
            ],
            AnomalyType.API_USAGE: [
                "Implement rate limiting",
                "Review API authentication",
                "Check for automated attacks",
                "Monitor API endpoint usage"
            ],
            AnomalyType.AUTHENTICATION: [
                "Force password reset",
                "Enable MFA for account",
                "Review login locations",
                "Check for credential stuffing"
            ]
        }
        
        return suggestions.get(self.anomaly_type, ["Investigate further", "Review security logs"])


class MLAnomalyDetectionService:
    """Main service for ML-based anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.models = {}
        self.redis_client = None
        self.is_initialized = False
        
        # Initialize models for each anomaly type
        for anomaly_type in AnomalyType:
            self.models[anomaly_type] = AnomalyDetectionModel(config, anomaly_type)
    
    async def initialize(self):
        """Initialize the anomaly detection service"""
        try:
            logger.info("Initializing ML Anomaly Detection Service")
            
            # Initialize Redis connection for caching
            try:
                self.redis_client = redis.Redis(
                    host='localhost', port=6379, db=1,
                    decode_responses=True
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.ping
                )
                logger.info("Redis connection established for anomaly detection")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")
                self.redis_client = None
            
            # Load pre-trained models if available
            await self._load_models()
            
            self.is_initialized = True
            logger.info("ML Anomaly Detection Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Anomaly Detection Service: {str(e)}")
            raise
    
    async def _load_models(self):
        """Load pre-trained models from storage"""
        model_dir = Path("models/anomaly_detection")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for anomaly_type in AnomalyType:
            model_file = model_dir / f"{anomaly_type.value}_model.pkl"
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    self.models[anomaly_type].models = model_data['models']
                    self.models[anomaly_type].scalers = model_data['scalers']
                    self.models[anomaly_type].is_trained = model_data['is_trained']
                    self.models[anomaly_type].model_version = model_data.get('model_version', '1.0')
                    
                    logger.info(f"Loaded pre-trained model for {anomaly_type.value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load model for {anomaly_type.value}: {str(e)}")
    
    async def save_models(self):
        """Save trained models to storage"""
        model_dir = Path("models/anomaly_detection")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for anomaly_type, model in self.models.items():
            if model.is_trained:
                model_file = model_dir / f"{anomaly_type.value}_model.pkl"
                try:
                    model_data = {
                        'models': model.models,
                        'scalers': model.scalers,
                        'is_trained': model.is_trained,
                        'model_version': model.model_version,
                        'last_training': model.last_training
                    }
                    
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    logger.info(f"Saved model for {anomaly_type.value}")
                    
                except Exception as e:
                    logger.error(f"Failed to save model for {anomaly_type.value}: {str(e)}")
    
    async def train_models(self, training_data: Dict[AnomalyType, List[Dict]]):
        """Train anomaly detection models"""
        logger.info("Starting model training for all anomaly types")
        
        training_tasks = []
        for anomaly_type, data in training_data.items():
            if anomaly_type in self.models and data:
                task = self.models[anomaly_type].train(data)
                training_tasks.append(task)
        
        # Train models concurrently
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Log training results
        successful_trainings = sum(1 for result in results if result is True)
        logger.info(f"Training completed: {successful_trainings}/{len(training_tasks)} models trained successfully")
        
        # Save trained models
        await self.save_models()
    
    async def detect_anomalies_batch(self, data: Dict[AnomalyType, List[Dict]]) -> Dict[AnomalyType, List[AnomalyEvent]]:
        """Detect anomalies in batch data"""
        if not self.is_initialized:
            logger.warning("Service not initialized")
            return {}
        
        detection_tasks = []
        anomaly_types = []
        
        for anomaly_type, type_data in data.items():
            if anomaly_type in self.models and type_data:
                task = self.models[anomaly_type].detect_anomalies(type_data)
                detection_tasks.append(task)
                anomaly_types.append(anomaly_type)
        
        # Run detection concurrently
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Combine results
        anomalies = {}
        for anomaly_type, result in zip(anomaly_types, results):
            if isinstance(result, list):
                anomalies[anomaly_type] = result
            else:
                logger.error(f"Detection failed for {anomaly_type.value}: {str(result)}")
                anomalies[anomaly_type] = []
        
        return anomalies
    
    async def detect_real_time(self, anomaly_type: AnomalyType, data: Dict) -> Optional[AnomalyEvent]:
        """Real-time anomaly detection for single event"""
        if not self.is_initialized or anomaly_type not in self.models:
            return None
        
        try:
            anomalies = await self.models[anomaly_type].detect_anomalies([data])
            return anomalies[0] if anomalies else None
            
        except Exception as e:
            logger.error(f"Real-time detection failed for {anomaly_type.value}: {str(e)}")
            return None
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            "service_initialized": self.is_initialized,
            "redis_connected": self.redis_client is not None,
            "models": {}
        }
        
        for anomaly_type, model in self.models.items():
            status["models"][anomaly_type.value] = {
                "trained": model.is_trained,
                "last_training": model.last_training.isoformat() if model.last_training else None,
                "model_version": model.model_version
            }
        
        return status
    
    async def get_anomaly_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            # Get anomaly counts from cache
            stats = {}
            for anomaly_type in AnomalyType:
                key = f"anomaly_count:{anomaly_type.value}:{hours}h"
                count = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                stats[anomaly_type.value] = int(count) if count else 0
            
            return {
                "time_window_hours": hours,
                "anomaly_counts": stats,
                "total_anomalies": sum(stats.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get anomaly statistics: {str(e)}")
            return {"error": str(e)}


# Global service instance
_anomaly_detection_service: Optional[MLAnomalyDetectionService] = None


async def get_anomaly_detection_service() -> MLAnomalyDetectionService:
    """Get the global anomaly detection service instance"""
    global _anomaly_detection_service
    
    if _anomaly_detection_service is None:
        config = AnomalyDetectionConfig()
        _anomaly_detection_service = MLAnomalyDetectionService(config)
        await _anomaly_detection_service.initialize()
    
    return _anomaly_detection_service


# Export main classes
__all__ = [
    "MLAnomalyDetectionService",
    "AnomalyDetectionConfig", 
    "AnomalyEvent",
    "AnomalyType",
    "AnomalySeverity",
    "get_anomaly_detection_service"
]