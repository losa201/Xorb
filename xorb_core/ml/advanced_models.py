"""
Advanced Machine Learning Models for Security Analysis
Implements sophisticated ML models for threat detection, behavioral analysis, and predictive security
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import joblib

# ML Libraries
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Deep Learning (optional - graceful fallback)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available, using sklearn models only")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    THREAT_CLASSIFICATION = "threat_classification"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    RISK_SCORING = "risk_scoring"
    ATTACK_PREDICTION = "attack_prediction"
    FEATURE_EXTRACTION = "feature_extraction"

class SecurityFeature(Enum):
    NETWORK_TRAFFIC = "network_traffic"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_LOGS = "system_logs"
    FILE_ACCESS = "file_access"
    PROCESS_ACTIVITY = "process_activity"
    AUTHENTICATION = "authentication"

@dataclass
class ModelConfig:
    model_type: ModelType
    features: List[SecurityFeature]
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_data_path: Optional[str] = None
    model_path: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = None

@dataclass
class PredictionResult:
    prediction: Union[int, float, str]
    confidence: float
    features_used: List[str]
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class TrainingResult:
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    feature_importance: Dict[str, float]
    cross_validation_scores: List[float]

class NetworkAnomalyDetector:
    """Advanced network anomaly detection using multiple algorithms"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def extract_network_features(self, network_data: Dict) -> np.ndarray:
        """Extract features from network traffic data"""
        
        features = []
        
        # Basic flow features
        features.extend([
            network_data.get('packet_count', 0),
            network_data.get('byte_count', 0),
            network_data.get('duration', 0),
            network_data.get('packets_per_second', 0),
            network_data.get('bytes_per_second', 0)
        ])
        
        # Protocol distribution
        protocols = network_data.get('protocols', {})
        features.extend([
            protocols.get('tcp', 0),
            protocols.get('udp', 0),
            protocols.get('icmp', 0),
            protocols.get('other', 0)
        ])
        
        # Port analysis
        ports = network_data.get('ports', {})
        features.extend([
            len(ports.get('unique_src_ports', [])),
            len(ports.get('unique_dst_ports', [])),
            ports.get('well_known_ports', 0),
            ports.get('high_ports', 0)
        ])
        
        # Statistical features
        packet_sizes = network_data.get('packet_sizes', [])
        if packet_sizes:
            features.extend([
                np.mean(packet_sizes),
                np.std(packet_sizes),
                np.min(packet_sizes),
                np.max(packet_sizes),
                np.percentile(packet_sizes, 95)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Time-based features
        inter_arrival_times = network_data.get('inter_arrival_times', [])
        if inter_arrival_times:
            features.extend([
                np.mean(inter_arrival_times),
                np.std(inter_arrival_times),
                np.var(inter_arrival_times)
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def train(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train the network anomaly detection model"""
        
        logger.info(f"Training network anomaly detector on {len(training_data)} samples")
        
        # Extract features
        X = np.array([self.extract_network_features(data) for data in training_data])
        
        # Store feature names
        self.feature_names = [
            'packet_count', 'byte_count', 'duration', 'packets_per_second', 'bytes_per_second',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'other_ratio',
            'unique_src_ports', 'unique_dst_ports', 'well_known_ports', 'high_ports',
            'mean_packet_size', 'std_packet_size', 'min_packet_size', 'max_packet_size', 'p95_packet_size',
            'mean_inter_arrival', 'std_inter_arrival', 'var_inter_arrival'
        ]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        
        # Calculate anomaly scores for validation
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        outliers = self.isolation_forest.predict(X_scaled)
        
        self.is_trained = True
        
        return {
            "samples_trained": len(training_data),
            "features_count": X.shape[1],
            "anomaly_threshold": np.percentile(anomaly_scores, 10),
            "outlier_percentage": np.sum(outliers == -1) / len(outliers) * 100
        }
    
    def predict(self, network_data: Dict) -> PredictionResult:
        """Predict if network traffic is anomalous"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and scale features
        features = self.extract_network_features(network_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and confidence
        prediction = self.isolation_forest.predict(features_scaled)[0]
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        
        # Convert to probability (confidence)
        confidence = abs(anomaly_score)
        
        return PredictionResult(
            prediction=int(prediction == -1),  # 1 for anomaly, 0 for normal
            confidence=confidence,
            features_used=self.feature_names,
            model_version="network_anomaly_v1.0",
            timestamp=datetime.now(),
            metadata={
                "anomaly_score": anomaly_score,
                "is_outlier": prediction == -1,
                "feature_values": features.tolist()
            }
        )

class BehavioralAnalysisModel:
    """Advanced behavioral analysis using deep learning and clustering"""
    
    def __init__(self):
        self.user_profiles = {}
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize neural network if PyTorch is available
        if TORCH_AVAILABLE:
            self.neural_network = self._create_neural_network()
        else:
            self.neural_network = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    
    def _create_neural_network(self):
        """Create PyTorch neural network for behavioral analysis"""
        
        class BehaviorNet(nn.Module):
            def __init__(self, input_size=20, hidden_size=64, output_size=2):
                super(BehaviorNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, output_size)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return self.softmax(x)
        
        return BehaviorNet()
    
    def extract_behavioral_features(self, user_activity: Dict) -> np.ndarray:
        """Extract behavioral features from user activity data"""
        
        features = []
        
        # Time-based features
        activity_times = user_activity.get('activity_times', [])
        if activity_times:
            # Convert to hours for analysis
            hours = [datetime.fromisoformat(t).hour for t in activity_times]
            features.extend([
                np.mean(hours),
                np.std(hours),
                len(set(hours)),  # unique hours active
                max(hours) - min(hours)  # activity span
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Activity patterns
        features.extend([
            user_activity.get('login_count', 0),
            user_activity.get('failed_login_count', 0),
            user_activity.get('file_access_count', 0),
            user_activity.get('command_count', 0),
            user_activity.get('network_connection_count', 0)
        ])
        
        # Resource usage patterns
        resource_usage = user_activity.get('resource_usage', {})
        features.extend([
            resource_usage.get('avg_cpu_usage', 0),
            resource_usage.get('max_cpu_usage', 0),
            resource_usage.get('avg_memory_usage', 0),
            resource_usage.get('max_memory_usage', 0),
            resource_usage.get('disk_io_ops', 0)
        ])
        
        # Access patterns
        access_patterns = user_activity.get('access_patterns', {})
        features.extend([
            len(access_patterns.get('unique_files', [])),
            len(access_patterns.get('unique_directories', [])),
            access_patterns.get('sensitive_file_access', 0),
            len(access_patterns.get('unique_ip_addresses', [])),
            access_patterns.get('privilege_escalation_attempts', 0)
        ])
        
        return np.array(features)
    
    def build_user_profile(self, user_id: str, activities: List[Dict]) -> Dict[str, Any]:
        """Build a behavioral profile for a specific user"""
        
        logger.info(f"Building behavioral profile for user {user_id}")
        
        # Extract features for all activities
        features_list = [self.extract_behavioral_features(activity) for activity in activities]
        X = np.array(features_list)
        
        # Calculate baseline statistics
        profile = {
            "user_id": user_id,
            "total_activities": len(activities),
            "feature_means": np.mean(X, axis=0).tolist(),
            "feature_stds": np.std(X, axis=0).tolist(),
            "feature_mins": np.min(X, axis=0).tolist(),
            "feature_maxs": np.max(X, axis=0).tolist(),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Perform clustering to identify behavior patterns
        if len(X) > 5:  # Need minimum samples for clustering
            X_scaled = self.scaler.fit_transform(X)
            clusters = self.clustering_model.fit_predict(X_scaled)
            
            profile["behavior_clusters"] = {
                "unique_clusters": len(set(clusters)),
                "cluster_labels": clusters.tolist(),
                "noise_points": np.sum(clusters == -1)
            }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def detect_anomalous_behavior(self, user_id: str, current_activity: Dict) -> PredictionResult:
        """Detect if current user activity is anomalous compared to their profile"""
        
        if user_id not in self.user_profiles:
            return PredictionResult(
                prediction=0,  # Unknown user, can't determine anomaly
                confidence=0.0,
                features_used=[],
                model_version="behavioral_v1.0",
                timestamp=datetime.now(),
                metadata={"error": "User profile not found"}
            )
        
        profile = self.user_profiles[user_id]
        current_features = self.extract_behavioral_features(current_activity)
        
        # Calculate deviation from user's normal behavior
        means = np.array(profile["feature_means"])
        stds = np.array(profile["feature_stds"])
        
        # Prevent division by zero
        stds = np.where(stds == 0, 1, stds)
        
        # Z-score based anomaly detection
        z_scores = np.abs((current_features - means) / stds)
        anomaly_score = np.mean(z_scores)
        
        # Consider it anomalous if average z-score > 2.5 (roughly 99% confidence)
        is_anomalous = anomaly_score > 2.5
        confidence = min(anomaly_score / 2.5, 1.0)  # Normalize to 0-1
        
        return PredictionResult(
            prediction=int(is_anomalous),
            confidence=confidence,
            features_used=[
                'activity_time_mean', 'activity_time_std', 'unique_hours', 'activity_span',
                'login_count', 'failed_login_count', 'file_access_count', 'command_count', 'network_connections',
                'avg_cpu', 'max_cpu', 'avg_memory', 'max_memory', 'disk_io',
                'unique_files', 'unique_dirs', 'sensitive_access', 'unique_ips', 'privilege_escalation'
            ],
            model_version="behavioral_v1.0",
            timestamp=datetime.now(),
            metadata={
                "anomaly_score": anomaly_score,
                "z_scores": z_scores.tolist(),
                "user_profile_activities": profile["total_activities"]
            }
        )

class ThreatClassificationModel:
    """Multi-class threat classification using ensemble methods"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.threat_classes = []
        
    def extract_threat_features(self, incident_data: Dict) -> np.ndarray:
        """Extract features for threat classification"""
        
        features = []
        
        # Incident metadata
        features.extend([
            incident_data.get('severity_level', 0),
            incident_data.get('duration_minutes', 0),
            incident_data.get('affected_systems_count', 0),
            incident_data.get('data_exfiltration_attempted', 0),
            incident_data.get('privilege_escalation', 0)
        ])
        
        # Attack vector indicators
        attack_vectors = incident_data.get('attack_vectors', {})
        features.extend([
            attack_vectors.get('network_based', 0),
            attack_vectors.get('email_based', 0),
            attack_vectors.get('web_based', 0),
            attack_vectors.get('physical_access', 0),
            attack_vectors.get('social_engineering', 0)
        ])
        
        # Technical indicators
        technical_indicators = incident_data.get('technical_indicators', {})
        features.extend([
            technical_indicators.get('malware_detected', 0),
            technical_indicators.get('suspicious_processes', 0),
            technical_indicators.get('network_anomalies', 0),
            technical_indicators.get('file_modifications', 0),
            technical_indicators.get('registry_changes', 0)
        ])
        
        # MITRE ATT&CK mapping
        mitre_tactics = incident_data.get('mitre_tactics', {})
        features.extend([
            mitre_tactics.get('initial_access', 0),
            mitre_tactics.get('execution', 0),
            mitre_tactics.get('persistence', 0),
            mitre_tactics.get('privilege_escalation', 0),
            mitre_tactics.get('defense_evasion', 0),
            mitre_tactics.get('credential_access', 0),
            mitre_tactics.get('discovery', 0),
            mitre_tactics.get('lateral_movement', 0),
            mitre_tactics.get('collection', 0),
            mitre_tactics.get('exfiltration', 0),
            mitre_tactics.get('impact', 0)
        ])
        
        return np.array(features)
    
    def train(self, training_data: List[Dict], labels: List[str]) -> TrainingResult:
        """Train the threat classification model"""
        
        logger.info(f"Training threat classifier on {len(training_data)} samples")
        
        # Extract features and encode labels
        X = np.array([self.extract_threat_features(data) for data in training_data])
        y = self.label_encoder.fit_transform(labels)
        
        self.threat_classes = self.label_encoder.classes_.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        start_time = datetime.now()
        self.classifier.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Feature importance
        feature_names = [
            'severity_level', 'duration', 'affected_systems', 'data_exfiltration', 'privilege_escalation',
            'network_attack', 'email_attack', 'web_attack', 'physical_attack', 'social_engineering',
            'malware', 'suspicious_processes', 'network_anomalies', 'file_modifications', 'registry_changes',
            'initial_access', 'execution', 'persistence', 'privilege_escalation_mitre', 'defense_evasion',
            'credential_access', 'discovery', 'lateral_movement', 'collection', 'exfiltration', 'impact'
        ]
        
        feature_importance = dict(zip(feature_names, self.classifier.feature_importances_))
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        return TrainingResult(
            model_id="threat_classifier_v1.0",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores.tolist()
        )
    
    def predict(self, incident_data: Dict) -> PredictionResult:
        """Classify the type of security threat"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and scale features
        features = self.extract_threat_features(incident_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Convert back to threat class name
        threat_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return PredictionResult(
            prediction=threat_class,
            confidence=confidence,
            features_used=[
                'severity_level', 'duration', 'affected_systems', 'data_exfiltration', 'privilege_escalation',
                'network_attack', 'email_attack', 'web_attack', 'physical_attack', 'social_engineering',
                'malware', 'suspicious_processes', 'network_anomalies', 'file_modifications', 'registry_changes'
            ] + [f'mitre_{tactic}' for tactic in ['initial_access', 'execution', 'persistence', 'privilege_escalation', 
                                                'defense_evasion', 'credential_access', 'discovery', 'lateral_movement', 
                                                'collection', 'exfiltration', 'impact']],
            model_version="threat_classifier_v1.0",
            timestamp=datetime.now(),
            metadata={
                "all_probabilities": dict(zip(self.threat_classes, probabilities.tolist())),
                "predicted_class_index": int(prediction)
            }
        )

class RiskScoringModel:
    """Advanced risk scoring using multiple factors and ML"""
    
    def __init__(self):
        self.risk_factors = {}
        self.weight_optimizer = RandomForestClassifier(n_estimators=50)
        self.is_calibrated = False
        
    def calculate_base_risk_score(self, asset_data: Dict) -> float:
        """Calculate base risk score for an asset"""
        
        risk_score = 0.0
        
        # Asset criticality (0-1)
        criticality = asset_data.get('criticality_level', 0.5)
        risk_score += criticality * 0.3
        
        # Vulnerability score (0-1)
        vulnerabilities = asset_data.get('vulnerabilities', [])
        if vulnerabilities:
            vuln_scores = [v.get('cvss_score', 0) / 10.0 for v in vulnerabilities]
            risk_score += max(vuln_scores) * 0.25
        
        # Exposure level (0-1)
        exposure = asset_data.get('exposure_level', 0.0)
        risk_score += exposure * 0.2
        
        # Threat activity (0-1)
        threat_activity = asset_data.get('recent_threat_activity', 0.0)
        risk_score += threat_activity * 0.15
        
        # Control effectiveness (0-1, inverted)
        control_effectiveness = asset_data.get('security_control_effectiveness', 1.0)
        risk_score += (1.0 - control_effectiveness) * 0.1
        
        return min(risk_score, 1.0)
    
    def calculate_dynamic_risk_score(self, asset_data: Dict, context_data: Dict) -> Dict[str, Any]:
        """Calculate dynamic risk score with contextual factors"""
        
        base_score = self.calculate_base_risk_score(asset_data)
        
        # Contextual multipliers
        multipliers = {
            'time_of_day': self._get_time_risk_multiplier(context_data.get('current_time')),
            'recent_incidents': self._get_incident_risk_multiplier(context_data.get('recent_incidents', [])),
            'threat_landscape': self._get_threat_landscape_multiplier(context_data.get('threat_intel', {})),
            'business_context': self._get_business_context_multiplier(context_data.get('business_context', {}))
        }
        
        # Apply multipliers
        dynamic_score = base_score
        for factor, multiplier in multipliers.items():
            dynamic_score *= multiplier
        
        # Ensure score stays within bounds
        dynamic_score = min(max(dynamic_score, 0.0), 1.0)
        
        return {
            'base_risk_score': base_score,
            'dynamic_risk_score': dynamic_score,
            'risk_factors': multipliers,
            'risk_level': self._categorize_risk(dynamic_score),
            'recommendations': self._generate_risk_recommendations(asset_data, dynamic_score)
        }
    
    def _get_time_risk_multiplier(self, current_time: Optional[str]) -> float:
        """Calculate time-based risk multiplier"""
        
        if not current_time:
            return 1.0
        
        try:
            dt = datetime.fromisoformat(current_time)
            hour = dt.hour
            
            # Higher risk during off-hours (6 PM - 6 AM)
            if hour < 6 or hour >= 18:
                return 1.2
            else:
                return 1.0
        except:
            return 1.0
    
    def _get_incident_risk_multiplier(self, recent_incidents: List[Dict]) -> float:
        """Calculate incident-based risk multiplier"""
        
        if not recent_incidents:
            return 1.0
        
        # More recent incidents = higher risk
        high_severity_incidents = len([i for i in recent_incidents if i.get('severity', 0) >= 7])
        
        if high_severity_incidents >= 3:
            return 1.5
        elif high_severity_incidents >= 1:
            return 1.3
        else:
            return 1.1
    
    def _get_threat_landscape_multiplier(self, threat_intel: Dict) -> float:
        """Calculate threat landscape risk multiplier"""
        
        if not threat_intel:
            return 1.0
        
        # Active campaigns targeting similar assets
        active_campaigns = threat_intel.get('active_campaigns', 0)
        
        if active_campaigns >= 5:
            return 1.4
        elif active_campaigns >= 2:
            return 1.2
        else:
            return 1.0
    
    def _get_business_context_multiplier(self, business_context: Dict) -> float:
        """Calculate business context risk multiplier"""
        
        if not business_context:
            return 1.0
        
        multiplier = 1.0
        
        # Critical business period
        if business_context.get('critical_business_period', False):
            multiplier *= 1.3
        
        # Regulatory scrutiny
        if business_context.get('under_regulatory_scrutiny', False):
            multiplier *= 1.2
        
        # Public visibility
        if business_context.get('high_public_visibility', False):
            multiplier *= 1.1
        
        return multiplier
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_risk_recommendations(self, asset_data: Dict, risk_score: float) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "Implement immediate additional monitoring",
                "Consider temporary access restrictions",
                "Activate incident response team",
                "Review and update security controls"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Increase monitoring frequency",
                "Review access permissions",
                "Update vulnerability patching priority",
                "Conduct security assessment"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Schedule security review",
                "Update security awareness training",
                "Review configuration settings"
            ])
        
        # Asset-specific recommendations
        vulnerabilities = asset_data.get('vulnerabilities', [])
        if vulnerabilities:
            high_vuln = [v for v in vulnerabilities if v.get('cvss_score', 0) >= 7]
            if high_vuln:
                recommendations.append(f"Prioritize patching {len(high_vuln)} high-severity vulnerabilities")
        
        return recommendations

class MLModelManager:
    """Centralized ML model management system"""
    
    def __init__(self, model_storage_path: str = "models"):
        self.storage_path = Path(model_storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.models = {
            ModelType.ANOMALY_DETECTION: NetworkAnomalyDetector(),
            ModelType.BEHAVIORAL_ANALYSIS: BehavioralAnalysisModel(),
            ModelType.THREAT_CLASSIFICATION: ThreatClassificationModel(),
            ModelType.RISK_SCORING: RiskScoringModel()
        }
        
        self.model_metadata = {}
        
    async def train_model(self, model_type: ModelType, training_data: Any, **kwargs) -> TrainingResult:
        """Train a specific model type"""
        
        logger.info(f"Training model: {model_type.value}")
        
        model = self.models[model_type]
        
        if model_type == ModelType.ANOMALY_DETECTION:
            result = model.train(training_data)
            # Create training result for consistency
            training_result = TrainingResult(
                model_id=f"{model_type.value}_v1.0",
                accuracy=0.0,  # Unsupervised learning
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_time=0.0,
                feature_importance={},
                cross_validation_scores=[]
            )
        elif model_type == ModelType.THREAT_CLASSIFICATION:
            labels = kwargs.get('labels', [])
            training_result = model.train(training_data, labels)
        elif model_type == ModelType.BEHAVIORAL_ANALYSIS:
            # For behavioral analysis, training_data should be dict of {user_id: activities}
            for user_id, activities in training_data.items():
                model.build_user_profile(user_id, activities)
            
            training_result = TrainingResult(
                model_id=f"{model_type.value}_v1.0",
                accuracy=0.0,  # Unsupervised learning
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_time=0.0,
                feature_importance={},
                cross_validation_scores=[]
            )
        else:
            raise ValueError(f"Training not implemented for {model_type}")
        
        # Save model
        await self.save_model(model_type)
        
        return training_result
    
    async def predict(self, model_type: ModelType, input_data: Any, **kwargs) -> PredictionResult:
        """Make prediction using specified model"""
        
        model = self.models[model_type]
        
        if model_type == ModelType.ANOMALY_DETECTION:
            return model.predict(input_data)
        elif model_type == ModelType.THREAT_CLASSIFICATION:
            return model.predict(input_data)
        elif model_type == ModelType.BEHAVIORAL_ANALYSIS:
            user_id = kwargs.get('user_id')
            return model.detect_anomalous_behavior(user_id, input_data)
        elif model_type == ModelType.RISK_SCORING:
            context_data = kwargs.get('context_data', {})
            risk_result = model.calculate_dynamic_risk_score(input_data, context_data)
            
            return PredictionResult(
                prediction=risk_result['dynamic_risk_score'],
                confidence=1.0,  # Deterministic calculation
                features_used=['criticality', 'vulnerabilities', 'exposure', 'threats', 'controls'],
                model_version="risk_scoring_v1.0",
                timestamp=datetime.now(),
                metadata=risk_result
            )
        else:
            raise ValueError(f"Prediction not implemented for {model_type}")
    
    async def save_model(self, model_type: ModelType):
        """Save model to disk"""
        
        model_file = self.storage_path / f"{model_type.value}.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.models[model_type], f)
            
            logger.info(f"Saved model: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_type}: {e}")
    
    async def load_model(self, model_type: ModelType):
        """Load model from disk"""
        
        model_file = self.storage_path / f"{model_type.value}.pkl"
        
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            return
        
        try:
            with open(model_file, 'rb') as f:
                self.models[model_type] = pickle.load(f)
            
            logger.info(f"Loaded model: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
    
    async def get_model_info(self, model_type: ModelType) -> Dict[str, Any]:
        """Get information about a model"""
        
        model = self.models[model_type]
        
        info = {
            "model_type": model_type.value,
            "is_trained": getattr(model, 'is_trained', False),
            "model_class": model.__class__.__name__
        }
        
        # Add model-specific information
        if hasattr(model, 'feature_names'):
            info["feature_count"] = len(model.feature_names)
        
        if hasattr(model, 'threat_classes'):
            info["output_classes"] = model.threat_classes
        
        if hasattr(model, 'user_profiles'):
            info["user_profiles_count"] = len(model.user_profiles)
        
        return info

# Demo and testing functions
async def demo_ml_models():
    """Demonstrate ML model capabilities"""
    
    logger.info("Starting ML models demonstration")
    
    # Initialize model manager
    model_manager = MLModelManager()
    
    # Generate sample data for network anomaly detection
    normal_network_data = []
    for _ in range(100):
        normal_network_data.append({
            'packet_count': np.random.normal(1000, 200),
            'byte_count': np.random.normal(50000, 10000),
            'duration': np.random.normal(60, 15),
            'packets_per_second': np.random.normal(16, 4),
            'bytes_per_second': np.random.normal(800, 200),
            'protocols': {'tcp': 0.7, 'udp': 0.2, 'icmp': 0.05, 'other': 0.05},
            'ports': {
                'unique_src_ports': list(range(5)),
                'unique_dst_ports': list(range(3)),
                'well_known_ports': 2,
                'high_ports': 3
            },
            'packet_sizes': [64, 128, 256, 512, 1024],
            'inter_arrival_times': [0.01, 0.02, 0.015, 0.025, 0.018]
        })
    
    # Train network anomaly detector
    print("Training network anomaly detector...")
    await model_manager.train_model(ModelType.ANOMALY_DETECTION, normal_network_data)
    
    # Test anomaly detection
    anomalous_traffic = {
        'packet_count': 10000,  # Unusual high count
        'byte_count': 500000,   # Unusual high bytes
        'duration': 300,        # Long duration
        'packets_per_second': 100,  # High rate
        'bytes_per_second': 5000,
        'protocols': {'tcp': 0.1, 'udp': 0.1, 'icmp': 0.8, 'other': 0.0},  # Unusual protocol mix
        'ports': {
            'unique_src_ports': list(range(50)),  # Many ports
            'unique_dst_ports': list(range(20)),
            'well_known_ports': 0,
            'high_ports': 70
        },
        'packet_sizes': [1500] * 20,  # All max size
        'inter_arrival_times': [0.001] * 20  # Very fast
    }
    
    anomaly_result = await model_manager.predict(ModelType.ANOMALY_DETECTION, anomalous_traffic)
    print(f"Anomaly detection result: {anomaly_result.prediction} (confidence: {anomaly_result.confidence:.3f})")
    
    # Generate sample threat classification data
    threat_training_data = []
    threat_labels = []
    
    threat_types = ['malware', 'phishing', 'ddos', 'insider_threat', 'apt']
    
    for _ in range(200):
        threat_type = np.random.choice(threat_types)
        threat_labels.append(threat_type)
        
        # Generate features based on threat type
        if threat_type == 'malware':
            data = {
                'severity_level': np.random.uniform(6, 10),
                'duration_minutes': np.random.uniform(30, 300),
                'affected_systems_count': np.random.randint(1, 10),
                'data_exfiltration_attempted': np.random.choice([0, 1]),
                'privilege_escalation': 1,
                'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 1, 'physical_access': 0, 'social_engineering': 0},
                'technical_indicators': {'malware_detected': 1, 'suspicious_processes': 1, 'network_anomalies': 0, 'file_modifications': 1, 'registry_changes': 1},
                'mitre_tactics': {'initial_access': 1, 'execution': 1, 'persistence': 1, 'privilege_escalation': 1, 'defense_evasion': 1, 'credential_access': 0, 'discovery': 0, 'lateral_movement': 0, 'collection': 0, 'exfiltration': 0, 'impact': 1}
            }
        elif threat_type == 'phishing':
            data = {
                'severity_level': np.random.uniform(4, 8),
                'duration_minutes': np.random.uniform(5, 60),
                'affected_systems_count': np.random.randint(1, 5),
                'data_exfiltration_attempted': np.random.choice([0, 1]),
                'privilege_escalation': 0,
                'attack_vectors': {'network_based': 0, 'email_based': 1, 'web_based': 1, 'physical_access': 0, 'social_engineering': 1},
                'technical_indicators': {'malware_detected': 0, 'suspicious_processes': 0, 'network_anomalies': 0, 'file_modifications': 0, 'registry_changes': 0},
                'mitre_tactics': {'initial_access': 1, 'execution': 0, 'persistence': 0, 'privilege_escalation': 0, 'defense_evasion': 0, 'credential_access': 1, 'discovery': 0, 'lateral_movement': 0, 'collection': 1, 'exfiltration': 0, 'impact': 0}
            }
        else:
            # Default pattern
            data = {
                'severity_level': np.random.uniform(1, 10),
                'duration_minutes': np.random.uniform(1, 500),
                'affected_systems_count': np.random.randint(1, 20),
                'data_exfiltration_attempted': np.random.choice([0, 1]),
                'privilege_escalation': np.random.choice([0, 1]),
                'attack_vectors': {k: np.random.choice([0, 1]) for k in ['network_based', 'email_based', 'web_based', 'physical_access', 'social_engineering']},
                'technical_indicators': {k: np.random.choice([0, 1]) for k in ['malware_detected', 'suspicious_processes', 'network_anomalies', 'file_modifications', 'registry_changes']},
                'mitre_tactics': {k: np.random.choice([0, 1]) for k in ['initial_access', 'execution', 'persistence', 'privilege_escalation', 'defense_evasion', 'credential_access', 'discovery', 'lateral_movement', 'collection', 'exfiltration', 'impact']}
            }
        
        threat_training_data.append(data)
    
    # Train threat classifier
    print("Training threat classifier...")
    training_result = await model_manager.train_model(ModelType.THREAT_CLASSIFICATION, threat_training_data, labels=threat_labels)
    print(f"Threat classifier accuracy: {training_result.accuracy:.3f}")
    
    # Test threat classification
    test_incident = {
        'severity_level': 8,
        'duration_minutes': 120,
        'affected_systems_count': 5,
        'data_exfiltration_attempted': 1,
        'privilege_escalation': 1,
        'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 1, 'physical_access': 0, 'social_engineering': 0},
        'technical_indicators': {'malware_detected': 1, 'suspicious_processes': 1, 'network_anomalies': 1, 'file_modifications': 1, 'registry_changes': 1},
        'mitre_tactics': {'initial_access': 1, 'execution': 1, 'persistence': 1, 'privilege_escalation': 1, 'defense_evasion': 1, 'credential_access': 0, 'discovery': 1, 'lateral_movement': 1, 'collection': 1, 'exfiltration': 1, 'impact': 1}
    }
    
    classification_result = await model_manager.predict(ModelType.THREAT_CLASSIFICATION, test_incident)
    print(f"Threat classification: {classification_result.prediction} (confidence: {classification_result.confidence:.3f})")
    
    # Test risk scoring
    print("Testing risk scoring...")
    asset_data = {
        'criticality_level': 0.8,
        'vulnerabilities': [
            {'cvss_score': 8.5, 'name': 'CVE-2023-1234'},
            {'cvss_score': 6.2, 'name': 'CVE-2023-5678'}
        ],
        'exposure_level': 0.6,
        'recent_threat_activity': 0.3,
        'security_control_effectiveness': 0.7
    }
    
    context_data = {
        'current_time': datetime.now().isoformat(),
        'recent_incidents': [
            {'severity': 8, 'type': 'malware'},
            {'severity': 6, 'type': 'phishing'}
        ],
        'threat_intel': {'active_campaigns': 3},
        'business_context': {'critical_business_period': True}
    }
    
    risk_result = await model_manager.predict(ModelType.RISK_SCORING, asset_data, context_data=context_data)
    print(f"Risk score: {risk_result.prediction:.3f} ({risk_result.metadata['risk_level']})")
    print(f"Recommendations: {risk_result.metadata['recommendations'][:2]}")
    
    print("✅ ML models demonstration completed")

if __name__ == "__main__":
    asyncio.run(demo_ml_models())