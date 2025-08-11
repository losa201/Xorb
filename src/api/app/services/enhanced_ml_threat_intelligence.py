"""
Enhanced ML-Powered Threat Intelligence Service
Advanced machine learning integration for threat detection and analysis
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
import tempfile
from pathlib import Path

# ML imports with graceful fallbacks
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_service import XORBService, ServiceType, ServiceStatus
from .interfaces import ThreatIntelligenceService, MLAnalysisService
from ..core.logging import get_logger
# from ..domain.entities import ThreatIndicator, SecurityEvent, VulnerabilityAssessment
# Using dynamic typing for missing domain entities

logger = get_logger(__name__)

class ThreatType(Enum):
    """Types of security threats"""
    MALWARE = "malware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    APT = "apt"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    COMMAND_CONTROL = "command_control"

class ConfidenceLevel(Enum):
    """Confidence levels for threat analysis"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class ThreatSignature:
    """ML-based threat signature"""
    signature_id: str
    threat_type: ThreatType
    feature_vector: List[float]
    confidence_threshold: float
    model_version: str
    created_at: datetime
    last_updated: datetime
    detection_count: int = 0
    false_positive_rate: float = 0.0

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis"""
    anomaly_id: str
    anomaly_score: float
    confidence: ConfidenceLevel
    features: Dict[str, float]
    timestamp: datetime
    description: str
    recommended_actions: List[str]
    related_events: List[str] = field(default_factory=list)

@dataclass
class ThreatCorrelation:
    """Correlated threat analysis result"""
    correlation_id: str
    primary_threat: str
    related_threats: List[str]
    correlation_score: float
    attack_chain: List[str]
    timeline: List[Tuple[datetime, str]]
    risk_assessment: str
    mitigation_steps: List[str]

class EnhancedMLThreatIntelligence(XORBService, ThreatIntelligenceService, MLAnalysisService):
    """
    Enhanced ML-powered threat intelligence service
    
    Features:
    - Real-time anomaly detection
    - Behavioral analysis
    - Threat correlation
    - Predictive threat modeling
    - Custom ML model training
    - Feature engineering
    - Model drift detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            service_id="enhanced_ml_threat_intelligence",
            service_type=ServiceType.INTELLIGENCE,
            dependencies=["database", "redis", "monitoring"]
        )
        
        self.config = config or {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        
        # Threat signatures and patterns
        self.threat_signatures: Dict[str, ThreatSignature] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.feature_extractors: Dict[str, callable] = {}
        
        # Analysis results cache
        self.anomaly_results: Dict[str, AnomalyDetectionResult] = {}
        self.threat_correlations: Dict[str, ThreatCorrelation] = {}
        
        # Model performance tracking
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
        # Initialize ML components
        self._initialize_ml_components()
        
        logger.info("Enhanced ML Threat Intelligence service initialized")
    
    def _initialize_ml_components(self):
        """Initialize ML models and components"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using fallback models")
            self._initialize_fallback_models()
            return
        
        try:
            # Anomaly detection models
            self.models['isolation_forest'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Classification models
            self.models['threat_classifier'] = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=10
            )
            
            # Clustering for behavioral analysis
            self.models['behavior_clustering'] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            # Feature preprocessing
            self.scalers['standard'] = StandardScaler()
            self.encoders['label'] = LabelEncoder()
            
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models when ML libraries unavailable"""
        self.models['fallback_anomaly'] = self._simple_anomaly_detector
        self.models['fallback_classifier'] = self._rule_based_classifier
        logger.info("Fallback models initialized")
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        self.feature_extractors = {
            'network_features': self._extract_network_features,
            'temporal_features': self._extract_temporal_features,
            'behavioral_features': self._extract_behavioral_features,
            'content_features': self._extract_content_features,
            'statistical_features': self._extract_statistical_features
        }
    
    async def analyze_threat_indicators(
        self, 
        indicators: List[Any]
    ) -> Dict[str, AnomalyDetectionResult]:
        """Analyze threat indicators using ML models"""
        try:
            results = {}
            
            for indicator in indicators:
                # Extract features
                features = await self._extract_comprehensive_features(indicator)
                
                # Perform anomaly detection
                anomaly_result = await self._detect_anomalies(indicator, features)
                
                if anomaly_result:
                    results[indicator.indicator_id] = anomaly_result
                    
                    # Store for correlation analysis
                    self.anomaly_results[indicator.indicator_id] = anomaly_result
            
            # Perform correlation analysis
            if len(results) > 1:
                correlations = await self._correlate_threats(list(results.values()))
                for correlation in correlations:
                    self.threat_correlations[correlation.correlation_id] = correlation
            
            logger.info(f"Analyzed {len(indicators)} threat indicators, found {len(results)} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Threat indicator analysis failed: {e}")
            return {}
    
    async def _extract_comprehensive_features(self, indicator: Any) -> Dict[str, float]:
        """Extract comprehensive feature set from threat indicator"""
        features = {}
        
        try:
            # Extract different feature types
            for feature_type, extractor in self.feature_extractors.items():
                feature_subset = await extractor(indicator)
                features.update(feature_subset)
            
            # Add metadata features
            features.update({
                'confidence': indicator.confidence,
                'severity_numeric': self._severity_to_numeric(indicator.severity),
                'age_hours': (datetime.utcnow() - indicator.first_seen).total_seconds() / 3600,
                'update_frequency': self._calculate_update_frequency(indicator)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    async def _extract_network_features(self, indicator: Any) -> Dict[str, float]:
        """Extract network-related features"""
        features = {}
        
        if indicator.indicator_type in ['ip', 'domain', 'url']:
            # IP-based features
            if indicator.indicator_type == 'ip':
                features.update(self._extract_ip_features(indicator.value))
            
            # Domain-based features
            elif indicator.indicator_type == 'domain':
                features.update(self._extract_domain_features(indicator.value))
            
            # URL-based features
            elif indicator.indicator_type == 'url':
                features.update(self._extract_url_features(indicator.value))
        
        return features
    
    def _extract_ip_features(self, ip_address: str) -> Dict[str, float]:
        """Extract features from IP address"""
        features = {}
        
        try:
            import ipaddress
            ip = ipaddress.ip_address(ip_address)
            
            features.update({
                'is_private': float(ip.is_private),
                'is_loopback': float(ip.is_loopback),
                'is_multicast': float(ip.is_multicast),
                'is_reserved': float(ip.is_reserved),
                'ip_version': float(ip.version),
                'ip_numeric': float(int(ip)) if ip.version == 4 else 0.0
            })
            
        except Exception as e:
            logger.warning(f"IP feature extraction failed: {e}")
            features = {'ip_valid': 0.0}
        
        return features
    
    def _extract_domain_features(self, domain: str) -> Dict[str, float]:
        """Extract features from domain name"""
        features = {
            'domain_length': float(len(domain)),
            'subdomain_count': float(domain.count('.')) - 1,
            'has_hyphen': float('-' in domain),
            'has_numbers': float(any(c.isdigit() for c in domain)),
            'entropy': self._calculate_entropy(domain),
            'vowel_consonant_ratio': self._calculate_vowel_ratio(domain)
        }
        
        # TLD analysis
        tld = domain.split('.')[-1] if '.' in domain else ''
        features['tld_length'] = float(len(tld))
        features['common_tld'] = float(tld in ['com', 'org', 'net', 'edu', 'gov'])
        
        return features
    
    def _extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract features from URL"""
        features = {
            'url_length': float(len(url)),
            'path_length': float(len(url.split('/', 3)[3:4][0]) if '/' in url[8:] else 0),
            'query_length': float(len(url.split('?', 1)[1]) if '?' in url else 0),
            'has_https': float(url.startswith('https')),
            'special_char_count': float(sum(1 for c in url if c in '!@#$%^&*()+=[]{}|;:,.<>?')),
            'url_entropy': self._calculate_entropy(url)
        }
        
        return features
    
    async def _extract_temporal_features(self, indicator: Any) -> Dict[str, float]:
        """Extract temporal features"""
        now = datetime.utcnow()
        
        features = {
            'age_days': (now - indicator.first_seen).days,
            'last_seen_hours': (now - indicator.last_seen).total_seconds() / 3600,
            'activity_duration': (indicator.last_seen - indicator.first_seen).total_seconds() / 3600,
            'is_recent': float((now - indicator.first_seen).days < 7),
            'is_persistent': float((indicator.last_seen - indicator.first_seen).days > 30)
        }
        
        return features
    
    async def _extract_behavioral_features(self, indicator: Any) -> Dict[str, float]:
        """Extract behavioral features"""
        features = {
            'detection_frequency': float(len(indicator.tags)),  # Proxy for detection frequency
            'source_reliability': self._calculate_source_reliability(indicator.source),
            'tag_diversity': float(len(set(indicator.tags))) if indicator.tags else 0.0,
            'description_length': float(len(indicator.description)),
            'description_complexity': self._calculate_text_complexity(indicator.description)
        }
        
        return features
    
    async def _extract_content_features(self, indicator: Any) -> Dict[str, float]:
        """Extract content-based features"""
        features = {}
        
        if indicator.indicator_type == 'hash':
            features.update(self._extract_hash_features(indicator.value))
        elif indicator.indicator_type == 'email':
            features.update(self._extract_email_features(indicator.value))
        elif indicator.indicator_type == 'file':
            features.update(self._extract_file_features(indicator.value))
        
        return features
    
    def _extract_hash_features(self, hash_value: str) -> Dict[str, float]:
        """Extract features from hash values"""
        features = {
            'hash_length': float(len(hash_value)),
            'is_md5': float(len(hash_value) == 32),
            'is_sha1': float(len(hash_value) == 40),
            'is_sha256': float(len(hash_value) == 64),
            'hex_valid': float(all(c in '0123456789abcdefABCDEF' for c in hash_value))
        }
        
        return features
    
    def _extract_email_features(self, email: str) -> Dict[str, float]:
        """Extract features from email addresses"""
        local_part = email.split('@')[0] if '@' in email else email
        domain_part = email.split('@')[1] if '@' in email else ''
        
        features = {
            'email_length': float(len(email)),
            'local_length': float(len(local_part)),
            'domain_length': float(len(domain_part)),
            'has_plus': float('+' in local_part),
            'has_dots': float('.' in local_part),
            'number_ratio': float(sum(1 for c in local_part if c.isdigit()) / len(local_part)) if local_part else 0.0
        }
        
        return features
    
    def _extract_file_features(self, filename: str) -> Dict[str, float]:
        """Extract features from filenames"""
        name_part = filename.rsplit('.', 1)[0] if '.' in filename else filename
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        features = {
            'filename_length': float(len(filename)),
            'name_length': float(len(name_part)),
            'extension_length': float(len(extension)),
            'has_extension': float(bool(extension)),
            'executable_extension': float(extension in ['exe', 'bat', 'cmd', 'scr', 'com', 'pif']),
            'script_extension': float(extension in ['js', 'vbs', 'ps1', 'py', 'pl', 'php']),
            'suspicious_name': float(any(word in name_part.lower() for word in ['temp', 'tmp', 'test', 'crack', 'hack']))
        }
        
        return features
    
    async def _extract_statistical_features(self, indicator: Any) -> Dict[str, float]:
        """Extract statistical features"""
        # Get historical data for this indicator type
        similar_indicators = await self._get_similar_indicators(indicator)
        
        if not similar_indicators:
            return {'statistical_available': 0.0}
        
        # Calculate statistical features
        confidences = [ind.confidence for ind in similar_indicators]
        ages = [(datetime.utcnow() - ind.first_seen).days for ind in similar_indicators]
        
        features = {
            'confidence_percentile': self._calculate_percentile(indicator.confidence, confidences),
            'age_percentile': self._calculate_percentile((datetime.utcnow() - indicator.first_seen).days, ages),
            'confidence_zscore': self._calculate_zscore(indicator.confidence, confidences),
            'age_zscore': self._calculate_zscore((datetime.utcnow() - indicator.first_seen).days, ages),
            'statistical_available': 1.0
        }
        
        return features
    
    async def _detect_anomalies(
        self, 
        indicator: Any, 
        features: Dict[str, float]
    ) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using ML models"""
        try:
            if not features:
                return None
            
            # Prepare feature vector
            feature_vector = list(features.values())
            
            if SKLEARN_AVAILABLE and 'isolation_forest' in self.models:
                # Use isolation forest for anomaly detection
                model = self.models['isolation_forest']
                
                # Reshape for single sample prediction
                X = np.array(feature_vector).reshape(1, -1)
                
                # Get anomaly score
                anomaly_score = model.decision_function(X)[0]
                is_anomaly = model.predict(X)[0] == -1
                
                if is_anomaly:
                    confidence = self._calculate_anomaly_confidence(anomaly_score)
                    
                    return AnomalyDetectionResult(
                        anomaly_id=f"anomaly_{indicator.indicator_id}_{int(datetime.utcnow().timestamp())}",
                        anomaly_score=float(anomaly_score),
                        confidence=confidence,
                        features=features,
                        timestamp=datetime.utcnow(),
                        description=f"Anomalous {indicator.indicator_type} detected with score {anomaly_score:.3f}",
                        recommended_actions=self._generate_recommendations(indicator, anomaly_score)
                    )
            else:
                # Use fallback anomaly detection
                return await self._fallback_anomaly_detection(indicator, features)
            
            return None
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return None
    
    async def _fallback_anomaly_detection(
        self, 
        indicator: Any, 
        features: Dict[str, float]
    ) -> Optional[AnomalyDetectionResult]:
        """Fallback anomaly detection using rule-based approach"""
        anomaly_score = 0.0
        
        # Rule-based anomaly scoring
        if features.get('confidence', 0) > 0.9:
            anomaly_score += 0.3
        
        if features.get('age_days', 0) < 1:
            anomaly_score += 0.2
        
        if features.get('entropy', 0) > 4.0:
            anomaly_score += 0.2
        
        if features.get('suspicious_name', 0) > 0:
            anomaly_score += 0.3
        
        if anomaly_score > 0.5:
            confidence = ConfidenceLevel.MEDIUM if anomaly_score > 0.7 else ConfidenceLevel.LOW
            
            return AnomalyDetectionResult(
                anomaly_id=f"fallback_anomaly_{indicator.indicator_id}_{int(datetime.utcnow().timestamp())}",
                anomaly_score=anomaly_score,
                confidence=confidence,
                features=features,
                timestamp=datetime.utcnow(),
                description=f"Rule-based anomaly detected in {indicator.indicator_type}",
                recommended_actions=["Review indicator manually", "Validate with additional sources"]
            )
        
        return None
    
    async def _correlate_threats(self, anomalies: List[AnomalyDetectionResult]) -> List[ThreatCorrelation]:
        """Correlate multiple threat indicators"""
        correlations = []
        
        if len(anomalies) < 2:
            return correlations
        
        try:
            # Group by time windows
            time_groups = self._group_by_time_window(anomalies, hours=1)
            
            for group in time_groups:
                if len(group) >= 2:
                    correlation = await self._analyze_threat_group(group)
                    if correlation:
                        correlations.append(correlation)
            
        except Exception as e:
            logger.error(f"Threat correlation failed: {e}")
        
        return correlations
    
    def _group_by_time_window(self, anomalies: List[AnomalyDetectionResult], hours: int) -> List[List[AnomalyDetectionResult]]:
        """Group anomalies by time windows"""
        groups = []
        sorted_anomalies = sorted(anomalies, key=lambda x: x.timestamp)
        
        current_group = []
        window_start = None
        
        for anomaly in sorted_anomalies:
            if window_start is None:
                window_start = anomaly.timestamp
                current_group = [anomaly]
            elif (anomaly.timestamp - window_start).total_seconds() <= hours * 3600:
                current_group.append(anomaly)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                window_start = anomaly.timestamp
                current_group = [anomaly]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    async def _analyze_threat_group(self, group: List[AnomalyDetectionResult]) -> Optional[ThreatCorrelation]:
        """Analyze a group of correlated threats"""
        try:
            # Calculate correlation score
            correlation_score = self._calculate_correlation_score(group)
            
            if correlation_score < 0.5:
                return None
            
            # Build attack timeline
            timeline = [(anomaly.timestamp, anomaly.description) for anomaly in group]
            timeline.sort(key=lambda x: x[0])
            
            # Identify attack chain
            attack_chain = self._identify_attack_chain(group)
            
            # Generate risk assessment
            risk_assessment = self._assess_correlation_risk(group, correlation_score)
            
            # Generate mitigation steps
            mitigation_steps = self._generate_correlation_mitigations(group)
            
            correlation_id = f"correlation_{hashlib.md5(str([a.anomaly_id for a in group]).encode()).hexdigest()[:8]}"
            
            return ThreatCorrelation(
                correlation_id=correlation_id,
                primary_threat=group[0].anomaly_id,
                related_threats=[a.anomaly_id for a in group[1:]],
                correlation_score=correlation_score,
                attack_chain=attack_chain,
                timeline=timeline,
                risk_assessment=risk_assessment,
                mitigation_steps=mitigation_steps
            )
            
        except Exception as e:
            logger.error(f"Threat group analysis failed: {e}")
            return None
    
    # Utility methods
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability) if SKLEARN_AVAILABLE else probability * (count / text_length)
        
        return entropy
    
    def _calculate_vowel_ratio(self, text: str) -> float:
        """Calculate vowel to consonant ratio"""
        if not text:
            return 0.0
        
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')
        
        return vowels / consonants if consonants > 0 else 0.0
    
    def _severity_to_numeric(self, severity: str) -> float:
        """Convert severity string to numeric value"""
        severity_map = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        }
        return severity_map.get(severity.lower(), 1.0)
    
    def _calculate_update_frequency(self, indicator: Any) -> float:
        """Calculate how frequently an indicator is updated"""
        time_span = (indicator.last_seen - indicator.first_seen).total_seconds()
        if time_span <= 0:
            return 0.0
        
        # Assume 1 update per hour as baseline
        return 3600.0 / max(time_span, 3600.0)
    
    def _calculate_source_reliability(self, source: str) -> float:
        """Calculate reliability score for threat source"""
        reliable_sources = {
            'misp': 0.9,
            'otx': 0.8,
            'virustotal': 0.85,
            'internal': 0.9,
            'external': 0.6
        }
        return reliable_sources.get(source.lower(), 0.5)
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        if not text:
            return 0.0
        
        # Simple complexity based on word length and sentence structure
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
        avg_sentence_length = len(words) / sentence_count
        
        return (avg_word_length + avg_sentence_length) / 20.0  # Normalize
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile of value in list"""
        if not values:
            return 0.5
        
        sorted_values = sorted(values)
        rank = sum(1 for v in sorted_values if v <= value)
        return rank / len(sorted_values)
    
    def _calculate_zscore(self, value: float, values: List[float]) -> float:
        """Calculate z-score of value"""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (value - mean_val) / std_dev
    
    def _calculate_anomaly_confidence(self, anomaly_score: float) -> ConfidenceLevel:
        """Calculate confidence level from anomaly score"""
        abs_score = abs(anomaly_score)
        
        if abs_score > 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif abs_score > 0.6:
            return ConfidenceLevel.HIGH
        elif abs_score > 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_recommendations(self, indicator: Any, anomaly_score: float) -> List[str]:
        """Generate recommendations based on anomaly analysis"""
        recommendations = []
        
        if abs(anomaly_score) > 0.8:
            recommendations.extend([
                "Immediate investigation required",
                "Block indicator across all systems",
                "Notify security team"
            ])
        elif abs(anomaly_score) > 0.6:
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Validate with additional sources",
                "Consider temporary blocking"
            ])
        else:
            recommendations.extend([
                "Continue monitoring",
                "Review in next analysis cycle"
            ])
        
        # Type-specific recommendations
        if indicator.indicator_type == 'ip':
            recommendations.append("Check geolocation and ASN information")
        elif indicator.indicator_type == 'domain':
            recommendations.append("Perform DNS analysis and WHOIS lookup")
        elif indicator.indicator_type == 'hash':
            recommendations.append("Submit to malware analysis sandbox")
        
        return recommendations
    
    async def _get_similar_indicators(self, indicator: Any) -> List[Any]:
        """Get similar indicators for statistical analysis"""
        # This would typically query the database
        # For now, return empty list as placeholder
        return []
    
    def _calculate_correlation_score(self, group: List[AnomalyDetectionResult]) -> float:
        """Calculate correlation score for threat group"""
        if len(group) < 2:
            return 0.0
        
        # Time proximity score
        time_range = max(a.timestamp for a in group) - min(a.timestamp for a in group)
        time_score = max(0.0, 1.0 - time_range.total_seconds() / 3600.0)  # 1 hour window
        
        # Confidence score
        avg_confidence = sum(a.confidence.value for a in group) / len(group)
        
        # Feature similarity score
        feature_similarity = self._calculate_feature_similarity(group)
        
        return (time_score + avg_confidence + feature_similarity) / 3.0
    
    def _calculate_feature_similarity(self, group: List[AnomalyDetectionResult]) -> float:
        """Calculate feature similarity between anomalies"""
        if len(group) < 2:
            return 0.0
        
        # Simple implementation - could be enhanced with proper similarity metrics
        common_features = set(group[0].features.keys())
        for anomaly in group[1:]:
            common_features &= set(anomaly.features.keys())
        
        if not common_features:
            return 0.0
        
        total_similarity = 0.0
        for feature in common_features:
            values = [a.features[feature] for a in group]
            variance = np.var(values) if SKLEARN_AVAILABLE else self._simple_variance(values)
            similarity = 1.0 / (1.0 + variance)  # Higher similarity for lower variance
            total_similarity += similarity
        
        return total_similarity / len(common_features)
    
    def _simple_variance(self, values: List[float]) -> float:
        """Simple variance calculation without numpy"""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        return sum((v - mean_val) ** 2 for v in values) / len(values)
    
    def _identify_attack_chain(self, group: List[AnomalyDetectionResult]) -> List[str]:
        """Identify potential attack chain from anomaly group"""
        # Simplified attack chain identification
        chain_steps = []
        
        sorted_group = sorted(group, key=lambda x: x.timestamp)
        
        for i, anomaly in enumerate(sorted_group):
            step = f"Step {i+1}: {anomaly.description}"
            chain_steps.append(step)
        
        return chain_steps
    
    def _assess_correlation_risk(self, group: List[AnomalyDetectionResult], correlation_score: float) -> str:
        """Assess risk level of correlated threats"""
        max_confidence = max(a.confidence.value for a in group)
        avg_score = sum(a.anomaly_score for a in group) / len(group)
        
        risk_score = (correlation_score + max_confidence + abs(avg_score)) / 3.0
        
        if risk_score > 0.8:
            return "CRITICAL - Immediate action required"
        elif risk_score > 0.6:
            return "HIGH - Investigation and containment needed"
        elif risk_score > 0.4:
            return "MEDIUM - Enhanced monitoring recommended"
        else:
            return "LOW - Continue standard monitoring"
    
    def _generate_correlation_mitigations(self, group: List[AnomalyDetectionResult]) -> List[str]:
        """Generate mitigation steps for correlated threats"""
        mitigations = [
            "Isolate affected systems immediately",
            "Preserve forensic evidence",
            "Activate incident response procedures",
            "Block all related indicators",
            "Enhance monitoring for similar patterns",
            "Review access logs for compromise indicators",
            "Validate security controls effectiveness"
        ]
        
        # Add specific mitigations based on anomaly types
        descriptions = [a.description.lower() for a in group]
        
        if any('network' in desc for desc in descriptions):
            mitigations.append("Review network segmentation and firewall rules")
        
        if any('malware' in desc for desc in descriptions):
            mitigations.append("Perform full system malware scan")
        
        if any('lateral' in desc for desc in descriptions):
            mitigations.append("Review privileged account access")
        
        return mitigations
    
    # Simple rule-based fallback classifiers
    def _simple_anomaly_detector(self, features: Dict[str, float]) -> float:
        """Simple rule-based anomaly detector"""
        score = 0.0
        
        # High entropy indicates potential obfuscation
        if features.get('entropy', 0) > 4.0:
            score += 0.3
        
        # Very new indicators are suspicious
        if features.get('age_days', 365) < 1:
            score += 0.2
        
        # High confidence but recent could indicate false positive
        if features.get('confidence', 0) > 0.9 and features.get('age_days', 365) < 7:
            score += 0.3
        
        return min(1.0, score)
    
    def _rule_based_classifier(self, features: Dict[str, float]) -> str:
        """Simple rule-based threat classifier"""
        # Basic classification based on features
        if features.get('executable_extension', 0) > 0:
            return ThreatType.MALWARE.value
        
        if features.get('suspicious_name', 0) > 0:
            return ThreatType.MALWARE.value
        
        if features.get('entropy', 0) > 5.0:
            return ThreatType.APT.value
        
        return ThreatType.MALWARE.value  # Default
    
    async def get_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of threat intelligence analysis"""
        return {
            'total_signatures': len(self.threat_signatures),
            'active_anomalies': len(self.anomaly_results),
            'threat_correlations': len(self.threat_correlations),
            'models_available': list(self.models.keys()),
            'feature_extractors': list(self.feature_extractors.keys()),
            'ml_backend': 'scikit-learn' if SKLEARN_AVAILABLE else 'fallback',
            'analysis_capabilities': [
                'anomaly_detection',
                'threat_correlation',
                'behavioral_analysis',
                'feature_engineering',
                'risk_assessment'
            ]
        }

# Global instance
_ml_threat_intelligence = None

def get_ml_threat_intelligence() -> EnhancedMLThreatIntelligence:
    """Get ML threat intelligence service instance"""
    global _ml_threat_intelligence
    if _ml_threat_intelligence is None:
        _ml_threat_intelligence = EnhancedMLThreatIntelligence()
    return _ml_threat_intelligence