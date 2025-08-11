"""
AI-Powered Threat Intelligence Engine
Advanced threat detection and correlation using machine learning models
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModel
import aiohttp
import hashlib
import uuid

logger = logging.getLogger(__name__)

class ThreatSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ThreatCategory(Enum):
    MALWARE = "malware"
    PHISHING = "phishing"
    VULNERABILITY = "vulnerability"
    ANOMALY = "anomaly"
    IOC = "indicator_of_compromise"
    APT = "advanced_persistent_threat"

@dataclass
class ThreatIndicator:
    id: str
    indicator_type: str
    value: str
    confidence: float
    severity: ThreatSeverity
    category: ThreatCategory
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str]
    context: Dict[str, Any]
    related_indicators: List[str]

@dataclass
class ThreatEvent:
    id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    payload: str
    indicators: List[str]
    severity: ThreatSeverity
    confidence: float
    metadata: Dict[str, Any]

class AIThreatIntelligenceEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threat_feeds = config.get('threat_feeds', [])
        self.ml_models = {}
        self.threat_indicators = {}
        self.threat_events = []
        self.correlation_rules = []
        
        # Initialize ML components
        self.anomaly_detector = None
        self.clustering_model = None
        self.nlp_pipeline = None
        self.scaler = StandardScaler()
        
        # Initialize external integrations
        self.session = None
        self.running = False
        
    async def initialize(self):
        """Initialize the threat intelligence engine"""
        logger.info("Initializing AI Threat Intelligence Engine...")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Load ML models
        await self.load_ml_models()
        
        # Initialize threat feeds
        await self.initialize_threat_feeds()
        
        # Load correlation rules
        await self.load_correlation_rules()
        
        self.running = True
        logger.info("AI Threat Intelligence Engine initialized successfully")
        
    async def load_ml_models(self):
        """Load and initialize machine learning models"""
        try:
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Clustering model for threat grouping
            self.clustering_model = DBSCAN(
                eps=0.3,
                min_samples=5
            )
            
            # NLP pipeline for text analysis
            self.nlp_pipeline = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium"
            )
            
            # Initialize neural network for advanced threat detection
            await self.initialize_neural_network()
            
            logger.info("Machine learning models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise
            
    async def initialize_neural_network(self):
        """Initialize TensorFlow neural network for advanced threat detection"""
        try:
            # Define neural network architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')  # 4 threat categories
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.ml_models['threat_classifier'] = model
            logger.info("Neural network initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize neural network: {e}")
            
    async def initialize_threat_feeds(self):
        """Initialize external threat intelligence feeds"""
        feeds = [
            {
                'name': 'MISP',
                'url': self.config.get('misp_url'),
                'api_key': self.config.get('misp_api_key'),
                'enabled': True
            },
            {
                'name': 'VirusTotal',
                'url': 'https://www.virustotal.com/vtapi/v2/',
                'api_key': self.config.get('virustotal_api_key'),
                'enabled': True
            },
            {
                'name': 'OTX',
                'url': 'https://otx.alienvault.com/api/v1/',
                'api_key': self.config.get('otx_api_key'),
                'enabled': True
            },
            {
                'name': 'ThreatCrowd',
                'url': 'https://www.threatcrowd.org/searchApi/v2/',
                'api_key': None,
                'enabled': True
            }
        ]
        
        for feed in feeds:
            if feed['enabled'] and feed['api_key']:
                self.threat_feeds.append(feed)
                
        logger.info(f"Initialized {len(self.threat_feeds)} threat intelligence feeds")
        
    async def load_correlation_rules(self):
        """Load threat correlation rules"""
        default_rules = [
            {
                'id': 'multiple_failed_logins',
                'name': 'Multiple Failed Login Attempts',
                'conditions': [
                    {'field': 'event_type', 'operator': 'equals', 'value': 'failed_login'},
                    {'field': 'count', 'operator': 'greater_than', 'value': 5},
                    {'field': 'time_window', 'operator': 'within', 'value': 300}
                ],
                'severity': ThreatSeverity.MEDIUM,
                'category': ThreatCategory.ANOMALY
            },
            {
                'id': 'suspicious_network_activity',
                'name': 'Suspicious Network Activity',
                'conditions': [
                    {'field': 'bytes_transferred', 'operator': 'greater_than', 'value': 1000000},
                    {'field': 'destination_port', 'operator': 'in', 'value': [22, 23, 3389]},
                    {'field': 'time_of_day', 'operator': 'between', 'value': [22, 6]}
                ],
                'severity': ThreatSeverity.HIGH,
                'category': ThreatCategory.ANOMALY
            }
        ]
        
        self.correlation_rules = default_rules
        logger.info(f"Loaded {len(self.correlation_rules)} correlation rules")
        
    async def collect_threat_intelligence(self):
        """Collect threat intelligence from various sources"""
        tasks = []
        
        for feed in self.threat_feeds:
            task = asyncio.create_task(self.fetch_from_feed(feed))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_indicators = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch from feed: {result}")
            else:
                total_indicators += len(result)
                
        logger.info(f"Collected {total_indicators} threat indicators")
        return total_indicators
        
    async def fetch_from_feed(self, feed: Dict[str, Any]) -> List[ThreatIndicator]:
        """Fetch threat indicators from a specific feed"""
        try:
            headers = {}
            if feed['api_key']:
                headers['Authorization'] = f"Bearer {feed['api_key']}"
                
            async with self.session.get(feed['url'], headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self.parse_feed_data(feed['name'], data)
                else:
                    logger.warning(f"Failed to fetch from {feed['name']}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching from {feed['name']}: {e}")
            return []
            
    async def parse_feed_data(self, feed_name: str, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse threat intelligence data from feeds"""
        indicators = []
        
        try:
            if feed_name == 'MISP':
                indicators = await self.parse_misp_data(data)
            elif feed_name == 'VirusTotal':
                indicators = await self.parse_virustotal_data(data)
            elif feed_name == 'OTX':
                indicators = await self.parse_otx_data(data)
            elif feed_name == 'ThreatCrowd':
                indicators = await self.parse_threatcrowd_data(data)
                
        except Exception as e:
            logger.error(f"Error parsing data from {feed_name}: {e}")
            
        return indicators
        
    async def parse_misp_data(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse MISP threat intelligence data"""
        indicators = []
        
        for event in data.get('response', []):
            for attribute in event.get('Attribute', []):
                indicator = ThreatIndicator(
                    id=str(uuid.uuid4()),
                    indicator_type=attribute.get('type', 'unknown'),
                    value=attribute.get('value', ''),
                    confidence=float(attribute.get('to_ids', 0)),
                    severity=self.map_severity(attribute.get('category', 'low')),
                    category=self.map_category(attribute.get('type', 'unknown')),
                    source='MISP',
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    tags=attribute.get('Tag', []),
                    context={'event_id': event.get('id')},
                    related_indicators=[]
                )
                indicators.append(indicator)
                
        return indicators
        
    async def analyze_threat_event(self, event: ThreatEvent) -> Dict[str, Any]:
        """Analyze a threat event using AI models"""
        analysis_results = {
            'event_id': event.id,
            'anomaly_score': 0.0,
            'threat_classification': None,
            'related_indicators': [],
            'recommendations': [],
            'confidence': 0.0
        }
        
        try:
            # Extract features for ML analysis
            features = await self.extract_event_features(event)
            
            # Anomaly detection
            anomaly_score = await self.detect_anomaly(features)
            analysis_results['anomaly_score'] = anomaly_score
            
            # Threat classification using neural network
            threat_class = await self.classify_threat(features)
            analysis_results['threat_classification'] = threat_class
            
            # Find related indicators
            related = await self.find_related_indicators(event)
            analysis_results['related_indicators'] = related
            
            # Generate recommendations
            recommendations = await self.generate_recommendations(event, analysis_results)
            analysis_results['recommendations'] = recommendations
            
            # Calculate overall confidence
            confidence = await self.calculate_confidence(analysis_results)
            analysis_results['confidence'] = confidence
            
        except Exception as e:
            logger.error(f"Error analyzing threat event {event.id}: {e}")
            
        return analysis_results
        
    async def extract_event_features(self, event: ThreatEvent) -> np.ndarray:
        """Extract numerical features from threat event for ML analysis"""
        features = []
        
        # Network features
        features.extend([
            len(event.source_ip.split('.')),
            len(event.destination_ip.split('.')),
            event.port,
            len(event.payload),
            hash(event.protocol) % 1000
        ])
        
        # Temporal features
        hour_of_day = event.timestamp.hour
        day_of_week = event.timestamp.weekday()
        features.extend([hour_of_day, day_of_week])
        
        # Payload analysis features
        payload_entropy = await self.calculate_entropy(event.payload)
        payload_suspicious_strings = await self.count_suspicious_strings(event.payload)
        features.extend([payload_entropy, payload_suspicious_strings])
        
        # Pad to 100 features for neural network
        while len(features) < 100:
            features.append(0.0)
            
        return np.array(features[:100])
        
    async def detect_anomaly(self, features: np.ndarray) -> float:
        """Detect anomalies using isolation forest"""
        try:
            if self.anomaly_detector is None:
                return 0.5
                
            # Reshape for single sample prediction
            features_reshaped = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_reshaped)
            
            # Predict anomaly score
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Normalize to 0-1 range
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return 0.5
            
    async def classify_threat(self, features: np.ndarray) -> str:
        """Classify threat using neural network"""
        try:
            if 'threat_classifier' not in self.ml_models:
                return 'unknown'
                
            model = self.ml_models['threat_classifier']
            
            # Reshape and predict
            features_reshaped = features.reshape(1, -1)
            prediction = model.predict(features_reshaped, verbose=0)
            
            # Map prediction to threat category
            categories = ['malware', 'phishing', 'vulnerability', 'anomaly']
            predicted_class = categories[np.argmax(prediction[0])]
            
            return predicted_class
            
        except Exception as e:
            logger.error(f"Error in threat classification: {e}")
            return 'unknown'
            
    async def find_related_indicators(self, event: ThreatEvent) -> List[str]:
        """Find related threat indicators for an event"""
        related = []
        
        for indicator_id, indicator in self.threat_indicators.items():
            if await self.is_indicator_related(event, indicator):
                related.append(indicator_id)
                
        return related
        
    async def is_indicator_related(self, event: ThreatEvent, indicator: ThreatIndicator) -> bool:
        """Check if an indicator is related to an event"""
        # Check IP addresses
        if indicator.indicator_type == 'ip-dst' and indicator.value == event.destination_ip:
            return True
            
        if indicator.indicator_type == 'ip-src' and indicator.value == event.source_ip:
            return True
            
        # Check domains in payload
        if indicator.indicator_type == 'domain' and indicator.value in event.payload:
            return True
            
        # Check file hashes
        if indicator.indicator_type in ['md5', 'sha1', 'sha256']:
            payload_hash = hashlib.md5(event.payload.encode()).hexdigest()
            if indicator.value == payload_hash:
                return True
                
        return False
        
    async def generate_recommendations(self, event: ThreatEvent, analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on threat analysis"""
        recommendations = []
        
        # High anomaly score recommendations
        if analysis['anomaly_score'] > 0.8:
            recommendations.append("Implement additional monitoring for this traffic pattern")
            recommendations.append("Consider blocking source IP temporarily")
            
        # Threat classification specific recommendations
        threat_class = analysis['threat_classification']
        if threat_class == 'malware':
            recommendations.extend([
                "Perform full system antivirus scan",
                "Check for persistence mechanisms",
                "Review recent file downloads and installations"
            ])
        elif threat_class == 'phishing':
            recommendations.extend([
                "Block sender domain/IP",
                "Notify users about phishing attempt",
                "Review email security policies"
            ])
        elif threat_class == 'vulnerability':
            recommendations.extend([
                "Apply security patches immediately",
                "Scan for similar vulnerabilities",
                "Review system configuration"
            ])
            
        # Network-based recommendations
        if event.port in [22, 23, 3389]:  # SSH, Telnet, RDP
            recommendations.append("Review remote access policies and authentication")
            
        return recommendations
        
    async def calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for threat analysis"""
        factors = []
        
        # Anomaly score factor
        anomaly_factor = analysis['anomaly_score']
        factors.append(anomaly_factor)
        
        # Related indicators factor
        indicators_count = len(analysis['related_indicators'])
        indicators_factor = min(1.0, indicators_count / 5.0)
        factors.append(indicators_factor)
        
        # Classification confidence (mock implementation)
        classification_factor = 0.8 if analysis['threat_classification'] != 'unknown' else 0.3
        factors.append(classification_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.3]
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, max(0.1, confidence))
        
    async def correlate_events(self, events: List[ThreatEvent]) -> List[Dict[str, Any]]:
        """Correlate multiple threat events to identify patterns"""
        correlations = []
        
        # Time-based correlation
        time_correlated = await self.correlate_by_time(events)
        correlations.extend(time_correlated)
        
        # IP-based correlation
        ip_correlated = await self.correlate_by_ip(events)
        correlations.extend(ip_correlated)
        
        # Pattern-based correlation
        pattern_correlated = await self.correlate_by_pattern(events)
        correlations.extend(pattern_correlated)
        
        return correlations
        
    async def correlate_by_time(self, events: List[ThreatEvent]) -> List[Dict[str, Any]]:
        """Correlate events by time proximity"""
        correlations = []
        time_window = timedelta(minutes=5)
        
        for i, event1 in enumerate(events):
            related_events = []
            for j, event2 in enumerate(events[i+1:], i+1):
                if abs((event1.timestamp - event2.timestamp).total_seconds()) <= time_window.total_seconds():
                    related_events.append(event2.id)
                    
            if len(related_events) >= 2:
                correlations.append({
                    'type': 'temporal',
                    'primary_event': event1.id,
                    'related_events': related_events,
                    'confidence': 0.7,
                    'description': f"Events occurring within {time_window.total_seconds()} seconds"
                })
                
        return correlations
        
    async def correlate_by_ip(self, events: List[ThreatEvent]) -> List[Dict[str, Any]]:
        """Correlate events by IP address patterns"""
        correlations = []
        ip_groups = {}
        
        # Group events by source IP
        for event in events:
            if event.source_ip not in ip_groups:
                ip_groups[event.source_ip] = []
            ip_groups[event.source_ip].append(event.id)
            
        # Find suspicious IP patterns
        for ip, event_ids in ip_groups.items():
            if len(event_ids) >= 3:
                correlations.append({
                    'type': 'ip_based',
                    'primary_event': event_ids[0],
                    'related_events': event_ids[1:],
                    'confidence': 0.8,
                    'description': f"Multiple events from same source IP: {ip}"
                })
                
        return correlations
        
    async def correlate_by_pattern(self, events: List[ThreatEvent]) -> List[Dict[str, Any]]:
        """Correlate events by behavioral patterns"""
        correlations = []
        
        # Use clustering to find patterns
        if len(events) < 3:
            return correlations
            
        try:
            # Extract features for clustering
            features_matrix = []
            for event in events:
                features = await self.extract_event_features(event)
                features_matrix.append(features)
                
            features_array = np.array(features_matrix)
            
            # Perform clustering
            clusters = self.clustering_model.fit_predict(features_array)
            
            # Group events by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id == -1:  # Noise points
                    continue
                    
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(events[i].id)
                
            # Create correlations for significant clusters
            for cluster_id, event_ids in cluster_groups.items():
                if len(event_ids) >= 2:
                    correlations.append({
                        'type': 'behavioral_pattern',
                        'primary_event': event_ids[0],
                        'related_events': event_ids[1:],
                        'confidence': 0.6,
                        'description': f"Events showing similar behavioral patterns (cluster {cluster_id})"
                    })
                    
        except Exception as e:
            logger.error(f"Error in pattern correlation: {e}")
            
        return correlations
        
    async def calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
            
        counts = {}
        for char in data:
            counts[char] = counts.get(char, 0) + 1
            
        entropy = 0.0
        length = len(data)
        
        for count in counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
                
        return entropy
        
    async def count_suspicious_strings(self, payload: str) -> int:
        """Count suspicious strings in payload"""
        suspicious_patterns = [
            'eval', 'exec', 'system', 'shell_exec', 'passthru',
            'base64_decode', 'gzinflate', 'str_rot13',
            'javascript:', 'vbscript:', 'data:',
            'cmd.exe', 'powershell', '/bin/sh',
            'SELECT', 'UNION', 'DROP', 'DELETE'
        ]
        
        count = 0
        payload_lower = payload.lower()
        
        for pattern in suspicious_patterns:
            count += payload_lower.count(pattern.lower())
            
        return count
        
    def map_severity(self, category: str) -> ThreatSeverity:
        """Map category to threat severity"""
        severity_mapping = {
            'critical': ThreatSeverity.CRITICAL,
            'high': ThreatSeverity.HIGH,
            'medium': ThreatSeverity.MEDIUM,
            'low': ThreatSeverity.LOW
        }
        return severity_mapping.get(category.lower(), ThreatSeverity.LOW)
        
    def map_category(self, indicator_type: str) -> ThreatCategory:
        """Map indicator type to threat category"""
        category_mapping = {
            'malware-sample': ThreatCategory.MALWARE,
            'filename': ThreatCategory.MALWARE,
            'md5': ThreatCategory.MALWARE,
            'sha1': ThreatCategory.MALWARE,
            'sha256': ThreatCategory.MALWARE,
            'domain': ThreatCategory.IOC,
            'hostname': ThreatCategory.IOC,
            'ip-dst': ThreatCategory.IOC,
            'ip-src': ThreatCategory.IOC,
            'url': ThreatCategory.PHISHING,
            'vulnerability': ThreatCategory.VULNERABILITY
        }
        return category_mapping.get(indicator_type.lower(), ThreatCategory.IOC)
        
    async def export_intelligence(self, format_type: str = 'json') -> str:
        """Export threat intelligence data"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'indicators_count': len(self.threat_indicators),
            'events_count': len(self.threat_events),
            'indicators': [asdict(indicator) for indicator in self.threat_indicators.values()],
            'events': [asdict(event) for event in self.threat_events]
        }
        
        if format_type == 'json':
            return json.dumps(export_data, indent=2, default=str)
        elif format_type == 'stix':
            return await self.export_stix_format(export_data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
            
    async def export_stix_format(self, data: Dict[str, Any]) -> str:
        """Export data in STIX format"""
        # Simplified STIX export
        stix_bundle = {
            "type": "bundle",
            "id": f"bundle--{uuid.uuid4()}",
            "objects": []
        }
        
        # Convert indicators to STIX format
        for indicator_data in data['indicators']:
            stix_indicator = {
                "type": "indicator",
                "id": f"indicator--{uuid.uuid4()}",
                "created": indicator_data['first_seen'],
                "modified": indicator_data['last_seen'],
                "pattern": f"[{indicator_data['indicator_type']}:value = '{indicator_data['value']}']",
                "labels": [indicator_data['category']],
                "confidence": int(indicator_data['confidence'] * 100)
            }
            stix_bundle["objects"].append(stix_indicator)
            
        return json.dumps(stix_bundle, indent=2)
        
    async def shutdown(self):
        """Shutdown the threat intelligence engine"""
        self.running = False
        
        if self.session:
            await self.session.close()
            
        logger.info("AI Threat Intelligence Engine shutdown complete")
        
    async def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threat landscape"""
        summary = {
            'total_indicators': len(self.threat_indicators),
            'total_events': len(self.threat_events),
            'severity_breakdown': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'category_breakdown': {
                'malware': 0,
                'phishing': 0,
                'vulnerability': 0,
                'anomaly': 0,
                'ioc': 0,
                'apt': 0
            },
            'recent_activity': len([
                e for e in self.threat_events 
                if e.timestamp > datetime.now() - timedelta(hours=24)
            ]),
            'active_feeds': len(self.threat_feeds),
            'last_updated': datetime.now().isoformat()
        }
        
        # Count by severity
        for indicator in self.threat_indicators.values():
            summary['severity_breakdown'][indicator.severity.value] += 1
            summary['category_breakdown'][indicator.category.value] += 1
            
        return summary

# Usage example and configuration
def create_threat_intelligence_engine():
    """Create and configure the threat intelligence engine"""
    config = {
        'misp_url': '',  # Configure via environment variables
        'misp_api_key': 'your-misp-api-key',
        'virustotal_api_key': 'your-virustotal-api-key',
        'otx_api_key': 'your-otx-api-key',
        'update_interval': 3600,  # 1 hour
        'max_indicators': 100000,
        'max_events': 50000
    }
    
    return AIThreatIntelligenceEngine(config)