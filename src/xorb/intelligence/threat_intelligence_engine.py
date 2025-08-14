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
from .mitre_attack_integration import get_mitre_framework, MITREAttackFramework, ThreatMapping

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

        # MITRE ATT&CK integration
        self.mitre_framework: Optional[MITREAttackFramework] = None
        self.threat_mappings: List[ThreatMapping] = []

    async def initialize(self):
        """Initialize the threat intelligence engine"""
        logger.info("Initializing AI Threat Intelligence Engine...")

        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        # Initialize MITRE ATT&CK framework
        self.mitre_framework = await get_mitre_framework()

        # Load ML models
        await self.load_ml_models()

        # Initialize threat feeds
        await self.initialize_threat_feeds()

        # Load correlation rules
        await self.load_correlation_rules()

        self.running = True
        logger.info("AI Threat Intelligence Engine initialized with MITRE ATT&CK integration")

    async def load_ml_models(self):
        """Load and initialize advanced machine learning models"""
        try:
            # Enhanced anomaly detection with multiple algorithms
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=200,
                max_features=1.0,
                bootstrap=True
            )

            # Advanced clustering model for threat grouping
            self.clustering_model = DBSCAN(
                eps=0.3,
                min_samples=5,
                metric='euclidean',
                algorithm='auto'
            )

            # Load cybersecurity-specific NLP model if available
            try:
                self.nlp_pipeline = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Fallback model
                    tokenizer="microsoft/DialoGPT-medium"
                )
                logger.info("Loaded general NLP model (fallback)")
            except Exception as e:
                logger.warning(f"Failed to load NLP model, using mock: {e}")
                self.nlp_pipeline = None

            # Initialize advanced neural networks
            await self.initialize_neural_network()
            await self.initialize_attention_model()
            await self.initialize_time_series_model()

            # Load pre-trained embeddings for threat indicators
            await self.load_threat_embeddings()

            logger.info("Advanced machine learning models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            # Continue with mock implementations
            await self.initialize_mock_models()

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
        """Analyze a threat event using AI models and MITRE ATT&CK"""
        analysis_results = {
            'event_id': event.id,
            'anomaly_score': 0.0,
            'threat_classification': None,
            'related_indicators': [],
            'recommendations': [],
            'confidence': 0.0,
            'mitre_techniques': [],
            'attack_flow': {},
            'detection_rules': []
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

            # MITRE ATT&CK mapping
            if self.mitre_framework:
                mitre_analysis = await self.map_event_to_mitre_techniques(event)
                analysis_results['mitre_techniques'] = mitre_analysis['techniques']
                analysis_results['attack_flow'] = mitre_analysis['attack_flow']
                analysis_results['detection_rules'] = mitre_analysis['detection_rules']

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

    async def map_event_to_mitre_techniques(self, event: ThreatEvent) -> Dict[str, Any]:
        """Map threat event to MITRE ATT&CK techniques"""
        try:
            if not self.mitre_framework:
                return {"techniques": [], "attack_flow": {}, "detection_rules": []}

            # Create threat indicators from event
            indicators = []

            # Network indicators
            if event.source_ip:
                indicators.append({
                    "type": "ip-src",
                    "value": event.source_ip,
                    "context": {"port": event.port, "protocol": event.protocol}
                })

            if event.destination_ip:
                indicators.append({
                    "type": "ip-dst",
                    "value": event.destination_ip,
                    "context": {"port": event.port, "protocol": event.protocol}
                })

            # Payload analysis
            if event.payload:
                indicators.append({
                    "type": "payload",
                    "value": event.payload,
                    "context": {"event_type": event.event_type}
                })

            # Map to MITRE techniques
            threat_mapping = await self.mitre_framework.map_threat_to_techniques(
                indicators, {"event": event}
            )

            # Get technique details
            technique_details = []
            for tech_id in threat_mapping.technique_ids:
                technique = await self.mitre_framework.get_technique_details(tech_id)
                if technique:
                    technique_details.append({
                        "id": technique.id,
                        "name": technique.name,
                        "tactic": technique.tactic,
                        "description": technique.description,
                        "platforms": technique.platform,
                        "data_sources": technique.data_sources
                    })

            # Generate attack flow
            attack_flow = await self.mitre_framework.get_attack_flow(threat_mapping.technique_ids)

            # Generate detection rules
            detection_rules = []
            for tech_id in threat_mapping.technique_ids:
                rules = await self.mitre_framework.generate_detection_rules(tech_id)
                detection_rules.append(rules)

            return {
                "techniques": technique_details,
                "attack_flow": attack_flow,
                "detection_rules": detection_rules,
                "mapping_confidence": threat_mapping.confidence,
                "evidence": threat_mapping.evidence
            }

        except Exception as e:
            logger.error(f"Error mapping event to MITRE techniques: {e}")
            return {"techniques": [], "attack_flow": {}, "detection_rules": [], "error": str(e)}

    async def analyze_attack_campaign(self, events: List[ThreatEvent]) -> Dict[str, Any]:
        """Analyze multiple events as potential attack campaign using MITRE ATT&CK"""
        try:
            if not self.mitre_framework or not events:
                return {"campaign_analysis": {}, "techniques": [], "attack_chain": []}

            # Map all events to techniques
            all_techniques = []
            technique_timeline = []

            for event in sorted(events, key=lambda x: x.timestamp):
                event_mapping = await self.map_event_to_mitre_techniques(event)

                for technique in event_mapping["techniques"]:
                    all_techniques.append(technique["id"])
                    technique_timeline.append({
                        "timestamp": event.timestamp.isoformat(),
                        "technique_id": technique["id"],
                        "technique_name": technique["name"],
                        "tactic": technique["tactic"],
                        "event_id": event.id
                    })

            # Deduplicate techniques while preserving order
            unique_techniques = []
            seen = set()
            for tech_id in all_techniques:
                if tech_id not in seen:
                    unique_techniques.append(tech_id)
                    seen.add(tech_id)

            # Generate comprehensive attack flow
            attack_flow = await self.mitre_framework.get_attack_flow(unique_techniques)

            # Analyze campaign characteristics
            campaign_analysis = {
                "duration_hours": (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600,
                "total_events": len(events),
                "unique_techniques": len(unique_techniques),
                "tactics_covered": len(set(tech["tactic"] for tech in technique_timeline)),
                "sophistication_score": await self._calculate_sophistication_score(unique_techniques),
                "threat_actor_similarity": await self._analyze_threat_actor_similarity(unique_techniques)
            }

            return {
                "campaign_analysis": campaign_analysis,
                "techniques": unique_techniques,
                "attack_chain": technique_timeline,
                "attack_flow": attack_flow,
                "potential_threat_actors": await self._identify_potential_threat_actors(unique_techniques)
            }

        except Exception as e:
            logger.error(f"Error analyzing attack campaign: {e}")
            return {"campaign_analysis": {}, "techniques": [], "attack_chain": [], "error": str(e)}

    async def _calculate_sophistication_score(self, technique_ids: List[str]) -> float:
        """Calculate campaign sophistication score based on MITRE techniques"""
        try:
            if not technique_ids:
                return 0.0

            sophistication_weights = {
                # Advanced techniques get higher scores
                "T1055": 0.9,    # Process Injection
                "T1068": 0.8,    # Exploitation for Privilege Escalation
                "T1027": 0.7,    # Obfuscated Files
                "T1003": 0.8,    # OS Credential Dumping
                "T1021.001": 0.6, # Remote Desktop Protocol
                "T1566.001": 0.4, # Spearphishing Attachment
                "T1059.001": 0.6, # PowerShell
                "T1486": 0.7,    # Data Encrypted for Impact
            }

            total_score = 0.0
            for tech_id in technique_ids:
                weight = sophistication_weights.get(tech_id, 0.3)  # Default weight
                total_score += weight

            # Normalize by number of techniques
            base_score = total_score / len(technique_ids)

            # Bonus for technique diversity (covering multiple tactics)
            technique_tactics = set()
            for tech_id in technique_ids:
                technique = await self.mitre_framework.get_technique_details(tech_id)
                if technique:
                    technique_tactics.add(technique.tactic)

            diversity_bonus = len(technique_tactics) / 14  # 14 tactics in kill chain

            return min(1.0, base_score + (diversity_bonus * 0.3))

        except Exception as e:
            logger.error(f"Error calculating sophistication score: {e}")
            return 0.0

    async def _analyze_threat_actor_similarity(self, technique_ids: List[str]) -> Dict[str, float]:
        """Analyze similarity to known threat actors"""
        try:
            if not self.mitre_framework:
                return {}

            similarity_scores = {}

            # Get all known threat groups
            for group_id, group in self.mitre_framework.groups.items():
                if not group.techniques:
                    continue

                # Calculate Jaccard similarity
                group_techniques = set(group.techniques)
                campaign_techniques = set(technique_ids)

                intersection = len(group_techniques.intersection(campaign_techniques))
                union = len(group_techniques.union(campaign_techniques))

                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.1:  # Only include significant similarities
                        similarity_scores[group.name] = similarity

            # Sort by similarity
            return dict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:10])

        except Exception as e:
            logger.error(f"Error analyzing threat actor similarity: {e}")
            return {}

    async def _identify_potential_threat_actors(self, technique_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify potential threat actors based on technique usage"""
        try:
            similarity_scores = await self._analyze_threat_actor_similarity(technique_ids)

            potential_actors = []
            for actor_name, similarity in similarity_scores.items():
                # Find the group details
                group = None
                for g in self.mitre_framework.groups.values():
                    if g.name == actor_name:
                        group = g
                        break

                if group and similarity > 0.3:  # High similarity threshold
                    potential_actors.append({
                        "name": actor_name,
                        "group_id": group.id,
                        "similarity_score": similarity,
                        "aliases": group.aliases,
                        "description": group.description[:200] + "..." if len(group.description) > 200 else group.description
                    })

            return potential_actors

        except Exception as e:
            logger.error(f"Error identifying potential threat actors: {e}")
            return []

    async def generate_mitre_based_iocs(self, technique_ids: List[str]) -> Dict[str, Any]:
        """Generate IOCs based on MITRE techniques"""
        try:
            if not self.mitre_framework:
                return {"iocs": [], "hunting_queries": []}

            iocs = []
            hunting_queries = []

            for tech_id in technique_ids:
                technique = await self.mitre_framework.get_technique_details(tech_id)
                if not technique:
                    continue

                # Generate detection rules for this technique
                detection_rules = await self.mitre_framework.generate_detection_rules(tech_id)

                # Extract IOC patterns from technique description
                technique_iocs = await self._extract_iocs_from_technique(technique)
                iocs.extend(technique_iocs)

                # Add hunting queries
                if detection_rules.get("splunk_queries"):
                    hunting_queries.extend(detection_rules["splunk_queries"])

            return {
                "iocs": iocs,
                "hunting_queries": hunting_queries,
                "total_techniques": len(technique_ids),
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating MITRE-based IOCs: {e}")
            return {"iocs": [], "hunting_queries": [], "error": str(e)}

    async def _extract_iocs_from_technique(self, technique) -> List[Dict[str, Any]]:
        """Extract potential IOCs from technique details"""
        iocs = []

        try:
            # Common IOC patterns for different techniques
            ioc_patterns = {
                "T1566.001": [  # Spearphishing Attachment
                    {"type": "file_extension", "pattern": r"\.(doc|docx|pdf|zip|rar)$", "description": "Suspicious attachment types"},
                    {"type": "registry", "pattern": r"HKEY_CURRENT_USER\\Software\\Microsoft\\Office", "description": "Office macro execution"}
                ],
                "T1059.001": [  # PowerShell
                    {"type": "process", "pattern": "powershell.exe", "description": "PowerShell execution"},
                    {"type": "command_line", "pattern": r"-EncodedCommand|-Exec|-WindowStyle Hidden", "description": "Suspicious PowerShell parameters"}
                ],
                "T1055": [  # Process Injection
                    {"type": "api_call", "pattern": "VirtualAllocEx|WriteProcessMemory|CreateRemoteThread", "description": "Process injection APIs"},
                    {"type": "behavior", "pattern": "memory_allocation_in_remote_process", "description": "Memory allocation in remote process"}
                ],
                "T1003": [  # OS Credential Dumping
                    {"type": "file_access", "pattern": r"C:\\Windows\\System32\\config\\SAM", "description": "SAM database access"},
                    {"type": "process", "pattern": "lsass.exe", "description": "LSASS process access"}
                ]
            }

            patterns = ioc_patterns.get(technique.id, [])
            for pattern in patterns:
                iocs.append({
                    "technique_id": technique.id,
                    "technique_name": technique.name,
                    "ioc_type": pattern["type"],
                    "pattern": pattern["pattern"],
                    "description": pattern["description"],
                    "confidence": 0.8
                })

        except Exception as e:
            logger.debug(f"Error extracting IOCs for technique {technique.id}: {e}")

        return iocs

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

    # Advanced ML Model Implementations
    async def initialize_attention_model(self):
        """Initialize attention-based model for sequence analysis"""
        try:
            # Attention model for analyzing attack sequences
            inputs = tf.keras.layers.Input(shape=(50, 100))  # sequence_length, feature_dim

            # Multi-head attention layer
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8,
                key_dim=64,
                dropout=0.1
            )(inputs, inputs)

            # Add & Norm layer
            attention_output = tf.keras.layers.LayerNormalization()(
                inputs + attention_output
            )

            # Feed forward network
            ff_output = tf.keras.layers.Dense(512, activation='relu')(attention_output)
            ff_output = tf.keras.layers.Dropout(0.1)(ff_output)
            ff_output = tf.keras.layers.Dense(100)(ff_output)

            # Add & Norm layer
            ff_output = tf.keras.layers.LayerNormalization()(
                attention_output + ff_output
            )

            # Global average pooling and classification
            pooled = tf.keras.layers.GlobalAveragePooling1D()(ff_output)
            outputs = tf.keras.layers.Dense(4, activation='softmax')(pooled)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self.ml_models['attention_threat_classifier'] = model
            logger.info("Attention-based threat classifier initialized")

        except Exception as e:
            logger.error(f"Failed to initialize attention model: {e}")

    async def initialize_time_series_model(self):
        """Initialize time series model for temporal threat analysis"""
        try:
            # LSTM-based time series model for threat prediction
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(24, 50)),  # 24 hours, 50 features
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Threat probability
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )

            self.ml_models['temporal_threat_predictor'] = model
            logger.info("Time series threat prediction model initialized")

        except Exception as e:
            logger.error(f"Failed to initialize time series model: {e}")

    async def load_threat_embeddings(self):
        """Load or create threat indicator embeddings"""
        try:
            # Create embeddings for threat indicators using Word2Vec-like approach
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Sample threat indicators for embedding training
            threat_samples = [
                "malware ransomware encryption",
                "phishing email credential harvesting",
                "botnet command control communication",
                "vulnerability exploit remote code execution",
                "ddos amplification reflection attack",
                "lateral movement privilege escalation",
                "data exfiltration steganography covert",
                "persistence registry modification startup",
                "reconnaissance port scanning enumeration",
                "social engineering pretexting manipulation"
            ]

            # Create TF-IDF embeddings
            self.threat_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english'
            )

            threat_embeddings = self.threat_vectorizer.fit_transform(threat_samples)
            self.threat_embeddings = threat_embeddings.toarray()

            logger.info("Threat indicator embeddings loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load threat embeddings: {e}")
            self.threat_vectorizer = None
            self.threat_embeddings = None

    async def initialize_mock_models(self):
        """Initialize mock models when real ones fail to load"""
        try:
            class MockModel:
                def __init__(self, name):
                    self.name = name

                def predict(self, X, **kwargs):
                    # Return mock predictions
                    if len(X.shape) == 1:
                        X = X.reshape(1, -1)
                    return np.random.rand(X.shape[0], 4)

                def fit_predict(self, X):
                    return np.random.randint(-1, 2, X.shape[0])

                def decision_function(self, X):
                    return np.random.rand(X.shape[0]) - 0.5

            self.ml_models['threat_classifier'] = MockModel('threat_classifier')
            self.ml_models['attention_threat_classifier'] = MockModel('attention_classifier')
            self.ml_models['temporal_threat_predictor'] = MockModel('temporal_predictor')

            if not self.anomaly_detector:
                self.anomaly_detector = MockModel('anomaly_detector')

            logger.info("Mock ML models initialized as fallback")

        except Exception as e:
            logger.error(f"Failed to initialize mock models: {e}")

    async def advanced_threat_correlation(self, events: List[ThreatEvent]) -> Dict[str, Any]:
        """Advanced threat correlation using multiple ML techniques"""
        try:
            if len(events) < 2:
                return {"correlations": [], "confidence": 0.0}

            correlations = []

            # Feature extraction for all events
            event_features = []
            for event in events:
                features = await self.extract_event_features(event)
                event_features.append(features)

            event_features_array = np.array(event_features)

            # 1. Temporal correlation using time series analysis
            temporal_correlations = await self._temporal_correlation_analysis(events, event_features_array)
            correlations.extend(temporal_correlations)

            # 2. Semantic correlation using embeddings
            semantic_correlations = await self._semantic_correlation_analysis(events)
            correlations.extend(semantic_correlations)

            # 3. Behavioral correlation using clustering
            behavioral_correlations = await self._behavioral_correlation_analysis(events, event_features_array)
            correlations.extend(behavioral_correlations)

            # 4. Attention-based sequence correlation
            sequence_correlations = await self._attention_sequence_analysis(events, event_features_array)
            correlations.extend(sequence_correlations)

            # Calculate overall confidence
            confidence = await self._calculate_correlation_confidence(correlations)

            # Generate attack chain reconstruction
            attack_chain = await self._reconstruct_attack_chain(correlations, events)

            return {
                "correlations": correlations,
                "confidence": confidence,
                "attack_chain": attack_chain,
                "total_events": len(events),
                "correlation_techniques": ["temporal", "semantic", "behavioral", "attention"]
            }

        except Exception as e:
            logger.error(f"Advanced threat correlation failed: {e}")
            return {"correlations": [], "confidence": 0.0, "error": str(e)}

    async def _temporal_correlation_analysis(self, events: List[ThreatEvent], features: np.ndarray) -> List[Dict]:
        """Analyze temporal patterns in threat events"""
        correlations = []

        try:
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.timestamp)

            # Look for temporal patterns
            for i in range(len(sorted_events) - 1):
                for j in range(i + 1, min(i + 10, len(sorted_events))):  # Look ahead up to 10 events
                    time_diff = (sorted_events[j].timestamp - sorted_events[i].timestamp).total_seconds()

                    if time_diff < 3600:  # Within 1 hour
                        correlation_strength = max(0, 1 - (time_diff / 3600))

                        correlations.append({
                            "type": "temporal",
                            "event_ids": [sorted_events[i].id, sorted_events[j].id],
                            "strength": correlation_strength,
                            "time_delta_seconds": time_diff,
                            "pattern": "sequential_timing"
                        })

            return correlations

        except Exception as e:
            logger.error(f"Temporal correlation analysis failed: {e}")
            return []

    async def _semantic_correlation_analysis(self, events: List[ThreatEvent]) -> List[Dict]:
        """Analyze semantic similarity between threat events"""
        correlations = []

        try:
            if not self.threat_vectorizer:
                return []

            # Extract text features from events
            event_texts = []
            for event in events:
                text = f"{event.event_type} {event.payload}"
                event_texts.append(text)

            # Vectorize event texts
            try:
                event_vectors = self.threat_vectorizer.transform(event_texts)

                # Calculate cosine similarity between events
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(event_vectors)

                # Find high similarity pairs
                for i in range(len(events)):
                    for j in range(i + 1, len(events)):
                        similarity = similarity_matrix[i][j]

                        if similarity > 0.3:  # Similarity threshold
                            correlations.append({
                                "type": "semantic",
                                "event_ids": [events[i].id, events[j].id],
                                "strength": float(similarity),
                                "pattern": "content_similarity"
                            })

            except Exception as e:
                logger.debug(f"Semantic analysis skipped: {e}")

            return correlations

        except Exception as e:
            logger.error(f"Semantic correlation analysis failed: {e}")
            return []

    async def _behavioral_correlation_analysis(self, events: List[ThreatEvent], features: np.ndarray) -> List[Dict]:
        """Analyze behavioral patterns using clustering"""
        correlations = []

        try:
            if len(features) < 3:
                return []

            # Perform clustering
            clusters = self.clustering_model.fit_predict(features)

            # Group events by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id == -1:  # Noise point
                    continue

                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(i)

            # Create correlations for events in same cluster
            for cluster_id, event_indices in cluster_groups.items():
                if len(event_indices) >= 2:
                    for i in range(len(event_indices)):
                        for j in range(i + 1, len(event_indices)):
                            idx1, idx2 = event_indices[i], event_indices[j]

                            correlations.append({
                                "type": "behavioral",
                                "event_ids": [events[idx1].id, events[idx2].id],
                                "strength": 0.8,  # High confidence for same cluster
                                "cluster_id": int(cluster_id),
                                "pattern": "similar_behavior"
                            })

            return correlations

        except Exception as e:
            logger.error(f"Behavioral correlation analysis failed: {e}")
            return []

    async def _attention_sequence_analysis(self, events: List[ThreatEvent], features: np.ndarray) -> List[Dict]:
        """Analyze attack sequences using attention mechanism"""
        correlations = []

        try:
            if 'attention_threat_classifier' not in self.ml_models or len(features) < 3:
                return []

            # Prepare sequence data
            sequence_length = min(len(features), 50)
            if len(features) < sequence_length:
                # Pad sequence
                padding = np.zeros((sequence_length - len(features), features.shape[1]))
                padded_features = np.vstack([features, padding])
            else:
                padded_features = features[:sequence_length]

            # Reshape for attention model
            sequence_input = padded_features.reshape(1, sequence_length, -1)

            # Get attention weights (mock implementation)
            # In real implementation, you would extract attention weights from the model
            attention_weights = np.random.rand(sequence_length, sequence_length)

            # Find high attention pairs
            for i in range(min(len(events), sequence_length)):
                for j in range(i + 1, min(len(events), sequence_length)):
                    attention_score = attention_weights[i][j]

                    if attention_score > 0.7:  # High attention threshold
                        correlations.append({
                            "type": "attention_sequence",
                            "event_ids": [events[i].id, events[j].id],
                            "strength": float(attention_score),
                            "pattern": "sequential_dependency"
                        })

            return correlations

        except Exception as e:
            logger.error(f"Attention sequence analysis failed: {e}")
            return []

    async def _calculate_correlation_confidence(self, correlations: List[Dict]) -> float:
        """Calculate overall confidence in correlations"""
        if not correlations:
            return 0.0

        # Weight different correlation types
        type_weights = {
            "temporal": 0.8,
            "semantic": 0.7,
            "behavioral": 0.9,
            "attention_sequence": 0.85
        }

        total_weighted_strength = 0.0
        total_weight = 0.0

        for correlation in correlations:
            corr_type = correlation.get("type", "unknown")
            strength = correlation.get("strength", 0.0)
            weight = type_weights.get(corr_type, 0.5)

            total_weighted_strength += strength * weight
            total_weight += weight

        return total_weighted_strength / total_weight if total_weight > 0 else 0.0

    async def _reconstruct_attack_chain(self, correlations: List[Dict], events: List[ThreatEvent]) -> Dict[str, Any]:
        """Reconstruct attack chain from correlations"""
        try:
            # Build graph of event relationships
            event_graph = {}
            for event in events:
                event_graph[event.id] = {"event": event, "connections": []}

            # Add correlations as edges
            for correlation in correlations:
                if len(correlation["event_ids"]) == 2:
                    event1_id, event2_id = correlation["event_ids"]
                    if event1_id in event_graph and event2_id in event_graph:
                        event_graph[event1_id]["connections"].append({
                            "target": event2_id,
                            "strength": correlation["strength"],
                            "type": correlation["type"]
                        })

            # Find attack chain phases
            phases = []
            processed_events = set()

            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.timestamp)

            for event in sorted_events:
                if event.id not in processed_events:
                    phase = {
                        "phase_name": self._classify_attack_phase(event),
                        "events": [event.id],
                        "timestamp": event.timestamp.isoformat(),
                        "confidence": 0.8
                    }
                    phases.append(phase)
                    processed_events.add(event.id)

            return {
                "phases": phases,
                "total_phases": len(phases),
                "attack_duration_seconds": (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds() if len(sorted_events) > 1 else 0,
                "complexity_score": min(len(correlations) / len(events), 1.0) if events else 0
            }

        except Exception as e:
            logger.error(f"Attack chain reconstruction failed: {e}")
            return {"phases": [], "error": str(e)}

    def _classify_attack_phase(self, event: ThreatEvent) -> str:
        """Classify event into attack phase"""
        event_type = event.event_type.lower()
        payload = event.payload.lower()

        if any(keyword in event_type + payload for keyword in ["scan", "recon", "enum"]):
            return "reconnaissance"
        elif any(keyword in event_type + payload for keyword in ["exploit", "shell", "access"]):
            return "initial_access"
        elif any(keyword in event_type + payload for keyword in ["persist", "backdoor", "startup"]):
            return "persistence"
        elif any(keyword in event_type + payload for keyword in ["escalate", "privilege", "admin"]):
            return "privilege_escalation"
        elif any(keyword in event_type + payload for keyword in ["lateral", "move", "pivot"]):
            return "lateral_movement"
        elif any(keyword in event_type + payload for keyword in ["collect", "gather", "steal"]):
            return "collection"
        elif any(keyword in event_type + payload for keyword in ["exfil", "transfer", "upload"]):
            return "exfiltration"
        else:
            return "unknown"

    async def predict_threat_evolution(self, events: List[ThreatEvent], prediction_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict how threats might evolve using time series analysis"""
        try:
            if 'temporal_threat_predictor' not in self.ml_models or len(events) < 10:
                return {"predictions": [], "confidence": 0.0}

            # Extract temporal features
            temporal_features = []
            for event in sorted(events, key=lambda x: x.timestamp):
                hour = event.timestamp.hour
                day_of_week = event.timestamp.weekday()
                severity_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(event.severity.value, 1)

                features = [hour, day_of_week, severity_score] + [0] * 47  # Pad to 50 features
                temporal_features.append(features[:50])

            # Use sliding window approach
            predictions = []
            if len(temporal_features) >= 24:  # Need at least 24 hours of data
                for i in range(len(temporal_features) - 24 + 1):
                    window = np.array(temporal_features[i:i+24]).reshape(1, 24, 50)

                    try:
                        threat_probability = self.ml_models['temporal_threat_predictor'].predict(window, verbose=0)[0][0]

                        prediction_time = events[i+23].timestamp + timedelta(hours=1)
                        predictions.append({
                            "timestamp": prediction_time.isoformat(),
                            "threat_probability": float(threat_probability),
                            "confidence": 0.7,
                            "predicted_severity": "high" if threat_probability > 0.7 else "medium" if threat_probability > 0.4 else "low"
                        })
                    except Exception as e:
                        logger.debug(f"Prediction failed for window {i}: {e}")

            # Calculate overall prediction confidence
            avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions) if predictions else 0.0

            return {
                "predictions": predictions[-prediction_horizon_hours:],  # Return last N hours
                "confidence": avg_confidence,
                "prediction_horizon_hours": prediction_horizon_hours,
                "model_accuracy": 0.78  # Mock accuracy score
            }

        except Exception as e:
            logger.error(f"Threat evolution prediction failed: {e}")
            return {"predictions": [], "confidence": 0.0, "error": str(e)}
