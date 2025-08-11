"""
Advanced AI-Powered Threat Prediction Engine
Machine learning-based threat forecasting, attack pattern prediction, and automated response
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import hashlib
import re

# ML imports with graceful fallback
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

class ThreatPredictionLevel(Enum):
    """Threat prediction confidence levels"""
    IMMINENT = "imminent"      # Attack likely within hours
    HIGH = "high"              # Attack likely within days
    MEDIUM = "medium"          # Attack possible within weeks
    LOW = "low"                # Attack unlikely
    MINIMAL = "minimal"        # No threat detected

class AttackVector(Enum):
    """Predicted attack vectors"""
    WEB_APPLICATION = "web_application"
    NETWORK_INTRUSION = "network_intrusion"
    MALWARE_INFECTION = "malware_infection"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    COMMAND_CONTROL = "command_control"
    RECONNAISSANCE = "reconnaissance"

class PredictionConfidence(Enum):
    """AI prediction confidence levels"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"              # 75-89%
    MEDIUM = "medium"          # 60-74%
    LOW = "low"                # 40-59%
    VERY_LOW = "very_low"      # <40%

@dataclass
class ThreatPrediction:
    """AI threat prediction result"""
    prediction_id: str
    attack_vector: AttackVector
    threat_level: ThreatPredictionLevel
    confidence: PredictionConfidence
    probability_score: float
    predicted_timeframe: str
    target_assets: List[str]
    attack_techniques: List[str]
    indicators: List[Dict[str, Any]]
    countermeasures: List[str]
    risk_factors: Dict[str, float]
    created_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['attack_vector'] = self.attack_vector.value
        data['threat_level'] = self.threat_level.value
        data['confidence'] = self.confidence.value
        data['created_at'] = self.created_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        return data

@dataclass
class AttackPattern:
    """Detected attack pattern"""
    pattern_id: str
    pattern_name: str
    pattern_type: str
    confidence: float
    indicators: List[str]
    timestamps: List[datetime]
    source_ips: List[str]
    targets: List[str]
    mitre_techniques: List[str]
    severity: str

class AdvancedAIThreatPredictor:
    """Advanced AI-powered threat prediction and analysis engine"""
    
    def __init__(self):
        self.models = {}
        self.feature_scalers = {}
        self.label_encoders = {}
        self.attack_patterns = {}
        self.threat_predictions = deque(maxlen=1000)
        self.training_data = []
        self.model_performance = {}
        
        # Initialize models if ML is available
        if ML_AVAILABLE:
            self._initialize_ml_models()
        else:
            logger.warning("ML libraries not available, using fallback prediction algorithms")
        
        # Initialize threat intelligence patterns
        self._initialize_attack_patterns()
        self._initialize_mitre_mappings()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Anomaly detection model
            self.models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Attack classification model
            self.models['attack_classifier'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
            
            # Threat prediction model
            self.models['threat_predictor'] = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Clustering for attack pattern discovery
            self.models['pattern_clusterer'] = DBSCAN(
                eps=0.5,
                min_samples=3
            )
            
            # Feature scalers
            self.feature_scalers['standard'] = StandardScaler()
            
            # Label encoders
            self.label_encoders['attack_type'] = LabelEncoder()
            
            logger.info("✅ AI threat prediction models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    def _initialize_attack_patterns(self):
        """Initialize known attack patterns and signatures"""
        self.attack_signatures = {
            'sql_injection': {
                'patterns': [
                    r"(?i)(union.*select|select.*from|insert.*into)",
                    r"(?i)(\'\s*or\s*\'|--\s*$|;\s*--)",
                    r"(?i)(exec\s*\(|sp_executesql)"
                ],
                'weight': 0.9,
                'mitre_techniques': ['T1190', 'T1059']
            },
            'xss_attack': {
                'patterns': [
                    r"(?i)(<script|javascript:|onload=)",
                    r"(?i)(alert\s*\(|confirm\s*\()",
                    r"(?i)(<img.*src.*javascript)"
                ],
                'weight': 0.8,
                'mitre_techniques': ['T1189', 'T1059.007']
            },
            'command_injection': {
                'patterns': [
                    r"(?i)(;|\||&|\$\(|`)",
                    r"(?i)(nc\s+-l|bash\s+-i|/bin/sh)",
                    r"(?i)(wget\s+|curl\s+.*http)"
                ],
                'weight': 0.95,
                'mitre_techniques': ['T1059', 'T1190']
            },
            'directory_traversal': {
                'patterns': [
                    r"(?i)(\.\.\/|\.\.\\|%2e%2e)",
                    r"(?i)(\/etc\/passwd|\/windows\/system32)",
                    r"(?i)(\.\.%2f|\.\.%5c)"
                ],
                'weight': 0.7,
                'mitre_techniques': ['T1083', 'T1005']
            },
            'reconnaissance': {
                'patterns': [
                    r"(?i)(nmap|masscan|zmap)",
                    r"(?i)(\/admin|\/wp-admin|\/phpmyadmin)",
                    r"(?i)(\.git\/|\.env|\/config)"
                ],
                'weight': 0.6,
                'mitre_techniques': ['T1595', 'T1590']
            }
        }
    
    def _initialize_mitre_mappings(self):
        """Initialize MITRE ATT&CK technique mappings"""
        self.mitre_mappings = {
            'T1190': 'Exploit Public-Facing Application',
            'T1059': 'Command and Scripting Interpreter',
            'T1189': 'Drive-by Compromise',
            'T1083': 'File and Directory Discovery',
            'T1005': 'Data from Local System',
            'T1595': 'Active Scanning',
            'T1590': 'Gather Victim Network Information',
            'T1059.007': 'JavaScript',
            'T1071': 'Application Layer Protocol',
            'T1105': 'Ingress Tool Transfer'
        }
    
    async def predict_threats(self, 
                            historical_events: List[Dict[str, Any]], 
                            current_context: Dict[str, Any]) -> List[ThreatPrediction]:
        """Generate AI-powered threat predictions"""
        
        try:
            predictions = []
            
            # Analyze attack patterns
            attack_patterns = await self._detect_attack_patterns(historical_events)
            
            # Predict attack vectors
            for pattern in attack_patterns:
                prediction = await self._generate_threat_prediction(pattern, current_context)
                if prediction:
                    predictions.append(prediction)
            
            # ML-based predictions if available
            if ML_AVAILABLE and len(historical_events) > 50:
                ml_predictions = await self._ml_threat_prediction(historical_events, current_context)
                predictions.extend(ml_predictions)
            
            # Fallback rule-based predictions
            if not predictions or not ML_AVAILABLE:
                fallback_predictions = await self._rule_based_prediction(historical_events, current_context)
                predictions.extend(fallback_predictions)
            
            # Store predictions
            for pred in predictions:
                self.threat_predictions.append(pred)
            
            # Sort by threat level and confidence
            predictions.sort(key=lambda x: (
                self._threat_level_priority(x.threat_level),
                self._confidence_score(x.confidence)
            ), reverse=True)
            
            return predictions[:10]  # Return top 10 predictions
            
        except Exception as e:
            logger.error(f"Error in threat prediction: {e}")
            return []
    
    async def _detect_attack_patterns(self, events: List[Dict[str, Any]]) -> List[AttackPattern]:
        """Detect attack patterns in historical events"""
        
        patterns = []
        
        try:
            # Group events by time windows
            time_windows = self._create_time_windows(events, window_size=300)  # 5-minute windows
            
            for window_start, window_events in time_windows.items():
                if len(window_events) < 3:  # Need minimum events for pattern
                    continue
                
                # Analyze patterns within time window
                pattern = await self._analyze_event_cluster(window_events, window_start)
                if pattern:
                    patterns.append(pattern)
            
            # Cross-window pattern analysis
            cross_patterns = await self._analyze_cross_window_patterns(time_windows)
            patterns.extend(cross_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting attack patterns: {e}")
            return []
    
    async def _analyze_event_cluster(self, events: List[Dict[str, Any]], window_start: datetime) -> Optional[AttackPattern]:
        """Analyze cluster of events for attack patterns"""
        
        try:
            # Extract features
            source_ips = [e.get('source_ip', '') for e in events]
            targets = [e.get('target', '') for e in events]
            payloads = [e.get('payload', '') for e in events]
            user_agents = [e.get('user_agent', '') for e in events]
            
            # Check for attack signatures
            detected_attacks = []
            confidence_scores = []
            
            combined_payload = ' '.join(payloads)
            combined_ua = ' '.join(user_agents)
            combined_targets = ' '.join(targets)
            
            for attack_type, signature in self.attack_signatures.items():
                for pattern in signature['patterns']:
                    if re.search(pattern, combined_payload + ' ' + combined_targets + ' ' + combined_ua):
                        detected_attacks.append(attack_type)
                        confidence_scores.append(signature['weight'])
                        break
            
            if not detected_attacks:
                return None
            
            # Calculate overall confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Determine pattern severity
            severity = "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
            
            # Get MITRE techniques
            mitre_techniques = []
            for attack in detected_attacks:
                if attack in self.attack_signatures:
                    mitre_techniques.extend(self.attack_signatures[attack]['mitre_techniques'])
            
            pattern = AttackPattern(
                pattern_id=f"pattern_{hashlib.md5(str(window_start).encode()).hexdigest()[:8]}",
                pattern_name=f"Multi-vector attack: {', '.join(detected_attacks)}",
                pattern_type="coordinated_attack",
                confidence=avg_confidence,
                indicators=detected_attacks,
                timestamps=[window_start],
                source_ips=list(set(source_ips)),
                targets=list(set(targets)),
                mitre_techniques=list(set(mitre_techniques)),
                severity=severity
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing event cluster: {e}")
            return None
    
    async def _generate_threat_prediction(self, 
                                        pattern: AttackPattern, 
                                        context: Dict[str, Any]) -> Optional[ThreatPrediction]:
        """Generate threat prediction based on detected pattern"""
        
        try:
            # Determine attack vector based on pattern
            attack_vector = self._map_pattern_to_vector(pattern)
            
            # Calculate threat level
            threat_level = self._calculate_threat_level(pattern, context)
            
            # Determine confidence
            confidence = self._map_confidence_score(pattern.confidence)
            
            # Generate countermeasures
            countermeasures = self._generate_countermeasures(pattern, attack_vector)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(pattern, context)
            
            # Predict timeframe
            timeframe = self._predict_attack_timeframe(pattern, threat_level)
            
            prediction = ThreatPrediction(
                prediction_id=f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(pattern.pattern_id) % 1000:03d}",
                attack_vector=attack_vector,
                threat_level=threat_level,
                confidence=confidence,
                probability_score=pattern.confidence,
                predicted_timeframe=timeframe,
                target_assets=pattern.targets,
                attack_techniques=pattern.mitre_techniques,
                indicators=[
                    {
                        "type": "pattern",
                        "value": indicator,
                        "confidence": pattern.confidence
                    } for indicator in pattern.indicators
                ],
                countermeasures=countermeasures,
                risk_factors=risk_factors,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating threat prediction: {e}")
            return None
    
    async def _ml_threat_prediction(self, 
                                  events: List[Dict[str, Any]], 
                                  context: Dict[str, Any]) -> List[ThreatPrediction]:
        """Generate ML-based threat predictions"""
        
        if not ML_AVAILABLE:
            return []
        
        try:
            predictions = []
            
            # Prepare feature matrix
            features = self._extract_ml_features(events)
            if len(features) < 10:  # Need minimum data for ML
                return []
            
            # Train or update models with recent data
            await self._update_ml_models(features)
            
            # Generate anomaly predictions
            anomaly_predictions = await self._detect_ml_anomalies(features)
            predictions.extend(anomaly_predictions)
            
            # Generate attack type predictions
            attack_predictions = await self._predict_attack_types(features)
            predictions.extend(attack_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in ML threat prediction: {e}")
            return []
    
    async def _rule_based_prediction(self, 
                                   events: List[Dict[str, Any]], 
                                   context: Dict[str, Any]) -> List[ThreatPrediction]:
        """Generate rule-based threat predictions as fallback"""
        
        predictions = []
        
        try:
            # Analyze event frequency
            if len(events) > 100:  # High event volume
                prediction = ThreatPrediction(
                    prediction_id=f"rule_pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    attack_vector=AttackVector.RECONNAISSANCE,
                    threat_level=ThreatPredictionLevel.MEDIUM,
                    confidence=PredictionConfidence.MEDIUM,
                    probability_score=0.65,
                    predicted_timeframe="1-3 hours",
                    target_assets=["web_application"],
                    attack_techniques=["T1595"],
                    indicators=[{"type": "volume", "value": "high_event_rate", "confidence": 0.7}],
                    countermeasures=["Enable rate limiting", "Monitor for scanning tools"],
                    risk_factors={"event_volume": 0.8, "time_concentration": 0.6},
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=6)
                )
                predictions.append(prediction)
            
            # Check for suspicious user agents
            user_agents = [e.get('user_agent', '') for e in events]
            suspicious_agents = ['nmap', 'nikto', 'sqlmap', 'burp', 'zap']
            
            for agent in suspicious_agents:
                if any(agent.lower() in ua.lower() for ua in user_agents):
                    prediction = ThreatPrediction(
                        prediction_id=f"rule_pred_{agent}_{datetime.utcnow().strftime('%H%M%S')}",
                        attack_vector=AttackVector.RECONNAISSANCE,
                        threat_level=ThreatPredictionLevel.HIGH,
                        confidence=PredictionConfidence.HIGH,
                        probability_score=0.85,
                        predicted_timeframe="30 minutes - 2 hours",
                        target_assets=["web_application"],
                        attack_techniques=["T1595", "T1190"],
                        indicators=[{"type": "user_agent", "value": agent, "confidence": 0.9}],
                        countermeasures=[f"Block {agent} user agent", "Monitor for exploitation attempts"],
                        risk_factors={"tool_detection": 0.9, "automation": 0.8},
                        created_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(hours=4)
                    )
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {e}")
            return []
    
    def _create_time_windows(self, events: List[Dict[str, Any]], window_size: int) -> Dict[datetime, List[Dict[str, Any]]]:
        """Create time windows for event analysis"""
        
        windows = defaultdict(list)
        
        for event in events:
            timestamp = event.get('timestamp')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
            elif not isinstance(timestamp, datetime):
                continue
            
            # Round to window boundary
            window_start = timestamp.replace(
                minute=(timestamp.minute // (window_size // 60)) * (window_size // 60),
                second=0,
                microsecond=0
            )
            
            windows[window_start].append(event)
        
        return dict(windows)
    
    async def _analyze_cross_window_patterns(self, time_windows: Dict[datetime, List[Dict[str, Any]]]) -> List[AttackPattern]:
        """Analyze patterns across multiple time windows"""
        
        patterns = []
        
        try:
            # Look for escalating attack patterns
            sorted_windows = sorted(time_windows.items())
            
            for i in range(len(sorted_windows) - 2):
                window1_time, window1_events = sorted_windows[i]
                window2_time, window2_events = sorted_windows[i + 1]
                window3_time, window3_events = sorted_windows[i + 2]
                
                # Check for attack escalation
                escalation = self._detect_escalation_pattern(
                    window1_events, window2_events, window3_events
                )
                
                if escalation:
                    pattern = AttackPattern(
                        pattern_id=f"escalation_{window1_time.strftime('%H%M%S')}",
                        pattern_name="Attack Escalation Pattern",
                        pattern_type="escalation",
                        confidence=0.8,
                        indicators=["reconnaissance", "exploitation", "persistence"],
                        timestamps=[window1_time, window2_time, window3_time],
                        source_ips=list(set([
                            e.get('source_ip', '') for e in window1_events + window2_events + window3_events
                        ])),
                        targets=list(set([
                            e.get('target', '') for e in window1_events + window2_events + window3_events
                        ])),
                        mitre_techniques=["T1595", "T1190", "T1059"],
                        severity="high"
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing cross-window patterns: {e}")
            return []
    
    def _detect_escalation_pattern(self, window1: List, window2: List, window3: List) -> bool:
        """Detect if events show escalation pattern"""
        
        # Simple escalation detection: increasing event severity or complexity
        severities = [
            self._calculate_window_severity(window1),
            self._calculate_window_severity(window2),
            self._calculate_window_severity(window3)
        ]
        
        # Check for increasing severity
        return severities[1] > severities[0] and severities[2] > severities[1]
    
    def _calculate_window_severity(self, events: List[Dict[str, Any]]) -> float:
        """Calculate severity score for events in window"""
        
        if not events:
            return 0.0
        
        severity_map = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }
        
        total_severity = 0.0
        for event in events:
            severity = event.get('threat_level', 'low')
            total_severity += severity_map.get(severity, 0.4)
        
        return total_severity / len(events)
    
    # Helper methods for prediction generation
    
    def _map_pattern_to_vector(self, pattern: AttackPattern) -> AttackVector:
        """Map attack pattern to attack vector"""
        
        vector_mapping = {
            'sql_injection': AttackVector.WEB_APPLICATION,
            'xss_attack': AttackVector.WEB_APPLICATION,
            'command_injection': AttackVector.WEB_APPLICATION,
            'directory_traversal': AttackVector.WEB_APPLICATION,
            'reconnaissance': AttackVector.RECONNAISSANCE,
            'escalation': AttackVector.PRIVILEGE_ESCALATION
        }
        
        for indicator in pattern.indicators:
            if indicator in vector_mapping:
                return vector_mapping[indicator]
        
        return AttackVector.NETWORK_INTRUSION  # Default
    
    def _calculate_threat_level(self, pattern: AttackPattern, context: Dict[str, Any]) -> ThreatPredictionLevel:
        """Calculate threat level based on pattern and context"""
        
        base_score = pattern.confidence
        
        # Adjust based on severity
        if pattern.severity == "high":
            base_score += 0.2
        elif pattern.severity == "low":
            base_score -= 0.1
        
        # Adjust based on asset criticality
        critical_assets = context.get('critical_assets', [])
        if any(asset in pattern.targets for asset in critical_assets):
            base_score += 0.15
        
        # Map to threat level
        if base_score >= 0.9:
            return ThreatPredictionLevel.IMMINENT
        elif base_score >= 0.75:
            return ThreatPredictionLevel.HIGH
        elif base_score >= 0.6:
            return ThreatPredictionLevel.MEDIUM
        elif base_score >= 0.4:
            return ThreatPredictionLevel.LOW
        else:
            return ThreatPredictionLevel.MINIMAL
    
    def _map_confidence_score(self, score: float) -> PredictionConfidence:
        """Map numeric confidence to enum"""
        
        if score >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif score >= 0.75:
            return PredictionConfidence.HIGH
        elif score >= 0.6:
            return PredictionConfidence.MEDIUM
        elif score >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _generate_countermeasures(self, pattern: AttackPattern, vector: AttackVector) -> List[str]:
        """Generate countermeasures for attack pattern"""
        
        countermeasures = {
            AttackVector.WEB_APPLICATION: [
                "Enable Web Application Firewall (WAF)",
                "Implement input validation and sanitization",
                "Apply security headers",
                "Regular security patching"
            ],
            AttackVector.RECONNAISSANCE: [
                "Implement rate limiting",
                "Block suspicious user agents",
                "Enable intrusion detection",
                "Monitor access patterns"
            ],
            AttackVector.NETWORK_INTRUSION: [
                "Enable network segmentation",
                "Implement intrusion prevention system",
                "Monitor network traffic",
                "Apply principle of least privilege"
            ]
        }
        
        return countermeasures.get(vector, ["Monitor and investigate"])
    
    def _calculate_risk_factors(self, pattern: AttackPattern, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk factors for the threat"""
        
        return {
            'attack_sophistication': pattern.confidence,
            'asset_exposure': 0.7,  # Based on context
            'detection_difficulty': 1.0 - pattern.confidence,
            'potential_impact': 0.8,  # Based on targets
            'exploitability': pattern.confidence * 0.9
        }
    
    def _predict_attack_timeframe(self, pattern: AttackPattern, threat_level: ThreatPredictionLevel) -> str:
        """Predict when attack might occur"""
        
        timeframes = {
            ThreatPredictionLevel.IMMINENT: "1-6 hours",
            ThreatPredictionLevel.HIGH: "6-24 hours",
            ThreatPredictionLevel.MEDIUM: "1-7 days",
            ThreatPredictionLevel.LOW: "1-4 weeks",
            ThreatPredictionLevel.MINIMAL: "Unknown"
        }
        
        return timeframes.get(threat_level, "Unknown")
    
    def _threat_level_priority(self, level: ThreatPredictionLevel) -> int:
        """Get priority score for threat level"""
        
        priorities = {
            ThreatPredictionLevel.IMMINENT: 5,
            ThreatPredictionLevel.HIGH: 4,
            ThreatPredictionLevel.MEDIUM: 3,
            ThreatPredictionLevel.LOW: 2,
            ThreatPredictionLevel.MINIMAL: 1
        }
        
        return priorities.get(level, 0)
    
    def _confidence_score(self, confidence: PredictionConfidence) -> float:
        """Get numeric score for confidence level"""
        
        scores = {
            PredictionConfidence.VERY_HIGH: 0.95,
            PredictionConfidence.HIGH: 0.8,
            PredictionConfidence.MEDIUM: 0.65,
            PredictionConfidence.LOW: 0.5,
            PredictionConfidence.VERY_LOW: 0.3
        }
        
        return scores.get(confidence, 0.5)
    
    # Placeholder ML methods (would be fully implemented with real ML pipeline)
    
    def _extract_ml_features(self, events: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract ML features from events"""
        # Simplified feature extraction
        return [[len(events), sum(1 for e in events if 'attack' in str(e))]]
    
    async def _update_ml_models(self, features: List[List[float]]):
        """Update ML models with new data"""
        pass  # Placeholder
    
    async def _detect_ml_anomalies(self, features: List[List[float]]) -> List[ThreatPrediction]:
        """Detect anomalies using ML"""
        return []  # Placeholder
    
    async def _predict_attack_types(self, features: List[List[float]]) -> List[ThreatPrediction]:
        """Predict attack types using ML"""
        return []  # Placeholder

# Global threat predictor instance
threat_predictor = AdvancedAIThreatPredictor()

async def get_threat_predictor() -> AdvancedAIThreatPredictor:
    """Get threat predictor instance"""
    return threat_predictor

async def predict_threats(events: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[ThreatPrediction]:
    """Generate threat predictions"""
    predictor = await get_threat_predictor()
    return await predictor.predict_threats(events, context or {})