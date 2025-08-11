#!/usr/bin/env python3
"""
XORB Advanced Threat Detection System
ML-powered threat detection with behavioral analysis and automated response
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import sqlite3
import pickle
from pathlib import Path
import aiohttp
import redis.asyncio as redis

# Machine Learning imports
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available, using rule-based detection only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ThreatCategory(Enum):
    """Types of threats"""
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_INJECTION = "command_injection"
    UNKNOWN = "unknown"

@dataclass
class ThreatIndicator:
    """Threat indicator information"""
    indicator_id: str
    value: str
    type: str  # ip, hash, url, domain, etc.
    threat_level: ThreatLevel
    category: ThreatCategory
    source: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['threat_level'] = self.threat_level.value
        data['category'] = self.category.value
        data['first_seen'] = self.first_seen.isoformat()
        data['last_seen'] = self.last_seen.isoformat()
        return data

@dataclass
class ThreatEvent:
    """Detected threat event"""
    event_id: str
    timestamp: datetime
    source_ip: str
    target_ip: str
    threat_level: ThreatLevel
    category: ThreatCategory
    description: str
    indicators: List[str]
    confidence: float
    raw_data: Dict[str, Any]
    automated_response: bool = False
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['threat_level'] = self.threat_level.value
        data['category'] = self.category.value
        return data

class BehavioralAnalyzer:
    """Advanced behavioral analysis engine"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict] = {}
        self.baseline_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        
    def analyze_user_behavior(self, user_id: str, activity_data: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze user behavior for anomalies"""
        if not ML_AVAILABLE:
            return self._rule_based_behavioral_analysis(user_id, activity_data)
        
        try:
            # Extract features from activity data
            features = self._extract_behavioral_features(activity_data)
            
            # Get or create user profile
            if user_id not in self.user_profiles:
                self._create_user_profile(user_id, features)
                return 0.1, []  # New user, low anomaly score
            
            # Detect anomalies
            anomaly_score = self._detect_behavioral_anomalies(user_id, features)
            anomalous_activities = self._identify_anomalous_activities(user_id, activity_data, features)
            
            # Update user profile
            self._update_user_profile(user_id, features)
            
            return anomaly_score, anomalous_activities
            
        except Exception as e:
            logger.error(f"Behavioral analysis error: {e}")
            return 0.0, []
    
    def _extract_behavioral_features(self, activity_data: List[Dict]) -> np.ndarray:
        """Extract behavioral features from activity data"""
        features = []
        
        # Time-based features
        timestamps = [datetime.fromisoformat(a['timestamp']) for a in activity_data]
        if timestamps:
            avg_interval = np.mean([
                (timestamps[i+1] - timestamps[i]).total_seconds() 
                for i in range(len(timestamps)-1)
            ]) if len(timestamps) > 1 else 0
            
            work_hours = sum(1 for ts in timestamps if 9 <= ts.hour <= 17)
            weekend_activity = sum(1 for ts in timestamps if ts.weekday() >= 5)
            
            features.extend([
                len(activity_data),  # Activity count
                avg_interval,        # Average time between activities
                work_hours / len(timestamps) if timestamps else 0,
                weekend_activity / len(timestamps) if timestamps else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Resource access patterns
        unique_resources = len(set(a.get('resource', '') for a in activity_data))
        failed_attempts = sum(1 for a in activity_data if not a.get('success', True))
        
        features.extend([
            unique_resources,
            failed_attempts,
            failed_attempts / len(activity_data) if activity_data else 0
        ])
        
        # Network behavior
        unique_ips = len(set(a.get('source_ip', '') for a in activity_data))
        data_transferred = sum(a.get('bytes_transferred', 0) for a in activity_data)
        
        features.extend([
            unique_ips,
            data_transferred,
            data_transferred / len(activity_data) if activity_data else 0
        ])
        
        return np.array(features)
    
    def _rule_based_behavioral_analysis(self, user_id: str, activity_data: List[Dict]) -> Tuple[float, List[str]]:
        """Rule-based behavioral analysis fallback"""
        anomaly_score = 0.0
        anomalous_activities = []
        
        if not activity_data:
            return anomaly_score, anomalous_activities
        
        # Check for suspicious patterns
        failed_logins = sum(1 for a in activity_data 
                           if a.get('activity_type') == 'login' and not a.get('success', True))
        
        if failed_logins > 5:
            anomaly_score += 0.3
            anomalous_activities.append("Multiple failed login attempts")
        
        # Check for unusual time access
        unusual_time_access = sum(1 for a in activity_data
                                if datetime.fromisoformat(a['timestamp']).hour < 6 or 
                                   datetime.fromisoformat(a['timestamp']).hour > 22)
        
        if unusual_time_access > len(activity_data) * 0.3:
            anomaly_score += 0.2
            anomalous_activities.append("Unusual time access patterns")
        
        # Check for privilege escalation attempts
        privileged_actions = sum(1 for a in activity_data 
                               if a.get('requires_admin', False))
        
        if privileged_actions > 10:
            anomaly_score += 0.4
            anomalous_activities.append("Multiple privileged actions")
        
        return min(anomaly_score, 1.0), anomalous_activities

class AdvancedThreatDetector:
    """Advanced threat detection system with ML capabilities"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.redis_url = redis_url
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.active_threats: Dict[str, ThreatEvent] = {}
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.detection_rules: List[Dict] = []
        self.ml_models: Dict[str, Any] = {}
        
        # Initialize threat intelligence feeds
        self.threat_intel_sources = [
            "https://rules.emergingthreats.net/open/suricata/rules/",
            "https://www.malwaredomainlist.com/mdlcsv.php",
            "https://reputation.alienvault.com/reputation.data"
        ]
        
        # Setup database
        self.db_path = Path("threat_detection.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for threat storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threat_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                source_ip TEXT,
                target_ip TEXT,
                threat_level INTEGER,
                category TEXT,
                description TEXT,
                confidence REAL,
                raw_data TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threat_indicators (
                indicator_id TEXT PRIMARY KEY,
                value TEXT,
                type TEXT,
                threat_level INTEGER,
                category TEXT,
                confidence REAL,
                first_seen TEXT,
                last_seen TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def initialize(self):
        """Initialize threat detection system"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load detection rules
            await self._load_detection_rules()
            
            # Initialize ML models
            if ML_AVAILABLE:
                await self._initialize_ml_models()
            
            # Load threat intelligence
            await self._update_threat_intelligence()
            
            logger.info("Advanced Threat Detection System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize threat detector: {e}")
            raise
    
    async def _load_detection_rules(self):
        """Load threat detection rules"""
        rules = [
            {
                "rule_id": "brute_force_login",
                "pattern": "multiple_failed_logins",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "threat_level": ThreatLevel.HIGH,
                "category": ThreatCategory.INTRUSION
            },
            {
                "rule_id": "privilege_escalation",
                "pattern": "sudo_usage_spike",
                "threshold": 10,
                "time_window": 600,
                "threat_level": ThreatLevel.CRITICAL,
                "category": ThreatCategory.PRIVILEGE_ESCALATION
            },
            {
                "rule_id": "data_exfiltration",
                "pattern": "large_data_transfer",
                "threshold": 1000000,  # 1MB
                "time_window": 60,
                "threat_level": ThreatLevel.HIGH,
                "category": ThreatCategory.DATA_EXFILTRATION
            },
            {
                "rule_id": "lateral_movement",
                "pattern": "multiple_host_access",
                "threshold": 5,
                "time_window": 1800,  # 30 minutes
                "threat_level": ThreatLevel.MEDIUM,
                "category": ThreatCategory.LATERAL_MOVEMENT
            }
        ]
        
        self.detection_rules.extend(rules)
        logger.info(f"Loaded {len(rules)} detection rules")
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models for threat detection"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Anomaly detection model
            self.ml_models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Classification model for threat categorization
            self.ml_models['threat_classifier'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Feature scaler
            self.ml_models['feature_scaler'] = StandardScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence from external sources"""
        try:
            # This would normally fetch from real threat intel feeds
            # For demo purposes, we'll create some sample indicators
            sample_indicators = [
                {
                    "value": "192.168.1.100",
                    "type": "ip",
                    "threat_level": ThreatLevel.HIGH,
                    "category": ThreatCategory.MALWARE,
                    "confidence": 0.85,
                    "source": "internal_detection"
                },
                {
                    "value": "malicious.example.com",
                    "type": "domain",
                    "threat_level": ThreatLevel.CRITICAL,
                    "category": ThreatCategory.COMMAND_INJECTION,
                    "confidence": 0.95,
                    "source": "threat_feed_1"
                }
            ]
            
            for indicator_data in sample_indicators:
                indicator = ThreatIndicator(
                    indicator_id=hashlib.md5(indicator_data["value"].encode()).hexdigest(),
                    value=indicator_data["value"],
                    type=indicator_data["type"],
                    threat_level=indicator_data["threat_level"],
                    category=indicator_data["category"],
                    confidence=indicator_data["confidence"],
                    source=indicator_data["source"],
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    metadata={}
                )
                
                self.threat_indicators[indicator.indicator_id] = indicator
            
            logger.info(f"Updated threat intelligence with {len(sample_indicators)} indicators")
            
        except Exception as e:
            logger.error(f"Failed to update threat intelligence: {e}")
    
    async def analyze_network_traffic(self, traffic_data: List[Dict]) -> List[ThreatEvent]:
        """Analyze network traffic for threats"""
        detected_threats = []
        
        try:
            for packet in traffic_data:
                # Check against threat indicators
                source_ip = packet.get('source_ip', '')
                dest_ip = packet.get('dest_ip', '')
                
                # IP-based detection
                for indicator in self.threat_indicators.values():
                    if indicator.type == 'ip' and indicator.value in [source_ip, dest_ip]:
                        threat = ThreatEvent(
                            event_id=hashlib.md5(f"{packet}_{datetime.utcnow()}".encode()).hexdigest(),
                            timestamp=datetime.utcnow(),
                            source_ip=source_ip,
                            target_ip=dest_ip,
                            threat_level=indicator.threat_level,
                            category=indicator.category,
                            description=f"Communication with known malicious IP: {indicator.value}",
                            indicators=[indicator.indicator_id],
                            confidence=indicator.confidence,
                            raw_data=packet
                        )
                        detected_threats.append(threat)
                
                # Rule-based detection
                rule_threats = await self._apply_detection_rules([packet])
                detected_threats.extend(rule_threats)
            
            # Store detected threats
            for threat in detected_threats:
                await self._store_threat_event(threat)
                self.active_threats[threat.event_id] = threat
            
            return detected_threats
            
        except Exception as e:
            logger.error(f"Network traffic analysis error: {e}")
            return []
    
    async def analyze_system_logs(self, log_entries: List[Dict]) -> List[ThreatEvent]:
        """Analyze system logs for threats"""
        detected_threats = []
        
        try:
            # Group logs by user for behavioral analysis
            user_activities = {}
            for log in log_entries:
                user_id = log.get('user_id', 'unknown')
                if user_id not in user_activities:
                    user_activities[user_id] = []
                user_activities[user_id].append(log)
            
            # Analyze each user's behavior
            for user_id, activities in user_activities.items():
                anomaly_score, anomalous_activities = self.behavioral_analyzer.analyze_user_behavior(
                    user_id, activities
                )
                
                if anomaly_score > 0.7:  # High anomaly threshold
                    threat = ThreatEvent(
                        event_id=hashlib.md5(f"{user_id}_{datetime.utcnow()}".encode()).hexdigest(),
                        timestamp=datetime.utcnow(),
                        source_ip=activities[0].get('source_ip', 'unknown'),
                        target_ip='system',
                        threat_level=ThreatLevel.HIGH if anomaly_score > 0.8 else ThreatLevel.MEDIUM,
                        category=ThreatCategory.ANOMALOUS_BEHAVIOR,
                        description=f"Anomalous behavior detected for user {user_id}: {', '.join(anomalous_activities)}",
                        indicators=[],
                        confidence=anomaly_score,
                        raw_data={'user_activities': activities[:10]}  # Limit raw data size
                    )
                    detected_threats.append(threat)
            
            # Apply detection rules to all logs
            rule_threats = await self._apply_detection_rules(log_entries)
            detected_threats.extend(rule_threats)
            
            # Store detected threats
            for threat in detected_threats:
                await self._store_threat_event(threat)
                self.active_threats[threat.event_id] = threat
            
            return detected_threats
            
        except Exception as e:
            logger.error(f"System log analysis error: {e}")
            return []
    
    async def _apply_detection_rules(self, data_entries: List[Dict]) -> List[ThreatEvent]:
        """Apply detection rules to data entries"""
        detected_threats = []
        
        for rule in self.detection_rules:
            try:
                pattern = rule['pattern']
                threshold = rule['threshold']
                time_window = rule['time_window']
                
                # Apply rule based on pattern
                if pattern == "multiple_failed_logins":
                    failed_logins = [
                        entry for entry in data_entries
                        if entry.get('event_type') == 'login_failed'
                    ]
                    
                    if len(failed_logins) >= threshold:
                        source_ip = failed_logins[0].get('source_ip', 'unknown')
                        threat = ThreatEvent(
                            event_id=hashlib.md5(f"{rule['rule_id']}_{source_ip}_{datetime.utcnow()}".encode()).hexdigest(),
                            timestamp=datetime.utcnow(),
                            source_ip=source_ip,
                            target_ip='system',
                            threat_level=rule['threat_level'],
                            category=rule['category'],
                            description=f"Multiple failed login attempts detected from {source_ip}",
                            indicators=[],
                            confidence=0.8,
                            raw_data={'failed_attempts': len(failed_logins)}
                        )
                        detected_threats.append(threat)
                
                elif pattern == "large_data_transfer":
                    large_transfers = [
                        entry for entry in data_entries
                        if entry.get('bytes_transferred', 0) > threshold
                    ]
                    
                    for transfer in large_transfers:
                        threat = ThreatEvent(
                            event_id=hashlib.md5(f"{rule['rule_id']}_{transfer}_{datetime.utcnow()}".encode()).hexdigest(),
                            timestamp=datetime.utcnow(),
                            source_ip=transfer.get('source_ip', 'unknown'),
                            target_ip=transfer.get('dest_ip', 'unknown'),
                            threat_level=rule['threat_level'],
                            category=rule['category'],
                            description=f"Large data transfer detected: {transfer.get('bytes_transferred', 0)} bytes",
                            indicators=[],
                            confidence=0.7,
                            raw_data=transfer
                        )
                        detected_threats.append(threat)
                
            except Exception as e:
                logger.error(f"Rule application error for {rule['rule_id']}: {e}")
        
        return detected_threats
    
    async def _store_threat_event(self, threat: ThreatEvent):
        """Store threat event in database and Redis"""
        try:
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO threat_events
                (event_id, timestamp, source_ip, target_ip, threat_level, category, description, confidence, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat.event_id,
                threat.timestamp.isoformat(),
                threat.source_ip,
                threat.target_ip,
                threat.threat_level.value,
                threat.category.value,
                threat.description,
                threat.confidence,
                json.dumps(threat.raw_data)
            ))
            conn.commit()
            conn.close()
            
            # Store in Redis for real-time access
            await self.redis_client.setex(
                f"threat_event:{threat.event_id}",
                86400,  # 24 hours
                json.dumps(threat.to_dict())
            )
            
        except Exception as e:
            logger.error(f"Failed to store threat event: {e}")
    
    async def get_active_threats(self, threat_level: Optional[ThreatLevel] = None) -> List[ThreatEvent]:
        """Get currently active threats"""
        threats = list(self.active_threats.values())
        
        if threat_level:
            threats = [t for t in threats if t.threat_level == threat_level]
        
        return sorted(threats, key=lambda x: x.timestamp, reverse=True)
    
    async def continuous_monitoring(self):
        """Continuous threat monitoring loop"""
        while True:
            try:
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                # Clean up old threats
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                expired_threats = [
                    event_id for event_id, threat in self.active_threats.items()
                    if threat.timestamp < cutoff_time
                ]
                
                for event_id in expired_threats:
                    del self.active_threats[event_id]
                
                logger.info(f"Cleaned up {len(expired_threats)} expired threats")
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

async def main():
    """Main function for testing threat detection system"""
    detector = AdvancedThreatDetector()
    await detector.initialize()
    
    # Start continuous monitoring
    monitoring_task = asyncio.create_task(detector.continuous_monitoring())
    
    # Example network traffic analysis
    sample_traffic = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "source_ip": "192.168.1.100",
            "dest_ip": "10.0.0.5",
            "port": 443,
            "protocol": "tcp",
            "bytes_transferred": 1500
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "source_ip": "external.malicious.com",
            "dest_ip": "10.0.0.10",
            "port": 22,
            "protocol": "tcp",
            "bytes_transferred": 500
        }
    ]
    
    threats = await detector.analyze_network_traffic(sample_traffic)
    print(f"Detected {len(threats)} threats from network traffic")
    
    # Example system log analysis
    sample_logs = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user001",
            "event_type": "login_failed",
            "source_ip": "192.168.1.50",
            "success": False
        }
    ] * 6  # Simulate multiple failed logins
    
    log_threats = await detector.analyze_system_logs(sample_logs)
    print(f"Detected {len(log_threats)} threats from system logs")
    
    # Get all active threats
    active_threats = await detector.get_active_threats()
    print(f"Total active threats: {len(active_threats)}")
    
    for threat in active_threats:
        print(f"Threat: {threat.description} (Level: {threat.threat_level.name})")
    
    # Keep monitoring running
    await monitoring_task

if __name__ == "__main__":
    asyncio.run(main())