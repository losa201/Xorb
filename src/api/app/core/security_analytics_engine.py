"""
Security Analytics Engine with Behavioral Analysis and Attack Pattern Recognition

This module provides enterprise-grade security analytics with:
- Advanced behavioral profiling and anomaly detection
- Real-time attack pattern recognition
- Risk scoring and threat hunting capabilities
- Security metrics and trend analysis
- Automated response recommendations
"""

import asyncio
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import uuid4
from collections import defaultdict, deque
import logging

import numpy as np
import structlog
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .advanced_threat_detection import ThreatCategory, ThreatSeverity, ConfidenceLevel

logger = structlog.get_logger("security_analytics_engine")


class RiskLevel(Enum):
    """Risk assessment levels"""
    MINIMAL = "minimal"         # 0-20
    LOW = "low"                # 21-40
    MODERATE = "moderate"      # 41-60
    HIGH = "high"              # 61-80
    SEVERE = "severe"          # 81-95
    EXTREME = "extreme"        # 96-100


class UserRiskCategory(Enum):
    """User risk categories"""
    TRUSTED = "trusted"
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    HIGH_RISK = "high_risk"
    COMPROMISED = "compromised"


class AttackStage(Enum):
    """MITRE ATT&CK framework stages"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class SecurityMetric:
    """Individual security metric with metadata"""
    metric_id: str
    name: str
    value: float
    timestamp: datetime
    category: str
    description: str
    severity: ThreatSeverity
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskScore:
    """Risk score with breakdown and explanation"""
    entity_id: str
    entity_type: str  # user, ip, tenant, endpoint
    overall_score: float
    risk_level: RiskLevel
    confidence: float
    
    # Score breakdown
    behavioral_score: float = 0.0
    temporal_score: float = 0.0
    geographical_score: float = 0.0
    access_pattern_score: float = 0.0
    threat_intel_score: float = 0.0
    
    # Contributing factors
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    
    # Trending
    score_trend: str = "stable"  # increasing, decreasing, stable
    previous_score: Optional[float] = None
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
    explanation: str = ""


@dataclass
class AttackChain:
    """Represents a sequence of related attack activities"""
    chain_id: str
    attacker_id: str  # IP, user, or identifier
    start_time: datetime
    last_activity: datetime
    
    # Attack progression
    stages: List[AttackStage] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    
    # Scope and impact
    affected_users: Set[str] = field(default_factory=set)
    affected_tenants: Set[str] = field(default_factory=set)
    affected_resources: Set[str] = field(default_factory=set)
    
    # Scoring
    severity: ThreatSeverity = ThreatSeverity.LOW
    confidence: float = 0.0
    impact_score: float = 0.0
    
    # Attribution
    source_ips: Set[str] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    geographical_indicators: Set[str] = field(default_factory=set)
    
    # Status
    is_active: bool = True
    is_blocked: bool = False
    human_reviewed: bool = False


@dataclass
class ThreatHuntingQuery:
    """Threat hunting query definition"""
    query_id: str
    name: str
    description: str
    query_logic: Dict[str, Any]
    severity: ThreatSeverity
    confidence: float
    
    # Execution
    is_active: bool = True
    execution_frequency: str = "real_time"  # real_time, hourly, daily
    last_execution: Optional[datetime] = None
    
    # Results
    matches_count: int = 0
    false_positive_rate: float = 0.0
    
    # MITRE mapping
    mitre_techniques: List[str] = field(default_factory=list)
    attack_stages: List[AttackStage] = field(default_factory=list)


class BehavioralProfiler:
    """Advanced behavioral profiling and anomaly detection"""
    
    def __init__(self):
        self.user_profiles = {}
        self.ip_profiles = {}
        self.tenant_profiles = {}
        
        # ML components
        self.scaler = MinMaxScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.pca = PCA(n_components=10)
        
        # Configuration
        self.learning_window_days = 30
        self.minimum_samples = 50
        self.anomaly_threshold = 0.7
        
    async def update_behavioral_profile(self, entity_type: str, entity_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update behavioral profile for an entity"""
        try:
            if entity_type == "user":
                profile = await self._update_user_profile(entity_id, event)
            elif entity_type == "ip":
                profile = await self._update_ip_profile(entity_id, event)
            elif entity_type == "tenant":
                profile = await self._update_tenant_profile(entity_id, event)
            else:
                return {}
            
            return profile
        
        except Exception as e:
            logger.error("Failed to update behavioral profile", 
                        entity_type=entity_type, entity_id=entity_id, error=str(e))
            return {}
    
    async def _update_user_profile(self, user_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update user behavioral profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "event_count": 0,
                
                # Temporal patterns
                "login_hours": defaultdict(int),
                "login_days": defaultdict(int),
                "session_durations": deque(maxlen=100),
                "activity_intervals": deque(maxlen=100),
                
                # Access patterns
                "endpoints_accessed": defaultdict(int),
                "user_agents": defaultdict(int),
                "source_ips": defaultdict(int),
                "geographic_locations": defaultdict(int),
                
                # Activity patterns
                "request_volumes": deque(maxlen=100),
                "data_volumes": deque(maxlen=100),
                "error_rates": deque(maxlen=100),
                "privilege_requests": deque(maxlen=100),
                
                # Security events
                "failed_logins": 0,
                "successful_logins": 0,
                "mfa_challenges": 0,
                "password_changes": 0,
                "privilege_escalations": 0,
                
                # Risk indicators
                "security_violations": 0,
                "policy_violations": 0,
                "anomaly_scores": deque(maxlen=50),
                "risk_scores": deque(maxlen=50),
                
                # Machine learning features
                "feature_vectors": deque(maxlen=1000),
                "cluster_assignments": deque(maxlen=100)
            }
        
        profile = self.user_profiles[user_id]
        
        # Update basic stats
        profile["last_seen"] = datetime.utcnow()
        profile["event_count"] += 1
        
        # Update temporal patterns
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        profile["login_hours"][timestamp.hour] += 1
        profile["login_days"][timestamp.weekday()] += 1
        
        if event.get('session_duration'):
            profile["session_durations"].append(event['session_duration'])
        
        # Update access patterns
        if event.get('endpoint'):
            profile["endpoints_accessed"][event['endpoint']] += 1
        
        if event.get('user_agent'):
            profile["user_agents"][event['user_agent']] += 1
        
        if event.get('source_ip'):
            profile["source_ips"][event['source_ip']] += 1
        
        if event.get('country_code'):
            profile["geographic_locations"][event['country_code']] += 1
        
        # Update activity patterns
        if event.get('request_count'):
            profile["request_volumes"].append(event['request_count'])
        
        if event.get('data_volume'):
            profile["data_volumes"].append(event['data_volume'])
        
        # Update security events
        if event.get('event_type') == 'login':
            if event.get('success'):
                profile["successful_logins"] += 1
            else:
                profile["failed_logins"] += 1
        
        if event.get('mfa_challenge'):
            profile["mfa_challenges"] += 1
        
        if event.get('privilege_escalation'):
            profile["privilege_escalations"] += 1
        
        # Calculate anomaly score
        anomaly_score = await self._calculate_user_anomaly_score(user_id, event)
        profile["anomaly_scores"].append(anomaly_score)
        
        return profile
    
    async def _update_ip_profile(self, ip_address: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update IP address behavioral profile"""
        if ip_address not in self.ip_profiles:
            self.ip_profiles[ip_address] = {
                "ip_address": ip_address,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "event_count": 0,
                
                # Geographic info
                "countries": defaultdict(int),
                "asns": defaultdict(int),
                "isp_names": defaultdict(int),
                
                # Users and tenants from this IP
                "unique_users": set(),
                "unique_tenants": set(),
                "user_count_history": deque(maxlen=100),
                
                # Activity patterns
                "request_volumes": deque(maxlen=100),
                "error_rates": deque(maxlen=100),
                "endpoints_accessed": defaultdict(int),
                "user_agents": defaultdict(int),
                
                # Reputation indicators
                "failed_login_attempts": 0,
                "successful_logins": 0,
                "blocked_requests": 0,
                "malicious_payloads": 0,
                
                # Temporal patterns
                "activity_hours": defaultdict(int),
                "activity_days": defaultdict(int),
                
                # Risk scoring
                "reputation_score": 50.0,  # Neutral start
                "threat_indicators": [],
                "whitelist_status": False,
                "blacklist_status": False
            }
        
        profile = self.ip_profiles[ip_address]
        
        # Update basic stats
        profile["last_seen"] = datetime.utcnow()
        profile["event_count"] += 1
        
        # Update geographic info
        if event.get('country_code'):
            profile["countries"][event['country_code']] += 1
        
        if event.get('asn'):
            profile["asns"][event['asn']] += 1
        
        # Track users and tenants
        if event.get('user_id'):
            profile["unique_users"].add(event['user_id'])
        
        if event.get('tenant_id'):
            profile["unique_tenants"].add(event['tenant_id'])
        
        profile["user_count_history"].append(len(profile["unique_users"]))
        
        # Update activity patterns
        if event.get('endpoint'):
            profile["endpoints_accessed"][event['endpoint']] += 1
        
        if event.get('user_agent'):
            profile["user_agents"][event['user_agent']] += 1
        
        # Update reputation indicators
        if event.get('event_type') == 'login':
            if event.get('success'):
                profile["successful_logins"] += 1
            else:
                profile["failed_login_attempts"] += 1
        
        if event.get('blocked'):
            profile["blocked_requests"] += 1
        
        # Update temporal patterns
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        profile["activity_hours"][timestamp.hour] += 1
        profile["activity_days"][timestamp.weekday()] += 1
        
        # Update reputation score
        await self._update_ip_reputation(ip_address, event)
        
        return profile
    
    async def _update_tenant_profile(self, tenant_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update tenant behavioral profile"""
        if tenant_id not in self.tenant_profiles:
            self.tenant_profiles[tenant_id] = {
                "tenant_id": tenant_id,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "event_count": 0,
                
                # User activity
                "active_users": set(),
                "user_activity_history": deque(maxlen=100),
                "new_users_per_day": deque(maxlen=30),
                
                # Resource usage
                "api_usage": deque(maxlen=100),
                "data_volume": deque(maxlen=100),
                "compute_usage": deque(maxlen=100),
                
                # Security posture
                "security_events": defaultdict(int),
                "compliance_violations": 0,
                "failed_authentications": 0,
                "privilege_escalations": 0,
                
                # Geographic distribution
                "user_countries": defaultdict(int),
                "source_ip_ranges": defaultdict(int),
                
                # Service usage patterns
                "endpoints_usage": defaultdict(int),
                "feature_adoption": defaultdict(int),
                
                # Risk metrics
                "risk_score": 25.0,  # Low risk start for tenants
                "threat_events": 0,
                "incidents": 0
            }
        
        profile = self.tenant_profiles[tenant_id]
        
        # Update basic stats
        profile["last_seen"] = datetime.utcnow()
        profile["event_count"] += 1
        
        # Track active users
        if event.get('user_id'):
            profile["active_users"].add(event['user_id'])
        
        profile["user_activity_history"].append(len(profile["active_users"]))
        
        # Update resource usage
        if event.get('api_calls'):
            profile["api_usage"].append(event['api_calls'])
        
        if event.get('data_volume'):
            profile["data_volume"].append(event['data_volume'])
        
        # Update security events
        if event.get('security_event_type'):
            profile["security_events"][event['security_event_type']] += 1
        
        if event.get('failed_auth'):
            profile["failed_authentications"] += 1
        
        # Update geographic distribution
        if event.get('country_code'):
            profile["user_countries"][event['country_code']] += 1
        
        # Update service usage
        if event.get('endpoint'):
            profile["endpoints_usage"][event['endpoint']] += 1
        
        return profile
    
    async def _calculate_user_anomaly_score(self, user_id: str, event: Dict[str, Any]) -> float:
        """Calculate anomaly score for user behavior"""
        if user_id not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user_id]
        score = 0.0
        factors = []
        
        # Temporal anomalies
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        typical_hours = set(profile["login_hours"].keys())
        if typical_hours and timestamp.hour not in typical_hours:
            score += 0.3
            factors.append("unusual_hour")
        
        # Geographic anomalies
        if event.get('country_code'):
            typical_countries = set(profile["geographic_locations"].keys())
            if typical_countries and event['country_code'] not in typical_countries:
                score += 0.4
                factors.append("unusual_location")
        
        # Access pattern anomalies
        if event.get('endpoint'):
            typical_endpoints = set(profile["endpoints_accessed"].keys())
            if typical_endpoints and event['endpoint'] not in typical_endpoints:
                score += 0.2
                factors.append("unusual_endpoint")
        
        # Volume anomalies
        if event.get('request_count') and profile["request_volumes"]:
            avg_volume = np.mean(profile["request_volumes"])
            current_volume = event['request_count']
            if current_volume > avg_volume * 3:
                score += 0.5
                factors.append("unusual_volume")
        
        # User agent anomalies
        if event.get('user_agent'):
            typical_agents = set(profile["user_agents"].keys())
            if typical_agents and event['user_agent'] not in typical_agents:
                score += 0.3
                factors.append("unusual_user_agent")
        
        # Cap the score
        score = min(1.0, score)
        
        # Store factors for explanation
        event['anomaly_factors'] = factors
        
        return score
    
    async def _update_ip_reputation(self, ip_address: str, event: Dict[str, Any]):
        """Update IP reputation score"""
        if ip_address not in self.ip_profiles:
            return
        
        profile = self.ip_profiles[ip_address]
        current_score = profile["reputation_score"]
        
        # Positive indicators
        if event.get('event_type') == 'login' and event.get('success'):
            current_score += 1.0
        
        if event.get('compliance_check_passed'):
            current_score += 0.5
        
        # Negative indicators
        if event.get('failed_login'):
            current_score -= 2.0
        
        if event.get('blocked'):
            current_score -= 5.0
        
        if event.get('malicious_payload'):
            current_score -= 10.0
        
        if event.get('security_violation'):
            current_score -= 7.0
        
        # Apply decay towards neutral (50)
        decay_factor = 0.99
        if current_score > 50:
            current_score = 50 + (current_score - 50) * decay_factor
        else:
            current_score = 50 - (50 - current_score) * decay_factor
        
        # Clamp to valid range
        profile["reputation_score"] = max(0.0, min(100.0, current_score))
    
    async def detect_behavioral_anomalies(self, entity_type: str, entity_id: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies for an entity"""
        anomalies = []
        
        try:
            if entity_type == "user":
                anomalies.extend(await self._detect_user_anomalies(entity_id, event))
            elif entity_type == "ip":
                anomalies.extend(await self._detect_ip_anomalies(entity_id, event))
            elif entity_type == "tenant":
                anomalies.extend(await self._detect_tenant_anomalies(entity_id, event))
        
        except Exception as e:
            logger.error("Failed to detect behavioral anomalies",
                        entity_type=entity_type, entity_id=entity_id, error=str(e))
        
        return anomalies
    
    async def _detect_user_anomalies(self, user_id: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect user-specific behavioral anomalies"""
        if user_id not in self.user_profiles:
            return []
        
        anomalies = []
        profile = self.user_profiles[user_id]
        
        # Check if we have enough data
        if profile["event_count"] < self.minimum_samples:
            return []
        
        # Time-based anomalies
        timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
        
        # Unusual login time
        login_hours = profile["login_hours"]
        if login_hours:
            total_logins = sum(login_hours.values())
            hour_probability = login_hours.get(timestamp.hour, 0) / total_logins
            
            if hour_probability < 0.05:  # Less than 5% of usual activity
                anomalies.append({
                    "type": "temporal_anomaly",
                    "subtype": "unusual_login_time",
                    "severity": "medium",
                    "confidence": 0.8,
                    "details": {
                        "current_hour": timestamp.hour,
                        "typical_hours": list(login_hours.keys()),
                        "probability": hour_probability
                    }
                })
        
        # Volume anomalies
        if event.get('request_count') and profile["request_volumes"]:
            current_volume = event['request_count']
            volumes = list(profile["request_volumes"])
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            if std_volume > 0:
                z_score = (current_volume - mean_volume) / std_volume
                if abs(z_score) > 3:  # 3 standard deviations
                    anomalies.append({
                        "type": "volume_anomaly",
                        "subtype": "unusual_request_volume",
                        "severity": "high" if z_score > 0 else "medium",
                        "confidence": 0.9,
                        "details": {
                            "current_volume": current_volume,
                            "mean_volume": mean_volume,
                            "z_score": z_score
                        }
                    })
        
        # Geographic anomalies
        if event.get('country_code'):
            locations = profile["geographic_locations"]
            if locations and event['country_code'] not in locations:
                anomalies.append({
                    "type": "geographic_anomaly",
                    "subtype": "new_location",
                    "severity": "high",
                    "confidence": 0.85,
                    "details": {
                        "new_country": event['country_code'],
                        "typical_countries": list(locations.keys())
                    }
                })
        
        return anomalies
    
    async def _detect_ip_anomalies(self, ip_address: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect IP-specific behavioral anomalies"""
        if ip_address not in self.ip_profiles:
            return []
        
        anomalies = []
        profile = self.ip_profiles[ip_address]
        
        # Multiple user accounts from single IP
        if len(profile["unique_users"]) > 10:  # Suspicious threshold
            anomalies.append({
                "type": "ip_anomaly",
                "subtype": "multiple_users",
                "severity": "medium",
                "confidence": 0.7,
                "details": {
                    "user_count": len(profile["unique_users"]),
                    "threshold": 10
                }
            })
        
        # Rapid user switching
        if len(profile["user_count_history"]) > 5:
            recent_counts = list(profile["user_count_history"])[-5:]
            if max(recent_counts) - min(recent_counts) > 5:
                anomalies.append({
                    "type": "ip_anomaly",
                    "subtype": "rapid_user_switching",
                    "severity": "high",
                    "confidence": 0.8,
                    "details": {
                        "user_count_variation": max(recent_counts) - min(recent_counts),
                        "recent_counts": recent_counts
                    }
                })
        
        # Low reputation score
        if profile["reputation_score"] < 30:
            anomalies.append({
                "type": "ip_anomaly",
                "subtype": "low_reputation",
                "severity": "high",
                "confidence": 0.9,
                "details": {
                    "reputation_score": profile["reputation_score"],
                    "failed_logins": profile["failed_login_attempts"],
                    "blocked_requests": profile["blocked_requests"]
                }
            })
        
        return anomalies
    
    async def _detect_tenant_anomalies(self, tenant_id: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect tenant-specific behavioral anomalies"""
        if tenant_id not in self.tenant_profiles:
            return []
        
        anomalies = []
        profile = self.tenant_profiles[tenant_id]
        
        # Unusual spike in API usage
        if profile["api_usage"]:
            recent_usage = list(profile["api_usage"])[-10:]  # Last 10 measurements
            if len(recent_usage) >= 5:
                avg_usage = np.mean(recent_usage[:-1])
                current_usage = recent_usage[-1]
                
                if current_usage > avg_usage * 3:
                    anomalies.append({
                        "type": "tenant_anomaly",
                        "subtype": "api_usage_spike",
                        "severity": "medium",
                        "confidence": 0.75,
                        "details": {
                            "current_usage": current_usage,
                            "average_usage": avg_usage,
                            "spike_factor": current_usage / avg_usage
                        }
                    })
        
        # High number of security events
        total_security_events = sum(profile["security_events"].values())
        if total_security_events > 50:  # Threshold
            anomalies.append({
                "type": "tenant_anomaly",
                "subtype": "high_security_events",
                "severity": "high",
                "confidence": 0.85,
                "details": {
                    "total_events": total_security_events,
                    "event_breakdown": dict(profile["security_events"])
                }
            })
        
        return anomalies


class AttackChainAnalyzer:
    """Analyzes attack chains and progression using MITRE ATT&CK framework"""
    
    def __init__(self):
        self.active_chains = {}
        self.completed_chains = {}
        self.attack_techniques = self._load_attack_techniques()
        self.chain_timeout_hours = 24
        
    def _load_attack_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Load MITRE ATT&CK techniques mapping"""
        return {
            # Initial Access
            "T1078": {"name": "Valid Accounts", "stage": AttackStage.INITIAL_ACCESS},
            "T1190": {"name": "Exploit Public-Facing Application", "stage": AttackStage.INITIAL_ACCESS},
            "T1566": {"name": "Phishing", "stage": AttackStage.INITIAL_ACCESS},
            
            # Execution
            "T1059": {"name": "Command and Scripting Interpreter", "stage": AttackStage.EXECUTION},
            "T1203": {"name": "Exploitation for Client Execution", "stage": AttackStage.EXECUTION},
            
            # Persistence
            "T1098": {"name": "Account Manipulation", "stage": AttackStage.PERSISTENCE},
            "T1136": {"name": "Create Account", "stage": AttackStage.PERSISTENCE},
            
            # Privilege Escalation
            "T1068": {"name": "Exploitation for Privilege Escalation", "stage": AttackStage.PRIVILEGE_ESCALATION},
            "T1134": {"name": "Access Token Manipulation", "stage": AttackStage.PRIVILEGE_ESCALATION},
            
            # Defense Evasion
            "T1027": {"name": "Obfuscated Files or Information", "stage": AttackStage.DEFENSE_EVASION},
            "T1055": {"name": "Process Injection", "stage": AttackStage.DEFENSE_EVASION},
            
            # Credential Access
            "T1110": {"name": "Brute Force", "stage": AttackStage.CREDENTIAL_ACCESS},
            "T1003": {"name": "OS Credential Dumping", "stage": AttackStage.CREDENTIAL_ACCESS},
            
            # Discovery
            "T1083": {"name": "File and Directory Discovery", "stage": AttackStage.DISCOVERY},
            "T1018": {"name": "Remote System Discovery", "stage": AttackStage.DISCOVERY},
            
            # Lateral Movement
            "T1021": {"name": "Remote Services", "stage": AttackStage.LATERAL_MOVEMENT},
            "T1570": {"name": "Lateral Tool Transfer", "stage": AttackStage.LATERAL_MOVEMENT},
            
            # Collection
            "T1005": {"name": "Data from Local System", "stage": AttackStage.COLLECTION},
            "T1039": {"name": "Data from Network Shared Drive", "stage": AttackStage.COLLECTION},
            
            # Exfiltration
            "T1041": {"name": "Exfiltration Over C2 Channel", "stage": AttackStage.EXFILTRATION},
            "T1048": {"name": "Exfiltration Over Alternative Protocol", "stage": AttackStage.EXFILTRATION},
            
            # Impact
            "T1485": {"name": "Data Destruction", "stage": AttackStage.IMPACT},
            "T1486": {"name": "Data Encrypted for Impact", "stage": AttackStage.IMPACT}
        }
    
    async def analyze_event_for_attack_chain(self, event: Dict[str, Any]) -> Optional[AttackChain]:
        """Analyze security event for attack chain progression"""
        try:
            # Extract attacker identifier
            attacker_id = self._extract_attacker_id(event)
            if not attacker_id:
                return None
            
            # Detect MITRE technique
            technique = await self._detect_mitre_technique(event)
            if not technique:
                return None
            
            # Get or create attack chain
            chain = await self._get_or_create_chain(attacker_id, event)
            
            # Update chain with new technique
            await self._update_attack_chain(chain, technique, event)
            
            # Analyze chain progression
            await self._analyze_chain_progression(chain)
            
            return chain
        
        except Exception as e:
            logger.error("Failed to analyze attack chain", error=str(e))
            return None
    
    def _extract_attacker_id(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract unique attacker identifier from event"""
        # Primary: Source IP
        if event.get('source_ip'):
            return f"ip:{event['source_ip']}"
        
        # Secondary: User ID (for insider threats)
        if event.get('user_id'):
            return f"user:{event['user_id']}"
        
        # Tertiary: Session ID
        if event.get('session_id'):
            return f"session:{event['session_id']}"
        
        return None
    
    async def _detect_mitre_technique(self, event: Dict[str, Any]) -> Optional[str]:
        """Detect MITRE ATT&CK technique from event"""
        # Brute force detection
        if (event.get('failed_login') and 
            event.get('failed_attempts_count', 0) > 5):
            return "T1110"
        
        # Valid accounts usage
        if (event.get('successful_login') and 
            event.get('unusual_location')):
            return "T1078"
        
        # Account manipulation
        if event.get('privilege_escalation'):
            return "T1098"
        
        # Data discovery
        if (event.get('endpoint', '').startswith('/api/v1/data') and
            event.get('unusual_data_access')):
            return "T1083"
        
        # Data collection
        if (event.get('large_data_download') or
            event.get('bulk_data_access')):
            return "T1005"
        
        # Lateral movement
        if (event.get('cross_tenant_access') or
            event.get('unusual_resource_access')):
            return "T1021"
        
        # Exfiltration
        if (event.get('data_exfiltration_indicators') or
            event.get('unusual_outbound_traffic')):
            return "T1041"
        
        return None
    
    async def _get_or_create_chain(self, attacker_id: str, event: Dict[str, Any]) -> AttackChain:
        """Get existing or create new attack chain"""
        current_time = datetime.utcnow()
        
        # Look for active chain
        if attacker_id in self.active_chains:
            chain = self.active_chains[attacker_id]
            
            # Check if chain is still active (within timeout)
            time_diff = (current_time - chain.last_activity).total_seconds() / 3600
            if time_diff <= self.chain_timeout_hours:
                return chain
            else:
                # Move to completed chains
                self.completed_chains[chain.chain_id] = chain
                del self.active_chains[attacker_id]
        
        # Create new chain
        chain = AttackChain(
            chain_id=str(uuid4()),
            attacker_id=attacker_id,
            start_time=current_time,
            last_activity=current_time
        )
        
        # Set initial attribution
        if event.get('source_ip'):
            chain.source_ips.add(event['source_ip'])
        if event.get('user_agent'):
            chain.user_agents.add(event['user_agent'])
        if event.get('country_code'):
            chain.geographical_indicators.add(event['country_code'])
        
        self.active_chains[attacker_id] = chain
        return chain
    
    async def _update_attack_chain(self, chain: AttackChain, technique: str, event: Dict[str, Any]):
        """Update attack chain with new technique"""
        chain.last_activity = datetime.utcnow()
        
        # Add technique if not already present
        if technique not in chain.techniques:
            chain.techniques.append(technique)
            
            # Add corresponding stage
            technique_info = self.attack_techniques.get(technique, {})
            stage = technique_info.get('stage')
            if stage and stage not in chain.stages:
                chain.stages.append(stage)
        
        # Update affected resources
        if event.get('user_id'):
            chain.affected_users.add(event['user_id'])
        if event.get('tenant_id'):
            chain.affected_tenants.add(event['tenant_id'])
        if event.get('endpoint'):
            chain.affected_resources.add(event['endpoint'])
        
        # Update attribution
        if event.get('source_ip'):
            chain.source_ips.add(event['source_ip'])
        if event.get('user_agent'):
            chain.user_agents.add(event['user_agent'])
        if event.get('country_code'):
            chain.geographical_indicators.add(event['country_code'])
    
    async def _analyze_chain_progression(self, chain: AttackChain):
        """Analyze attack chain progression and calculate risk"""
        # Calculate severity based on stages reached
        stage_scores = {
            AttackStage.RECONNAISSANCE: 10,
            AttackStage.INITIAL_ACCESS: 20,
            AttackStage.EXECUTION: 30,
            AttackStage.PERSISTENCE: 40,
            AttackStage.PRIVILEGE_ESCALATION: 60,
            AttackStage.DEFENSE_EVASION: 50,
            AttackStage.CREDENTIAL_ACCESS: 70,
            AttackStage.DISCOVERY: 45,
            AttackStage.LATERAL_MOVEMENT: 80,
            AttackStage.COLLECTION: 85,
            AttackStage.EXFILTRATION: 95,
            AttackStage.IMPACT: 100
        }
        
        max_stage_score = max([stage_scores.get(stage, 0) for stage in chain.stages], default=0)
        
        # Calculate confidence based on technique count and diversity
        technique_count = len(chain.techniques)
        stage_diversity = len(set(chain.stages))
        
        confidence = min(1.0, (technique_count * 0.2) + (stage_diversity * 0.3))
        
        # Calculate impact based on scope
        impact_factors = [
            len(chain.affected_users) * 10,
            len(chain.affected_tenants) * 25,
            len(chain.affected_resources) * 5,
            len(chain.source_ips) * 2
        ]
        impact_score = min(100, sum(impact_factors))
        
        # Update chain scoring
        chain.severity = self._score_to_severity(max_stage_score)
        chain.confidence = confidence
        chain.impact_score = impact_score
        
        # Check if chain should trigger alerts
        if (max_stage_score >= 60 or  # Privilege escalation or higher
            len(chain.stages) >= 4 or   # Multiple stages
            impact_score >= 50):        # High impact
            
            logger.warning("High-risk attack chain detected",
                          chain_id=chain.chain_id,
                          attacker_id=chain.attacker_id,
                          stages=len(chain.stages),
                          techniques=len(chain.techniques),
                          severity=chain.severity.value,
                          impact_score=impact_score)
    
    def _score_to_severity(self, score: float) -> ThreatSeverity:
        """Convert numerical score to severity enum"""
        if score >= 95:
            return ThreatSeverity.EMERGENCY
        elif score >= 80:
            return ThreatSeverity.CRITICAL
        elif score >= 60:
            return ThreatSeverity.HIGH
        elif score >= 40:
            return ThreatSeverity.MEDIUM
        elif score >= 20:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO
    
    async def get_active_chains(self) -> List[AttackChain]:
        """Get all active attack chains"""
        return list(self.active_chains.values())
    
    async def get_chain_by_attacker(self, attacker_id: str) -> Optional[AttackChain]:
        """Get attack chain for specific attacker"""
        return self.active_chains.get(attacker_id)


class ThreatHuntingEngine:
    """Advanced threat hunting with custom queries and automated hunting"""
    
    def __init__(self):
        self.hunting_queries = {}
        self.query_results = defaultdict(list)
        self.execution_schedule = {}
        self.load_default_queries()
    
    def load_default_queries(self):
        """Load default threat hunting queries"""
        default_queries = [
            {
                "query_id": "credential_stuffing",
                "name": "Credential Stuffing Detection",
                "description": "Detect credential stuffing attacks",
                "query_logic": {
                    "conditions": [
                        {"field": "failed_logins", "operator": ">", "value": 100, "timeframe": "1h"},
                        {"field": "unique_usernames", "operator": ">", "value": 50, "timeframe": "1h"},
                        {"field": "source_ip_diversity", "operator": "<", "value": 5}
                    ],
                    "aggregation": "source_ip"
                },
                "severity": ThreatSeverity.HIGH,
                "confidence": 0.85,
                "mitre_techniques": ["T1110.004"],
                "attack_stages": [AttackStage.CREDENTIAL_ACCESS]
            },
            {
                "query_id": "data_exfiltration",
                "name": "Unusual Data Access Pattern",
                "description": "Detect potential data exfiltration",
                "query_logic": {
                    "conditions": [
                        {"field": "data_volume_gb", "operator": ">", "value": 1.0, "timeframe": "1h"},
                        {"field": "time_of_day", "operator": "outside", "value": "business_hours"},
                        {"field": "user_typical_volume_gb", "operator": "<", "value": 0.1}
                    ],
                    "aggregation": "user_id"
                },
                "severity": ThreatSeverity.CRITICAL,
                "confidence": 0.75,
                "mitre_techniques": ["T1041", "T1048"],
                "attack_stages": [AttackStage.EXFILTRATION]
            },
            {
                "query_id": "lateral_movement",
                "name": "Cross-Tenant Access",
                "description": "Detect lateral movement between tenants",
                "query_logic": {
                    "conditions": [
                        {"field": "tenants_accessed", "operator": ">", "value": 3, "timeframe": "4h"},
                        {"field": "user_typical_tenants", "operator": "<=", "value": 1},
                        {"field": "privilege_level", "operator": ">=", "value": "admin"}
                    ],
                    "aggregation": "user_id"
                },
                "severity": ThreatSeverity.HIGH,
                "confidence": 0.80,
                "mitre_techniques": ["T1021"],
                "attack_stages": [AttackStage.LATERAL_MOVEMENT]
            },
            {
                "query_id": "privilege_escalation",
                "name": "Suspicious Privilege Escalation",
                "description": "Detect unauthorized privilege escalation attempts",
                "query_logic": {
                    "conditions": [
                        {"field": "role_changes", "operator": ">", "value": 0, "timeframe": "1h"},
                        {"field": "role_elevation", "operator": "=", "value": True},
                        {"field": "justification_provided", "operator": "=", "value": False}
                    ],
                    "aggregation": "user_id"
                },
                "severity": ThreatSeverity.HIGH,
                "confidence": 0.90,
                "mitre_techniques": ["T1068", "T1134"],
                "attack_stages": [AttackStage.PRIVILEGE_ESCALATION]
            }
        ]
        
        for query_data in default_queries:
            query = ThreatHuntingQuery(**query_data)
            self.hunting_queries[query.query_id] = query
    
    async def execute_hunt(self, query_id: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a specific hunting query"""
        if query_id not in self.hunting_queries:
            return []
        
        query = self.hunting_queries[query_id]
        
        try:
            # Apply query logic
            matches = await self._apply_query_logic(query, events)
            
            # Update query statistics
            query.matches_count += len(matches)
            query.last_execution = datetime.utcnow()
            
            # Store results
            self.query_results[query_id].extend(matches)
            
            # Keep only recent results (last 1000)
            if len(self.query_results[query_id]) > 1000:
                self.query_results[query_id] = self.query_results[query_id][-1000:]
            
            logger.info("Threat hunting query executed",
                       query_id=query_id,
                       matches=len(matches),
                       total_events=len(events))
            
            return matches
        
        except Exception as e:
            logger.error("Failed to execute hunting query",
                        query_id=query_id, error=str(e))
            return []
    
    async def _apply_query_logic(self, query: ThreatHuntingQuery, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply query logic to events"""
        logic = query.query_logic
        conditions = logic.get("conditions", [])
        aggregation = logic.get("aggregation")
        
        if not conditions:
            return []
        
        # Group events by aggregation field
        if aggregation:
            event_groups = defaultdict(list)
            for event in events:
                key = event.get(aggregation)
                if key:
                    event_groups[key].append(event)
        else:
            event_groups = {"all": events}
        
        matches = []
        
        # Check each group against conditions
        for group_key, group_events in event_groups.items():
            if await self._check_conditions(conditions, group_events):
                # Create match record
                match = {
                    "query_id": query.query_id,
                    "query_name": query.name,
                    "match_time": datetime.utcnow().isoformat(),
                    "aggregation_key": group_key,
                    "event_count": len(group_events),
                    "severity": query.severity.value,
                    "confidence": query.confidence,
                    "mitre_techniques": query.mitre_techniques,
                    "attack_stages": [stage.value for stage in query.attack_stages],
                    "sample_events": group_events[:5]  # Include sample events
                }
                matches.append(match)
        
        return matches
    
    async def _check_conditions(self, conditions: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> bool:
        """Check if events meet all conditions"""
        for condition in conditions:
            if not await self._check_single_condition(condition, events):
                return False
        return True
    
    async def _check_single_condition(self, condition: Dict[str, Any], events: List[Dict[str, Any]]) -> bool:
        """Check a single condition against events"""
        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]
        timeframe = condition.get("timeframe", "1h")
        
        # Filter events by timeframe
        current_time = datetime.utcnow()
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            cutoff_time = current_time - timedelta(hours=hours)
        elif timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
            cutoff_time = current_time - timedelta(minutes=minutes)
        else:
            cutoff_time = current_time - timedelta(hours=1)  # Default 1 hour
        
        recent_events = [
            event for event in events
            if datetime.fromisoformat(event.get('timestamp', '2024-01-01')) >= cutoff_time
        ]
        
        # Calculate field value
        field_value = await self._calculate_field_value(field, recent_events)
        
        # Apply operator
        if operator == ">":
            return field_value > value
        elif operator == "<":
            return field_value < value
        elif operator == ">=":
            return field_value >= value
        elif operator == "<=":
            return field_value <= value
        elif operator == "=":
            return field_value == value
        elif operator == "!=":
            return field_value != value
        elif operator == "outside" and field == "time_of_day" and value == "business_hours":
            # Check if events occurred outside business hours
            outside_hours = 0
            for event in recent_events:
                timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
                if timestamp.hour < 9 or timestamp.hour > 17:
                    outside_hours += 1
            return outside_hours > 0
        
        return False
    
    async def _calculate_field_value(self, field: str, events: List[Dict[str, Any]]) -> Union[int, float]:
        """Calculate field value from events"""
        if field == "failed_logins":
            return sum(1 for event in events if event.get('failed_login'))
        
        elif field == "unique_usernames":
            usernames = set(event.get('username') for event in events if event.get('username'))
            return len(usernames)
        
        elif field == "source_ip_diversity":
            ips = set(event.get('source_ip') for event in events if event.get('source_ip'))
            return len(ips)
        
        elif field == "data_volume_gb":
            total_bytes = sum(event.get('data_volume', 0) for event in events)
            return total_bytes / (1024**3)  # Convert to GB
        
        elif field == "tenants_accessed":
            tenants = set(event.get('tenant_id') for event in events if event.get('tenant_id'))
            return len(tenants)
        
        elif field == "role_changes":
            return sum(1 for event in events if event.get('role_change'))
        
        elif field == "role_elevation":
            return any(event.get('role_elevation') for event in events)
        
        elif field == "justification_provided":
            return any(event.get('justification') for event in events)
        
        # Default: count events
        return len(events)
    
    async def execute_all_active_queries(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute all active hunting queries"""
        results = {}
        
        for query_id, query in self.hunting_queries.items():
            if query.is_active:
                matches = await self.execute_hunt(query_id, events)
                if matches:
                    results[query_id] = matches
        
        return results


class SecurityAnalyticsEngine:
    """Main security analytics engine orchestrating all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.behavioral_profiler = BehavioralProfiler()
        self.attack_chain_analyzer = AttackChainAnalyzer()
        self.threat_hunting_engine = ThreatHuntingEngine()
        
        # Analytics state
        self.risk_scores = {}
        self.security_metrics = deque(maxlen=10000)
        self.analytics_enabled = True
        
        # Statistics
        self.events_processed = 0
        self.threats_detected = 0
        self.risk_assessments = 0
        
        logger.info("Security analytics engine initialized")
    
    async def process_security_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process security events through all analytics components"""
        if not self.analytics_enabled:
            return {}
        
        try:
            results = {
                "events_processed": len(events),
                "behavioral_anomalies": [],
                "attack_chains": [],
                "threat_hunting_matches": {},
                "risk_scores": {},
                "security_metrics": []
            }
            
            # 1. Behavioral profiling and anomaly detection
            for event in events:
                # Update profiles for different entities
                entities = [
                    ("user", event.get('user_id')),
                    ("ip", event.get('source_ip')),
                    ("tenant", event.get('tenant_id'))
                ]
                
                for entity_type, entity_id in entities:
                    if entity_id:
                        # Update profile
                        await self.behavioral_profiler.update_behavioral_profile(
                            entity_type, entity_id, event
                        )
                        
                        # Detect anomalies
                        anomalies = await self.behavioral_profiler.detect_behavioral_anomalies(
                            entity_type, entity_id, event
                        )
                        results["behavioral_anomalies"].extend(anomalies)
            
            # 2. Attack chain analysis
            for event in events:
                chain = await self.attack_chain_analyzer.analyze_event_for_attack_chain(event)
                if chain:
                    results["attack_chains"].append({
                        "chain_id": chain.chain_id,
                        "attacker_id": chain.attacker_id,
                        "stages": [stage.value for stage in chain.stages],
                        "techniques": chain.techniques,
                        "severity": chain.severity.value,
                        "confidence": chain.confidence,
                        "impact_score": chain.impact_score
                    })
            
            # 3. Threat hunting
            hunting_results = await self.threat_hunting_engine.execute_all_active_queries(events)
            results["threat_hunting_matches"] = hunting_results
            
            # 4. Risk scoring
            risk_scores = await self._calculate_risk_scores(events)
            results["risk_scores"] = risk_scores
            
            # 5. Generate security metrics
            metrics = await self._generate_security_metrics(events, results)
            results["security_metrics"] = metrics
            
            # Update statistics
            self.events_processed += len(events)
            self.threats_detected += len(results["behavioral_anomalies"]) + len(results["attack_chains"])
            self.risk_assessments += len(risk_scores)
            
            return results
        
        except Exception as e:
            logger.error("Failed to process security events", error=str(e))
            return {}
    
    async def _calculate_risk_scores(self, events: List[Dict[str, Any]]) -> Dict[str, RiskScore]:
        """Calculate risk scores for entities"""
        risk_scores = {}
        
        # Group events by entity
        entities = defaultdict(list)
        for event in events:
            if event.get('user_id'):
                entities[f"user:{event['user_id']}"].append(event)
            if event.get('source_ip'):
                entities[f"ip:{event['source_ip']}"].append(event)
            if event.get('tenant_id'):
                entities[f"tenant:{event['tenant_id']}"].append(event)
        
        # Calculate risk score for each entity
        for entity_key, entity_events in entities.items():
            entity_type, entity_id = entity_key.split(":", 1)
            
            risk_score = await self._calculate_entity_risk_score(
                entity_type, entity_id, entity_events
            )
            risk_scores[entity_key] = risk_score
        
        return risk_scores
    
    async def _calculate_entity_risk_score(
        self, 
        entity_type: str, 
        entity_id: str, 
        events: List[Dict[str, Any]]
    ) -> RiskScore:
        """Calculate risk score for a specific entity"""
        # Base risk factors
        behavioral_score = 0.0
        temporal_score = 0.0
        geographical_score = 0.0
        access_pattern_score = 0.0
        threat_intel_score = 0.0
        
        risk_factors = []
        protective_factors = []
        
        # Behavioral scoring
        if entity_type == "user":
            profile = self.behavioral_profiler.user_profiles.get(entity_id, {})
            if profile:
                # Failed login rate
                failed_rate = profile.get("failed_logins", 0) / max(1, profile.get("successful_logins", 1))
                if failed_rate > 0.1:
                    behavioral_score += failed_rate * 30
                    risk_factors.append("high_failed_login_rate")
                
                # Privilege escalation attempts
                priv_attempts = profile.get("privilege_escalations", 0)
                if priv_attempts > 0:
                    behavioral_score += min(40, priv_attempts * 10)
                    risk_factors.append("privilege_escalation_attempts")
                
                # MFA usage
                if profile.get("mfa_challenges", 0) > 0:
                    protective_factors.append("mfa_enabled")
                    behavioral_score -= 10
        
        elif entity_type == "ip":
            profile = self.behavioral_profiler.ip_profiles.get(entity_id, {})
            if profile:
                # Reputation score
                reputation = profile.get("reputation_score", 50)
                if reputation < 30:
                    threat_intel_score += (50 - reputation) * 2
                    risk_factors.append("low_ip_reputation")
                elif reputation > 70:
                    protective_factors.append("good_ip_reputation")
                    threat_intel_score -= 10
                
                # Multiple users from single IP
                user_count = len(profile.get("unique_users", set()))
                if user_count > 10:
                    behavioral_score += min(30, user_count)
                    risk_factors.append("multiple_users_single_ip")
        
        # Temporal scoring
        current_time = datetime.utcnow()
        for event in events:
            timestamp = datetime.fromisoformat(event.get('timestamp', '2024-01-01'))
            
            # Off-hours activity
            if timestamp.hour < 6 or timestamp.hour > 22:
                temporal_score += 5
                if "off_hours_activity" not in risk_factors:
                    risk_factors.append("off_hours_activity")
            
            # Weekend activity
            if timestamp.weekday() >= 5:
                temporal_score += 3
                if "weekend_activity" not in risk_factors:
                    risk_factors.append("weekend_activity")
        
        # Geographical scoring
        countries = set()
        for event in events:
            if event.get('country_code'):
                countries.add(event['country_code'])
        
        if len(countries) > 2:
            geographical_score += len(countries) * 5
            risk_factors.append("multiple_countries")
        
        # Access pattern scoring
        endpoints = set()
        sensitive_endpoints = 0
        for event in events:
            if event.get('endpoint'):
                endpoints.add(event['endpoint'])
                if any(sensitive in event['endpoint'] for sensitive in ['admin', 'config', 'secret']):
                    sensitive_endpoints += 1
        
        if sensitive_endpoints > 0:
            access_pattern_score += sensitive_endpoints * 10
            risk_factors.append("sensitive_endpoint_access")
        
        # Calculate overall score
        component_scores = [
            behavioral_score,
            temporal_score,
            geographical_score,
            access_pattern_score,
            threat_intel_score
        ]
        
        overall_score = sum(component_scores)
        
        # Apply protective factors
        protection_discount = len(protective_factors) * 5
        overall_score = max(0, overall_score - protection_discount)
        
        # Cap at 100
        overall_score = min(100, overall_score)
        
        # Determine risk level
        risk_level = self._score_to_risk_level(overall_score)
        
        # Calculate confidence based on sample size
        sample_count = len(events)
        confidence = min(1.0, sample_count / 50)  # Full confidence at 50+ samples
        
        # Create explanation
        explanation = self._generate_risk_explanation(
            overall_score, risk_factors, protective_factors
        )
        
        return RiskScore(
            entity_id=entity_id,
            entity_type=entity_type,
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            behavioral_score=behavioral_score,
            temporal_score=temporal_score,
            geographical_score=geographical_score,
            access_pattern_score=access_pattern_score,
            threat_intel_score=threat_intel_score,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            sample_count=sample_count,
            explanation=explanation
        )
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numerical score to risk level"""
        if score >= 96:
            return RiskLevel.EXTREME
        elif score >= 81:
            return RiskLevel.SEVERE
        elif score >= 61:
            return RiskLevel.HIGH
        elif score >= 41:
            return RiskLevel.MODERATE
        elif score >= 21:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _generate_risk_explanation(
        self, 
        score: float, 
        risk_factors: List[str], 
        protective_factors: List[str]
    ) -> str:
        """Generate human-readable explanation of risk score"""
        explanation_parts = [f"Risk score: {score:.1f}/100"]
        
        if risk_factors:
            explanation_parts.append(f"Risk factors: {', '.join(risk_factors)}")
        
        if protective_factors:
            explanation_parts.append(f"Protective factors: {', '.join(protective_factors)}")
        
        return ". ".join(explanation_parts)
    
    async def _generate_security_metrics(
        self, 
        events: List[Dict[str, Any]], 
        analysis_results: Dict[str, Any]
    ) -> List[SecurityMetric]:
        """Generate security metrics from events and analysis"""
        metrics = []
        current_time = datetime.utcnow()
        
        # Event volume metrics
        metrics.append(SecurityMetric(
            metric_id=str(uuid4()),
            name="events_per_hour",
            value=len(events),
            timestamp=current_time,
            category="volume",
            description="Number of security events processed",
            severity=ThreatSeverity.INFO
        ))
        
        # Threat detection metrics
        threats_detected = len(analysis_results.get("behavioral_anomalies", [])) + \
                          len(analysis_results.get("attack_chains", []))
        
        metrics.append(SecurityMetric(
            metric_id=str(uuid4()),
            name="threats_detected",
            value=threats_detected,
            timestamp=current_time,
            category="threats",
            description="Number of threats detected",
            severity=ThreatSeverity.MEDIUM if threats_detected > 0 else ThreatSeverity.INFO
        ))
        
        # Risk score metrics
        risk_scores = analysis_results.get("risk_scores", {})
        if risk_scores:
            avg_risk = np.mean([score.overall_score for score in risk_scores.values()])
            metrics.append(SecurityMetric(
                metric_id=str(uuid4()),
                name="average_risk_score",
                value=avg_risk,
                timestamp=current_time,
                category="risk",
                description="Average risk score across entities",
                severity=ThreatSeverity.HIGH if avg_risk > 60 else ThreatSeverity.MEDIUM if avg_risk > 40 else ThreatSeverity.INFO
            ))
        
        # Store metrics for trending
        self.security_metrics.extend(metrics)
        
        return metrics
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            "events_processed": self.events_processed,
            "threats_detected": self.threats_detected,
            "risk_assessments": self.risk_assessments,
            "active_attack_chains": len(self.attack_chain_analyzer.active_chains),
            "user_profiles": len(self.behavioral_profiler.user_profiles),
            "ip_profiles": len(self.behavioral_profiler.ip_profiles),
            "tenant_profiles": len(self.behavioral_profiler.tenant_profiles),
            "hunting_queries": len(self.threat_hunting_engine.hunting_queries),
            "recent_metrics_count": len(self.security_metrics)
        }
    
    async def shutdown(self):
        """Shutdown the analytics engine"""
        self.analytics_enabled = False
        logger.info("Security analytics engine shutdown")


# Global instance
_security_analytics_engine: Optional[SecurityAnalyticsEngine] = None


def get_security_analytics_engine() -> SecurityAnalyticsEngine:
    """Get global security analytics engine instance"""
    global _security_analytics_engine
    if _security_analytics_engine is None:
        _security_analytics_engine = SecurityAnalyticsEngine()
    return _security_analytics_engine


async def initialize_security_analytics_engine(config: Optional[Dict[str, Any]] = None) -> SecurityAnalyticsEngine:
    """Initialize global security analytics engine"""
    global _security_analytics_engine
    _security_analytics_engine = SecurityAnalyticsEngine(config)
    return _security_analytics_engine