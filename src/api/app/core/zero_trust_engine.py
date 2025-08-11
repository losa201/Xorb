"""
Zero Trust Security Engine - Advanced Implementation
Implements comprehensive zero-trust architecture with continuous verification
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
import ipaddress
from contextlib import asynccontextmanager

import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)

class TrustLevel(Enum):
    """Trust levels for zero-trust evaluation"""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4

class RiskLevel(Enum):
    """Risk assessment levels"""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class AccessDecision(Enum):
    """Access control decisions"""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    MONITOR = "monitor"
    QUARANTINE = "quarantine"

@dataclass
class SecurityContext:
    """Security context for zero-trust evaluation"""
    user_id: str
    session_id: str
    device_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    location: Optional[Dict[str, str]] = None
    mfa_verified: bool = False
    device_trusted: bool = False
    network_trusted: bool = False
    behavior_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    previous_sessions: List[str] = field(default_factory=list)

@dataclass
class PolicyRule:
    """Zero-trust policy rule"""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    priority: int
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ThreatIndicator:
    """Security threat indicator"""
    indicator_id: str
    indicator_type: str
    value: str
    confidence: float
    severity: RiskLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str
    tags: List[str] = field(default_factory=list)

class ZeroTrustEngine:
    """
    Advanced Zero Trust Security Engine
    
    Implements comprehensive zero-trust architecture with:
    - Continuous identity verification
    - Device trust assessment
    - Network security monitoring
    - Behavioral analytics
    - Risk-based access control
    - Automated threat response
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Trust and risk thresholds
        self.trust_thresholds = {
            TrustLevel.VERIFIED: 0.9,
            TrustLevel.HIGH: 0.75,
            TrustLevel.MEDIUM: 0.5,
            TrustLevel.LOW: 0.25,
            TrustLevel.UNKNOWN: 0.0
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'device_trust': 0.25,
            'network_trust': 0.20,
            'behavior_score': 0.30,
            'location_anomaly': 0.15,
            'time_anomaly': 0.10
        }
        
        # Policy engine
        self.policies: Dict[str, PolicyRule] = {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        
        # Behavioral baselines
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        self.device_profiles: Dict[str, Dict[str, Any]] = {}
        self.network_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Session tracking
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_cache: Dict[str, Dict[str, Any]] = {}
        
        # Threat intelligence
        self.threat_feed_cache = {}
        self.threat_feed_updated = datetime.min
        
        logger.info("Zero Trust Engine initialized with advanced security policies")
    
    async def evaluate_trust(self, context: SecurityContext) -> Tuple[TrustLevel, float, AccessDecision]:
        """
        Comprehensive trust evaluation using multiple factors
        
        Returns:
            Tuple of (trust_level, confidence_score, access_decision)
        """
        try:
            # Initialize scoring components
            scores = {
                'identity_score': await self._evaluate_identity(context),
                'device_score': await self._evaluate_device(context),
                'network_score': await self._evaluate_network(context),
                'behavior_score': await self._evaluate_behavior(context),
                'location_score': await self._evaluate_location(context),
                'time_score': await self._evaluate_temporal_patterns(context)
            }
            
            # Check for threat indicators
            threat_score = await self._check_threat_indicators(context)
            
            # Calculate weighted trust score
            trust_score = self._calculate_weighted_score(scores)
            
            # Apply threat penalty
            trust_score = max(0.0, trust_score - threat_score)
            
            # Determine trust level
            trust_level = self._determine_trust_level(trust_score)
            
            # Make access decision
            access_decision = await self._make_access_decision(
                trust_level, trust_score, context, scores
            )
            
            # Log evaluation
            await self._log_trust_evaluation(context, trust_level, trust_score, access_decision, scores)
            
            # Update behavioral baselines
            await self._update_behavioral_baseline(context, scores)
            
            return trust_level, trust_score, access_decision
            
        except Exception as e:
            logger.error("Trust evaluation failed", error=str(e), context=context.user_id)
            return TrustLevel.UNKNOWN, 0.0, AccessDecision.DENY
    
    async def _evaluate_identity(self, context: SecurityContext) -> float:
        """Evaluate identity trust factors"""
        score = 0.5  # Base score
        
        # MFA verification boost
        if context.mfa_verified:
            score += 0.3
        
        # Session continuity
        if context.session_id in self.session_cache:
            cached_session = self.session_cache[context.session_id]
            if cached_session.get('verified_recently', False):
                score += 0.2
        
        # Previous authentication history
        if len(context.previous_sessions) > 0:
            recent_sessions = [
                s for s in context.previous_sessions 
                if s in self.session_cache
            ]
            if recent_sessions:
                score += min(0.2, len(recent_sessions) * 0.05)
        
        return min(1.0, score)
    
    async def _evaluate_device(self, context: SecurityContext) -> float:
        """Evaluate device trust factors"""
        score = 0.3  # Base score for unknown devices
        
        if context.device_trusted:
            score = 0.8
        
        # Check device profile
        if context.device_id in self.device_profiles:
            profile = self.device_profiles[context.device_id]
            
            # Consistent user agent
            if profile.get('user_agent') == context.user_agent:
                score += 0.1
            
            # Regular usage pattern
            last_seen = profile.get('last_seen')
            if last_seen and (datetime.utcnow() - last_seen).days < 7:
                score += 0.1
            
            # No security incidents
            if not profile.get('security_incidents', []):
                score += 0.1
        
        return min(1.0, score)
    
    async def _evaluate_network(self, context: SecurityContext) -> float:
        """Evaluate network trust factors"""
        score = 0.5  # Base score
        
        if context.network_trusted:
            score = 0.9
        
        try:
            ip = ipaddress.ip_address(context.ip_address)
            
            # Check if IP is in trusted ranges
            trusted_ranges = self.config.get('trusted_ip_ranges', [])
            for range_str in trusted_ranges:
                if ip in ipaddress.ip_network(range_str):
                    score += 0.3
                    break
            
            # Check for known malicious IPs
            if await self._is_malicious_ip(context.ip_address):
                score -= 0.5
            
            # Geographic consistency
            if context.location:
                expected_country = self.user_baselines.get(
                    context.user_id, {}
                ).get('typical_country')
                if expected_country == context.location.get('country'):
                    score += 0.1
            
        except ValueError:
            # Invalid IP address
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _evaluate_behavior(self, context: SecurityContext) -> float:
        """Evaluate behavioral patterns"""
        if context.user_id not in self.user_baselines:
            return 0.5  # Neutral score for new users
        
        baseline = self.user_baselines[context.user_id]
        score = 0.5
        
        # Time-based patterns
        current_hour = context.timestamp.hour
        typical_hours = baseline.get('typical_hours', [])
        if current_hour in typical_hours:
            score += 0.2
        elif abs(current_hour - min(typical_hours, default=12)) > 6:
            score -= 0.1
        
        # Access patterns
        typical_resources = baseline.get('typical_resources', set())
        recent_resources = baseline.get('recent_resources', set())
        
        if recent_resources and typical_resources:
            overlap = len(recent_resources & typical_resources) / len(typical_resources)
            score += overlap * 0.3
        
        # Velocity checks
        last_login = baseline.get('last_login')
        if last_login:
            time_diff = (context.timestamp - last_login).total_seconds()
            if time_diff < 60:  # Too fast
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _evaluate_location(self, context: SecurityContext) -> float:
        """Evaluate location-based risk factors"""
        if not context.location:
            return 0.5
        
        score = 0.7  # Base score
        
        # Check against user baseline
        if context.user_id in self.user_baselines:
            baseline = self.user_baselines[context.user_id]
            typical_countries = baseline.get('typical_countries', set())
            
            if context.location.get('country') in typical_countries:
                score += 0.2
            else:
                score -= 0.1
        
        # High-risk countries
        high_risk_countries = self.config.get('high_risk_countries', [])
        if context.location.get('country') in high_risk_countries:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _evaluate_temporal_patterns(self, context: SecurityContext) -> float:
        """Evaluate temporal access patterns"""
        score = 0.7
        
        # Business hours check
        current_hour = context.timestamp.hour
        if 8 <= current_hour <= 18:  # Business hours
            score += 0.2
        elif current_hour < 6 or current_hour > 22:  # Unusual hours
            score -= 0.1
        
        # Weekend check
        if context.timestamp.weekday() >= 5:  # Weekend
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _check_threat_indicators(self, context: SecurityContext) -> float:
        """Check for threat indicators"""
        threat_score = 0.0
        
        # Check IP against threat feeds
        ip_threats = await self._check_ip_threats(context.ip_address)
        threat_score += ip_threats
        
        # Check user agent for suspicious patterns
        ua_threats = await self._check_user_agent_threats(context.user_agent)
        threat_score += ua_threats
        
        # Check for rapid access patterns
        if await self._check_velocity_threats(context):
            threat_score += 0.3
        
        return min(1.0, threat_score)
    
    async def _check_ip_threats(self, ip_address: str) -> float:
        """Check IP against threat intelligence"""
        # Check local threat indicators
        for indicator in self.threat_indicators.values():
            if (indicator.indicator_type == 'ip' and 
                indicator.value == ip_address and
                indicator.confidence > 0.7):
                return 0.5
        
        # Could integrate with external threat feeds here
        return 0.0
    
    async def _check_user_agent_threats(self, user_agent: str) -> float:
        """Check user agent for threats"""
        suspicious_patterns = [
            'bot', 'crawler', 'scanner', 'sqlmap', 'nikto',
            'nessus', 'openvas', 'nmap', 'masscan'
        ]
        
        ua_lower = user_agent.lower()
        for pattern in suspicious_patterns:
            if pattern in ua_lower:
                return 0.3
        
        return 0.0
    
    async def _check_velocity_threats(self, context: SecurityContext) -> bool:
        """Check for velocity-based threats"""
        # Check if user has multiple recent sessions
        recent_sessions = [
            sess for sess in self.active_sessions.values()
            if (sess.user_id == context.user_id and
                (context.timestamp - sess.timestamp).total_seconds() < 300)
        ]
        
        return len(recent_sessions) > 3
    
    async def _is_malicious_ip(self, ip_address: str) -> bool:
        """Check if IP is known malicious"""
        # This could integrate with threat intelligence feeds
        malicious_ips = self.config.get('malicious_ips', set())
        return ip_address in malicious_ips
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted trust score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        score_weights = {
            'identity_score': 0.25,
            'device_score': 0.20,
            'network_score': 0.20,
            'behavior_score': 0.20,
            'location_score': 0.10,
            'time_score': 0.05
        }
        
        for score_type, score_value in scores.items():
            weight = score_weights.get(score_type, 0.0)
            weighted_sum += score_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Determine trust level based on score"""
        for level, threshold in sorted(
            self.trust_thresholds.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if trust_score >= threshold:
                return level
        return TrustLevel.UNKNOWN
    
    async def _make_access_decision(
        self, 
        trust_level: TrustLevel, 
        trust_score: float, 
        context: SecurityContext,
        scores: Dict[str, float]
    ) -> AccessDecision:
        """Make access control decision"""
        
        # High trust - allow
        if trust_level in [TrustLevel.VERIFIED, TrustLevel.HIGH]:
            return AccessDecision.ALLOW
        
        # Unknown/Low trust - deny
        if trust_level in [TrustLevel.UNKNOWN, TrustLevel.LOW]:
            return AccessDecision.DENY
        
        # Medium trust - conditional access
        if trust_level == TrustLevel.MEDIUM:
            # Check if MFA verified
            if context.mfa_verified:
                return AccessDecision.ALLOW
            else:
                return AccessDecision.CHALLENGE
        
        # Default to monitoring
        return AccessDecision.MONITOR
    
    async def _log_trust_evaluation(
        self, 
        context: SecurityContext, 
        trust_level: TrustLevel, 
        trust_score: float,
        access_decision: AccessDecision,
        scores: Dict[str, float]
    ):
        """Log trust evaluation for audit purposes"""
        logger.info(
            "Zero trust evaluation completed",
            user_id=context.user_id,
            session_id=context.session_id,
            trust_level=trust_level.name,
            trust_score=trust_score,
            access_decision=access_decision.value,
            detailed_scores=scores,
            ip_address=context.ip_address,
            device_id=context.device_id
        )
    
    async def _update_behavioral_baseline(self, context: SecurityContext, scores: Dict[str, float]):
        """Update user behavioral baseline"""
        if context.user_id not in self.user_baselines:
            self.user_baselines[context.user_id] = {
                'typical_hours': [],
                'typical_countries': set(),
                'typical_resources': set(),
                'last_login': None,
                'login_count': 0
            }
        
        baseline = self.user_baselines[context.user_id]
        
        # Update typical hours
        current_hour = context.timestamp.hour
        if current_hour not in baseline['typical_hours']:
            baseline['typical_hours'].append(current_hour)
            # Keep only recent typical hours (last 50 logins worth)
            if len(baseline['typical_hours']) > 50:
                baseline['typical_hours'] = baseline['typical_hours'][-50:]
        
        # Update location info
        if context.location and context.location.get('country'):
            baseline['typical_countries'].add(context.location['country'])
        
        # Update login tracking
        baseline['last_login'] = context.timestamp
        baseline['login_count'] += 1
    
    async def add_threat_indicator(self, indicator: ThreatIndicator):
        """Add new threat indicator"""
        self.threat_indicators[indicator.indicator_id] = indicator
        logger.warning(
            "Threat indicator added",
            indicator_id=indicator.indicator_id,
            indicator_type=indicator.indicator_type,
            value=indicator.value,
            severity=indicator.severity.name
        )
    
    async def add_policy_rule(self, rule: PolicyRule):
        """Add new policy rule"""
        self.policies[rule.rule_id] = rule
        logger.info(
            "Policy rule added",
            rule_id=rule.rule_id,
            name=rule.name,
            priority=rule.priority
        )
    
    async def start_session_monitoring(self, context: SecurityContext):
        """Start monitoring session"""
        self.active_sessions[context.session_id] = context
        self.session_cache[context.session_id] = {
            'verified_recently': True,
            'created_at': context.timestamp
        }
    
    async def end_session_monitoring(self, session_id: str):
        """End session monitoring"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.session_cache:
            del self.session_cache[session_id]
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring"""
        total_sessions = len(self.active_sessions)
        trusted_sessions = sum(
            1 for sess in self.active_sessions.values()
            if sess.device_trusted and sess.network_trusted
        )
        
        return {
            'active_sessions': total_sessions,
            'trusted_sessions': trusted_sessions,
            'trust_ratio': trusted_sessions / total_sessions if total_sessions > 0 else 0,
            'threat_indicators': len(self.threat_indicators),
            'policy_rules': len(self.policies),
            'user_baselines': len(self.user_baselines)
        }

# Global instance
_zero_trust_engine = None

def get_zero_trust_engine() -> ZeroTrustEngine:
    """Get zero trust engine instance"""
    global _zero_trust_engine
    if _zero_trust_engine is None:
        _zero_trust_engine = ZeroTrustEngine()
    return _zero_trust_engine

async def evaluate_request_trust(
    user_id: str,
    session_id: str,
    device_id: str,
    ip_address: str,
    user_agent: str,
    **kwargs
) -> Tuple[TrustLevel, float, AccessDecision]:
    """
    Convenience function to evaluate request trust
    
    Returns:
        Tuple of (trust_level, confidence_score, access_decision)
    """
    engine = get_zero_trust_engine()
    
    context = SecurityContext(
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        ip_address=ip_address,
        user_agent=user_agent,
        timestamp=datetime.utcnow(),
        **kwargs
    )
    
    return await engine.evaluate_trust(context)