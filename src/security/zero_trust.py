"""
Zero Trust Security Architecture for XORB
Implements "never trust, always verify" security model
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import ipaddress
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in Zero Trust model"""
    DENIED = 0
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    VERIFIED = 100


class RiskScore(Enum):
    """Risk assessment scores"""
    CRITICAL = 90
    HIGH = 70
    MEDIUM = 50
    LOW = 30
    MINIMAL = 10


@dataclass
class SecurityContext:
    """Complete security context for access decisions"""
    user_id: str
    session_id: str
    device_id: str
    ip_address: str
    location: Dict[str, Any]
    user_agent: str
    authentication_method: str
    mfa_verified: bool
    device_trusted: bool
    network_trusted: bool
    risk_score: int
    trust_level: TrustLevel
    last_verified: datetime
    created_at: datetime


@dataclass
class AccessRequest:
    """Access request for Zero Trust evaluation"""
    request_id: str
    user_id: str
    resource: str
    action: str
    context: SecurityContext
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class PolicyRule:
    """Zero Trust policy rule"""
    rule_id: str
    name: str
    description: str
    resource_pattern: str
    action_pattern: str
    conditions: List[Dict[str, Any]]
    required_trust_level: TrustLevel
    risk_threshold: int
    enabled: bool = True


@dataclass
class AccessDecision:
    """Access control decision"""
    request_id: str
    decision: str  # allow, deny, challenge
    trust_level: TrustLevel
    risk_score: int
    reasons: List[str]
    conditions: List[str]  # Additional conditions for access
    expires_at: Optional[datetime]
    requires_step_up: bool = False


class ZeroTrustEngine:
    """Core Zero Trust security engine"""

    def __init__(self):
        self.policies = self._load_default_policies()
        self.trust_cache = {}
        self.risk_engine = RiskAssessmentEngine()
        self.device_manager = DeviceTrustManager()
        self.network_analyzer = NetworkTrustAnalyzer()

    def _load_default_policies(self) -> List[PolicyRule]:
        """Load default Zero Trust policies"""
        return [
            # High-privilege operations require verified trust
            PolicyRule(
                rule_id="admin_operations",
                name="Administrative Operations",
                description="High-privilege admin operations require highest trust",
                resource_pattern="admin/*",
                action_pattern="create|update|delete",
                conditions=[
                    {"type": "mfa_verified", "value": True},
                    {"type": "device_trusted", "value": True},
                    {"type": "session_age_max", "value": 3600}  # 1 hour
                ],
                required_trust_level=TrustLevel.VERIFIED,
                risk_threshold=30
            ),

            # Sensitive data access requires high trust
            PolicyRule(
                rule_id="sensitive_data",
                name="Sensitive Data Access",
                description="Access to sensitive data requires high trust level",
                resource_pattern="data/sensitive/*",
                action_pattern="read|export",
                conditions=[
                    {"type": "mfa_verified", "value": True},
                    {"type": "location_verified", "value": True}
                ],
                required_trust_level=TrustLevel.HIGH,
                risk_threshold=50
            ),

            # External API access requires medium trust
            PolicyRule(
                rule_id="external_api",
                name="External API Access",
                description="External API calls require medium trust",
                resource_pattern="api/external/*",
                action_pattern="*",
                conditions=[
                    {"type": "device_known", "value": True},
                    {"type": "network_trusted", "value": True}
                ],
                required_trust_level=TrustLevel.MEDIUM,
                risk_threshold=60
            ),

            # Financial operations require verified trust
            PolicyRule(
                rule_id="financial_operations",
                name="Financial Operations",
                description="Financial transactions require verified trust",
                resource_pattern="finance/*",
                action_pattern="transfer|payment|withdrawal",
                conditions=[
                    {"type": "mfa_verified", "value": True},
                    {"type": "device_trusted", "value": True},
                    {"type": "time_restriction", "value": "business_hours"}
                ],
                required_trust_level=TrustLevel.VERIFIED,
                risk_threshold=20
            )
        ]

    async def evaluate_access(self, request: AccessRequest) -> AccessDecision:
        """Evaluate access request using Zero Trust principles"""
        logger.info(f"Evaluating Zero Trust access for request: {request.request_id}")

        # Find applicable policies
        applicable_policies = self._find_applicable_policies(request)

        if not applicable_policies:
            # Default deny - no applicable policy
            return AccessDecision(
                request_id=request.request_id,
                decision="deny",
                trust_level=TrustLevel.DENIED,
                risk_score=100,
                reasons=["No applicable policy found"],
                conditions=[],
                expires_at=None
            )

        # Evaluate each policy
        policy_results = []
        for policy in applicable_policies:
            result = await self._evaluate_policy(request, policy)
            policy_results.append(result)

        # Combine results (most restrictive wins)
        final_decision = self._combine_policy_results(request, policy_results)

        # Log decision for audit
        await self._log_access_decision(request, final_decision)

        return final_decision

    def _find_applicable_policies(self, request: AccessRequest) -> List[PolicyRule]:
        """Find policies applicable to the access request"""
        applicable = []

        for policy in self.policies:
            if not policy.enabled:
                continue

            # Check resource pattern
            if self._matches_pattern(request.resource, policy.resource_pattern):
                # Check action pattern
                if self._matches_pattern(request.action, policy.action_pattern):
                    applicable.append(policy)

        return applicable

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards)"""
        if pattern == "*":
            return True

        if "*" in pattern:
            import re
            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(f"^{regex_pattern}$", value))

        return value == pattern

    async def _evaluate_policy(self, request: AccessRequest, policy: PolicyRule) -> Dict[str, Any]:
        """Evaluate a single policy against the request"""

        # Check trust level requirement
        trust_sufficient = request.context.trust_level.value >= policy.required_trust_level.value

        # Check risk threshold
        risk_acceptable = request.context.risk_score <= policy.risk_threshold

        # Evaluate conditions
        conditions_met = await self._evaluate_conditions(request, policy.conditions)

        # Determine if policy allows access
        policy_allows = trust_sufficient and risk_acceptable and conditions_met

        return {
            "policy": policy,
            "allows": policy_allows,
            "trust_sufficient": trust_sufficient,
            "risk_acceptable": risk_acceptable,
            "conditions_met": conditions_met,
            "reasons": self._get_policy_failure_reasons(
                trust_sufficient, risk_acceptable, conditions_met, policy
            )
        }

    async def _evaluate_conditions(self, request: AccessRequest, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate policy conditions"""
        for condition in conditions:
            condition_type = condition["type"]
            expected_value = condition["value"]

            if condition_type == "mfa_verified":
                if request.context.mfa_verified != expected_value:
                    return False

            elif condition_type == "device_trusted":
                if request.context.device_trusted != expected_value:
                    return False

            elif condition_type == "session_age_max":
                session_age = (datetime.utcnow() - request.context.created_at).seconds
                if session_age > expected_value:
                    return False

            elif condition_type == "location_verified":
                location_verified = await self._verify_location(request.context)
                if location_verified != expected_value:
                    return False

            elif condition_type == "time_restriction":
                if not await self._check_time_restriction(expected_value):
                    return False

        return True

    async def _verify_location(self, context: SecurityContext) -> bool:
        """Verify if user location is trusted"""
        # Check against known good locations
        user_location = context.location

        # For demo, assume location is verified if country is known
        return user_location.get("country") in ["US", "CA", "GB", "DE", "AU"]

    async def _check_time_restriction(self, restriction: str) -> bool:
        """Check time-based restrictions"""
        if restriction == "business_hours":
            current_hour = datetime.utcnow().hour
            return 8 <= current_hour <= 18  # 8 AM to 6 PM UTC

        return True

    def _get_policy_failure_reasons(
        self,
        trust_sufficient: bool,
        risk_acceptable: bool,
        conditions_met: bool,
        policy: PolicyRule
    ) -> List[str]:
        """Get reasons why policy evaluation failed"""
        reasons = []

        if not trust_sufficient:
            reasons.append(f"Insufficient trust level (required: {policy.required_trust_level.name})")

        if not risk_acceptable:
            reasons.append(f"Risk score too high (max: {policy.risk_threshold})")

        if not conditions_met:
            reasons.append("Policy conditions not met")

        return reasons

    def _combine_policy_results(
        self,
        request: AccessRequest,
        policy_results: List[Dict[str, Any]]
    ) -> AccessDecision:
        """Combine multiple policy evaluation results"""

        # Check if any policy explicitly allows
        allows_access = any(result["allows"] for result in policy_results)

        # Collect all reasons
        all_reasons = []
        for result in policy_results:
            all_reasons.extend(result["reasons"])

        # Determine if step-up authentication is needed
        requires_step_up = (
            request.context.trust_level.value < TrustLevel.HIGH.value and
            any(r["policy"].required_trust_level.value >= TrustLevel.HIGH.value
                for r in policy_results)
        )

        if allows_access:
            decision = "allow"
        elif requires_step_up:
            decision = "challenge"
        else:
            decision = "deny"

        # Set expiration for temporary access
        expires_at = None
        if decision == "allow":
            expires_at = datetime.utcnow() + timedelta(hours=1)

        return AccessDecision(
            request_id=request.request_id,
            decision=decision,
            trust_level=request.context.trust_level,
            risk_score=request.context.risk_score,
            reasons=all_reasons[:5],  # Limit to top 5 reasons
            conditions=[],
            expires_at=expires_at,
            requires_step_up=requires_step_up
        )

    async def _log_access_decision(self, request: AccessRequest, decision: AccessDecision):
        """Log access decision for audit trail"""
        audit_log = {
            "event_type": "zero_trust_decision",
            "request_id": request.request_id,
            "user_id": request.user_id,
            "resource": request.resource,
            "action": request.action,
            "decision": decision.decision,
            "trust_level": decision.trust_level.name,
            "risk_score": decision.risk_score,
            "reasons": decision.reasons,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": request.context.ip_address,
            "device_id": request.context.device_id
        }

        # In production, this would go to a secure audit log
        logger.info(f"Zero Trust Decision: {json.dumps(audit_log)}")

    async def continuous_verification(self, session_id: str) -> SecurityContext:
        """Continuously verify user session and update trust level"""

        # Get current context
        context = await self._get_session_context(session_id)
        if not context:
            raise ValueError(f"Session not found: {session_id}")

        # Perform continuous checks
        new_risk_score = await self.risk_engine.calculate_risk(context)
        device_still_trusted = await self.device_manager.verify_device(context.device_id)
        network_still_trusted = await self.network_analyzer.analyze_network(context.ip_address)

        # Update context
        context.risk_score = new_risk_score
        context.device_trusted = device_still_trusted
        context.network_trusted = network_still_trusted
        context.last_verified = datetime.utcnow()

        # Recalculate trust level
        context.trust_level = await self._calculate_trust_level(context)

        # Cache updated context
        self.trust_cache[session_id] = context

        return context

    async def _get_session_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context for session"""
        return self.trust_cache.get(session_id)

    async def _calculate_trust_level(self, context: SecurityContext) -> TrustLevel:
        """Calculate trust level based on security context"""
        base_score = 0

        # Authentication factors
        if context.mfa_verified:
            base_score += 30
        else:
            base_score += 10

        # Device trust
        if context.device_trusted:
            base_score += 25
        else:
            base_score += 5

        # Network trust
        if context.network_trusted:
            base_score += 20
        else:
            base_score += 5

        # Risk adjustment
        risk_adjustment = max(0, 50 - context.risk_score)
        base_score += risk_adjustment

        # Session freshness
        session_age = (datetime.utcnow() - context.created_at).seconds
        if session_age < 3600:  # Less than 1 hour
            base_score += 15
        elif session_age < 14400:  # Less than 4 hours
            base_score += 10
        else:
            base_score += 0

        # Map to trust level
        if base_score >= 90:
            return TrustLevel.VERIFIED
        elif base_score >= 70:
            return TrustLevel.HIGH
        elif base_score >= 50:
            return TrustLevel.MEDIUM
        elif base_score >= 25:
            return TrustLevel.LOW
        else:
            return TrustLevel.DENIED


class RiskAssessmentEngine:
    """Risk assessment for Zero Trust decisions"""

    async def calculate_risk(self, context: SecurityContext) -> int:
        """Calculate risk score based on multiple factors"""
        risk_score = 0

        # Location-based risk
        risk_score += await self._assess_location_risk(context.location)

        # Device-based risk
        risk_score += await self._assess_device_risk(context.device_id)

        # Behavioral risk
        risk_score += await self._assess_behavioral_risk(context.user_id)

        # Network risk
        risk_score += await self._assess_network_risk(context.ip_address)

        return min(risk_score, 100)  # Cap at 100

    async def _assess_location_risk(self, location: Dict[str, Any]) -> int:
        """Assess risk based on location"""
        country = location.get("country", "Unknown")

        # High-risk countries
        high_risk_countries = ["XX", "YY", "ZZ"]  # Placeholder
        if country in high_risk_countries:
            return 40

        # Unknown location
        if country == "Unknown":
            return 20

        return 0

    async def _assess_device_risk(self, device_id: str) -> int:
        """Assess risk based on device characteristics"""
        # Check device reputation
        device_info = await self._get_device_info(device_id)

        if device_info.get("jailbroken", False):
            return 30

        if device_info.get("unknown_device", True):
            return 15

        return 0

    async def _assess_behavioral_risk(self, user_id: str) -> int:
        """Assess risk based on user behavior"""
        # Analyze user behavior patterns
        behavior = await self._get_user_behavior(user_id)

        if behavior.get("unusual_activity", False):
            return 25

        if behavior.get("off_hours_access", False):
            return 10

        return 0

    async def _assess_network_risk(self, ip_address: str) -> int:
        """Assess risk based on network characteristics"""
        try:
            ip = ipaddress.ip_address(ip_address)

            # Check if it's a private IP
            if ip.is_private:
                return 0

            # Check threat intelligence
            threat_info = await self._check_threat_intelligence(ip_address)
            if threat_info.get("malicious", False):
                return 50

            # Check if it's a known VPN/proxy
            if threat_info.get("proxy", False):
                return 15

        except ValueError:
            return 20  # Invalid IP address

        return 5  # Default for external IPs

    async def _get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get device information"""
        # In production, this would query device management system
        return {"jailbroken": False, "unknown_device": False}

    async def _get_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """Get user behavior analysis"""
        # In production, this would analyze user behavior patterns
        return {"unusual_activity": False, "off_hours_access": False}

    async def _check_threat_intelligence(self, ip_address: str) -> Dict[str, Any]:
        """Check IP against threat intelligence feeds"""
        # In production, this would query threat intelligence APIs
        return {"malicious": False, "proxy": False}


class DeviceTrustManager:
    """Manages device trust and compliance"""

    async def verify_device(self, device_id: str) -> bool:
        """Verify device trust status"""
        device_info = await self._get_device_compliance(device_id)

        # Check compliance requirements
        return (
            device_info.get("encryption_enabled", False) and
            device_info.get("os_updated", False) and
            device_info.get("antivirus_active", False) and
            not device_info.get("jailbroken", True)
        )

    async def _get_device_compliance(self, device_id: str) -> Dict[str, Any]:
        """Get device compliance status"""
        # In production, this would integrate with MDM solutions
        return {
            "encryption_enabled": True,
            "os_updated": True,
            "antivirus_active": True,
            "jailbroken": False
        }


class NetworkTrustAnalyzer:
    """Analyzes network trust and security"""

    async def analyze_network(self, ip_address: str) -> bool:
        """Analyze network trust level"""
        try:
            ip = ipaddress.ip_address(ip_address)

            # Corporate networks are trusted
            if ip.is_private:
                return True

            # Check against trusted external networks
            trusted_networks = await self._get_trusted_networks()
            for network in trusted_networks:
                if ip in ipaddress.ip_network(network):
                    return True

            # Check threat intelligence
            threat_info = await self._check_network_threats(ip_address)
            return not threat_info.get("malicious", False)

        except ValueError:
            return False

    async def _get_trusted_networks(self) -> List[str]:
        """Get list of trusted network ranges"""
        return [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16"
        ]

    async def _check_network_threats(self, ip_address: str) -> Dict[str, Any]:
        """Check network against threat feeds"""
        # In production, integrate with threat intelligence
        return {"malicious": False}


# Global Zero Trust engine
zero_trust_engine = ZeroTrustEngine()


async def evaluate_zero_trust_access(
    user_id: str,
    resource: str,
    action: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """API endpoint for Zero Trust access evaluation"""

    # Create security context
    security_context = SecurityContext(
        user_id=user_id,
        session_id=context.get("session_id", str(uuid.uuid4())),
        device_id=context.get("device_id", "unknown"),
        ip_address=context.get("ip_address", "127.0.0.1"),
        location=context.get("location", {}),
        user_agent=context.get("user_agent", ""),
        authentication_method=context.get("auth_method", "password"),
        mfa_verified=context.get("mfa_verified", False),
        device_trusted=context.get("device_trusted", False),
        network_trusted=context.get("network_trusted", False),
        risk_score=context.get("risk_score", 50),
        trust_level=TrustLevel(context.get("trust_level", 25)),
        last_verified=datetime.utcnow(),
        created_at=datetime.utcnow()
    )

    # Create access request
    request = AccessRequest(
        request_id=str(uuid.uuid4()),
        user_id=user_id,
        resource=resource,
        action=action,
        context=security_context,
        timestamp=datetime.utcnow(),
        metadata=context.get("metadata", {})
    )

    # Evaluate access
    decision = await zero_trust_engine.evaluate_access(request)

    return asdict(decision)


async def continuous_trust_verification(session_id: str) -> Dict[str, Any]:
    """API endpoint for continuous trust verification"""
    context = await zero_trust_engine.continuous_verification(session_id)
    return asdict(context)
