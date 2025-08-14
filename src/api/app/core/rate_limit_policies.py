"""
Hierarchical rate limiting policy system with tenant/role/endpoint overrides.

This module provides enterprise policy management with:
- Global defaults → tenant overrides → role overrides → endpoint overrides
- Hard cap guardrails that cannot be exceeded
- Dynamic policy loading and hot reloading
- Integration with RBAC and tenant context
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID
import structlog

from .adaptive_rate_limiter import (
    RateLimitPolicy, PolicyScope, LimitAlgorithm, ReputationLevel
)

logger = structlog.get_logger("rate_limit_policies")


class PolicyType(Enum):
    """Policy hierarchy types"""
    GLOBAL_DEFAULT = "global_default"
    TENANT_OVERRIDE = "tenant_override"
    ROLE_OVERRIDE = "role_override"
    USER_OVERRIDE = "user_override"
    ENDPOINT_OVERRIDE = "endpoint_override"
    EMERGENCY_OVERRIDE = "emergency_override"


@dataclass
class PolicyOverride:
    """Policy override configuration"""
    policy_type: PolicyType
    scope: PolicyScope
    identifier: str  # tenant_id, role_name, endpoint_pattern, etc.
    
    # Override values (None means inherit from parent)
    requests_per_second: Optional[float] = None
    burst_size: Optional[int] = None
    window_seconds: Optional[int] = None
    algorithm: Optional[LimitAlgorithm] = None
    enabled: Optional[bool] = None
    
    # Hard caps (cannot be exceeded by any override)
    max_requests_per_second: Optional[float] = None
    max_burst_size: Optional[int] = None
    
    # Conditions for applying this override
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Priority within same type (lower = higher priority)
    priority: int = 100
    
    # Metadata
    created_by: Optional[str] = None
    description: Optional[str] = None
    valid_until: Optional[float] = None  # Unix timestamp


@dataclass
class RateLimitContext:
    """Context for rate limit policy resolution"""
    scope: PolicyScope
    endpoint: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    role_names: Set[str] = field(default_factory=set)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Request characteristics
    is_authenticated: bool = False
    is_admin: bool = False
    is_service_account: bool = False
    
    # Time-based context
    request_time: Optional[float] = None
    business_hours: bool = True
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


class HierarchicalPolicyManager:
    """
    Manages hierarchical rate limiting policies with inheritance and overrides.
    
    Policy Resolution Order:
    1. Emergency overrides (highest priority)
    2. User-specific overrides
    3. Role-based overrides (most specific role wins)
    4. Endpoint-specific overrides
    5. Tenant-specific overrides
    6. Global defaults (lowest priority)
    
    Hard caps are enforced at all levels and cannot be exceeded.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
        # Policy storage
        self.global_policies: Dict[PolicyScope, RateLimitPolicy] = {}
        self.tenant_overrides: Dict[str, List[PolicyOverride]] = {}
        self.role_overrides: Dict[str, List[PolicyOverride]] = {}
        self.user_overrides: Dict[str, List[PolicyOverride]] = {}
        self.endpoint_overrides: List[PolicyOverride] = []
        self.emergency_overrides: List[PolicyOverride] = []
        
        # Hard caps (cannot be exceeded by any override)
        self.hard_caps = self._get_default_hard_caps()
        
        # Cache for resolved policies
        self.policy_cache: Dict[str, RateLimitPolicy] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Load default policies
        self._load_default_policies()
        
        logger.info("Hierarchical policy manager initialized")
    
    def _get_default_hard_caps(self) -> Dict[PolicyScope, Dict[str, float]]:
        """Get default hard caps that cannot be exceeded"""
        return {
            PolicyScope.GLOBAL: {
                "max_requests_per_second": 10000.0,
                "max_burst_size": 50000
            },
            PolicyScope.IP: {
                "max_requests_per_second": 100.0,
                "max_burst_size": 500
            },
            PolicyScope.USER: {
                "max_requests_per_second": 1000.0,
                "max_burst_size": 5000
            },
            PolicyScope.TENANT: {
                "max_requests_per_second": 5000.0,
                "max_burst_size": 25000
            },
            PolicyScope.ENDPOINT: {
                "max_requests_per_second": 500.0,
                "max_burst_size": 2500
            }
        }
    
    def _load_default_policies(self):
        """Load default rate limiting policies"""
        
        # Global IP-based rate limiting (strictest for anonymous users)
        self.global_policies[PolicyScope.IP] = RateLimitPolicy(
            scope=PolicyScope.IP,
            algorithm=LimitAlgorithm.SLIDING_WINDOW,
            requests_per_second=10.0,  # 10 req/sec
            burst_size=50,             # Allow burst of 50
            window_seconds=60,
            priority=100,
            circuit_breaker_enabled=True,
            failure_threshold=100,
            failure_window_seconds=60,
            recovery_timeout_seconds=300
        )
        
        # User-based rate limiting (more generous for authenticated users)
        self.global_policies[PolicyScope.USER] = RateLimitPolicy(
            scope=PolicyScope.USER,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=50.0,   # 50 req/sec
            burst_size=200,             # Burst of 200
            window_seconds=60,
            priority=90,
            circuit_breaker_enabled=True,
            failure_threshold=200,
            failure_window_seconds=60,
            recovery_timeout_seconds=180
        )
        
        # Tenant-based rate limiting (highest for multi-tenant)
        self.global_policies[PolicyScope.TENANT] = RateLimitPolicy(
            scope=PolicyScope.TENANT,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=200.0,  # 200 req/sec per tenant
            burst_size=1000,            # Large burst allowance
            window_seconds=60,
            priority=80,
            circuit_breaker_enabled=True,
            failure_threshold=500,
            failure_window_seconds=120,
            recovery_timeout_seconds=300
        )
        
        # Endpoint-specific policies for sensitive operations
        auth_endpoints_policy = RateLimitPolicy(
            scope=PolicyScope.ENDPOINT,
            algorithm=LimitAlgorithm.SLIDING_WINDOW,
            requests_per_second=2.0,    # Very strict for auth
            burst_size=5,
            window_seconds=60,
            priority=70,
            circuit_breaker_enabled=True,
            failure_threshold=20,
            failure_window_seconds=300,
            recovery_timeout_seconds=600
        )
        auth_endpoints_policy.decision_metadata = {
            'endpoint_patterns': ['/api/v1/auth/', '/api/v1/login', '/api/v1/token']
        }
        
        ptaas_endpoints_policy = RateLimitPolicy(
            scope=PolicyScope.ENDPOINT,
            algorithm=LimitAlgorithm.TOKEN_BUCKET,
            requests_per_second=5.0,    # Moderate for PTaaS operations
            burst_size=20,
            window_seconds=60,
            priority=75,
            circuit_breaker_enabled=True,
            failure_threshold=50,
            failure_window_seconds=120,
            recovery_timeout_seconds=300
        )
        ptaas_endpoints_policy.decision_metadata = {
            'endpoint_patterns': ['/api/v1/ptaas/', '/api/v1/scans/']
        }
        
        # Store endpoint policies in overrides list
        self.endpoint_overrides.extend([
            PolicyOverride(
                policy_type=PolicyType.ENDPOINT_OVERRIDE,
                scope=PolicyScope.ENDPOINT,
                identifier="auth_endpoints",
                requests_per_second=auth_endpoints_policy.requests_per_second,
                burst_size=auth_endpoints_policy.burst_size,
                window_seconds=auth_endpoints_policy.window_seconds,
                algorithm=auth_endpoints_policy.algorithm,
                conditions={'endpoint_patterns': ['/api/v1/auth/', '/api/v1/login', '/api/v1/token']},
                priority=70,
                description="Strict rate limiting for authentication endpoints"
            ),
            PolicyOverride(
                policy_type=PolicyType.ENDPOINT_OVERRIDE,
                scope=PolicyScope.ENDPOINT,
                identifier="ptaas_endpoints",
                requests_per_second=ptaas_endpoints_policy.requests_per_second,
                burst_size=ptaas_endpoints_policy.burst_size,
                window_seconds=ptaas_endpoints_policy.window_seconds,
                algorithm=ptaas_endpoints_policy.algorithm,
                conditions={'endpoint_patterns': ['/api/v1/ptaas/', '/api/v1/scans/']},
                priority=75,
                description="Moderate rate limiting for PTaaS endpoints"
            )
        ])
        
        logger.info("Default policies loaded", 
                   global_policies=len(self.global_policies),
                   endpoint_overrides=len(self.endpoint_overrides))
    
    def resolve_policy(self, context: RateLimitContext) -> Optional[RateLimitPolicy]:
        """
        Resolve the applicable rate limiting policy based on context.
        
        Returns the most specific policy after applying all applicable overrides.
        """
        # Check cache first
        cache_key = self._generate_cache_key(context)
        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]
        
        try:
            # Start with global default policy
            base_policy = self.global_policies.get(context.scope)
            if not base_policy:
                logger.warning(f"No global policy found for scope {context.scope.value}")
                return None
            
            # Create working copy
            resolved_policy = RateLimitPolicy(
                scope=base_policy.scope,
                algorithm=base_policy.algorithm,
                requests_per_second=base_policy.requests_per_second,
                burst_size=base_policy.burst_size,
                window_seconds=base_policy.window_seconds,
                cost_multiplier=base_policy.cost_multiplier,
                enabled=base_policy.enabled,
                priority=base_policy.priority,
                reputation_multiplier=base_policy.reputation_multiplier.copy(),
                backoff_base_seconds=base_policy.backoff_base_seconds,
                backoff_max_seconds=base_policy.backoff_max_seconds,
                backoff_multiplier=base_policy.backoff_multiplier,
                circuit_breaker_enabled=base_policy.circuit_breaker_enabled,
                failure_threshold=base_policy.failure_threshold,
                failure_window_seconds=base_policy.failure_window_seconds,
                recovery_timeout_seconds=base_policy.recovery_timeout_seconds
            )
            
            # Apply overrides in order of precedence
            overrides_applied = []
            
            # 1. Emergency overrides (highest priority)
            emergency_override = self._find_applicable_override(
                self.emergency_overrides, context
            )
            if emergency_override:
                self._apply_override(resolved_policy, emergency_override)
                overrides_applied.append(emergency_override.policy_type.value)
            
            # 2. User-specific overrides
            if context.user_id:
                user_overrides = self.user_overrides.get(context.user_id, [])
                user_override = self._find_applicable_override(user_overrides, context)
                if user_override:
                    self._apply_override(resolved_policy, user_override)
                    overrides_applied.append(user_override.policy_type.value)
            
            # 3. Role-based overrides (most permissive role wins)
            if context.role_names:
                best_role_override = None
                best_rps = 0
                
                for role_name in context.role_names:
                    role_overrides = self.role_overrides.get(role_name, [])
                    role_override = self._find_applicable_override(role_overrides, context)
                    if role_override and (role_override.requests_per_second or 0) > best_rps:
                        best_role_override = role_override
                        best_rps = role_override.requests_per_second or 0
                
                if best_role_override:
                    self._apply_override(resolved_policy, best_role_override)
                    overrides_applied.append(best_role_override.policy_type.value)
            
            # 4. Endpoint-specific overrides
            endpoint_override = self._find_applicable_override(
                self.endpoint_overrides, context
            )
            if endpoint_override:
                self._apply_override(resolved_policy, endpoint_override)
                overrides_applied.append(endpoint_override.policy_type.value)
            
            # 5. Tenant-specific overrides
            if context.tenant_id:
                tenant_overrides = self.tenant_overrides.get(str(context.tenant_id), [])
                tenant_override = self._find_applicable_override(tenant_overrides, context)
                if tenant_override:
                    self._apply_override(resolved_policy, tenant_override)
                    overrides_applied.append(tenant_override.policy_type.value)
            
            # Apply hard caps
            self._enforce_hard_caps(resolved_policy)
            
            # Add metadata about applied overrides
            resolved_policy.decision_metadata = {
                'base_policy': context.scope.value,
                'overrides_applied': overrides_applied,
                'cache_key': cache_key
            }
            
            # Cache the resolved policy
            self.policy_cache[cache_key] = resolved_policy
            
            logger.debug("Policy resolved",
                        scope=context.scope.value,
                        overrides_applied=overrides_applied,
                        final_rps=resolved_policy.requests_per_second,
                        final_burst=resolved_policy.burst_size)
            
            return resolved_policy
        
        except Exception as e:
            logger.error("Policy resolution failed", error=str(e), scope=context.scope.value)
            return base_policy  # Fallback to base policy
    
    def _find_applicable_override(
        self, 
        overrides: List[PolicyOverride], 
        context: RateLimitContext
    ) -> Optional[PolicyOverride]:
        """Find the most applicable override from a list"""
        applicable_overrides = []
        
        for override in overrides:
            if self._is_override_applicable(override, context):
                applicable_overrides.append(override)
        
        if not applicable_overrides:
            return None
        
        # Sort by priority (lower = higher priority)
        applicable_overrides.sort(key=lambda o: o.priority)
        return applicable_overrides[0]
    
    def _is_override_applicable(
        self, 
        override: PolicyOverride, 
        context: RateLimitContext
    ) -> bool:
        """Check if an override is applicable to the context"""
        # Check scope match
        if override.scope != context.scope:
            return False
        
        # Check if override is enabled
        if override.enabled is False:
            return False
        
        # Check validity period
        if override.valid_until and context.request_time:
            if context.request_time > override.valid_until:
                return False
        
        # Check conditions
        if override.conditions:
            if not self._check_conditions(override.conditions, context):
                return False
        
        return True
    
    def _check_conditions(
        self, 
        conditions: Dict[str, Any], 
        context: RateLimitContext
    ) -> bool:
        """Check if conditions are met for the context"""
        
        # Endpoint pattern matching
        if 'endpoint_patterns' in conditions and context.endpoint:
            patterns = conditions['endpoint_patterns']
            if not any(context.endpoint.startswith(pattern) for pattern in patterns):
                return False
        
        # Role requirements
        if 'required_roles' in conditions:
            required_roles = set(conditions['required_roles'])
            if not context.role_names.intersection(required_roles):
                return False
        
        # Authentication requirements
        if 'requires_auth' in conditions:
            if conditions['requires_auth'] and not context.is_authenticated:
                return False
        
        # Admin requirements
        if 'requires_admin' in conditions:
            if conditions['requires_admin'] and not context.is_admin:
                return False
        
        # Business hours check
        if 'business_hours_only' in conditions:
            if conditions['business_hours_only'] and not context.business_hours:
                return False
        
        # Custom attribute checks
        if 'custom_attributes' in conditions:
            for key, expected_value in conditions['custom_attributes'].items():
                if context.custom_attributes.get(key) != expected_value:
                    return False
        
        return True
    
    def _apply_override(self, policy: RateLimitPolicy, override: PolicyOverride):
        """Apply an override to a policy"""
        if override.requests_per_second is not None:
            policy.requests_per_second = override.requests_per_second
        
        if override.burst_size is not None:
            policy.burst_size = override.burst_size
        
        if override.window_seconds is not None:
            policy.window_seconds = override.window_seconds
        
        if override.algorithm is not None:
            policy.algorithm = override.algorithm
        
        if override.enabled is not None:
            policy.enabled = override.enabled
    
    def _enforce_hard_caps(self, policy: RateLimitPolicy):
        """Enforce hard caps that cannot be exceeded"""
        caps = self.hard_caps.get(policy.scope, {})
        
        max_rps = caps.get('max_requests_per_second')
        if max_rps and policy.requests_per_second > max_rps:
            logger.warning(
                "Policy exceeds hard cap for requests_per_second",
                scope=policy.scope.value,
                requested=policy.requests_per_second,
                capped_at=max_rps
            )
            policy.requests_per_second = max_rps
        
        max_burst = caps.get('max_burst_size')
        if max_burst and policy.burst_size > max_burst:
            logger.warning(
                "Policy exceeds hard cap for burst_size",
                scope=policy.scope.value,
                requested=policy.burst_size,
                capped_at=max_burst
            )
            policy.burst_size = max_burst
    
    def _generate_cache_key(self, context: RateLimitContext) -> str:
        """Generate cache key for resolved policy"""
        key_parts = [
            context.scope.value,
            context.endpoint or "none",
            context.user_id or "anonymous",
            str(context.tenant_id) if context.tenant_id else "no_tenant",
            "|".join(sorted(context.role_names)) if context.role_names else "no_roles",
            str(context.is_authenticated),
            str(context.is_admin),
            str(context.business_hours)
        ]
        return ":".join(key_parts)
    
    def add_tenant_override(
        self,
        tenant_id: str,
        scope: PolicyScope,
        override: PolicyOverride
    ):
        """Add a tenant-specific override"""
        if tenant_id not in self.tenant_overrides:
            self.tenant_overrides[tenant_id] = []
        
        override.policy_type = PolicyType.TENANT_OVERRIDE
        override.scope = scope
        override.identifier = tenant_id
        
        self.tenant_overrides[tenant_id].append(override)
        self._clear_cache()
        
        logger.info(
            "Tenant override added",
            tenant_id=tenant_id,
            scope=scope.value,
            requests_per_second=override.requests_per_second
        )
    
    def add_role_override(
        self,
        role_name: str,
        scope: PolicyScope,
        override: PolicyOverride
    ):
        """Add a role-specific override"""
        if role_name not in self.role_overrides:
            self.role_overrides[role_name] = []
        
        override.policy_type = PolicyType.ROLE_OVERRIDE
        override.scope = scope
        override.identifier = role_name
        
        self.role_overrides[role_name].append(override)
        self._clear_cache()
        
        logger.info(
            "Role override added",
            role_name=role_name,
            scope=scope.value,
            requests_per_second=override.requests_per_second
        )
    
    def add_emergency_override(
        self,
        scope: PolicyScope,
        override: PolicyOverride,
        duration_seconds: int = 3600
    ):
        """Add an emergency override (highest priority)"""
        import time
        
        override.policy_type = PolicyType.EMERGENCY_OVERRIDE
        override.scope = scope
        override.identifier = f"emergency_{int(time.time())}"
        override.valid_until = time.time() + duration_seconds
        override.priority = 1  # Highest priority
        
        self.emergency_overrides.append(override)
        self._clear_cache()
        
        logger.critical(
            "Emergency override added",
            scope=scope.value,
            duration_seconds=duration_seconds,
            requests_per_second=override.requests_per_second
        )
    
    def remove_expired_overrides(self):
        """Remove expired overrides (maintenance task)"""
        import time
        current_time = time.time()
        
        removed_count = 0
        
        # Remove expired emergency overrides
        self.emergency_overrides = [
            o for o in self.emergency_overrides
            if not o.valid_until or o.valid_until > current_time
        ]
        
        # Remove expired overrides from all collections
        for collection in [self.tenant_overrides, self.role_overrides, self.user_overrides]:
            for key, overrides in collection.items():
                original_count = len(overrides)
                collection[key] = [
                    o for o in overrides
                    if not o.valid_until or o.valid_until > current_time
                ]
                removed_count += original_count - len(collection[key])
        
        if removed_count > 0:
            self._clear_cache()
            logger.info("Expired overrides removed", count=removed_count)
    
    def _clear_cache(self):
        """Clear the policy cache"""
        self.policy_cache.clear()
        logger.debug("Policy cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy manager statistics"""
        return {
            "global_policies": len(self.global_policies),
            "tenant_overrides": {
                tenant: len(overrides) 
                for tenant, overrides in self.tenant_overrides.items()
            },
            "role_overrides": {
                role: len(overrides)
                for role, overrides in self.role_overrides.items()
            },
            "user_overrides": {
                user: len(overrides)
                for user, overrides in self.user_overrides.items()
            },
            "endpoint_overrides": len(self.endpoint_overrides),
            "emergency_overrides": len(self.emergency_overrides),
            "cache_size": len(self.policy_cache),
            "hard_caps": self.hard_caps
        }
    
    async def load_policies_from_redis(self):
        """Load policies from Redis (for persistent storage)"""
        if not self.redis:
            return
        
        try:
            # Load tenant overrides
            tenant_keys = await self.redis.keys("policy:tenant:*")
            for key in tenant_keys:
                tenant_id = key.split(":")[-1]
                data = await self.redis.get(key)
                if data:
                    overrides = json.loads(data)
                    self.tenant_overrides[tenant_id] = [
                        PolicyOverride(**override) for override in overrides
                    ]
            
            # Load role overrides
            role_keys = await self.redis.keys("policy:role:*")
            for key in role_keys:
                role_name = key.split(":")[-1]
                data = await self.redis.get(key)
                if data:
                    overrides = json.loads(data)
                    self.role_overrides[role_name] = [
                        PolicyOverride(**override) for override in overrides
                    ]
            
            self._clear_cache()
            logger.info("Policies loaded from Redis")
        
        except Exception as e:
            logger.error("Failed to load policies from Redis", error=str(e))
    
    async def save_policies_to_redis(self):
        """Save policies to Redis (for persistence)"""
        if not self.redis:
            return
        
        try:
            # Save tenant overrides
            for tenant_id, overrides in self.tenant_overrides.items():
                key = f"policy:tenant:{tenant_id}"
                data = json.dumps([asdict(override) for override in overrides])
                await self.redis.set(key, data)
            
            # Save role overrides
            for role_name, overrides in self.role_overrides.items():
                key = f"policy:role:{role_name}"
                data = json.dumps([asdict(override) for override in overrides])
                await self.redis.set(key, data)
            
            logger.info("Policies saved to Redis")
        
        except Exception as e:
            logger.error("Failed to save policies to Redis", error=str(e))


# Predefined role-based overrides for common scenarios
DEFAULT_ROLE_OVERRIDES = {
    "admin": PolicyOverride(
        policy_type=PolicyType.ROLE_OVERRIDE,
        scope=PolicyScope.USER,
        identifier="admin",
        requests_per_second=200.0,
        burst_size=1000,
        description="Higher limits for admin users"
    ),
    "premium_user": PolicyOverride(
        policy_type=PolicyType.ROLE_OVERRIDE,
        scope=PolicyScope.USER,
        identifier="premium_user",
        requests_per_second=100.0,
        burst_size=500,
        description="Higher limits for premium users"
    ),
    "api_service": PolicyOverride(
        policy_type=PolicyType.ROLE_OVERRIDE,
        scope=PolicyScope.USER,
        identifier="api_service",
        requests_per_second=500.0,
        burst_size=2500,
        description="High limits for service accounts"
    ),
    "readonly_user": PolicyOverride(
        policy_type=PolicyType.ROLE_OVERRIDE,
        scope=PolicyScope.USER,
        identifier="readonly_user",
        requests_per_second=20.0,
        burst_size=100,
        description="Conservative limits for read-only users"
    )
}