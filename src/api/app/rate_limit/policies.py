"""
Rate Limit Policy Definition and Resolution System
Production-ready policy management with hierarchical resolution and adaptive controls
"""

import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from uuid import UUID
import json
import hashlib

from ..core.logging import get_logger

logger = get_logger(__name__)


class RateLimitScope(Enum):
    """Rate limit scope types with priority order (highest to lowest)"""
    ENDPOINT = "endpoint"     # Highest priority - specific endpoint limits
    USER = "user"            # User-specific limits
    ROLE = "role"            # Role-based limits
    TENANT = "tenant"        # Tenant-wide limits
    IP = "ip"                # IP-based limits (security)
    GLOBAL = "global"        # Platform-wide limits (lowest priority)


class RateLimitMode(Enum):
    """Rate limiting enforcement modes"""
    SHADOW = "shadow"        # Log violations but don't enforce
    ENFORCE = "enforce"      # Actively block requests
    DISABLED = "disabled"    # No rate limiting


class BurstStrategy(Enum):
    """Burst handling strategies"""
    STRICT = "strict"        # No burst allowance
    ADAPTIVE = "adaptive"    # Dynamic burst based on usage patterns
    FIXED = "fixed"          # Fixed burst allowance


@dataclass
class RateLimitWindow:
    """Time window configuration for rate limits"""
    duration_seconds: int
    max_requests: int
    burst_allowance: int = 0
    burst_strategy: BurstStrategy = BurstStrategy.ADAPTIVE
    
    def __post_init__(self):
        """Validate window configuration"""
        if self.duration_seconds <= 0:
            raise ValueError("Window duration must be positive")
        if self.max_requests <= 0:
            raise ValueError("Max requests must be positive")
        if self.burst_allowance < 0:
            raise ValueError("Burst allowance cannot be negative")


@dataclass
class AdaptiveConfig:
    """Adaptive rate limiting configuration"""
    enable_reputation_scoring: bool = True
    violation_penalty_multiplier: float = 2.0
    reputation_decay_hours: int = 24
    min_reputation_score: float = 0.1
    max_reputation_score: float = 1.0
    adaptive_burst_factor: float = 1.5
    escalation_thresholds: Dict[int, float] = field(default_factory=lambda: {
        3: 0.5,    # 3 violations: 50% rate reduction
        5: 0.25,   # 5 violations: 75% rate reduction
        10: 0.1    # 10 violations: 90% rate reduction
    })


@dataclass
class RateLimitPolicy:
    """
    Comprehensive rate limit policy with hierarchical resolution support
    
    Features:
    - Multiple time windows (e.g., per-minute, per-hour, per-day)
    - Adaptive burst handling
    - Reputation-based adjustments
    - Emergency circuit breaker
    - Shadow vs enforce modes
    """
    
    # Policy identification
    name: str
    scope: RateLimitScope
    scope_values: Set[str] = field(default_factory=set)  # Specific IPs, roles, etc.
    
    # Time windows (multiple windows can be active)
    windows: List[RateLimitWindow] = field(default_factory=list)
    
    # Enforcement mode
    mode: RateLimitMode = RateLimitMode.ENFORCE
    
    # Adaptive configuration
    adaptive_config: Optional[AdaptiveConfig] = None
    
    # Priority for policy resolution
    priority: int = 0
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 1000  # Global emergency threshold
    circuit_breaker_window_seconds: int = 60
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate policy configuration"""
        if not self.windows:
            # Default window if none specified
            self.windows = [RateLimitWindow(duration_seconds=60, max_requests=60)]
        
        if not self.adaptive_config:
            self.adaptive_config = AdaptiveConfig()
    
    @property
    def policy_id(self) -> str:
        """Generate unique policy ID based on configuration"""
        key_data = f"{self.name}:{self.scope.value}:{sorted(self.scope_values)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get_effective_limits(self, reputation_score: float = 1.0) -> List[RateLimitWindow]:
        """
        Get effective rate limits adjusted for reputation
        
        Args:
            reputation_score: Current reputation score (0.0 to 1.0)
            
        Returns:
            List of adjusted rate limit windows
        """
        if not self.adaptive_config or not self.adaptive_config.enable_reputation_scoring:
            return self.windows
        
        adjusted_windows = []
        for window in self.windows:
            # Apply reputation-based adjustment
            adjusted_max = int(window.max_requests * reputation_score)
            adjusted_burst = int(window.burst_allowance * reputation_score)
            
            # Ensure minimum viable limits
            adjusted_max = max(1, adjusted_max)
            adjusted_burst = max(0, adjusted_burst)
            
            adjusted_window = RateLimitWindow(
                duration_seconds=window.duration_seconds,
                max_requests=adjusted_max,
                burst_allowance=adjusted_burst,
                burst_strategy=window.burst_strategy
            )
            adjusted_windows.append(adjusted_window)
        
        return adjusted_windows
    
    def should_escalate(self, violation_count: int) -> bool:
        """Check if escalation should be applied based on violations"""
        if not self.adaptive_config:
            return False
        
        return violation_count >= min(self.adaptive_config.escalation_thresholds.keys())
    
    def get_escalation_factor(self, violation_count: int) -> float:
        """Get rate reduction factor based on violation count"""
        if not self.adaptive_config:
            return 1.0
        
        # Find the highest applicable threshold
        factor = 1.0
        for threshold, reduction in sorted(self.adaptive_config.escalation_thresholds.items()):
            if violation_count >= threshold:
                factor = reduction
        
        return factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy to dictionary"""
        return {
            "name": self.name,
            "scope": self.scope.value,
            "scope_values": list(self.scope_values),
            "windows": [
                {
                    "duration_seconds": w.duration_seconds,
                    "max_requests": w.max_requests,
                    "burst_allowance": w.burst_allowance,
                    "burst_strategy": w.burst_strategy.value
                }
                for w in self.windows
            ],
            "mode": self.mode.value,
            "priority": self.priority,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_window_seconds": self.circuit_breaker_window_seconds,
            "adaptive_config": {
                "enable_reputation_scoring": self.adaptive_config.enable_reputation_scoring,
                "violation_penalty_multiplier": self.adaptive_config.violation_penalty_multiplier,
                "reputation_decay_hours": self.adaptive_config.reputation_decay_hours,
                "escalation_thresholds": self.adaptive_config.escalation_thresholds
            } if self.adaptive_config else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RateLimitPolicy':
        """Deserialize policy from dictionary"""
        windows = [
            RateLimitWindow(
                duration_seconds=w["duration_seconds"],
                max_requests=w["max_requests"],
                burst_allowance=w.get("burst_allowance", 0),
                burst_strategy=BurstStrategy(w.get("burst_strategy", "adaptive"))
            )
            for w in data["windows"]
        ]
        
        adaptive_config = None
        if data.get("adaptive_config"):
            ac = data["adaptive_config"]
            adaptive_config = AdaptiveConfig(
                enable_reputation_scoring=ac.get("enable_reputation_scoring", True),
                violation_penalty_multiplier=ac.get("violation_penalty_multiplier", 2.0),
                reputation_decay_hours=ac.get("reputation_decay_hours", 24),
                escalation_thresholds=ac.get("escalation_thresholds", {})
            )
        
        return cls(
            name=data["name"],
            scope=RateLimitScope(data["scope"]),
            scope_values=set(data.get("scope_values", [])),
            windows=windows,
            mode=RateLimitMode(data.get("mode", "enforce")),
            adaptive_config=adaptive_config,
            priority=data.get("priority", 0),
            circuit_breaker_threshold=data.get("circuit_breaker_threshold", 1000),
            circuit_breaker_window_seconds=data.get("circuit_breaker_window_seconds", 60),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            created_by=data.get("created_by", "system"),
            description=data.get("description", ""),
            tags=set(data.get("tags", []))
        )


class PolicyResolver:
    """
    Hierarchical policy resolution with caching and conflict handling
    
    Resolution order:
    1. Endpoint-specific policies (highest priority)
    2. User-specific policies
    3. Role-based policies
    4. Tenant-wide policies
    5. IP-based policies
    6. Global policies (lowest priority)
    """
    
    def __init__(self):
        self.policies: Dict[str, RateLimitPolicy] = {}
        self._cache: Dict[str, RateLimitPolicy] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}
    
    def add_policy(self, policy: RateLimitPolicy) -> None:
        """Add or update a rate limit policy"""
        policy.updated_at = datetime.utcnow()
        self.policies[policy.policy_id] = policy
        self._invalidate_cache()
        
        logger.info(f"Added rate limit policy: {policy.name} (scope: {policy.scope.value})")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a rate limit policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self._invalidate_cache()
            logger.info(f"Removed rate limit policy: {policy_id}")
            return True
        return False
    
    def get_policy(self, policy_id: str) -> Optional[RateLimitPolicy]:
        """Get a specific policy by ID"""
        return self.policies.get(policy_id)
    
    def list_policies(self, scope: Optional[RateLimitScope] = None) -> List[RateLimitPolicy]:
        """List all policies, optionally filtered by scope"""
        policies = list(self.policies.values())
        if scope:
            policies = [p for p in policies if p.scope == scope]
        return sorted(policies, key=lambda p: (p.scope.value, p.priority), reverse=True)
    
    def resolve_policy(
        self,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        roles: Optional[Set[str]] = None,
        endpoint: Optional[str] = None
    ) -> Optional[RateLimitPolicy]:
        """
        Resolve the effective rate limit policy for given context
        
        Args:
            ip_address: Client IP address
            user_id: User identifier
            tenant_id: Tenant identifier
            roles: Set of user roles
            endpoint: API endpoint path
            
        Returns:
            Most specific applicable policy or None
        """
        # Create cache key
        cache_key = self._create_cache_key(ip_address, user_id, tenant_id, roles, endpoint)
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        
        # Resolve policy hierarchy
        resolved_policy = self._resolve_hierarchy(ip_address, user_id, tenant_id, roles, endpoint)
        
        # Cache result
        if resolved_policy:
            self._cache[cache_key] = resolved_policy
            self._cache_timestamps[cache_key] = time.time()
        
        return resolved_policy
    
    def _resolve_hierarchy(
        self,
        ip_address: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[Set[str]],
        endpoint: Optional[str]
    ) -> Optional[RateLimitPolicy]:
        """Resolve policy using hierarchical priority"""
        
        # Priority order: endpoint > user > role > tenant > ip > global
        contexts = [
            (RateLimitScope.ENDPOINT, {endpoint} if endpoint else set()),
            (RateLimitScope.USER, {user_id} if user_id else set()),
            (RateLimitScope.ROLE, roles or set()),
            (RateLimitScope.TENANT, {tenant_id} if tenant_id else set()),
            (RateLimitScope.IP, {ip_address} if ip_address else set()),
            (RateLimitScope.GLOBAL, {"*"})
        ]
        
        for scope, values in contexts:
            matching_policies = []
            
            for policy in self.policies.values():
                if policy.scope != scope or policy.mode == RateLimitMode.DISABLED:
                    continue
                
                # Check if policy applies to current context
                if self._policy_matches(policy, values):
                    matching_policies.append(policy)
            
            if matching_policies:
                # Return highest priority policy for this scope
                return max(matching_policies, key=lambda p: p.priority)
        
        return None
    
    def _policy_matches(self, policy: RateLimitPolicy, values: Set[str]) -> bool:
        """Check if policy matches given values"""
        if not policy.scope_values:
            # Empty scope_values means applies to all in scope
            return True
        
        # Check if any value matches policy scope
        return bool(policy.scope_values.intersection(values))
    
    def _create_cache_key(
        self,
        ip_address: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[Set[str]],
        endpoint: Optional[str]
    ) -> str:
        """Create cache key for policy resolution"""
        key_parts = [
            f"ip:{ip_address or 'none'}",
            f"user:{user_id or 'none'}",
            f"tenant:{tenant_id or 'none'}",
            f"roles:{','.join(sorted(roles)) if roles else 'none'}",
            f"endpoint:{endpoint or 'none'}"
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached policy is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        
        return (time.time() - self._cache_timestamps[cache_key]) < self._cache_ttl
    
    def _invalidate_cache(self) -> None:
        """Invalidate policy cache"""
        self._cache.clear()
        self._cache_timestamps.clear()


# Default policy configurations
def create_default_policies() -> List[RateLimitPolicy]:
    """Create default rate limit policies for common scenarios"""
    
    policies = []
    
    # Global default policy - conservative limits
    policies.append(RateLimitPolicy(
        name="global_default",
        scope=RateLimitScope.GLOBAL,
        scope_values={"*"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=100, burst_allowance=20),
            RateLimitWindow(duration_seconds=3600, max_requests=1000, burst_allowance=100),
            RateLimitWindow(duration_seconds=86400, max_requests=10000, burst_allowance=500)
        ],
        mode=RateLimitMode.ENFORCE,
        priority=1,
        description="Default global rate limits"
    ))
    
    # Authentication endpoints - stricter limits
    policies.append(RateLimitPolicy(
        name="auth_endpoints",
        scope=RateLimitScope.ENDPOINT,
        scope_values={"/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/reset"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=5, burst_allowance=2),
            RateLimitWindow(duration_seconds=3600, max_requests=20, burst_allowance=5)
        ],
        mode=RateLimitMode.ENFORCE,
        priority=100,
        adaptive_config=AdaptiveConfig(
            violation_penalty_multiplier=3.0,
            escalation_thresholds={2: 0.5, 3: 0.1, 5: 0.05}
        ),
        description="Strict limits for authentication endpoints"
    ))
    
    # Admin endpoints - very strict
    policies.append(RateLimitPolicy(
        name="admin_endpoints",
        scope=RateLimitScope.ENDPOINT,
        scope_values={"/api/v1/admin", "/api/v1/users/manage", "/api/v1/tenants/manage"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=10, burst_allowance=2),
            RateLimitWindow(duration_seconds=3600, max_requests=50, burst_allowance=10)
        ],
        mode=RateLimitMode.ENFORCE,
        priority=90,
        description="Strict limits for administrative endpoints"
    ))
    
    # PTaaS scan endpoints - moderate limits due to resource intensity
    policies.append(RateLimitPolicy(
        name="ptaas_scan_endpoints",
        scope=RateLimitScope.ENDPOINT,
        scope_values={"/api/v1/ptaas/sessions", "/api/v1/ptaas/scans"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=5, burst_allowance=2),
            RateLimitWindow(duration_seconds=3600, max_requests=20, burst_allowance=5),
            RateLimitWindow(duration_seconds=86400, max_requests=100, burst_allowance=20)
        ],
        mode=RateLimitMode.ENFORCE,
        priority=80,
        description="Resource-aware limits for PTaaS scan operations"
    ))
    
    # High-privilege roles - relaxed limits
    policies.append(RateLimitPolicy(
        name="admin_role_limits",
        scope=RateLimitScope.ROLE,
        scope_values={"super_admin", "tenant_admin"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=200, burst_allowance=50),
            RateLimitWindow(duration_seconds=3600, max_requests=2000, burst_allowance=200)
        ],
        mode=RateLimitMode.ENFORCE,
        priority=70,
        description="Relaxed limits for administrative roles"
    ))
    
    # IP-based emergency limits (for suspicious IPs)
    policies.append(RateLimitPolicy(
        name="emergency_ip_limits",
        scope=RateLimitScope.IP,
        scope_values=set(),  # To be populated dynamically
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=5, burst_allowance=0),
            RateLimitWindow(duration_seconds=3600, max_requests=10, burst_allowance=0)
        ],
        mode=RateLimitMode.SHADOW,  # Start in shadow mode
        priority=95,
        description="Emergency limits for suspicious IP addresses"
    ))
    
    return policies


def create_testing_policies() -> List[RateLimitPolicy]:
    """Create policies for testing environments with higher limits"""
    
    policies = []
    
    # Testing global policy - higher limits
    policies.append(RateLimitPolicy(
        name="testing_global",
        scope=RateLimitScope.GLOBAL,
        scope_values={"*"},
        windows=[
            RateLimitWindow(duration_seconds=60, max_requests=1000, burst_allowance=200),
            RateLimitWindow(duration_seconds=3600, max_requests=10000, burst_allowance=1000)
        ],
        mode=RateLimitMode.SHADOW,  # Shadow mode for testing
        priority=1,
        description="High limits for testing environment"
    ))
    
    return policies