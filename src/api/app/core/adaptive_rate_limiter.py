"""
Production-ready adaptive rate limiting with multi-algorithm support and observability.

This module provides enterprise-grade rate limiting with:
- Token Bucket (burst tolerance) + Sliding Window (precision)
- Hierarchical policy enforcement (global → tenant → role → endpoint)
- Adaptive controls (reputation scoring, progressive backoff)
- Circuit breakers for DoS protection
- Comprehensive observability and shadow mode
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID
import logging

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog

from ..core.metrics import get_metrics_registry
from ..domain.value_objects import RateLimitInfo


logger = structlog.get_logger("adaptive_rate_limiter")


class LimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class PolicyScope(Enum):
    """Policy enforcement scopes"""
    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    TENANT = "tenant"
    ENDPOINT = "endpoint"
    ASN = "asn"
    DEVICE = "device"


class ReputationLevel(Enum):
    """User/IP reputation levels"""
    EXCELLENT = "excellent"  # 1.5x limits
    GOOD = "good"           # 1.2x limits
    NEUTRAL = "neutral"     # 1.0x limits
    POOR = "poor"          # 0.7x limits
    BLOCKED = "blocked"    # 0.0x limits


@dataclass
class RateLimitPolicy:
    """Rate limiting policy configuration"""
    scope: PolicyScope
    algorithm: LimitAlgorithm
    requests_per_second: float
    burst_size: int
    window_seconds: int = 60
    cost_multiplier: float = 1.0
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    
    # Adaptive controls
    reputation_multiplier: Dict[ReputationLevel, float] = field(default_factory=lambda: {
        ReputationLevel.EXCELLENT: 1.5,
        ReputationLevel.GOOD: 1.2,
        ReputationLevel.NEUTRAL: 1.0,
        ReputationLevel.POOR: 0.7,
        ReputationLevel.BLOCKED: 0.0
    })
    
    # Progressive backoff
    backoff_base_seconds: int = 60
    backoff_max_seconds: int = 3600
    backoff_multiplier: float = 2.0
    
    # Circuit breaker thresholds
    circuit_breaker_enabled: bool = False
    failure_threshold: int = 100
    failure_window_seconds: int = 60
    recovery_timeout_seconds: int = 300


@dataclass
class RateLimitResult:
    """Rate limiting decision result"""
    allowed: bool
    policy_matched: RateLimitPolicy
    tokens_remaining: int
    retry_after_seconds: Optional[int]
    reputation_level: ReputationLevel
    backoff_level: int
    circuit_breaker_open: bool
    
    # Observability
    algorithm_used: LimitAlgorithm
    computation_time_ms: float
    redis_hits: int
    cache_hits: int
    
    # Correlation
    correlation_id: str
    decision_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    failures: int = 0
    failure_window_start: float = 0
    state: str = "closed"  # closed, open, half_open
    last_failure_time: float = 0
    recovery_start_time: float = 0


class AdaptiveRateLimiter:
    """
    Enterprise adaptive rate limiter with multiple algorithms and observability.
    
    Features:
    - Multi-algorithm support (Token Bucket + Sliding Window)
    - Hierarchical policy enforcement
    - Reputation-based adaptive limits
    - Progressive backoff for repeat offenders
    - Circuit breakers for DoS protection
    - Comprehensive metrics and tracing
    - Shadow mode for testing
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        policies: List[RateLimitPolicy],
        shadow_mode: bool = False,
        enable_tracing: bool = True,
        metric_prefix: str = "rate_limiter"
    ):
        self.redis = redis_client
        self.policies = sorted(policies, key=lambda p: p.priority)
        self.shadow_mode = shadow_mode
        self.enable_tracing = enable_tracing
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Local cache for hot path optimization
        self.local_cache: Dict[str, Tuple[float, Any]] = {}
        self.cache_ttl = 5.0  # 5 second local cache
        
        # Metrics
        self._setup_metrics(metric_prefix)
        
        # Lua scripts for atomic operations
        self._load_lua_scripts()
        
        logger.info(
            "Adaptive rate limiter initialized",
            policies_count=len(policies),
            shadow_mode=shadow_mode,
            enable_tracing=enable_tracing
        )
    
    def _setup_metrics(self, prefix: str):
        """Setup Prometheus metrics"""
        registry = get_metrics_registry()
        
        self.metrics = {
            'requests_total': Counter(
                f'{prefix}_requests_total',
                'Total rate limit checks',
                ['scope', 'algorithm', 'result', 'reputation'],
                registry=registry
            ),
            'tokens_remaining': Gauge(
                f'{prefix}_tokens_remaining',
                'Remaining tokens in bucket',
                ['scope', 'key_hash'],
                registry=registry
            ),
            'computation_time': Histogram(
                f'{prefix}_computation_time_seconds',
                'Rate limit computation time',
                ['algorithm'],
                registry=registry
            ),
            'redis_operations': Counter(
                f'{prefix}_redis_operations_total',
                'Redis operations count',
                ['operation', 'result'],
                registry=registry
            ),
            'circuit_breaker_state': Gauge(
                f'{prefix}_circuit_breaker_open',
                'Circuit breaker state (1=open, 0=closed)',
                ['scope'],
                registry=registry
            ),
            'reputation_scores': Histogram(
                f'{prefix}_reputation_scores',
                'Reputation score distribution',
                ['scope'],
                registry=registry
            ),
            'backoff_levels': Counter(
                f'{prefix}_backoff_levels_total',
                'Progressive backoff level distribution',
                ['level'],
                registry=registry
            )
        }
    
    def _load_lua_scripts(self):
        """Load Lua scripts for atomic Redis operations"""
        
        # Token bucket algorithm with burst tolerance
        self.token_bucket_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local reputation_multiplier = tonumber(ARGV[5])
        
        -- Apply reputation multiplier to capacity and rate
        capacity = math.floor(capacity * reputation_multiplier)
        refill_rate = refill_rate * reputation_multiplier
        
        -- Get current state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add based on elapsed time
        local elapsed = now - last_refill
        local tokens_to_add = elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- Check if request can be satisfied
        if tokens >= cost then
            tokens = tokens - cost
            
            -- Update state
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
            
            return {1, math.floor(tokens), 0}  -- allowed, remaining, retry_after
        else
            -- Calculate retry after
            local deficit = cost - tokens
            local retry_after = math.ceil(deficit / refill_rate)
            
            -- Update last_refill even on reject to prevent gaming
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            
            return {0, math.floor(tokens), retry_after}
        end
        """
        
        # Sliding window log with precise tracking
        self.sliding_window_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_seconds = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local reputation_multiplier = tonumber(ARGV[5])
        
        -- Apply reputation multiplier to limit
        limit = math.floor(limit * reputation_multiplier)
        
        local window_start = now - window_seconds
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- Count current requests
        local current_count = redis.call('ZCARD', key)
        
        -- Check if adding cost would exceed limit
        if current_count + cost > limit then
            -- Get oldest entry for retry calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if oldest[2] then
                retry_after = math.ceil(tonumber(oldest[2]) + window_seconds - now)
            end
            
            return {0, limit - current_count, retry_after}
        end
        
        -- Add current request(s)
        for i = 1, cost do
            local score = now + (i * 0.001)  -- Sub-millisecond precision
            redis.call('ZADD', key, score, score .. ':' .. math.random(1000000))
        end
        
        -- Set expiration
        redis.call('EXPIRE', key, window_seconds + 60)
        
        return {1, limit - current_count - cost, 0}
        """
        
        # Circuit breaker state management
        self.circuit_breaker_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local failure_threshold = tonumber(ARGV[2])
        local failure_window = tonumber(ARGV[3])
        local recovery_timeout = tonumber(ARGV[4])
        local is_success = tonumber(ARGV[5])  -- 1 for success, 0 for failure
        
        -- Get current state
        local state = redis.call('HMGET', key, 'failures', 'window_start', 'state', 'recovery_start')
        local failures = tonumber(state[1]) or 0
        local window_start = tonumber(state[2]) or now
        local cb_state = state[3] or 'closed'
        local recovery_start = tonumber(state[4]) or 0
        
        -- Reset window if expired
        if now - window_start > failure_window then
            failures = 0
            window_start = now
        end
        
        -- State machine logic
        if cb_state == 'closed' then
            if is_success == 1 then
                failures = math.max(0, failures - 1)  -- Gradual recovery
            else
                failures = failures + 1
                if failures >= failure_threshold then
                    cb_state = 'open'
                    recovery_start = now
                end
            end
        elseif cb_state == 'open' then
            if now - recovery_start > recovery_timeout then
                cb_state = 'half_open'
            end
        elseif cb_state == 'half_open' then
            if is_success == 1 then
                cb_state = 'closed'
                failures = 0
            else
                cb_state = 'open'
                recovery_start = now
                failures = failures + 1
            end
        end
        
        -- Update state
        redis.call('HMSET', key, 'failures', failures, 'window_start', window_start, 
                   'state', cb_state, 'recovery_start', recovery_start)
        redis.call('EXPIRE', key, failure_window * 2)
        
        -- Return: state_open (1/0), failures, state_name
        local state_open = (cb_state == 'open') and 1 or 0
        return {state_open, failures, cb_state}
        """
    
    async def check_rate_limit(
        self,
        identifier: str,
        scope: PolicyScope,
        endpoint: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        cost: int = 1,
        correlation_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit with comprehensive policy evaluation.
        
        Returns decision based on hierarchical policy matching and adaptive controls.
        """
        start_time = time.time()
        correlation_id = correlation_id or self._generate_correlation_id()
        
        try:
            # Find applicable policy
            policy = self._find_applicable_policy(scope, endpoint)
            if not policy:
                # No policy found, allow by default but log
                logger.warning(
                    "No rate limit policy found for scope",
                    scope=scope.value,
                    endpoint=endpoint,
                    correlation_id=correlation_id
                )
                return self._create_allow_result(policy=None, correlation_id=correlation_id)
            
            # Check circuit breaker first
            circuit_breaker_open = await self._check_circuit_breaker(policy, identifier)
            if circuit_breaker_open and not self.shadow_mode:
                return self._create_circuit_breaker_result(policy, correlation_id)
            
            # Get reputation level
            reputation = await self._get_reputation_level(identifier, scope, ip_address)
            
            # Check for progressive backoff
            backoff_level = await self._get_backoff_level(identifier)
            if backoff_level > 0 and not self.shadow_mode:
                return self._create_backoff_result(policy, backoff_level, correlation_id)
            
            # Apply rate limiting algorithm
            if policy.algorithm == LimitAlgorithm.TOKEN_BUCKET:
                result = await self._check_token_bucket(policy, identifier, cost, reputation, correlation_id)
            elif policy.algorithm == LimitAlgorithm.SLIDING_WINDOW:
                result = await self._check_sliding_window(policy, identifier, cost, reputation, correlation_id)
            else:
                # Fallback to sliding window
                result = await self._check_sliding_window(policy, identifier, cost, reputation, correlation_id)
            
            # Record metrics
            computation_time = (time.time() - start_time) * 1000
            self._record_metrics(result, computation_time, reputation)
            
            # Update circuit breaker on success
            if result.allowed:
                await self._update_circuit_breaker(policy, identifier, success=True)
            else:
                await self._update_circuit_breaker(policy, identifier, success=False)
                # Increment backoff on reject
                await self._increment_backoff(identifier)
            
            # Shadow mode: log decision but always allow
            if self.shadow_mode:
                logger.info(
                    "Shadow mode rate limit decision",
                    identifier=identifier,
                    decision=result.allowed,
                    policy_scope=policy.scope.value,
                    algorithm=policy.algorithm.value,
                    correlation_id=correlation_id
                )
                result.allowed = True
            
            return result
        
        except Exception as e:
            # Fail open on errors but log extensively
            logger.error(
                "Rate limiter error - failing open",
                error=str(e),
                identifier=identifier,
                scope=scope.value,
                correlation_id=correlation_id,
                exc_info=True
            )
            self.metrics['redis_operations'].labels(operation='error', result='fail_open').inc()
            return self._create_error_result(correlation_id)
    
    async def _check_token_bucket(
        self,
        policy: RateLimitPolicy,
        identifier: str,
        cost: int,
        reputation: ReputationLevel,
        correlation_id: str
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm"""
        key = self._generate_key(policy.scope, identifier, policy.algorithm)
        reputation_multiplier = policy.reputation_multiplier[reputation]
        
        start_time = time.time()
        
        try:
            result = await self.redis.eval(
                self.token_bucket_script,
                1,
                key,
                time.time(),
                policy.burst_size,
                policy.requests_per_second,
                cost,
                reputation_multiplier
            )
            
            allowed = bool(result[0])
            tokens_remaining = int(result[1])
            retry_after = int(result[2]) if result[2] > 0 else None
            
            self.metrics['redis_operations'].labels(operation='token_bucket', result='success').inc()
            
            return RateLimitResult(
                allowed=allowed,
                policy_matched=policy,
                tokens_remaining=tokens_remaining,
                retry_after_seconds=retry_after,
                reputation_level=reputation,
                backoff_level=0,
                circuit_breaker_open=False,
                algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
                computation_time_ms=(time.time() - start_time) * 1000,
                redis_hits=1,
                cache_hits=0,
                correlation_id=correlation_id,
                decision_metadata={
                    "key": self._hash_key(key),
                    "reputation_multiplier": reputation_multiplier,
                    "cost": cost
                }
            )
        
        except Exception as e:
            logger.error("Token bucket check failed", error=str(e), key=key)
            self.metrics['redis_operations'].labels(operation='token_bucket', result='error').inc()
            raise
    
    async def _check_sliding_window(
        self,
        policy: RateLimitPolicy,
        identifier: str,
        cost: int,
        reputation: ReputationLevel,
        correlation_id: str
    ) -> RateLimitResult:
        """Check rate limit using sliding window log algorithm"""
        key = self._generate_key(policy.scope, identifier, policy.algorithm)
        reputation_multiplier = policy.reputation_multiplier[reputation]
        limit = int(policy.requests_per_second * policy.window_seconds)
        
        start_time = time.time()
        
        try:
            result = await self.redis.eval(
                self.sliding_window_script,
                1,
                key,
                time.time(),
                policy.window_seconds,
                limit,
                cost,
                reputation_multiplier
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = int(result[2]) if result[2] > 0 else None
            
            self.metrics['redis_operations'].labels(operation='sliding_window', result='success').inc()
            
            return RateLimitResult(
                allowed=allowed,
                policy_matched=policy,
                tokens_remaining=remaining,
                retry_after_seconds=retry_after,
                reputation_level=reputation,
                backoff_level=0,
                circuit_breaker_open=False,
                algorithm_used=LimitAlgorithm.SLIDING_WINDOW,
                computation_time_ms=(time.time() - start_time) * 1000,
                redis_hits=1,
                cache_hits=0,
                correlation_id=correlation_id,
                decision_metadata={
                    "key": self._hash_key(key),
                    "window_seconds": policy.window_seconds,
                    "limit": limit,
                    "reputation_multiplier": reputation_multiplier
                }
            )
        
        except Exception as e:
            logger.error("Sliding window check failed", error=str(e), key=key)
            self.metrics['redis_operations'].labels(operation='sliding_window', result='error').inc()
            raise
    
    async def _check_circuit_breaker(self, policy: RateLimitPolicy, identifier: str) -> bool:
        """Check if circuit breaker is open for this policy/identifier"""
        if not policy.circuit_breaker_enabled:
            return False
        
        key = f"cb:{policy.scope.value}:{self._hash_key(identifier)}"
        
        try:
            # Check local cache first
            cache_key = f"cb_state:{key}"
            if cache_key in self.local_cache:
                cache_time, state = self.local_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    self.metrics['redis_operations'].labels(operation='circuit_breaker', result='cache_hit').inc()
                    return state == 'open'
            
            # Check Redis
            state_data = await self.redis.hmget(key, 'state', 'recovery_start')
            state = state_data[0] or 'closed'
            recovery_start = float(state_data[1] or 0)
            
            # Update local cache
            self.local_cache[cache_key] = (time.time(), state)
            
            # Check if we should transition from open to half_open
            if state == 'open' and time.time() - recovery_start > policy.recovery_timeout_seconds:
                state = 'half_open'
            
            is_open = state == 'open'
            self.metrics['circuit_breaker_state'].labels(scope=policy.scope.value).set(1 if is_open else 0)
            
            return is_open
        
        except Exception as e:
            logger.error("Circuit breaker check failed", error=str(e), key=key)
            return False  # Fail closed on errors
    
    async def _update_circuit_breaker(self, policy: RateLimitPolicy, identifier: str, success: bool):
        """Update circuit breaker state"""
        if not policy.circuit_breaker_enabled:
            return
        
        key = f"cb:{policy.scope.value}:{self._hash_key(identifier)}"
        
        try:
            await self.redis.eval(
                self.circuit_breaker_script,
                1,
                key,
                time.time(),
                policy.failure_threshold,
                policy.failure_window_seconds,
                policy.recovery_timeout_seconds,
                1 if success else 0
            )
        except Exception as e:
            logger.error("Circuit breaker update failed", error=str(e), key=key)
    
    async def _get_reputation_level(
        self,
        identifier: str,
        scope: PolicyScope,
        ip_address: Optional[str] = None
    ) -> ReputationLevel:
        """Get reputation level for identifier"""
        # Simple implementation - can be enhanced with ML models
        reputation_key = f"reputation:{scope.value}:{self._hash_key(identifier)}"
        
        try:
            # Check local cache
            cache_key = f"reputation:{reputation_key}"
            if cache_key in self.local_cache:
                cache_time, reputation = self.local_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl * 2:  # Longer cache for reputation
                    return reputation
            
            # Get from Redis
            reputation_data = await self.redis.hmget(
                reputation_key, 
                'score', 'violations', 'last_violation'
            )
            
            score = float(reputation_data[0] or 50.0)  # Default neutral score
            violations = int(reputation_data[1] or 0)
            last_violation = float(reputation_data[2] or 0)
            
            # Decay violations over time (24 hour half-life)
            time_since_violation = time.time() - last_violation
            if time_since_violation > 0:
                decay_factor = 0.5 ** (time_since_violation / 86400)  # 24 hours
                violations = int(violations * decay_factor)
            
            # Calculate reputation level
            if score >= 80 and violations == 0:
                reputation = ReputationLevel.EXCELLENT
            elif score >= 60 and violations <= 2:
                reputation = ReputationLevel.GOOD
            elif score >= 20 and violations <= 10:
                reputation = ReputationLevel.NEUTRAL
            elif violations > 50:
                reputation = ReputationLevel.BLOCKED
            else:
                reputation = ReputationLevel.POOR
            
            # Update cache
            self.local_cache[cache_key] = (time.time(), reputation)
            
            return reputation
        
        except Exception as e:
            logger.error("Reputation check failed", error=str(e), identifier=identifier)
            return ReputationLevel.NEUTRAL  # Fail to neutral
    
    async def _get_backoff_level(self, identifier: str) -> int:
        """Get progressive backoff level"""
        backoff_key = f"backoff:{self._hash_key(identifier)}"
        
        try:
            backoff_data = await self.redis.hmget(backoff_key, 'level', 'expires')
            level = int(backoff_data[0] or 0)
            expires = float(backoff_data[1] or 0)
            
            if time.time() > expires:
                # Backoff expired, reset
                await self.redis.delete(backoff_key)
                return 0
            
            return level
        
        except Exception as e:
            logger.error("Backoff check failed", error=str(e), identifier=identifier)
            return 0
    
    async def _increment_backoff(self, identifier: str):
        """Increment progressive backoff level"""
        backoff_key = f"backoff:{self._hash_key(identifier)}"
        
        try:
            # Get current level
            current_level = await self._get_backoff_level(identifier)
            new_level = min(current_level + 1, 10)  # Max 10 levels
            
            # Calculate exponential backoff duration
            base_duration = 60  # 1 minute base
            max_duration = 3600  # 1 hour max
            duration = min(base_duration * (2 ** new_level), max_duration)
            expires = time.time() + duration
            
            await self.redis.hmset(backoff_key, {
                'level': new_level,
                'expires': expires
            })
            await self.redis.expire(backoff_key, int(duration) + 60)
            
            self.metrics['backoff_levels'].labels(level=str(new_level)).inc()
        
        except Exception as e:
            logger.error("Backoff increment failed", error=str(e), identifier=identifier)
    
    def _find_applicable_policy(self, scope: PolicyScope, endpoint: Optional[str] = None) -> Optional[RateLimitPolicy]:
        """Find the most specific applicable policy"""
        # Policies are sorted by priority (lower = higher priority)
        for policy in self.policies:
            if policy.scope == scope and policy.enabled:
                # For endpoint policies, check if endpoint matches
                if scope == PolicyScope.ENDPOINT and endpoint:
                    # Simple prefix matching - can be enhanced with regex
                    if any(endpoint.startswith(pattern) for pattern in policy.decision_metadata.get('endpoint_patterns', [endpoint])):
                        return policy
                else:
                    return policy
        return None
    
    def _generate_key(self, scope: PolicyScope, identifier: str, algorithm: LimitAlgorithm) -> str:
        """Generate Redis key for rate limit tracking"""
        return f"rl:{scope.value}:{algorithm.value}:{self._hash_key(identifier)}"
    
    def _hash_key(self, key: str) -> str:
        """Hash sensitive keys for privacy"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for request tracking"""
        return hashlib.md5(f"{time.time()}:{id(self)}".encode()).hexdigest()[:12]
    
    def _create_allow_result(
        self,
        policy: Optional[RateLimitPolicy],
        correlation_id: str
    ) -> RateLimitResult:
        """Create allow result"""
        return RateLimitResult(
            allowed=True,
            policy_matched=policy,
            tokens_remaining=999999,
            retry_after_seconds=None,
            reputation_level=ReputationLevel.NEUTRAL,
            backoff_level=0,
            circuit_breaker_open=False,
            algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
            computation_time_ms=0.1,
            redis_hits=0,
            cache_hits=1,
            correlation_id=correlation_id
        )
    
    def _create_circuit_breaker_result(
        self,
        policy: RateLimitPolicy,
        correlation_id: str
    ) -> RateLimitResult:
        """Create circuit breaker open result"""
        return RateLimitResult(
            allowed=False,
            policy_matched=policy,
            tokens_remaining=0,
            retry_after_seconds=policy.recovery_timeout_seconds,
            reputation_level=ReputationLevel.POOR,
            backoff_level=0,
            circuit_breaker_open=True,
            algorithm_used=policy.algorithm,
            computation_time_ms=0.5,
            redis_hits=1,
            cache_hits=0,
            correlation_id=correlation_id,
            decision_metadata={"reason": "circuit_breaker_open"}
        )
    
    def _create_backoff_result(
        self,
        policy: RateLimitPolicy,
        backoff_level: int,
        correlation_id: str
    ) -> RateLimitResult:
        """Create progressive backoff result"""
        retry_after = policy.backoff_base_seconds * (policy.backoff_multiplier ** backoff_level)
        
        return RateLimitResult(
            allowed=False,
            policy_matched=policy,
            tokens_remaining=0,
            retry_after_seconds=min(int(retry_after), policy.backoff_max_seconds),
            reputation_level=ReputationLevel.POOR,
            backoff_level=backoff_level,
            circuit_breaker_open=False,
            algorithm_used=policy.algorithm,
            computation_time_ms=0.2,
            redis_hits=1,
            cache_hits=0,
            correlation_id=correlation_id,
            decision_metadata={"reason": "progressive_backoff", "level": backoff_level}
        )
    
    def _create_error_result(self, correlation_id: str) -> RateLimitResult:
        """Create error result (fail open)"""
        return RateLimitResult(
            allowed=True,
            policy_matched=None,
            tokens_remaining=999999,
            retry_after_seconds=None,
            reputation_level=ReputationLevel.NEUTRAL,
            backoff_level=0,
            circuit_breaker_open=False,
            algorithm_used=LimitAlgorithm.TOKEN_BUCKET,
            computation_time_ms=1.0,
            redis_hits=0,
            cache_hits=0,
            correlation_id=correlation_id,
            decision_metadata={"reason": "limiter_error_fail_open"}
        )
    
    def _record_metrics(self, result: RateLimitResult, computation_time: float, reputation: ReputationLevel):
        """Record comprehensive metrics"""
        # Request outcome
        self.metrics['requests_total'].labels(
            scope=result.policy_matched.scope.value if result.policy_matched else 'unknown',
            algorithm=result.algorithm_used.value,
            result='allowed' if result.allowed else 'blocked',
            reputation=reputation.value
        ).inc()
        
        # Computation time
        self.metrics['computation_time'].labels(
            algorithm=result.algorithm_used.value
        ).observe(computation_time / 1000)  # Convert to seconds
        
        # Reputation scores
        reputation_score = {
            ReputationLevel.EXCELLENT: 90,
            ReputationLevel.GOOD: 70,
            ReputationLevel.NEUTRAL: 50,
            ReputationLevel.POOR: 30,
            ReputationLevel.BLOCKED: 10
        }[reputation]
        
        if result.policy_matched:
            self.metrics['reputation_scores'].labels(
                scope=result.policy_matched.scope.value
            ).observe(reputation_score)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter statistics"""
        try:
            # Get Redis info
            redis_info = await self.redis.info()
            
            # Count keys by scope
            key_counts = {}
            for scope in PolicyScope:
                pattern = f"rl:{scope.value}:*"
                keys = await self.redis.keys(pattern)
                key_counts[scope.value] = len(keys)
            
            return {
                "policies": {
                    "total": len(self.policies),
                    "enabled": len([p for p in self.policies if p.enabled]),
                    "by_scope": {scope.value: len([p for p in self.policies if p.scope == scope]) 
                               for scope in PolicyScope}
                },
                "redis": {
                    "connected_clients": redis_info.get('connected_clients', 0),
                    "used_memory_human": redis_info.get('used_memory_human', '0B'),
                    "keyspace_hits": redis_info.get('keyspace_hits', 0),
                    "keyspace_misses": redis_info.get('keyspace_misses', 0)
                },
                "keys": key_counts,
                "circuit_breakers": len(self.circuit_breakers),
                "local_cache_size": len(self.local_cache),
                "shadow_mode": self.shadow_mode
            }
        
        except Exception as e:
            logger.error("Failed to get rate limiter stats", error=str(e))
            return {"error": "stats_unavailable"}
    
    async def cleanup_expired_keys(self):
        """Cleanup expired keys and states (maintenance task)"""
        try:
            # Cleanup expired backoff entries
            backoff_keys = await self.redis.keys("backoff:*")
            for key in backoff_keys:
                expires = await self.redis.hget(key, 'expires')
                if expires and time.time() > float(expires):
                    await self.redis.delete(key)
            
            # Cleanup local cache
            current_time = time.time()
            expired_keys = [
                k for k, (cache_time, _) in self.local_cache.items()
                if current_time - cache_time > self.cache_ttl * 2
            ]
            for key in expired_keys:
                del self.local_cache[key]
            
            logger.debug(
                "Rate limiter cleanup completed",
                backoff_keys_checked=len(backoff_keys),
                local_cache_expired=len(expired_keys)
            )
        
        except Exception as e:
            logger.error("Rate limiter cleanup failed", error=str(e))


# Emergency rate limiting with kill-switch
class EmergencyRateLimiter:
    """Emergency rate limiter with kill-switch functionality"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.emergency_active = False
        self.kill_switch_active = False
    
    async def check_emergency_mode(self) -> bool:
        """Check if emergency rate limiting is active"""
        try:
            status = await self.redis.get("emergency:rate_limit")
            return status == "active"
        except:
            return False
    
    async def activate_emergency_mode(self, duration_seconds: int = 300):
        """Activate emergency rate limiting"""
        await self.redis.setex("emergency:rate_limit", duration_seconds, "active")
        self.emergency_active = True
        logger.critical("Emergency rate limiting activated", duration_seconds=duration_seconds)
    
    async def activate_kill_switch(self):
        """Activate kill-switch (block all requests)"""
        await self.redis.set("kill_switch:active", "true")
        self.kill_switch_active = True
        logger.critical("Rate limiter kill-switch activated")
    
    async def deactivate_kill_switch(self):
        """Deactivate kill-switch"""
        await self.redis.delete("kill_switch:active")
        self.kill_switch_active = False
        logger.critical("Rate limiter kill-switch deactivated")
    
    async def is_kill_switch_active(self) -> bool:
        """Check if kill-switch is active"""
        try:
            status = await self.redis.get("kill_switch:active")
            return status == "true"
        except:
            return False