"""
Advanced API Rate Limiting and Quotas Middleware
Sophisticated rate limiting with multiple algorithms, tenant-aware quotas, and adaptive throttling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import redis.asyncio as redis
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import math

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class QuotaType(Enum):
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"


class RateLimitScope(Enum):
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    quota_type: QuotaType
    limit: int
    window_size: int  # in seconds
    
    # Advanced settings
    burst_limit: Optional[int] = None  # Allow bursts up to this limit
    leak_rate: Optional[float] = None  # For leaky bucket (requests per second)
    replenishment_rate: Optional[float] = None  # For token bucket (tokens per second)
    
    # Conditions
    path_patterns: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    tenant_tiers: List[str] = field(default_factory=list)
    
    # Adaptive behavior
    adaptive_scaling: bool = False
    scale_factor_range: Tuple[float, float] = (0.5, 2.0)
    
    # Penalties
    penalty_factor: float = 1.0  # Multiply limit by this on violations
    penalty_duration: int = 300  # Penalty duration in seconds
    
    def matches_request(self, request: Request, tenant_tier: Optional[str] = None) -> bool:
        """Check if rule applies to request"""
        # Check path patterns
        if self.path_patterns:
            import re
            path_match = any(re.match(pattern, request.url.path) for pattern in self.path_patterns)
            if not path_match:
                return False
                
        # Check methods
        if self.methods and request.method not in self.methods:
            return False
            
        # Check tenant tiers
        if self.tenant_tiers and tenant_tier not in self.tenant_tiers:
            return False
            
        return True


@dataclass
class RateLimitState:
    """Current rate limiting state"""
    key: str
    rule: RateLimitRule
    current_count: int = 0
    window_start: float = 0.0
    last_request: float = 0.0
    
    # Token bucket specific
    tokens: float = 0.0
    last_refill: float = 0.0
    
    # Sliding window specific
    request_timestamps: List[float] = field(default_factory=list)
    
    # Adaptive scaling
    scale_factor: float = 1.0
    violation_count: int = 0
    last_violation: Optional[float] = None
    
    # Penalty state
    penalty_active: bool = False
    penalty_until: Optional[float] = None
    
    def get_effective_limit(self) -> int:
        """Get current effective limit considering penalties and scaling"""
        base_limit = self.rule.limit
        
        if self.penalty_active and self.penalty_until and time.time() < self.penalty_until:
            base_limit = int(base_limit * self.rule.penalty_factor)
            
        if self.rule.adaptive_scaling:
            base_limit = int(base_limit * self.scale_factor)
            
        return max(1, base_limit)  # Ensure at least 1 request allowed


class AdvancedRateLimiter:
    """Advanced rate limiting middleware with multiple algorithms and adaptive behavior"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.rules: List[RateLimitRule] = []
        self.global_circuit_breaker = False
        
        # Metrics and monitoring
        self.request_counter = 0
        self.rejection_counter = 0
        self.last_stats_update = time.time()
        
        # Initialize default rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        # Global rate limiting
        self.rules.append(RateLimitRule(
            name="global_per_second",
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=RateLimitScope.GLOBAL,
            quota_type=QuotaType.PER_SECOND,
            limit=1000,
            window_size=1,
            adaptive_scaling=True
        ))
        
        # Per-IP rate limiting
        self.rules.append(RateLimitRule(
            name="ip_per_minute",
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.IP,
            quota_type=QuotaType.PER_MINUTE,
            limit=100,
            window_size=60,
            burst_limit=20,
            replenishment_rate=100/60,  # 100 tokens per minute
            penalty_factor=0.5,
            penalty_duration=300
        ))
        
        # Tenant-based rate limiting
        self.rules.append(RateLimitRule(
            name="tenant_basic_per_hour",
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.TENANT,
            quota_type=QuotaType.PER_HOUR,
            limit=10000,
            window_size=3600,
            tenant_tiers=["basic"]
        ))
        
        self.rules.append(RateLimitRule(
            name="tenant_professional_per_hour",
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.TENANT,
            quota_type=QuotaType.PER_HOUR,
            limit=100000,
            window_size=3600,
            tenant_tiers=["professional"]
        ))
        
        self.rules.append(RateLimitRule(
            name="tenant_enterprise_per_hour",
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.TENANT,
            quota_type=QuotaType.PER_HOUR,
            limit=1000000,
            window_size=3600,
            tenant_tiers=["enterprise"]
        ))
        
        # API endpoint specific limits
        self.rules.append(RateLimitRule(
            name="auth_endpoint_limit",
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
            scope=RateLimitScope.IP,
            quota_type=QuotaType.PER_MINUTE,
            limit=10,
            window_size=60,
            leak_rate=10/60,  # 10 requests per minute
            path_patterns=[r"/api/v1/auth/.*"],
            penalty_factor=0.2,
            penalty_duration=600  # 10 minutes penalty
        ))
        
        # Heavy computation endpoints
        self.rules.append(RateLimitRule(
            name="scan_endpoint_limit",
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.USER,
            quota_type=QuotaType.PER_HOUR,
            limit=50,
            window_size=3600,
            burst_limit=5,
            replenishment_rate=50/3600,
            path_patterns=[r"/api/v1/scans.*"],
            methods=["POST"]
        ))
        
    async def middleware(self, request: Request, call_next: Callable) -> Response:
        """Rate limiting middleware function"""
        start_time = time.time()
        self.request_counter += 1
        
        # Skip rate limiting for health checks and internal endpoints
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
            
        # Check global circuit breaker
        if self.global_circuit_breaker:
            return self._create_rate_limit_response(
                "Service temporarily unavailable due to high load",
                status.HTTP_503_SERVICE_UNAVAILABLE,
                retry_after=60
            )
            
        # Extract request context
        context = await self._extract_request_context(request)
        
        # Apply rate limiting rules
        for rule in self.rules:
            if rule.matches_request(request, context.get("tenant_tier")):
                allowed, state, retry_after = await self._check_rate_limit(rule, context)
                
                if not allowed:
                    self.rejection_counter += 1
                    
                    # Update violation tracking
                    await self._record_violation(rule, context, state)
                    
                    # Create response with rate limit headers
                    response = self._create_rate_limit_response(
                        f"Rate limit exceeded for {rule.name}",
                        status.HTTP_429_TOO_MANY_REQUESTS,
                        retry_after=retry_after,
                        rule=rule,
                        state=state
                    )
                    
                    # Log violation
                    logger.warning(
                        f"Rate limit violation: {rule.name} for {context.get('identifier', 'unknown')} "
                        f"({state.current_count}/{state.get_effective_limit()})"
                    )
                    
                    return response
                    
        # All rate limits passed, process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        await self._add_rate_limit_headers(response, context)
        
        # Update adaptive scaling based on response time
        processing_time = time.time() - start_time
        await self._update_adaptive_scaling(processing_time, context)
        
        return response
        
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request"""
        skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        return any(request.url.path.startswith(path) for path in skip_paths)
        
    async def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract context information from request"""
        context = {
            "ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
            "method": request.method,
            "path": request.url.path,
            "timestamp": time.time()
        }
        
        # Extract tenant information
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            context["tenant_id"] = tenant_id
            # In production, fetch tenant tier from database/cache
            context["tenant_tier"] = "professional"  # Placeholder
            
        # Extract user information
        user_id = request.headers.get("X-User-ID")
        if user_id:
            context["user_id"] = user_id
            
        # Extract API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            context["api_key"] = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            
        # Create identifier for rate limiting
        context["identifier"] = self._create_identifier(context)
        
        return context
        
    def _create_identifier(self, context: Dict[str, Any]) -> str:
        """Create unique identifier for rate limiting"""
        # Priority order: API key > User ID > Tenant ID > IP
        if "api_key" in context:
            return f"api_key:{context['api_key']}"
        elif "user_id" in context:
            return f"user:{context['user_id']}"
        elif "tenant_id" in context:
            return f"tenant:{context['tenant_id']}"
        else:
            return f"ip:{context['ip']}"
            
    async def _check_rate_limit(self, rule: RateLimitRule, context: Dict[str, Any]) -> Tuple[bool, RateLimitState, Optional[int]]:
        """Check if request passes rate limit rule"""
        # Create rate limit key
        key = self._create_rate_limit_key(rule, context)
        
        # Get current state
        state = await self._get_rate_limit_state(key, rule)
        
        # Apply rate limiting algorithm
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            allowed, retry_after = await self._apply_token_bucket(state)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            allowed, retry_after = await self._apply_sliding_window(state)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            allowed, retry_after = await self._apply_fixed_window(state)
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            allowed, retry_after = await self._apply_leaky_bucket(state)
        else:
            allowed, retry_after = True, None
            
        # Save updated state
        await self._save_rate_limit_state(state)
        
        return allowed, state, retry_after
        
    def _create_rate_limit_key(self, rule: RateLimitRule, context: Dict[str, Any]) -> str:
        """Create Redis key for rate limit state"""
        scope_key = ""
        
        if rule.scope == RateLimitScope.GLOBAL:
            scope_key = "global"
        elif rule.scope == RateLimitScope.TENANT:
            scope_key = f"tenant:{context.get('tenant_id', 'unknown')}"
        elif rule.scope == RateLimitScope.USER:
            scope_key = f"user:{context.get('user_id', 'unknown')}"
        elif rule.scope == RateLimitScope.IP:
            scope_key = f"ip:{context['ip']}"
        elif rule.scope == RateLimitScope.API_KEY:
            scope_key = f"api_key:{context.get('api_key', 'unknown')}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            scope_key = f"endpoint:{hashlib.md5(context['path'].encode()).hexdigest()[:8]}"
            
        return f"rate_limit:{rule.name}:{scope_key}"
        
    async def _get_rate_limit_state(self, key: str, rule: RateLimitRule) -> RateLimitState:
        """Get rate limit state from Redis"""
        try:
            data = await self.redis.get(key)
            if data:
                state_dict = json.loads(data.decode())
                state = RateLimitState(
                    key=key,
                    rule=rule,
                    current_count=state_dict.get("current_count", 0),
                    window_start=state_dict.get("window_start", time.time()),
                    last_request=state_dict.get("last_request", 0.0),
                    tokens=state_dict.get("tokens", rule.burst_limit or rule.limit),
                    last_refill=state_dict.get("last_refill", time.time()),
                    request_timestamps=state_dict.get("request_timestamps", []),
                    scale_factor=state_dict.get("scale_factor", 1.0),
                    violation_count=state_dict.get("violation_count", 0),
                    last_violation=state_dict.get("last_violation"),
                    penalty_active=state_dict.get("penalty_active", False),
                    penalty_until=state_dict.get("penalty_until")
                )
                return state
        except Exception as e:
            logger.error(f"Failed to get rate limit state for {key}: {e}")
            
        # Return new state
        return RateLimitState(
            key=key,
            rule=rule,
            window_start=time.time(),
            tokens=rule.burst_limit or rule.limit,
            last_refill=time.time()
        )
        
    async def _save_rate_limit_state(self, state: RateLimitState):
        """Save rate limit state to Redis"""
        try:
            state_dict = {
                "current_count": state.current_count,
                "window_start": state.window_start,
                "last_request": state.last_request,
                "tokens": state.tokens,
                "last_refill": state.last_refill,
                "request_timestamps": state.request_timestamps,
                "scale_factor": state.scale_factor,
                "violation_count": state.violation_count,
                "last_violation": state.last_violation,
                "penalty_active": state.penalty_active,
                "penalty_until": state.penalty_until
            }
            
            # Set TTL based on window size plus some buffer
            ttl = state.rule.window_size + 3600  # Add 1 hour buffer
            
            await self.redis.setex(
                state.key,
                ttl,
                json.dumps(state_dict)
            )
        except Exception as e:
            logger.error(f"Failed to save rate limit state for {state.key}: {e}")
            
    async def _apply_token_bucket(self, state: RateLimitState) -> Tuple[bool, Optional[int]]:
        """Apply token bucket algorithm"""
        now = time.time()
        rule = state.rule
        
        # Calculate tokens to add based on replenishment rate
        if rule.replenishment_rate:
            time_passed = now - state.last_refill
            tokens_to_add = time_passed * rule.replenishment_rate
            state.tokens = min(rule.burst_limit or rule.limit, state.tokens + tokens_to_add)
            state.last_refill = now
            
        # Check if token available
        if state.tokens >= 1:
            state.tokens -= 1
            state.last_request = now
            return True, None
        else:
            # Calculate retry after based on replenishment rate
            retry_after = int(1.0 / rule.replenishment_rate) if rule.replenishment_rate else rule.window_size
            return False, retry_after
            
    async def _apply_sliding_window(self, state: RateLimitState) -> Tuple[bool, Optional[int]]:
        """Apply sliding window algorithm"""
        now = time.time()
        rule = state.rule
        window_start = now - rule.window_size
        
        # Clean old timestamps
        state.request_timestamps = [
            ts for ts in state.request_timestamps if ts >= window_start
        ]
        
        # Check if under limit
        effective_limit = state.get_effective_limit()
        if len(state.request_timestamps) < effective_limit:
            state.request_timestamps.append(now)
            state.current_count = len(state.request_timestamps)
            state.last_request = now
            return True, None
        else:
            # Calculate retry after (time until oldest request expires)
            if state.request_timestamps:
                retry_after = int(state.request_timestamps[0] + rule.window_size - now) + 1
            else:
                retry_after = rule.window_size
            return False, max(1, retry_after)
            
    async def _apply_fixed_window(self, state: RateLimitState) -> Tuple[bool, Optional[int]]:
        """Apply fixed window algorithm"""
        now = time.time()
        rule = state.rule
        
        # Check if we need to reset the window
        if now >= state.window_start + rule.window_size:
            state.window_start = now
            state.current_count = 0
            
        # Check if under limit
        effective_limit = state.get_effective_limit()
        if state.current_count < effective_limit:
            state.current_count += 1
            state.last_request = now
            return True, None
        else:
            # Calculate retry after (time until window resets)
            retry_after = int(state.window_start + rule.window_size - now) + 1
            return False, max(1, retry_after)
            
    async def _apply_leaky_bucket(self, state: RateLimitState) -> Tuple[bool, Optional[int]]:
        """Apply leaky bucket algorithm"""
        now = time.time()
        rule = state.rule
        
        # Calculate requests that have "leaked" out
        if rule.leak_rate:
            time_passed = now - state.last_request
            leaked_requests = time_passed * rule.leak_rate
            state.current_count = max(0, state.current_count - int(leaked_requests))
            
        # Check if bucket has capacity
        if state.current_count < rule.limit:
            state.current_count += 1
            state.last_request = now
            return True, None
        else:
            # Calculate retry after based on leak rate
            retry_after = int(1.0 / rule.leak_rate) if rule.leak_rate else rule.window_size
            return False, retry_after
            
    async def _record_violation(self, rule: RateLimitRule, context: Dict[str, Any], state: RateLimitState):
        """Record rate limit violation"""
        now = time.time()
        state.violation_count += 1
        state.last_violation = now
        
        # Apply penalty if configured
        if rule.penalty_factor < 1.0:
            state.penalty_active = True
            state.penalty_until = now + rule.penalty_duration
            
        # Update adaptive scaling
        if rule.adaptive_scaling:
            # Decrease scale factor on violations
            min_scale, max_scale = rule.scale_factor_range
            state.scale_factor = max(min_scale, state.scale_factor * 0.9)
            
        # Log violation details
        await self._log_violation(rule, context, state)
        
    async def _log_violation(self, rule: RateLimitRule, context: Dict[str, Any], state: RateLimitState):
        """Log rate limit violation for monitoring"""
        violation_data = {
            "rule": rule.name,
            "scope": rule.scope.value,
            "identifier": context.get("identifier", "unknown"),
            "ip": context["ip"],
            "user_agent": context.get("user_agent", ""),
            "path": context["path"],
            "method": context["method"],
            "current_count": state.current_count,
            "effective_limit": state.get_effective_limit(),
            "violation_count": state.violation_count,
            "timestamp": time.time()
        }
        
        # Store in Redis for monitoring dashboard
        await self.redis.lpush(
            "rate_limit_violations",
            json.dumps(violation_data)
        )
        
        # Keep only last 1000 violations
        await self.redis.ltrim("rate_limit_violations", 0, 999)
        
    def _create_rate_limit_response(self, message: str, status_code: int, 
                                  retry_after: Optional[int] = None,
                                  rule: Optional[RateLimitRule] = None,
                                  state: Optional[RateLimitState] = None) -> JSONResponse:
        """Create rate limit error response"""
        headers = {}
        
        if retry_after:
            headers["Retry-After"] = str(retry_after)
            
        if rule and state:
            headers["X-RateLimit-Limit"] = str(state.get_effective_limit())
            headers["X-RateLimit-Remaining"] = str(max(0, state.get_effective_limit() - state.current_count))
            headers["X-RateLimit-Reset"] = str(int(state.window_start + rule.window_size))
            headers["X-RateLimit-Rule"] = rule.name
            
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "retry_after": retry_after
            },
            headers=headers
        )
        
    async def _add_rate_limit_headers(self, response: Response, context: Dict[str, Any]):
        """Add rate limit headers to successful responses"""
        # Find the most restrictive active rule for this request
        most_restrictive = None
        lowest_remaining = float('inf')
        
        for rule in self.rules:
            if rule.matches_request(None, context.get("tenant_tier")):  # Simplified check
                key = self._create_rate_limit_key(rule, context)
                state = await self._get_rate_limit_state(key, rule)
                
                remaining = state.get_effective_limit() - state.current_count
                if remaining < lowest_remaining:
                    lowest_remaining = remaining
                    most_restrictive = (rule, state)
                    
        if most_restrictive:
            rule, state = most_restrictive
            response.headers["X-RateLimit-Limit"] = str(state.get_effective_limit())
            response.headers["X-RateLimit-Remaining"] = str(max(0, lowest_remaining))
            response.headers["X-RateLimit-Reset"] = str(int(state.window_start + rule.window_size))
            
    async def _update_adaptive_scaling(self, processing_time: float, context: Dict[str, Any]):
        """Update adaptive scaling based on system performance"""
        # Update scaling for adaptive rules based on processing time
        target_response_time = 0.5  # 500ms target
        
        for rule in self.rules:
            if not rule.adaptive_scaling:
                continue
                
            key = self._create_rate_limit_key(rule, context)
            state = await self._get_rate_limit_state(key, rule)
            
            # Adjust scale factor based on performance
            if processing_time > target_response_time * 2:
                # Slow response, decrease limit
                min_scale, max_scale = rule.scale_factor_range
                state.scale_factor = max(min_scale, state.scale_factor * 0.95)
            elif processing_time < target_response_time * 0.5:
                # Fast response, can increase limit
                min_scale, max_scale = rule.scale_factor_range
                state.scale_factor = min(max_scale, state.scale_factor * 1.02)
                
            await self._save_rate_limit_state(state)
            
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        now = time.time()
        uptime = now - (self.last_stats_update or now)
        
        # Get recent violations
        violations = await self.redis.lrange("rate_limit_violations", 0, 99)
        recent_violations = []
        
        for violation in violations:
            try:
                data = json.loads(violation.decode())
                if now - data["timestamp"] < 3600:  # Last hour
                    recent_violations.append(data)
            except:
                continue
                
        # Calculate rates
        request_rate = self.request_counter / max(uptime, 1)
        rejection_rate = self.rejection_counter / max(uptime, 1)
        success_rate = (self.request_counter - self.rejection_counter) / max(self.request_counter, 1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_counter,
            "total_rejections": self.rejection_counter,
            "request_rate": request_rate,
            "rejection_rate": rejection_rate,
            "success_rate": success_rate,
            "recent_violations": len(recent_violations),
            "global_circuit_breaker": self.global_circuit_breaker,
            "active_rules": len(self.rules)
        }
        
    async def set_global_circuit_breaker(self, enabled: bool, duration: Optional[int] = None):
        """Enable/disable global circuit breaker"""
        self.global_circuit_breaker = enabled
        
        if enabled and duration:
            # Auto-disable after duration
            await asyncio.sleep(duration)
            self.global_circuit_breaker = False
            
    def add_rule(self, rule: RateLimitRule):
        """Add custom rate limiting rule"""
        self.rules.append(rule)
        logger.info(f"Added rate limiting rule: {rule.name}")
        
    def remove_rule(self, rule_name: str):
        """Remove rate limiting rule"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"Removed rate limiting rule: {rule_name}")
        
    async def clear_rate_limit(self, identifier: str, rule_name: Optional[str] = None):
        """Clear rate limit for specific identifier"""
        if rule_name:
            # Clear specific rule
            rule = next((r for r in self.rules if r.name == rule_name), None)
            if rule:
                key = f"rate_limit:{rule_name}:{identifier}"
                await self.redis.delete(key)
        else:
            # Clear all rules for identifier
            pattern = f"rate_limit:*:{identifier}"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                
        logger.info(f"Cleared rate limits for {identifier} (rule: {rule_name or 'all'})")


# Factory function
def create_advanced_rate_limiter(redis_client: redis.Redis, config: Dict[str, Any]) -> AdvancedRateLimiter:
    """Create and configure advanced rate limiter"""
    return AdvancedRateLimiter(redis_client, config)