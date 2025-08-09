"""
Advanced rate limiting middleware with Redis backend
Implements multiple rate limiting strategies and adaptive protection
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import redis.asyncio as redis

from ..domain.exceptions import RateLimitExceeded


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class RateLimitRule:
    """Rate limiting rule configuration"""
    
    def __init__(
        self,
        name: str,
        limit: int,
        window_seconds: int,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        burst_limit: Optional[int] = None,
        penalty_multiplier: float = 2.0,
        whitelist_ips: Optional[List[str]] = None,
        blacklist_ips: Optional[List[str]] = None
    ):
        self.name = name
        self.limit = limit
        self.window_seconds = window_seconds
        self.strategy = strategy
        self.burst_limit = burst_limit or limit * 2
        self.penalty_multiplier = penalty_multiplier
        self.whitelist_ips = whitelist_ips or []
        self.blacklist_ips = blacklist_ips or []


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on system load and threats"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
        # Default rate limiting rules
        self.rules = {
            "global": RateLimitRule("global", 1000, 60),  # 1000 requests per minute globally
            "per_ip": RateLimitRule("per_ip", 100, 60),   # 100 requests per minute per IP
            "auth": RateLimitRule("auth", 5, 300),        # 5 auth attempts per 5 minutes
            "api_key": RateLimitRule("api_key", 500, 60), # 500 requests per minute per API key
            "admin": RateLimitRule("admin", 50, 60),      # 50 admin requests per minute
            "upload": RateLimitRule("upload", 10, 300),   # 10 uploads per 5 minutes
        }
        
        # Threat indicators that trigger stricter limits
        self.threat_indicators = {
            "brute_force": {"threshold": 10, "penalty": 5.0},
            "scraping": {"threshold": 200, "penalty": 3.0},
            "ddos": {"threshold": 500, "penalty": 10.0}
        }
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule_name: str,
        request: Request
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be rate limited
        Returns (is_allowed, rate_limit_info)
        """
        rule = self.rules.get(rule_name)
        if not rule:
            return True, {}
        
        # Check whitelist/blacklist
        client_ip = self._get_client_ip(request)
        if client_ip in rule.blacklist_ips:
            return False, {"reason": "blacklisted_ip", "client_ip": client_ip}
        
        if client_ip in rule.whitelist_ips:
            return True, {"reason": "whitelisted_ip", "client_ip": client_ip}
        
        # Apply adaptive adjustments
        adjusted_rule = await self._apply_adaptive_adjustments(rule, identifier, request)
        
        # Execute rate limiting based on strategy
        if adjusted_rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(identifier, adjusted_rule)
        elif adjusted_rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(identifier, adjusted_rule)
        elif adjusted_rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(identifier, adjusted_rule)
        else:  # ADAPTIVE
            return await self._adaptive_check(identifier, adjusted_rule, request)
    
    async def _sliding_window_check(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting"""
        now = time.time()
        window_start = now - rule.window_seconds
        
        key = f"rate_limit:sliding:{rule.name}:{identifier}"
        
        # Use Redis sorted set for sliding window
        async with self.redis_client.pipeline() as pipe:
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Set expiration
            pipe.expire(key, rule.window_seconds + 1)
            
            results = await pipe.execute()
        
        current_count = results[1]
        
        rate_limit_info = {
            "limit": rule.limit,
            "remaining": max(0, rule.limit - current_count - 1),
            "reset_time": int(now + rule.window_seconds),
            "window_seconds": rule.window_seconds,
            "strategy": "sliding_window"
        }
        
        if current_count >= rule.limit:
            # Log rate limit violation
            await self._log_rate_limit_violation(identifier, rule, current_count)
            return False, rate_limit_info
        
        return True, rate_limit_info
    
    async def _token_bucket_check(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""
        now = time.time()
        key = f"rate_limit:bucket:{rule.name}:{identifier}"
        
        # Get current bucket state
        bucket_data = await self.redis_client.hgetall(key)
        
        if bucket_data:
            tokens = float(bucket_data.get(b'tokens', rule.limit))
            last_refill = float(bucket_data.get(b'last_refill', now))
        else:
            tokens = rule.limit
            last_refill = now
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = (time_elapsed / rule.window_seconds) * rule.limit
        tokens = min(rule.limit, tokens + tokens_to_add)
        
        rate_limit_info = {
            "limit": rule.limit,
            "remaining": int(tokens - 1) if tokens >= 1 else 0,
            "reset_time": int(now + (rule.window_seconds * (1 - tokens / rule.limit))),
            "window_seconds": rule.window_seconds,
            "strategy": "token_bucket"
        }
        
        if tokens >= 1:
            # Consume token
            tokens -= 1
            
            # Update bucket state
            await self.redis_client.hset(key, mapping={
                "tokens": str(tokens),
                "last_refill": str(now)
            })
            await self.redis_client.expire(key, rule.window_seconds * 2)
            
            return True, rate_limit_info
        else:
            await self._log_rate_limit_violation(identifier, rule, 0)
            return False, rate_limit_info
    
    async def _fixed_window_check(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""
        now = time.time()
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        
        key = f"rate_limit:fixed:{rule.name}:{identifier}:{window_start}"
        
        # Increment counter
        current_count = await self.redis_client.incr(key)
        if current_count == 1:
            await self.redis_client.expire(key, rule.window_seconds)
        
        rate_limit_info = {
            "limit": rule.limit,
            "remaining": max(0, rule.limit - current_count),
            "reset_time": int(window_start + rule.window_seconds),
            "window_seconds": rule.window_seconds,
            "strategy": "fixed_window"
        }
        
        if current_count > rule.limit:
            await self._log_rate_limit_violation(identifier, rule, current_count)
            return False, rate_limit_info
        
        return True, rate_limit_info
    
    async def _adaptive_check(
        self,
        identifier: str,
        rule: RateLimitRule,
        request: Request
    ) -> Tuple[bool, Dict[str, Any]]:
        """Adaptive rate limiting based on system conditions"""
        # Get system load metrics
        system_load = await self._get_system_load()
        
        # Adjust limits based on load
        if system_load > 0.8:  # High load
            adjusted_limit = int(rule.limit * 0.5)
        elif system_load > 0.6:  # Medium load
            adjusted_limit = int(rule.limit * 0.7)
        else:  # Normal load
            adjusted_limit = rule.limit
        
        # Create adjusted rule
        adjusted_rule = RateLimitRule(
            rule.name,
            adjusted_limit,
            rule.window_seconds,
            RateLimitStrategy.SLIDING_WINDOW
        )
        
        return await self._sliding_window_check(identifier, adjusted_rule)
    
    async def _apply_adaptive_adjustments(
        self,
        rule: RateLimitRule,
        identifier: str,
        request: Request
    ) -> RateLimitRule:
        """Apply adaptive adjustments based on threat detection"""
        client_ip = self._get_client_ip(request)
        
        # Check for threat indicators
        threat_level = await self._assess_threat_level(client_ip, identifier)
        
        if threat_level > 0:
            # Apply penalty
            penalty_factor = 1.0 + (threat_level * rule.penalty_multiplier)
            adjusted_limit = max(1, int(rule.limit / penalty_factor))
            
            return RateLimitRule(
                rule.name,
                adjusted_limit,
                rule.window_seconds,
                rule.strategy,
                rule.burst_limit,
                rule.penalty_multiplier
            )
        
        return rule
    
    async def _assess_threat_level(self, client_ip: str, identifier: str) -> float:
        """Assess threat level for adaptive rate limiting"""
        threat_level = 0.0
        
        # Check for brute force patterns
        brute_force_key = f"threat:brute_force:{client_ip}"
        brute_force_count = await self.redis_client.get(brute_force_key)
        if brute_force_count and int(brute_force_count) > self.threat_indicators["brute_force"]["threshold"]:
            threat_level += 1.0
        
        # Check for scraping patterns
        scraping_key = f"threat:scraping:{client_ip}"
        scraping_count = await self.redis_client.get(scraping_key)
        if scraping_count and int(scraping_count) > self.threat_indicators["scraping"]["threshold"]:
            threat_level += 0.5
        
        # Check global threat level
        global_threat = await self.redis_client.get("global_threat_level")
        if global_threat:
            threat_level += float(global_threat)
        
        return min(threat_level, 3.0)  # Cap at 3.0
    
    async def _get_system_load(self) -> float:
        """Get current system load (placeholder)"""
        # In a real implementation, this would check:
        # - CPU usage
        # - Memory usage
        # - Request queue length
        # - Database connection pool status
        # For now, return a simulated value
        load_key = "system_load"
        load = await self.redis_client.get(load_key)
        return float(load) if load else 0.1
    
    async def _log_rate_limit_violation(
        self,
        identifier: str,
        rule: RateLimitRule,
        current_count: int
    ):
        """Log rate limit violations for monitoring"""
        violation_data = {
            "identifier": identifier,
            "rule": rule.name,
            "limit": rule.limit,
            "current_count": current_count,
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": rule.strategy.value
        }
        
        # Store violation
        violation_key = f"rate_limit_violations:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis_client.lpush(violation_key, json.dumps(violation_data))
        await self.redis_client.expire(violation_key, 86400 * 7)  # Keep for 7 days
        
        # Update threat indicators
        await self._update_threat_indicators(identifier, rule.name)
    
    async def _update_threat_indicators(self, identifier: str, rule_name: str):
        """Update threat indicators based on violations"""
        if rule_name == "auth":
            # Increment brute force counter
            key = f"threat:brute_force:{identifier}"
            count = await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # 1 hour
        elif rule_name in ["per_ip", "global"]:
            # Increment scraping counter
            key = f"threat:scraping:{identifier}"
            count = await self.redis_client.incr(key)
            await self.redis_client.expire(key, 1800)  # 30 minutes
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {
            "violations_today": 0,
            "top_violators": [],
            "rule_stats": {},
            "threat_levels": {}
        }
        
        # Get violations for today
        today = datetime.utcnow().strftime('%Y-%m-%d')
        violation_key = f"rate_limit_violations:{today}"
        violations = await self.redis_client.lrange(violation_key, 0, -1)
        
        stats["violations_today"] = len(violations)
        
        # Analyze violations
        violator_counts = {}
        rule_counts = {}
        
        for violation_str in violations:
            try:
                violation = json.loads(violation_str)
                identifier = violation["identifier"]
                rule = violation["rule"]
                
                violator_counts[identifier] = violator_counts.get(identifier, 0) + 1
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            except:
                continue
        
        # Top violators
        stats["top_violators"] = sorted(
            violator_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats["rule_stats"] = rule_counts
        
        return stats


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.rate_limiter = AdaptiveRateLimiter(redis_client)
        
        # Endpoint-specific rate limiting rules
        self.endpoint_rules = {
            "/auth/token": "auth",
            "/auth/login": "auth",
            "/auth/register": "auth",
            "/admin": "admin",
            "/v1/embeddings": "api_key",
            "/upload": "upload"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Determine rate limiting rules to apply
        rules_to_check = ["global", "per_ip"]  # Always check global and per-IP limits
        
        # Add endpoint-specific rules
        for endpoint, rule_name in self.endpoint_rules.items():
            if request.url.path.startswith(endpoint):
                rules_to_check.append(rule_name)
                break
        
        # Check API key rate limiting
        api_key = request.headers.get("x-api-key") or request.headers.get("authorization")
        if api_key and not request.url.path.startswith("/auth"):
            rules_to_check.append("api_key")
        
        # Get identifiers
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        
        identifiers = {
            "global": "global",
            "per_ip": client_ip,
            "auth": client_ip,
            "api_key": api_key or client_ip,
            "admin": user_id or client_ip,
            "upload": user_id or client_ip
        }
        
        # Check all applicable rate limits
        rate_limit_info = {}
        
        for rule_name in rules_to_check:
            identifier = identifiers.get(rule_name, client_ip)
            is_allowed, info = await self.rate_limiter.check_rate_limit(
                identifier, rule_name, request
            )
            
            if not is_allowed:
                # Rate limit exceeded
                return self._create_rate_limit_response(info, rule_name)
            
            # Store the most restrictive rate limit info
            if not rate_limit_info or info.get("remaining", float('inf')) < rate_limit_info.get("remaining", float('inf')):
                rate_limit_info = info
                rate_limit_info["rule"] = rule_name
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if rate_limit_info:
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info.get("limit", "unknown"))
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.get("remaining", "unknown"))
            response.headers["X-RateLimit-Reset"] = str(rate_limit_info.get("reset_time", "unknown"))
            response.headers["X-RateLimit-Rule"] = rate_limit_info.get("rule", "unknown")
        
        return response
    
    def _create_rate_limit_response(self, rate_limit_info: Dict[str, Any], rule_name: str) -> JSONResponse:
        """Create rate limit exceeded response"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Rule: {rule_name}",
                "limit": rate_limit_info.get("limit"),
                "remaining": 0,
                "reset_time": rate_limit_info.get("reset_time"),
                "retry_after": rate_limit_info.get("window_seconds", 60)
            },
            headers={
                "X-RateLimit-Limit": str(rate_limit_info.get("limit", "unknown")),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_limit_info.get("reset_time", "unknown")),
                "X-RateLimit-Rule": rule_name,
                "Retry-After": str(rate_limit_info.get("window_seconds", 60))
            }
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request state"""
        try:
            user = getattr(request.state, "user", None)
            return str(user.id) if user else None
        except:
            return None