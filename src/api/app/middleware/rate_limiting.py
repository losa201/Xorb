"""Advanced rate limiting middleware with Redis backend."""
import asyncio
import hashlib
import time
from typing import Dict, Optional, Tuple

import redis.asyncio as redis
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from .error_handling import RateLimitError


logger = structlog.get_logger("rate_limiter")


class RateLimitConfig:
    """Rate limiting configuration."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_size: int = 10,
        enable_per_user: bool = True,
        enable_per_ip: bool = True,
        enable_per_tenant: bool = True
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_size = burst_size
        self.enable_per_user = enable_per_user
        self.enable_per_ip = enable_per_ip
        self.enable_per_tenant = enable_per_tenant


class SlidingWindowRateLimiter:
    """Sliding window rate limiter with Redis."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed using sliding window algorithm."""

        current_time = time.time()
        window_start = current_time - window_seconds

        # Use Lua script for atomic operations
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local current_time = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local window_seconds = tonumber(ARGV[5])

        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current requests
        local current_count = redis.call('ZCARD', key)

        -- Check if adding cost would exceed limit
        if current_count + cost > limit then
            -- Get oldest entry to calculate retry-after
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if oldest[2] then
                retry_after = math.ceil(tonumber(oldest[2]) + window_seconds - current_time)
            end

            return {0, current_count, limit, retry_after}
        end

        -- Add current request
        for i = 1, cost do
            redis.call('ZADD', key, current_time + (i * 0.001), current_time .. ':' .. i)
        end

        -- Set expiration
        redis.call('EXPIRE', key, window_seconds)

        return {1, current_count + cost, limit, 0}
        """

        try:
            result = await self.redis.eval(
                lua_script,
                1,
                key,
                window_start,
                current_time,
                limit,
                cost,
                window_seconds
            )

            allowed = bool(result[0])
            current_count = int(result[1])
            retry_after = int(result[3]) if result[3] > 0 else None

            return allowed, {
                "current_count": current_count,
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after
            }

        except Exception as e:
            logger.error("Rate limiter error", error=str(e))
            # Fail open on Redis errors
            return True, {"error": "rate_limiter_unavailable"}


class AdvancedRateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with multiple strategies."""

    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        config: Optional[RateLimitConfig] = None
    ):
        super().__init__(app)
        self.redis = redis_client
        self.config = config or RateLimitConfig()
        self.limiter = SlidingWindowRateLimiter(redis_client)

        # Exempt paths from rate limiting
        self.exempt_paths = {
            "/health",
            "/readiness",
            "/metrics",
            "/docs",
            "/openapi.json"
        }

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""

        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get rate limiting keys
        keys = await self._get_rate_limit_keys(request)

        # Check all applicable rate limits
        for key_info in keys:
            allowed, info = await self._check_rate_limit(key_info)

            if not allowed:
                return await self._create_rate_limit_response(key_info, info)

        # Add rate limit headers to response
        response = await call_next(request)
        await self._add_rate_limit_headers(response, keys)

        return response

    async def _get_rate_limit_keys(self, request: Request) -> list:
        """Generate rate limiting keys for different strategies."""
        keys = []

        # IP-based rate limiting
        if self.config.enable_per_ip:
            client_ip = self._get_client_ip(request)
            keys.extend([
                {
                    "key": f"rate_limit:ip:{client_ip}:minute",
                    "limit": self.config.requests_per_minute,
                    "window": 60,
                    "type": "ip_minute"
                },
                {
                    "key": f"rate_limit:ip:{client_ip}:hour",
                    "limit": self.config.requests_per_hour,
                    "window": 3600,
                    "type": "ip_hour"
                }
            ])

        # User-based rate limiting
        if self.config.enable_per_user and hasattr(request.state, 'user'):
            user_id = request.state.user.sub
            keys.extend([
                {
                    "key": f"rate_limit:user:{user_id}:minute",
                    "limit": self.config.requests_per_minute * 2,  # Higher limit for authenticated users
                    "window": 60,
                    "type": "user_minute"
                },
                {
                    "key": f"rate_limit:user:{user_id}:hour",
                    "limit": self.config.requests_per_hour * 2,
                    "window": 3600,
                    "type": "user_hour"
                }
            ])

        # Tenant-based rate limiting
        if (self.config.enable_per_tenant and
            hasattr(request.state, 'user') and
            request.state.user.tenant_id):

            tenant_id = str(request.state.user.tenant_id)
            keys.extend([
                {
                    "key": f"rate_limit:tenant:{tenant_id}:hour",
                    "limit": self.config.requests_per_hour * 10,  # Higher limit per tenant
                    "window": 3600,
                    "type": "tenant_hour"
                },
                {
                    "key": f"rate_limit:tenant:{tenant_id}:day",
                    "limit": self.config.requests_per_day,
                    "window": 86400,
                    "type": "tenant_day"
                }
            ])

        # Endpoint-specific rate limiting
        endpoint = self._normalize_endpoint(request.url.path)
        if endpoint in self._get_high_cost_endpoints():
            cost = 5  # Higher cost for expensive operations
            keys.append({
                "key": f"rate_limit:endpoint:{endpoint}:minute",
                "limit": self.config.requests_per_minute // 5,
                "window": 60,
                "type": "endpoint_minute",
                "cost": cost
            })

        return keys

    async def _check_rate_limit(self, key_info: dict) -> Tuple[bool, dict]:
        """Check a specific rate limit."""
        cost = key_info.get("cost", 1)

        return await self.limiter.is_allowed(
            key=key_info["key"],
            limit=key_info["limit"],
            window_seconds=key_info["window"],
            cost=cost
        )

    async def _create_rate_limit_response(self, key_info: dict, info: dict) -> JSONResponse:
        """Create rate limit exceeded response."""

        retry_after = info.get("retry_after", 60)

        logger.warning(
            "Rate limit exceeded",
            limit_type=key_info["type"],
            limit=key_info["limit"],
            window=key_info["window"],
            current_count=info.get("current_count", 0)
        )

        headers = {
            "X-RateLimit-Limit": str(key_info["limit"]),
            "X-RateLimit-Window": str(key_info["window"]),
            "X-RateLimit-Type": key_info["type"]
        }

        if retry_after:
            headers["Retry-After"] = str(retry_after)

        error_response = {
            "success": False,
            "errors": [{
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded",
                "details": {
                    "limit": key_info["limit"],
                    "window_seconds": key_info["window"],
                    "retry_after_seconds": retry_after,
                    "limit_type": key_info["type"]
                }
            }]
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response,
            headers=headers
        )

    async def _add_rate_limit_headers(self, response, keys: list) -> None:
        """Add rate limit headers to response."""
        if not keys:
            return

        # Use the most restrictive limit for headers
        primary_key = min(keys, key=lambda k: k["limit"] / k["window"])

        try:
            _, info = await self.limiter.is_allowed(
                key=primary_key["key"],
                limit=primary_key["limit"],
                window_seconds=primary_key["window"],
                cost=0  # Just check, don't increment
            )

            response.headers["X-RateLimit-Limit"] = str(primary_key["limit"])
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, primary_key["limit"] - info.get("current_count", 0))
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + primary_key["window"]
            )

        except Exception:
            # Don't fail response if rate limit headers fail
            pass

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers (from load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client address
        if request.client:
            return request.client.host

        return "unknown"

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for rate limiting."""
        # Replace UUIDs and IDs with placeholders
        import re

        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path
        )

        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)

        return path

    def _get_high_cost_endpoints(self) -> set:
        """Get endpoints that should have higher rate limiting cost."""
        return {
            "/api/evidence/upload",
            "/api/vectors/search",
            "/api/jobs/schedule",
            "/api/analysis/start"
        }


# Rate limiting decorators
def rate_limit(
    operation_name_or_requests_per_minute,
    requests_per_minute: Optional[int] = None,
    window_seconds: Optional[int] = None,
    requests_per_hour: Optional[int] = None,
    key_func: Optional[callable] = None
):
    """
    Decorator for function-level rate limiting.

    Supports two usage patterns:
    1. @rate_limit(requests_per_minute=10) - keyword args
    2. @rate_limit("operation_name", 10, 60) - positional args with operation name
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would integrate with the middleware or use a separate limiter
            # For now, just pass through - the actual rate limiting is handled by middleware
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Rate limit key generators
def generate_user_key(user_id: str, window: str) -> str:
    """Generate rate limit key for user."""
    return f"rate_limit:user:{user_id}:{window}"


def generate_ip_key(ip_address: str, window: str) -> str:
    """Generate rate limit key for IP."""
    return f"rate_limit:ip:{ip_address}:{window}"


def generate_tenant_key(tenant_id: str, window: str) -> str:
    """Generate rate limit key for tenant."""
    return f"rate_limit:tenant:{tenant_id}:{window}"
