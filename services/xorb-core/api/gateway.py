#!/usr/bin/env python3
"""
Hardened API Gateway for XORB with Circuit Breakers and Rate Limiting
Enterprise-grade security and reliability features
"""

import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import logging
from collections import defaultdict, deque
import ipaddress

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import jwt

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimitType(str, Enum):
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitoring_window: int = 300  # 5 minutes


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 20


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_jwt_auth: bool = True
    enable_api_keys: bool = True
    enable_ip_whitelist: bool = False
    enable_ddos_protection: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    blocked_ips: Set[str] = field(default_factory=set)
    whitelisted_ips: Set[str] = field(default_factory=set)


class CircuitBreaker:
    """Advanced circuit breaker implementation"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.request_history = deque(maxlen=1000)

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
            else:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")

        try:
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            response_time = time.time() - start_time

            await self._record_success(response_time)
            return result

        except Exception as e:
            await self._record_failure(str(e))
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True

        return (time.time() - self.last_failure_time) >= self.config.timeout_seconds

    async def _record_success(self, response_time: float):
        """Record successful request"""
        self.request_history.append({
            "timestamp": time.time(),
            "success": True,
            "response_time": response_time
        })

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

        self.last_success_time = time.time()

    async def _record_failure(self, error: str):
        """Record failed request"""
        self.request_history.append({
            "timestamp": time.time(),
            "success": False,
            "error": error
        })

        self.failure_count += 1
        self.last_failure_time = time.time()

        if (self.state == CircuitBreakerState.CLOSED and
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        recent_requests = [
            r for r in self.request_history
            if time.time() - r["timestamp"] <= 300  # Last 5 minutes
        ]

        success_count = sum(1 for r in recent_requests if r["success"])
        total_count = len(recent_requests)

        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "total_requests": total_count,
            "last_failure": self.last_failure_time,
            "last_success": self.last_success_time
        }


class RateLimiter:
    """Advanced rate limiter with multiple time windows"""

    def __init__(self, config: RateLimitConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        self.local_cache = defaultdict(lambda: defaultdict(deque))
        self.cache_cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()

    async def is_allowed(self, identifier: str, endpoint: str = "default") -> bool:
        """Check if request is allowed under rate limits"""

        current_time = time.time()

        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cache_cleanup_interval:
            await self._cleanup_cache()

        # Check each time window
        limits = {
            RateLimitType.PER_SECOND: (self.config.requests_per_second, 1),
            RateLimitType.PER_MINUTE: (self.config.requests_per_minute, 60),
            RateLimitType.PER_HOUR: (self.config.requests_per_hour, 3600),
            RateLimitType.PER_DAY: (self.config.requests_per_day, 86400)
        }

        for limit_type, (max_requests, window_seconds) in limits.items():
            key = f"{identifier}:{endpoint}:{limit_type.value}"

            if self.redis_client:
                allowed = await self._check_redis_limit(key, max_requests, window_seconds)
            else:
                allowed = await self._check_local_limit(key, max_requests, window_seconds, current_time)

            if not allowed:
                logger.warning(f"Rate limit exceeded for {identifier} on {endpoint} ({limit_type.value})")
                return False

        return True

    async def _check_redis_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check rate limit using Redis sliding window"""
        try:
            pipe = self.redis_client.pipeline()
            current_time = time.time()
            window_start = current_time - window_seconds

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(key, window_seconds + 1)

            results = await pipe.execute()
            current_count = results[1]

            return current_count < max_requests

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True  # Fail open

    async def _check_local_limit(self, key: str, max_requests: int,
                                window_seconds: int, current_time: float) -> bool:
        """Check rate limit using local memory"""

        # Get or create request queue for this key
        request_queue = self.local_cache[key]["requests"]

        # Remove requests outside the window
        while request_queue and current_time - request_queue[0] > window_seconds:
            request_queue.popleft()

        # Check if limit exceeded
        if len(request_queue) >= max_requests:
            return False

        # Add current request
        request_queue.append(current_time)
        return True

    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()

        for key in list(self.local_cache.keys()):
            request_queue = self.local_cache[key]["requests"]

            # Remove old requests
            while request_queue and current_time - request_queue[0] > 86400:  # 1 day
                request_queue.popleft()

            # Remove empty queues
            if not request_queue:
                del self.local_cache[key]

        self.last_cleanup = current_time

    async def get_limits(self, identifier: str, endpoint: str = "default") -> Dict[str, Any]:
        """Get current rate limit status"""
        current_time = time.time()
        status = {}

        limits = {
            "per_second": (self.config.requests_per_second, 1),
            "per_minute": (self.config.requests_per_minute, 60),
            "per_hour": (self.config.requests_per_hour, 3600),
            "per_day": (self.config.requests_per_day, 86400)
        }

        for limit_name, (max_requests, window_seconds) in limits.items():
            key = f"{identifier}:{endpoint}:{limit_name}"

            if self.redis_client:
                try:
                    window_start = current_time - window_seconds
                    count = await self.redis_client.zcount(key, window_start, current_time)
                except:
                    count = 0
            else:
                request_queue = self.local_cache[key]["requests"]
                count = len([r for r in request_queue if current_time - r <= window_seconds])

            status[limit_name] = {
                "limit": max_requests,
                "used": count,
                "remaining": max(0, max_requests - count),
                "reset_time": current_time + window_seconds
            }

        return status


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""

    def __init__(self, app: FastAPI, config: SecurityConfig,
                 rate_limiter: RateLimiter, circuit_breaker: CircuitBreaker):
        super().__init__(app)
        self.config = config
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.security = HTTPBearer(auto_error=False)

    async def dispatch(self, request: Request, call_next):
        """Main security middleware dispatch"""

        try:
            # Security checks
            await self._check_ip_security(request)
            await self._check_request_size(request)
            await self._check_ddos_protection(request)

            # Rate limiting
            client_id = await self._get_client_identifier(request)
            endpoint = f"{request.method}:{request.url.path}"

            if not await self.rate_limiter.is_allowed(client_id, endpoint):
                return Response(
                    content=json.dumps({"error": "Rate limit exceeded"}),
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": "60"}
                )

            # Authentication
            await self._authenticate_request(request)

            # Circuit breaker protection
            response = await self.circuit_breaker.call(call_next, request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["X-Rate-Limit-Status"] = json.dumps(
                await self.rate_limiter.get_limits(client_id, endpoint)
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            raise HTTPException(status_code=500, detail="Internal security error")

    async def _check_ip_security(self, request: Request):
        """Check IP-based security rules"""
        client_ip = self._get_client_ip(request)

        # Check blocked IPs
        if client_ip in self.config.blocked_ips:
            raise HTTPException(status_code=403, detail="IP address blocked")

        # Check IP whitelist (if enabled)
        if (self.config.enable_ip_whitelist and
            self.config.whitelisted_ips and
            client_ip not in self.config.whitelisted_ips):
            raise HTTPException(status_code=403, detail="IP address not whitelisted")

    async def _check_request_size(self, request: Request):
        """Check request size limits"""
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.config.max_request_size:
                    raise HTTPException(status_code=413, detail="Request too large")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid content-length header")

    async def _check_ddos_protection(self, request: Request):
        """Basic DDoS protection checks"""
        if not self.config.enable_ddos_protection:
            return

        # Check for suspicious patterns
        user_agent = request.headers.get("user-agent", "")

        # Block requests without user agent
        if not user_agent.strip():
            raise HTTPException(status_code=403, detail="User agent required")

        # Block known malicious user agents
        malicious_patterns = ["sqlmap", "nikto", "dirb", "gobuster", "masscan"]
        if any(pattern in user_agent.lower() for pattern in malicious_patterns):
            raise HTTPException(status_code=403, detail="Malicious user agent detected")

    async def _authenticate_request(self, request: Request):
        """Authenticate request using JWT or API key"""

        # Skip authentication for health checks and public endpoints
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return

        auth_header = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")

        authenticated = False

        # Try JWT authentication
        if self.config.enable_jwt_auth and auth_header:
            try:
                credentials = await self.security(request)
                if credentials:
                    # Verify JWT token (simplified - use proper secret in production)
                    payload = jwt.decode(credentials.credentials,
                                       "your-secret-key",
                                       algorithms=["HS256"])
                    request.state.user = payload
                    authenticated = True
            except jwt.InvalidTokenError:
                pass

        # Try API key authentication
        if self.config.enable_api_keys and api_key and not authenticated:
            # Verify API key (simplified - use proper validation in production)
            if await self._validate_api_key(api_key):
                request.state.api_key = api_key
                authenticated = True

        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )

    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key (simplified implementation)"""
        # In production, validate against database or external service
        valid_keys = ["xorb-api-key-2025", "test-api-key"]
        return api_key in valid_keys

    async def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""

        # Try to get authenticated user ID
        if hasattr(request.state, "user") and "sub" in request.state.user:
            return f"user:{request.state.user['sub']}"

        # Try to get API key
        if hasattr(request.state, "api_key"):
            return f"api_key:{hashlib.sha256(request.state.api_key.encode()).hexdigest()[:8]}"

        # Fall back to IP address
        return f"ip:{self._get_client_ip(request)}"

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies"""

        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        forwarded = request.headers.get("forwarded")
        if forwarded:
            # Parse forwarded header (simplified)
            for part in forwarded.split(";"):
                if part.strip().startswith("for="):
                    return part.split("=")[1].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to client host
        return request.client.host if request.client else "unknown"


class HardenedAPIGateway:
    """Main hardened API gateway class"""

    def __init__(self, app: FastAPI, redis_url: Optional[str] = None):
        self.app = app
        self.redis_client = None

        # Initialize Redis if URL provided
        if redis_url:
            self.redis_client = redis.from_url(redis_url)

        # Configuration
        self.circuit_breaker_config = CircuitBreakerConfig()
        self.rate_limit_config = RateLimitConfig()
        self.security_config = SecurityConfig()

        # Components
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.rate_limiter = RateLimiter(self.rate_limit_config, self.redis_client)

        # Add middleware
        self.security_middleware = SecurityMiddleware(
            app, self.security_config, self.rate_limiter, self.circuit_breaker
        )
        app.add_middleware(SecurityMiddleware,
                          config=self.security_config,
                          rate_limiter=self.rate_limiter,
                          circuit_breaker=self.circuit_breaker)

    def configure_circuit_breaker(self, **kwargs):
        """Configure circuit breaker settings"""
        for key, value in kwargs.items():
            if hasattr(self.circuit_breaker_config, key):
                setattr(self.circuit_breaker_config, key, value)

    def configure_rate_limiting(self, **kwargs):
        """Configure rate limiting settings"""
        for key, value in kwargs.items():
            if hasattr(self.rate_limit_config, key):
                setattr(self.rate_limit_config, key, value)

    def configure_security(self, **kwargs):
        """Configure security settings"""
        for key, value in kwargs.items():
            if hasattr(self.security_config, key):
                setattr(self.security_config, key, value)

    async def get_gateway_stats(self) -> Dict[str, Any]:
        """Get comprehensive gateway statistics"""
        return {
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "security_config": {
                "jwt_auth_enabled": self.security_config.enable_jwt_auth,
                "api_keys_enabled": self.security_config.enable_api_keys,
                "ip_whitelist_enabled": self.security_config.enable_ip_whitelist,
                "ddos_protection_enabled": self.security_config.enable_ddos_protection,
                "blocked_ips_count": len(self.security_config.blocked_ips),
                "whitelisted_ips_count": len(self.security_config.whitelisted_ips)
            },
            "rate_limiting": {
                "requests_per_second": self.rate_limit_config.requests_per_second,
                "requests_per_minute": self.rate_limit_config.requests_per_minute,
                "requests_per_hour": self.rate_limit_config.requests_per_hour,
                "requests_per_day": self.rate_limit_config.requests_per_day,
                "burst_allowance": self.rate_limit_config.burst_allowance
            },
            "redis_connected": self.redis_client is not None
        }

    async def block_ip(self, ip: str):
        """Block an IP address"""
        try:
            # Validate IP address
            ipaddress.ip_address(ip)
            self.security_config.blocked_ips.add(ip)
            logger.info(f"Blocked IP address: {ip}")
        except ValueError:
            raise ValueError(f"Invalid IP address: {ip}")

    async def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.security_config.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP address: {ip}")

    async def whitelist_ip(self, ip: str):
        """Add IP to whitelist"""
        try:
            ipaddress.ip_address(ip)
            self.security_config.whitelisted_ips.add(ip)
            logger.info(f"Whitelisted IP address: {ip}")
        except ValueError:
            raise ValueError(f"Invalid IP address: {ip}")

    async def remove_from_whitelist(self, ip: str):
        """Remove IP from whitelist"""
        self.security_config.whitelisted_ips.discard(ip)
        logger.info(f"Removed IP from whitelist: {ip}")

    async def health_check(self) -> Dict[str, Any]:
        """Gateway health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "components": {
                "rate_limiter": "healthy",
                "circuit_breaker": "healthy",
                "security_middleware": "healthy"
            }
        }

        # Check Redis connection
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["components"]["redis"] = "healthy"
            except Exception:
                health["components"]["redis"] = "unhealthy"
                health["status"] = "degraded"

        return health
