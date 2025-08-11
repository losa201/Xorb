"""
Production API Middleware Stack for XORB Platform
Comprehensive middleware for security, performance, and observability
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
from ..infrastructure.redis_compatibility import get_redis_client
from urllib.parse import urlparse
import ipaddress

from ..services.monitoring_service import get_monitoring_service, record_api_request, create_alert, AlertSeverity

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add comprehensive security headers to all responses"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            # XSS Protection
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS and Transport Security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https: wss:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            ),
            
            # Server Information Hiding
            "Server": "XORB Security Platform",
            
            # Cache Control for sensitive endpoints
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Apply security headers
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        # Remove server version information
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Track requests with unique IDs and correlation"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Monitor API performance and record metrics"""
    
    def __init__(self, app, slow_request_threshold: float = 2.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Record metrics
            await record_api_request(
                method=request.method,
                endpoint=self._normalize_endpoint(request.url.path),
                status_code=response.status_code,
                duration_seconds=duration
            )
            
            # Alert on slow requests
            if duration > self.slow_request_threshold:
                await create_alert(
                    title="Slow API Request",
                    description=f"Request to {request.method} {request.url.path} took {duration:.2f}s",
                    severity=AlertSeverity.WARNING,
                    source_service="api",
                    metadata={
                        "method": request.method,
                        "endpoint": request.url.path,
                        "duration_seconds": duration,
                        "request_id": getattr(request.state, "request_id", "unknown")
                    }
                )
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            await record_api_request(
                method=request.method,
                endpoint=self._normalize_endpoint(request.url.path),
                status_code=500,
                duration_seconds=duration
            )
            
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics"""
        # Replace UUID patterns with placeholder
        import re
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        path = re.sub(r'/\d+', '/{id}', path)
        return path


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with Redis backend and tenant isolation"""
    
    def __init__(self, app, redis_url: str = "redis://localhost:6379", 
                 default_rate_limit: int = 60, default_window: int = 60):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis_client = None
        self.default_rate_limit = default_rate_limit
        self.default_window = default_window
        
        # Rate limit configurations by endpoint
        self.endpoint_limits = {
            "/api/v1/auth/login": {"limit": 5, "window": 300},  # 5 per 5 minutes
            "/api/v1/auth/register": {"limit": 3, "window": 3600},  # 3 per hour
            "/api/v1/ptaas/sessions": {"limit": 10, "window": 300},  # 10 scans per 5 minutes
            "/api/v1/intelligence/analyze": {"limit": 30, "window": 60},  # 30 per minute
            "/api/v1/vectors/search": {"limit": 100, "window": 60},  # 100 per minute
        }
        
        # In-memory fallback for rate limiting
        self.memory_cache = {}
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            try:
                self.redis_client = await aioredis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Rate limiting Redis connection established")
            except Exception as e:
                logger.warning(f"Rate limiting Redis connection failed: {e}")
                self.redis_client = None
    
    async def dispatch(self, request: Request, call_next):
        await self.init_redis()
        
        # Get client identifier
        client_id = await self._get_client_id(request)
        endpoint = self._normalize_endpoint(request.url.path)
        
        # Check rate limits
        if await self._is_rate_limited(client_id, endpoint, request):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": self._get_retry_after(endpoint)
                },
                headers={
                    "Retry-After": str(self._get_retry_after(endpoint)),
                    "X-RateLimit-Limit": str(self._get_rate_limit(endpoint)),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_id, endpoint)
        response.headers["X-RateLimit-Limit"] = str(self._get_rate_limit(endpoint))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self._get_window(endpoint)))
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get tenant ID from JWT token
        try:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # Decode without verification for rate limiting purposes
                payload = jwt.decode(token, options={"verify_signature": False})
                tenant_id = payload.get("tenant_id")
                if tenant_id:
                    return f"tenant:{tenant_id}"
        except:
            pass
        
        # Fallback to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_id: str, endpoint: str, request: Request) -> bool:
        """Check if client is rate limited"""
        rate_limit = self._get_rate_limit(endpoint)
        window = self._get_window(endpoint)
        
        if self.redis_client:
            return await self._redis_rate_limit_check(client_id, endpoint, rate_limit, window)
        else:
            return await self._memory_rate_limit_check(client_id, endpoint, rate_limit, window)
    
    async def _redis_rate_limit_check(self, client_id: str, endpoint: str, 
                                    rate_limit: int, window: int) -> bool:
        """Redis-based rate limiting"""
        try:
            key = f"rate_limit:{client_id}:{endpoint}"
            
            # Use sliding window log approach
            current_time = time.time()
            window_start = current_time - window
            
            # Remove old entries and count current requests
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            return current_requests >= rate_limit
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return False
    
    async def _memory_rate_limit_check(self, client_id: str, endpoint: str,
                                     rate_limit: int, window: int) -> bool:
        """Memory-based rate limiting fallback"""
        key = f"{client_id}:{endpoint}"
        current_time = time.time()
        
        if key not in self.memory_cache:
            self.memory_cache[key] = []
        
        # Remove old entries
        self.memory_cache[key] = [
            timestamp for timestamp in self.memory_cache[key]
            if current_time - timestamp < window
        ]
        
        # Check rate limit
        if len(self.memory_cache[key]) >= rate_limit:
            return True
        
        # Add current request
        self.memory_cache[key].append(current_time)
        return False
    
    async def _get_remaining_requests(self, client_id: str, endpoint: str) -> int:
        """Get remaining requests for client"""
        rate_limit = self._get_rate_limit(endpoint)
        window = self._get_window(endpoint)
        
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}:{endpoint}"
                current_time = time.time()
                window_start = current_time - window
                
                # Count current requests in window
                await self.redis_client.zremrangebyscore(key, 0, window_start)
                current_requests = await self.redis_client.zcard(key)
                
                return max(0, rate_limit - current_requests)
            except:
                pass
        
        # Fallback to memory cache
        key = f"{client_id}:{endpoint}"
        if key in self.memory_cache:
            current_time = time.time()
            valid_requests = [
                timestamp for timestamp in self.memory_cache[key]
                if current_time - timestamp < window
            ]
            return max(0, rate_limit - len(valid_requests))
        
        return rate_limit
    
    def _get_rate_limit(self, endpoint: str) -> int:
        """Get rate limit for endpoint"""
        return self.endpoint_limits.get(endpoint, {}).get("limit", self.default_rate_limit)
    
    def _get_window(self, endpoint: str) -> int:
        """Get rate limit window for endpoint"""
        return self.endpoint_limits.get(endpoint, {}).get("window", self.default_window)
    
    def _get_retry_after(self, endpoint: str) -> int:
        """Get retry after seconds"""
        return self._get_window(endpoint)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path"""
        import re
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        path = re.sub(r'/\d+', '/{id}', path)
        return path


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelisting and geoblocking middleware"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Default configurations
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.whitelist_ips = set(self.config.get("whitelist_ips", []))
        self.whitelist_cidrs = [ipaddress.ip_network(cidr) for cidr in self.config.get("whitelist_cidrs", [])]
        
        self.blacklist_enabled = self.config.get("blacklist_enabled", True)
        self.blacklist_ips = set(self.config.get("blacklist_ips", []))
        self.blacklist_cidrs = [ipaddress.ip_network(cidr) for cidr in self.config.get("blacklist_cidrs", [])]
        
        # Blocked countries (ISO 3166-1 alpha-2 codes)
        self.blocked_countries = set(self.config.get("blocked_countries", []))
        
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        
        try:
            ip_addr = ipaddress.ip_address(client_ip)
            
            # Check blacklist first
            if self.blacklist_enabled and self._is_blacklisted(ip_addr):
                await self._log_blocked_request(request, client_ip, "blacklisted")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied", "message": "Your IP address is not allowed"}
                )
            
            # Check whitelist if enabled
            if self.whitelist_enabled and not self._is_whitelisted(ip_addr):
                await self._log_blocked_request(request, client_ip, "not_whitelisted")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied", "message": "Your IP address is not whitelisted"}
                )
            
        except ValueError:
            # Invalid IP address
            await self._log_blocked_request(request, client_ip, "invalid_ip")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "message": "Invalid client IP"}
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_whitelisted(self, ip_addr: ipaddress.IPv4Address) -> bool:
        """Check if IP is whitelisted"""
        # Check exact IP matches
        if str(ip_addr) in self.whitelist_ips:
            return True
        
        # Check CIDR ranges
        for cidr in self.whitelist_cidrs:
            if ip_addr in cidr:
                return True
        
        return False
    
    def _is_blacklisted(self, ip_addr: ipaddress.IPv4Address) -> bool:
        """Check if IP is blacklisted"""
        # Check exact IP matches
        if str(ip_addr) in self.blacklist_ips:
            return True
        
        # Check CIDR ranges
        for cidr in self.blacklist_cidrs:
            if ip_addr in cidr:
                return True
        
        return False
    
    async def _log_blocked_request(self, request: Request, client_ip: str, reason: str):
        """Log blocked request attempt"""
        await create_alert(
            title="Blocked IP Access Attempt",
            description=f"Access attempt from {client_ip} blocked: {reason}",
            severity=AlertSeverity.WARNING,
            source_service="security",
            metadata={
                "client_ip": client_ip,
                "reason": reason,
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("User-Agent", "unknown")
            }
        )


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation and sanitization"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Request size limits
        self.max_request_size = self.config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
        self.max_header_size = self.config.get("max_header_size", 8192)  # 8KB
        
        # Content type restrictions
        self.allowed_content_types = set(self.config.get("allowed_content_types", [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        ]))
        
        # Malicious pattern detection
        self.malicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # XSS
            r"on\w+\s*=",  # Event handlers
            r"(?:union|select|insert|update|delete|drop|create|alter)\s+",  # SQL injection
            r"\.\./",  # Path traversal
            r"cmd\s*=",  # Command injection
            r"eval\s*\(",  # Code injection
        ]
        
    async def dispatch(self, request: Request, call_next):
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large", "message": f"Maximum size is {self.max_request_size} bytes"}
            )
        
        # Check header sizes
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > self.max_header_size:
            return JSONResponse(
                status_code=431,
                content={"error": "Request headers too large"}
            )
        
        # Validate content type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            if content_type and content_type not in self.allowed_content_types:
                return JSONResponse(
                    status_code=415,
                    content={"error": "Unsupported content type"}
                )
        
        # Check for malicious patterns in URL and headers
        if await self._contains_malicious_patterns(request):
            await create_alert(
                title="Malicious Request Detected",
                description=f"Potentially malicious request from {self._get_client_ip(request)}",
                severity=AlertSeverity.ERROR,
                source_service="security",
                metadata={
                    "client_ip": self._get_client_ip(request),
                    "endpoint": request.url.path,
                    "method": request.method,
                    "user_agent": request.headers.get("User-Agent", "unknown")
                }
            )
            
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "message": "Request contains invalid characters"}
            )
        
        return await call_next(request)
    
    async def _contains_malicious_patterns(self, request: Request) -> bool:
        """Check for malicious patterns in request"""
        import re
        
        # Check URL path and query parameters
        full_url = str(request.url)
        for pattern in self.malicious_patterns:
            if re.search(pattern, full_url, re.IGNORECASE):
                return True
        
        # Check headers
        for header_name, header_value in request.headers.items():
            for pattern in self.malicious_patterns:
                if re.search(pattern, header_value, re.IGNORECASE):
                    return True
        
        return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling and logging"""
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
        
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "HTTP Error",
                    "message": e.detail,
                    "status_code": e.status_code
                }
            )
            
        except Exception as e:
            # Log the error
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception(f"Unhandled error in request {request_id}: {str(e)}")
            
            # Create alert for server errors
            await create_alert(
                title="Unhandled Server Error",
                description=f"Unhandled exception in {request.method} {request.url.path}: {str(e)}",
                severity=AlertSeverity.ERROR,
                source_service="api",
                metadata={
                    "request_id": request_id,
                    "method": request.method,
                    "endpoint": request.url.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            # Return error response
            error_response = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
            
            if self.debug:
                error_response["debug"] = {
                    "type": type(e).__name__,
                    "message": str(e)
                }
            
            return JSONResponse(
                status_code=500,
                content=error_response
            )


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with enhanced security"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        self.allowed_origins = set(self.config.get("allowed_origins", ["*"]))
        self.allowed_methods = set(self.config.get("allowed_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"]))
        self.allowed_headers = set(self.config.get("allowed_headers", [
            "Authorization", "Content-Type", "X-Requested-With", "X-Request-ID"
        ]))
        self.allow_credentials = self.config.get("allow_credentials", True)
        self.max_age = self.config.get("max_age", 86400)  # 24 hours
        
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("Origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, origin)
            return response
        
        # Process actual request
        response = await call_next(request)
        self._add_cors_headers(response, origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: Optional[str]):
        """Add CORS headers to response"""
        if origin and (self._is_origin_allowed(origin) or "*" in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if origin in self.allowed_origins:
            return True
        
        # Check for wildcard domains
        for allowed_origin in self.allowed_origins:
            if allowed_origin.startswith("*."):
                domain = allowed_origin[2:]
                if origin.endswith(domain):
                    return True
        
        return False


# Middleware factory function
def create_middleware_stack(app, config: Dict[str, Any] = None):
    """Create complete middleware stack for production"""
    config = config or {}
    
    # Add middleware in reverse order (they are applied in reverse)
    middleware_stack = [
        # Error handling (outermost)
        (ErrorHandlingMiddleware, {"debug": config.get("debug", False)}),
        
        # Security headers
        (SecurityHeadersMiddleware, {}),
        
        # CORS
        (CORSMiddleware, config.get("cors", {})),
        
        # Request validation
        (RequestValidationMiddleware, config.get("validation", {})),
        
        # IP filtering
        (IPWhitelistMiddleware, config.get("ip_filtering", {})),
        
        # Rate limiting
        (RateLimitingMiddleware, {
            "redis_url": config.get("redis_url", "redis://localhost:6379"),
            "default_rate_limit": config.get("default_rate_limit", 60),
            "default_window": config.get("default_window", 60)
        }),
        
        # Performance monitoring
        (PerformanceMonitoringMiddleware, {
            "slow_request_threshold": config.get("slow_request_threshold", 2.0)
        }),
        
        # Request tracking (innermost)
        (RequestTrackingMiddleware, {}),
    ]
    
    # Apply middleware to app
    for middleware_class, middleware_config in middleware_stack:
        app.add_middleware(middleware_class, **middleware_config)
    
    return app