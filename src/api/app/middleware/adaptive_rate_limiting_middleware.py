"""
Production adaptive rate limiting middleware with shadow mode and emergency controls.

This middleware integrates:
- Adaptive rate limiter with multiple algorithms
- Hierarchical policy system with tenant/role/endpoint overrides
- Comprehensive observability and alerting
- Shadow mode for safe rollouts
- Emergency kill-switch functionality
- Integration with existing auth and tenant context
"""

import asyncio
import time
from typing import Optional, Dict, Any, Set
from uuid import uuid4

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import structlog

from ..core.adaptive_rate_limiter import (
    AdaptiveRateLimiter, RateLimitPolicy, PolicyScope, LimitAlgorithm, 
    EmergencyRateLimiter, ReputationLevel
)
from ..core.rate_limit_policies import (
    HierarchicalPolicyManager, RateLimitContext, PolicyOverride, PolicyType
)
from ..core.rate_limit_observability import (
    RateLimitObservability, DecisionOutcome, AlertSeverity
)
from ..auth.rbac_dependencies import get_user_permissions, RBACContext
from ..core.secure_tenant_context import get_tenant_context

logger = structlog.get_logger("adaptive_rate_limiting_middleware")


class AdaptiveRateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Enterprise adaptive rate limiting middleware.
    
    Features:
    - Multi-scope enforcement (IP, user, tenant, endpoint, global)
    - Token bucket + sliding window algorithms
    - Hierarchical policy system with overrides
    - Shadow mode for safe deployment
    - Emergency controls and kill-switch
    - Comprehensive observability
    - Integration with RBAC and tenant context
    """
    
    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        shadow_mode: bool = False,
        enable_emergency_controls: bool = True,
        enable_observability: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.redis = redis_client
        self.shadow_mode = shadow_mode
        self.enable_emergency_controls = enable_emergency_controls
        self.enable_observability = enable_observability
        
        # Configuration
        self.config = config or {}
        
        # Initialize components
        self.policy_manager = HierarchicalPolicyManager(redis_client)
        self.rate_limiter = AdaptiveRateLimiter(
            redis_client=redis_client,
            policies=list(self.policy_manager.global_policies.values()),
            shadow_mode=shadow_mode,
            enable_tracing=enable_observability
        )
        self.emergency_limiter = EmergencyRateLimiter(redis_client) if enable_emergency_controls else None
        self.observability = RateLimitObservability() if enable_observability else None
        
        # Exempt paths that should not be rate limited
        self.exempt_paths = {
            "/health",
            "/readiness", 
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
            "/.well-known/"
        }
        
        # High-cost endpoints requiring special handling
        self.high_cost_endpoints = {
            "/api/v1/ptaas/sessions": 5,
            "/api/v1/ptaas/scans": 3,
            "/api/v1/auth/login": 2,
            "/api/v1/auth/token": 2,
            "/api/v1/evidence/upload": 10,
            "/api/v1/vectors/search": 5
        }
        
        # Load role-based overrides
        self._load_role_overrides()
        
        # Start maintenance tasks
        self._start_maintenance_tasks()
        
        logger.info(
            "Adaptive rate limiting middleware initialized",
            shadow_mode=shadow_mode,
            enable_emergency_controls=enable_emergency_controls,
            enable_observability=enable_observability,
            exempt_paths_count=len(self.exempt_paths)
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process request through adaptive rate limiting"""
        start_time = time.time()
        correlation_id = str(uuid4())[:12]
        
        # Add correlation ID to request state
        request.state.rate_limit_correlation_id = correlation_id
        
        try:
            # Check if path is exempt
            if self._is_exempt_path(request.url.path):
                return await call_next(request)
            
            # Check emergency kill-switch first
            if self.emergency_limiter and await self.emergency_limiter.is_kill_switch_active():
                logger.critical("Kill-switch active - blocking all requests", correlation_id=correlation_id)
                return self._create_emergency_response("Kill-switch active", correlation_id)
            
            # Build rate limiting context
            context = await self._build_rate_limit_context(request)
            
            # Start tracing if enabled
            trace_span = None
            if self.observability:
                trace_span = self.observability.start_trace(context, correlation_id)
            
            # Check multiple scopes in order of precedence
            decision_results = []
            
            try:
                # 1. Check global circuit breakers and emergency mode
                if self.emergency_limiter and await self.emergency_limiter.check_emergency_mode():
                    logger.warning("Emergency mode active", correlation_id=correlation_id)
                    # Apply emergency rate limiting (very restrictive)
                    emergency_result = await self._apply_emergency_rate_limiting(context, correlation_id)
                    if not emergency_result.allowed and not self.shadow_mode:
                        return self._create_rate_limit_response(emergency_result, correlation_id)
                
                # 2. Check IP-based rate limiting (broadest scope)
                if context.ip_address:
                    ip_result = await self._check_scope_rate_limit(
                        context, PolicyScope.IP, context.ip_address, correlation_id
                    )
                    decision_results.append(ip_result)
                    if not ip_result.allowed and not self.shadow_mode:
                        return self._create_rate_limit_response(ip_result, correlation_id)
                
                # 3. Check user-based rate limiting (if authenticated)
                if context.user_id:
                    user_result = await self._check_scope_rate_limit(
                        context, PolicyScope.USER, context.user_id, correlation_id
                    )
                    decision_results.append(user_result)
                    if not user_result.allowed and not self.shadow_mode:
                        return self._create_rate_limit_response(user_result, correlation_id)
                
                # 4. Check tenant-based rate limiting (if multi-tenant)
                if context.tenant_id:
                    tenant_result = await self._check_scope_rate_limit(
                        context, PolicyScope.TENANT, str(context.tenant_id), correlation_id
                    )
                    decision_results.append(tenant_result)
                    if not tenant_result.allowed and not self.shadow_mode:
                        return self._create_rate_limit_response(tenant_result, correlation_id)
                
                # 5. Check endpoint-specific rate limiting
                endpoint_result = await self._check_scope_rate_limit(
                    context, PolicyScope.ENDPOINT, context.endpoint or "", correlation_id
                )
                decision_results.append(endpoint_result)
                if not endpoint_result.allowed and not self.shadow_mode:
                    return self._create_rate_limit_response(endpoint_result, correlation_id)
                
                # All checks passed - proceed with request
                response = await call_next(request)
                
                # Add rate limit headers
                if decision_results:
                    self._add_rate_limit_headers(response, decision_results)
                
                # Record successful completion
                if self.observability:
                    for result in decision_results:
                        self.observability.record_decision(context, result, correlation_id)
                
                return response
            
            finally:
                # Ensure tracing span is completed
                if trace_span:
                    if decision_results:
                        primary_result = decision_results[0] if decision_results else None
                        if primary_result and self.observability:
                            event = self.observability.record_decision(context, primary_result, correlation_id)
                            self.observability.tracer.record_decision(trace_span, primary_result, event)
                    trace_span.end()
        
        except Exception as e:
            # Log error and fail open
            logger.error(
                "Rate limiting middleware error - failing open",
                error=str(e),
                correlation_id=correlation_id,
                path=request.url.path,
                exc_info=True
            )
            
            # In production, you might want to fail closed on certain critical errors
            return await call_next(request)
    
    async def _build_rate_limit_context(self, request: Request) -> RateLimitContext:
        """Build comprehensive rate limiting context from request"""
        context = RateLimitContext(
            scope=PolicyScope.IP,  # Default scope
            endpoint=self._normalize_endpoint(request.url.path),
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_time=time.time(),
            business_hours=self._is_business_hours()
        )
        
        # Extract user context if authenticated
        if hasattr(request.state, 'user') and request.state.user:
            user = request.state.user
            context.user_id = user.sub
            context.is_authenticated = True
            context.is_admin = 'admin' in getattr(user, 'roles', [])
            context.is_service_account = 'service' in getattr(user, 'roles', [])
            context.role_names = set(getattr(user, 'roles', []))
        
        # Extract tenant context
        if hasattr(request.state, 'tenant_context'):
            tenant_context = request.state.tenant_context
            context.tenant_id = tenant_context.tenant_id
        
        # Add custom attributes based on request
        context.custom_attributes = {
            'method': request.method,
            'content_type': request.headers.get('Content-Type', ''),
            'has_auth_header': 'Authorization' in request.headers,
            'request_size': int(request.headers.get('Content-Length', 0))
        }
        
        return context
    
    async def _check_scope_rate_limit(
        self,
        context: RateLimitContext,
        scope: PolicyScope,
        identifier: str,
        correlation_id: str
    ) -> Any:
        """Check rate limit for a specific scope"""
        # Update context scope
        scope_context = RateLimitContext(**context.__dict__)
        scope_context.scope = scope
        
        # Resolve policy for this scope
        policy = self.policy_manager.resolve_policy(scope_context)
        if not policy:
            # No policy, allow by default
            return self._create_allow_result(correlation_id)
        
        # Calculate cost for this request
        cost = self._calculate_request_cost(context)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            scope=scope,
            endpoint=context.endpoint,
            user_id=context.user_id,
            tenant_id=str(context.tenant_id) if context.tenant_id else None,
            ip_address=context.ip_address,
            cost=cost,
            correlation_id=correlation_id
        )
        
        return result
    
    async def _apply_emergency_rate_limiting(
        self,
        context: RateLimitContext,
        correlation_id: str
    ) -> Any:
        """Apply emergency rate limiting with very restrictive limits"""
        # Create emergency policy
        emergency_policy = RateLimitPolicy(
            scope=PolicyScope.GLOBAL,
            algorithm=LimitAlgorithm.SLIDING_WINDOW,
            requests_per_second=1.0,  # Very restrictive
            burst_size=3,
            window_seconds=60,
            priority=1
        )
        
        # Apply emergency limiting
        identifier = context.ip_address or "unknown"
        result = await self.rate_limiter.check_rate_limit(
            identifier=f"emergency:{identifier}",
            scope=PolicyScope.GLOBAL,
            endpoint=context.endpoint,
            cost=1,
            correlation_id=correlation_id
        )
        
        if self.observability:
            self.observability.metrics.emergency_mode_activations.labels(
                trigger_reason="emergency_mode"
            ).inc()
        
        return result
    
    def _calculate_request_cost(self, context: RateLimitContext) -> int:
        """Calculate the cost/weight of this request"""
        base_cost = 1
        
        # Higher cost for expensive endpoints
        if context.endpoint:
            for pattern, cost in self.high_cost_endpoints.items():
                if context.endpoint.startswith(pattern):
                    base_cost = cost
                    break
        
        # Higher cost for unauthenticated requests
        if not context.is_authenticated:
            base_cost *= 2
        
        # Higher cost for large requests
        request_size = context.custom_attributes.get('request_size', 0)
        if request_size > 1024 * 1024:  # > 1MB
            base_cost *= 3
        elif request_size > 100 * 1024:  # > 100KB
            base_cost *= 2
        
        # Higher cost for certain HTTP methods
        method = context.custom_attributes.get('method', 'GET')
        if method in ['POST', 'PUT', 'DELETE']:
            base_cost = int(base_cost * 1.5)
        
        return max(1, base_cost)
    
    def _create_allow_result(self, correlation_id: str) -> Any:
        """Create a mock allow result when no policy is found"""
        from ..core.adaptive_rate_limiter import RateLimitResult
        
        return RateLimitResult(
            allowed=True,
            policy_matched=None,
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
    
    def _create_rate_limit_response(self, result: Any, correlation_id: str) -> JSONResponse:
        """Create rate limit exceeded response"""
        retry_after = result.retry_after_seconds or 60
        
        # Determine status code based on reason
        if result.circuit_breaker_open:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_code = "CIRCUIT_BREAKER_OPEN"
            message = "Service temporarily unavailable due to circuit breaker"
        elif result.backoff_level > 0:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            error_code = "PROGRESSIVE_BACKOFF"
            message = f"Rate limited due to repeated violations (level {result.backoff_level})"
        else:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            error_code = "RATE_LIMIT_EXCEEDED"
            message = "Rate limit exceeded"
        
        headers = {
            "X-RateLimit-Remaining": str(result.tokens_remaining),
            "X-RateLimit-Reset": str(int(time.time()) + retry_after),
            "X-Correlation-ID": correlation_id,
            "Retry-After": str(retry_after)
        }
        
        if result.policy_matched:
            headers["X-RateLimit-Limit"] = str(int(result.policy_matched.requests_per_second * result.policy_matched.window_seconds))
            headers["X-RateLimit-Window"] = str(result.policy_matched.window_seconds)
            headers["X-RateLimit-Scope"] = result.policy_matched.scope.value
        
        error_response = {
            "success": False,
            "errors": [{
                "code": error_code,
                "message": message,
                "details": {
                    "tokens_remaining": result.tokens_remaining,
                    "retry_after_seconds": retry_after,
                    "reputation_level": result.reputation_level.value,
                    "correlation_id": correlation_id
                }
            }]
        }
        
        # Log the block
        logger.warning(
            "Request rate limited",
            correlation_id=correlation_id,
            error_code=error_code,
            tokens_remaining=result.tokens_remaining,
            retry_after_seconds=retry_after,
            reputation_level=result.reputation_level.value,
            algorithm=result.algorithm_used.value
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers=headers
        )
    
    def _create_emergency_response(self, reason: str, correlation_id: str) -> JSONResponse:
        """Create emergency response (kill-switch active)"""
        error_response = {
            "success": False,
            "errors": [{
                "code": "EMERGENCY_RATE_LIMITING",
                "message": f"Service temporarily unavailable: {reason}",
                "details": {
                    "correlation_id": correlation_id,
                    "retry_after_seconds": 300
                }
            }]
        }
        
        headers = {
            "X-Correlation-ID": correlation_id,
            "Retry-After": "300"
        }
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, results: list):
        """Add rate limit headers to response"""
        if not results:
            return
        
        # Use the most restrictive result for headers
        primary_result = min(results, key=lambda r: r.tokens_remaining)
        
        response.headers["X-RateLimit-Remaining"] = str(primary_result.tokens_remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        if primary_result.policy_matched:
            response.headers["X-RateLimit-Limit"] = str(int(
                primary_result.policy_matched.requests_per_second * 
                primary_result.policy_matched.window_seconds
            ))
            response.headers["X-RateLimit-Scope"] = primary_result.policy_matched.scope.value
        
        # Add reputation and performance headers
        response.headers["X-RateLimit-Reputation"] = primary_result.reputation_level.value
        response.headers["X-RateLimit-Computation-Time"] = f"{primary_result.computation_time_ms:.2f}ms"
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address with proxy support"""
        # Check X-Forwarded-For header first (from load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (client IP)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for consistent rate limiting"""
        import re
        
        # Remove query parameters
        path = path.split("?")[0]
        
        # Replace UUIDs with placeholder
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace other common patterns
        path = re.sub(r'/[a-f0-9]{32}', '/{hash}', path)  # MD5 hashes
        path = re.sub(r'/[a-f0-9]{64}', '/{hash256}', path)  # SHA256 hashes
        
        return path
    
    def _is_business_hours(self) -> bool:
        """Check if current time is during business hours"""
        import datetime
        
        now = datetime.datetime.now()
        
        # Business hours: Monday-Friday, 6 AM - 10 PM
        if now.weekday() >= 5:  # Weekend
            return False
        
        if now.hour < 6 or now.hour >= 22:  # Outside business hours
            return False
        
        return True
    
    def _load_role_overrides(self):
        """Load default role-based overrides"""
        from ..core.rate_limit_policies import DEFAULT_ROLE_OVERRIDES
        
        for role_name, override in DEFAULT_ROLE_OVERRIDES.items():
            self.policy_manager.add_role_override(role_name, override.scope, override)
        
        logger.info("Role-based overrides loaded", count=len(DEFAULT_ROLE_OVERRIDES))
    
    def _start_maintenance_tasks(self):
        """Start background maintenance tasks"""
        # In production, these would be proper background tasks
        # For now, we'll add them as instance methods that can be called periodically
        
        async def cleanup_task():
            """Periodic cleanup task"""
            try:
                # Cleanup expired keys
                await self.rate_limiter.cleanup_expired_keys()
                
                # Remove expired policy overrides
                self.policy_manager.remove_expired_overrides()
                
                # Update health score
                if self.observability:
                    health_score = self.observability.get_health_score()
                    self.observability.metrics.system_health_score.set(health_score)
                
                logger.debug("Rate limiter maintenance completed")
            
            except Exception as e:
                logger.error("Rate limiter maintenance failed", error=str(e))
        
        # Store cleanup task for external scheduling
        self.cleanup_task = cleanup_task
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive middleware statistics"""
        stats = {
            "middleware": {
                "shadow_mode": self.shadow_mode,
                "emergency_controls_enabled": self.enable_emergency_controls,
                "observability_enabled": self.enable_observability,
                "exempt_paths": len(self.exempt_paths),
                "high_cost_endpoints": len(self.high_cost_endpoints)
            }
        }
        
        # Add rate limiter stats
        if hasattr(self.rate_limiter, 'get_stats'):
            stats["rate_limiter"] = await self.rate_limiter.get_stats()
        
        # Add policy manager stats
        if hasattr(self.policy_manager, 'get_stats'):
            stats["policy_manager"] = self.policy_manager.get_stats()
        
        # Add observability stats
        if self.observability:
            stats["observability"] = self.observability.get_stats()
        
        return stats
    
    async def activate_emergency_mode(self, duration_seconds: int = 300):
        """Activate emergency rate limiting"""
        if self.emergency_limiter:
            await self.emergency_limiter.activate_emergency_mode(duration_seconds)
            logger.critical("Emergency rate limiting activated via middleware", duration_seconds=duration_seconds)
    
    async def activate_kill_switch(self):
        """Activate emergency kill-switch"""
        if self.emergency_limiter:
            await self.emergency_limiter.activate_kill_switch()
            logger.critical("Rate limiter kill-switch activated via middleware")
    
    async def deactivate_kill_switch(self):
        """Deactivate emergency kill-switch"""
        if self.emergency_limiter:
            await self.emergency_limiter.deactivate_kill_switch()
            logger.critical("Rate limiter kill-switch deactivated via middleware")