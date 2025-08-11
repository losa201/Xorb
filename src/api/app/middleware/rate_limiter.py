#!/usr/bin/env python3
"""
Xorb Rate Limiting Middleware with Plan-Based Quotas (Clean Architecture)
Integrated with domain services and dependency injection
"""

import time
from dataclasses import dataclass
from datetime import datetime

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram

from ..container import get_container
from ..domain.entities import Organization
from ..domain.exceptions import RateLimitExceeded
from ..infrastructure.redis_manager import get_redis_manager, rate_limit_check

logger = structlog.get_logger("xorb.rate_limiter")

# Phase 5.4 Required Metrics
rate_limit_block_total = Counter(
    'rate_limit_block_total',
    'Total rate limit blocks',
    ['org_id', 'plan_type', 'resource_type']
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['container', 'service']
)

# Additional rate limiting metrics
rate_limit_requests_total = Counter(
    'rate_limit_requests_total',
    'Total rate-limited requests',
    ['org_id', 'endpoint', 'plan_type', 'result']
)

rate_limit_quota_usage = Gauge(
    'rate_limit_quota_usage_ratio',
    'Quota usage ratio (0-1)',
    ['org_id', 'resource_type', 'plan_type']
)

rate_limit_processing_duration = Histogram(
    'rate_limit_processing_duration_seconds',
    'Rate limit processing duration'
)

@dataclass
class PlanLimits:
    """Plan-based resource limits"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    scans_per_day: int
    assets_max: int
    api_burst_size: int
    concurrent_scans: int

    # Memory and compute limits
    memory_limit_mb: int
    cpu_limit_cores: float

@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: int | None = None
    resource_type: str = "requests"

class ResourceBudgetManager:
    """Manages resource budgets and rate limits per organization and plan"""

    def __init__(self):
        self.redis_manager = None

        # Plan configurations (Phase 5.4 requirements)
        self.plan_limits = {
            'Growth': PlanLimits(
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=10000,
                scans_per_day=50,
                assets_max=150,
                api_burst_size=20,
                concurrent_scans=2,
                memory_limit_mb=1024,
                cpu_limit_cores=1.0
            ),
            'Pro': PlanLimits(
                requests_per_minute=500,
                requests_per_hour=5000,
                requests_per_day=50000,
                scans_per_day=200,
                assets_max=500,
                api_burst_size=50,
                concurrent_scans=5,
                memory_limit_mb=4096,
                cpu_limit_cores=4.0
            ),
            'Enterprise': PlanLimits(
                requests_per_minute=2000,
                requests_per_hour=20000,
                requests_per_day=200000,
                scans_per_day=1000,
                assets_max=2000,
                api_burst_size=100,
                concurrent_scans=20,
                memory_limit_mb=16384,
                cpu_limit_cores=16.0
            )
        }

        # SLO violation thresholds (Phase 5.4)
        self.slo_thresholds = {
            'scan_latency_p95': 15.0,      # 15 seconds
            'queue_depth_max': 1000,       # 1000 items
            'memory_usage_threshold': 0.85, # 85% of limit
            'cpu_usage_threshold': 0.80    # 80% of limit
        }

    async def initialize(self, redis_url: str = "redis://redis:6379/0"):
        """Initialize the resource budget manager"""
        self.redis_manager = await get_redis_manager()
        if self.redis_manager and self.redis_manager.is_healthy:
            logger.info("Resource Budget Manager initialized with Redis")
        else:
            logger.warning("Resource Budget Manager initialized without Redis - using in-memory fallback")

    async def check_rate_limit(
        self,
        org_id: str,
        plan_type: str,
        resource_type: str = "requests",
        endpoint: str = "api"
    ) -> RateLimitResult:
        """Check rate limit for organization and resource type"""

        with rate_limit_processing_duration.time():
            try:
                limits = self.plan_limits.get(plan_type, self.plan_limits['Growth'])

                if resource_type == "requests":
                    # Check minute, hour, and daily limits
                    minute_result = await self._check_time_window_limit(
                        org_id, "requests_per_minute", limits.requests_per_minute, 60
                    )
                    if not minute_result.allowed:
                        rate_limit_block_total.labels(
                            org_id=org_id,
                            plan_type=plan_type,
                            resource_type="requests_per_minute"
                        ).inc()
                        return minute_result

                    hour_result = await self._check_time_window_limit(
                        org_id, "requests_per_hour", limits.requests_per_hour, 3600
                    )
                    if not hour_result.allowed:
                        rate_limit_block_total.labels(
                            org_id=org_id,
                            plan_type=plan_type,
                            resource_type="requests_per_hour"
                        ).inc()
                        return hour_result

                    daily_result = await self._check_time_window_limit(
                        org_id, "requests_per_day", limits.requests_per_day, 86400
                    )
                    if not daily_result.allowed:
                        rate_limit_block_total.labels(
                            org_id=org_id,
                            plan_type=plan_type,
                            resource_type="requests_per_day"
                        ).inc()
                        return daily_result

                    # All checks passed
                    rate_limit_requests_total.labels(
                        org_id=org_id,
                        endpoint=endpoint,
                        plan_type=plan_type,
                        result="allowed"
                    ).inc()

                    return minute_result  # Return most restrictive (minute) for headers

                elif resource_type == "scans":
                    return await self._check_time_window_limit(
                        org_id, "scans_per_day", limits.scans_per_day, 86400
                    )

                elif resource_type == "concurrent_scans":
                    return await self._check_concurrent_limit(
                        org_id, "concurrent_scans", limits.concurrent_scans
                    )

                elif resource_type == "assets":
                    return await self._check_static_limit(
                        org_id, "assets", limits.assets_max
                    )

                else:
                    # Unknown resource type, allow by default
                    return RateLimitResult(
                        allowed=True,
                        limit=1000,
                        remaining=1000,
                        reset_time=int(time.time()) + 60,
                        resource_type=resource_type
                    )

            except Exception as e:
                logger.error("Rate limit check failed", error=str(e))
                # Fail open - allow request if rate limiting fails
                return RateLimitResult(
                    allowed=True,
                    limit=1000,
                    remaining=1000,
                    reset_time=int(time.time()) + 60,
                    resource_type=resource_type
                )

    async def _check_time_window_limit(
        self,
        org_id: str,
        limit_type: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitResult:
        """Check time-based rate limit using sliding window"""

        key = f"rate_limit:{org_id}:{limit_type}"
        
        # Use the new Redis manager rate limiting function
        result = await rate_limit_check(key, limit, window_seconds)
        
        # Update quota usage metric
        current_count = result.get("current_count", 0)
        usage_ratio = current_count / limit if limit > 0 else 0
        rate_limit_quota_usage.labels(
            org_id=org_id,
            resource_type=limit_type,
            plan_type="unknown"  # Would need to be passed in
        ).set(usage_ratio)

        return RateLimitResult(
            allowed=result["allowed"],
            limit=result["limit"],
            remaining=result["remaining"],
            reset_time=result["reset_time"],
            retry_after=1 if not result["allowed"] else None,
            resource_type=limit_type
        )

    async def _check_concurrent_limit(
        self,
        org_id: str,
        limit_type: str,
        limit: int
    ) -> RateLimitResult:
        """Check concurrent resource limit"""

        key = f"concurrent:{org_id}:{limit_type}"
        
        if self.redis_manager and self.redis_manager.is_healthy:
            current_value = await self.redis_manager.get(key)
            current_count = int(current_value) if current_value else 0
        else:
            current_count = 0

        remaining = max(0, limit - current_count)

        return RateLimitResult(
            allowed=current_count < limit,
            limit=limit,
            remaining=remaining,
            reset_time=int(time.time()) + 60,  # Arbitrary reset time for concurrent limits
            resource_type=limit_type
        )

    async def _check_static_limit(
        self,
        org_id: str,
        limit_type: str,
        limit: int
    ) -> RateLimitResult:
        """Check static resource limit (e.g., total assets)"""

        # This would query the database for current count
        # For now, simulate with Redis
        key = f"static:{org_id}:{limit_type}"
        
        if self.redis_manager and self.redis_manager.is_healthy:
            current_value = await self.redis_manager.get(key)
            current_count = int(current_value) if current_value else 0
        else:
            current_count = 0

        remaining = max(0, limit - current_count)

        return RateLimitResult(
            allowed=current_count < limit,
            limit=limit,
            remaining=remaining,
            reset_time=0,  # Static limits don't reset
            resource_type=limit_type
        )

    async def increment_concurrent(self, org_id: str, resource_type: str, count: int = 1):
        """Increment concurrent resource counter"""
        key = f"concurrent:{org_id}:{resource_type}"
        if self.redis_manager and self.redis_manager.is_healthy:
            current_value = await self.redis_manager.incr(key, count)
            await self.redis_manager.expire(key, 3600)  # Auto-expire after 1 hour

    async def decrement_concurrent(self, org_id: str, resource_type: str, count: int = 1):
        """Decrement concurrent resource counter"""
        key = f"concurrent:{org_id}:{resource_type}"
        if self.redis_manager and self.redis_manager.is_healthy:
            current_value = await self.redis_manager.get(key)
            current = int(current_value) if current_value else 0
            new_value = max(0, current - count)
            await self.redis_manager.set(key, str(new_value), ex=3600)

    async def update_static_count(self, org_id: str, resource_type: str, count: int):
        """Update static resource counter"""
        key = f"static:{org_id}:{resource_type}"
        if self.redis_manager and self.redis_manager.is_healthy:
            await self.redis_manager.set(key, str(count))

    async def check_slo_violations(self) -> dict[str, bool]:
        """Check for SLO violations that trigger alerts"""
        violations = {}

        try:
            # Check scan latency (would integrate with Prometheus)
            # For now, simulate
            violations['scan_latency_p95'] = False

            # Check queue depth
            if self.redis_manager and self.redis_manager.is_healthy:
                queue_depth_value = await self.redis_manager.get("queue_depth:scans")
                queue_depth = int(queue_depth_value) if queue_depth_value else 0
            else:
                queue_depth = 0
            violations['queue_depth'] = queue_depth > self.slo_thresholds['queue_depth_max']

            # Check memory usage (would integrate with container metrics)
            violations['memory_usage'] = False

            # Check CPU usage
            violations['cpu_usage'] = False

            return violations

        except Exception as e:
            logger.error("SLO violation check failed", error=str(e))
            return {}

    async def get_resource_usage_report(self, org_id: str, plan_type: str) -> dict:
        """Get comprehensive resource usage report"""
        try:
            limits = self.plan_limits[plan_type]
            report = {
                "org_id": org_id,
                "plan_type": plan_type,
                "limits": {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "requests_per_day": limits.requests_per_day,
                    "scans_per_day": limits.scans_per_day,
                    "assets_max": limits.assets_max,
                    "concurrent_scans": limits.concurrent_scans
                },
                "current_usage": {},
                "usage_ratios": {},
                "timestamp": datetime.now().isoformat()
            }

            # Get current usage for each resource type
            resources = [
                ("requests_per_minute", 60),
                ("requests_per_hour", 3600),
                ("requests_per_day", 86400),
                ("scans_per_day", 86400)
            ]

            for resource, window in resources:
                key = f"rate_limit:{org_id}:{resource}"
                
                if self.redis_manager and self.redis_manager.is_healthy:
                    now = int(time.time())
                    window_start = now - window
                    current_usage = await self.redis_manager.zcount(key, window_start, now)
                else:
                    current_usage = 0
                    
                report["current_usage"][resource] = current_usage

                # Calculate usage ratio
                limit = getattr(limits, resource)
                ratio = current_usage / limit if limit > 0 else 0
                report["usage_ratios"][resource] = ratio

            # Get concurrent usage
            if self.redis_manager and self.redis_manager.is_healthy:
                concurrent_scans_value = await self.redis_manager.get(f"concurrent:{org_id}:concurrent_scans")
                concurrent_scans = int(concurrent_scans_value) if concurrent_scans_value else 0
            else:
                concurrent_scans = 0
            report["current_usage"]["concurrent_scans"] = concurrent_scans
            report["usage_ratios"]["concurrent_scans"] = concurrent_scans / limits.concurrent_scans

            # Get asset count
            if self.redis_manager and self.redis_manager.is_healthy:
                asset_count_value = await self.redis_manager.get(f"static:{org_id}:assets")
                asset_count = int(asset_count_value) if asset_count_value else 0
            else:
                asset_count = 0
            report["current_usage"]["assets"] = asset_count
            report["usage_ratios"]["assets"] = asset_count / limits.assets_max

            return report

        except Exception as e:
            logger.error("Failed to generate resource usage report", error=str(e))
            return {"error": str(e)}

# Global instance
resource_manager = ResourceBudgetManager()

async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting using clean architecture"""

    # Skip rate limiting for health checks and metrics
    if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
        return await call_next(request)

    # Extract organization info from request headers
    org_id = request.headers.get("X-Org-ID")
    plan_type = request.headers.get("X-Plan-Type", "Growth")

    if not org_id:
        # No org ID, allow request but log warning
        logger.warning("Request without org ID", path=request.url.path)
        return await call_next(request)

    try:
        # Create organization entity from headers
        org = Organization.create(name=f"Org-{org_id}", plan_type=plan_type)
        org.id = org_id  # Override with actual ID from header
        
        # Use legacy rate limiter for now - in a full implementation,
        # this would use a proper RateLimitService from the container
        result = await resource_manager.check_rate_limit(
            org_id=org_id,
            plan_type=plan_type,
            resource_type="requests",
            endpoint=request.url.path
        )

        if not result.allowed:
            # Create domain exception
            rate_limit_exception = RateLimitExceeded(
                message=f"Too many {result.resource_type} for plan {plan_type}",
                limit=result.limit,
                remaining=result.remaining,
                reset_time=result.reset_time
            )
            
            # Record metrics
            rate_limit_requests_total.labels(
                org_id=org_id,
                endpoint=request.url.path,
                plan_type=plan_type,
                result="blocked"
            ).inc()

            headers = {
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": str(result.reset_time),
                "X-RateLimit-Resource": result.resource_type
            }

            if result.retry_after:
                headers["Retry-After"] = str(result.retry_after)

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": rate_limit_exception.message,
                    "limit": result.limit,
                    "remaining": result.remaining,
                    "reset_time": result.reset_time
                },
                headers=headers
            )

        # Rate limit passed, process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_time)
        response.headers["X-RateLimit-Resource"] = result.resource_type

        return response
        
    except Exception as e:
        logger.error("Rate limiting failed", error=str(e))
        # Fail open - allow request if rate limiting fails
        return await call_next(request)

def get_resource_manager() -> ResourceBudgetManager:
    """Dependency injection for resource manager"""
    return resource_manager
