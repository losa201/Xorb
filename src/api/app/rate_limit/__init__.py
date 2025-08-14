"""
Adaptive Rate Limiting System
Production-ready rate limiting with multi-scope enforcement and observability
"""

from .policies import (
    RateLimitPolicy,
    RateLimitWindow,
    RateLimitScope,
    RateLimitMode,
    BurstStrategy,
    AdaptiveConfig,
    PolicyResolver,
    create_default_policies,
    create_testing_policies
)

from .limiter import (
    AdaptiveRateLimiter,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    CircuitBreakerLimiter,
    RateLimitResult,
    RateLimitMetrics
)

from .middleware import (
    RateLimitMiddleware,
    create_rate_limit_middleware,
    get_rate_limit_status
)

__all__ = [
    # Policy classes
    "RateLimitPolicy",
    "RateLimitWindow", 
    "RateLimitScope",
    "RateLimitMode",
    "BurstStrategy",
    "AdaptiveConfig",
    "PolicyResolver",
    "create_default_policies",
    "create_testing_policies",
    
    # Limiter classes
    "AdaptiveRateLimiter",
    "TokenBucketLimiter",
    "SlidingWindowLimiter", 
    "CircuitBreakerLimiter",
    "RateLimitResult",
    "RateLimitMetrics",
    
    # Middleware
    "RateLimitMiddleware",
    "create_rate_limit_middleware",
    "get_rate_limit_status"
]

# Version info
__version__ = "1.0.0"
__author__ = "XORB Security Team"
__description__ = "Production-ready adaptive rate limiting system"