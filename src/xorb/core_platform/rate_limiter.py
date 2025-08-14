import time

import aioredis

from xorb.shared.config import PlatformConfig

# Unified Rate Limiting Service
class UnifiedRateLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def check_rate_limit(self, key: str, limit: int = PlatformConfig.RATE_LIMIT_REQUESTS, window: int = 60) -> tuple[bool, int]:
        """Check rate limit for a key. Returns (allowed, remaining)."""
        current_time = int(time.time())
        window_start = current_time - window

        # Use sliding window counter
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
        pipe.zcard(f"rate_limit:{key}")
        pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
        pipe.expire(f"rate_limit:{key}", window)

        results = await pipe.execute()
        current_requests = results[1]

        if current_requests >= limit:
            return False, 0

        return True, limit - current_requests - 1
