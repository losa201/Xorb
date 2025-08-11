"""
Production Rate Limiter Implementation
Redis-based token bucket and sliding window with Lua scripts for atomic operations
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from .policies import RateLimitPolicy, RateLimitWindow, RateLimitMode, BurstStrategy
from ..core.logging import get_logger
from ..core.config import get_cache_config

logger = get_logger(__name__)


class RateLimitResult(NamedTuple):
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int]
    algorithm: str
    policy_name: str
    violation_count: int = 0


@dataclass
class RateLimitMetrics:
    """Metrics for rate limit operations"""
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    policy_violations: int = 0
    average_decision_time_ms: float = 0.0
    circuit_breaker_trips: int = 0
    reputation_adjustments: int = 0


class TokenBucketLimiter:
    """
    Redis-based token bucket rate limiter with atomic operations
    
    Features:
    - Atomic token consumption with Lua scripts
    - Burst tolerance with configurable strategies
    - Reputation-based rate adjustments
    - Circuit breaker for global protection
    """
    
    # Lua script for atomic token bucket operations
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local tokens_requested = tonumber(ARGV[2])
    local refill_rate = tonumber(ARGV[3])
    local refill_period = tonumber(ARGV[4])
    local current_time = tonumber(ARGV[5])
    local burst_allowance = tonumber(ARGV[6])
    local reputation_score = tonumber(ARGV[7])
    
    -- Get current bucket state
    local bucket_data = redis.call('HMGET', key, 'tokens', 'last_refill', 'violations', 'burst_used')
    local current_tokens = tonumber(bucket_data[1]) or capacity
    local last_refill = tonumber(bucket_data[2]) or current_time
    local violations = tonumber(bucket_data[3]) or 0
    local burst_used = tonumber(bucket_data[4]) or 0
    
    -- Apply reputation adjustment to effective capacity
    local effective_capacity = math.max(1, math.floor(capacity * reputation_score))
    local effective_refill_rate = math.max(1, math.floor(refill_rate * reputation_score))
    
    -- Calculate tokens to add based on time elapsed
    local time_elapsed = current_time - last_refill
    local tokens_to_add = 0
    
    if time_elapsed > 0 then
        -- Calculate refill amount
        local refill_periods = math.floor(time_elapsed / refill_period)
        tokens_to_add = refill_periods * effective_refill_rate
        
        -- Update tokens (don't exceed capacity)
        current_tokens = math.min(effective_capacity, current_tokens + tokens_to_add)
        last_refill = current_time
        
        -- Reset burst usage if enough time has passed
        if refill_periods > 0 then
            burst_used = math.max(0, burst_used - refill_periods)
        end
    end
    
    -- Check if request can be served
    local can_serve = false
    local remaining_tokens = current_tokens
    local retry_after = 0
    
    if current_tokens >= tokens_requested then
        -- Normal case: sufficient tokens
        can_serve = true
        remaining_tokens = current_tokens - tokens_requested
    elseif burst_allowance > 0 and burst_used < burst_allowance then
        -- Burst case: use burst allowance
        local burst_needed = tokens_requested - current_tokens
        if burst_needed <= (burst_allowance - burst_used) then
            can_serve = true
            remaining_tokens = 0
            burst_used = burst_used + burst_needed
        end
    end
    
    -- Calculate retry_after if request denied
    if not can_serve then
        local tokens_needed = tokens_requested - current_tokens
        retry_after = math.ceil((tokens_needed / effective_refill_rate) * refill_period)
        violations = violations + 1
    end
    
    -- Update bucket state
    redis.call('HMSET', key, 
        'tokens', remaining_tokens,
        'last_refill', last_refill,
        'violations', violations,
        'burst_used', burst_used
    )
    
    -- Set expiration (cleanup old buckets)
    redis.call('EXPIRE', key, math.max(3600, refill_period * 2))
    
    return {
        can_serve and 1 or 0,  -- allowed (1) or denied (0)
        remaining_tokens,       -- remaining tokens
        last_refill + refill_period, -- reset time
        retry_after,           -- retry after seconds
        violations             -- total violations
    }
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.script_sha = None
        self._metrics = RateLimitMetrics()
        
    async def initialize(self):
        """Initialize Lua scripts in Redis"""
        try:
            self.script_sha = await self.redis.script_load(self.TOKEN_BUCKET_SCRIPT)
            logger.info("Token bucket Lua script loaded")
        except Exception as e:
            logger.error(f"Failed to load token bucket script: {e}")
            raise
    
    async def check_rate_limit(
        self,
        key: str,
        window: RateLimitWindow,
        policy: RateLimitPolicy,
        reputation_score: float = 1.0,
        tokens_requested: int = 1
    ) -> RateLimitResult:
        """
        Check rate limit using token bucket algorithm
        
        Args:
            key: Unique identifier for the rate limit bucket
            window: Rate limit window configuration
            policy: Rate limit policy
            reputation_score: Reputation adjustment (0.0 to 1.0)
            tokens_requested: Number of tokens to consume
            
        Returns:
            RateLimitResult with decision and metadata
        """
        start_time = time.time()
        
        try:
            # Calculate refill parameters
            refill_rate = window.max_requests
            refill_period = window.duration_seconds
            current_time = time.time()
            
            # Execute atomic token bucket operation
            if not self.script_sha:
                await self.initialize()
            
            result = await self.redis.evalsha(
                self.script_sha,
                1,  # Number of keys
                key,
                window.max_requests,        # capacity
                tokens_requested,           # tokens_requested
                refill_rate,               # refill_rate
                refill_period,             # refill_period
                current_time,              # current_time
                window.burst_allowance,    # burst_allowance
                reputation_score           # reputation_score
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            reset_time = float(result[2])
            retry_after = int(result[3]) if result[3] > 0 else None
            violations = int(result[4])
            
            # Update metrics
            self._metrics.total_requests += 1
            if allowed:
                self._metrics.allowed_requests += 1
            else:
                self._metrics.blocked_requests += 1
                self._metrics.policy_violations += 1
            
            decision_time = (time.time() - start_time) * 1000
            self._metrics.average_decision_time_ms = (
                (self._metrics.average_decision_time_ms * (self._metrics.total_requests - 1) + decision_time) /
                self._metrics.total_requests
            )
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                algorithm="token_bucket",
                policy_name=policy.name,
                violation_count=violations
            )
            
        except Exception as e:
            logger.error(f"Token bucket rate limit check failed: {e}")
            # Fail open in case of Redis issues
            return RateLimitResult(
                allowed=True,
                remaining=window.max_requests,
                reset_time=time.time() + window.duration_seconds,
                retry_after=None,
                algorithm="token_bucket",
                policy_name=policy.name
            )


class SlidingWindowLimiter:
    """
    Redis-based sliding window log limiter for exact rate enforcement
    
    Used for critical endpoints where precise rate limiting is essential.
    More resource intensive than token bucket but provides exact counts.
    """
    
    # Lua script for atomic sliding window operations
    SLIDING_WINDOW_SCRIPT = """
    local key = KEYS[1]
    local window_size = tonumber(ARGV[1])
    local max_requests = tonumber(ARGV[2])
    local current_time = tonumber(ARGV[3])
    local request_id = ARGV[4]
    local reputation_score = tonumber(ARGV[5])
    
    -- Apply reputation adjustment
    local effective_max = math.max(1, math.floor(max_requests * reputation_score))
    
    -- Remove old entries outside the window
    local window_start = current_time - window_size
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
    
    -- Count current requests in window
    local current_count = redis.call('ZCARD', key)
    
    local allowed = current_count < effective_max
    local remaining = math.max(0, effective_max - current_count)
    
    if allowed then
        -- Add current request to window
        redis.call('ZADD', key, current_time, request_id)
        remaining = remaining - 1
    end
    
    -- Set expiration
    redis.call('EXPIRE', key, math.max(3600, window_size))
    
    -- Calculate reset time (when oldest entry expires)
    local oldest_entries = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local reset_time = current_time + window_size
    if #oldest_entries > 0 then
        reset_time = oldest_entries[2] + window_size
    end
    
    -- Calculate retry_after if denied
    local retry_after = 0
    if not allowed then
        -- Time until we can serve another request
        if #oldest_entries > 0 then
            retry_after = math.ceil(oldest_entries[2] + window_size - current_time)
        else
            retry_after = window_size
        end
    end
    
    return {
        allowed and 1 or 0,
        remaining,
        reset_time,
        retry_after,
        current_count
    }
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.script_sha = None
        
    async def initialize(self):
        """Initialize Lua scripts in Redis"""
        try:
            self.script_sha = await self.redis.script_load(self.SLIDING_WINDOW_SCRIPT)
            logger.info("Sliding window Lua script loaded")
        except Exception as e:
            logger.error(f"Failed to load sliding window script: {e}")
            raise
    
    async def check_rate_limit(
        self,
        key: str,
        window: RateLimitWindow,
        policy: RateLimitPolicy,
        reputation_score: float = 1.0,
        request_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit using sliding window log algorithm
        
        Args:
            key: Unique identifier for the rate limit window
            window: Rate limit window configuration
            policy: Rate limit policy
            reputation_score: Reputation adjustment (0.0 to 1.0)
            request_id: Unique request identifier
            
        Returns:
            RateLimitResult with decision and metadata
        """
        if not request_id:
            request_id = f"{time.time()}:{hash(key) % 10000}"
        
        try:
            if not self.script_sha:
                await self.initialize()
            
            current_time = time.time()
            
            result = await self.redis.evalsha(
                self.script_sha,
                1,  # Number of keys
                key,
                window.duration_seconds,    # window_size
                window.max_requests,        # max_requests
                current_time,              # current_time
                request_id,                # request_id
                reputation_score           # reputation_score
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            reset_time = float(result[2])
            retry_after = int(result[3]) if result[3] > 0 else None
            current_count = int(result[4])
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                algorithm="sliding_window",
                policy_name=policy.name
            )
            
        except Exception as e:
            logger.error(f"Sliding window rate limit check failed: {e}")
            # Fail open in case of Redis issues
            return RateLimitResult(
                allowed=True,
                remaining=window.max_requests,
                reset_time=time.time() + window.duration_seconds,
                retry_after=None,
                algorithm="sliding_window",
                policy_name=policy.name
            )


class CircuitBreakerLimiter:
    """
    Global circuit breaker for platform-wide protection
    
    Monitors global request patterns and can temporarily restrict access
    during anomalous traffic patterns or system stress.
    """
    
    CIRCUIT_BREAKER_SCRIPT = """
    local key = KEYS[1]
    local threshold = tonumber(ARGV[1])
    local window_seconds = tonumber(ARGV[2])
    local current_time = tonumber(ARGV[3])
    
    -- Get current state
    local state_data = redis.call('HMGET', key, 'state', 'trip_time', 'request_count', 'window_start')
    local state = state_data[1] or 'closed'
    local trip_time = tonumber(state_data[2]) or 0
    local request_count = tonumber(state_data[3]) or 0
    local window_start = tonumber(state_data[4]) or current_time
    
    -- Reset window if needed
    if current_time - window_start >= window_seconds then
        request_count = 0
        window_start = current_time
    end
    
    -- Increment request count
    request_count = request_count + 1
    
    -- State transitions
    local allow_request = true
    local new_state = state
    
    if state == 'closed' then
        if request_count > threshold then
            new_state = 'open'
            trip_time = current_time
            allow_request = false
        end
    elseif state == 'open' then
        if current_time - trip_time >= window_seconds then
            new_state = 'half_open'
            request_count = 1  -- Reset for half-open test
        else
            allow_request = false
        end
    elseif state == 'half_open' then
        if request_count > (threshold / 4) then  -- Lower threshold for recovery
            new_state = 'closed'
            request_count = 0
        end
    end
    
    -- Update state
    redis.call('HMSET', key,
        'state', new_state,
        'trip_time', trip_time,
        'request_count', request_count,
        'window_start', window_start
    )
    redis.call('EXPIRE', key, window_seconds * 2)
    
    return {
        allow_request and 1 or 0,
        new_state,
        request_count,
        threshold
    }
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.script_sha = None
        
    async def initialize(self):
        """Initialize circuit breaker Lua script"""
        try:
            self.script_sha = await self.redis.script_load(self.CIRCUIT_BREAKER_SCRIPT)
            logger.info("Circuit breaker Lua script loaded")
        except Exception as e:
            logger.error(f"Failed to load circuit breaker script: {e}")
            raise
    
    async def check_circuit_breaker(
        self,
        threshold: int = 1000,
        window_seconds: int = 60
    ) -> Tuple[bool, str]:
        """
        Check global circuit breaker state
        
        Args:
            threshold: Request threshold for tripping breaker
            window_seconds: Time window for counting requests
            
        Returns:
            Tuple of (allow_request, circuit_state)
        """
        try:
            if not self.script_sha:
                await self.initialize()
            
            key = "circuit_breaker:global"
            current_time = time.time()
            
            result = await self.redis.evalsha(
                self.script_sha,
                1,  # Number of keys
                key,
                threshold,
                window_seconds,
                current_time
            )
            
            allowed = bool(result[0])
            state = result[1]
            request_count = int(result[2])
            
            if not allowed:
                logger.warning(f"Circuit breaker tripped: state={state}, requests={request_count}")
            
            return allowed, state
            
        except Exception as e:
            logger.error(f"Circuit breaker check failed: {e}")
            # Fail open to maintain availability
            return True, "closed"


class AdaptiveRateLimiter:
    """
    Main rate limiter that combines token bucket, sliding window, and circuit breaker
    with adaptive reputation scoring and comprehensive observability.
    """
    
    def __init__(self, redis_client: Optional[Redis] = None):
        if redis_client:
            self.redis = redis_client
        else:
            # Create Redis client from configuration
            cache_config = get_cache_config()
            self.redis = redis.from_url(
                cache_config.redis_url,
                max_connections=cache_config.redis_max_connections,
                socket_timeout=cache_config.redis_socket_timeout
            )
        
        self.token_bucket = TokenBucketLimiter(self.redis)
        self.sliding_window = SlidingWindowLimiter(self.redis)
        self.circuit_breaker = CircuitBreakerLimiter(self.redis)
        
        self._reputation_cache: Dict[str, Tuple[float, float]] = {}  # key -> (score, timestamp)
        self._reputation_ttl = 3600  # 1 hour cache
    
    async def initialize(self):
        """Initialize all rate limiter components"""
        await asyncio.gather(
            self.token_bucket.initialize(),
            self.sliding_window.initialize(),
            self.circuit_breaker.initialize()
        )
        logger.info("Adaptive rate limiter initialized")
    
    async def check_rate_limit(
        self,
        key: str,
        policy: RateLimitPolicy,
        request_id: Optional[str] = None,
        use_sliding_window: bool = False
    ) -> List[RateLimitResult]:
        """
        Check rate limits across all configured windows
        
        Args:
            key: Unique identifier for rate limiting
            policy: Rate limit policy to apply
            request_id: Unique request identifier
            use_sliding_window: Use sliding window for exact counting
            
        Returns:
            List of RateLimitResult for each window
        """
        if policy.mode == RateLimitMode.DISABLED:
            # Return allowed for all windows
            return [
                RateLimitResult(
                    allowed=True,
                    remaining=window.max_requests,
                    reset_time=time.time() + window.duration_seconds,
                    retry_after=None,
                    algorithm="disabled",
                    policy_name=policy.name
                )
                for window in policy.windows
            ]
        
        # Check global circuit breaker first
        circuit_allowed, circuit_state = await self.circuit_breaker.check_circuit_breaker(
            policy.circuit_breaker_threshold,
            policy.circuit_breaker_window_seconds
        )
        
        if not circuit_allowed:
            logger.warning(f"Circuit breaker blocked request: {key}")
            return [
                RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=time.time() + policy.circuit_breaker_window_seconds,
                    retry_after=policy.circuit_breaker_window_seconds,
                    algorithm="circuit_breaker",
                    policy_name=policy.name
                )
            ]
        
        # Get reputation score for adaptive limits
        reputation_score = await self._get_reputation_score(key)
        
        # Apply reputation-based adjustments to policy
        effective_windows = policy.get_effective_limits(reputation_score)
        
        # Check each window
        results = []
        limiter = self.sliding_window if use_sliding_window else self.token_bucket
        
        for i, window in enumerate(effective_windows):
            window_key = f"{key}:w{i}:{window.duration_seconds}"
            
            result = await limiter.check_rate_limit(
                window_key,
                window,
                policy,
                reputation_score,
                request_id=request_id
            )
            
            results.append(result)
            
            # If any window denies the request, update reputation
            if not result.allowed:
                await self._update_reputation_score(key, violation=True)
                break
        
        return results
    
    async def _get_reputation_score(self, key: str) -> float:
        """Get current reputation score for a key"""
        if key in self._reputation_cache:
            score, timestamp = self._reputation_cache[key]
            if time.time() - timestamp < self._reputation_ttl:
                return score
        
        # Load from Redis
        try:
            reputation_key = f"reputation:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            data = await self.redis.hgetall(reputation_key)
            
            if data:
                score = float(data.get(b'score', 1.0))
                violations = int(data.get(b'violations', 0))
                last_violation = float(data.get(b'last_violation', 0))
                
                # Apply time-based decay
                hours_since_violation = (time.time() - last_violation) / 3600
                if hours_since_violation > 24:  # 24-hour decay period
                    score = min(1.0, score + (hours_since_violation - 24) * 0.01)
                    violations = max(0, violations - int(hours_since_violation / 24))
                
                # Cache the score
                self._reputation_cache[key] = (score, time.time())
                return score
        except Exception as e:
            logger.error(f"Failed to get reputation score: {e}")
        
        return 1.0  # Default to full reputation
    
    async def _update_reputation_score(self, key: str, violation: bool = False):
        """Update reputation score based on behavior"""
        try:
            reputation_key = f"reputation:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            current_time = time.time()
            
            if violation:
                # Decrease reputation on violation
                current_score = await self._get_reputation_score(key)
                new_score = max(0.1, current_score * 0.8)  # 20% penalty
                
                await self.redis.hset(reputation_key, mapping={
                    'score': new_score,
                    'violations': await self.redis.hincrby(reputation_key, 'violations', 1),
                    'last_violation': current_time
                })
                
                self._reputation_cache[key] = (new_score, current_time)
                logger.debug(f"Reputation decreased: {key} -> {new_score}")
            else:
                # Slowly improve reputation for good behavior
                current_score = await self._get_reputation_score(key)
                new_score = min(1.0, current_score + 0.01)
                
                await self.redis.hset(reputation_key, 'score', new_score)
                self._reputation_cache[key] = (new_score, current_time)
            
            # Set expiration
            await self.redis.expire(reputation_key, 86400 * 7)  # 7 days
            
        except Exception as e:
            logger.error(f"Failed to update reputation score: {e}")
    
    async def get_metrics(self) -> RateLimitMetrics:
        """Get current rate limiter metrics"""
        return self.token_bucket._metrics
    
    async def reset_reputation(self, key: str):
        """Reset reputation score for a key"""
        try:
            reputation_key = f"reputation:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            await self.redis.delete(reputation_key)
            if key in self._reputation_cache:
                del self._reputation_cache[key]
            logger.info(f"Reputation reset for key: {key}")
        except Exception as e:
            logger.error(f"Failed to reset reputation: {e}")
    
    async def close(self):
        """Close Redis connections"""
        if self.redis:
            await self.redis.close()