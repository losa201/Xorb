"""
Production-ready Redis Manager for XORB Platform
Replaces aioredis with redis-py for Python 3.12 compatibility
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis connection configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30


class XORBRedisManager:
    """Enterprise Redis connection manager with health monitoring"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy: bool = False

    async def initialize(self) -> bool:
        """Initialize Redis connection pool"""
        try:
            # Create connection pool
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=True
            )

            # Create Redis client
            self._client = Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            self._is_healthy = True

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(f"Redis connection initialized: {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._is_healthy = False
            return False

    async def shutdown(self):
        """Gracefully shutdown Redis connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        if self._pool:
            await self._pool.aclose()

        logger.info("Redis connection shutdown complete")

    @property
    def client(self) -> Optional[Redis]:
        """Get Redis client instance"""
        return self._client

    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy"""
        return self._is_healthy

    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                if self._client:
                    await self._client.ping()
                    if not self._is_healthy:
                        self._is_healthy = True
                        logger.info("Redis connection restored")
            except Exception as e:
                if self._is_healthy:
                    self._is_healthy = False
                    logger.error(f"Redis health check failed: {e}")

    # High-level Redis operations
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis with error handling"""
        if not self._client or not self._is_healthy:
            return None

        try:
            return await self._client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis with error handling"""
        if not self._client or not self._is_healthy:
            return False

        try:
            await self._client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self._client or not self._is_healthy:
            return False

        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self._client or not self._is_healthy:
            return False

        try:
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment value in Redis"""
        if not self._client or not self._is_healthy:
            return None

        try:
            return await self._client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR error for key {key}: {e}")
            return None

    async def expire(self, key: str, time: int) -> bool:
        """Set expiration on key"""
        if not self._client or not self._is_healthy:
            return False

        try:
            return await self._client.expire(key, time)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False

    # Advanced Redis operations
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add to sorted set"""
        if not self._client or not self._is_healthy:
            return 0

        try:
            return await self._client.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Redis ZADD error for key {key}: {e}")
            return 0

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove from sorted set by score range"""
        if not self._client or not self._is_healthy:
            return 0

        try:
            return await self._client.zremrangebyscore(key, min_score, max_score)
        except Exception as e:
            logger.error(f"Redis ZREMRANGEBYSCORE error for key {key}: {e}")
            return 0

    async def zcard(self, key: str) -> int:
        """Get sorted set cardinality"""
        if not self._client or not self._is_healthy:
            return 0

        try:
            return await self._client.zcard(key)
        except Exception as e:
            logger.error(f"Redis ZCARD error for key {key}: {e}")
            return 0

    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        """Count elements in sorted set score range"""
        if not self._client or not self._is_healthy:
            return 0

        try:
            return await self._client.zcount(key, min_score, max_score)
        except Exception as e:
            logger.error(f"Redis ZCOUNT error for key {key}: {e}")
            return 0

    async def pipeline(self):
        """Get Redis pipeline for batch operations"""
        if not self._client or not self._is_healthy:
            return None

        try:
            return self._client.pipeline()
        except Exception as e:
            logger.error(f"Redis PIPELINE error: {e}")
            return None

    async def execute_pipeline(self, pipe) -> Optional[List[Any]]:
        """Execute Redis pipeline with error handling"""
        if not pipe:
            return None

        try:
            return await pipe.execute()
        except Exception as e:
            logger.error(f"Redis pipeline execution error: {e}")
            return None


# Global Redis manager instance
_redis_manager: Optional[XORBRedisManager] = None


async def get_redis_manager() -> Optional[XORBRedisManager]:
    """Get global Redis manager instance"""
    global _redis_manager
    return _redis_manager


async def initialize_redis(config: Optional[RedisConfig] = None) -> bool:
    """Initialize global Redis manager"""
    global _redis_manager

    if not config:
        config = RedisConfig()

    _redis_manager = XORBRedisManager(config)
    return await _redis_manager.initialize()


async def shutdown_redis():
    """Shutdown global Redis manager"""
    global _redis_manager
    if _redis_manager:
        await _redis_manager.shutdown()
        _redis_manager = None


@asynccontextmanager
async def redis_transaction():
    """Context manager for Redis transactions"""
    manager = await get_redis_manager()
    if not manager or not manager.client:
        yield None
        return

    pipe = await manager.pipeline()
    if not pipe:
        yield None
        return

    try:
        yield pipe
        await manager.execute_pipeline(pipe)
    except Exception as e:
        logger.error(f"Redis transaction error: {e}")
        # Transaction failed, but we don't need to rollback explicitly
        # Redis transactions are atomic


# Utility functions for common patterns
async def cache_get_or_set(
    key: str,
    factory_func,
    ttl: int = 3600,
    *args,
    **kwargs
) -> Any:
    """Get from cache or compute and cache the value"""
    manager = await get_redis_manager()
    if not manager:
        # No cache available, compute directly
        return await factory_func(*args, **kwargs) if asyncio.iscoroutinefunction(factory_func) else factory_func(*args, **kwargs)

    # Try to get from cache
    cached_value = await manager.get(key)
    if cached_value is not None:
        try:
            return json.loads(cached_value)
        except json.JSONDecodeError:
            return cached_value

    # Compute value
    value = await factory_func(*args, **kwargs) if asyncio.iscoroutinefunction(factory_func) else factory_func(*args, **kwargs)

    # Cache the value
    try:
        cache_value = json.dumps(value) if not isinstance(value, str) else value
        await manager.set(key, cache_value, ex=ttl)
    except Exception as e:
        logger.error(f"Failed to cache value for key {key}: {e}")

    return value


async def rate_limit_check(
    key: str,
    limit: int,
    window_seconds: int
) -> Dict[str, Any]:
    """Check rate limit using sliding window"""
    manager = await get_redis_manager()
    if not manager:
        # No Redis available, allow request
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit,
            "reset_time": int(time.time()) + window_seconds
        }

    now = time.time()
    window_start = now - window_seconds

    pipe = await manager.pipeline()
    if not pipe:
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit,
            "reset_time": int(now) + window_seconds
        }

    try:
        # Remove expired entries
        await pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        await pipe.zcard(key)

        # Add current request
        await pipe.zadd(key, {str(now): now})

        # Set expiration
        await pipe.expire(key, window_seconds + 60)

        results = await manager.execute_pipeline(pipe)
        current_count = results[1] + 1  # +1 for current request

        remaining = max(0, limit - current_count)
        reset_time = int(now + window_seconds)

        return {
            "allowed": current_count <= limit,
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
            "current_count": current_count
        }

    except Exception as e:
        logger.error(f"Rate limit check error for key {key}: {e}")
        # Fail open
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit,
            "reset_time": int(now) + window_seconds
        }
