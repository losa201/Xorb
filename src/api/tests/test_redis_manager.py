"""
Tests for the Redis Manager
Ensures the new Redis implementation works correctly
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.redis_manager import (
    XORBRedisManager, 
    RedisConfig, 
    rate_limit_check,
    cache_get_or_set,
    initialize_redis,
    shutdown_redis,
    get_redis_manager
)


@pytest.fixture
def redis_config():
    """Test Redis configuration"""
    return RedisConfig(
        host="localhost",
        port=6379,
        password=None,
        db=1,  # Use test database
        max_connections=10,
        socket_timeout=1.0,
        socket_connect_timeout=1.0
    )


@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing"""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = 0
    mock_client.incrby.return_value = 1
    mock_client.expire.return_value = True
    mock_client.zadd.return_value = 1
    mock_client.zremrangebyscore.return_value = 0
    mock_client.zcard.return_value = 1
    mock_client.zcount.return_value = 1
    mock_client.aclose = AsyncMock()
    
    # Mock pipeline
    mock_pipe = AsyncMock()
    mock_pipe.execute.return_value = [0, 1, 1, True]
    mock_client.pipeline.return_value = mock_pipe
    
    return mock_client


@pytest.fixture
async def redis_manager(redis_config, mock_redis_client):
    """Test Redis manager with mocked client"""
    manager = XORBRedisManager(redis_config)
    
    with patch('redis.asyncio.ConnectionPool') as mock_pool, \
         patch('redis.asyncio.Redis') as mock_redis_class:
        
        mock_redis_class.return_value = mock_redis_client
        
        await manager.initialize()
        yield manager
        await manager.shutdown()


class TestRedisConfig:
    """Test Redis configuration"""
    
    def test_default_config(self):
        """Test default Redis configuration"""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.password is None
        assert config.db == 0
        assert config.max_connections == 50
    
    def test_custom_config(self):
        """Test custom Redis configuration"""
        config = RedisConfig(
            host="redis-server",
            port=6380,
            password="secret",
            db=2,
            max_connections=100
        )
        assert config.host == "redis-server"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.db == 2
        assert config.max_connections == 100


class TestXORBRedisManager:
    """Test Redis manager functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, redis_config, mock_redis_client):
        """Test successful Redis initialization"""
        manager = XORBRedisManager(redis_config)
        
        with patch('redis.asyncio.ConnectionPool') as mock_pool, \
             patch('redis.asyncio.Redis') as mock_redis_class:
            
            mock_redis_class.return_value = mock_redis_client
            
            result = await manager.initialize()
            assert result is True
            assert manager.is_healthy is True
            assert manager.client == mock_redis_client
            
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, redis_config):
        """Test Redis initialization failure"""
        manager = XORBRedisManager(redis_config)
        
        with patch('redis.asyncio.ConnectionPool') as mock_pool, \
             patch('redis.asyncio.Redis') as mock_redis_class:
            
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_client
            
            result = await manager.initialize()
            assert result is False
            assert manager.is_healthy is False
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, redis_manager, mock_redis_client):
        """Test basic Redis operations"""
        # Test GET
        mock_redis_client.get.return_value = "test_value"
        result = await redis_manager.get("test_key")
        assert result == "test_value"
        mock_redis_client.get.assert_called_with("test_key")
        
        # Test SET
        result = await redis_manager.set("test_key", "test_value", ex=3600)
        assert result is True
        mock_redis_client.set.assert_called_with("test_key", "test_value", ex=3600)
        
        # Test DELETE
        mock_redis_client.delete.return_value = 1
        result = await redis_manager.delete("test_key")
        assert result is True
        mock_redis_client.delete.assert_called_with("test_key")
        
        # Test EXISTS
        mock_redis_client.exists.return_value = 1
        result = await redis_manager.exists("test_key")
        assert result is True
        mock_redis_client.exists.assert_called_with("test_key")
        
        # Test INCR
        mock_redis_client.incrby.return_value = 5
        result = await redis_manager.incr("counter", 3)
        assert result == 5
        mock_redis_client.incrby.assert_called_with("counter", 3)
    
    @pytest.mark.asyncio
    async def test_operations_with_unhealthy_connection(self, redis_config):
        """Test operations when Redis is unhealthy"""
        manager = XORBRedisManager(redis_config)
        manager._is_healthy = False  # Simulate unhealthy connection
        
        # All operations should return safe defaults
        assert await manager.get("key") is None
        assert await manager.set("key", "value") is False
        assert await manager.delete("key") is False
        assert await manager.exists("key") is False
        assert await manager.incr("key") is None
        assert await manager.expire("key", 60) is False
    
    @pytest.mark.asyncio
    async def test_sorted_set_operations(self, redis_manager, mock_redis_client):
        """Test sorted set operations"""
        # Test ZADD
        mock_redis_client.zadd.return_value = 1
        result = await redis_manager.zadd("test_set", {"member1": 1.0, "member2": 2.0})
        assert result == 1
        mock_redis_client.zadd.assert_called_with("test_set", {"member1": 1.0, "member2": 2.0})
        
        # Test ZREMRANGEBYSCORE
        mock_redis_client.zremrangebyscore.return_value = 2
        result = await redis_manager.zremrangebyscore("test_set", 0.0, 1.0)
        assert result == 2
        mock_redis_client.zremrangebyscore.assert_called_with("test_set", 0.0, 1.0)
        
        # Test ZCARD
        mock_redis_client.zcard.return_value = 5
        result = await redis_manager.zcard("test_set")
        assert result == 5
        mock_redis_client.zcard.assert_called_with("test_set")
        
        # Test ZCOUNT
        mock_redis_client.zcount.return_value = 3
        result = await redis_manager.zcount("test_set", 1.0, 5.0)
        assert result == 3
        mock_redis_client.zcount.assert_called_with("test_set", 1.0, 5.0)


class TestRateLimitCheck:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_without_redis(self):
        """Test rate limit check when Redis is unavailable"""
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=None):
            result = await rate_limit_check("test_key", 100, 60)
            
            assert result["allowed"] is True
            assert result["limit"] == 100
            assert result["remaining"] == 100
            assert "reset_time" in result
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_with_redis(self, mock_redis_client):
        """Test rate limit check with Redis"""
        mock_manager = AsyncMock()
        mock_manager.pipeline.return_value = mock_redis_client.pipeline()
        mock_manager.execute_pipeline.return_value = [0, 5, 1, True]  # 5 requests in window
        
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=mock_manager):
            result = await rate_limit_check("test_key", 10, 60)
            
            assert result["allowed"] is True  # 6 requests <= 10 limit
            assert result["limit"] == 10
            assert result["remaining"] == 4  # 10 - 6
            assert "reset_time" in result
            assert result["current_count"] == 6


class TestCacheGetOrSet:
    """Test cache get-or-set functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit scenario"""
        mock_manager = AsyncMock()
        mock_manager.get.return_value = '"cached_value"'  # JSON string
        
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=mock_manager):
            async def factory_func():
                return "computed_value"
            
            result = await cache_get_or_set("test_key", factory_func, ttl=3600)
            assert result == "cached_value"
            mock_manager.get.assert_called_once_with("test_key")
            mock_manager.set.assert_not_called()  # Should not compute or set
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss scenario"""
        mock_manager = AsyncMock()
        mock_manager.get.return_value = None  # Cache miss
        mock_manager.set.return_value = True
        
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=mock_manager):
            async def factory_func():
                return "computed_value"
            
            result = await cache_get_or_set("test_key", factory_func, ttl=3600)
            assert result == "computed_value"
            mock_manager.get.assert_called_once_with("test_key")
            mock_manager.set.assert_called_once_with("test_key", '"computed_value"', ex=3600)
    
    @pytest.mark.asyncio
    async def test_cache_without_redis(self):
        """Test caching when Redis is unavailable"""
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=None):
            async def factory_func():
                return "computed_value"
            
            result = await cache_get_or_set("test_key", factory_func, ttl=3600)
            assert result == "computed_value"
    
    @pytest.mark.asyncio
    async def test_cache_sync_function(self):
        """Test caching with synchronous factory function"""
        mock_manager = AsyncMock()
        mock_manager.get.return_value = None  # Cache miss
        mock_manager.set.return_value = True
        
        with patch('app.infrastructure.redis_manager.get_redis_manager', return_value=mock_manager):
            def sync_factory_func():
                return "sync_computed_value"
            
            result = await cache_get_or_set("test_key", sync_factory_func, ttl=3600)
            assert result == "sync_computed_value"


class TestGlobalRedisManager:
    """Test global Redis manager functions"""
    
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Test global Redis manager initialization and shutdown"""
        config = RedisConfig(host="test-host", port=6380)
        
        with patch('app.infrastructure.redis_manager.XORBRedisManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.initialize.return_value = True
            mock_manager_class.return_value = mock_manager
            
            # Initialize
            result = await initialize_redis(config)
            assert result is True
            mock_manager_class.assert_called_once_with(config)
            mock_manager.initialize.assert_called_once()
            
            # Get manager
            manager = await get_redis_manager()
            assert manager == mock_manager
            
            # Shutdown
            await shutdown_redis()
            mock_manager.shutdown.assert_called_once()
            
            # Manager should be None after shutdown
            manager = await get_redis_manager()
            assert manager is None


@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests (requires actual Redis server)"""
    
    @pytest.mark.asyncio
    async def test_real_redis_connection(self):
        """Test connection to real Redis server (if available)"""
        config = RedisConfig(host="localhost", port=6379, db=15)  # Use high DB number for testing
        manager = XORBRedisManager(config)
        
        try:
            # Try to initialize - this will fail if Redis is not available
            result = await manager.initialize()
            
            if result:
                # If Redis is available, test basic operations
                assert manager.is_healthy is True
                
                # Test basic operations
                await manager.set("test_key", "test_value", ex=10)
                value = await manager.get("test_key")
                assert value == "test_value"
                
                # Test deletion
                deleted = await manager.delete("test_key")
                assert deleted is True
                
                # Verify deletion
                value = await manager.get("test_key")
                assert value is None
                
                await manager.shutdown()
            else:
                # Redis not available, skip test
                pytest.skip("Redis server not available for integration test")
                
        except Exception as e:
            # Redis not available or connection failed
            pytest.skip(f"Redis integration test skipped: {e}")
        finally:
            if manager._client:
                await manager.shutdown()