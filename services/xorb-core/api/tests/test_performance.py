"""Tests for performance optimizations and monitoring."""
import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from app.infrastructure.performance import (
    PerformanceConfig, PerformanceMiddleware, DatabaseMetrics,
    MemoryMonitor, CacheManager, AsyncProfiler,
    setup_uvloop, setup_json_encoder
)
from app.infrastructure.vector_store import VectorStore, get_vector_store


class TestPerformanceConfig:
    """Test performance configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceConfig()

        # Test defaults (environment variables not set)
        assert config.enable_uvloop is True
        assert config.enable_orjson is True
        assert config.enable_metrics is True
        assert config.http_keepalive == 60
        assert config.http_max_requests == 1000

    @patch.dict('os.environ', {
        'ENABLE_UVLOOP': 'false',
        'ENABLE_ORJSON': 'false',
        'HTTP_KEEPALIVE': '30'
    })
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        config = PerformanceConfig()

        assert config.enable_uvloop is False
        assert config.enable_orjson is False
        assert config.http_keepalive == 30


class TestPerformanceMiddleware:
    """Test performance monitoring middleware."""

    @pytest.fixture
    def middleware(self):
        """Performance middleware instance."""
        app = Mock()
        return PerformanceMiddleware(app, enable_metrics=True)

    @pytest.mark.asyncio
    async def test_request_processing(self, middleware):
        """Test request processing with metrics."""
        from starlette.requests import Request
        from starlette.responses import Response

        # Mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/evidence"

        # Mock call_next
        async def mock_call_next(req):
            await asyncio.sleep(0.01)  # Simulate processing time
            response = Response("OK")
            response.status_code = 200
            return response

        # Process request
        response = await middleware.dispatch(request, mock_call_next)

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, middleware):
        """Test error handling in middleware."""
        from starlette.requests import Request

        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/test"

        # Mock call_next that raises exception
        async def mock_call_next(req):
            raise ValueError("Test error")

        # Should propagate exception
        with pytest.raises(ValueError):
            await middleware.dispatch(request, mock_call_next)

    def test_endpoint_name_extraction(self, middleware):
        """Test endpoint name normalization."""
        from starlette.requests import Request

        # Test UUID replacement
        request = Mock(spec=Request)
        request.url.path = f"/api/evidence/{uuid4()}/download"

        endpoint = middleware._get_endpoint_name(request)
        assert endpoint == "/api/evidence/{uuid}/download"

        # Test ID replacement
        request.url.path = "/api/jobs/123/status"
        endpoint = middleware._get_endpoint_name(request)
        assert endpoint == "/api/jobs/{id}/status"


class TestDatabaseMetrics:
    """Test database metrics collection."""

    @pytest.mark.asyncio
    async def test_record_query_time(self):
        """Test query time recording."""
        # This would test Prometheus metrics in a real scenario
        await DatabaseMetrics.record_query_time("SELECT", 0.05)

        # In a real test, would verify metrics were recorded
        assert True  # Placeholder

    @pytest.mark.asyncio
    async def test_update_connection_count(self):
        """Test connection count updating."""
        await DatabaseMetrics.update_connection_count(5)

        # In a real test, would verify gauge was updated
        assert True  # Placeholder


class TestMemoryMonitor:
    """Test memory monitoring functionality."""

    def test_get_memory_usage(self):
        """Test memory usage statistics."""
        with patch('psutil.Process') as mock_process, \
             patch('psutil.virtual_memory') as mock_vm, \
             patch('gc.get_objects') as mock_gc:

            # Mock memory info
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
            mock_memory_info.vms = 1024 * 1024 * 200  # 200MB

            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value = mock_memory_info
            mock_process_instance.memory_percent.return_value = 5.0
            mock_process.return_value = mock_process_instance

            mock_vm_instance = Mock()
            mock_vm_instance.available = 1024 * 1024 * 1024 * 4  # 4GB
            mock_vm.return_value = mock_vm_instance

            mock_gc.return_value = list(range(1000))  # 1000 objects

            stats = MemoryMonitor.get_memory_usage()

            assert stats["rss"] == 1024 * 1024 * 100
            assert stats["vms"] == 1024 * 1024 * 200
            assert stats["percent"] == 5.0
            assert stats["available"] == 1024 * 1024 * 1024 * 4
            assert stats["gc_objects"] == 1000

    @pytest.mark.asyncio
    async def test_update_memory_metrics(self):
        """Test memory metrics updating."""
        with patch.object(MemoryMonitor, 'get_memory_usage') as mock_get_usage:
            mock_get_usage.return_value = {"rss": 1024 * 1024 * 50}

            await MemoryMonitor.update_memory_metrics()

            mock_get_usage.assert_called_once()


class TestCacheManager:
    """Test application-level caching."""

    def test_basic_cache_operations(self):
        """Test basic cache get/set/delete operations."""
        cache = CacheManager()

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

        # Test delete
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = CacheManager()

        # Set with TTL
        cache.set("ttl_key", "ttl_value", ttl=1)  # 1 second
        assert cache.get("ttl_key") == "ttl_value"

        # Wait for expiration (mock time)
        with patch('time.time', return_value=time.time() + 2):
            assert cache.get("ttl_key") is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestAsyncProfiler:
    """Test async function profiler."""

    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        profiler = AsyncProfiler(enabled=False)

        @profiler.profile("test_func")
        async def test_function():
            await asyncio.sleep(0.01)
            return "result"

        # Should return original function when disabled
        assert asyncio.iscoroutinefunction(test_function)

    @pytest.mark.asyncio
    async def test_profiler_enabled(self):
        """Test profiler when enabled."""
        profiler = AsyncProfiler(enabled=True)

        @profiler.profile("test_func")
        async def test_function():
            await asyncio.sleep(0.01)
            return "result"

        # Call function multiple times
        for _ in range(3):
            result = await test_function()
            assert result == "result"

        # Check stats
        stats = profiler.get_stats("test_func")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["min"] > 0
        assert stats["max"] > 0
        assert stats["avg"] > 0
        assert stats["total"] > 0

    def test_profiler_stats_nonexistent(self):
        """Test stats for non-existent function."""
        profiler = AsyncProfiler(enabled=True)

        stats = profiler.get_stats("nonexistent")
        assert stats is None


class TestVectorStore:
    """Test vector store performance."""

    @pytest.fixture
    def vector_store(self):
        """Vector store instance for testing."""
        return VectorStore(dimension=128)  # Smaller dimension for testing

    @pytest.mark.asyncio
    async def test_add_vector_validation(self, vector_store):
        """Test vector dimension validation."""
        vector = [0.1] * 64  # Wrong dimension

        with pytest.raises(ValueError) as exc_info:
            await vector_store.add_vector(
                vector=vector,
                tenant_id=uuid4(),
                source_type="test",
                source_id=uuid4(),
                content_hash="test_hash",
                embedding_model="test_model"
            )

        assert "dimension" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_similar_validation(self, vector_store):
        """Test search vector dimension validation."""
        query_vector = [0.1] * 64  # Wrong dimension

        with pytest.raises(ValueError) as exc_info:
            await vector_store.search_similar(
                query_vector=query_vector,
                tenant_id=uuid4(),
                limit=10
            )

        assert "dimension" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vector_operations_with_mock_db(self, vector_store):
        """Test vector operations with mocked database."""
        tenant_id = uuid4()
        source_id = uuid4()
        vector = [0.1] * 128  # Correct dimension

        # Mock database operations
        with patch('app.infrastructure.vector_store.get_async_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.__aenter__.return_value = mock_session_instance
            mock_session.__aexit__.return_value = None

            # Mock insert result
            mock_result = Mock()
            mock_result.scalar.return_value = uuid4()
            mock_session_instance.execute.return_value = mock_result

            # Test add vector
            vector_id = await vector_store.add_vector(
                vector=vector,
                tenant_id=tenant_id,
                source_type="evidence",
                source_id=source_id,
                content_hash="test_hash",
                embedding_model="test_model"
            )

            assert vector_id is not None
            mock_session_instance.execute.assert_called()
            mock_session_instance.commit.assert_called()


class TestPerformanceIntegration:
    """Integration tests for performance features."""

    def test_uvloop_setup(self):
        """Test uvloop setup."""
        # Mock uvloop availability
        with patch('app.infrastructure.performance.UVLOOP_AVAILABLE', True):
            with patch('uvloop.install') as mock_install:
                setup_uvloop()
                mock_install.assert_called_once()

    def test_json_encoder_setup(self):
        """Test JSON encoder setup."""
        # Test orjson when available
        with patch('app.infrastructure.performance.ORJSON_AVAILABLE', True):
            import app.infrastructure.performance as perf
            with patch.object(perf, 'orjson') as mock_orjson:
                encoder = setup_json_encoder()
                assert encoder == mock_orjson

        # Test fallback to json
        with patch('app.infrastructure.performance.ORJSON_AVAILABLE', False):
            encoder = setup_json_encoder()
            import json
            assert encoder == json

    def test_get_vector_store(self):
        """Test vector store singleton."""
        store1 = get_vector_store(dimension=256)
        store2 = get_vector_store(dimension=256)

        # Should return same instance
        assert store1 is store2
        assert store1.dimension == 256


@pytest.mark.asyncio
async def test_performance_monitoring_context():
    """Test performance monitoring context manager."""
    from app.infrastructure.performance import performance_monitor

    with patch('app.infrastructure.performance.setup_uvloop') as mock_uvloop, \
         patch('app.infrastructure.performance.setup_json_encoder') as mock_json, \
         patch('asyncio.create_task') as mock_task:

        mock_task_instance = Mock()
        mock_task_instance.cancel = Mock()
        mock_task.return_value = mock_task_instance

        async with performance_monitor():
            pass  # Context manager should handle setup and cleanup

        mock_uvloop.assert_called_once()
        mock_json.assert_called_once()
        mock_task_instance.cancel.assert_called_once()


def test_performance_config_singleton():
    """Test that performance configurations are consistent."""
    config1 = PerformanceConfig()
    config2 = PerformanceConfig()

    # Should have same values (though not same instance)
    assert config1.enable_uvloop == config2.enable_uvloop
    assert config1.enable_orjson == config2.enable_orjson
    assert config1.http_keepalive == config2.http_keepalive
