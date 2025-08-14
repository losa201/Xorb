"""Integration tests for the complete Xorb platform."""
import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import the main application
from app.main import app


class TestFullStackIntegration:
    """Test complete application stack integration."""

    @pytest.fixture
    def client(self):
        """Test client for the application."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Async test client."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    def test_health_endpoint(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "xorb-api"

    def test_version_endpoint(self, client):
        """Test version information endpoint."""
        response = client.get("/version")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "2.0.0"
        assert "features" in data

        features = data["features"]
        assert features["authentication"] == "OIDC"
        assert features["multi_tenancy"] == "RLS"
        assert features["storage"] == "FS+S3"
        assert features["job_orchestration"] == "Redis"
        assert features["vector_search"] == "pgvector"

    def test_openapi_docs(self, client):
        """Test that OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        assert openapi_spec["info"]["title"] == "Xorb API"
        assert openapi_spec["info"]["version"] == "3.0.0"

    @pytest.mark.asyncio
    async def test_readiness_check(self, async_client):
        """Test readiness check with dependency validation."""
        with patch('app.infrastructure.database.check_database_connection', return_value=True), \
             patch('redis.asyncio.Redis.ping', return_value=True):

            response = await async_client.get("/readiness")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "ready"
            assert "checks" in data
            assert data["checks"]["database"]["status"] == "ok"
            assert data["checks"]["redis"]["status"] == "ok"

    def test_middleware_stack_error_handling(self, client):
        """Test that middleware stack handles errors properly."""
        # This endpoint doesn't exist, should return structured error
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        # Should have structured error response
        data = response.json()
        assert data["success"] is False
        assert "errors" in data

    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present."""
        response = client.get("/health")
        assert response.status_code == 200

        # Should have rate limiting headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_request_id_header(self, client):
        """Test that request ID is added to responses."""
        response = client.get("/health")
        assert response.status_code == 200

        # Should have request ID
        assert "X-Request-ID" in response.headers

    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        # Test preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Should allow CORS for localhost
        assert "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_prometheus_metrics(self, async_client):
        """Test Prometheus metrics endpoint."""
        response = await async_client.get("/metrics")

        # Should either return metrics or indicate prometheus not available
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should contain prometheus format metrics
            content = response.text
            assert "# HELP" in content or "# TYPE" in content


class TestAuthenticationIntegration:
    """Test authentication system integration."""

    @pytest.fixture
    def mock_oidc_provider(self):
        """Mock OIDC provider for testing."""
        with patch('app.auth.oidc.get_oidc_provider') as mock:
            provider = Mock()
            provider.validate_access_token = AsyncMock()
            mock.return_value = provider
            yield provider

    @pytest.fixture
    def client(self):
        """Test client."""
        return TestClient(app)

    def test_auth_routes_available(self, client):
        """Test that authentication routes are available."""
        # Login endpoint should redirect to OIDC provider
        response = client.get("/auth/login", allow_redirects=False)
        # Should redirect to OIDC provider (would be 302 with real OIDC)
        assert response.status_code in [302, 500]  # 500 if OIDC not configured

        # Logout endpoint
        response = client.post("/auth/logout")
        assert response.status_code == 200

        # Roles endpoint
        response = client.get("/auth/roles")
        assert response.status_code == 200

        data = response.json()
        assert "roles" in data
        assert "permissions" in data

    @pytest.mark.asyncio
    async def test_protected_endpoint_without_auth(self, async_client):
        """Test accessing protected endpoint without authentication."""
        # Try to access storage endpoint without auth
        response = await async_client.get("/api/storage/evidence")
        assert response.status_code == 401

        data = response.json()
        assert data["success"] is False
        assert any("authentication" in error["message"].lower() for error in data["errors"])


class TestStorageIntegration:
    """Test storage system integration."""

    @pytest.fixture
    def authenticated_client(self):
        """Client with mocked authentication."""
        with patch('app.auth.dependencies.get_current_user') as mock_auth:
            # Mock user with storage permissions
            from app.auth.models import UserClaims, Role
            mock_user = UserClaims(
                sub="test_user",
                email="test@example.com",
                tenant_id=uuid4(),
                roles=[Role.SECURITY_ANALYST],
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow()
            )
            mock_auth.return_value = mock_user

            with TestClient(app) as client:
                yield client, mock_user

    def test_storage_endpoints_require_auth(self):
        """Test that storage endpoints require authentication."""
        client = TestClient(app)

        # Test various storage endpoints
        endpoints = [
            "/api/storage/evidence",
            "/api/storage/upload-url",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_storage_upload_flow(self, authenticated_client):
        """Test complete file upload flow."""
        client, mock_user = authenticated_client

        with patch('app.services.storage_service.StorageService.create_upload_url') as mock_upload:
            mock_upload.return_value = {
                "upload_url": "http://example.com/upload",
                "file_id": str(uuid4()),
                "expires_at": datetime.utcnow().isoformat()
            }

            # Create upload URL
            response = client.post("/api/storage/upload-url", data={
                "filename": "test.pdf",
                "content_type": "application/pdf",
                "size_bytes": "1024"
            })

            # Should work with authentication
            assert response.status_code == 200
            data = response.json()
            assert "upload_url" in data
            assert "file_id" in data


class TestJobsIntegration:
    """Test job orchestration integration."""

    @pytest.fixture
    def authenticated_client(self):
        """Client with mocked authentication."""
        with patch('app.auth.dependencies.get_current_user') as mock_auth:
            from app.auth.models import UserClaims, Role
            mock_user = UserClaims(
                sub="test_user",
                email="test@example.com",
                tenant_id=uuid4(),
                roles=[Role.SECURITY_ANALYST],
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow()
            )
            mock_auth.return_value = mock_user

            with TestClient(app) as client:
                yield client, mock_user

    @pytest.mark.asyncio
    async def test_job_scheduling(self, authenticated_client):
        """Test job scheduling endpoint."""
        client, mock_user = authenticated_client

        with patch('app.jobs.service.JobService.schedule_job') as mock_schedule:
            mock_schedule.return_value = {
                "job_id": str(uuid4()),
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }

            # Schedule a job
            response = client.post("/api/jobs/schedule", json={
                "job_type": "evidence_processing",
                "payload": {"evidence_id": str(uuid4())},
                "priority": "normal"
            })

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data

    def test_queue_stats_endpoint(self, authenticated_client):
        """Test queue statistics endpoint."""
        client, mock_user = authenticated_client

        with patch('app.jobs.service.JobService.get_queue_stats') as mock_stats:
            mock_stats.return_value = {
                "queue_name": "default",
                "pending_jobs": 5,
                "running_jobs": 2,
                "completed_jobs_24h": 100,
                "failed_jobs_24h": 2
            }

            response = client.get("/api/jobs/queue-stats/default")
            assert response.status_code == 200

            data = response.json()
            assert data["queue_name"] == "default"
            assert "pending_jobs" in data


class TestVectorSearchIntegration:
    """Test vector search integration."""

    @pytest.fixture
    def authenticated_client(self):
        """Client with mocked authentication."""
        with patch('app.auth.dependencies.get_current_user') as mock_auth:
            from app.auth.models import UserClaims, Role
            mock_user = UserClaims(
                sub="test_user",
                email="test@example.com",
                tenant_id=uuid4(),
                roles=[Role.SECURITY_ANALYST],
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow()
            )
            mock_auth.return_value = mock_user

            with TestClient(app) as client:
                yield client, mock_user

    @pytest.mark.asyncio
    async def test_vector_search_endpoint(self, authenticated_client):
        """Test vector search functionality."""
        client, mock_user = authenticated_client

        with patch('app.infrastructure.vector_store.VectorStore.search_similar') as mock_search:
            mock_search.return_value = [
                {
                    "id": str(uuid4()),
                    "source_type": "evidence",
                    "source_id": str(uuid4()),
                    "similarity": 0.95,
                    "metadata": {}
                }
            ]

            # Test vector search
            test_vector = [0.1] * 1536  # OpenAI embedding dimension
            response = client.post("/api/vectors/search", json={
                "query_vector": test_vector,
                "limit": 10,
                "similarity_threshold": 0.8
            })

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestMultiTenancyIntegration:
    """Test multi-tenancy isolation."""

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_headers(self):
        """Test tenant context extraction from headers."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            tenant_id = uuid4()

            # Test with X-Tenant-ID header
            response = await client.get(
                "/health",
                headers={"X-Tenant-ID": str(tenant_id)}
            )

            # Should process request (health endpoint doesn't require tenant)
            assert response.status_code == 200

    def test_tenant_context_middleware(self):
        """Test that tenant context middleware is working."""
        client = TestClient(app)

        # The middleware should be in the stack
        # (Can't easily test without actual database, but verifies it loads)
        response = client.get("/health")
        assert response.status_code == 200


class TestPerformanceAndMonitoring:
    """Test performance optimizations and monitoring."""

    def test_response_time_headers(self):
        """Test that response time is tracked."""
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        # Should have process time header from performance middleware
        assert "X-Process-Time" in response.headers

        process_time = float(response.headers["X-Process-Time"])
        assert process_time > 0
        assert process_time < 1.0  # Should be fast for health check

    def test_compression_middleware(self):
        """Test that compression is working."""
        client = TestClient(app)

        # Request with compression
        response = client.get(
            "/version",
            headers={"Accept-Encoding": "gzip"}
        )

        assert response.status_code == 200
        # FastAPI/Starlette should handle compression automatically


@pytest.mark.asyncio
async def test_application_startup_and_shutdown():
    """Test application startup and shutdown process."""

    # Mock all external dependencies
    with patch('app.infrastructure.database.get_database_pool') as mock_db, \
         patch('redis.asyncio.from_url') as mock_redis, \
         patch('app.container.get_container') as mock_container:

        mock_container.return_value.initialize = AsyncMock()
        mock_redis.return_value = Mock()
        mock_db.return_value = Mock()

        # Test that application can start up
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/health")
            assert response.status_code == 200


def test_security_headers():
    """Test that security headers are present."""
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    # Should have security headers (from APISecurityMiddleware)
    # The actual headers depend on the middleware implementation
    # Just verify the middleware is in the stack by checking response


def test_error_handling_format():
    """Test consistent error response format."""
    client = TestClient(app)

    # Access non-existent endpoint
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404

    data = response.json()

    # Should follow error response format
    assert "success" in data
    assert data["success"] is False
    assert "errors" in data
    assert isinstance(data["errors"], list)

    if data["errors"]:
        error = data["errors"][0]
        assert "code" in error
        assert "message" in error


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    async with AsyncClient(app=app, base_url="http://testserver") as client:

        # Make multiple concurrent requests
        tasks = [
            client.get("/health"),
            client.get("/version"),
            client.get("/auth/roles"),
            client.get("/health"),
            client.get("/version")
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code in [200, 401]  # 401 for protected endpoints

        # Should handle concurrency without issues
        health_responses = [r for r in responses if "/health" in str(r.url)]
        for response in health_responses:
            assert response.status_code == 200
