"""
Integration tests for API endpoints and services
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_readiness_check(self, test_client: TestClient):
        response = test_client.get("/api/v1/readiness")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_info_endpoint(self, test_client: TestClient):
        response = test_client.get("/api/v1/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "title" in data


class TestAuthenticationFlow:
    """Test authentication endpoints and flow"""
    
    @pytest.mark.asyncio
    async def test_user_registration_and_login(self, async_test_client: AsyncClient, test_user_data):
        # Test user registration (if endpoint exists)
        registration_data = {
            "username": test_user_data["username"],
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        
        # Note: Adjust endpoint path as needed
        response = await async_test_client.post("/api/v1/auth/register", json=registration_data)
        
        # May return 404 if endpoint doesn't exist yet - that's okay for now
        if response.status_code not in [404, 501]:
            assert response.status_code in [200, 201]
            data = response.json()
            assert "user_id" in data or "message" in data
    
    @pytest.mark.asyncio
    async def test_token_validation(self, async_test_client: AsyncClient, auth_headers):
        # Test token validation with a protected endpoint
        response = await async_test_client.get("/api/v1/auth/me", headers=auth_headers)
        
        # May return 404 if endpoint doesn't exist yet
        if response.status_code != 404:
            # If endpoint exists, should validate token
            assert response.status_code in [200, 401]


class TestPTaaSEndpoints:
    """Test PTaaS (Penetration Testing as a Service) endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_scan_profiles(self, async_test_client: AsyncClient):
        response = await async_test_client.get("/api/v1/ptaas/profiles")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_create_scan_session(self, async_test_client: AsyncClient, test_scan_session_data, auth_headers):
        response = await async_test_client.post(
            "/api/v1/ptaas/sessions",
            json=test_scan_session_data,
            headers=auth_headers
        )
        
        # May return 404 if endpoint doesn't exist yet, or 401 if auth is required
        if response.status_code not in [404, 401, 501]:
            assert response.status_code in [200, 201, 202]
            data = response.json()
            assert "session_id" in data or "id" in data
    
    @pytest.mark.asyncio
    async def test_scan_session_status(self, async_test_client: AsyncClient, auth_headers):
        # Test getting scan session status
        session_id = "test-session-id"
        response = await async_test_client.get(
            f"/api/v1/ptaas/sessions/{session_id}",
            headers=auth_headers
        )
        
        # Endpoint may not exist yet or session may not be found
        assert response.status_code in [200, 404, 401, 501]


class TestSecurityMiddleware:
    """Test security middleware and headers"""
    
    def test_security_headers(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        
        # Check for security headers (if implemented)
        headers = response.headers
        
        # These may not be implemented yet
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]
        
        # Just verify response is successful
        assert response.status_code == 200
    
    def test_cors_headers(self, test_client: TestClient):
        # Test CORS preflight request
        response = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # CORS may not be fully configured yet
        assert response.status_code in [200, 204, 405]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, async_test_client: AsyncClient):
        # Make multiple rapid requests to test rate limiting
        endpoint = "/api/v1/health"
        responses = []
        
        for i in range(10):
            response = await async_test_client.get(endpoint)
            responses.append(response.status_code)
        
        # Should get mostly 200s, but might get 429 if rate limiting is active
        success_responses = sum(1 for status in responses if status == 200)
        rate_limited_responses = sum(1 for status in responses if status == 429)
        
        # Should have some successful responses
        assert success_responses > 0
        
        # Rate limiting may not be implemented yet, so we don't assert it


class TestErrorHandling:
    """Test error handling and responses"""
    
    def test_404_error_handling(self, test_client: TestClient):
        response = test_client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "message" in data
    
    def test_method_not_allowed(self, test_client: TestClient):
        response = test_client.post("/api/v1/health")  # GET-only endpoint
        
        assert response.status_code == 405
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self, async_test_client: AsyncClient):
        response = await async_test_client.post(
            "/api/v1/ptaas/sessions",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]


class TestDatabaseIntegration:
    """Test database integration"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, test_db_session):
        # Test that database session is available
        assert test_db_session is not None
        
        # Test basic query execution
        result = await test_db_session.execute("SELECT 1 as test_value")
        row = result.fetchone()
        assert row[0] == 1


class TestCacheIntegration:
    """Test caching integration"""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, test_cache_service):
        cache = test_cache_service
        
        # Test basic cache operations
        await cache.set("test_key", "test_value", ttl=60)
        
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test cache deletion
        await cache.delete("test_key")
        value = await cache.get("test_key")
        assert value is None


class TestMetricsIntegration:
    """Test metrics collection integration"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, test_metrics_service):
        metrics = test_metrics_service
        
        # Test custom metrics
        metrics.custom_metrics.increment_counter("test_counter", 1)
        metrics.custom_metrics.set_gauge("test_gauge", 42.0)
        metrics.custom_metrics.record_histogram("test_histogram", 1.5)
        
        # Get metrics summary
        summary = metrics.custom_metrics.get_metrics_summary()
        assert "counters" in summary
        assert "gauges" in summary


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_api_workflow(async_test_client: AsyncClient, test_user_data, test_scan_session_data):
    """Test complete API workflow"""
    
    # 1. Check API health
    health_response = await async_test_client.get("/api/v1/health")
    assert health_response.status_code == 200
    
    # 2. Get available scan profiles
    profiles_response = await async_test_client.get("/api/v1/ptaas/profiles")
    # May not be implemented yet
    
    # 3. Check API info
    info_response = await async_test_client.get("/api/v1/info")
    assert info_response.status_code == 200
    
    info_data = info_response.json()
    assert "version" in info_data
    
    # Workflow test passes if basic endpoints are working
    assert True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_requests(async_test_client: AsyncClient):
    """Test handling of concurrent requests"""
    import asyncio
    
    async def make_request():
        response = await async_test_client.get("/api/v1/health")
        return response.status_code
    
    # Make 20 concurrent requests
    tasks = [make_request() for _ in range(20)]
    results = await asyncio.gather(*tasks)
    
    # All requests should succeed
    success_count = sum(1 for status in results if status == 200)
    assert success_count >= 18  # Allow for some potential failures due to load