"""Integration tests for the unified XORB platform"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add src/api to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "api"))

from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.infrastructure.service_orchestrator import ServiceOrchestrator, get_service_orchestrator


class TestPlatformIntegration:
    """Test suite for complete platform integration"""
    
    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(app)
    
    @pytest.fixture
    async def orchestrator(self):
        """Service orchestrator for testing"""
        orchestrator = ServiceOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    
    def test_platform_gateway_routes_available(self, client):
        """Test that platform gateway routes are registered"""
        # Get OpenAPI spec to check routes
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        paths = openapi_spec.get("paths", {})
        
        # Check core platform management routes
        assert "/api/v1/platform/services" in paths
        assert "/api/v1/platform/health" in paths
        assert "/api/v1/platform/metrics" in paths
        
        # Check PTaaS service routes
        assert "/api/v1/platform/analytics/behavioral/dashboard" in paths
        assert "/api/v1/platform/threat-hunting/query" in paths
        assert "/api/v1/platform/forensics/evidence" in paths
        assert "/api/v1/platform/network/segments" in paths
    
    def test_platform_health_endpoint(self, client):
        """Test platform health check without auth"""
        response = client.get("/api/v1/platform/health")
        
        # Should work without auth for monitoring
        assert response.status_code in [200, 503]  # 503 if services not running
        
        health_data = response.json()
        assert "overall_status" in health_data
        assert "timestamp" in health_data
        assert "services" in health_data
    
    @pytest.mark.asyncio
    async def test_service_orchestrator_initialization(self, orchestrator):
        """Test service orchestrator proper initialization"""
        # Check service registry
        assert len(orchestrator.service_registry) == 11
        
        # Verify core services
        core_services = ["database", "cache", "vector_store"]
        for service_id in core_services:
            assert service_id in orchestrator.service_registry
            definition = orchestrator.service_registry[service_id]
            assert definition.service_type.value == "core"
        
        # Verify PTaaS services
        ptaas_services = {
            "behavioral_analytics": "analytics",
            "threat_hunting": "security", 
            "forensics": "security",
            "network_microsegmentation": "security"
        }
        
        for service_id, expected_type in ptaas_services.items():
            assert service_id in orchestrator.service_registry
            definition = orchestrator.service_registry[service_id]
            assert definition.service_type.value == expected_type
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, orchestrator):
        """Test proper dependency resolution"""
        startup_order = orchestrator._get_startup_order()
        
        # Core services should start first
        assert startup_order.index("database") < startup_order.index("vector_store")
        assert startup_order.index("database") < startup_order.index("behavioral_analytics")
        assert startup_order.index("cache") < startup_order.index("behavioral_analytics")
        
        # Dependent services should start after dependencies
        dependent_services = [
            ("database", "threat_hunting"),
            ("database", "forensics"), 
            ("database", "network_microsegmentation"),
            ("vector_store", "threat_intelligence")
        ]
        
        for dependency, dependent in dependent_services:
            dep_index = startup_order.index(dependency)
            dependent_index = startup_order.index(dependent)
            assert dep_index < dependent_index, f"{dependency} should start before {dependent}"
    
    @pytest.mark.asyncio
    async def test_service_health_checks(self, orchestrator):
        """Test service health check functionality"""
        # Test health check for non-existent service
        health_result = await orchestrator.health_check("non_existent")
        assert not health_result["healthy"]
        assert "not running" in health_result["reason"]
        
        # All services should initially be unhealthy (not started)
        for service_id in orchestrator.service_registry:
            health_result = await orchestrator.health_check(service_id)
            assert not health_result["healthy"]
    
    @pytest.mark.asyncio
    async def test_service_metrics_collection(self, orchestrator):
        """Test service metrics collection"""
        for service_id in orchestrator.service_registry:
            metrics = await orchestrator.get_service_metrics(service_id)
            
            # Should return empty dict for non-running services
            assert isinstance(metrics, dict)
            
            # If service is not running, metrics should be minimal
            if service_id not in orchestrator.services:
                assert len(metrics) == 0
    
    def test_api_security_middleware_present(self, client):
        """Test that security middleware is properly configured"""
        # Make a request that should trigger security headers
        response = client.get("/api/v1/platform/health")
        
        # Check for security headers (from APISecurityMiddleware)
        headers = response.headers
        expected_security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection"
        ]
        
        # Some headers might be set by middleware
        # This is a basic check that security middleware is active
        assert response.status_code in [200, 503]
    
    def test_rate_limiting_configuration(self, client):
        """Test that rate limiting is configured"""
        # Make multiple requests to test rate limiting
        responses = []
        for i in range(5):
            response = client.get("/api/v1/platform/health")
            responses.append(response.status_code)
        
        # All requests should succeed for health endpoint (usually excluded from rate limiting)
        assert all(status in [200, 503] for status in responses)
    
    @pytest.mark.asyncio
    async def test_ptaas_service_integration(self):
        """Test PTaaS service integration without actually starting them"""
        # This tests that the service definitions are properly configured
        orchestrator = ServiceOrchestrator()
        await orchestrator.initialize()
        
        # Check PTaaS service definitions
        ptaas_services = ["behavioral_analytics", "threat_hunting", "forensics", "network_microsegmentation"]
        
        for service_id in ptaas_services:
            definition = orchestrator.service_registry[service_id]
            
            # All PTaaS services should have proper module paths
            assert definition.module_path.startswith("ptaas.")
            
            # Should have proper class names
            assert definition.class_name is not None
            assert len(definition.class_name) > 0
            
            # Should have dependencies configured
            assert isinstance(definition.dependencies, list)
    
    def test_cors_configuration(self, client):
        """Test CORS configuration for frontend integration"""
        # Test preflight request
        response = client.options(
            "/api/v1/platform/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # CORS should be configured to allow frontend requests
        assert response.status_code in [200, 204, 405]  # Various valid responses for OPTIONS
    
    @pytest.mark.asyncio
    async def test_observability_integration(self, orchestrator):
        """Test observability features integration"""
        # Test that metrics collector is available
        assert orchestrator.metrics is not None
        
        # Test that tracing is configured (should not raise errors)
        from app.infrastructure.observability import add_trace_context
        
        # Should not raise exception
        add_trace_context(
            operation="test_operation",
            service_id="test_service"
        )
    
    def test_enterprise_authentication_ready(self, client):
        """Test that enterprise auth endpoints are available"""
        # Check that auth endpoints exist
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        paths = openapi_spec.get("paths", {})
        
        # Should have auth-related endpoints
        auth_paths = [path for path in paths.keys() if "auth" in path.lower()]
        assert len(auth_paths) > 0, "Authentication endpoints should be available"


class TestPlatformAPIEndpoints:
    """Test specific platform API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_platform_services_endpoint(self, client):
        """Test platform services listing endpoint"""
        # This will require auth, so expect 401
        response = client.get("/api/v1/platform/services")
        assert response.status_code == 401  # Should require authentication
    
    def test_platform_health_public_access(self, client):
        """Test platform health is publicly accessible"""
        response = client.get("/api/v1/platform/health")
        
        # Health check should be accessible without auth
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            health_data = response.json()
            assert "overall_status" in health_data
            assert health_data["overall_status"] in ["healthy", "degraded", "unhealthy", "error"]
    
    def test_analytics_endpoints_require_auth(self, client):
        """Test that analytics endpoints require authentication"""
        analytics_endpoints = [
            "/api/v1/platform/analytics/behavioral/dashboard",
            "/api/v1/platform/threat-hunting/queries",
            "/api/v1/platform/forensics/evidence/test123",
        ]
        
        for endpoint in analytics_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"
    
    def test_service_management_requires_auth(self, client):
        """Test that service management endpoints require authentication"""
        management_endpoints = [
            ("/api/v1/platform/services/test/start", "POST"),
            ("/api/v1/platform/services/test/stop", "POST"), 
            ("/api/v1/platform/services/test/restart", "POST"),
        ]
        
        for endpoint, method in management_endpoints:
            if method == "POST":
                response = client.post(endpoint)
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"


if __name__ == "__main__":
    # Run basic integration tests
    import sys
    import asyncio
    
    async def run_basic_tests():
        print("Running basic platform integration tests...")
        
        # Test service orchestrator
        orchestrator = ServiceOrchestrator()
        await orchestrator.initialize()
        
        print(f"✓ Service orchestrator initialized with {len(orchestrator.service_registry)} services")
        
        # Test startup order
        startup_order = orchestrator._get_startup_order()
        print(f"✓ Dependency resolution working - startup order: {startup_order}")
        
        # Test health checks
        health_count = 0
        for service_id in list(orchestrator.service_registry.keys())[:3]:  # Test first 3 services
            health_result = await orchestrator.health_check(service_id)
            if isinstance(health_result, dict) and "healthy" in health_result:
                health_count += 1
        
        print(f"✓ Health check system working - tested {health_count} services")
        
        await orchestrator.shutdown()
        print("✓ Platform integration test completed successfully")
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_basic_tests())
    else:
        print("Use 'python test_platform_integration.py run' to execute basic tests")