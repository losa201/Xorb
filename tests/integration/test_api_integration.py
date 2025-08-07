"""
Integration tests for XORB API service.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app.main import app


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus metrics format
        assert "xorb_" in response.text

    def test_agent_discovery_endpoint(self, client):
        """Test agent discovery endpoint."""
        response = client.get("/agents/discovery")
        assert response.status_code == 200
        agents = response.json()
        assert isinstance(agents, list)

    @pytest.mark.asyncio
    async def test_campaign_lifecycle(self, client):
        """Test complete campaign lifecycle."""
        # Create campaign
        campaign_data = {
            "name": "test_campaign",
            "targets": ["example.com"],
            "agents": ["discovery_agent"],
            "config": {"timeout": 300}
        }

        response = client.post("/campaigns", json=campaign_data)
        assert response.status_code == 201
        campaign = response.json()
        campaign_id = campaign["id"]

        # Get campaign status
        response = client.get(f"/campaigns/{campaign_id}")
        assert response.status_code == 200

        # Stop campaign
        response = client.post(f"/campaigns/{campaign_id}/stop")
        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseIntegration:
    """Integration tests with database."""

    @pytest.fixture
    def db_session(self):
        """Create database session for testing."""
        # Setup test database session

    def test_campaign_persistence(self, db_session):
        """Test campaign data persistence."""
        # Test database operations
        assert True  # Placeholder

    def test_agent_results_storage(self, db_session):
        """Test agent results storage."""
        # Test results storage
        assert True  # Placeholder
