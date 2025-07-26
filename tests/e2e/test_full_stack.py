"""
End-to-end tests for complete XORB stack.
"""

import pytest
import asyncio
import httpx
import time
from typing import Dict, Any


@pytest.mark.e2e
@pytest.mark.slow
class TestFullStackE2E:
    """End-to-end tests for the complete XORB ecosystem."""
    
    @pytest.fixture(scope="class")
    def stack_config(self):
        """Configuration for the test stack."""
        return {
            "api_url": "http://localhost:8000",
            "worker_url": "http://localhost:9000",
            "orchestrator_url": "http://localhost:8001",
            "timeout": 60
        }
    
    @pytest.mark.asyncio
    async def test_stack_health(self, stack_config):
        """Test that all services are healthy."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test API health
            response = await client.get(f"{stack_config['api_url']}/health")
            assert response.status_code == 200
            
            # Test Worker health
            response = await client.get(f"{stack_config['worker_url']}/health")
            assert response.status_code == 200
            
            # Test Orchestrator health
            response = await client.get(f"{stack_config['orchestrator_url']}/health")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_agent_execution_workflow(self, stack_config):
        """Test complete agent execution workflow."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. Create a campaign
            campaign_data = {
                "name": "e2e_test_campaign",
                "targets": ["httpbin.org"],
                "agents": ["discovery_agent"],
                "config": {
                    "timeout": 30,
                    "max_depth": 1
                }
            }
            
            response = await client.post(
                f"{stack_config['api_url']}/campaigns",
                json=campaign_data
            )
            assert response.status_code == 201
            campaign = response.json()
            campaign_id = campaign["id"]
            
            # 2. Wait for campaign to start
            await asyncio.sleep(5)
            
            # 3. Check campaign status
            response = await client.get(
                f"{stack_config['api_url']}/campaigns/{campaign_id}"
            )
            assert response.status_code == 200
            status = response.json()
            assert status["status"] in ["running", "pending", "completed"]
            
            # 4. Wait for completion or timeout
            max_wait = 60
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = await client.get(
                    f"{stack_config['api_url']}/campaigns/{campaign_id}"
                )
                status = response.json()
                
                if status["status"] == "completed":
                    break
                elif status["status"] == "failed":
                    pytest.fail(f"Campaign failed: {status.get('error', 'Unknown error')}")
                
                await asyncio.sleep(2)
            
            # 5. Verify results were generated
            response = await client.get(
                f"{stack_config['api_url']}/campaigns/{campaign_id}/results"
            )
            assert response.status_code == 200
            results = response.json()
            assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_fabric_integration(self, stack_config):
        """Test knowledge fabric integration."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test knowledge storage
            knowledge_data = {
                "type": "vulnerability",
                "content": "Test vulnerability finding",
                "confidence": 0.8,
                "source": "e2e_test"
            }
            
            response = await client.post(
                f"{stack_config['api_url']}/knowledge",
                json=knowledge_data
            )
            assert response.status_code == 201
            
            # Test knowledge retrieval
            response = await client.get(
                f"{stack_config['api_url']}/knowledge?source=e2e_test"
            )
            assert response.status_code == 200
            knowledge = response.json()
            assert len(knowledge) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_metrics(self, stack_config):
        """Test monitoring and metrics collection."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test metrics endpoints
            response = await client.get(f"{stack_config['api_url']}/metrics")
            assert response.status_code == 200
            metrics = response.text
            
            # Verify XORB-specific metrics are present
            assert "xorb_campaigns_total" in metrics or "xorb_" in metrics
            
            response = await client.get(f"{stack_config['worker_url']}/metrics")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, stack_config):
        """Test error handling and recovery mechanisms."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test invalid campaign creation
            invalid_campaign = {
                "name": "",  # Invalid empty name
                "targets": [],  # Invalid empty targets
                "agents": ["nonexistent_agent"]  # Invalid agent
            }
            
            response = await client.post(
                f"{stack_config['api_url']}/campaigns",
                json=invalid_campaign
            )
            assert response.status_code == 422  # Validation error
            
            # Test invalid endpoint
            response = await client.get(
                f"{stack_config['api_url']}/nonexistent_endpoint"
            )
            assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.security
class TestSecurityE2E:
    """End-to-end security tests."""
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, stack_config):
        """Test authentication and authorization."""
        # Test without authentication
        # Test with valid authentication
        # Test with invalid authentication
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, stack_config):
        """Test input sanitization across the stack."""
        # Test SQL injection protection
        # Test XSS protection
        # Test command injection protection
        assert True  # Placeholder