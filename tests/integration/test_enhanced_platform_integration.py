"""
Enhanced Platform Integration Tests
Comprehensive tests for the enhanced PTaaS platform with all advanced components
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Test framework imports
import pytest_asyncio

# Application imports
from src.api.app.enhanced_container import EnhancedContainer, get_enhanced_container
from src.api.app.services.enhanced_security_scanner_service import AdvancedSecurityScannerService
from src.api.app.services.advanced_threat_intelligence_engine import AdvancedThreatIntelligenceEngine
from src.api.app.services.advanced_orchestration_engine import AdvancedOrchestrationEngine
from src.api.app.services.advanced_reporting_engine import AdvancedReportingEngine
from src.api.app.infrastructure.enhanced_database_repositories import (
    EnhancedPostgreSQLScanSessionRepository, EnhancedPostgreSQLTenantRepository, EnhancedRedisCacheRepository
)
from src.api.app.domain.tenant_entities import ScanTarget


class TestEnhancedPlatformIntegration:
    """Integration tests for the enhanced PTaaS platform"""
    
    @pytest_asyncio.fixture
    async def enhanced_container(self):
        """Create enhanced container for testing"""
        # Override config for testing
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'test',
            'USE_PRODUCTION_DB': 'false',
            'REDIS_URL': 'redis://localhost:6379/15',  # Test database
            'ENABLE_ML_ANALYSIS': 'true',
            'ENABLE_THREAT_INTELLIGENCE': 'true',
            'ENABLE_ORCHESTRATION': 'true',
            'ENABLE_ADVANCED_REPORTING': 'true'
        }):
            container = EnhancedContainer()
            
            # Initialize services
            await container.initialize_all_services()
            
            yield container
            
            # Cleanup
            await container.shutdown_all_services()
    
    @pytest_asyncio.fixture
    async def sample_scan_target(self):
        """Create sample scan target for testing"""
        return ScanTarget(
            host="test.example.com",
            ports=[80, 443, 22],
            scan_profile="comprehensive",
            stealth_mode=False
        )
    
    @pytest_asyncio.fixture
    async def sample_scan_results(self):
        """Create sample scan results for testing"""
        return {
            "scan_id": "test_scan_001",
            "target": "test.example.com",
            "scan_type": "comprehensive",
            "status": "completed",
            "vulnerabilities": [
                {
                    "scanner": "nmap",
                    "name": "SSH Service Running",
                    "severity": "info",
                    "description": "SSH service detected on port 22",
                    "port": 22,
                    "service": "ssh",
                    "remediation": "Ensure SSH is properly configured"
                },
                {
                    "scanner": "nuclei",
                    "name": "Apache Server Info Disclosure",
                    "severity": "medium",
                    "description": "Apache server version disclosed",
                    "port": 80,
                    "service": "http",
                    "remediation": "Hide server version information"
                },
                {
                    "scanner": "nikto",
                    "name": "SQL Injection Vulnerability",
                    "severity": "high",
                    "description": "Potential SQL injection in login form",
                    "port": 80,
                    "service": "http",
                    "remediation": "Implement parameterized queries"
                }
            ],
            "open_ports": [
                {"port": 22, "protocol": "tcp", "service": "ssh"},
                {"port": 80, "protocol": "tcp", "service": "http"},
                {"port": 443, "protocol": "tcp", "service": "https"}
            ],
            "services": [
                {"port": 22, "name": "ssh", "version": "OpenSSH 8.0"},
                {"port": 80, "name": "http", "product": "Apache", "version": "2.4.41"},
                {"port": 443, "name": "https", "product": "Apache", "version": "2.4.41"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_container_initialization(self, enhanced_container):
        """Test that the enhanced container initializes all services correctly"""
        
        # Verify container is initialized
        assert enhanced_container is not None
        
        # Check service status
        status = enhanced_container.get_service_status()
        
        # Verify core metrics
        assert status["registered_services"] > 0
        assert status["initialized_services"] > 0
        assert len(status["initialization_order"]) > 0
        
        # Verify critical services are registered
        critical_services = [
            "cache_repository",
            "auth_service", 
            "scanner_service",
            "threat_intelligence_service",
            "orchestration_service",
            "reporting_service"
        ]
        
        for service_name in critical_services:
            assert service_name in status["services"]
            service_info = status["services"][service_name]
            assert service_info["initialized"] is True
            assert service_info["has_instance"] is True
    
    @pytest.mark.asyncio
    async def test_service_health_checks(self, enhanced_container):
        """Test health checks for all enhanced services"""
        
        # Perform health checks on all services
        health_results = await enhanced_container.health_check_all_services()
        
        # Verify overall health
        assert "overall_status" in health_results
        assert health_results["total_services"] > 0
        
        # Check individual service health
        services = health_results["services"]
        
        # Verify that most services are healthy (allow for some degraded states in test environment)
        healthy_count = sum(1 for s in services.values() if s["status"] == "healthy")
        total_count = len(services)
        
        # At least 70% of services should be healthy
        assert healthy_count / total_count >= 0.7, f"Only {healthy_count}/{total_count} services are healthy"
    
    @pytest.mark.asyncio
    async def test_enhanced_scanner_service_integration(self, enhanced_container, sample_scan_target):
        """Test enhanced scanner service integration"""
        
        # Get scanner service
        scanner_service = enhanced_container.get("scanner_service")
        assert isinstance(scanner_service, AdvancedSecurityScannerService)
        
        # Test service health
        health = await scanner_service.health_check()
        assert health is not None
        
        # Test scanner detection
        assert len(scanner_service.scanners) > 0
        
        # Test mock scan (since we don't have actual scanners in test environment)
        with patch.object(scanner_service, 'advanced_comprehensive_scan') as mock_scan:
            mock_scan.return_value = AsyncMock()
            mock_scan.return_value.scan_id = "test_scan_001"
            mock_scan.return_value.status = "completed"
            mock_scan.return_value.vulnerabilities = []
            
            result = await scanner_service.advanced_comprehensive_scan(sample_scan_target)
            assert result is not None
            mock_scan.assert_called_once_with(sample_scan_target)
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_integration(self, enhanced_container):
        """Test threat intelligence engine integration"""
        
        # Get threat intelligence service
        threat_intel_service = enhanced_container.get("threat_intelligence_service")
        assert isinstance(threat_intel_service, AdvancedThreatIntelligenceEngine)
        
        # Test service health
        health = await threat_intel_service.health_check()
        assert health is not None
        
        # Test indicator analysis
        test_indicators = [
            "192.0.2.1",
            "malicious-domain.com",
            "d41d8cd98f00b204e9800998ecf8427e"
        ]
        
        analysis_result = await threat_intel_service.analyze_indicators(test_indicators)
        
        # Verify analysis structure
        assert "total_indicators" in analysis_result
        assert "threat_score" in analysis_result
        assert "recommendations" in analysis_result
        assert analysis_result["total_indicators"] == len(test_indicators)
    
    @pytest.mark.asyncio
    async def test_orchestration_engine_integration(self, enhanced_container):
        """Test orchestration engine integration"""
        
        # Get orchestration service
        orchestration_service = enhanced_container.get("orchestration_service")
        assert isinstance(orchestration_service, AdvancedOrchestrationEngine)
        
        # Test service health
        health = await orchestration_service.health_check()
        assert health is not None
        
        # Test workflow creation
        workflow_definition = {
            "name": "Test Security Assessment",
            "description": "Test workflow for integration testing",
            "workflow_type": "security_assessment",
            "priority": "medium",
            "tasks": [
                {
                    "task_id": "test_scan",
                    "name": "Test Scan",
                    "task_type": "network_scan",
                    "description": "Test network scanning task",
                    "executor": "scanner_service",
                    "parameters": {"scan_type": "quick"}
                }
            ]
        }
        
        workflow_id = await orchestration_service.create_workflow(
            workflow_definition=workflow_definition,
            user_id="test_user",
            organization_id="test_org"
        )
        
        assert workflow_id is not None
        assert workflow_id.startswith("workflow_")
        
        # Verify workflow is registered
        assert workflow_id in orchestration_service.workflow_definitions
    
    @pytest.mark.asyncio
    async def test_reporting_engine_integration(self, enhanced_container, sample_scan_results):
        """Test reporting engine integration"""
        
        # Get reporting service
        reporting_service = enhanced_container.get("reporting_service")
        assert isinstance(reporting_service, AdvancedReportingEngine)
        
        # Test service health
        health = await reporting_service.health_check()
        assert health is not None
        
        # Test report generation
        report_config = {
            "report_type": "vulnerability_report",
            "format": "json",
            "title": "Test Vulnerability Report",
            "description": "Integration test report",
            "include_executive_summary": True,
            "include_technical_details": True,
            "include_recommendations": True
        }
        
        data_sources = {
            "scan_results": sample_scan_results,
            "organization": "Test Organization"
        }
        
        report_result = await reporting_service.generate_report(
            report_config=report_config,
            data_sources=data_sources
        )
        
        # Verify report structure
        assert report_result["status"] == "completed"
        assert "report_id" in report_result
        assert "metadata" in report_result
        assert report_result["file_size"] > 0
    
    @pytest.mark.asyncio
    async def test_database_repository_integration(self, enhanced_container):
        """Test enhanced database repository integration"""
        
        # Test cache repository
        cache_repo = enhanced_container.get("cache_repository")
        assert isinstance(cache_repo, EnhancedRedisCacheRepository)
        
        # Test cache operations
        test_key = "integration_test_key"
        test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        
        # Set and get
        success = await cache_repo.set(test_key, test_value, ttl=60)
        assert success is True
        
        retrieved_value = await cache_repo.get(test_key)
        assert retrieved_value is not None
        assert retrieved_value["test"] == "data"
        
        # Test existence and deletion
        exists = await cache_repo.exists(test_key)
        assert exists is True
        
        deleted = await cache_repo.delete(test_key)
        assert deleted is True
        
        exists_after_delete = await cache_repo.exists(test_key)
        assert exists_after_delete is False
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_assessment_workflow(self, enhanced_container, sample_scan_target):
        """Test complete end-to-end security assessment workflow"""
        
        # Get all required services
        scanner_service = enhanced_container.get("scanner_service")
        threat_intel_service = enhanced_container.get("threat_intelligence_service")
        orchestration_service = enhanced_container.get("orchestration_service")
        reporting_service = enhanced_container.get("reporting_service")
        
        # Mock the actual scanning to avoid external dependencies
        with patch.object(scanner_service, 'advanced_comprehensive_scan') as mock_scan:
            # Setup mock scan result
            mock_scan_result = MagicMock()
            mock_scan_result.scan_id = "e2e_test_scan"
            mock_scan_result.status = "completed"
            mock_scan_result.vulnerabilities = [
                {
                    "scanner": "nuclei",
                    "name": "Test Vulnerability",
                    "severity": "high",
                    "description": "Test vulnerability for e2e testing",
                    "port": 80,
                    "remediation": "Test remediation"
                }
            ]
            mock_scan_result.open_ports = [{"port": 80, "service": "http"}]
            mock_scan_result.services = [{"port": 80, "name": "http"}]
            mock_scan_result.recommendations = ["Test recommendation"]
            
            mock_scan.return_value = mock_scan_result
            
            # Step 1: Execute scan
            scan_result = await scanner_service.advanced_comprehensive_scan(sample_scan_target)
            assert scan_result.scan_id == "e2e_test_scan"
            assert scan_result.status == "completed"
            
            # Step 2: Analyze threats
            indicators = ["test.example.com", "192.0.2.1"]
            threat_analysis = await threat_intel_service.analyze_indicators(indicators)
            assert "threat_score" in threat_analysis
            
            # Step 3: Generate comprehensive report
            report_config = {
                "report_type": "technical_detailed",
                "format": "json",
                "title": "End-to-End Security Assessment",
                "description": "Comprehensive security assessment report"
            }
            
            data_sources = {
                "scan_results": {
                    "vulnerabilities": mock_scan_result.vulnerabilities,
                    "open_ports": mock_scan_result.open_ports,
                    "services": mock_scan_result.services
                },
                "threat_intelligence": threat_analysis,
                "organization": "Test Organization"
            }
            
            report_result = await reporting_service.generate_report(
                report_config=report_config,
                data_sources=data_sources
            )
            
            # Verify end-to-end workflow completion
            assert report_result["status"] == "completed"
            assert "report_id" in report_result
            
            # Parse report content to verify integration
            if "content" in report_result and report_result["content"]:
                content = json.loads(report_result["content"]) if isinstance(report_result["content"], str) else report_result["content"]
                metadata = content.get("metadata", {})
                report_content = content.get("content", {})
                
                # Verify report includes scan data
                assert "vulnerability_metrics" in report_content
                assert "threat_intelligence" in report_content
    
    @pytest.mark.asyncio
    async def test_service_dependency_resolution(self, enhanced_container):
        """Test that service dependencies are correctly resolved"""
        
        # Test that services with dependencies can access their dependencies
        orchestration_service = enhanced_container.get("orchestration_service")
        
        # Verify orchestration service has access to its dependencies
        assert orchestration_service.scanner_service is not None
        assert orchestration_service.threat_intel_service is not None
        
        # Test reporting service dependencies
        reporting_service = enhanced_container.get("reporting_service")
        
        # Verify reporting service is functional (has templates loaded)
        assert len(reporting_service.report_templates) > 0
        assert len(reporting_service.default_configs) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self, enhanced_container):
        """Test concurrent operations across multiple services"""
        
        # Get services
        cache_repo = enhanced_container.get("cache_repository")
        threat_intel_service = enhanced_container.get("threat_intelligence_service")
        
        # Test concurrent cache operations
        async def cache_operation(key: str, value: Any):
            await cache_repo.set(f"concurrent_test_{key}", value)
            result = await cache_repo.get(f"concurrent_test_{key}")
            await cache_repo.delete(f"concurrent_test_{key}")
            return result
        
        # Test concurrent threat analysis
        async def threat_analysis_operation(indicators: List[str]):
            return await threat_intel_service.analyze_indicators(indicators)
        
        # Execute operations concurrently
        cache_tasks = [
            cache_operation(f"key_{i}", {"data": f"value_{i}"})
            for i in range(5)
        ]
        
        threat_tasks = [
            threat_analysis_operation([f"192.0.2.{i}", f"test{i}.example.com"])
            for i in range(3)
        ]
        
        # Wait for all operations to complete
        cache_results = await asyncio.gather(*cache_tasks)
        threat_results = await asyncio.gather(*threat_tasks)
        
        # Verify results
        assert len(cache_results) == 5
        assert all(result is not None for result in cache_results)
        
        assert len(threat_results) == 3
        assert all("threat_score" in result for result in threat_results)
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, enhanced_container):
        """Test error handling across enhanced services"""
        
        threat_intel_service = enhanced_container.get("threat_intelligence_service")
        
        # Test with invalid indicators
        invalid_indicators = ["", None, "invalid_format_indicator"]
        
        # Should handle gracefully without crashing
        try:
            result = await threat_intel_service.analyze_indicators(invalid_indicators)
            # Should return some result even with invalid input
            assert "total_indicators" in result
        except Exception as e:
            # If exception is raised, it should be handled gracefully
            assert isinstance(e, (ValueError, TypeError))
    
    @pytest.mark.asyncio
    async def test_container_lifecycle_management(self):
        """Test container lifecycle management"""
        
        # Create new container
        container = EnhancedContainer()
        
        # Test initialization
        init_result = await container.initialize_all_services()
        assert init_result["initialized"] > 0
        assert init_result["failed"] == 0
        
        # Test health checks
        health_result = await container.health_check_all_services()
        assert "overall_status" in health_result
        
        # Test shutdown
        shutdown_result = await container.shutdown_all_services()
        assert shutdown_result["shutdown"] > 0
        assert shutdown_result["failed"] == 0


class TestEnhancedServicePerformance:
    """Performance tests for enhanced services"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, enhanced_container):
        """Test cache repository performance"""
        
        cache_repo = enhanced_container.get("cache_repository")
        
        # Test bulk operations
        start_time = datetime.utcnow()
        
        # Perform 100 cache operations
        for i in range(100):
            await cache_repo.set(f"perf_test_{i}", {"index": i, "timestamp": datetime.utcnow().isoformat()})
        
        for i in range(100):
            result = await cache_repo.get(f"perf_test_{i}")
            assert result is not None
            assert result["index"] == i
        
        for i in range(100):
            await cache_repo.delete(f"perf_test_{i}")
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete 300 operations in reasonable time (adjust threshold as needed)
        assert duration < 10.0, f"Cache operations took {duration} seconds, expected < 10 seconds"
    
    @pytest.mark.asyncio
    async def test_threat_analysis_performance(self, enhanced_container):
        """Test threat intelligence analysis performance"""
        
        threat_intel_service = enhanced_container.get("threat_intelligence_service")
        
        # Test with multiple indicators
        indicators = [
            f"192.0.2.{i}" for i in range(1, 51)  # 50 IP addresses
        ] + [
            f"test{i}.example.com" for i in range(1, 51)  # 50 domains
        ]
        
        start_time = datetime.utcnow()
        result = await threat_intel_service.analyze_indicators(indicators)
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        # Verify result
        assert result["total_indicators"] == 100
        
        # Should complete analysis in reasonable time
        assert duration < 30.0, f"Threat analysis took {duration} seconds, expected < 30 seconds"


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])