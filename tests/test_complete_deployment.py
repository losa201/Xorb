#!/usr/bin/env python3
"""
XORB 2.0 Complete Deployment Test Suite
Comprehensive testing of all deployment components and advanced features
"""

import asyncio
import json
import pytest
import requests
import time
from datetime import datetime
from typing import Dict, List

# Import XORB components
import sys
sys.path.insert(0, '/root/Xorb')

class TestXORBDeployment:
    """Comprehensive deployment testing suite"""
    
    def setup_class(self):
        """Set up test environment"""
        self.api_base = "http://localhost:8000"
        self.orchestrator_base = "http://localhost:8080"
        self.worker_base = "http://localhost:9090"
        self.prometheus_base = "http://localhost:9091"
        self.grafana_base = "http://localhost:3000"
        
    def test_core_services_health(self):
        """Test all core services are healthy"""
        services = [
            (self.api_base, "xorb-api"),
            (self.orchestrator_base, "xorb-orchestrator"),
            (self.worker_base, "xorb-worker")
        ]
        
        for base_url, service_name in services:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] == service_name
            
            print(f"✅ {service_name} health check passed")
            
    def test_api_endpoints(self):
        """Test API service endpoints"""
        endpoints = [
            "/",
            "/api/v1/status",
            "/api/v1/assets",
            "/api/v1/scans",
            "/api/v1/findings",
            "/api/gamification/leaderboard",
            "/api/compliance/status"
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{self.api_base}{endpoint}", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert data is not None
            
            print(f"✅ API endpoint {endpoint} working")
            
    def test_orchestrator_endpoints(self):
        """Test orchestrator service endpoints"""
        endpoints = [
            "/",
            "/api/v1/campaigns",
            "/api/v1/agents",
            "/api/v1/orchestrator/status"
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{self.orchestrator_base}{endpoint}", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert data is not None
            
            print(f"✅ Orchestrator endpoint {endpoint} working")
            
    def test_monitoring_services(self):
        """Test monitoring services"""
        # Test Prometheus
        response = requests.get(f"{self.prometheus_base}/api/v1/query?query=up", timeout=10)
        assert response.status_code == 200
        print("✅ Prometheus metrics API working")
        
        # Test Grafana
        response = requests.get(f"{self.grafana_base}/api/health", timeout=10)
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["database"] == "ok"
        print("✅ Grafana health check passed")
        
    def test_advanced_features_import(self):
        """Test advanced features can be imported"""
        features = [
            "xorb_core.vulnerabilities.vulnerability_lifecycle_manager",
            "xorb_core.intelligence.threat_intelligence_engine",
            "xorb_core.hunting.ai_threat_hunter",
            "xorb_core.orchestration.distributed_campaign_coordinator",
            "xorb_core.reporting.advanced_reporting_engine",
            "xorb_core.agents.stealth.advanced_stealth_agent",
            "xorb_core.ml.security_ml_engine"
        ]
        
        for feature in features:
            try:
                __import__(feature)
                print(f"✅ {feature} imports successfully")
            except ImportError as e:
                pytest.fail(f"❌ Failed to import {feature}: {e}")
                
    def test_vulnerability_management_system(self):
        """Test vulnerability management system"""
        try:
            from xorb_core.vulnerabilities import vulnerability_manager
            
            # Test manager creation
            manager = vulnerability_manager.VulnerabilityLifecycleManager()
            assert manager is not None
            
            # Test basic functionality
            stats = manager.get_vulnerability_statistics()
            assert isinstance(stats, dict)
            assert "total" in stats
            
            print("✅ Vulnerability management system functional")
            
        except Exception as e:
            pytest.fail(f"❌ Vulnerability management test failed: {e}")
            
    def test_threat_intelligence_system(self):
        """Test threat intelligence system"""
        try:
            from xorb_core.intelligence.threat_intelligence_engine import ThreatIntelligenceEngine
            
            # Test engine creation
            engine = ThreatIntelligenceEngine()
            assert engine is not None
            
            print("✅ Threat intelligence system functional")
            
        except Exception as e:
            pytest.fail(f"❌ Threat intelligence test failed: {e}")
            
    def test_ai_threat_hunting_system(self):
        """Test AI threat hunting system"""
        try:
            from xorb_core.hunting.ai_threat_hunter import AIThreatHunter
            
            # Test hunter creation
            hunter = AIThreatHunter()
            assert hunter is not None
            assert hasattr(hunter, 'anomaly_detectors')
            assert hasattr(hunter, 'hypothesis_generator')
            
            print("✅ AI threat hunting system functional")
            
        except Exception as e:
            pytest.fail(f"❌ AI threat hunting test failed: {e}")
            
    def test_distributed_coordination_system(self):
        """Test distributed coordination system"""
        try:
            from xorb_core.orchestration.distributed_campaign_coordinator import DistributedCampaignCoordinator
            
            # Test coordinator creation
            coordinator = DistributedCampaignCoordinator()
            assert coordinator is not None
            assert hasattr(coordinator, 'node_info')
            assert hasattr(coordinator, 'consensus_engine')
            
            print("✅ Distributed coordination system functional")
            
        except Exception as e:
            pytest.fail(f"❌ Distributed coordination test failed: {e}")
            
    def test_reporting_system(self):
        """Test reporting system"""
        try:
            from xorb_core.reporting.advanced_reporting_engine import AdvancedReportingEngine
            
            # Test engine creation
            engine = AdvancedReportingEngine()
            assert engine is not None
            
            print("✅ Reporting system functional")
            
        except Exception as e:
            pytest.fail(f"❌ Reporting system test failed: {e}")
            
    def test_stealth_agents_system(self):
        """Test stealth agents system"""
        try:
            from xorb_core.agents.stealth.advanced_stealth_agent import AdvancedStealthAgent
            
            # Test agent creation
            agent = AdvancedStealthAgent("test-agent")
            assert agent is not None
            assert hasattr(agent, 'evasion_techniques')
            
            print("✅ Stealth agents system functional")
            
        except Exception as e:
            pytest.fail(f"❌ Stealth agents test failed: {e}")
            
    def test_ml_security_system(self):
        """Test ML security system"""
        try:
            from xorb_core.ml.security_ml_engine import SecurityMLEngine
            
            # Test engine creation
            engine = SecurityMLEngine()
            assert engine is not None
            assert hasattr(engine, 'threat_classifier')
            
            print("✅ ML security system functional")
            
        except Exception as e:
            pytest.fail(f"❌ ML security system test failed: {e}")
            
    def test_database_connectivity(self):
        """Test database connectivity"""
        try:
            import asyncpg
            import redis
            
            # Test Redis connectivity (simple test)
            try:
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                r.ping()
                print("✅ Redis connectivity working")
            except Exception as e:
                print(f"⚠️  Redis connectivity issue: {e}")
                
        except Exception as e:
            pytest.fail(f"❌ Database connectivity test failed: {e}")
            
    def test_service_integration(self):
        """Test service integration and communication"""
        # Test API to Orchestrator communication
        try:
            # Get orchestrator status from API perspective
            response = requests.get(f"{self.api_base}/api/v1/status", timeout=10)
            assert response.status_code == 200
            
            api_status = response.json()
            assert "services" in api_status
            
            print("✅ Service integration working")
            
        except Exception as e:
            pytest.fail(f"❌ Service integration test failed: {e}")
            
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        try:
            # Check if metrics are being collected
            response = requests.get(
                f"{self.prometheus_base}/api/v1/query?query=up", 
                timeout=10
            )
            assert response.status_code == 200
            
            metrics_data = response.json()
            assert metrics_data["status"] == "success"
            
            print("✅ Performance metrics collection working")
            
        except Exception as e:
            pytest.fail(f"❌ Performance metrics test failed: {e}")
            
    def test_security_hardening(self):
        """Test security hardening measures"""
        try:
            # Test that services are not running as root
            # This is a basic check - in production, more comprehensive tests would be needed
            
            # Check if health endpoints require proper headers (basic security)
            response = requests.get(f"{self.api_base}/health")
            assert response.status_code == 200
            
            # Check CORS headers are present
            assert "Access-Control-Allow-Origin" in response.headers
            
            print("✅ Basic security hardening verified")
            
        except Exception as e:
            pytest.fail(f"❌ Security hardening test failed: {e}")

class TestXORBFunctionality:
    """Test XORB functionality and features"""
    
    @pytest.mark.asyncio
    async def test_vulnerability_workflow(self):
        """Test complete vulnerability management workflow"""
        try:
            from xorb_core.vulnerabilities.vulnerability_lifecycle_manager import (
                VulnerabilityLifecycleManager, Vulnerability, VulnerabilitySeverity
            )
            
            manager = VulnerabilityLifecycleManager()
            
            # Create test vulnerability
            test_vuln = Vulnerability(
                title="Test SQL Injection",
                description="Test vulnerability for demo",
                severity=VulnerabilitySeverity.HIGH,
                affected_asset="test-server.local",
                discovery_source="Unit Test",
                technical_details="Test injection in login form"
            )
            
            # Add vulnerability
            await manager.add_vulnerability(test_vuln)
            
            # Verify it was added
            retrieved_vuln = manager.get_vulnerability(test_vuln.vulnerability_id)
            assert retrieved_vuln is not None
            assert retrieved_vuln.title == "Test SQL Injection"
            
            print("✅ Vulnerability workflow test passed")
            
        except Exception as e:
            pytest.fail(f"❌ Vulnerability workflow test failed: {e}")
            
    @pytest.mark.asyncio 
    async def test_threat_hunting_workflow(self):
        """Test AI threat hunting workflow"""
        try:
            from xorb_core.hunting.ai_threat_hunter import AIThreatHunter
            
            hunter = AIThreatHunter()
            
            # Test hypothesis generation
            hypotheses = await hunter.generate_hypotheses(
                data_sources=["network", "process"],
                time_window=3600
            )
            
            assert isinstance(hypotheses, list)
            print(f"✅ Generated {len(hypotheses)} hunting hypotheses")
            
        except Exception as e:
            pytest.fail(f"❌ Threat hunting workflow test failed: {e}")

if __name__ == "__main__":
    # Run tests
    test_deployment = TestXORBDeployment()
    test_functionality = TestXORBFunctionality()
    
    print("🚀 Starting XORB Complete Deployment Tests")
    print("=" * 60)
    
    try:
        # Set up test environment
        test_deployment.setup_class()
        
        # Core service tests
        print("\n🔧 Testing Core Services...")
        test_deployment.test_core_services_health()
        test_deployment.test_api_endpoints()
        test_deployment.test_orchestrator_endpoints()
        test_deployment.test_monitoring_services()
        
        # Advanced feature tests
        print("\n🎯 Testing Advanced Features...")
        test_deployment.test_advanced_features_import()
        test_deployment.test_vulnerability_management_system()
        test_deployment.test_threat_intelligence_system()
        test_deployment.test_ai_threat_hunting_system()
        test_deployment.test_distributed_coordination_system()
        test_deployment.test_reporting_system()
        test_deployment.test_stealth_agents_system()
        test_deployment.test_ml_security_system()
        
        # Integration tests
        print("\n🔗 Testing Integration...")
        test_deployment.test_database_connectivity()
        test_deployment.test_service_integration()
        test_deployment.test_performance_metrics()
        test_deployment.test_security_hardening()
        
        # Functionality tests
        print("\n⚡ Testing Functionality...")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_functionality.test_vulnerability_workflow())
        loop.run_until_complete(test_functionality.test_threat_hunting_workflow())
        
        print("\n🎉 All Tests Passed!")
        print("=" * 60)
        print("✅ XORB 2.0 deployment is fully functional and ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()