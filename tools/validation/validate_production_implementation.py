#!/usr/bin/env python3
"""
Production Implementation Validation Script
Comprehensive validation of all production service implementations
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionImplementationValidator:
    """Comprehensive validator for production implementations"""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "validator": "Production Implementation Validator v1.0",
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "test_results": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        self.start_time = time.time()
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all production implementations"""
        logger.info("🚀 Starting Production Implementation Validation...")
        
        try:
            # Test service imports
            await self._test_service_imports()
            
            # Test service instantiation
            await self._test_service_instantiation()
            
            # Test authentication service
            await self._test_authentication_service()
            
            # Test PTaaS service
            await self._test_ptaas_service()
            
            # Test threat intelligence service
            await self._test_threat_intelligence_service()
            
            # Test service factory
            await self._test_service_factory()
            
            # Test production container
            await self._test_production_container()
            
            # Test router integration
            await self._test_router_integration()
            
            # Calculate final results
            self._calculate_final_results()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.validation_results["overall_status"] = "failed"
            self.validation_results["error"] = str(e)
            return self.validation_results
    
    async def _test_service_imports(self):
        """Test that all production services can be imported"""
        test_name = "service_imports"
        logger.info("📦 Testing Service Imports...")
        
        try:
            # Test production service implementations
            from src.api.app.services.production_service_implementations import (
                ProductionAuthenticationService, ProductionAuthorizationService,
                ProductionPTaaSService, ProductionHealthService, create_service_instances
            )
            
            # Test production intelligence service
            from src.api.app.services.production_intelligence_service import (
                ProductionThreatIntelligenceService
            )
            
            # Test production container
            from src.api.app.services.production_container_orchestrator import (
                ProductionServiceContainer, create_production_container
            )
            
            # Test production factory
            from src.api.app.services.production_service_factory import (
                ProductionServiceFactory, get_production_factory
            )
            
            # Test production router
            from src.api.app.routers.production_security_platform import router
            
            self._record_test_result(test_name, True, "All production services imported successfully")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Import failed: {e}")
    
    async def _test_service_instantiation(self):
        """Test that production services can be instantiated"""
        test_name = "service_instantiation"
        logger.info("🏗️ Testing Service Instantiation...")
        
        try:
            from src.api.app.services.production_service_implementations import (
                ProductionAuthenticationService, ProductionPTaaSService
            )
            
            # Test authentication service
            auth_service = ProductionAuthenticationService()
            assert auth_service is not None
            
            # Test PTaaS service
            ptaas_service = ProductionPTaaSService()
            assert ptaas_service is not None
            assert len(ptaas_service.scan_profiles) > 0
            
            self._record_test_result(test_name, True, "All services instantiated successfully")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Instantiation failed: {e}")
    
    async def _test_authentication_service(self):
        """Test authentication service functionality"""
        test_name = "authentication_service"
        logger.info("🔐 Testing Authentication Service...")
        
        try:
            from src.api.app.services.production_service_implementations import (
                ProductionAuthenticationService
            )
            
            auth_service = ProductionAuthenticationService()
            
            # Test password hashing
            password = "test_password"
            hashed = auth_service.hash_password(password)
            assert hashed != password
            assert auth_service.verify_password(password, hashed)
            
            # Test authentication
            credentials = {"username": "admin", "password": "admin"}
            auth_result = await auth_service.authenticate_user(credentials)
            assert auth_result["success"] is True
            assert "access_token" in auth_result
            
            # Test token validation
            token = auth_result["access_token"]
            validation_result = await auth_service.validate_token(token)
            assert validation_result["valid"] is True
            
            self._record_test_result(test_name, True, "Authentication service working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Authentication test failed: {e}")
    
    async def _test_ptaas_service(self):
        """Test PTaaS service functionality"""
        test_name = "ptaas_service"
        logger.info("🎯 Testing PTaaS Service...")
        
        try:
            from src.api.app.services.production_service_implementations import (
                ProductionPTaaSService
            )
            
            ptaas_service = ProductionPTaaSService()
            
            # Test scan profiles
            profiles = await ptaas_service.get_available_scan_profiles()
            assert len(profiles) >= 4  # Should have at least 4 profiles
            
            # Test scan session creation
            mock_user = {"id": "test-user-id"}
            mock_org = {"id": "test-org-id"}
            
            targets = [{"host": "scanme.nmap.org", "ports": [80, 443]}]
            scan_session = await ptaas_service.create_scan_session(
                targets=targets,
                scan_type="quick",
                user=mock_user,
                org=mock_org
            )
            
            assert "session_id" in scan_session
            assert scan_session["status"] == "created"
            
            # Test scan status
            session_id = scan_session["session_id"]
            status = await ptaas_service.get_scan_status(session_id, mock_user)
            assert "status" in status
            
            # Wait a moment for scan to start
            await asyncio.sleep(2)
            
            # Test scan cancellation
            cancelled = await ptaas_service.cancel_scan(session_id, mock_user)
            assert cancelled is True
            
            self._record_test_result(test_name, True, "PTaaS service working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"PTaaS test failed: {e}")
    
    async def _test_threat_intelligence_service(self):
        """Test threat intelligence service functionality"""
        test_name = "threat_intelligence_service"
        logger.info("🧠 Testing Threat Intelligence Service...")
        
        try:
            from src.api.app.services.production_intelligence_service import (
                ProductionThreatIntelligenceService
            )
            
            intel_service = ProductionThreatIntelligenceService()
            
            # Test indicator analysis
            mock_user = {"id": "test-user-id"}
            indicators = ["192.0.2.1", "malicious-domain.com", "hash123"]
            context = {"source": "test", "environment": "test"}
            
            analysis_result = await intel_service.analyze_indicators(
                indicators=indicators,
                context=context,
                user=mock_user
            )
            
            assert "analysis_id" in analysis_result
            assert "indicators" in analysis_result
            assert len(analysis_result["indicators"]) == len(indicators)
            assert "overall_threat_score" in analysis_result
            
            # Test threat correlation
            scan_results = {
                "vulnerabilities": [{"indicators": ["test.com"]}],
                "network_data": {"suspicious_ips": ["192.0.2.1"]}
            }
            
            correlation_result = await intel_service.correlate_threats(scan_results)
            assert "correlation_id" in correlation_result
            
            # Test threat prediction
            environment_data = {"asset_count": 100, "vulnerability_count": 5}
            prediction_result = await intel_service.get_threat_prediction(environment_data)
            assert "prediction_id" in prediction_result
            assert "predictions" in prediction_result
            
            self._record_test_result(test_name, True, "Threat intelligence service working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Threat intelligence test failed: {e}")
    
    async def _test_service_factory(self):
        """Test production service factory"""
        test_name = "service_factory"
        logger.info("🏭 Testing Service Factory...")
        
        try:
            from src.api.app.services.production_service_factory import (
                ProductionServiceFactory, get_production_factory
            )
            
            # Test factory creation
            config = {"environment": "test", "enable_ml_analysis": True}
            factory = ProductionServiceFactory(config)
            
            # Test service creation
            auth_service = factory.create_authentication_service()
            assert auth_service is not None
            
            ptaas_service = factory.create_ptaas_service()
            assert ptaas_service is not None
            
            intel_service = factory.create_threat_intelligence_service()
            assert intel_service is not None
            
            # Test all services creation
            all_services = factory.create_all_services()
            assert len(all_services) >= 5
            
            # Test status
            status = factory.get_service_status()
            assert "service_count" in status
            assert status["service_count"] >= 5
            
            self._record_test_result(test_name, True, "Service factory working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Service factory test failed: {e}")
    
    async def _test_production_container(self):
        """Test production container orchestrator"""
        test_name = "production_container"
        logger.info("📦 Testing Production Container...")
        
        try:
            from src.api.app.services.production_container_orchestrator import (
                ProductionServiceContainer, create_production_container
            )
            
            # Test container creation
            config = {"environment": "test"}
            container = await create_production_container(config)
            
            assert container is not None
            assert container._is_initialized is True
            
            # Test service retrieval
            auth_service = container.get_service("authentication_service")
            assert auth_service is not None
            
            ptaas_service = container.get_service("ptaas_service")
            assert ptaas_service is not None
            
            # Test health check
            health_results = await container.health_check_all_services()
            assert "overall_status" in health_results
            assert "services" in health_results
            
            # Test service status
            status = container.get_service_status()
            assert "container_initialized" in status
            assert status["container_initialized"] is True
            
            # Test shutdown
            shutdown_result = await container.shutdown_all_services()
            assert "shutdown" in shutdown_result
            
            self._record_test_result(test_name, True, "Production container working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Production container test failed: {e}")
    
    async def _test_router_integration(self):
        """Test router integration"""
        test_name = "router_integration"
        logger.info("🛣️ Testing Router Integration...")
        
        try:
            from src.api.app.routers.production_security_platform import router
            from fastapi import FastAPI
            
            # Test router import and configuration
            assert router is not None
            assert router.prefix == "/api/v1/production-security"
            assert len(router.routes) > 0
            
            # Test router can be added to app
            test_app = FastAPI()
            test_app.include_router(router)
            
            # Count endpoints
            endpoint_count = len([route for route in router.routes if hasattr(route, 'methods')])
            assert endpoint_count >= 10  # Should have at least 10 endpoints
            
            self._record_test_result(test_name, True, f"Router integration working with {endpoint_count} endpoints")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Router integration test failed: {e}")
    
    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        self.validation_results["test_results"][test_name] = {
            "passed": passed,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if passed:
            self.validation_results["tests_passed"] += 1
            logger.info(f"✅ {test_name}: {message}")
        else:
            self.validation_results["tests_failed"] += 1
            logger.error(f"❌ {test_name}: {message}")
        
        self.validation_results["total_tests"] += 1
    
    def _calculate_final_results(self):
        """Calculate final validation results"""
        total_tests = self.validation_results["total_tests"]
        passed_tests = self.validation_results["tests_passed"]
        
        if total_tests == 0:
            success_rate = 0.0
        else:
            success_rate = (passed_tests / total_tests) * 100
        
        # Performance metrics
        total_time = time.time() - self.start_time
        self.validation_results["performance_metrics"] = {
            "total_validation_time": f"{total_time:.2f} seconds",
            "tests_per_second": f"{total_tests / total_time:.2f}",
            "success_rate": f"{success_rate:.1f}%"
        }
        
        # Overall status
        if success_rate >= 90:
            self.validation_results["overall_status"] = "excellent"
        elif success_rate >= 75:
            self.validation_results["overall_status"] = "good"
        elif success_rate >= 50:
            self.validation_results["overall_status"] = "fair"
        else:
            self.validation_results["overall_status"] = "poor"
        
        # Recommendations
        if success_rate < 100:
            failed_tests = [
                test_name for test_name, result in self.validation_results["test_results"].items()
                if not result["passed"]
            ]
            self.validation_results["recommendations"].append(
                f"Review and fix failed tests: {', '.join(failed_tests)}"
            )
        
        if success_rate >= 90:
            self.validation_results["recommendations"].append(
                "Production implementation is ready for deployment"
            )
        else:
            self.validation_results["recommendations"].append(
                "Additional testing and fixes required before production deployment"
            )


async def main():
    """Main validation function"""
    print("🚀 XORB Production Implementation Validation")
    print("=" * 60)
    
    validator = ProductionImplementationValidator()
    results = await validator.run_comprehensive_validation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print(f"Success Rate: {results['performance_metrics']['success_rate']}")
    print(f"Validation Time: {results['performance_metrics']['total_validation_time']}")
    
    # Print test details
    print("\n📋 TEST DETAILS:")
    for test_name, result in results["test_results"].items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status} {test_name}: {result['message']}")
    
    # Print recommendations
    if results["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  • {rec}")
    
    # Save results to file
    with open("production_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full results saved to: production_validation_results.json")
    
    # Exit with appropriate code
    if results["overall_status"] in ["excellent", "good"]:
        print("\n🎉 Production implementation validation completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Production implementation needs attention before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())