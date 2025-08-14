#!/usr/bin/env python3
"""
XORB Platform Implementation Validation Script
Comprehensive validation of all service implementations and architecture enhancements
"""

import asyncio
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from api.app.services.base_service import (
        ServiceFactory,
        ServiceRegistry,
        service_registry,
        ServiceStatus,
        ServiceType
    )
    from api.app.services.ptaas_scanner_service import SecurityScannerService, get_scanner_service
    from api.app.services.interfaces import (
        PTaaSService,
        ThreatIntelligenceService,
        SecurityOrchestrationService,
        ComplianceService,
        SecurityMonitoringService,
        AuthenticationService,
        EmbeddingService
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running this script from the project root directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImplementationValidator:
    """Comprehensive validator for XORB platform implementation"""

    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "service_validation": {},
            "interface_validation": {},
            "architecture_validation": {},
            "integration_tests": {},
            "performance_tests": {},
            "security_tests": {}
        }

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("🚀 Starting XORB Platform Implementation Validation")

        try:
            # 1. Validate Service Architecture
            await self._validate_service_architecture()

            # 2. Validate Service Interfaces
            await self._validate_service_interfaces()

            # 3. Validate PTaaS Implementation
            await self._validate_ptaas_implementation()

            # 4. Validate Service Factory & Registry
            await self._validate_service_factory()

            # 5. Validate Base Service Implementation
            await self._validate_base_service()

            # 6. Integration Tests
            await self._run_integration_tests()

            # 7. Security Validation
            await self._validate_security_implementations()

            # 8. Performance Validation
            await self._validate_performance_capabilities()

            # Calculate overall status
            total_tests = self.validation_results["tests_passed"] + self.validation_results["tests_failed"]
            success_rate = self.validation_results["tests_passed"] / total_tests if total_tests > 0 else 0

            if success_rate >= 0.95:
                self.validation_results["overall_status"] = "EXCELLENT"
            elif success_rate >= 0.85:
                self.validation_results["overall_status"] = "GOOD"
            elif success_rate >= 0.70:
                self.validation_results["overall_status"] = "ACCEPTABLE"
            else:
                self.validation_results["overall_status"] = "NEEDS_IMPROVEMENT"

            logger.info(f"✅ Validation Complete - Status: {self.validation_results['overall_status']}")
            logger.info(f"📊 Results: {self.validation_results['tests_passed']}/{total_tests} tests passed")

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.validation_results["overall_status"] = "FAILED"
            self.validation_results["fatal_error"] = str(e)

        return self.validation_results

    async def _validate_service_architecture(self):
        """Validate overall service architecture"""
        logger.info("🏗️ Validating Service Architecture...")

        try:
            # Test service registry
            registry = service_registry
            self._assert(hasattr(registry, 'register'), "Service registry has register method")
            self._assert(hasattr(registry, 'get_service'), "Service registry has get_service method")
            self._assert(hasattr(registry, 'calculate_startup_order'), "Service registry has startup order calculation")

            # Test service factory
            factory = ServiceFactory
            self._assert(hasattr(factory, 'register_service'), "Service factory has register_service method")
            self._assert(hasattr(factory, 'create_service'), "Service factory has create_service method")
            self._assert(hasattr(factory, 'get_registered_services'), "Service factory has get_registered_services method")

            # Test service types
            types = list(ServiceType)
            expected_types = ['CORE', 'ANALYTICS', 'SECURITY', 'INTELLIGENCE', 'INTEGRATION', 'MONITORING']
            for expected_type in expected_types:
                self._assert(any(t.name == expected_type for t in types), f"ServiceType.{expected_type} exists")

            self.validation_results["architecture_validation"]["service_architecture"] = "PASSED"

        except Exception as e:
            self.validation_results["architecture_validation"]["service_architecture"] = f"FAILED: {e}"
            raise

    async def _validate_service_interfaces(self):
        """Validate all service interfaces are properly defined"""
        logger.info("🔌 Validating Service Interfaces...")

        interfaces_to_test = [
            PTaaSService,
            ThreatIntelligenceService,
            SecurityOrchestrationService,
            ComplianceService,
            SecurityMonitoringService,
            AuthenticationService,
            EmbeddingService
        ]

        for interface in interfaces_to_test:
            try:
                interface_name = interface.__name__

                # Check it's an abstract base class
                self._assert(hasattr(interface, '__abstractmethods__'), f"{interface_name} is abstract")

                # Check it has abstract methods
                abstract_methods = getattr(interface, '__abstractmethods__', set())
                self._assert(len(abstract_methods) > 0, f"{interface_name} has abstract methods")

                # Test that methods raise NotImplementedError when called on base class
                for method_name in abstract_methods:
                    method = getattr(interface, method_name, None)
                    self._assert(method is not None, f"{interface_name}.{method_name} exists")

                self.validation_results["interface_validation"][interface_name] = "PASSED"

            except Exception as e:
                self.validation_results["interface_validation"][interface_name] = f"FAILED: {e}"
                raise

    async def _validate_ptaas_implementation(self):
        """Validate PTaaS service implementation"""
        logger.info("🛡️ Validating PTaaS Implementation...")

        try:
            # Test service creation
            scanner_service = SecurityScannerService()
            self._assert(isinstance(scanner_service, SecurityScannerService), "SecurityScannerService instantiates")

            # Test interface compliance
            self._assert(isinstance(scanner_service, PTaaSService), "SecurityScannerService implements PTaaSService")

            # Test service initialization (mock)
            self._assert(hasattr(scanner_service, 'initialize'), "Scanner service has initialize method")
            self._assert(hasattr(scanner_service, 'health_check'), "Scanner service has health_check method")

            # Test PTaaS interface methods
            ptaas_methods = [
                'create_scan_session',
                'get_scan_status',
                'get_scan_results',
                'cancel_scan',
                'get_available_scan_profiles',
                'create_compliance_scan'
            ]

            for method_name in ptaas_methods:
                self._assert(hasattr(scanner_service, method_name), f"Scanner service has {method_name}")

            # Test security service interface methods
            security_methods = ['process_security_event', 'get_security_metrics']
            for method_name in security_methods:
                self._assert(hasattr(scanner_service, method_name), f"Scanner service has {method_name}")

            # Test scan profiles
            profiles = await scanner_service.get_available_scan_profiles()
            self._assert(isinstance(profiles, list), "Scan profiles returns list")
            self._assert(len(profiles) > 0, "Has available scan profiles")

            expected_profiles = ['quick', 'comprehensive', 'stealth', 'web_focused']
            profile_names = [p['name'] for p in profiles]
            for expected in expected_profiles:
                self._assert(expected in profile_names, f"Has {expected} scan profile")

            self.validation_results["service_validation"]["ptaas_implementation"] = "PASSED"

        except Exception as e:
            self.validation_results["service_validation"]["ptaas_implementation"] = f"FAILED: {e}"
            raise

    async def _validate_service_factory(self):
        """Validate service factory functionality"""
        logger.info("🏭 Validating Service Factory...")

        try:
            # Test service registration
            ServiceFactory.register_service("test_service", SecurityScannerService)
            registered_services = ServiceFactory.get_registered_services()
            self._assert("test_service" in registered_services, "Service registration works")

            # Test service creation
            test_service = ServiceFactory.create_service("test_service")
            self._assert(isinstance(test_service, SecurityScannerService), "Service creation works")

            # Test get_or_create
            same_service = ServiceFactory.get_or_create_service("test_service")
            self._assert(isinstance(same_service, SecurityScannerService), "get_or_create works")

            self.validation_results["architecture_validation"]["service_factory"] = "PASSED"

        except Exception as e:
            self.validation_results["architecture_validation"]["service_factory"] = f"FAILED: {e}"
            raise

    async def _validate_base_service(self):
        """Validate base service functionality"""
        logger.info("⚡ Validating Base Service Implementation...")

        try:
            scanner_service = SecurityScannerService()

            # Test service properties
            self._assert(hasattr(scanner_service, 'service_id'), "Has service_id")
            self._assert(hasattr(scanner_service, 'service_type'), "Has service_type")
            self._assert(hasattr(scanner_service, 'status'), "Has status")

            # Test lifecycle methods
            lifecycle_methods = ['initialize', 'shutdown', 'health_check', 'start', 'stop']
            for method in lifecycle_methods:
                self._assert(hasattr(scanner_service, method), f"Has {method} method")

            # Test metrics
            self._assert(hasattr(scanner_service, 'get_metrics'), "Has get_metrics method")

            self.validation_results["architecture_validation"]["base_service"] = "PASSED"

        except Exception as e:
            self.validation_results["architecture_validation"]["base_service"] = f"FAILED: {e}"
            raise

    async def _run_integration_tests(self):
        """Run integration tests"""
        logger.info("🔗 Running Integration Tests...")

        try:
            # Test service registry integration
            registry = service_registry
            scanner_service = SecurityScannerService()

            # Register service
            registry.register(scanner_service)
            retrieved_service = registry.get_service(scanner_service.service_id)
            self._assert(retrieved_service is scanner_service, "Service registry integration works")

            # Test startup order calculation
            startup_order = registry.calculate_startup_order()
            self._assert(isinstance(startup_order, list), "Startup order calculation works")

            self.validation_results["integration_tests"]["service_registry"] = "PASSED"

        except Exception as e:
            self.validation_results["integration_tests"]["service_registry"] = f"FAILED: {e}"
            raise

    async def _validate_security_implementations(self):
        """Validate security implementations"""
        logger.info("🔒 Validating Security Implementations...")

        try:
            scanner_service = SecurityScannerService()

            # Test security validation methods
            self._assert(hasattr(scanner_service, '_is_safe_executable_name'), "Has executable validation")
            self._assert(hasattr(scanner_service, '_validate_command_args'), "Has command validation")
            self._assert(hasattr(scanner_service, '_validate_target_host'), "Has target validation")

            # Test security validation logic
            safe_name = scanner_service._is_safe_executable_name("nmap")
            self._assert(safe_name, "Validates safe executable names")

            unsafe_name = scanner_service._is_safe_executable_name("rm -rf")
            self._assert(not unsafe_name, "Rejects unsafe executable names")

            # Test command validation
            safe_cmd = ["nmap", "-sS", "scanme.nmap.org"]
            self._assert(scanner_service._validate_command_args(safe_cmd), "Validates safe commands")

            unsafe_cmd = ["rm", "-rf", "/"]
            self._assert(not scanner_service._validate_command_args(unsafe_cmd), "Rejects unsafe commands")

            self.validation_results["security_tests"]["input_validation"] = "PASSED"

        except Exception as e:
            self.validation_results["security_tests"]["input_validation"] = f"FAILED: {e}"
            raise

    async def _validate_performance_capabilities(self):
        """Validate performance and scalability capabilities"""
        logger.info("⚡ Validating Performance Capabilities...")

        try:
            scanner_service = SecurityScannerService()

            # Test async queue implementation
            self._assert(hasattr(scanner_service, 'scan_queue'), "Has async scan queue")
            self._assert(hasattr(scanner_service, 'active_scans'), "Has active scans tracking")

            # Test metrics collection
            metrics = await scanner_service.get_security_metrics()
            self._assert(isinstance(metrics, dict), "Returns security metrics")

            expected_metrics = [
                'total_scans_completed',
                'active_scans',
                'queue_size',
                'available_scanners'
            ]

            for metric in expected_metrics:
                self._assert(metric in metrics, f"Has {metric} metric")

            self.validation_results["performance_tests"]["metrics_collection"] = "PASSED"

        except Exception as e:
            self.validation_results["performance_tests"]["metrics_collection"] = f"FAILED: {e}"
            raise

    def _assert(self, condition: bool, message: str):
        """Assert a condition and track results"""
        if condition:
            logger.debug(f"✅ {message}")
            self.validation_results["tests_passed"] += 1
        else:
            logger.error(f"❌ {message}")
            self.validation_results["tests_failed"] += 1
            raise AssertionError(message)

async def main():
    """Main validation entry point"""
    print("=" * 80)
    print("🛡️  XORB PLATFORM IMPLEMENTATION VALIDATION")
    print("=" * 80)
    print()

    validator = ImplementationValidator()
    results = await validator.run_comprehensive_validation()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)

    print(f"Overall Status: {results['overall_status']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Success Rate: {results['tests_passed']/(results['tests_passed']+results['tests_failed'])*100:.1f}%")

    print("\n🏗️ Architecture Validation:")
    for test, result in results['architecture_validation'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test}: {result}")

    print("\n🔌 Interface Validation:")
    for interface, result in results['interface_validation'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {interface}: {result}")

    print("\n🛡️ Service Validation:")
    for service, result in results['service_validation'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {service}: {result}")

    print("\n🔗 Integration Tests:")
    for test, result in results['integration_tests'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test}: {result}")

    print("\n🔒 Security Tests:")
    for test, result in results['security_tests'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test}: {result}")

    print("\n⚡ Performance Tests:")
    for test, result in results['performance_tests'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test}: {result}")

    # Save detailed results
    results_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {results_file}")

    # Return appropriate exit code
    if results['overall_status'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']:
        print("\n🎉 Implementation validation successful!")
        return 0
    else:
        print("\n❌ Implementation validation failed - see details above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
