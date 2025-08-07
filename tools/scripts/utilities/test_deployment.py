#!/usr/bin/env python3
"""
XORB Enterprise Deployment Test Suite
Validates the deployment system functionality
"""

import asyncio
import sys
import logging
from pathlib import Path
from deploy_xorb_enterprise import XORBEnterpriseDeployer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentTester:
    """Test suite for XORB enterprise deployment"""
    
    def __init__(self):
        self.deployer = XORBEnterpriseDeployer('config/deployment.yaml')
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all deployment tests"""
        logger.info("🧪 Starting XORB Deployment Test Suite")
        
        tests = [
            ('Configuration Loading', self.test_config_loading),
            ('Environment Validation', self.test_environment_validation),
            ('Docker Availability', self.test_docker_check),
            ('Resource Checks', self.test_resource_checks),
            ('Configuration Generation', self.test_config_generation),
            ('Report Generation', self.test_report_generation)
        ]
        
        passed = 0 
        failed = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"🔍 Running test: {test_name}")
                await test_func()
                logger.info(f"✅ Test passed: {test_name}")
                self.test_results.append((test_name, "PASSED", None))
                passed += 1
            except Exception as e:
                logger.error(f"❌ Test failed: {test_name} - {e}")
                self.test_results.append((test_name, "FAILED", str(e)))
                failed += 1
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("🧪 DEPLOYMENT TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Tests Passed: {passed}")
        logger.info(f"❌ Tests Failed: {failed}")
        logger.info(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        return passed, failed
    
    async def test_config_loading(self):
        """Test configuration loading"""
        config = self.deployer.load_deployment_config()
        assert config is not None, "Configuration should not be None"
        assert 'deployment' in config, "Configuration should have deployment section"
        assert 'services' in config, "Configuration should have services section"
        
    async def test_environment_validation(self):
        """Test environment validation components"""
        await self.deployer.check_domain_configuration()
        await self.deployer.check_resource_availability()
        
    async def test_docker_check(self):
        """Test Docker availability check"""
        await self.deployer.check_docker_availability()
        
    async def test_resource_checks(self):
        """Test resource availability checks"""
        await self.deployer.check_resource_availability()
        
    async def test_config_generation(self):
        """Test Kubernetes configuration generation"""
        await self.deployer.create_configuration()
        
    async def test_report_generation(self):
        """Test deployment report generation"""
        report = await self.deployer.generate_deployment_report()
        
        assert report is not None, "Report should not be None"
        assert 'deployment_id' in report, "Report should have deployment ID"
        assert 'timestamp' in report, "Report should have timestamp"

async def main():
    """Main test function"""
    tester = DeploymentTester()
    
    try:
        passed, failed = await tester.run_all_tests()
        
        if failed > 0:
            logger.warning(f"⚠️ Some tests failed. Please review the issues above.")
            sys.exit(1)
        else:
            logger.info("🎉 All tests passed! Deployment system is ready.")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())