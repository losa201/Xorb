#!/usr/bin/env python3
"""
XORB Configuration Validation Test Suite

Validates the auto-configuration system across different hardware profiles
to ensure proper detection, classification, and optimization.

Author: XORB DevOps AI
Version: 2.0.0
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import unittest
from unittest.mock import patch, MagicMock

# Import the autoconfigure module
import sys
sys.path.append(str(Path(__file__).parent))
from autoconfigure import XorbAutoConfigurator, SystemProfile, XorbMode, SystemCapabilities


class TestXorbAutoConfiguration(unittest.TestCase):
    """Test suite for XORB auto-configuration system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.configurator = XorbAutoConfigurator()
        self.configurator.project_root = Path(self.temp_dir)
        self.configurator.logs_dir = Path(self.temp_dir) / "logs"
        self.configurator.logs_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_capabilities(self, profile: SystemProfile) -> SystemCapabilities:
        """Create mock system capabilities for testing"""
        profiles = {
            SystemProfile.RPI: {
                'cpu_cores': 4, 'cpu_threads': 4, 'ram_total_gb': 4.0,
                'is_arm': True, 'is_virtualized': False
            },
            SystemProfile.CLOUD_MICRO: {
                'cpu_cores': 1, 'cpu_threads': 2, 'ram_total_gb': 2.0,
                'is_arm': False, 'is_virtualized': True
            },
            SystemProfile.CLOUD_SMALL: {
                'cpu_cores': 2, 'cpu_threads': 4, 'ram_total_gb': 4.0,
                'is_arm': False, 'is_virtualized': True
            },
            SystemProfile.CLOUD_MEDIUM: {
                'cpu_cores': 4, 'cpu_threads': 8, 'ram_total_gb': 8.0,
                'is_arm': False, 'is_virtualized': True
            },
            SystemProfile.BARE_METAL: {
                'cpu_cores': 16, 'cpu_threads': 32, 'ram_total_gb': 32.0,
                'is_arm': False, 'is_virtualized': False
            },
            SystemProfile.EPYC_SERVER: {
                'cpu_cores': 64, 'cpu_threads': 128, 'ram_total_gb': 128.0,
                'is_arm': False, 'is_virtualized': False
            }
        }
        
        data = profiles[profile]
        
        return SystemCapabilities(
            os_type="Linux",
            os_version="6.8.0",
            architecture="aarch64" if data['is_arm'] else "x86_64",
            cpu_cores=data['cpu_cores'],
            cpu_threads=data['cpu_threads'],
            cpu_frequency=2400.0,
            ram_total_gb=data['ram_total_gb'],
            ram_available_gb=data['ram_total_gb'] * 0.8,
            disk_space_gb=50.0,
            is_arm=data['is_arm'],
            is_virtualized=data['is_virtualized'],
            docker_version="24.0.7",
            docker_buildkit=True,
            docker_compose_version="2.21.0",
            podman_available=False,
            network_interfaces=["eth0", "lo"],
            dns_servers=["8.8.8.8"],
            system_load=0.5,
            profile=profile
        )
    
    def test_raspberry_pi_configuration(self):
        """Test Raspberry Pi configuration"""
        capabilities = self.create_mock_capabilities(SystemProfile.RPI)
        config = self.configurator.generate_configuration(capabilities)
        
        # Verify Raspberry Pi specific settings
        self.assertEqual(config.mode, XorbMode.SIMPLE)
        self.assertEqual(config.system_profile, SystemProfile.RPI)
        self.assertEqual(config.agent_concurrency, 2)
        self.assertEqual(config.max_concurrent_missions, 1)
        self.assertFalse(config.monitoring_enabled)
        self.assertIn("postgres", config.services_enabled)
        self.assertIn("api", config.services_enabled)
        self.assertNotIn("prometheus", config.services_enabled)
        
        # Check environment variables
        self.assertEqual(config.environment_variables["XORB_IS_ARM"], "true")
        self.assertEqual(config.environment_variables["XORB_PI5_OPTIMIZATION"], "true")
    
    def test_cloud_micro_configuration(self):
        """Test cloud micro instance configuration"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MICRO)
        config = self.configurator.generate_configuration(capabilities)
        
        self.assertEqual(config.mode, XorbMode.SIMPLE)
        self.assertEqual(config.agent_concurrency, 4)
        self.assertEqual(config.max_concurrent_missions, 2)
        self.assertFalse(config.monitoring_enabled)
    
    def test_cloud_medium_configuration(self):
        """Test cloud medium instance configuration"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        self.assertEqual(config.mode, XorbMode.ENHANCED)
        self.assertEqual(config.agent_concurrency, 16)
        self.assertEqual(config.max_concurrent_missions, 5)
        self.assertTrue(config.monitoring_enabled)
        self.assertIn("prometheus", config.services_enabled)
    
    def test_epyc_server_configuration(self):
        """Test EPYC server configuration"""
        capabilities = self.create_mock_capabilities(SystemProfile.EPYC_SERVER)
        config = self.configurator.generate_configuration(capabilities)
        
        self.assertEqual(config.mode, XorbMode.FULL)
        self.assertEqual(config.agent_concurrency, 64)
        self.assertEqual(config.max_concurrent_missions, 20)
        self.assertTrue(config.monitoring_enabled)
        self.assertIn("grafana", config.services_enabled)
        self.assertIn("tempo", config.services_enabled)
    
    def test_resource_limits_generation(self):
        """Test resource limits generation"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        # Check that resource limits are generated
        self.assertIsInstance(config.resource_limits, dict)
        
        # Check postgres limits
        if "postgres" in config.resource_limits:
            postgres_limits = config.resource_limits["postgres"]
            self.assertIn("deploy", postgres_limits)
            self.assertIn("resources", postgres_limits["deploy"])
            
    def test_environment_variables_generation(self):
        """Test environment variables generation"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        env_vars = config.environment_variables
        
        # Check required variables
        required_vars = [
            "XORB_MODE", "XORB_AGENT_CONCURRENCY", "XORB_MONITORING_ENABLED",
            "XORB_SYSTEM_PROFILE", "DOCKER_BUILDKIT"
        ]
        
        for var in required_vars:
            self.assertIn(var, env_vars)
        
        # Check values
        self.assertEqual(env_vars["XORB_MODE"], "CLOUD_MEDIUM")
        self.assertEqual(env_vars["XORB_AGENT_CONCURRENCY"], "16")
        self.assertEqual(env_vars["DOCKER_BUILDKIT"], "1")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        # Valid configuration should pass
        self.assertTrue(self.configurator.validate_configuration(config))
        
        # Invalid configuration should fail
        config.agent_concurrency = 0
        self.assertFalse(self.configurator.validate_configuration(config))
        
        config.agent_concurrency = 16
        config.services_enabled = []
        self.assertFalse(self.configurator.validate_configuration(config))
    
    def test_env_file_writing(self):
        """Test environment file writing"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        env_file = self.configurator.write_environment_file(config)
        
        # Check file was created
        self.assertTrue(env_file.exists())
        
        # Check content
        content = env_file.read_text()
        self.assertIn("XORB_MODE=CLOUD_MEDIUM", content)
        self.assertIn("XORB_AGENT_CONCURRENCY=16", content)
        self.assertIn("Generated:", content)
    
    def test_bootstrap_report_generation(self):
        """Test bootstrap report generation"""
        capabilities = self.create_mock_capabilities(SystemProfile.CLOUD_MEDIUM)
        config = self.configurator.generate_configuration(capabilities)
        
        report_file = self.configurator.generate_bootstrap_report(capabilities, config)
        
        # Check file was created
        self.assertTrue(report_file.exists())
        
        # Check content
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        self.assertIn("system_capabilities", report)
        self.assertIn("generated_configuration", report)
        self.assertIn("deployment_readiness", report)
        self.assertEqual(report["system_capabilities"]["profile"], "CLOUD_MEDIUM")
    
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('platform.machine')
    def test_system_detection_mocking(self, mock_machine, mock_memory, mock_cpu):
        """Test system detection with mocked values"""
        # Mock ARM system
        mock_machine.return_value = "aarch64"
        mock_cpu.return_value = 4
        mock_memory.return_value = MagicMock(total=4*1024**3, available=3*1024**3)
        
        capabilities = self.configurator.detect_system_capabilities()
        
        self.assertTrue(capabilities.is_arm)
        self.assertEqual(capabilities.cpu_cores, 4)
        self.assertEqual(capabilities.profile, SystemProfile.RPI)
    
    def test_performance_tuning_values(self):
        """Test performance tuning value generation"""
        # Test cycle times
        self.assertEqual(self.configurator._get_cycle_time(SystemProfile.RPI), "800")
        self.assertEqual(self.configurator._get_cycle_time(SystemProfile.EPYC_SERVER), "100")
        
        # Test pool sizes
        self.assertEqual(self.configurator._get_db_pool_size(SystemProfile.RPI), "5")
        self.assertEqual(self.configurator._get_db_pool_size(SystemProfile.EPYC_SERVER), "50")


class TestProfileClassification(unittest.TestCase):
    """Test system profile classification logic"""
    
    def setUp(self):
        self.configurator = XorbAutoConfigurator()
    
    def test_arm_classification(self):
        """Test ARM device classification"""
        profile = self.configurator._classify_system_profile(4, 4.0, True, False)
        self.assertEqual(profile, SystemProfile.RPI)
    
    def test_high_end_server_classification(self):
        """Test high-end server classification"""
        profile = self.configurator._classify_system_profile(32, 64.0, False, False)
        self.assertEqual(profile, SystemProfile.EPYC_SERVER)
        
        profile = self.configurator._classify_system_profile(16, 32.0, False, False)
        self.assertEqual(profile, SystemProfile.BARE_METAL)
    
    def test_cloud_classification(self):
        """Test cloud instance classification"""
        profile = self.configurator._classify_system_profile(2, 2.0, False, True)
        self.assertEqual(profile, SystemProfile.CLOUD_MICRO)
        
        profile = self.configurator._classify_system_profile(4, 8.0, False, True)
        self.assertEqual(profile, SystemProfile.CLOUD_SMALL)
        
        profile = self.configurator._classify_system_profile(8, 16.0, False, True)
        self.assertEqual(profile, SystemProfile.CLOUD_MEDIUM)


def run_validation_tests():
    """Run all validation tests"""
    print("🧪 Running XORB Auto-Configuration Validation Tests")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestXorbAutoConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestProfileClassification))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ All tests passed! Auto-configuration system is working correctly.")
    else:
        print(f"❌ {len(result.failures + result.errors)} tests failed.")
        for test, error in result.failures + result.errors:
            print(f"   - {test}: {error.split('AssertionError:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)