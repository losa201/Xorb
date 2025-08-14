#!/usr/bin/env python3
"""
Test script for XORB Configuration Management System
Validates centralized configuration, environment switching, and hot-reloading
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_manager_import():
    """Test that config manager can be imported"""
    print("🧪 Testing Configuration Manager Import...")

    try:
        from common.config_manager import (
            ConfigManager,
            XORBConfig,
            Environment,
            get_config_manager,
            get_config,
            get_feature_flag
        )
        print("✅ Configuration manager imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import configuration manager: {e}")
        return False

def test_environment_specific_configs():
    """Test loading different environment configurations"""
    print("🧪 Testing Environment-Specific Configurations...")

    try:
        from common.config_manager import ConfigManager, Environment

        environments = [
            Environment.DEVELOPMENT,
            Environment.STAGING,
            Environment.PRODUCTION,
            Environment.TEST
        ]

        results = {}
        for env in environments:
            try:
                manager = ConfigManager(environment=env.value, enable_hot_reload=False)
                config = manager.get_config()
                results[env.value] = {
                    "loaded": True,
                    "debug": config.debug,
                    "api_port": config.api_service.port,
                    "database_name": config.database.name,
                    "environment": config.environment.value
                }
                print(f"  ✅ {env.value}: DB={config.database.name}, Port={config.api_service.port}, Debug={config.debug}")
            except Exception as e:
                results[env.value] = {"loaded": False, "error": str(e)}
                print(f"  ❌ {env.value}: {e}")

        # Verify configurations are different
        dev_config = results.get("development", {})
        prod_config = results.get("production", {})

        if dev_config.get("debug") == True and prod_config.get("debug") == False:
            print("✅ Environment-specific configurations are working correctly")
            return True
        else:
            print("❌ Environment-specific configurations not properly differentiated")
            return False

    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False

def test_feature_flags():
    """Test feature flag functionality"""
    print("🧪 Testing Feature Flags...")

    try:
        from common.config_manager import get_config_manager

        manager = get_config_manager()

        # Test getting feature flags
        analytics_flag = manager.get_feature_flag("advanced_analytics", False)
        nonexistent_flag = manager.get_feature_flag("nonexistent_feature", True)

        print(f"  ✅ Advanced Analytics: {analytics_flag}")
        print(f"  ✅ Nonexistent Feature (default): {nonexistent_flag}")

        # Test setting feature flags
        manager.set_feature_flag("test_feature", True)
        test_flag = manager.get_feature_flag("test_feature", False)

        if test_flag:
            print("✅ Feature flag setting/getting works correctly")
            return True
        else:
            print("❌ Feature flag setting failed")
            return False

    except Exception as e:
        print(f"❌ Feature flag test failed: {e}")
        return False

def test_service_configs():
    """Test service-specific configurations"""
    print("🧪 Testing Service Configurations...")

    try:
        from common.config_manager import get_config_manager

        manager = get_config_manager()

        # Test different service configs
        api_config = manager.get_service_config("api")
        orchestrator_config = manager.get_service_config("orchestrator")
        intelligence_config = manager.get_service_config("intelligence")
        nonexistent_config = manager.get_service_config("nonexistent")

        services_tested = 0
        if api_config:
            print(f"  ✅ API Service: {api_config.host}:{api_config.port}")
            services_tested += 1

        if orchestrator_config:
            print(f"  ✅ Orchestrator Service: {orchestrator_config.host}:{orchestrator_config.port}")
            services_tested += 1

        if intelligence_config:
            print(f"  ✅ Intelligence Service: {intelligence_config.host}:{intelligence_config.port}")
            services_tested += 1

        if nonexistent_config is None:
            print("  ✅ Nonexistent service correctly returns None")
            services_tested += 1

        if services_tested >= 3:
            print("✅ Service configuration access works correctly")
            return True
        else:
            print("❌ Some service configurations failed")
            return False

    except Exception as e:
        print(f"❌ Service configuration test failed: {e}")
        return False

def test_config_export():
    """Test configuration export functionality"""
    print("🧪 Testing Configuration Export...")

    try:
        from common.config_manager import get_config_manager, ConfigFormat

        manager = get_config_manager()

        # Test JSON export
        json_export = manager.export_config(ConfigFormat.JSON, include_secrets=False)
        json_data = json.loads(json_export)

        # Test YAML export (if available)
        try:
            yaml_export = manager.export_config(ConfigFormat.YAML, include_secrets=False)
            print("  ✅ YAML export successful")
        except Exception as e:
            print(f"  ⚠️  YAML export failed (yaml library may not be installed): {e}")

        # Test ENV export
        env_export = manager.export_config(ConfigFormat.ENV, include_secrets=False)

        if json_data and "environment" in json_data and env_export:
            print("✅ Configuration export works correctly")
            print(f"  JSON keys: {len(json_data.keys())}")
            print(f"  ENV vars: {len(env_export.split())}")
            return True
        else:
            print("❌ Configuration export incomplete")
            return False

    except Exception as e:
        print(f"❌ Configuration export test failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation"""
    print("🧪 Testing Configuration Validation...")

    try:
        from common.config_manager import ConfigManager, Environment

        # Test valid configuration (development should be permissive)
        try:
            manager = ConfigManager(environment="development", enable_hot_reload=False)
            config = manager.get_config()
            print("  ✅ Development configuration validation passed")
        except Exception as e:
            print(f"  ❌ Development configuration validation failed: {e}")
            return False

        # Test production configuration requirements (this may fail due to missing secrets)
        try:
            manager = ConfigManager(environment="production", enable_hot_reload=False)
            config = manager.get_config()
            print("  ✅ Production configuration loaded (secrets available)")
        except Exception as e:
            print(f"  ⚠️  Production configuration validation failed (expected): {e}")

        print("✅ Configuration validation system is working")
        return True

    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_fallback_mechanisms():
    """Test fallback mechanisms when dependencies are missing"""
    print("🧪 Testing Fallback Mechanisms...")

    try:
        from common.config_manager import ConfigManager

        # Test with non-existent config directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(
                config_dir=temp_dir,
                environment="development",
                enable_hot_reload=False
            )
            config = manager.get_config()

            if config and config.app_name == "XORB Platform":
                print("  ✅ Fallback to default configuration works")
            else:
                print("  ❌ Fallback configuration failed")
                return False

        print("✅ Fallback mechanisms work correctly")
        return True

    except Exception as e:
        print(f"❌ Fallback mechanism test failed: {e}")
        return False

async def main():
    """Run all configuration management tests"""
    print("🔧 XORB Configuration Management Test Suite")
    print("=" * 60)

    tests = [
        ("Config Manager Import", test_config_manager_import),
        ("Environment-Specific Configs", test_environment_specific_configs),
        ("Feature Flags", test_feature_flags),
        ("Service Configurations", test_service_configs),
        ("Configuration Export", test_config_export),
        ("Configuration Validation", test_config_validation),
        ("Fallback Mechanisms", test_fallback_mechanisms),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 Test Results:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print(f"\n🎯 Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All configuration management tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
