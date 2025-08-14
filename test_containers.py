#!/usr/bin/env python3
"""
XORB Container Testing Suite
Tests Docker containers for functionality, security, and performance
"""

import asyncio
import docker
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_docker_environment():
    """Test Docker environment and connectivity"""
    print("🐳 Testing Docker Environment...")

    try:
        client = docker.from_env()

        # Test Docker daemon connectivity
        info = client.info()
        print(f"  ✅ Docker daemon connected (version: {info.get('ServerVersion', 'unknown')})")

        # Check available resources
        containers = client.containers.list(all=True)
        print(f"  ✅ Found {len(containers)} containers")

        images = client.images.list()
        print(f"  ✅ Found {len(images)} images")

        return True

    except Exception as e:
        print(f"  ❌ Docker environment test failed: {e}")
        return False

def build_test_images():
    """Build Docker images for testing"""
    print("🔨 Building Test Images...")

    try:
        client = docker.from_env()
        services = ['api', 'orchestrator', 'worker']
        built_images = {}

        for service in services:
            print(f"  Building {service} service...")

            dockerfile_path = f"src/{service}/Dockerfile"
            if not os.path.exists(dockerfile_path):
                print(f"  ⚠️ Dockerfile not found: {dockerfile_path}")
                continue

            # Build development target
            image, logs = client.images.build(
                path=".",
                dockerfile=dockerfile_path,
                target="development",
                tag=f"xorb-test/{service}:development",
                rm=True,
                pull=False
            )

            built_images[service] = image
            print(f"  ✅ Built {service} image: {image.short_id}")

        return built_images

    except Exception as e:
        print(f"  ❌ Image building failed: {e}")
        return {}

def test_container_security():
    """Test container security configurations"""
    print("🔒 Testing Container Security...")

    try:
        client = docker.from_env()
        services = ['api', 'orchestrator', 'worker']
        security_issues = []

        for service in services:
            image_name = f"xorb-test/{service}:development"

            try:
                # Inspect image
                image = client.images.get(image_name)
                config = image.attrs.get('Config', {})

                # Check if running as root
                user = config.get('User', 'root')
                if user == 'root' or user == '':
                    security_issues.append(f"{service}: Running as root user")
                else:
                    print(f"  ✅ {service}: Running as non-root user ({user})")

                # Check for exposed ports
                exposed_ports = config.get('ExposedPorts', {})
                if exposed_ports:
                    print(f"  ✅ {service}: Exposes ports {list(exposed_ports.keys())}")

                # Check labels
                labels = config.get('Labels', {})
                security_label = labels.get('security.non-root')
                if security_label == 'true':
                    print(f"  ✅ {service}: Security labels present")

            except docker.errors.ImageNotFound:
                print(f"  ⚠️ {service}: Image not found for security testing")

        if security_issues:
            print("  ❌ Security issues found:")
            for issue in security_issues:
                print(f"    - {issue}")
            return False
        else:
            print("  ✅ All containers pass security checks")
            return True

    except Exception as e:
        print(f"  ❌ Container security test failed: {e}")
        return False

def test_container_startup():
    """Test container startup and health checks"""
    print("🚀 Testing Container Startup...")

    try:
        client = docker.from_env()
        services = ['api', 'orchestrator', 'worker']
        startup_results = {}

        # Test each service individually
        for service in services:
            print(f"  Testing {service} startup...")

            image_name = f"xorb-test/{service}:development"

            try:
                # Run container with minimal environment
                container = client.containers.run(
                    image_name,
                    environment={
                        'XORB_ENV': 'test',
                        'DATABASE_HOST': 'localhost',
                        'REDIS_HOST': 'localhost',
                        'DEBUG': 'true'
                    },
                    detach=True,
                    remove=True
                )

                # Wait for container to start
                time.sleep(10)

                # Check if container is still running
                container.reload()
                if container.status == 'running':
                    print(f"    ✅ {service}: Container started successfully")
                    startup_results[service] = True
                else:
                    print(f"    ❌ {service}: Container failed to start (status: {container.status})")
                    # Get logs for debugging
                    logs = container.logs().decode('utf-8')
                    print(f"    Logs: {logs[-500:]}")  # Last 500 characters
                    startup_results[service] = False

                # Stop container
                container.stop(timeout=5)

            except docker.errors.ImageNotFound:
                print(f"    ⚠️ {service}: Image not found for startup testing")
                startup_results[service] = False
            except Exception as e:
                print(f"    ❌ {service}: Startup test failed: {e}")
                startup_results[service] = False

        success_count = sum(startup_results.values())
        total_count = len(startup_results)

        print(f"  📊 Startup Results: {success_count}/{total_count} services started successfully")
        return success_count == total_count

    except Exception as e:
        print(f"  ❌ Container startup test failed: {e}")
        return False

def test_configuration_integration():
    """Test configuration management integration in containers"""
    print("⚙️ Testing Configuration Integration...")

    try:
        client = docker.from_env()

        # Test API service configuration
        print("  Testing API service configuration...")

        # Create temporary config directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test configuration
            config_data = {
                "environment": "test",
                "app_name": "XORB Test",
                "debug": True,
                "database": {
                    "host": "test-db",
                    "port": 5432,
                    "name": "xorb_test"
                },
                "redis": {
                    "host": "test-redis",
                    "port": 6379
                },
                "api_service": {
                    "port": 8000
                }
            }

            config_file = Path(temp_dir) / "test.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            # Test configuration loading in container
            container = client.containers.run(
                "xorb-test/api:development",
                command=["python", "-c", """
import sys
sys.path.append('/app/src')
from common.config_manager import ConfigManager, Environment
try:
    manager = ConfigManager(environment='test', enable_hot_reload=False)
    config = manager.get_config()
    print(f'✅ Config loaded: {config.app_name}')
    print(f'✅ Environment: {config.environment.value}')
    print(f'✅ API Port: {config.api_service.port}')
    print(f'✅ Database: {config.database.host}:{config.database.port}')
except Exception as e:
    print(f'❌ Config test failed: {e}')
    sys.exit(1)
"""],
                environment={
                    'XORB_ENV': 'test',
                    'PYTHONPATH': '/app:/app/src'
                },
                volumes={
                    str(config_file.parent): {'bind': '/app/config', 'mode': 'ro'}
                },
                detach=True,
                remove=True
            )

            # Wait for execution and get results
            result = container.wait(timeout=30)
            logs = container.logs().decode('utf-8')

            if result['StatusCode'] == 0:
                print("    ✅ Configuration integration successful")
                print(f"    Output: {logs}")
                return True
            else:
                print(f"    ❌ Configuration integration failed (exit code: {result['StatusCode']})")
                print(f"    Logs: {logs}")
                return False

    except Exception as e:
        print(f"  ❌ Configuration integration test failed: {e}")
        return False

def test_image_sizes():
    """Test and report image sizes"""
    print("📊 Testing Image Sizes...")

    try:
        client = docker.from_env()
        services = ['api', 'orchestrator', 'worker']
        size_report = []

        print(f"{'Service':<15} {'Target':<12} {'Size':<10} {'Status'}")
        print("-" * 50)

        for service in services:
            for target in ['development', 'production']:
                image_name = f"xorb-test/{service}:{target}"

                try:
                    image = client.images.get(image_name)
                    size_mb = round(image.attrs['Size'] / (1024 * 1024), 1)

                    # Determine if size is reasonable
                    status = "✅ OK"
                    if size_mb > 1000:  # Greater than 1GB
                        status = "⚠️ LARGE"
                    elif size_mb > 500:  # Greater than 500MB
                        status = "⚠️ BIG"

                    print(f"{service:<15} {target:<12} {size_mb}MB{'':<5} {status}")
                    size_report.append((service, target, size_mb, status))

                except docker.errors.ImageNotFound:
                    print(f"{service:<15} {target:<12} {'N/A':<10} ❌ NOT_FOUND")
                    size_report.append((service, target, 0, "NOT_FOUND"))

        # Calculate totals
        total_size = sum(size for _, _, size, status in size_report if status != "NOT_FOUND")
        print(f"\n  📊 Total size: {total_size}MB")

        return True

    except Exception as e:
        print(f"  ❌ Image size test failed: {e}")
        return False

def test_health_endpoints():
    """Test health endpoints in containers"""
    print("🏥 Testing Health Endpoints...")

    try:
        import requests
    except ImportError:
        print("  ⚠️ Requests library not available - skipping health endpoint tests")
        return True

    try:
        client = docker.from_env()

        # Test API health endpoint
        print("  Testing API health endpoint...")

        # Start API container with port mapping
        api_container = client.containers.run(
            "xorb-test/api:development",
            environment={
                'XORB_ENV': 'test',
                'DATABASE_HOST': 'localhost',
                'REDIS_HOST': 'localhost',
                'DEBUG': 'true'
            },
            ports={'8000/tcp': 8000},
            detach=True,
            remove=True
        )

        # Wait for service to start
        time.sleep(15)

        try:
            # Test health endpoint
            response = requests.get('http://localhost:8000/health', timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                print(f"    ✅ API health endpoint responding: {health_data.get('status', 'unknown')}")
                health_result = True
            else:
                print(f"    ❌ API health endpoint failed: HTTP {response.status_code}")
                health_result = False

        except requests.exceptions.RequestException as e:
            print(f"    ❌ API health endpoint unreachable: {e}")
            health_result = False

        # Cleanup
        api_container.stop(timeout=5)

        return health_result

    except Exception as e:
        print(f"  ❌ Health endpoint test failed: {e}")
        return False

def test_volume_mounts():
    """Test volume mounts and file permissions"""
    print("💾 Testing Volume Mounts...")

    try:
        client = docker.from_env()

        # Test volume mounts with API service
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test_config.json"
            test_file.write_text('{"test": true}')

            # Test read-only mount
            container = client.containers.run(
                "xorb-test/api:development",
                command=["ls", "-la", "/app/config"],
                volumes={
                    temp_dir: {'bind': '/app/config', 'mode': 'ro'}
                },
                detach=True,
                remove=True
            )

            result = container.wait(timeout=10)
            logs = container.logs().decode('utf-8')

            if result['StatusCode'] == 0:
                print("    ✅ Volume mount successful")
                print(f"    Files: {logs}")
                return True
            else:
                print(f"    ❌ Volume mount failed (exit code: {result['StatusCode']})")
                print(f"    Logs: {logs}")
                return False

    except Exception as e:
        print(f"  ❌ Volume mount test failed: {e}")
        return False

async def main():
    """Run all container tests"""
    print("🔧 XORB Container Test Suite")
    print("=" * 60)

    tests = [
        ("Docker Environment", test_docker_environment),
        ("Build Test Images", build_test_images),
        ("Container Security", test_container_security),
        ("Container Startup", test_container_startup),
        ("Configuration Integration", test_configuration_integration),
        ("Image Sizes", test_image_sizes),
        ("Health Endpoints", test_health_endpoints),
        ("Volume Mounts", test_volume_mounts),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_name == "Build Test Images":
                # Special handling for build test
                built_images = test_func()
                result = len(built_images) > 0
            else:
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
        print("🎉 All container tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
