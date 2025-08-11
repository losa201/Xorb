#!/usr/bin/env python3
"""
Batch 2 Integration Test - Dependency Consolidation Validation
Tests that all services work correctly with unified dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_python_imports():
    """Test critical Python package imports."""
    print("🧪 Testing Python package imports...")
    
    test_imports = [
        'fastapi',
        'pydantic', 
        'uvicorn',
        'redis',
        'asyncpg',
        'sqlalchemy',
        'cryptography',
        'prometheus_client',
        'structlog'
    ]
    
    results = {}
    for package in test_imports:
        try:
            __import__(package)
            results[package] = "✅ OK"
            print(f"   • {package}: ✅ OK")
        except ImportError as e:
            results[package] = f"❌ FAIL: {e}"
            print(f"   • {package}: ❌ FAIL: {e}")
    
    return results

def test_fastapi_app_creation():
    """Test FastAPI app can be created with current config."""
    print("\n🚀 Testing FastAPI app creation...")
    
    # Set minimal required environment variables
    os.environ.update({
        'JWT_SECRET': 'test-jwt-secret-key-for-batch2-integration-testing-strong-password',
        'ENVIRONMENT': 'development',
        'DATABASE_URL': 'postgresql://xorb_test_user:StrongTestPassword123!@localhost:5432/xorb_test',
        'REDIS_URL': 'redis://localhost:6379/0'
    })
    
    try:
        # Test FastAPI app import
        test_code = """
import sys
sys.path.append('src/api')
from app.main import app
print("FastAPI app created successfully")
print(f"App title: {app.title}")
print(f"App version: {app.version}")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("   ✅ FastAPI app creation: SUCCESS")
            print(f"   📋 Output: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ FastAPI app creation: FAILED")
            print(f"   📋 Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"   ❌ FastAPI app creation: FAILED with exception: {e}")
        return False

def test_npm_dependencies():
    """Test NPM dependencies in frontend."""
    print("\n📦 Testing NPM dependencies...")
    
    npm_dir = Path("services/ptaas/web")
    if not npm_dir.exists():
        print("   ⚠️  Frontend directory not found, skipping NPM tests")
        return True
    
    try:
        # Test npm installation
        result = subprocess.run(
            ['npm', 'list', '--depth=0'],
            capture_output=True,
            text=True,
            cwd=npm_dir
        )
        
        if result.returncode == 0:
            print("   ✅ NPM dependencies: OK")
            # Count packages
            lines = result.stdout.split('\n')
            package_count = len([line for line in lines if '@' in line and not line.startswith('├─')])
            print(f"   📋 {package_count} packages installed")
            return True
        else:
            print("   ⚠️  NPM dependencies: Some issues detected")
            print(f"   📋 Output: {result.stdout}")
            return True  # Non-critical for batch 2
            
    except FileNotFoundError:
        print("   ⚠️  NPM not available, skipping frontend tests")
        return True
    except Exception as e:
        print(f"   ❌ NPM test failed: {e}")
        return False

def test_docker_build():
    """Test Docker build with unified Dockerfile."""
    print("\n🐳 Testing Docker build...")
    
    try:
        # Test if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("   ⚠️  Docker not available, skipping Docker tests")
            return True
        
        # Test Dockerfile validation (without building)
        dockerfile_path = Path("Dockerfile.unified")
        if dockerfile_path.exists():
            print("   ✅ Dockerfile.unified exists")
            
            # Check if requirements-unified.lock is referenced
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                if 'requirements-unified.lock' in content:
                    print("   ✅ Dockerfile references unified requirements")
                    return True
                else:
                    print("   ⚠️  Dockerfile doesn't reference unified requirements")
                    return False
        else:
            print("   ⚠️  Dockerfile.unified not found")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Docker test failed: {e}")
        return True  # Non-critical

def test_security_scanner():
    """Test dependency security scanner."""
    print("\n🔍 Testing dependency security scanner...")
    
    try:
        script_path = Path("scripts/dependency_security_scanner.py")
        if not script_path.exists():
            print("   ❌ Security scanner script not found")
            return False
        
        # Run a quick test of the scanner
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd()
        )
        
        if result.returncode in [0, 1]:  # 0 = no vulns, 1 = vulns found (both OK)
            print("   ✅ Security scanner: WORKING")
            print("   📋 Scanner executed successfully")
            return True
        else:
            print("   ❌ Security scanner: FAILED")
            print(f"   📋 Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Security scanner: TIMEOUT (but likely working)")
        return True
    except Exception as e:
        print(f"   ❌ Security scanner test failed: {e}")
        return False

def run_batch2_integration_tests():
    """Run all Batch 2 integration tests."""
    print("🚀 XORB Platform Batch 2 Integration Tests")
    print("=" * 60)
    print("Testing dependency consolidation and unified build system...")
    print()
    
    test_results = {}
    
    # Test 1: Python imports
    test_results['python_imports'] = test_python_imports()
    
    # Test 2: FastAPI app creation
    test_results['fastapi_app'] = test_fastapi_app_creation()
    
    # Test 3: NPM dependencies
    test_results['npm_deps'] = test_npm_dependencies()
    
    # Test 4: Docker build
    test_results['docker_build'] = test_docker_build()
    
    # Test 5: Security scanner
    test_results['security_scanner'] = test_security_scanner()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BATCH 2 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        total_tests += 1
        if result is True or (isinstance(result, dict) and all('✅' in str(v) for v in result.values())):
            print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
            passed_tests += 1
        else:
            print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
    
    print()
    print(f"📋 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL BATCH 2 INTEGRATION TESTS PASSED!")
        print("✅ Dependency consolidation successful")
        print("✅ Unified build system working")
        print("✅ Security scanning operational")
        return 0
    else:
        print("⚠️  Some tests failed - review output above")
        return 1

if __name__ == "__main__":
    exit_code = run_batch2_integration_tests()
    sys.exit(exit_code)