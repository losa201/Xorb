#!/usr/bin/env python3
"""
XORB Deployment Verification Script
Comprehensive testing of XORB deployment fixes and service functionality
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def test_imports():
    """Test critical imports are working."""
    print("🔍 Testing Python imports...")
    
    results = {}
    
    # Test core packages
    try:
        from packages.xorb_core.xorb_core.logging import configure_logging, get_logger
        results['xorb_core.logging'] = '✅ SUCCESS'
    except ImportError as e:
        results['xorb_core.logging'] = f'❌ FAILED: {e}'
    
    try:
        from packages.xorb_core.xorb_core.models import DiscoveryTarget, Finding
        results['xorb_core.models'] = '✅ SUCCESS'
    except ImportError as e:
        results['xorb_core.models'] = f'❌ FAILED: {e}'
        
    # Test service imports
    try:
        from services.api.app.main import app
        results['api.main'] = '✅ SUCCESS'
    except ImportError as e:
        results['api.main'] = f'❌ FAILED: {e}'
    
    try:
        from services.worker.workflows import DynamicScanWorkflow
        results['worker.workflows'] = '✅ SUCCESS'
    except ImportError as e:
        results['worker.workflows'] = f'❌ FAILED: {e}'
    
    try:
        import temporalio
        results['temporalio'] = '✅ SUCCESS'
    except ImportError as e:
        results['temporalio'] = f'❌ FAILED: {e}'
        
    try:
        import fastapi
        results['fastapi'] = '✅ SUCCESS'
    except ImportError as e:
        results['fastapi'] = f'❌ FAILED: {e}'
    
    try:
        import neo4j
        results['neo4j'] = '✅ SUCCESS'
    except ImportError as e:
        results['neo4j'] = f'❌ FAILED: {e}'
        
    # Print results
    for package, status in results.items():
        print(f"  {package:<20} {status}")
    
    return all('SUCCESS' in status for status in results.values())


def test_configuration_files():
    """Test that configuration files are valid."""
    print("\n📋 Testing configuration files...")
    
    results = {}
    
    # Test docker-compose.yml
    try:
        import yaml
        with open('docker-compose.yml', 'r') as f:
            yaml.safe_load(f)
        results['docker-compose.yml'] = '✅ SUCCESS'
    except Exception as e:
        results['docker-compose.yml'] = f'❌ FAILED: {e}'
    
    # Test targets.json
    try:
        with open('targets.json', 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                results['targets.json'] = '✅ SUCCESS'
            else:
                results['targets.json'] = '❌ FAILED: Invalid format'
    except Exception as e:
        results['targets.json'] = f'❌ FAILED: {e}'
    
    # Test requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'temporalio' in content and 'neo4j' in content:
                results['requirements.txt'] = '✅ SUCCESS'
            else:
                results['requirements.txt'] = '❌ FAILED: Missing critical dependencies'
    except Exception as e:
        results['requirements.txt'] = f'❌ FAILED: {e}'
    
    # Print results
    for file, status in results.items():
        print(f"  {file:<20} {status}")
    
    return all('SUCCESS' in status for status in results.values())


def test_model_creation():
    """Test that model classes can be instantiated."""
    print("\n🏗️  Testing model creation...")
    
    try:
        from packages.xorb_core.xorb_core.models import DiscoveryTarget, Finding, TargetType, FindingSeverity
        from datetime import datetime
        
        # Test DiscoveryTarget creation
        target = DiscoveryTarget(
            target_type=TargetType.DOMAIN,
            value="test.example.com",
            scope="test",
            metadata={"test": True}
        )
        
        # Test Finding creation
        finding = Finding(
            id="test-finding-001",
            title="Test Finding",
            description="A test security finding",
            severity=FindingSeverity.LOW,
            target="test.example.com",
            discovery_method="unit_test", 
            timestamp=datetime.utcnow()
        )
        
        print("  ✅ Model creation successful")
        print(f"    Target: {target.value} ({target.target_type})")
        print(f"    Finding: {finding.title} [{finding.severity}]")
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def test_logging():
    """Test logging configuration."""
    print("\n📝 Testing logging system...")
    
    try:
        from packages.xorb_core.xorb_core.logging import configure_logging, get_logger
        
        # Configure logging
        configure_logging(level="INFO", service_name="verification-test")
        
        # Get logger and test logging
        logger = get_logger("test_logger")
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        print("  ✅ Logging system operational")
        return True
        
    except Exception as e:
        print(f"  ❌ Logging system failed: {e}")
        return False


async def test_workflow_creation():
    """Test workflow instantiation."""
    print("\n🔄 Testing workflow creation...")
    
    try:
        from services.worker.workflows import DynamicScanWorkflow, DiscoveryTarget
        
        # Create workflow instance
        workflow = DynamicScanWorkflow()
        
        # Create test target
        target = DiscoveryTarget(
            target_type="domain",
            value="test.example.com"
        )
        
        print("  ✅ Workflow creation successful")
        print(f"    Workflow: {workflow.__class__.__name__}")
        print(f"    Test target: {target.value}")
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow creation failed: {e}")
        return False


def generate_deployment_report(test_results: Dict[str, bool]):
    """Generate a comprehensive deployment report."""
    print("\n" + "="*60)
    print("🎯 XORB DEPLOYMENT VERIFICATION REPORT")
    print("="*60)
    
    all_passed = all(test_results.values())
    
    print(f"\n📊 Overall Status: {'✅ PASS' if all_passed else '❌ FAIL'}")
    print(f"📈 Success Rate: {sum(test_results.values())}/{len(test_results)} ({sum(test_results.values())/len(test_results)*100:.1f}%)")
    
    print("\n📋 Test Results:")
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    if all_passed:
        print("\n🎉 DEPLOYMENT READY!")
        print("   All core issues have been resolved.")
        print("   You can now run: docker-compose up -d")
    else:
        print("\n⚠️  DEPLOYMENT ISSUES DETECTED")
        print("   Please review failed tests above.")
        print("   Fix issues before deployment.")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: docker-compose build")
    print("   2. Run: docker-compose up -d")
    print("   3. Check: curl http://localhost:8000/health")
    print("   4. Monitor: docker-compose logs -f")
    
    return all_passed


async def main():
    """Run all deployment verification tests."""
    print("🚀 XORB DEPLOYMENT VERIFICATION")
    print("Checking deployment fixes and service readiness...")
    print("="*60)
    
    # Run all tests
    test_results = {
        'Python Imports': test_imports(),
        'Configuration Files': test_configuration_files(),
        'Model Creation': test_model_creation(),
        'Logging System': test_logging(),
        'Workflow Creation': await test_workflow_creation()
    }
    
    # Generate report
    deployment_ready = generate_deployment_report(test_results)
    
    # Return appropriate exit code
    return 0 if deployment_ready else 1


if __name__ == "__main__":
    # Set up Python path
    import sys
    from pathlib import Path
    
    root_dir = Path(__file__).parent
    sys.path.insert(0, str(root_dir))
    sys.path.insert(0, str(root_dir / "packages"))
    sys.path.insert(0, str(root_dir / "xorb_core"))
    
    # Run verification
    exit_code = asyncio.run(main())
    sys.exit(exit_code)