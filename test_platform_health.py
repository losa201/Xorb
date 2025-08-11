#!/usr/bin/env python3
"""
XORB Platform Health Check Script
Tests critical components and provides debugging information
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test critical module imports"""
    results = {
        "imports": {},
        "syntax_checks": {},
        "service_health": {}
    }
    
    # Test core imports
    import_tests = [
        ("fastapi", "FastAPI framework"),
        ("pydantic", "Data validation"),
        ("asyncio", "Async support"),
        ("redis", "Redis client"),
        ("json", "JSON processing"),
        ("logging", "Logging system"),
        ("uuid", "UUID generation"),
        ("datetime", "Date/time handling")
    ]
    
    logger.info("Testing critical imports...")
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            results["imports"][module_name] = {
                "status": "success",
                "description": description
            }
            logger.info(f"✓ {module_name}: {description}")
        except ImportError as e:
            results["imports"][module_name] = {
                "status": "failed",
                "description": description,
                "error": str(e)
            }
            logger.error(f"✗ {module_name}: {e}")
    
    return results

def test_file_syntax():
    """Test Python file syntax"""
    import ast
    
    results = {}
    
    # Key files to test
    test_files = [
        "src/api/app/main.py",
        "src/api/app/infrastructure/database_models.py",
        "src/api/app/controllers/advanced_orchestration_controller.py",
        "src/api/app/services/interfaces.py"
    ]
    
    logger.info("Testing file syntax...")
    
    for file_path in test_files:
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                ast.parse(content)
                results[file_path] = {"status": "valid", "error": None}
                logger.info(f"✓ {file_path}: Syntax valid")
            else:
                results[file_path] = {"status": "missing", "error": "File not found"}
                logger.warning(f"- {file_path}: File not found")
        except SyntaxError as e:
            results[file_path] = {"status": "syntax_error", "error": str(e)}
            logger.error(f"✗ {file_path}: {e}")
        except Exception as e:
            results[file_path] = {"status": "error", "error": str(e)}
            logger.error(f"✗ {file_path}: {e}")
    
    return results

def test_configuration():
    """Test configuration files"""
    results = {}
    
    config_files = [
        "config/production.json",
        "config/development.json", 
        "requirements.txt",
        "docker-compose.yml"
    ]
    
    logger.info("Testing configuration files...")
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                if config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        json.load(f)
                    results[config_file] = {"status": "valid", "type": "json"}
                else:
                    # Just check if file exists and is readable
                    with open(config_file, 'r') as f:
                        f.read()
                    results[config_file] = {"status": "valid", "type": "text"}
                logger.info(f"✓ {config_file}: Valid")
            except Exception as e:
                results[config_file] = {"status": "invalid", "error": str(e)}
                logger.error(f"✗ {config_file}: {e}")
        else:
            results[config_file] = {"status": "missing"}
            logger.warning(f"- {config_file}: Missing")
    
    return results

def generate_health_report():
    """Generate comprehensive health report"""
    
    logger.info("🔍 Starting XORB Platform Health Check...")
    
    health_report = {
        "platform": "XORB Enterprise Cybersecurity Platform",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "health_check_id": f"health_{int(datetime.utcnow().timestamp())}",
        "summary": {},
        "details": {}
    }
    
    # Run tests
    import_results = test_imports()
    syntax_results = test_file_syntax()
    config_results = test_configuration()
    
    health_report["details"]["imports"] = import_results["imports"]
    health_report["details"]["syntax_checks"] = syntax_results
    health_report["details"]["configuration"] = config_results
    
    # Calculate summary
    total_imports = len(import_results["imports"])
    successful_imports = sum(1 for result in import_results["imports"].values() if result["status"] == "success")
    
    total_files = len(syntax_results)
    valid_files = sum(1 for result in syntax_results.values() if result["status"] == "valid")
    
    total_configs = len(config_results)
    valid_configs = sum(1 for result in config_results.values() if result["status"] == "valid")
    
    health_report["summary"] = {
        "overall_status": "healthy" if (successful_imports/total_imports > 0.8 and valid_files/total_files > 0.7) else "degraded",
        "import_success_rate": f"{successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)",
        "syntax_success_rate": f"{valid_files}/{total_files} ({valid_files/total_files*100:.1f}%)",
        "config_success_rate": f"{valid_configs}/{total_configs} ({valid_configs/total_configs*100:.1f}%)",
        "critical_issues": [],
        "recommendations": []
    }
    
    # Identify critical issues
    critical_issues = []
    recommendations = []
    
    if successful_imports < total_imports:
        critical_issues.append("Missing Python dependencies")
        recommendations.append("Install missing packages: pip install -r requirements.txt")
    
    if valid_files < total_files:
        critical_issues.append("Python syntax errors detected")
        recommendations.append("Fix syntax errors in Python files")
    
    if valid_configs < total_configs:
        critical_issues.append("Configuration file issues")
        recommendations.append("Verify configuration file formats")
    
    health_report["summary"]["critical_issues"] = critical_issues
    health_report["summary"]["recommendations"] = recommendations
    
    return health_report

def main():
    """Main execution function"""
    try:
        # Generate health report
        report = generate_health_report()
        
        # Save report
        report_file = f"platform_health_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 XORB PLATFORM HEALTH CHECK SUMMARY")
        print("="*60)
        print(f"Overall Status: {report['summary']['overall_status'].upper()}")
        print(f"Import Success: {report['summary']['import_success_rate']}")
        print(f"Syntax Success: {report['summary']['syntax_success_rate']}")
        print(f"Config Success: {report['summary']['config_success_rate']}")
        
        if report['summary']['critical_issues']:
            print(f"\n🚨 Critical Issues ({len(report['summary']['critical_issues'])}):")
            for issue in report['summary']['critical_issues']:
                print(f"  • {issue}")
        
        if report['summary']['recommendations']:
            print(f"\n💡 Recommendations ({len(report['summary']['recommendations'])}):")
            for rec in report['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n📄 Full report saved to: {report_file}")
        print("="*60)
        
        # Exit with appropriate code
        if report['summary']['overall_status'] == 'healthy':
            logger.info("✅ Platform health check completed successfully")
            sys.exit(0)
        else:
            logger.warning("⚠️ Platform health check found issues")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()