#!/usr/bin/env python3
"""
XORB 2.0 Deployment Validation Script  
Quick validation of all deployment components without external dependencies
"""

import json
import requests
import sys
import time
sys.path.insert(0, '/root/Xorb')

def test_service_health(url, service_name):
    """Test if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print(f"✅ {service_name} - HEALTHY")
                return True
        print(f"❌ {service_name} - UNHEALTHY")
        return False
    except Exception as e:
        print(f"❌ {service_name} - CONNECTION FAILED: {e}")
        return False

def test_advanced_feature_imports():
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
    
    success_count = 0
    for feature in features:
        try:
            __import__(feature)
            print(f"✅ {feature} - IMPORT SUCCESS")
            success_count += 1
        except ImportError as e:
            print(f"❌ {feature} - IMPORT FAILED: {e}")
            
    return success_count, len(features)

def main():
    """Run complete deployment validation"""
    print("🚀 XORB 2.0 Deployment Validation")
    print("=" * 50)
    
    # Test core services
    print("\n🔧 Testing Core Services...")
    services = [
        ("http://localhost:8000", "API Service"),
        ("http://localhost:8080", "Orchestrator Service"),
        ("http://localhost:9090", "Worker Service")
    ]
    
    healthy_services = 0
    for url, name in services:
        if test_service_health(url, name):
            healthy_services += 1
            
    # Test monitoring services
    print("\n📊 Testing Monitoring Services...")
    monitoring_tests = [
        ("http://localhost:9091/api/v1/query?query=up", "Prometheus"),
        ("http://localhost:3000/api/health", "Grafana")
    ]
    
    monitoring_healthy = 0
    for url, name in monitoring_tests:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} - HEALTHY")
                monitoring_healthy += 1
            else:
                print(f"❌ {name} - UNHEALTHY")
        except Exception as e:
            print(f"❌ {name} - CONNECTION FAILED: {e}")
            
    # Test advanced features
    print("\n🎯 Testing Advanced Features...")
    successful_imports, total_imports = test_advanced_feature_imports()
    
    # Test API endpoints
    print("\n🌐 Testing API Endpoints...")
    api_endpoints = [
        "/",
        "/api/v1/status", 
        "/api/v1/assets",
        "/api/v1/scans",
        "/api/v1/findings"
    ]
    
    working_endpoints = 0
    for endpoint in api_endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - WORKING")
                working_endpoints += 1
            else:
                print(f"❌ {endpoint} - FAILED")
        except Exception as e:
            print(f"❌ {endpoint} - ERROR: {e}")
            
    # Summary
    print("\n🎉 Validation Summary")
    print("=" * 50)
    print(f"Core Services: {healthy_services}/{len(services)} healthy")
    print(f"Monitoring: {monitoring_healthy}/{len(monitoring_tests)} healthy")
    print(f"Advanced Features: {successful_imports}/{total_imports} working")
    print(f"API Endpoints: {working_endpoints}/{len(api_endpoints)} working")
    
    total_score = healthy_services + monitoring_healthy + successful_imports + working_endpoints
    max_score = len(services) + len(monitoring_tests) + total_imports + len(api_endpoints)
    
    success_rate = (total_score / max_score) * 100
    print(f"\n📈 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 XORB 2.0 Deployment: EXCELLENT!")
    elif success_rate >= 75:
        print("✅ XORB 2.0 Deployment: GOOD")
    elif success_rate >= 50:
        print("⚠️  XORB 2.0 Deployment: NEEDS ATTENTION")
    else:
        print("❌ XORB 2.0 Deployment: MAJOR ISSUES")
        
    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)