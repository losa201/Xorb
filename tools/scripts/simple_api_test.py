#!/usr/bin/env python3
"""
Simple XORB API Test
Basic validation of key API functionality
"""

import json
import sys
import os
import asyncio
from datetime import datetime

# Add the API path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))

try:
    import httpx
    from app.main import app
    from app.security.auth import authenticator, Role
    print("✅ Successfully imported XORB API components")
    API_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import XORB API: {e}")
    API_AVAILABLE = False

def test_basic_functionality():
    """Test basic API functionality"""
    if not API_AVAILABLE:
        print("❌ API not available for testing")
        return False
    
    print("\n🧪 Starting Basic XORB API Tests")
    print("=" * 50)
    
    # Create test client
    client = httpx.Client(app=app, base_url="http://testserver")
    
    # Generate test token
    token = authenticator.generate_jwt(
        user_id="test_user",
        client_id="test_client", 
        roles=[Role.ADMIN]
    )
    
    headers = {"Authorization": f"Bearer {token}"}
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health Check
    total_tests += 1
    print(f"\n📋 Test 1: Health Check")
    try:
        response = client.get("/health")
        if response.status_code == 200 and response.json().get("status") == "ok":
            print("✅ Health check passed")
            tests_passed += 1
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: List Agents
    total_tests += 1
    print(f"\n📋 Test 2: List Agents")
    try:
        response = client.get("/v1/agents", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Agents endpoint working - Found {len(data.get('agents', []))} agents")
            tests_passed += 1
        else:
            print(f"❌ List agents failed: {response.status_code}")
    except Exception as e:
        print(f"❌ List agents error: {e}")
    
    # Test 3: Create Agent
    total_tests += 1
    print(f"\n📋 Test 3: Create Agent")
    try:
        agent_data = {
            "name": "Test Security Agent",
            "agent_type": "security_analyst",
            "capabilities": ["threat_intelligence", "log_analysis"],
            "description": "Test agent for validation"
        }
        
        response = client.post("/v1/agents", headers=headers, json=agent_data)
        if response.status_code == 201:
            agent = response.json()
            print(f"✅ Agent created successfully: {agent.get('name')} ({agent.get('id')})")
            tests_passed += 1
            
            # Test agent status
            agent_id = agent.get('id')
            if agent_id:
                status_response = client.get(f"/v1/agents/{agent_id}/status", headers=headers)
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"✅ Agent status check: {status.get('status')}")
                else:
                    print(f"⚠️ Agent status check failed: {status_response.status_code}")
        else:
            print(f"❌ Create agent failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Create agent error: {e}")
    
    # Test 4: List Tasks
    total_tests += 1
    print(f"\n📋 Test 4: List Tasks")
    try:
        response = client.get("/v1/orchestration/tasks", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Tasks endpoint working - Found {len(data.get('tasks', []))} tasks")
            tests_passed += 1
        else:
            print(f"❌ List tasks failed: {response.status_code}")
    except Exception as e:
        print(f"❌ List tasks error: {e}")
    
    # Test 5: Create Task
    total_tests += 1
    print(f"\n📋 Test 5: Create Task")
    try:
        task_data = {
            "name": "Test Vulnerability Scan",
            "task_type": "vulnerability_scan",
            "priority": "medium",
            "parameters": {
                "target": "test_network",
                "options": {"deep_scan": True}
            },
            "description": "Test task for validation"
        }
        
        response = client.post("/v1/orchestration/tasks", headers=headers, json=task_data)
        if response.status_code == 201:
            task = response.json()
            print(f"✅ Task created successfully: {task.get('name')} ({task.get('id')})")
            tests_passed += 1
        else:
            print(f"❌ Create task failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Create task error: {e}")
    
    # Test 6: Security Threats
    total_tests += 1
    print(f"\n📋 Test 6: List Security Threats")
    try:
        response = client.get("/v1/security/threats", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Security threats endpoint working - Found {len(data.get('threats', []))} threats")
            tests_passed += 1
        else:
            print(f"❌ List threats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ List threats error: {e}")
    
    # Test 7: Security Metrics
    total_tests += 1
    print(f"\n📋 Test 7: Security Metrics")
    try:
        response = client.get("/v1/security/metrics", headers=headers)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Security metrics working - Health score: {metrics.get('system_health_score', 'N/A')}")
            tests_passed += 1
        else:
            print(f"❌ Security metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Security metrics error: {e}")
    
    # Test 8: Intelligence Models
    total_tests += 1
    print(f"\n📋 Test 8: Intelligence Models")
    try:
        response = client.get("/v1/intelligence/models", headers=headers)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Intelligence models working - Found {len(models)} models")
            tests_passed += 1
        else:
            print(f"❌ Intelligence models failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Intelligence models error: {e}")
    
    # Test 9: AI Decision Request
    total_tests += 1
    print(f"\n📋 Test 9: AI Decision Request")
    try:
        decision_data = {
            "decision_type": "threat_classification",
            "context": {
                "scenario": "test_scenario",
                "available_data": {
                    "indicators": 2,
                    "severity_score": 0.7
                },
                "urgency_level": "medium"
            }
        }
        
        response = client.post("/v1/intelligence/decisions", headers=headers, json=decision_data)
        if response.status_code == 200:
            decision = response.json()
            print(f"✅ AI decision working - Recommendation: {decision.get('recommendation')} (confidence: {decision.get('confidence_score', 0):.2f})")
            tests_passed += 1
        else:
            print(f"❌ AI decision failed: {response.status_code}")
    except Exception as e:
        print(f"❌ AI decision error: {e}")
    
    # Test 10: API Documentation
    total_tests += 1
    print(f"\n📋 Test 10: API Documentation")
    try:
        response = client.get("/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            tests_passed += 1
        else:
            print(f"❌ API documentation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API documentation error: {e}")
    
    # Generate Report
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {tests_passed} ✅")
    print(f"Failed: {total_tests - tests_passed} ❌")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! XORB API is fully functional.")
        print("\n🚀 Key Features Validated:")
        print("  ✅ Health monitoring")
        print("  ✅ Agent management")  
        print("  ✅ Task orchestration")
        print("  ✅ Security operations")
        print("  ✅ Intelligence integration")
        print("  ✅ API documentation")
        return True
    else:
        print(f"\n⚠️ {total_tests - tests_passed} tests failed. See details above.")
        return False

def main():
    """Main test execution"""
    print("🚀 XORB API Simple Test Suite")
    print("Testing core API functionality...")
    
    success = test_basic_functionality()
    
    if success:
        print("\n✅ XORB API is ready for deployment!")
        return 0
    else:
        print("\n❌ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())