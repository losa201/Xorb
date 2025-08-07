#!/usr/bin/env python3
"""
XORB LLM Cognitive Cortex Integration Test

Comprehensive test suite for the LLM cognitive cortex system including
routing, adaptive learning, telemetry, and security features.
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Add paths
sys.path.insert(0, '/root/Xorb')
sys.path.insert(0, '/root/Xorb/xorb_core')

# Test configuration
BASE_URL = "http://localhost:8009"
TEST_AGENT_ID = "test-agent-001"


class LLMCortexTester:
    """Comprehensive LLM cortex test suite"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        self.start_time = time.time()
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        print("🧠 XORB LLM Cognitive Cortex Integration Test")
        print("=" * 60)
        print(f"Base URL: {BASE_URL}")
        print(f"Test Agent ID: {TEST_AGENT_ID}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    async def cleanup(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self):
        """Test health check endpoint"""
        print("🔍 Testing health check...")
        
        try:
            async with self.session.get(f"{BASE_URL}/llm/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Health check passed: {data['status']}")
                    print(f"   📊 Models available: {data['models_available']}")
                    self.test_results.append(("health_check", True, "Health check passed"))
                    return True
                else:
                    print(f"   ❌ Health check failed: {response.status}")
                    self.test_results.append(("health_check", False, f"Status {response.status}"))
                    return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            self.test_results.append(("health_check", False, str(e)))
            return False
    
    async def test_llm_request(self):
        """Test LLM request routing"""
        print("🤖 Testing LLM request routing...")
        
        test_cases = [
            {
                "task_type": "narrative",
                "prompt": "Write a brief security incident summary for a SQL injection attack.",
                "expected_model_contains": "horizon"
            },
            {
                "task_type": "exploit",
                "prompt": "Analyze potential SQL injection vectors in this query: SELECT * FROM users WHERE id = ?",
                "expected_model_contains": "glm"
            },
            {
                "task_type": "code_analysis",
                "prompt": "Review this Python function for security vulnerabilities:\ndef get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')",
                "expected_model_contains": "qwen"
            }
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"   Test {i+1}: {test_case['task_type']}")
            
            try:
                payload = {
                    "prompt": test_case["prompt"],
                    "task_type": test_case["task_type"],
                    "agent_id": TEST_AGENT_ID
                }
                
                start_time = time.time()
                async with self.session.post(f"{BASE_URL}/llm/request", json=payload) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Validate response structure
                        required_fields = ["content", "model", "task_type", "latency_ms", "confidence_score"]
                        missing_fields = [field for field in required_fields if field not in data]
                        
                        if missing_fields:
                            print(f"   ❌ Missing fields: {missing_fields}")
                            self.test_results.append((f"llm_request_{test_case['task_type']}", False, f"Missing fields: {missing_fields}"))
                            continue
                        
                        # Check model routing
                        model_correct = test_case["expected_model_contains"] in data["model"].lower()
                        
                        print(f"      ✅ Model: {data['model']} ({'✓' if model_correct else '✗'})")
                        print(f"      ⚡ Latency: {data['latency_ms']:.1f}ms")
                        print(f"      📊 Confidence: {data['confidence_score']:.2f}")
                        print(f"      📝 Content length: {len(data['content'])} chars")
                        
                        if model_correct and data['confidence_score'] > 0.5:
                            success_count += 1
                            self.test_results.append((f"llm_request_{test_case['task_type']}", True, "Request successful"))
                        else:
                            self.test_results.append((f"llm_request_{test_case['task_type']}", False, f"Model routing or confidence issue"))
                    
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Request failed: {response.status} - {error_text}")
                        self.test_results.append((f"llm_request_{test_case['task_type']}", False, f"Status {response.status}"))
                        
            except Exception as e:
                print(f"   ❌ Request error: {e}")
                self.test_results.append((f"llm_request_{test_case['task_type']}", False, str(e)))
        
        print(f"   📊 LLM Requests: {success_count}/{len(test_cases)} successful")
        return success_count == len(test_cases)
    
    async def test_routing_recommendation(self):
        """Test routing recommendation endpoint"""
        print("🔀 Testing routing recommendations...")
        
        task_types = ["narrative", "exploit", "code_analysis", "multi_hop_threat"]
        success_count = 0
        
        for task_type in task_types:
            try:
                payload = {"task_type": task_type, "agent_id": TEST_AGENT_ID}
                
                async with self.session.post(f"{BASE_URL}/llm/route", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        print(f"   ✅ {task_type}: {data['recommended_model']}")
                        if data.get('model_profile'):
                            print(f"      Profile: {data['model_profile']}")
                        
                        success_count += 1
                        self.test_results.append((f"routing_{task_type}", True, "Routing recommendation successful"))
                    else:
                        print(f"   ❌ {task_type}: {response.status}")
                        self.test_results.append((f"routing_{task_type}", False, f"Status {response.status}"))
                        
            except Exception as e:
                print(f"   ❌ {task_type}: {e}")
                self.test_results.append((f"routing_{task_type}", False, str(e)))
        
        print(f"   📊 Routing: {success_count}/{len(task_types)} successful")
        return success_count == len(task_types)
    
    async def test_content_scoring(self):
        """Test content scoring endpoint"""
        print("📊 Testing content scoring...")
        
        test_contents = [
            {
                "content": "This is a well-structured security analysis of the SQL injection vulnerability.",
                "expected_task": "code_analysis",
                "expected_quality": 0.7
            },
            {
                "content": "I don't have access to real-time data about this.",
                "expected_task": "narrative",
                "expected_quality": 0.3  # Should score low due to hallucination pattern
            }
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(test_contents):
            try:
                payload = {
                    "content": test_case["content"],
                    "expected_task": test_case["expected_task"]
                }
                
                async with self.session.post(f"{BASE_URL}/llm/score", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        print(f"   Test {i+1}:")
                        print(f"      Quality Score: {data['quality_score']:.2f}")
                        print(f"      Confidence: {data['confidence_score']:.2f}")
                        print(f"      Hallucination: {data['hallucination_score']:.2f}")
                        
                        # Check if quality score is in expected range
                        quality_ok = abs(data['quality_score'] - test_case['expected_quality']) < 0.3
                        
                        if quality_ok:
                            success_count += 1
                            print(f"      ✅ Score in expected range")
                            self.test_results.append((f"scoring_{i}", True, "Scoring successful"))
                        else:
                            print(f"      ❌ Score not in expected range")
                            self.test_results.append((f"scoring_{i}", False, "Score out of range"))
                    
                    else:
                        print(f"   ❌ Test {i+1}: {response.status}")
                        self.test_results.append((f"scoring_{i}", False, f"Status {response.status}"))
                        
            except Exception as e:
                print(f"   ❌ Test {i+1}: {e}")
                self.test_results.append((f"scoring_{i}", False, str(e)))
        
        print(f"   📊 Scoring: {success_count}/{len(test_contents)} successful")
        return success_count == len(test_contents)
    
    async def test_feedback_system(self):
        """Test feedback collection system"""
        print("🔄 Testing feedback system...")
        
        try:
            # Submit some feedback
            payload = {
                "agent_id": TEST_AGENT_ID,
                "learning_tag": f"task_narrative_{int(time.time())}",
                "success_score": 0.85,
                "feedback_type": "test",
                "details": {"test": True}
            }
            
            async with self.session.post(f"{BASE_URL}/llm/feedback", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print(f"   ✅ Feedback submitted: {data['status']}")
                    print(f"   📊 Score: {data['success_score']}")
                    
                    self.test_results.append(("feedback", True, "Feedback submission successful"))
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Feedback failed: {response.status} - {error_text}")
                    self.test_results.append(("feedback", False, f"Status {response.status}"))
                    return False
                    
        except Exception as e:
            print(f"   ❌ Feedback error: {e}")
            self.test_results.append(("feedback", False, str(e)))
            return False
    
    async def test_telemetry_endpoints(self):
        """Test telemetry endpoints"""
        print("📡 Testing telemetry endpoints...")
        
        endpoints = [
            "/telemetry/llm/core-metrics",
            "/telemetry/llm/dashboard",
            "/telemetry/llm/agent-flowgraph"
        ]
        
        success_count = 0
        
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{BASE_URL}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        endpoint_name = endpoint.split("/")[-1]
                        
                        print(f"   ✅ {endpoint_name}: Data retrieved")
                        
                        # Basic validation
                        if isinstance(data, dict) and "timestamp" in data:
                            success_count += 1
                            self.test_results.append((f"telemetry_{endpoint_name}", True, "Telemetry data retrieved"))
                        else:
                            print(f"   ❌ {endpoint_name}: Invalid data structure")
                            self.test_results.append((f"telemetry_{endpoint_name}", False, "Invalid data structure"))
                    
                    else:
                        print(f"   ❌ {endpoint_name}: {response.status}")
                        self.test_results.append((f"telemetry_{endpoint_name}", False, f"Status {response.status}"))
                        
            except Exception as e:
                print(f"   ❌ {endpoint}: {e}")
                self.test_results.append((f"telemetry_{endpoint.split('/')[-1]}", False, str(e)))
        
        print(f"   📊 Telemetry: {success_count}/{len(endpoints)} successful")
        return success_count == len(endpoints)
    
    async def test_audit_logs(self):
        """Test audit log retrieval"""
        print("🔍 Testing audit logs...")
        
        try:
            async with self.session.get(f"{BASE_URL}/llm/audit?limit=10") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print(f"   ✅ Audit logs retrieved")
                    print(f"   📊 Entries: {data.get('total', 0)}")
                    
                    self.test_results.append(("audit", True, "Audit logs retrieved"))
                    return True
                else:
                    print(f"   ❌ Audit logs failed: {response.status}")
                    self.test_results.append(("audit", False, f"Status {response.status}"))
                    return False
                    
        except Exception as e:
            print(f"   ❌ Audit logs error: {e}")
            self.test_results.append(("audit", False, str(e)))
            return False
    
    async def run_all_tests(self):
        """Run all test suites"""
        await self.setup()
        
        test_functions = [
            self.test_health_check,
            self.test_llm_request,
            self.test_routing_recommendation,
            self.test_content_scoring,
            self.test_feedback_system,
            self.test_telemetry_endpoints,
            self.test_audit_logs
        ]
        
        results = []
        
        for test_func in test_functions:
            try:
                result = await test_func()
                results.append(result)
                print()  # Add spacing between tests
            except Exception as e:
                print(f"   ❌ Test {test_func.__name__} failed: {e}")
                results.append(False)
                print()
        
        await self.cleanup()
        
        # Print summary
        self.print_summary(results)
        
        return all(results)
    
    def print_summary(self, results):
        """Print test summary"""
        total_time = time.time() - self.start_time
        passed = sum(results)
        total = len(results)
        
        print("=" * 60)
        print("🧠 XORB LLM Cognitive Cortex Test Summary")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"✅ Passed: {passed}/{total} test suites")
        print(f"❌ Failed: {total - passed}/{total} test suites")
        print()
        
        # Detailed results
        print("📊 Detailed Results:")
        success_count = 0
        for test_name, success, message in self.test_results:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {message}")
            if success:
                success_count += 1
        
        print()
        print(f"📈 Individual Tests: {success_count}/{len(self.test_results)} passed")
        
        # Overall result
        if passed == total:
            print("🎉 ALL TESTS PASSED - LLM Cognitive Cortex is operational!")
        else:
            print("⚠️  Some tests failed - Check service configuration")
        
        print("=" * 60)


async def main():
    """Main test runner"""
    # Set default environment variables for testing
    if not os.getenv('OPENROUTER_API_KEY'):
        print("⚠️  OPENROUTER_API_KEY not set - some tests may fail")
    
    if not os.getenv('NVIDIA_API_KEY'):
        print("⚠️  NVIDIA_API_KEY not set - some tests may fail")
    
    tester = LLMCortexTester()
    success = await tester.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())