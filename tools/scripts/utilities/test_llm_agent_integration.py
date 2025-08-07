#!/usr/bin/env python3
"""
LLM-Enhanced Agent Integration Test

Comprehensive test suite for LLM-enhanced agents including the LLM Security
Analyst Agent and base LLM functionality integration.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add paths
sys.path.insert(0, '/root/Xorb')
sys.path.insert(0, '/root/Xorb/xorb_core')

from xorb_core.agents.llm_enhanced_base_agent import LLMEnhancedBaseAgent
from xorb_core.agents.llm_security_analyst_agent import LLMSecurityAnalystAgent
from xorb_core.agents.base_agent import AgentTask


class LLMAgentTester:
    """Comprehensive LLM-enhanced agent test suite"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
    
    async def test_llm_enhanced_base_agent(self):
        """Test LLM-enhanced base agent functionality"""
        print("🤖 Testing LLM-Enhanced Base Agent...")
        
        agent = LLMEnhancedBaseAgent(
            agent_id="test-llm-base-001",
            config={
                "llm_enabled": True,
                "llm_confidence_threshold": 0.6,
                "adaptive_learning": True
            }
        )
        
        success_count = 0
        
        # Test 1: Basic LLM analysis
        print("   Test 1: Basic LLM analysis")
        try:
            response = await agent.llm_analyze(
                "Analyze the security implications of using default credentials",
                "narrative"
            )
            
            if response and response.content:
                print(f"      ✅ Analysis completed: {len(response.content)} chars")
                print(f"      📊 Confidence: {response.confidence_score:.2f}")
                print(f"      ⚡ Latency: {response.latency_ms:.1f}ms")
                print(f"      🧠 Model: {response.model}")
                success_count += 1
                self.test_results.append(("llm_base_analysis", True, "Basic analysis successful"))
            else:
                print("      ❌ Analysis failed - no response")
                self.test_results.append(("llm_base_analysis", False, "No response"))
                
        except Exception as e:
            print(f"      ❌ Analysis failed: {e}")
            self.test_results.append(("llm_base_analysis", False, str(e)))
        
        # Test 2: Code analysis
        print("   Test 2: Code security analysis")
        try:
            test_code = """
def authenticate(username, password):
    if username == "admin" and password == "password123":
        return True
    return False
"""
            
            response = await agent.llm_code_analysis(test_code, "python", "security")
            
            if response and response.content:
                print(f"      ✅ Code analysis completed")
                print(f"      📊 Confidence: {response.confidence_score:.2f}")
                
                # Check if it identifies the hardcoded credential issue
                if "hardcoded" in response.content.lower() or "password" in response.content.lower():
                    print("      🔍 Security issue detected correctly")
                
                success_count += 1
                self.test_results.append(("llm_code_analysis", True, "Code analysis successful"))
            else:
                print("      ❌ Code analysis failed")
                self.test_results.append(("llm_code_analysis", False, "No response"))
                
        except Exception as e:
            print(f"      ❌ Code analysis failed: {e}")
            self.test_results.append(("llm_code_analysis", False, str(e)))
        
        # Test 3: Exploit analysis
        print("   Test 3: Exploit analysis")
        try:
            target_info = {
                "host": "example.com",
                "service": "web",
                "version": "Apache 2.4.41",
                "platform": "Linux"
            }
            
            vulnerability = {
                "type": "SQL Injection",
                "severity": "high",
                "description": "Unsanitized user input in login form",
                "cve": "CVE-2023-XXXX"
            }
            
            response = await agent.llm_exploit_analysis(target_info, vulnerability)
            
            if response and response.content:
                print(f"      ✅ Exploit analysis completed")
                print(f"      📊 Confidence: {response.confidence_score:.2f}")
                success_count += 1
                self.test_results.append(("llm_exploit_analysis", True, "Exploit analysis successful"))
            else:
                print("      ❌ Exploit analysis failed")
                self.test_results.append(("llm_exploit_analysis", False, "No response"))
                
        except Exception as e:
            print(f"      ❌ Exploit analysis failed: {e}")
            self.test_results.append(("llm_exploit_analysis", False, str(e)))
        
        # Test 4: Metrics retrieval
        print("   Test 4: LLM metrics")
        try:
            metrics = agent.get_llm_metrics()
            
            expected_keys = ["llm_enabled", "total_requests", "success_rate", "average_latency_ms"]
            missing_keys = [key for key in expected_keys if key not in metrics]
            
            if not missing_keys:
                print(f"      ✅ Metrics retrieved: {metrics['total_requests']} requests")
                print(f"      📈 Success rate: {metrics['success_rate']:.2f}")
                success_count += 1
                self.test_results.append(("llm_metrics", True, "Metrics retrieved successfully"))
            else:
                print(f"      ❌ Missing metrics keys: {missing_keys}")
                self.test_results.append(("llm_metrics", False, f"Missing keys: {missing_keys}"))
                
        except Exception as e:
            print(f"      ❌ Metrics retrieval failed: {e}")
            self.test_results.append(("llm_metrics", False, str(e)))
        
        print(f"   📊 Base Agent Tests: {success_count}/4 successful")
        return success_count == 4
    
    async def test_llm_security_analyst_agent(self):
        """Test LLM Security Analyst Agent functionality"""
        print("🛡️ Testing LLM Security Analyst Agent...")
        
        agent = LLMSecurityAnalystAgent(
            agent_id="test-security-analyst-001",
            config={
                "llm_enabled": True,
                "llm_confidence_threshold": 0.6,
                "adaptive_learning": True,
                "max_code_analysis_size": 5000
            }
        )
        
        success_count = 0
        
        # Test 1: Vulnerability Analysis Task
        print("   Test 1: Vulnerability analysis task")
        try:
            task = AgentTask(
                task_id="vuln-001",
                task_type="vulnerability_analysis",
                target="example.com",
                parameters={
                    "vulnerability": {
                        "type": "SQL Injection",
                        "severity": "high",
                        "description": "Union-based SQL injection in search parameter",
                        "cve": "CVE-2023-1234"
                    },
                    "target_info": {
                        "host": "example.com",
                        "service": "web application",
                        "version": "PHP 7.4",
                        "platform": "Linux Ubuntu 20.04"
                    }
                }
            )
            
            result = await agent.enhanced_execute(task)
            
            if result.success:
                print(f"      ✅ Vulnerability analysis completed")
                print(f"      📊 Confidence: {result.confidence:.2f}")
                print(f"      🔍 Findings: {len(result.findings)}")
                print(f"      ⚡ Execution time: {result.execution_time:.2f}s")
                
                if "vulnerability_analysis" in result.data:
                    print(f"      📝 Analysis length: {len(result.data['vulnerability_analysis'])} chars")
                
                success_count += 1
                self.test_results.append(("security_vuln_analysis", True, "Vulnerability analysis successful"))
            else:
                print(f"      ❌ Vulnerability analysis failed: {result.errors}")
                self.test_results.append(("security_vuln_analysis", False, f"Errors: {result.errors}"))
                
        except Exception as e:
            print(f"      ❌ Vulnerability analysis error: {e}")
            self.test_results.append(("security_vuln_analysis", False, str(e)))
        
        # Test 2: Code Security Review Task
        print("   Test 2: Code security review task")
        try:
            vulnerable_code = """
import sqlite3
from flask import Flask, request

app = Flask(__name__)

@app.route('/user')
def get_user():
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Vulnerable SQL query
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    
    return result
"""
            
            task = AgentTask(
                task_id="code-001",
                task_type="code_security_review",
                target="user_authentication_module",
                parameters={
                    "code": vulnerable_code,
                    "language": "python"
                }
            )
            
            result = await agent.enhanced_execute(task)
            
            if result.success:
                print(f"      ✅ Code review completed")
                print(f"      📊 Confidence: {result.confidence:.2f}")
                print(f"      🔍 Findings: {len(result.findings)}")
                
                # Check if SQL injection was detected
                sql_injection_detected = any(
                    "sql" in finding.get("type", "").lower() or 
                    "injection" in finding.get("description", "").lower()
                    for finding in result.findings
                )
                
                if sql_injection_detected:
                    print("      🎯 SQL injection vulnerability detected correctly")
                
                success_count += 1
                self.test_results.append(("security_code_review", True, "Code review successful"))
            else:
                print(f"      ❌ Code review failed: {result.errors}")
                self.test_results.append(("security_code_review", False, f"Errors: {result.errors}"))
                
        except Exception as e:
            print(f"      ❌ Code review error: {e}")
            self.test_results.append(("security_code_review", False, str(e)))
        
        # Test 3: Threat Report Generation
        print("   Test 3: Threat report generation")
        try:
            sample_findings = [
                {
                    "type": "sql_injection",
                    "severity": "high",
                    "description": "SQL injection in user login form",
                    "impact": "Data breach potential"
                },
                {
                    "type": "xss",
                    "severity": "medium", 
                    "description": "Reflected XSS in search functionality",
                    "impact": "Session hijacking risk"
                },
                {
                    "type": "weak_password_policy",
                    "severity": "low",
                    "description": "Weak password requirements",
                    "impact": "Brute force vulnerability"
                }
            ]
            
            task = AgentTask(
                task_id="report-001",
                task_type="threat_report_generation",
                target="webapp.example.com",
                parameters={
                    "findings": sample_findings,
                    "target": "webapp.example.com",
                    "report_type": "comprehensive"
                }
            )
            
            result = await agent.enhanced_execute(task)
            
            if result.success:
                print(f"      ✅ Threat report generated")
                print(f"      📊 Confidence: {result.confidence:.2f}")
                print(f"      📝 Report length: {len(result.data.get('threat_report', ''))} chars")
                print(f"      📈 Quality score: {result.data.get('quality_score', 0):.2f}")
                
                success_count += 1
                self.test_results.append(("security_threat_report", True, "Threat report successful"))
            else:
                print(f"      ❌ Threat report failed: {result.errors}")
                self.test_results.append(("security_threat_report", False, f"Errors: {result.errors}"))
                
        except Exception as e:
            print(f"      ❌ Threat report error: {e}")
            self.test_results.append(("security_threat_report", False, str(e)))
        
        # Test 4: Exploit Feasibility Assessment
        print("   Test 4: Exploit feasibility assessment")
        try:
            task = AgentTask(
                task_id="exploit-001",
                task_type="exploit_feasibility_assessment",
                target="target.example.com",
                parameters={
                    "vulnerability": {
                        "type": "Buffer Overflow",
                        "severity": "critical",
                        "description": "Stack-based buffer overflow in service daemon",
                        "cve": "CVE-2023-5678"
                    },
                    "target_info": {
                        "host": "target.example.com",
                        "service": "custom_daemon",
                        "version": "1.2.3",
                        "platform": "Linux x86_64"
                    },
                    "environment": {
                        "network_segmentation": False,
                        "monitoring_enabled": True,
                        "patches_current": False
                    }
                }
            )
            
            result = await agent.enhanced_execute(task)
            
            if result.success:
                print(f"      ✅ Exploit assessment completed")
                print(f"      📊 Confidence: {result.confidence:.2f}")
                print(f"      🎯 Feasibility score: {result.data.get('feasibility_score', 0):.2f}")
                print(f"      ⚠️ Risk level: {result.data.get('risk_assessment', 'unknown')}")
                
                success_count += 1
                self.test_results.append(("security_exploit_assessment", True, "Exploit assessment successful"))
            else:
                print(f"      ❌ Exploit assessment failed: {result.errors}")
                self.test_results.append(("security_exploit_assessment", False, f"Errors: {result.errors}"))
                
        except Exception as e:
            print(f"      ❌ Exploit assessment error: {e}")
            self.test_results.append(("security_exploit_assessment", False, str(e)))
        
        # Test 5: Specialized Metrics
        print("   Test 5: Specialized metrics")
        try:
            metrics = agent.get_specialized_metrics()
            
            specialized_keys = [
                "vulnerabilities_analyzed", "code_reviews_completed",
                "threat_reports_generated", "exploit_assessments"
            ]
            
            missing_keys = [key for key in specialized_keys if key not in metrics]
            
            if not missing_keys:
                print(f"      ✅ Specialized metrics retrieved")
                print(f"      🔍 Vulnerabilities analyzed: {metrics['vulnerabilities_analyzed']}")
                print(f"      📝 Code reviews: {metrics['code_reviews_completed']}")
                print(f"      📊 Reports generated: {metrics['threat_reports_generated']}")
                
                success_count += 1
                self.test_results.append(("security_metrics", True, "Specialized metrics successful"))
            else:
                print(f"      ❌ Missing specialized metrics: {missing_keys}")
                self.test_results.append(("security_metrics", False, f"Missing keys: {missing_keys}"))
                
        except Exception as e:
            print(f"      ❌ Specialized metrics error: {e}")
            self.test_results.append(("security_metrics", False, str(e)))
        
        print(f"   📊 Security Analyst Tests: {success_count}/5 successful")
        return success_count == 5
    
    async def test_adaptive_learning_workflow(self):
        """Test adaptive learning workflow"""
        print("🧠 Testing Adaptive Learning Workflow...")
        
        agent = LLMSecurityAnalystAgent(
            agent_id="test-learning-001",
            config={"adaptive_learning": True}
        )
        
        success_count = 0
        
        # Test learning feedback submission
        print("   Test 1: Learning feedback submission")
        try:
            # Simulate a successful analysis
            learning_tag = f"test_learning_{int(time.time())}"
            await agent.submit_llm_feedback(learning_tag, 0.9, {"test": True})
            
            print("      ✅ Feedback submitted successfully")
            print(f"      📈 Feedback submissions: {agent.feedback_submissions}")
            
            success_count += 1
            self.test_results.append(("adaptive_learning", True, "Feedback submission successful"))
            
        except Exception as e:
            print(f"      ❌ Feedback submission failed: {e}")
            self.test_results.append(("adaptive_learning", False, str(e)))
        
        print(f"   📊 Adaptive Learning Tests: {success_count}/1 successful")
        return success_count == 1
    
    async def run_all_tests(self):
        """Run all LLM agent integration tests"""
        print("🧠 XORB LLM-Enhanced Agent Integration Tests")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        test_functions = [
            self.test_llm_enhanced_base_agent,
            self.test_llm_security_analyst_agent,
            self.test_adaptive_learning_workflow
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
        
        # Print summary
        self.print_summary(results)
        
        return all(results)
    
    def print_summary(self, results):
        """Print test summary"""
        total_time = time.time() - self.start_time
        passed = sum(results)
        total = len(results)
        
        print("=" * 70)
        print("🧠 LLM-Enhanced Agent Test Summary")
        print("=" * 70)
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
            print("🎉 ALL TESTS PASSED - LLM-Enhanced Agents are operational!")
        else:
            print("⚠️  Some tests failed - Check LLM cortex service and configuration")
        
        print("=" * 70)


async def main():
    """Main test runner"""
    # Check if LLM cortex service is running
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8009/llm/health") as response:
                if response.status != 200:
                    print("❌ LLM Cortex service not available at localhost:8009")
                    print("   Please start the service first: python3 llm_cognitive_cortex_service.py")
                    sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to LLM Cortex service: {e}")
        print("   Please start the service first: python3 llm_cognitive_cortex_service.py")
        sys.exit(1)
    
    tester = LLMAgentTester()
    success = await tester.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())