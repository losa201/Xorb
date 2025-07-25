#!/usr/bin/env python3
"""
Test Fixtures for Vector Store Similarity Testing
Phase 5.1 - Smart Triage Optimization Testing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List

from vector_store_service import VectorStoreService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test vulnerability data with known similarity patterns
TEST_VULNERABILITIES = [
    {
        "id": "test_vuln_001",
        "title": "SQL Injection in Login Form",
        "description": "Authentication bypass vulnerability in login endpoint allowing SQL injection attacks through username parameter",
        "severity": "high",
        "target": "https://example.com/login",
        "metadata": {"cvss_score": 8.1, "cwe_id": "CWE-89"}
    },
    {
        "id": "test_vuln_002", 
        "title": "SQL Injection in Authentication System",
        "description": "SQL injection vulnerability in authentication mechanism enabling bypass through malicious input in username field",
        "severity": "high",
        "target": "https://example.com/auth/login",
        "metadata": {"cvss_score": 8.3, "cwe_id": "CWE-89"}
    },
    {
        "id": "test_vuln_003",
        "title": "Cross-Site Scripting in Search Function", 
        "description": "Reflected XSS vulnerability in search functionality allowing execution of malicious JavaScript code",
        "severity": "medium",
        "target": "https://example.com/search",
        "metadata": {"cvss_score": 6.1, "cwe_id": "CWE-79"}
    },
    {
        "id": "test_vuln_004",
        "title": "Cross-Site Scripting in Search Results",
        "description": "XSS vulnerability in search results page enabling injection of arbitrary JavaScript through search parameters",
        "severity": "medium", 
        "target": "https://example.com/search/results",
        "metadata": {"cvss_score": 6.3, "cwe_id": "CWE-79"}
    },
    {
        "id": "test_vuln_005",
        "title": "Insecure Direct Object Reference",
        "description": "IDOR vulnerability allowing unauthorized access to user accounts through predictable user ID manipulation",
        "severity": "high",
        "target": "https://example.com/user/profile",
        "metadata": {"cvss_score": 7.5, "cwe_id": "CWE-639"}
    },
    {
        "id": "test_vuln_006",
        "title": "Command Injection in File Upload",
        "description": "OS command injection vulnerability in file upload functionality enabling remote code execution",
        "severity": "critical",
        "target": "https://example.com/upload",
        "metadata": {"cvss_score": 9.8, "cwe_id": "CWE-78"}
    },
    {
        "id": "test_vuln_007",
        "title": "Path Traversal in File Download",
        "description": "Directory traversal vulnerability in file download endpoint allowing access to sensitive system files",
        "severity": "high",
        "target": "https://example.com/download",
        "metadata": {"cvss_score": 7.5, "cwe_id": "CWE-22"}
    },
    {
        "id": "test_vuln_008",
        "title": "SQL Injection via User Registration",
        "description": "SQL injection in user registration form through email parameter enabling database manipulation",
        "severity": "high",
        "target": "https://example.com/register",
        "metadata": {"cvss_score": 8.0, "cwe_id": "CWE-89"}
    },
    {
        "id": "test_vuln_009",
        "title": "Stored XSS in Comment System",
        "description": "Persistent cross-site scripting vulnerability in comment functionality allowing stored JavaScript injection",
        "severity": "high",
        "target": "https://example.com/comments",
        "metadata": {"cvss_score": 7.2, "cwe_id": "CWE-79"}
    },
    {
        "id": "test_vuln_010",
        "title": "Authentication Bypass in Admin Panel",
        "description": "Logic flaw in admin authentication allowing unauthorized access to administrative functions",
        "severity": "critical",
        "target": "https://example.com/admin",
        "metadata": {"cvss_score": 9.1, "cwe_id": "CWE-287"}
    }
]

# Expected similarity relationships for testing
EXPECTED_SIMILARITIES = {
    "test_vuln_001": ["test_vuln_002", "test_vuln_008"],  # SQL injection variants
    "test_vuln_002": ["test_vuln_001", "test_vuln_008"],  # SQL injection variants
    "test_vuln_003": ["test_vuln_004", "test_vuln_009"],  # XSS variants
    "test_vuln_004": ["test_vuln_003", "test_vuln_009"],  # XSS variants
    "test_vuln_008": ["test_vuln_001", "test_vuln_002"],  # SQL injection variants
    "test_vuln_009": ["test_vuln_003", "test_vuln_004"],  # XSS variants
}

class SimilarityTestSuite:
    """Test suite for vulnerability similarity detection"""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.test_results = {}
        
    async def initialize(self):
        """Initialize test suite"""
        logger.info("Initializing similarity test suite...")
        await self.vector_store.initialize()
        
    async def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        logger.info("Running similarity test suite...")
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {}
        }
        
        # Test 1: Basic similarity detection
        test_results["tests"]["basic_similarity"] = await self.test_basic_similarity()
        
        # Test 2: Duplicate detection accuracy
        test_results["tests"]["duplicate_detection"] = await self.test_duplicate_detection()
        
        # Test 3: False positive rates
        test_results["tests"]["false_positive_rates"] = await self.test_false_positive_rates()
        
        # Test 4: Performance benchmarks
        test_results["tests"]["performance"] = await self.test_performance()
        
        # Test 5: GPT reranking effectiveness
        test_results["tests"]["gpt_reranking"] = await self.test_gpt_reranking()
        
        # Generate summary
        test_results["summary"] = self._generate_test_summary(test_results["tests"])
        
        logger.info("Test suite completed", summary=test_results["summary"])
        return test_results
    
    async def test_basic_similarity(self) -> Dict:
        """Test basic similarity detection functionality"""
        logger.info("Testing basic similarity detection...")
        
        results = {
            "name": "Basic Similarity Detection",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Add all test vulnerabilities to vector store
        for vuln in TEST_VULNERABILITIES:
            await self.vector_store.add_vulnerability_vector(
                vulnerability_id=vuln["id"],
                title=vuln["title"],
                description=vuln["description"],
                severity=vuln["severity"],
                target=vuln["target"],
                metadata=vuln["metadata"]
            )
        
        # Test similarity detection for each vulnerability
        for vuln in TEST_VULNERABILITIES:
            similar_findings = await self.vector_store.find_similar_vulnerabilities(
                vulnerability_id=vuln["id"],
                title=vuln["title"],
                description=vuln["description"],
                severity=vuln["severity"],
                target=vuln["target"],
                k=5
            )
            
            expected_similar = EXPECTED_SIMILARITIES.get(vuln["id"], [])
            found_similar = [sf.vulnerability_id for sf in similar_findings]
            
            # Check if expected similarities were found
            matches = len(set(expected_similar) & set(found_similar))
            expected_count = len(expected_similar)
            
            test_detail = {
                "vulnerability_id": vuln["id"],
                "expected_similar": expected_similar,
                "found_similar": found_similar[:3],  # Top 3
                "matches": matches,
                "expected_count": expected_count,
                "accuracy": matches / expected_count if expected_count > 0 else 1.0
            }
            
            if matches >= expected_count * 0.5:  # At least 50% accuracy
                results["passed"] += 1
                test_detail["status"] = "PASS"
            else:
                results["failed"] += 1
                test_detail["status"] = "FAIL"
            
            results["details"].append(test_detail)
        
        results["overall_accuracy"] = results["passed"] / (results["passed"] + results["failed"])
        return results
    
    async def test_duplicate_detection(self) -> Dict:
        """Test duplicate detection accuracy"""
        logger.info("Testing duplicate detection...")
        
        results = {
            "name": "Duplicate Detection Accuracy",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Test with near-duplicate vulnerabilities
        near_duplicates = [
            {
                "id": "duplicate_test_001",
                "title": "SQL Injection in Login Form",  # Exact match
                "description": "Authentication bypass vulnerability in login endpoint allowing SQL injection attacks through username parameter",
                "severity": "high",
                "target": "https://example.com/login",
                "should_be_duplicate": True,
                "duplicate_of": "test_vuln_001"
            },
            {
                "id": "duplicate_test_002", 
                "title": "SQL Injection in Login Endpoint",  # Very similar
                "description": "SQL injection vulnerability in login form allowing authentication bypass via username field manipulation",
                "severity": "high",
                "target": "https://example.com/login",
                "should_be_duplicate": True,
                "duplicate_of": "test_vuln_001"
            },
            {
                "id": "duplicate_test_003",
                "title": "Buffer Overflow in Memory Management",  # Completely different
                "description": "Buffer overflow vulnerability in memory allocation routines enabling arbitrary code execution",
                "severity": "critical",
                "target": "https://different.com/api",
                "should_be_duplicate": False,
                "duplicate_of": None
            }
        ]
        
        for test_vuln in near_duplicates:
            dedupe_result = await self.vector_store.detect_duplicate(
                vulnerability_id=test_vuln["id"],
                title=test_vuln["title"],
                description=test_vuln["description"],
                severity=test_vuln["severity"],
                target=test_vuln["target"],
                use_gpt_fallback=True  # Enable GPT for better accuracy
            )
            
            expected_duplicate = test_vuln["should_be_duplicate"]
            detected_duplicate = dedupe_result.is_duplicate
            
            test_detail = {
                "vulnerability_id": test_vuln["id"],
                "expected_duplicate": expected_duplicate,
                "detected_duplicate": detected_duplicate,
                "confidence": dedupe_result.confidence,
                "duplicate_of": dedupe_result.duplicate_of,
                "expected_duplicate_of": test_vuln["duplicate_of"],
                "reasoning": dedupe_result.reasoning
            }
            
            if expected_duplicate == detected_duplicate:
                results["passed"] += 1
                test_detail["status"] = "PASS"
            else:
                results["failed"] += 1
                test_detail["status"] = "FAIL"
            
            results["details"].append(test_detail)
        
        results["accuracy"] = results["passed"] / (results["passed"] + results["failed"])
        return results
    
    async def test_false_positive_rates(self) -> Dict:
        """Test false positive detection rates"""
        logger.info("Testing false positive rates...")
        
        results = {
            "name": "False Positive Rate Analysis",
            "total_tests": 0,
            "false_positives": 0,
            "details": []
        }
        
        # Test with dissimilar vulnerabilities that should NOT be duplicates
        dissimilar_tests = [
            ("test_vuln_005", "test_vuln_006"),  # IDOR vs Command Injection
            ("test_vuln_007", "test_vuln_010"),  # Path Traversal vs Auth Bypass
            ("test_vuln_006", "test_vuln_003"),  # Command Injection vs XSS
        ]
        
        for vuln1_id, vuln2_id in dissimilar_tests:
            vuln1 = next(v for v in TEST_VULNERABILITIES if v["id"] == vuln1_id)
            
            # Test if vuln1 is incorrectly detected as duplicate of vuln2
            dedupe_result = await self.vector_store.detect_duplicate(
                vulnerability_id=f"fp_test_{vuln1_id}",
                title=vuln1["title"],
                description=vuln1["description"],
                severity=vuln1["severity"],
                target=vuln1["target"]
            )
            
            results["total_tests"] += 1
            
            test_detail = {
                "test_vuln": vuln1_id,
                "compared_to": vuln2_id,
                "detected_duplicate": dedupe_result.is_duplicate,
                "confidence": dedupe_result.confidence,
                "similarity_scores": [sf.similarity_score for sf in dedupe_result.similar_findings[:3]]
            }
            
            if dedupe_result.is_duplicate:
                results["false_positives"] += 1
                test_detail["status"] = "FALSE_POSITIVE"
            else:
                test_detail["status"] = "CORRECT"
            
            results["details"].append(test_detail)
        
        results["false_positive_rate"] = results["false_positives"] / results["total_tests"] if results["total_tests"] > 0 else 0
        return results
    
    async def test_performance(self) -> Dict:
        """Test performance benchmarks"""
        logger.info("Testing performance...")
        
        import time
        
        results = {
            "name": "Performance Benchmarks",
            "metrics": {}
        }
        
        # Test embedding generation speed
        start_time = time.time()
        test_text = "SQL injection vulnerability in authentication system"
        for _ in range(10):
            await self.vector_store.generate_embedding(test_text)
        embedding_time = (time.time() - start_time) / 10
        results["metrics"]["avg_embedding_time"] = embedding_time
        
        # Test similarity search speed
        start_time = time.time()
        for _ in range(10):
            await self.vector_store.find_similar_vulnerabilities(
                vulnerability_id="perf_test",
                title="Test vulnerability",
                description="Test description for performance testing",
                severity="high",
                target="https://test.com"
            )
        search_time = (time.time() - start_time) / 10
        results["metrics"]["avg_search_time"] = search_time
        
        # Test full deduplication pipeline speed
        start_time = time.time()
        for i in range(5):
            await self.vector_store.detect_duplicate(
                vulnerability_id=f"perf_test_{i}",
                title=f"Performance test vulnerability {i}",
                description="Test description for performance measurement",
                severity="medium",
                target="https://perftest.com",
                use_gpt_fallback=False  # Disable GPT for speed test
            )
        dedupe_time = (time.time() - start_time) / 5
        results["metrics"]["avg_deduplication_time"] = dedupe_time
        
        # Performance thresholds
        results["performance_analysis"] = {
            "embedding_performance": "GOOD" if embedding_time < 0.1 else "NEEDS_IMPROVEMENT",
            "search_performance": "GOOD" if search_time < 0.05 else "NEEDS_IMPROVEMENT", 
            "deduplication_performance": "GOOD" if dedupe_time < 0.5 else "NEEDS_IMPROVEMENT"
        }
        
        return results
    
    async def test_gpt_reranking(self) -> Dict:
        """Test GPT reranking effectiveness"""
        logger.info("Testing GPT reranking...")
        
        results = {
            "name": "GPT Reranking Effectiveness",
            "with_gpt": 0,
            "without_gpt": 0,
            "improvements": 0,
            "details": []
        }
        
        # Test borderline cases where GPT should help
        borderline_cases = [
            {
                "id": "gpt_test_001",
                "title": "Authentication Bypass via SQL Injection",
                "description": "SQL injection in login allowing bypass of authentication checks",
                "severity": "high",
                "target": "https://example.com/auth"
            }
        ]
        
        for test_case in borderline_cases:
            # Test without GPT
            result_without_gpt = await self.vector_store.detect_duplicate(
                vulnerability_id=f"{test_case['id']}_no_gpt",
                title=test_case["title"],
                description=test_case["description"],
                severity=test_case["severity"],
                target=test_case["target"],
                use_gpt_fallback=False
            )
            
            # Test with GPT
            result_with_gpt = await self.vector_store.detect_duplicate(
                vulnerability_id=f"{test_case['id']}_with_gpt",
                title=test_case["title"],
                description=test_case["description"],
                severity=test_case["severity"],
                target=test_case["target"],
                use_gpt_fallback=True
            )
            
            test_detail = {
                "test_case": test_case["id"],
                "without_gpt": {
                    "is_duplicate": result_without_gpt.is_duplicate,
                    "confidence": result_without_gpt.confidence
                },
                "with_gpt": {
                    "is_duplicate": result_with_gpt.is_duplicate,
                    "confidence": result_with_gpt.confidence,
                    "gpt_analysis": result_with_gpt.gpt_analysis is not None
                },
                "improvement": abs(result_with_gpt.confidence - result_without_gpt.confidence)
            }
            
            if result_with_gpt.gpt_analysis:
                results["with_gpt"] += 1
                if test_detail["improvement"] > 0.1:  # Significant improvement
                    results["improvements"] += 1
            else:
                results["without_gpt"] += 1
            
            results["details"].append(test_detail)
        
        return results
    
    def _generate_test_summary(self, test_results: Dict) -> Dict:
        """Generate overall test summary"""
        summary = {
            "total_tests": len(test_results),
            "passed_tests": 0,
            "failed_tests": 0,
            "overall_score": 0.0,
            "recommendations": []
        }
        
        scores = []
        
        for test_name, test_result in test_results.items():
            if "accuracy" in test_result:
                scores.append(test_result["accuracy"])
                if test_result["accuracy"] >= 0.8:
                    summary["passed_tests"] += 1
                else:
                    summary["failed_tests"] += 1
            elif "false_positive_rate" in test_result:
                scores.append(1.0 - test_result["false_positive_rate"])  # Invert FP rate
                if test_result["false_positive_rate"] <= 0.1:  # Less than 10% FP rate
                    summary["passed_tests"] += 1
                else:
                    summary["failed_tests"] += 1
        
        if scores:
            summary["overall_score"] = sum(scores) / len(scores)
        
        # Generate recommendations
        if summary["overall_score"] < 0.8:
            summary["recommendations"].append("Consider tuning similarity thresholds")
        
        if test_results.get("false_positive_rates", {}).get("false_positive_rate", 0) > 0.1:
            summary["recommendations"].append("Review false positive detection logic")
        
        if not summary["recommendations"]:
            summary["recommendations"].append("System performing within acceptable parameters")
        
        return summary

async def run_similarity_tests():
    """Run the complete similarity test suite"""
    test_suite = SimilarityTestSuite()
    
    try:
        await test_suite.initialize()
        results = await test_suite.run_all_tests()
        
        # Save results
        with open("/tmp/similarity_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("SIMILARITY TEST SUITE RESULTS")
        print("="*60)
        print(f"Overall Score: {results['summary']['overall_score']:.2%}")
        print(f"Passed Tests: {results['summary']['passed_tests']}")
        print(f"Failed Tests: {results['summary']['failed_tests']}")
        print("\nRecommendations:")
        for rec in results['summary']['recommendations']:
            print(f"  - {rec}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run tests when script is executed directly
    results = asyncio.run(run_similarity_tests())
    exit(0 if results.get("summary", {}).get("overall_score", 0) >= 0.8 else 1)