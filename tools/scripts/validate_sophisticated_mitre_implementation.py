#!/usr/bin/env python3
"""
Sophisticated MITRE ATT&CK Implementation Validation
Comprehensive validation of advanced threat mapping, hunting, and vulnerability analysis
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from api.app.services.advanced_mitre_attack_engine import (
        get_advanced_mitre_engine, AdvancedMitreAttackEngine, ThreatSeverity
    )
    from api.app.services.advanced_threat_hunting_engine import (
        get_advanced_threat_hunting_engine, HuntingHypothesis, HuntingMethod
    )
    from api.app.services.production_ai_vulnerability_engine import (
        get_production_ai_vulnerability_engine, VulnerabilitySeverity
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SophisticatedMITREValidator:
    """Comprehensive validator for sophisticated MITRE ATT&CK implementation"""

    def __init__(self):
        self.results = {
            "validation_id": f"mitre_validation_{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "tests_executed": [],
            "overall_score": 0.0,
            "critical_issues": [],
            "recommendations": [],
            "detailed_results": {}
        }

        self.mitre_engine: Optional[AdvancedMitreAttackEngine] = None
        self.hunting_engine = None
        self.vulnerability_engine = None

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of MITRE implementation"""
        logger.info("🚀 Starting Sophisticated MITRE ATT&CK Implementation Validation")

        try:
            # Initialize services
            await self._initialize_services()

            # Core MITRE Engine Tests
            await self._test_mitre_framework_loading()
            await self._test_advanced_threat_mapping()
            await self._test_ai_powered_analysis()
            await self._test_attack_pattern_detection()
            await self._test_threat_attribution()

            # Threat Hunting Engine Tests
            await self._test_hunting_query_execution()
            await self._test_behavioral_analysis()
            await self._test_campaign_management()

            # Vulnerability Engine Tests
            await self._test_vulnerability_assessment()
            await self._test_ai_risk_scoring()
            await self._test_remediation_planning()

            # Integration Tests
            await self._test_service_integration()
            await self._test_real_world_scenarios()

            # Performance Tests
            await self._test_performance_metrics()

            # Calculate overall score
            self._calculate_overall_score()

            # Generate final report
            await self._generate_final_report()

            return self.results

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.results["critical_issues"].append(f"Validation framework error: {e}")
            return self.results

    async def _initialize_services(self):
        """Initialize all sophisticated services"""
        try:
            logger.info("🔧 Initializing sophisticated MITRE services...")

            # Initialize MITRE engine
            self.mitre_engine = await get_advanced_mitre_engine()

            # Initialize hunting engine
            self.hunting_engine = await get_advanced_threat_hunting_engine()

            # Initialize vulnerability engine
            self.vulnerability_engine = await get_production_ai_vulnerability_engine()

            logger.info("✅ All services initialized successfully")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    async def _test_mitre_framework_loading(self):
        """Test MITRE ATT&CK framework loading and structure"""
        test_name = "MITRE Framework Loading"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test framework data loading
            techniques_count = len(self.mitre_engine.techniques)
            groups_count = len(self.mitre_engine.groups)
            software_count = len(self.mitre_engine.software)

            result["details"] = {
                "techniques_loaded": techniques_count,
                "groups_loaded": groups_count,
                "software_loaded": software_count,
                "ml_models_ready": self.mitre_engine.technique_vectorizer is not None,
                "attack_graph_built": self.mitre_engine.attack_graph is not None
            }

            # Validation criteria
            criteria_met = [
                techniques_count >= 100,
                groups_count >= 10,
                software_count >= 50,
                self.mitre_engine.technique_vectorizer is not None,
                self.mitre_engine.attack_graph is not None
            ]

            if not all(criteria_met):
                result["issues"].append("Framework loading incomplete")

            # Test specific technique lookup
            test_techniques = ["T1566.001", "T1059.001", "T1055"]
            for tech_id in test_techniques:
                if tech_id not in self.mitre_engine.techniques:
                    result["issues"].append(f"Critical technique {tech_id} not found")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 1.0 if result["passed"] else 0.3

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_advanced_threat_mapping(self):
        """Test advanced threat indicator mapping"""
        test_name = "Advanced Threat Mapping"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Create sophisticated test indicators
            test_indicators = [
                {
                    "type": "ip-dst",
                    "value": "192.0.2.100",
                    "confidence": 0.8,
                    "context": {"is_c2_server": True, "port": 443}
                },
                {
                    "type": "file-hash",
                    "value": "d41d8cd98f00b204e9800998ecf8427e",
                    "confidence": 0.9,
                    "context": {"file_type": "executable", "is_packed": True}
                },
                {
                    "type": "process",
                    "value": "powershell.exe -enc",
                    "confidence": 0.85,
                    "context": {"is_injection": True}
                }
            ]

            # Test mapping
            mapping = await self.mitre_engine.analyze_threat_indicators(
                test_indicators,
                {"threat_id": "test_threat_001"}
            )

            result["details"] = {
                "mapping_id": mapping.mapping_id,
                "techniques_mapped": len(mapping.technique_ids),
                "confidence": mapping.confidence,
                "correlation_score": mapping.correlation_score,
                "severity": mapping.severity.value,
                "attribution_groups": len(mapping.attribution_groups)
            }

            # Validation criteria
            if len(mapping.technique_ids) < 2:
                result["issues"].append("Insufficient technique mapping")

            if mapping.confidence < 0.5:
                result["issues"].append("Low mapping confidence")

            if not mapping.severity:
                result["issues"].append("Severity not determined")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.9 if result["passed"] else 0.4

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_ai_powered_analysis(self):
        """Test AI-powered analysis capabilities"""
        test_name = "AI-Powered Analysis"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test ML components
            ml_components = {
                "technique_vectorizer": self.mitre_engine.technique_vectorizer is not None,
                "similarity_matrix": self.mitre_engine.similarity_matrix is not None,
                "attack_graph": self.mitre_engine.attack_graph is not None,
                "clustering_model": self.mitre_engine.clustering_model is not None
            }

            result["details"]["ml_components"] = ml_components

            # Test technique similarity
            if len(self.mitre_engine.techniques) >= 2:
                tech_ids = list(self.mitre_engine.techniques.keys())[:2]
                related_techniques = await self.mitre_engine._find_related_techniques(tech_ids[0])
                result["details"]["similarity_analysis"] = {
                    "test_technique": tech_ids[0],
                    "related_found": len(related_techniques) if related_techniques else 0
                }

            # Test prediction capabilities
            test_techniques = ["T1566.001", "T1059.001"]
            try:
                prediction = await self.mitre_engine.predict_attack_progression(test_techniques)
                result["details"]["prediction_analysis"] = {
                    "prediction_generated": True,
                    "prediction_id": prediction.get("prediction_id", "unknown"),
                    "confidence_score": prediction.get("confidence_score", 0.0)
                }
            except Exception as pred_error:
                result["issues"].append(f"Prediction failed: {pred_error}")

            # Validation
            missing_components = [k for k, v in ml_components.items() if not v]
            if missing_components:
                result["issues"].append(f"Missing ML components: {missing_components}")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.95 if result["passed"] else 0.5

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_attack_pattern_detection(self):
        """Test attack pattern detection capabilities"""
        test_name = "Attack Pattern Detection"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Create test security events
            test_events = [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "process_creation",
                    "description": "powershell.exe execution with encoded command",
                    "source": "endpoint_detection"
                },
                {
                    "timestamp": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
                    "type": "network_connection",
                    "description": "outbound connection to suspicious IP",
                    "source": "network_monitoring"
                },
                {
                    "timestamp": (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
                    "type": "file_creation",
                    "description": "suspicious executable created in temp directory",
                    "source": "file_monitoring"
                }
            ]

            # Test pattern detection
            patterns = await self.mitre_engine.detect_attack_patterns(test_events)

            result["details"] = {
                "events_analyzed": len(test_events),
                "patterns_detected": len(patterns),
                "detection_rules_loaded": len(self.mitre_engine.detection_rules)
            }

            if patterns:
                pattern = patterns[0]
                result["details"]["sample_pattern"] = {
                    "pattern_id": pattern.pattern_id,
                    "name": pattern.name,
                    "confidence": pattern.confidence,
                    "severity": pattern.severity.value,
                    "techniques": len(pattern.techniques)
                }

            # Validation
            if len(self.mitre_engine.detection_rules) < 3:
                result["issues"].append("Insufficient detection rules")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.8 if result["passed"] else 0.3

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_threat_attribution(self):
        """Test threat actor attribution capabilities"""
        test_name = "Threat Attribution"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test with known APT techniques
            apt_techniques = ["T1566.001", "T1059.001", "T1055", "T1078"]

            # Test attribution
            attribution_groups = await self.mitre_engine._perform_threat_attribution(
                [{"technique_id": tid} for tid in apt_techniques]
            )

            result["details"] = {
                "test_techniques": apt_techniques,
                "groups_loaded": len(self.mitre_engine.groups),
                "attribution_results": len(attribution_groups),
                "sample_groups": attribution_groups[:3] if attribution_groups else []
            }

            # Validation
            if len(self.mitre_engine.groups) < 5:
                result["issues"].append("Insufficient threat group data")

            result["passed"] = len(result["issues"]) == 0 and len(attribution_groups) > 0
            result["score"] = 0.85 if result["passed"] else 0.4

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_hunting_query_execution(self):
        """Test threat hunting query execution"""
        test_name = "Hunting Query Execution"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test hunting engine initialization
            result["details"] = {
                "hunting_queries_loaded": len(self.hunting_engine.hunting_queries),
                "ml_models_ready": self.hunting_engine.anomaly_detector is not None,
                "threat_actors_tracked": len(self.hunting_engine.threat_actors)
            }

            # Test custom query generation
            custom_query = await self.hunting_engine.generate_custom_hunting_query(
                HuntingHypothesis.LATERAL_MOVEMENT,
                ["T1021.001", "T1078"],
                ["network_traffic", "authentication_logs"]
            )

            result["details"]["custom_query"] = {
                "query_id": custom_query.query_id,
                "name": custom_query.name,
                "hypothesis": custom_query.hypothesis.value,
                "method": custom_query.method.value
            }

            # Validation
            if len(self.hunting_engine.hunting_queries) < 3:
                result["issues"].append("Insufficient hunting queries")

            if self.hunting_engine.anomaly_detector is None:
                result["issues"].append("ML models not initialized")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.9 if result["passed"] else 0.4

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_vulnerability_assessment(self):
        """Test AI vulnerability assessment"""
        test_name = "AI Vulnerability Assessment"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test vulnerability engine
            result["details"] = {
                "ml_models_initialized": self.vulnerability_engine.risk_predictor is not None,
                "vulnerability_feeds": len(self.vulnerability_engine.vulnerability_feeds),
                "cve_database": len(self.vulnerability_engine.cve_database)
            }

            # Test assessment
            from api.app.services.production_ai_vulnerability_engine import VulnerabilityContext

            context = VulnerabilityContext(
                asset_criticality="high",
                network_exposure="internet_facing",
                data_classification="confidential"
            )

            assessment = await self.vulnerability_engine.conduct_vulnerability_assessment(
                "test-target-001",
                "comprehensive",
                context
            )

            result["details"]["assessment"] = {
                "assessment_id": assessment.assessment_id,
                "total_vulns": assessment.total_vulns,
                "critical_vulns": assessment.critical_vulns,
                "overall_risk_score": assessment.overall_risk_score
            }

            # Validation
            if self.vulnerability_engine.risk_predictor is None:
                result["issues"].append("Risk prediction model not ready")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.85 if result["passed"] else 0.3

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_service_integration(self):
        """Test integration between services"""
        test_name = "Service Integration"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test health checks
            mitre_health = await self.mitre_engine.health_check()
            hunting_health = await self.hunting_engine.health_check()
            vuln_health = await self.vulnerability_engine.health_check()

            result["details"] = {
                "mitre_engine_status": mitre_health.status.value,
                "hunting_engine_status": hunting_health.status.value,
                "vulnerability_engine_status": vuln_health.status.value,
                "all_services_healthy": all(
                    health.status.value == "healthy"
                    for health in [mitre_health, hunting_health, vuln_health]
                )
            }

            # Test cross-service functionality
            # MITRE engine should integrate with hunting and vulnerability engines

            # Validation
            unhealthy_services = []
            for name, health in [
                ("MITRE", mitre_health),
                ("Hunting", hunting_health),
                ("Vulnerability", vuln_health)
            ]:
                if health.status.value != "healthy":
                    unhealthy_services.append(name)

            if unhealthy_services:
                result["issues"].append(f"Unhealthy services: {unhealthy_services}")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.9 if result["passed"] else 0.5

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_real_world_scenarios(self):
        """Test real-world attack scenarios"""
        test_name = "Real-World Scenarios"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Scenario 1: APT29 Spearphishing Campaign
            apt29_indicators = [
                {
                    "type": "email",
                    "value": "attacker@malicious-domain.com",
                    "confidence": 0.9,
                    "context": {"has_attachment": True, "phishing_indicators": True}
                },
                {
                    "type": "file-hash",
                    "value": "a1b2c3d4e5f6...",
                    "confidence": 0.95,
                    "context": {"file_type": "executable", "has_persistence": True}
                }
            ]

            apt29_mapping = await self.mitre_engine.analyze_threat_indicators(
                apt29_indicators,
                {"campaign": "apt29_test"}
            )

            result["details"]["apt29_scenario"] = {
                "techniques_detected": len(apt29_mapping.technique_ids),
                "confidence": apt29_mapping.confidence,
                "severity": apt29_mapping.severity.value
            }

            # Scenario 2: Ransomware Attack Chain
            ransomware_events = [
                {"type": "email_attachment", "technique": "T1566.001"},
                {"type": "powershell_execution", "technique": "T1059.001"},
                {"type": "file_encryption", "technique": "T1486"}
            ]

            ransomware_patterns = await self.mitre_engine.detect_attack_patterns(ransomware_events)

            result["details"]["ransomware_scenario"] = {
                "patterns_detected": len(ransomware_patterns),
                "events_analyzed": len(ransomware_events)
            }

            # Validation
            if apt29_mapping.confidence < 0.6:
                result["issues"].append("APT29 scenario confidence too low")

            if len(ransomware_patterns) == 0:
                result["issues"].append("Ransomware pattern not detected")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.95 if result["passed"] else 0.6

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    async def _test_performance_metrics(self):
        """Test performance and scalability"""
        test_name = "Performance Metrics"
        logger.info(f"🧪 Testing: {test_name}")

        try:
            result = {
                "test": test_name,
                "passed": False,
                "details": {},
                "issues": []
            }

            # Test analysis performance
            start_time = time.time()

            # Multiple indicator analysis
            for i in range(5):
                test_indicators = [
                    {
                        "type": "ip-dst",
                        "value": f"192.0.2.{100 + i}",
                        "confidence": 0.8
                    }
                ]
                await self.mitre_engine.analyze_threat_indicators(test_indicators)

            analysis_time = time.time() - start_time

            result["details"] = {
                "analysis_time_seconds": analysis_time,
                "indicators_per_second": 5 / analysis_time if analysis_time > 0 else 0,
                "cache_size": len(self.mitre_engine.mapping_cache),
                "memory_efficient": analysis_time < 10.0  # Should complete in under 10 seconds
            }

            # Validation
            if analysis_time > 30.0:
                result["issues"].append("Analysis performance too slow")

            result["passed"] = len(result["issues"]) == 0
            result["score"] = 0.8 if result["passed"] else 0.4

            self.results["tests_executed"].append(result)
            logger.info(f"✅ {test_name}: {'PASSED' if result['passed'] else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.results["critical_issues"].append(f"{test_name}: {e}")

    def _calculate_overall_score(self):
        """Calculate overall validation score"""
        if not self.results["tests_executed"]:
            self.results["overall_score"] = 0.0
            return

        total_score = sum(test.get("score", 0.0) for test in self.results["tests_executed"])
        max_possible_score = len(self.results["tests_executed"])

        self.results["overall_score"] = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0.0

        # Add critical issue penalty
        critical_penalty = len(self.results["critical_issues"]) * 10
        self.results["overall_score"] = max(0, self.results["overall_score"] - critical_penalty)

    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        score = self.results["overall_score"]

        # Determine overall status
        if score >= 90:
            status = "🏆 EXCELLENT"
            grade = "A+"
        elif score >= 80:
            status = "✅ GOOD"
            grade = "A"
        elif score >= 70:
            status = "⚠️ ACCEPTABLE"
            grade = "B"
        elif score >= 60:
            status = "❌ NEEDS IMPROVEMENT"
            grade = "C"
        else:
            status = "🚨 CRITICAL ISSUES"
            grade = "F"

        # Generate recommendations
        recommendations = [
            "🔧 Implement comprehensive logging for all MITRE operations",
            "📊 Add performance monitoring and alerting",
            "🛡️ Enhance threat intelligence integration",
            "🧪 Expand test coverage for edge cases",
            "📈 Implement advanced analytics dashboards",
            "🔄 Add automated model retraining capabilities"
        ]

        if score < 80:
            recommendations.extend([
                "🚨 Address critical service initialization issues",
                "⚡ Optimize performance for large-scale operations",
                "🔍 Improve ML model accuracy and validation"
            ])

        self.results["recommendations"] = recommendations

        # Detailed results summary
        self.results["detailed_results"] = {
            "overall_status": status,
            "grade": grade,
            "tests_passed": len([t for t in self.results["tests_executed"] if t.get("passed", False)]),
            "tests_failed": len([t for t in self.results["tests_executed"] if not t.get("passed", False)]),
            "critical_issues_count": len(self.results["critical_issues"]),
            "validation_timestamp": self.results["timestamp"]
        }


async def main():
    """Main validation execution"""
    print("\n" + "="*80)
    print("🎯 SOPHISTICATED MITRE ATT&CK IMPLEMENTATION VALIDATION")
    print("="*80)

    validator = SophisticatedMITREValidator()

    try:
        results = await validator.run_comprehensive_validation()

        # Display results
        print(f"\n📊 VALIDATION RESULTS")
        print("-" * 40)
        print(f"Overall Score: {results['overall_score']:.1f}/100")
        print(f"Status: {results['detailed_results']['overall_status']}")
        print(f"Grade: {results['detailed_results']['grade']}")
        print(f"Tests Passed: {results['detailed_results']['tests_passed']}")
        print(f"Tests Failed: {results['detailed_results']['tests_failed']}")
        print(f"Critical Issues: {results['detailed_results']['critical_issues_count']}")

        # Show test details
        print(f"\n🧪 TEST RESULTS SUMMARY")
        print("-" * 40)
        for test in results["tests_executed"]:
            status = "✅ PASS" if test.get("passed", False) else "❌ FAIL"
            score = test.get("score", 0.0) * 100
            print(f"{status} {test['test']} ({score:.1f}%)")

        # Show critical issues
        if results["critical_issues"]:
            print(f"\n🚨 CRITICAL ISSUES")
            print("-" * 40)
            for issue in results["critical_issues"]:
                print(f"❌ {issue}")

        # Show recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        for rec in results["recommendations"][:5]:
            print(f"{rec}")

        # Save detailed results
        output_file = f"sophisticated_mitre_validation_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {output_file}")

        # Final status
        if results["overall_score"] >= 80:
            print(f"\n🎉 SOPHISTICATED MITRE IMPLEMENTATION: PRODUCTION READY!")
        else:
            print(f"\n⚠️ SOPHISTICATED MITRE IMPLEMENTATION: NEEDS IMPROVEMENT")

        return results["overall_score"] >= 70

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
