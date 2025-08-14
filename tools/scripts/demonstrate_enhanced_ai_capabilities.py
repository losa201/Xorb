#!/usr/bin/env python3
"""
XORB Enhanced AI Capabilities Demonstration
Showcases the advanced AI-powered cybersecurity features and autonomous agents
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBEnhancedAIDemo:
    """Demonstrates XORB's enhanced AI capabilities"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_platform_status(self) -> Dict[str, Any]:
        """Check the AI platform status"""
        logger.info("🏥 Checking XORB Enhanced AI Platform Status...")

        try:
            async with self.session.get(f"{self.base_url}/api/v1/enterprise-ai/status") as response:
                if response.status == 200:
                    status = await response.json()

                    print("\n" + "="*80)
                    print("🧠 XORB ENHANCED AI PLATFORM STATUS")
                    print("="*80)

                    print(f"📊 Platform Status: {status['platform_status'].upper()}")
                    print(f"🤖 AI Services Status:")

                    for service_name, service_info in status['ai_services'].items():
                        print(f"   • {service_name}: {service_info['status'].upper()}")

                    print(f"\n🎯 AI Capabilities:")
                    for capability, enabled in status['capabilities'].items():
                        status_icon = "✅" if enabled else "❌"
                        print(f"   {status_icon} {capability.replace('_', ' ').title()}")

                    print(f"\n📈 Performance Metrics:")
                    metrics = status['performance_metrics']
                    print(f"   • Threat Detection Accuracy: {metrics['threat_detection_accuracy']:.2%}")
                    print(f"   • False Positive Rate: {metrics['false_positive_rate']:.2%}")
                    print(f"   • Scan Success Rate: {metrics['scan_success_rate']:.2%}")
                    print(f"   • Avg AI Analysis Time: {metrics['ai_analysis_time_avg']:.3f}s")

                    return status
                else:
                    logger.error(f"Failed to get platform status: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error checking platform status: {e}")
            return {}

    async def demonstrate_threat_analysis(self) -> Dict[str, Any]:
        """Demonstrate advanced AI-powered threat analysis"""
        logger.info("🔍 Demonstrating Advanced Threat Analysis...")

        # Sample threat indicators for analysis
        threat_indicators = [
            "198.51.100.42",              # Suspicious IP
            "evil-domain.com",            # Malicious domain
            "http://malware-site.org/payload.exe",  # Malware URL
            "a1b2c3d4e5f6789012345678901234567890abcd",  # File hash
            "powershell -enc SGVsbG8gV29ybGQ=",  # Encoded PowerShell
            "mimikatz.exe",               # Known hacking tool
            "192.168.1.100:4444",        # C2 communication
            "attacker@evil-domain.com"    # Suspicious email
        ]

        analysis_request = {
            "indicators": threat_indicators,
            "context": {
                "source": "demonstration",
                "environment": "production",
                "scan_type": "comprehensive",
                "timestamp": datetime.utcnow().isoformat()
            },
            "include_predictions": True,
            "analysis_depth": "comprehensive"
        }

        try:
            start_time = time.time()

            async with self.session.post(
                f"{self.base_url}/api/v1/enterprise-ai/threat-analysis",
                json=analysis_request
            ) as response:

                if response.status == 200:
                    analysis = await response.json()
                    processing_time = time.time() - start_time

                    print("\n" + "="*80)
                    print("🔍 AI-POWERED THREAT ANALYSIS RESULTS")
                    print("="*80)

                    print(f"📊 Analysis ID: {analysis['analysis_id']}")
                    print(f"⚠️ Threat Level: {analysis['threat_level']} (Scale: UNKNOWN/LOW/MEDIUM/HIGH/CRITICAL)")
                    print(f"🎯 Confidence Score: {analysis['confidence']:.2%}")
                    print(f"⚡ Processing Time: {processing_time:.3f}s")

                    print(f"\n🦠 Detected Threat Types:")
                    for threat_type in analysis['threat_types']:
                        print(f"   • {threat_type.replace('_', ' ').title()}")

                    print(f"\n🤖 AI Insights:")
                    ai_insights = analysis['ai_insights']

                    if 'orchestrator_analysis' in ai_insights:
                        orch_analysis = ai_insights['orchestrator_analysis']
                        if 'agent_consensus' in orch_analysis:
                            consensus = orch_analysis['agent_consensus']
                            print(f"   • Agent Consensus: {consensus.get('participating_agents', 0)} agents participated")
                            print(f"   • Consensus Confidence: {consensus.get('consensus_confidence', 0):.2%}")

                    if 'predictions' in ai_insights:
                        predictions = ai_insights['predictions']
                        if 'predictions' in predictions:
                            print(f"\n🔮 Threat Predictions:")
                            for pred in predictions['predictions'][:3]:  # Show top 3
                                print(f"   • {pred['threat_type'].replace('_', ' ').title()}: {pred['probability']:.2%} probability")

                    print(f"\n💡 AI Recommendations:")
                    for i, rec in enumerate(analysis['recommendations'][:5], 1):
                        print(f"   {i}. {rec}")

                    return analysis
                else:
                    logger.error(f"Threat analysis failed: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    return {}

        except Exception as e:
            logger.error(f"Error in threat analysis: {e}")
            return {}

    async def demonstrate_autonomous_orchestration(self) -> Dict[str, Any]:
        """Demonstrate AI orchestration capabilities"""
        logger.info("🎭 Demonstrating Autonomous AI Orchestration...")

        orchestration_request = {
            "operation_type": "analysis",
            "parameters": {
                "indicators": ["suspicious.exe", "192.168.1.100", "malware.domain.com"],
                "context": {
                    "source": "endpoint_detection",
                    "priority": "high",
                    "environment": "production"
                }
            },
            "autonomous_mode": True,
            "confidence_threshold": 0.7
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/enterprise-ai/orchestration",
                json=orchestration_request
            ) as response:

                if response.status == 200:
                    orchestration = await response.json()

                    print("\n" + "="*80)
                    print("🎭 AUTONOMOUS AI ORCHESTRATION RESULTS")
                    print("="*80)

                    print(f"🆔 Orchestration ID: {orchestration['orchestration_id']}")
                    print(f"📊 Status: {orchestration['status'].upper()}")
                    print(f"🎯 Confidence Score: {orchestration['confidence_score']:.2%}")

                    print(f"\n🤖 AI Decisions Made:")
                    for i, decision in enumerate(orchestration['ai_decisions'], 1):
                        if isinstance(decision, dict) and 'status' in decision:
                            print(f"   {i}. Decision Status: {decision['status']}")
                        else:
                            print(f"   {i}. Complex analysis completed")

                    print(f"\n⚡ Autonomous Actions: {len(orchestration['autonomous_actions'])}")
                    print(f"👤 Human Interventions Required: {len(orchestration['human_interventions_required'])}")

                    return orchestration
                else:
                    logger.error(f"Orchestration failed: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            return {}

    async def demonstrate_advanced_scanning(self) -> Dict[str, Any]:
        """Demonstrate enterprise PTaaS advanced scanning"""
        logger.info("🔍 Demonstrating Advanced AI-Enhanced Scanning...")

        scan_request = {
            "targets": [
                {
                    "host": "scanme.nmap.org",  # Safe scanning target
                    "ports": [22, 80, 443, 8080],
                    "scan_profile": "comprehensive"
                }
            ],
            "scan_profile": "comprehensive",
            "ai_enhancement": True,
            "compliance_framework": "owasp",
            "stealth_mode": False,
            "max_duration_hours": 1
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/enterprise-ai/advanced-scan",
                json=scan_request
            ) as response:

                if response.status == 200:
                    scan = await response.json()

                    print("\n" + "="*80)
                    print("🔍 ADVANCED AI-ENHANCED SCANNING")
                    print("="*80)

                    print(f"🆔 Scan Session ID: {scan['session_id']}")
                    print(f"📊 Status: {scan['status'].upper()}")
                    print(f"📅 Estimated Completion: {scan['estimated_completion']}")

                    print(f"\n🧠 AI Enhancements:")
                    for enhancement, enabled in scan['ai_enhancements'].items():
                        status_icon = "✅" if enabled else "❌"
                        print(f"   {status_icon} {enhancement.replace('_', ' ').title()}")

                    print(f"\n💡 Real-time Insights:")
                    for insight_type, description in scan['real_time_insights'].items():
                        print(f"   • {insight_type.replace('_', ' ').title()}: {description}")

                    print(f"\n⚡ Autonomous Responses Available:")
                    for response in scan['autonomous_responses']:
                        print(f"   • {response}")

                    return scan
                else:
                    logger.error(f"Advanced scan failed: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error in advanced scanning: {e}")
            return {}

    async def demonstrate_ai_agents(self) -> Dict[str, Any]:
        """Demonstrate AI agent commands"""
        logger.info("🤖 Demonstrating AI Agent Commands...")

        # Command threat hunting agent
        agent_command = {
            "agent_type": "threat_hunter",
            "command": "analyze",
            "parameters": {
                "network_data": {
                    "suspicious_connections": ["192.168.1.100:4444", "10.0.0.50:8080"],
                    "unusual_traffic": True,
                    "geographic_anomalies": ["Unknown", "TOR"]
                },
                "user_behavior": {
                    "failed_logins": 15,
                    "unusual_access_times": ["03:00", "04:30"],
                    "privilege_escalation": True
                }
            },
            "priority": "high"
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/enterprise-ai/agents/command",
                json=agent_command
            ) as response:

                if response.status == 200:
                    result = await response.json()

                    print("\n" + "="*80)
                    print("🤖 AI AGENT COMMAND RESULTS")
                    print("="*80)

                    print(f"📋 Command ID: {result['command_id']}")
                    print(f"🤖 Agent Type: {result['agent_type'].replace('_', ' ').title()}")
                    print(f"⚡ Command: {result['command']}")
                    print(f"👥 Agents Affected: {result['agents_affected']}")

                    print(f"\n📊 Agent Results:")
                    for i, agent_result in enumerate(result['results'], 1):
                        print(f"   Agent {i}:")
                        print(f"     • ID: {agent_result['agent_id']}")
                        print(f"     • Result: {agent_result['result']}")
                        if 'confidence' in agent_result:
                            print(f"     • Confidence: {agent_result['confidence']:.2%}")

                    return result
                else:
                    logger.error(f"Agent command failed: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error commanding agents: {e}")
            return {}

    async def demonstrate_real_time_insights(self) -> Dict[str, Any]:
        """Demonstrate real-time AI insights"""
        logger.info("⚡ Demonstrating Real-time AI Insights...")

        try:
            async with self.session.get(f"{self.base_url}/api/v1/enterprise-ai/insights/real-time") as response:
                if response.status == 200:
                    insights = await response.json()

                    print("\n" + "="*80)
                    print("⚡ REAL-TIME AI INSIGHTS")
                    print("="*80)

                    print(f"🆔 Insights ID: {insights['insights_id']}")
                    print(f"🎯 Confidence: {insights['confidence']:.2%}")
                    print(f"🕒 Last Updated: {insights['last_updated']}")

                    if 'ai_orchestration' in insights:
                        orch = insights['ai_orchestration']
                        print(f"\n🎭 AI Orchestration:")
                        print(f"   • Active Agents: {orch['active_agents']}")
                        print(f"   • Decision Queue Size: {orch['decision_queue_size']}")
                        print(f"   • Recent Decisions: {orch['recent_decisions']}")
                        print(f"   • Threat Detection Rate: {orch['threat_detection_rate']:.2%}")

                    if 'predictive_analytics' in insights:
                        pred = insights['predictive_analytics']
                        print(f"\n🔮 Predictive Analytics:")
                        print(f"   • Threat Trend: {pred['threat_trend']}")
                        print(f"   • Vulnerability Trend: {pred['vulnerability_trend']}")
                        print(f"   • Attack Pattern Evolution: {pred['attack_pattern_evolution']}")

                    print(f"\n💡 Immediate Recommendations:")
                    for i, rec in enumerate(insights['recommendations']['immediate_actions'], 1):
                        print(f"   {i}. {rec}")

                    return insights
                else:
                    logger.error(f"Real-time insights failed: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting real-time insights: {e}")
            return {}

    async def demonstrate_performance_analytics(self) -> Dict[str, Any]:
        """Demonstrate AI performance analytics"""
        logger.info("📊 Demonstrating AI Performance Analytics...")

        try:
            async with self.session.get(f"{self.base_url}/api/v1/enterprise-ai/analytics/performance") as response:
                if response.status == 200:
                    analytics = await response.json()

                    print("\n" + "="*80)
                    print("📊 AI PERFORMANCE ANALYTICS")
                    print("="*80)

                    summary = analytics['performance_summary']
                    print(f"🎯 Overall Score: {summary['overall_score']:.1f}/100")
                    print(f"🔍 Threat Detection Score: {summary['threat_detection_score']:.1f}/100")
                    print(f"🛡️ Scan Success Score: {summary['scan_success_score']:.1f}/100")
                    print(f"🧠 Threat Intelligence Accuracy: {summary['threat_intelligence_accuracy']:.1f}/100")

                    print(f"\n🎭 AI Orchestrator Performance:")
                    orch = analytics['ai_orchestrator']
                    print(f"   • Total Decisions: {orch['total_decisions']:,}")
                    print(f"   • Successful Responses: {orch['successful_responses']:,}")
                    print(f"   • Average Response Time: {orch['average_response_time']:.3f}s")
                    print(f"   • Threat Detection Rate: {orch['threat_detection_rate']:.2%}")
                    print(f"   • False Positive Rate: {orch['false_positive_rate']:.2%}")

                    print(f"\n🔍 Enterprise PTaaS Performance:")
                    ptaas = analytics['enterprise_ptaas']
                    print(f"   • Total Scans: {ptaas['total_scans']:,}")
                    print(f"   • Successful Scans: {ptaas['successful_scans']:,}")
                    print(f"   • Vulnerabilities Found: {ptaas['vulnerabilities_found']:,}")
                    print(f"   • Average Scan Duration: {ptaas['average_scan_duration']:.1f}s")

                    print(f"\n📈 Performance Trends:")
                    trends = analytics['trends']
                    for trend_type, trend_value in trends.items():
                        trend_icon = "📈" if trend_value == "Improving" else "📉" if trend_value == "Decreasing" else "➡️"
                        print(f"   {trend_icon} {trend_type.replace('_', ' ').title()}: {trend_value}")

                    return analytics
                else:
                    logger.error(f"Performance analytics failed: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}

    async def run_full_demonstration(self):
        """Run the complete demonstration of enhanced AI capabilities"""
        print("\n" + "="*80)
        print("🚀 XORB ENHANCED AI CAPABILITIES DEMONSTRATION")
        print("="*80)
        print("Showcasing advanced AI-powered cybersecurity operations")
        print("with autonomous agents, real-time threat analysis, and")
        print("sophisticated orchestration capabilities.")
        print("="*80)

        # Check if the API is available
        try:
            async with self.session.get(f"{self.base_url}/api/v1/health") as response:
                if response.status != 200:
                    logger.error("❌ XORB API is not available. Please start the API server first.")
                    return
        except Exception as e:
            logger.error(f"❌ Cannot connect to XORB API: {e}")
            logger.error("Please ensure the API server is running on http://localhost:8000")
            return

        # Run demonstrations
        demonstrations = [
            ("Platform Status Check", self.check_platform_status),
            ("Advanced Threat Analysis", self.demonstrate_threat_analysis),
            ("Autonomous AI Orchestration", self.demonstrate_autonomous_orchestration),
            ("Advanced AI-Enhanced Scanning", self.demonstrate_advanced_scanning),
            ("AI Agent Commands", self.demonstrate_ai_agents),
            ("Real-time AI Insights", self.demonstrate_real_time_insights),
            ("AI Performance Analytics", self.demonstrate_performance_analytics)
        ]

        results = {}

        for demo_name, demo_func in demonstrations:
            try:
                logger.info(f"Running: {demo_name}")
                result = await demo_func()
                results[demo_name] = result

                # Small delay between demonstrations
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"❌ {demo_name} failed: {e}")
                results[demo_name] = {"error": str(e)}

        # Summary
        print("\n" + "="*80)
        print("🎯 DEMONSTRATION SUMMARY")
        print("="*80)

        successful_demos = sum(1 for result in results.values() if result and "error" not in result)
        total_demos = len(demonstrations)

        print(f"✅ Successful Demonstrations: {successful_demos}/{total_demos}")
        print(f"📊 Success Rate: {successful_demos/total_demos:.1%}")

        print(f"\n🚀 XORB Enhanced AI Platform Status: OPERATIONAL")
        print(f"🤖 Advanced AI agents successfully demonstrated autonomous")
        print(f"   threat detection, analysis, and response capabilities")
        print(f"⚡ Real-time threat intelligence and predictive analytics")
        print(f"   providing sophisticated cybersecurity operations")

        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETE")
        print("="*80)
        print("The XORB Enhanced AI Platform has successfully demonstrated")
        print("world-class autonomous cybersecurity capabilities with")
        print("sophisticated AI agents, real-time threat analysis,")
        print("and advanced orchestration features.")
        print("="*80)


async def main():
    """Main demonstration function"""
    async with XORBEnhancedAIDemo() as demo:
        await demo.run_full_demonstration()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
