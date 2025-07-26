#!/usr/bin/env python3
"""
AI-Powered Threat Hunting Demonstration

This script demonstrates the comprehensive AI-driven threat hunting capabilities
including anomaly detection, behavioral analysis, hypothesis generation,
and automated threat hunting workflows.
"""

import asyncio
import json
import random
import time
from pathlib import Path
import sys

# Add the xorb_core to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xorb_core.hunting import (
    AIThreatHunter,
    DataPoint,
    HuntTrigger,
    ThreatType,
    AnomalyType,
    ai_threat_hunter
)
from xorb_core.intelligence.threat_intelligence_engine import (
    IoC, IoC_Type, ThreatLevel, threat_intel_engine
)
from xorb_core.vulnerabilities import (
    Vulnerability, VulnerabilitySeverity, VulnerabilityCategory, vulnerability_manager
)

import structlog

logger = structlog.get_logger(__name__)


class AIThreatHuntingDemo:
    """Comprehensive AI threat hunting demonstration."""
    
    def __init__(self):
        self.demo_data_points = []
        self.demo_session_ids = []
        
    async def run_demo(self):
        """Run the complete AI threat hunting demonstration."""
        print("🤖 AI-POWERED THREAT HUNTING DEMONSTRATION")
        print("=" * 60)
        print("This demo showcases advanced AI threat hunting capabilities:")
        print("• Intelligent anomaly detection and behavioral analysis")
        print("• AI-powered threat hypothesis generation")
        print("• Automated hunt session management")
        print("• Integration with threat intelligence and vulnerabilities")
        print("• Real-time threat correlation and risk assessment")
        print("=" * 60)
        
        try:
            # Initialize systems
            await self._initialize_systems()
            
            # Run demonstration scenarios
            await self._demo_data_ingestion()
            await self._demo_anomaly_detection()
            await self._demo_threat_hypothesis_generation()
            await self._demo_automated_hunting()
            await self._demo_threat_correlation()
            await self._demo_reporting_analytics()
            
            # Generate final report
            await self._generate_summary_report()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self._cleanup_systems()
    
    async def _initialize_systems(self):
        """Initialize the AI threat hunting systems."""
        print("\n🚀 Initializing AI Threat Hunting Systems...")
        
        # Initialize the AI threat hunter (demo mode)
        ai_threat_hunter.running = True
        
        # Add some sample IoCs to threat intelligence
        sample_iocs = [
            IoC(
                value="203.0.113.10",
                ioc_type=IoC_Type.IP_ADDRESS,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                source="demo_feed",
                description="Known C2 server"
            ),
            IoC(
                value="malware.example.com",
                ioc_type=IoC_Type.DOMAIN,
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.95,
                source="demo_feed",
                description="Malware distribution domain"
            ),
            IoC(
                value="a1b2c3d4e5f6789...",
                ioc_type=IoC_Type.FILE_HASH,
                threat_level=ThreatLevel.HIGH,
                confidence=0.8,
                source="demo_feed",
                description="Malicious file hash"
            )
        ]
        
        for ioc in sample_iocs:
            threat_intel_engine.ioc_cache[ioc.get_id()] = ioc
        
        # Add sample vulnerabilities
        sample_vuln = Vulnerability(
            title="Critical RCE in Web Application",
            description="Remote code execution vulnerability discovered",
            severity=VulnerabilitySeverity.CRITICAL,
            category=VulnerabilityCategory.INJECTION,
            target_name="Production Web Server",
            location="https://webapp.example.com/api/exec"
        )
        await vulnerability_manager.add_vulnerability(sample_vuln)
        
        print("✅ AI threat hunting systems initialized")
    
    async def _demo_data_ingestion(self):
        """Demonstrate data ingestion and preprocessing."""
        print("\n📊 DATA INGESTION & PREPROCESSING DEMONSTRATION")
        print("-" * 55)
        
        print("🔄 Ingesting security event data...")
        
        # Generate diverse sample data points
        sample_data = [
            # Normal login activities
            *[{
                "timestamp": time.time() - random.randint(0, 86400),
                "source": "auth_system",
                "event_type": "login",
                "data": {
                    "user_id": f"user_{random.randint(1, 20)}",
                    "source_ip": f"192.168.1.{random.randint(1, 50)}",
                    "success": True,
                    "duration": random.uniform(0.5, 3.0),
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }
            } for _ in range(50)],
            
            # Suspicious after-hours activities
            *[{
                "timestamp": time.time() - random.randint(0, 86400),
                "source": "auth_system", 
                "event_type": "login",
                "data": {
                    "user_id": "admin_user",
                    "source_ip": "203.0.113.10",  # Matches IoC
                    "success": True,
                    "duration": random.uniform(0.1, 0.5),
                    "user_agent": "curl/7.68.0"
                }
            } for _ in range(8)],
            
            # Excessive file access
            *[{
                "timestamp": time.time() - random.randint(0, 3600),
                "source": "file_system",
                "event_type": "file_access",
                "data": {
                    "user_id": "intern_user",
                    "resource": f"/sensitive/data/file_{i}.txt",
                    "action": "read",
                    "size": random.randint(1000, 100000),
                    "success": True
                }
            } for i in range(100)],
            
            # Network anomalies
            *[{
                "timestamp": time.time() - random.randint(0, 7200),
                "source": "network_monitor",
                "event_type": "network_connection",
                "data": {
                    "source_ip": f"192.168.1.{random.randint(100, 200)}",
                    "dest_ip": "203.0.113.10",  # Matches IoC
                    "port": 4444,
                    "bytes_transferred": random.randint(1000, 10000),
                    "protocol": "tcp",
                    "duration": random.uniform(10, 300)
                }
            } for _ in range(20)],
            
            # High-volume data transfers
            *[{
                "timestamp": time.time() - random.randint(0, 1800),
                "source": "network_monitor",
                "event_type": "data_transfer",
                "data": {
                    "user_id": "contractor_user",
                    "source_ip": "192.168.1.150",
                    "dest_ip": "external.example.com",
                    "bytes_transferred": random.randint(100000000, 1000000000),  # 100MB-1GB
                    "protocol": "https",
                    "duration": random.uniform(300, 3600)
                }
            } for _ in range(5)]
        ]
        
        # Convert to DataPoint objects and add to hunter
        for data in sample_data:
            data_point = DataPoint(
                timestamp=data["timestamp"],
                source=data["source"],
                event_type=data["event_type"],
                data=data["data"]
            )
            ai_threat_hunter.data_buffer.append(data_point)
            self.demo_data_points.append(data_point)
        
        print(f"   • Ingested {len(sample_data)} security events")
        print(f"   • Event types: login, file_access, network_connection, data_transfer")
        print(f"   • Time range: Last 24 hours")
        print(f"   • Data buffer size: {len(ai_threat_hunter.data_buffer)} events")
        
        print("\n✅ Data ingestion completed")
    
    async def _demo_anomaly_detection(self):
        """Demonstrate anomaly detection capabilities."""
        print("\n🔍 ANOMALY DETECTION DEMONSTRATION")
        print("-" * 50)
        
        print("🤖 Running AI anomaly detection algorithms...")
        
        # Run anomaly detection on our sample data
        all_anomalies = []
        
        for detector in ai_threat_hunter.anomaly_detectors:
            print(f"\n📈 Running {detector.get_detector_name()}...")
            
            anomalies = await detector.detect_anomalies(self.demo_data_points)
            all_anomalies.extend(anomalies)
            
            print(f"   • Detected {len(anomalies)} anomalies")
            
            # Show some example anomalies
            for i, anomaly in enumerate(anomalies[:3]):  # Show first 3
                print(f"     {i+1}. {anomaly.title}")
                print(f"        Severity: {anomaly.severity:.2f}, Confidence: {anomaly.confidence:.2f}")
                print(f"        Type: {anomaly.anomaly_type.value}")
        
        print(f"\n📊 Anomaly Detection Summary:")
        print(f"   • Total anomalies detected: {len(all_anomalies)}")
        
        # Group by type
        by_type = {}
        for anomaly in all_anomalies:
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
        
        for anomaly_type, count in by_type.items():
            print(f"   • {anomaly_type.replace('_', ' ').title()}: {count}")
        
        # Show high-severity anomalies
        high_severity = [a for a in all_anomalies if a.severity > 0.6]
        print(f"   • High-severity anomalies: {len(high_severity)}")
        
        if high_severity:
            print(f"\n🚨 High-Severity Anomalies:")
            for anomaly in high_severity[:5]:  # Show top 5
                print(f"   • {anomaly.title}")
                print(f"     Severity: {anomaly.severity:.2f}, Confidence: {anomaly.confidence:.2f}")
                if anomaly.affected_entities:
                    print(f"     Affected: {', '.join(anomaly.affected_entities)}")
        
        print("\n✅ Anomaly detection completed")
        return all_anomalies
    
    async def _demo_threat_hypothesis_generation(self):
        """Demonstrate AI threat hypothesis generation."""
        print("\n🧠 AI THREAT HYPOTHESIS GENERATION DEMONSTRATION")
        print("-" * 60)
        
        print("🤖 Generating AI-powered threat hypotheses...")
        
        # Run anomaly detection first
        all_anomalies = []
        for detector in ai_threat_hunter.anomaly_detectors:
            anomalies = await detector.detect_anomalies(self.demo_data_points)
            all_anomalies.extend(anomalies)
        
        # Get relevant IoCs
        relevant_iocs = list(threat_intel_engine.ioc_cache.values())
        
        # Generate hypotheses
        hypotheses = await ai_threat_hunter.hypothesis_generator.generate_hypotheses(
            all_anomalies, relevant_iocs
        )
        
        print(f"\n📋 Generated {len(hypotheses)} threat hypotheses")
        
        if hypotheses:
            print(f"\n🎯 Threat Hypotheses (ranked by risk score):")
            
            for i, hypothesis in enumerate(hypotheses[:5], 1):  # Show top 5
                print(f"\n   {i}. {hypothesis.title}")
                print(f"      Threat Type: {hypothesis.threat_type.value.replace('_', ' ').title()}")
                print(f"      Risk Score: {hypothesis.risk_score:.3f}")
                print(f"      Confidence: {hypothesis.confidence:.2f}")
                print(f"      Impact: {hypothesis.impact_score:.2f}, Likelihood: {hypothesis.likelihood_score:.2f}")
                print(f"      Description: {hypothesis.description}")
                
                if hypothesis.supporting_evidence:
                    print(f"      Evidence: {len(hypothesis.supporting_evidence)} items")
                    for evidence in hypothesis.supporting_evidence[:2]:
                        print(f"        • {evidence}")
                
                if hypothesis.recommended_actions:
                    print(f"      Recommended Actions:")
                    for action in hypothesis.recommended_actions[:2]:
                        print(f"        • {action}")
        
        # Show statistics by threat type
        by_threat_type = {}
        for hypothesis in hypotheses:
            threat_type = hypothesis.threat_type.value
            by_threat_type[threat_type] = by_threat_type.get(threat_type, 0) + 1
        
        if by_threat_type:
            print(f"\n📊 Hypotheses by Threat Type:")
            for threat_type, count in sorted(by_threat_type.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {threat_type.replace('_', ' ').title()}: {count}")
        
        print("\n✅ Threat hypothesis generation completed")
        return hypotheses
    
    async def _demo_automated_hunting(self):
        """Demonstrate automated threat hunting sessions."""
        print("\n🎯 AUTOMATED THREAT HUNTING DEMONSTRATION")
        print("-" * 55)
        
        print("🚀 Starting automated threat hunting sessions...")
        
        # Start different types of hunt sessions
        hunt_scenarios = [
            {
                "trigger": HuntTrigger.VULNERABILITY_DISCOVERED,
                "focus_areas": ["vulnerability_exploitation", "lateral_movement"],
                "time_window_hours": 6,
                "description": "Hunt triggered by critical vulnerability discovery"
            },
            {
                "trigger": HuntTrigger.IOC_DETECTED,
                "focus_areas": ["malware_activity", "command_control"],
                "time_window_hours": 12,
                "description": "Hunt triggered by IoC detection"
            },
            {
                "trigger": HuntTrigger.ANOMALY_DETECTED,
                "focus_areas": ["insider_threat", "data_exfiltration"],
                "time_window_hours": 24,
                "description": "Hunt triggered by behavioral anomalies"
            }
        ]
        
        for i, scenario in enumerate(hunt_scenarios, 1):
            print(f"\n🔍 Hunt Session {i}: {scenario['description']}")
            
            session_id = await ai_threat_hunter.start_hunt_session(
                trigger=scenario["trigger"],
                hunter_name="ai_demo_hunter",
                focus_areas=scenario["focus_areas"],
                time_window_hours=scenario["time_window_hours"]
            )
            
            self.demo_session_ids.append(session_id)
            
            print(f"   • Session ID: {session_id}")
            print(f"   • Trigger: {scenario['trigger'].value}")
            print(f"   • Focus Areas: {', '.join(scenario['focus_areas'])}")
            print(f"   • Time Window: {scenario['time_window_hours']} hours")
            
            # Process the session
            session = ai_threat_hunter.get_session(session_id)
            if session:
                await ai_threat_hunter._process_hunt_session(session)
                
                print(f"   • Data Points Analyzed: {session.data_points_analyzed}")
                print(f"   • Anomalies Found: {len(session.anomalies_found)}")
                print(f"   • Hypotheses Generated: {len(session.hypotheses_generated)}")
                print(f"   • Processing Time: {session.processing_time:.2f}s")
        
        # Show active sessions summary
        active_sessions = ai_threat_hunter.get_active_sessions()
        print(f"\n📊 Active Hunt Sessions: {len(active_sessions)}")
        
        for session in active_sessions:
            duration = session.get_duration()
            print(f"   • {session.session_id}: {duration:.0f}s elapsed, {len(session.hypotheses_generated)} hypotheses")
        
        print("\n✅ Automated hunting demonstration completed")
    
    async def _demo_threat_correlation(self):
        """Demonstrate threat intelligence correlation."""
        print("\n🔗 THREAT INTELLIGENCE CORRELATION DEMONSTRATION")
        print("-" * 60)
        
        print("🧩 Correlating threats across multiple data sources...")
        
        # Get recent threats from hunt sessions
        recent_threats = ai_threat_hunter.get_recent_threats(hours=24)
        
        # Get IoCs from threat intelligence
        threat_iocs = list(threat_intel_engine.ioc_cache.values())
        
        # Get vulnerabilities
        recent_vulns = [v for v in vulnerability_manager.vulnerabilities.values() 
                       if v.discovered_at > time.time() - 86400]
        
        print(f"\n📊 Correlation Analysis:")
        print(f"   • Recent Threat Hypotheses: {len(recent_threats)}")
        print(f"   • Active IoCs: {len(threat_iocs)}")
        print(f"   • Recent Vulnerabilities: {len(recent_vulns)}")
        
        # Correlation analysis
        correlations = []
        
        # Correlate threats with IoCs
        for threat in recent_threats:
            for ioc in threat.iocs:
                for cached_ioc in threat_iocs:
                    if ioc.value == cached_ioc.value and cached_ioc.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        correlations.append({
                            "type": "threat_ioc_correlation",
                            "threat": threat.title,
                            "ioc": ioc.value,
                            "threat_level": cached_ioc.threat_level.value,
                            "confidence": threat.confidence * ioc.confidence
                        })
        
        # Correlate threats with vulnerabilities
        for threat in recent_threats:
            if threat.threat_type in [ThreatType.APT, ThreatType.MALWARE]:
                for vuln in recent_vulns:
                    if vuln.severity in [VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL]:
                        correlations.append({
                            "type": "threat_vulnerability_correlation",
                            "threat": threat.title,
                            "vulnerability": vuln.title,
                            "severity": vuln.severity.value,
                            "confidence": threat.confidence * 0.8
                        })
        
        # Show correlations
        if correlations:
            print(f"\n🔗 Discovered {len(correlations)} threat correlations:")
            
            for i, correlation in enumerate(correlations[:5], 1):  # Show top 5
                print(f"\n   {i}. {correlation['type'].replace('_', ' ').title()}")
                print(f"      Threat: {correlation['threat']}")
                if 'ioc' in correlation:
                    print(f"      IoC: {correlation['ioc']} ({correlation['threat_level']})")
                if 'vulnerability' in correlation:
                    print(f"      Vulnerability: {correlation['vulnerability']} ({correlation['severity']})")
                print(f"      Correlation Confidence: {correlation['confidence']:.2f}")
        else:
            print("   • No high-confidence correlations found")
        
        # Risk assessment
        high_risk_indicators = 0
        if len([t for t in recent_threats if t.risk_score > 0.7]) > 0:
            high_risk_indicators += 1
        if len([i for i in threat_iocs if i.threat_level == ThreatLevel.CRITICAL]) > 0:
            high_risk_indicators += 1
        if len([v for v in recent_vulns if v.severity == VulnerabilitySeverity.CRITICAL]) > 0:
            high_risk_indicators += 1
        
        print(f"\n⚠️ Risk Assessment:")
        print(f"   • High-Risk Indicators: {high_risk_indicators}/3")
        
        if high_risk_indicators >= 2:
            print("   • Overall Risk Level: HIGH")
            print("   • Recommendation: Immediate investigation and response")
        elif high_risk_indicators >= 1:
            print("   • Overall Risk Level: MEDIUM")
            print("   • Recommendation: Enhanced monitoring and analysis")
        else:
            print("   • Overall Risk Level: LOW")
            print("   • Recommendation: Continue normal monitoring")
        
        print("\n✅ Threat correlation analysis completed")
    
    async def _demo_reporting_analytics(self):
        """Demonstrate reporting and analytics capabilities."""
        print("\n📈 REPORTING & ANALYTICS DEMONSTRATION")
        print("-" * 50)
        
        print("📊 Generating comprehensive threat hunting reports...")
        
        # Get system statistics
        stats = ai_threat_hunter.get_system_statistics()
        
        print(f"\n📋 System Overview:")
        print(f"   • Active Hunt Sessions: {stats['active_sessions']}")
        print(f"   • Historical Hunts: {stats['historical_hunts']}")
        print(f"   • Anomaly Detectors: {stats['anomaly_detectors']}")
        print(f"   • Data Points Buffered: {stats['data_points_buffered']}")
        
        print(f"\n🎯 Hunt Performance:")
        print(f"   • Hunts Executed: {stats['hunts_executed']}")
        print(f"   • Threats Detected: {stats['threats_detected']}")
        print(f"   • Anomalies Found: {stats['anomalies_found']}")
        print(f"   • Hypotheses Generated: {stats['hypotheses_generated']}")
        
        # Analysis of recent activity
        recent_threats = ai_threat_hunter.get_recent_threats(hours=24)
        
        if recent_threats:
            print(f"\n🔍 Recent Threat Analysis (24h):")
            
            # By threat type
            by_type = {}
            for threat in recent_threats:
                threat_type = threat.threat_type.value
                by_type[threat_type] = by_type.get(threat_type, 0) + 1
            
            print(f"   • Threat Types:")
            for threat_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {threat_type.replace('_', ' ').title()}: {count}")
            
            # By confidence level
            high_confidence = len([t for t in recent_threats if t.confidence > 0.7])
            medium_confidence = len([t for t in recent_threats if 0.4 <= t.confidence <= 0.7])
            low_confidence = len([t for t in recent_threats if t.confidence < 0.4])
            
            print(f"   • Confidence Distribution:")
            print(f"     - High (>0.7): {high_confidence}")
            print(f"     - Medium (0.4-0.7): {medium_confidence}")
            print(f"     - Low (<0.4): {low_confidence}")
            
            # Top threats by risk score
            top_threats = sorted(recent_threats, key=lambda t: t.risk_score, reverse=True)[:3]
            print(f"   • Top Risk Threats:")
            for i, threat in enumerate(top_threats, 1):
                print(f"     {i}. {threat.title} (Risk: {threat.risk_score:.3f})")
        
        # Hunt session analysis
        active_sessions = ai_threat_hunter.get_active_sessions()
        if active_sessions:
            print(f"\n⏱️ Active Session Analysis:")
            
            total_data_points = sum(s.data_points_analyzed for s in active_sessions)
            total_anomalies = sum(len(s.anomalies_found) for s in active_sessions)
            total_hypotheses = sum(len(s.hypotheses_generated) for s in active_sessions)
            
            print(f"   • Total Data Points Analyzed: {total_data_points}")
            print(f"   • Total Anomalies Detected: {total_anomalies}")
            print(f"   • Total Hypotheses Generated: {total_hypotheses}")
            
            if total_data_points > 0:
                anomaly_rate = (total_anomalies / total_data_points) * 100
                print(f"   • Anomaly Detection Rate: {anomaly_rate:.2f}%")
        
        # Recommendations
        print(f"\n💡 AI Recommendations:")
        
        if stats['threats_detected'] > 5:
            print("   • High threat activity detected - consider increasing hunt frequency")
        
        if stats['anomalies_found'] > stats['threats_detected'] * 3:
            print("   • Many anomalies with few threats - consider tuning detection thresholds")
        
        if len(active_sessions) > 3:
            print("   • Multiple active hunts - ensure adequate analyst capacity")
        
        print("   • Continue monitoring for emerging threat patterns")
        print("   • Regularly update threat intelligence feeds")
        print("   • Review and adjust anomaly detection parameters")
        
        print("\n✅ Reporting and analytics completed")
    
    async def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n📊 COMPREHENSIVE AI THREAT HUNTING SUMMARY")
        print("=" * 55)
        
        stats = ai_threat_hunter.get_system_statistics()
        recent_threats = ai_threat_hunter.get_recent_threats(hours=24)
        
        print(f"🎯 AI Threat Hunting System Summary:")
        print(f"   • Hunt Sessions Executed: {stats['hunts_executed']}")
        print(f"   • Data Points Analyzed: {stats['data_points_buffered']}")
        print(f"   • Anomalies Detected: {stats['anomalies_found']}")
        print(f"   • Threat Hypotheses Generated: {stats['hypotheses_generated']}")
        print(f"   • High-Confidence Threats: {len([t for t in recent_threats if t.confidence > 0.7])}")
        
        print(f"\n🤖 AI Capabilities Demonstrated:")
        print(f"   • ✅ Intelligent behavioral anomaly detection")
        print(f"   • ✅ Statistical anomaly analysis with Z-score and IQR")
        print(f"   • ✅ AI-powered threat hypothesis generation")
        print(f"   • ✅ Automated hunt session management")
        print(f"   • ✅ Multi-source threat intelligence correlation")
        print(f"   • ✅ Real-time risk assessment and scoring")
        print(f"   • ✅ Comprehensive reporting and analytics")
        
        print(f"\n📈 Business Impact:")
        print(f"   • Reduced false positive rate through AI filtering")
        print(f"   • Accelerated threat detection and response")
        print(f"   • Automated hypothesis generation saves analyst time")
        print(f"   • Proactive threat hunting based on intelligence")
        print(f"   • Enhanced visibility into complex attack patterns")
        
        print(f"\n🔮 Advanced AI Features:")
        print(f"   • Machine learning-based anomaly detection")
        print(f"   • Pattern recognition and behavioral modeling")
        print(f"   • Automated threat actor attribution")
        print(f"   • Predictive threat intelligence analysis")
        print(f"   • Natural language hypothesis generation")
        
        print(f"\n✅ AI THREAT HUNTING DEMONSTRATION COMPLETE")
        print(f"All advanced AI-powered threat hunting capabilities successfully demonstrated!")
    
    async def _cleanup_systems(self):
        """Clean up the demonstration systems."""
        print("\n🧹 Cleaning up demonstration...")
        
        # Complete active hunt sessions
        for session_id in self.demo_session_ids:
            session = ai_threat_hunter.active_sessions.get(session_id)
            if session:
                await ai_threat_hunter._complete_hunt_session(session)
                ai_threat_hunter.active_sessions.pop(session_id, None)
        
        ai_threat_hunter.running = False
        print("✅ Cleanup completed")


async def main():
    """Main demonstration function."""
    demo = AIThreatHuntingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())