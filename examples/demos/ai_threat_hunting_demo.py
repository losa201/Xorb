#!/usr/bin/env python3
"""
XORB 2.0 AI-Powered Threat Hunting Demo
Demonstrates machine learning-enhanced threat detection and hunting capabilities
"""

import asyncio
import random

# Import XORB components
import sys
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, "/root/Xorb")

from xorb_core.hunting.ai_threat_hunter import AIThreatHunter


class AIThreatHuntingDemo:
    """Comprehensive AI-powered threat hunting demonstration"""

    def __init__(self):
        self.hunter = AIThreatHunter()
        self.demo_sessions = []

    async def setup_demo_environment(self):
        """Set up demo environment with simulated network data"""
        print("🔧 Setting up AI Threat Hunting Demo Environment...")

        # Generate simulated network telemetry
        self.network_data = self.generate_network_telemetry()
        self.process_data = self.generate_process_telemetry()
        self.authentication_data = self.generate_auth_telemetry()

        print(f"✅ Generated {len(self.network_data)} network events")
        print(f"✅ Generated {len(self.process_data)} process events")
        print(f"✅ Generated {len(self.authentication_data)} authentication events")

    def generate_network_telemetry(self) -> list[dict[str, Any]]:
        """Generate realistic network telemetry data"""
        events = []
        base_time = datetime.now() - timedelta(hours=24)

        # Normal traffic patterns
        for i in range(1000):
            event = {
                "timestamp": base_time + timedelta(minutes=i * 1.44),
                "source_ip": f"192.168.1.{random.randint(10, 250)}",
                "dest_ip": f"10.0.0.{random.randint(1, 100)}",
                "source_port": random.randint(32768, 65535),
                "dest_port": random.choice([80, 443, 22, 3389, 53]),
                "protocol": random.choice(["TCP", "UDP"]),
                "bytes_sent": random.randint(100, 10000),
                "bytes_received": random.randint(50, 5000),
                "connection_state": random.choice(
                    ["ESTABLISHED", "CLOSED", "SYN_SENT"]
                ),
            }
            events.append(event)

        # Inject anomalous patterns
        # 1. Data exfiltration pattern
        for i in range(5):
            event = {
                "timestamp": base_time + timedelta(hours=random.randint(1, 20)),
                "source_ip": "192.168.1.42",
                "dest_ip": "203.0.113.66",  # External IP
                "source_port": random.randint(32768, 65535),
                "dest_port": 443,
                "protocol": "TCP",
                "bytes_sent": random.randint(50000000, 100000000),  # Large outbound
                "bytes_received": random.randint(1000, 5000),
                "connection_state": "ESTABLISHED",
                "suspicious": True,
            }
            events.append(event)

        # 2. Port scanning pattern
        scanner_ip = "192.168.1.199"
        for port in range(1, 1025):
            event = {
                "timestamp": base_time + timedelta(hours=12, seconds=port),
                "source_ip": scanner_ip,
                "dest_ip": "192.168.1.100",
                "source_port": random.randint(32768, 65535),
                "dest_port": port,
                "protocol": "TCP",
                "bytes_sent": 60,
                "bytes_received": 0,
                "connection_state": "SYN_SENT",
                "suspicious": True,
            }
            events.append(event)

        return events

    def generate_process_telemetry(self) -> list[dict[str, Any]]:
        """Generate realistic process execution data"""
        events = []
        base_time = datetime.now() - timedelta(hours=24)

        # Normal processes
        normal_processes = [
            "chrome.exe",
            "firefox.exe",
            "notepad.exe",
            "explorer.exe",
            "winword.exe",
            "excel.exe",
            "outlook.exe",
            "teams.exe",
        ]

        for i in range(500):
            event = {
                "timestamp": base_time + timedelta(minutes=i * 2.88),
                "hostname": f"workstation-{random.randint(1, 50):02d}",
                "process_name": random.choice(normal_processes),
                "process_id": random.randint(1000, 9999),
                "parent_process": "explorer.exe",
                "command_line": "",
                "user": f"user{random.randint(1, 20)}",
                "integrity_level": "Medium",
            }
            events.append(event)

        # Inject suspicious processes
        suspicious_events = [
            {
                "timestamp": base_time + timedelta(hours=8),
                "hostname": "workstation-23",
                "process_name": "powershell.exe",
                "process_id": 3456,
                "parent_process": "winword.exe",
                "command_line": "-WindowStyle Hidden -EncodedCommand SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0AA==",
                "user": "user15",
                "integrity_level": "High",
                "suspicious": True,
            },
            {
                "timestamp": base_time + timedelta(hours=15),
                "hostname": "workstation-05",
                "process_name": "cmd.exe",
                "process_id": 7891,
                "parent_process": "svchost.exe",
                "command_line": "whoami /all && net user administrator /active:yes",
                "user": "SYSTEM",
                "integrity_level": "System",
                "suspicious": True,
            },
        ]

        events.extend(suspicious_events)
        return events

    def generate_auth_telemetry(self) -> list[dict[str, Any]]:
        """Generate authentication event data"""
        events = []
        base_time = datetime.now() - timedelta(hours=24)

        # Normal authentication patterns
        users = [f"user{i:02d}" for i in range(1, 21)]

        for i in range(200):
            event = {
                "timestamp": base_time + timedelta(minutes=i * 7.2),
                "event_type": "logon",
                "username": random.choice(users),
                "source_ip": f"192.168.1.{random.randint(10, 250)}",
                "logon_type": random.choice(
                    ["Interactive", "Network", "RemoteInteractive"]
                ),
                "authentication_package": "NTLM",
                "result": "Success",
            }
            events.append(event)

        # Inject credential stuffing attack
        for i in range(50):
            event = {
                "timestamp": base_time + timedelta(hours=3, seconds=i * 30),
                "event_type": "logon",
                "username": random.choice(users),
                "source_ip": "203.0.113.45",
                "logon_type": "Network",
                "authentication_package": "NTLM",
                "result": "Failure",
                "failure_reason": "Bad username or password",
                "suspicious": True,
            }
            events.append(event)

        return events

    async def demonstrate_anomaly_detection(self):
        """Demonstrate AI-powered anomaly detection"""
        print("\n🤖 Demonstrating AI-Powered Anomaly Detection...")

        # Behavioral anomaly detection
        print("\n🧠 Running Behavioral Analysis...")
        behavioral_detector = BehavioralAnomalyDetector()

        # Feed network data to behavioral detector
        for event in self.network_data:
            await behavioral_detector.process_event(event)

        behavioral_anomalies = await behavioral_detector.detect_anomalies()

        print(f"   🎯 Detected {len(behavioral_anomalies)} behavioral anomalies:")
        for anomaly in behavioral_anomalies[:3]:  # Show top 3
            print(f"      • {anomaly.get('description', 'Unknown anomaly')}")
            print(f"        Risk Score: {anomaly.get('risk_score', 0)}/100")
            print(f"        Confidence: {anomaly.get('confidence', 0) * 100:.1f}%")

        # Statistical anomaly detection
        print("\n📊 Running Statistical Analysis...")
        statistical_detector = StatisticalAnomalyDetector()

        for event in self.process_data:
            await statistical_detector.process_event(event)

        statistical_anomalies = await statistical_detector.detect_anomalies()

        print(f"   📈 Detected {len(statistical_anomalies)} statistical anomalies:")
        for anomaly in statistical_anomalies[:3]:
            print(f"      • {anomaly.get('description', 'Statistical anomaly')}")
            print(
                f"        Deviation: {anomaly.get('deviation', 0):.2f} standard deviations"
            )
            print(f"        P-value: {anomaly.get('p_value', 0):.6f}")

    async def demonstrate_hypothesis_generation(self):
        """Demonstrate AI hypothesis generation"""
        print("\n🔍 Demonstrating AI Hypothesis Generation...")

        # Generate hunting hypotheses based on detected patterns
        hypotheses = await self.hunter.generate_hypotheses(
            data_sources=["network", "process", "authentication"],
            time_window=24 * 60 * 60,  # 24 hours
        )

        print(f"\n💡 Generated {len(hypotheses)} hunting hypotheses:")

        for i, hypothesis in enumerate(hypotheses[:5], 1):
            print(f"\n   {i}. {hypothesis.title}")
            print(f"      Description: {hypothesis.description}")
            print(f"      MITRE ATT&CK: {', '.join(hypothesis.mitre_techniques)}")
            print(f"      Priority: {hypothesis.priority.value}")
            print(f"      Confidence: {hypothesis.confidence * 100:.1f}%")
            print(f"      IOCs: {len(hypothesis.indicators)} indicators")

    async def demonstrate_active_hunting(self):
        """Demonstrate active threat hunting session"""
        print("\n🎯 Demonstrating Active Threat Hunting Session...")

        # Start a hunting session for APT activity
        session = await self.hunter.start_hunt_session(
            hypothesis_title="Potential APT Lateral Movement",
            description="Investigating suspicious process execution and network patterns",
            analyst="SOC Analyst",
            priority="High",
        )

        print(f"\n🔬 Started hunting session: {session.session_id}")
        print(f"   Hypothesis: {session.hypothesis.title}")
        print(f"   Priority: {session.priority}")
        print(f"   Analyst: {session.analyst}")

        # Simulate hunt progression
        print("\n📋 Hunt Progress:")
        hunt_steps = [
            "Analyzing process execution patterns...",
            "Correlating network connections...",
            "Examining authentication events...",
            "Searching for IOCs in threat intelligence...",
            "Building attack timeline...",
            "Generating threat report...",
        ]

        for i, step in enumerate(hunt_steps, 1):
            print(f"   {i}/6 {step}")
            await asyncio.sleep(1.5)

            # Update session progress
            session.progress = i / len(hunt_steps)

        # Generate findings
        findings = await self.hunter.generate_hunt_findings(session.session_id)

        print("\n📊 Hunt Results:")
        print(f"   Status: {session.status}")
        print(f"   Findings: {len(findings)} threats identified")
        print(f"   Confidence: {session.confidence * 100:.1f}%")

        for finding in findings[:3]:
            print(f"\n   🚨 {finding.get('title', 'Threat Detection')}")
            print(f"      Severity: {finding.get('severity', 'Unknown')}")
            print(f"      Description: {finding.get('description', 'No description')}")
            print(
                f"      Affected Assets: {len(finding.get('affected_assets', []))} hosts"
            )

    async def demonstrate_threat_correlation(self):
        """Demonstrate threat intelligence correlation"""
        print("\n🕵️ Demonstrating Threat Intelligence Correlation...")

        # Simulate IOC correlation
        iocs = [
            {"type": "ip", "value": "203.0.113.66", "source": "AlienVault OTX"},
            {
                "type": "hash",
                "value": "d41d8cd98f00b204e9800998ecf8427e",
                "source": "VirusTotal",
            },
            {
                "type": "domain",
                "value": "malicious-domain.evil",
                "source": "Emerging Threats",
            },
        ]

        print("🔍 Correlating with threat intelligence feeds...")

        for ioc in iocs:
            print(f"\n   📍 Analyzing {ioc['type'].upper()}: {ioc['value']}")

            # Simulate threat intel lookup
            await asyncio.sleep(1)

            # Mock threat intelligence results
            threat_data = {
                "reputation": random.choice(["Malicious", "Suspicious", "Clean"]),
                "first_seen": (
                    datetime.now() - timedelta(days=random.randint(1, 30))
                ).isoformat(),
                "threat_types": random.sample(
                    ["Malware", "C&C", "Phishing", "Botnet"], 2
                ),
                "confidence": random.uniform(0.7, 0.95),
            }

            print(f"      Reputation: {threat_data['reputation']}")
            print(f"      First Seen: {threat_data['first_seen'][:10]}")
            print(f"      Threat Types: {', '.join(threat_data['threat_types'])}")
            print(f"      Confidence: {threat_data['confidence'] * 100:.1f}%")

    async def demonstrate_automated_response(self):
        """Demonstrate automated threat response"""
        print("\n⚡ Demonstrating Automated Threat Response...")

        # Simulate high-confidence threat detection
        threat = {
            "id": "THR-2024-001",
            "title": "APT29 Lateral Movement Detected",
            "severity": "Critical",
            "confidence": 0.92,
            "affected_assets": ["workstation-23", "server-01", "dc-01"],
            "attack_stages": ["Initial Access", "Persistence", "Lateral Movement"],
        }

        print(f"🚨 Critical Threat Detected: {threat['title']}")
        print(f"   Severity: {threat['severity']}")
        print(f"   Confidence: {threat['confidence'] * 100:.1f}%")
        print(f"   Affected Assets: {len(threat['affected_assets'])} systems")

        # Automated response actions
        response_actions = [
            "Isolating affected workstations from network...",
            "Blocking malicious IP addresses at firewall...",
            "Disabling compromised user accounts...",
            "Collecting forensic evidence...",
            "Notifying incident response team...",
            "Generating executive summary report...",
        ]

        print("\n🤖 Executing Automated Response:")
        for i, action in enumerate(response_actions, 1):
            print(f"   {i}/6 {action}")
            await asyncio.sleep(1)

        print("\n✅ Automated response completed successfully!")

    async def run_complete_demo(self):
        """Run the complete AI threat hunting demonstration"""
        print("🚀 Starting XORB AI-Powered Threat Hunting Demo")
        print("=" * 60)

        try:
            # Setup
            await self.setup_demo_environment()

            # Run demonstration scenarios
            await self.demonstrate_anomaly_detection()
            await self.demonstrate_hypothesis_generation()
            await self.demonstrate_active_hunting()
            await self.demonstrate_threat_correlation()
            await self.demonstrate_automated_response()

            print("\n🎉 AI Threat Hunting Demo Complete!")
            print("=" * 60)
            print("✅ Demonstrated capabilities:")
            print("   • Machine learning anomaly detection")
            print("   • AI-powered hypothesis generation")
            print("   • Active threat hunting workflows")
            print("   • Threat intelligence correlation")
            print("   • MITRE ATT&CK technique mapping")
            print("   • Automated threat response")
            print("   • Executive reporting and metrics")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    demo = AIThreatHuntingDemo()
    asyncio.run(demo.run_complete_demo())
