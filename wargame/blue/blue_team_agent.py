#!/usr/bin/env python3
"""
Blue Team AI Defense Agent
Simulates advanced defense capabilities with detection and response
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

@dataclass 
class DefenseAction:
    timestamp: str
    category: str  # detection, prevention, response, deception, hunting
    technique: str
    target: str
    action_taken: str
    effectiveness: str  # high, medium, low
    artifacts: List[str] = None
    cost: int = 0  # resource cost (1-10 scale)

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

@dataclass
class ThreatDetection:
    timestamp: str
    detection_type: str
    confidence: float
    source: str
    indicator: str
    threat_description: str
    severity: str
    false_positive_risk: float

class BlueTeamAgent:
    def __init__(self, environment_state_path: str):
        self.environment_state_path = environment_state_path
        self.defense_actions: List[DefenseAction] = []
        self.detected_threats: List[ThreatDetection] = []
        self.deployed_countermeasures = set()
        self.monitoring_coverage = {
            "network": 0.7,
            "endpoints": 0.6,
            "applications": 0.8,
            "cloud": 0.5
        }
        self.threat_intelligence = []
        self.honeypots_deployed = []
        
    def load_environment_state(self) -> Dict:
        """Load current Purple environment state"""
        with open(self.environment_state_path, 'r') as f:
            return json.load(f)
    
    def load_red_team_actions(self, round_id: int) -> Optional[Dict]:
        """Load Red Team actions from previous round"""
        try:
            with open(f"/root/Xorb/wargame/reports/red/attacks_round_{round_id}.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def log_defense_action(self, action: DefenseAction):
        """Log defense action for reporting"""
        self.defense_actions.append(action)
        print(f"[BLUE] {action.timestamp} - {action.category}: {action.technique} -> {action.target} ({action.effectiveness})")
    
    def analyze_threat_indicators(self, red_actions: Dict) -> List[ThreatDetection]:
        """Analyze Red Team actions for detectable indicators"""
        detections = []
        
        if not red_actions:
            return detections
        
        for action_data in red_actions.get('actions', []):
            # Network-based detections
            if action_data['technique'].startswith('T1590') or 'nmap' in action_data.get('payload', ''):
                detection = ThreatDetection(
                    timestamp=datetime.now().isoformat(),
                    detection_type="network_scan",
                    confidence=0.85,
                    source="Network IDS",
                    indicator=f"Multiple port scans from external IP targeting {action_data['target']}",
                    threat_description="Potential reconnaissance activity detected",
                    severity="medium",
                    false_positive_risk=0.15
                )
                detections.append(detection)
                self.detected_threats.append(detection)
            
            # Web application attack detection
            if action_data['technique'] == 'T1190 - Exploit Public-Facing Application':
                detection = ThreatDetection(
                    timestamp=datetime.now().isoformat(),
                    detection_type="web_attack",
                    confidence=0.92,
                    source="Web Application Firewall",
                    indicator=f"Suspicious POST request to {action_data['target']}",
                    threat_description="Potential web application exploit attempt",
                    severity="high",
                    false_positive_risk=0.08
                )
                detections.append(detection)
                self.detected_threats.append(detection)
            
            # Credential-based detection
            if action_data['technique'] == 'T1078.001 - Default Accounts':
                # This would likely be detected after successful login
                detection = ThreatDetection(
                    timestamp=datetime.now().isoformat(),
                    detection_type="suspicious_login",
                    confidence=0.75,
                    source="Application Logs",
                    indicator=f"Admin login from unusual location to {action_data['target']}",
                    threat_description="Potentially compromised administrative account",
                    severity="high",
                    false_positive_risk=0.25
                )
                detections.append(detection)
                self.detected_threats.append(detection)
            
            # Data exfiltration detection
            if action_data['technique'] == 'T1041 - Exfiltration Over C2 Channel':
                detection = ThreatDetection(
                    timestamp=datetime.now().isoformat(),
                    detection_type="data_exfiltration",
                    confidence=0.88,
                    source="DLP System",
                    indicator="Large database export outside business hours",
                    threat_description="Potential data exfiltration detected",
                    severity="critical",
                    false_positive_risk=0.12
                )
                detections.append(detection)
                self.detected_threats.append(detection)
        
        return detections
    
    def deploy_preventive_measures(self, env_state: Dict) -> List[DefenseAction]:
        """Deploy preventive security measures"""
        preventive_actions = []
        
        # Patch vulnerable WordPress plugin
        vuln_004 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-004'), None)
        if vuln_004:
            action = DefenseAction(
                timestamp=datetime.now().isoformat(),
                category="prevention",
                technique="Vulnerability Patching",
                target="Corporate Website - Contact Form 7 Plugin",
                action_taken="Updated plugin to version 5.6.4 (patches CVE-2023-6000)",
                effectiveness="high",
                artifacts=["patch_deployment.log", "plugin_update_confirmation.txt"],
                cost=3
            )
            preventive_actions.append(action)
            self.log_defense_action(action)
            self.deployed_countermeasures.add("wordpress_plugin_patch")
        
        # Secure S3 bucket configuration
        vuln_003 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-003'), None)
        if vuln_003:
            action = DefenseAction(
                timestamp=datetime.now().isoformat(),
                category="prevention",
                technique="Access Control Remediation",
                target=f"S3 Bucket: {vuln_003['bucket']}",
                action_taken="Removed public read access, implemented IAM policies",
                effectiveness="high",
                artifacts=["s3_policy_update.json", "access_audit.log"],
                cost=2
            )
            preventive_actions.append(action)
            self.log_defense_action(action)
            self.deployed_countermeasures.add("s3_access_control")
        
        # Disable debug endpoint
        vuln_002 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-002'), None)
        if vuln_002:
            action = DefenseAction(
                timestamp=datetime.now().isoformat(),
                category="prevention",
                technique="Attack Surface Reduction",
                target="File Storage API",
                action_taken="Disabled /debug endpoint in production configuration",
                effectiveness="high",
                artifacts=["api_config_change.log"],
                cost=1
            )
            preventive_actions.append(action)
            self.log_defense_action(action)
            self.deployed_countermeasures.add("api_endpoint_hardening")
        
        return preventive_actions
    
    def deploy_detection_systems(self, env_state: Dict) -> List[DefenseAction]:
        """Deploy enhanced detection capabilities"""
        detection_actions = []
        
        # Deploy network monitoring
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="detection",
            technique="Network Traffic Analysis",
            target="DMZ and Internal Networks",
            action_taken="Deployed Suricata IDS with custom rules for reconnaissance detection",
            effectiveness="high",
            artifacts=["suricata_rules.txt", "network_baseline.json"],
            cost=5
        )
        detection_actions.append(action)
        self.log_defense_action(action)
        self.monitoring_coverage["network"] = 0.9
        
        # Enhanced web application monitoring
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="detection",
            technique="Web Application Firewall",
            target="All Public Web Applications",
            action_taken="Deployed ModSecurity with OWASP Core Rule Set",
            effectiveness="high",
            artifacts=["waf_rules.conf", "baseline_traffic.log"],
            cost=4
        )
        detection_actions.append(action)
        self.log_defense_action(action)
        self.monitoring_coverage["applications"] = 0.95
        
        # Database activity monitoring
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="detection",
            technique="Database Activity Monitoring",
            target="PostgreSQL Database",
            action_taken="Implemented pgAudit for comprehensive database logging",
            effectiveness="medium",
            artifacts=["pgaudit_config.conf", "db_baseline_queries.log"],
            cost=3
        )
        detection_actions.append(action)
        self.log_defense_action(action)
        
        return detection_actions
    
    def deploy_deception_technology(self, env_state: Dict) -> List[DefenseAction]:
        """Deploy honeypots and deception mechanisms"""
        deception_actions = []
        
        # Deploy web honeypot
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="deception",
            technique="Honeypot Deployment",
            target="DMZ Network",
            action_taken="Deployed Cowrie SSH honeypot on 192.168.1.99",
            effectiveness="medium",
            artifacts=["honeypot_config.json", "cowrie_setup.log"],
            cost=2
        )
        deception_actions.append(action)
        self.log_defense_action(action)
        self.honeypots_deployed.append("ssh_honeypot")
        
        # Deploy fake admin portal
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="deception",
            technique="Decoy Service",
            target="Internal Network",
            action_taken="Created fake admin portal at admin-backup.meridiandynamics.com",
            effectiveness="high",
            artifacts=["decoy_portal_logs.txt", "fake_credentials.json"],
            cost=3
        )
        deception_actions.append(action)
        self.log_defense_action(action)
        self.honeypots_deployed.append("admin_decoy")
        
        # Deploy canary tokens in S3
        action = DefenseAction(
            timestamp=datetime.now().isoformat(),
            category="deception",
            technique="Canary Tokens",
            target="AWS S3 Bucket",
            action_taken="Planted fake credentials and documents with embedded tracking",
            effectiveness="high",
            artifacts=["canary_tokens.json", "tracking_urls.txt"],
            cost=2
        )
        deception_actions.append(action)
        self.log_defense_action(action)
        self.honeypots_deployed.append("s3_canaries")
        
        return deception_actions
    
    def incident_response_actions(self, detections: List[ThreatDetection]) -> List[DefenseAction]:
        """Execute incident response based on detections"""
        response_actions = []
        
        for detection in detections:
            if detection.severity == "critical" and detection.confidence > 0.8:
                # Immediate containment for high-confidence critical threats
                action = DefenseAction(
                    timestamp=datetime.now().isoformat(),
                    category="response",
                    technique="Network Isolation", 
                    target=detection.source,
                    action_taken="Isolated affected systems and blocked suspicious IPs",
                    effectiveness="high",
                    artifacts=["isolation_commands.log", "blocked_ips.txt"],
                    cost=7
                )
                response_actions.append(action)
                self.log_defense_action(action)
            
            elif detection.severity == "high":
                # Enhanced monitoring for high severity threats
                action = DefenseAction(
                    timestamp=datetime.now().isoformat(),
                    category="response",
                    technique="Enhanced Monitoring",
                    target=detection.source,
                    action_taken="Increased logging verbosity and deployed additional sensors",
                    effectiveness="medium",
                    artifacts=["enhanced_monitoring.conf", "additional_sensors.log"],
                    cost=4
                )
                response_actions.append(action)
                self.log_defense_action(action)
            
            elif detection.severity == "medium":
                # Threat hunting for medium severity
                action = DefenseAction(
                    timestamp=datetime.now().isoformat(),
                    category="hunting",
                    technique="Threat Hunting",
                    target="Environment-wide",
                    action_taken=f"Initiated hunting query for similar indicators: {detection.indicator}",
                    effectiveness="medium",
                    artifacts=["hunting_queries.sql", "threat_hunt_results.json"],
                    cost=3
                )
                response_actions.append(action)
                self.log_defense_action(action)
        
        return response_actions
    
    def threat_intelligence_update(self, red_actions: Dict) -> List[DefenseAction]:
        """Update threat intelligence based on observed attacks"""
        ti_actions = []
        
        if red_actions:
            # Extract IOCs from red actions
            iocs = []
            for action_data in red_actions.get('actions', []):
                if 'payload' in action_data and action_data['payload']:
                    ioc_hash = hashlib.md5(action_data['payload'].encode()).hexdigest()
                    iocs.append({
                        "type": "payload_hash",
                        "value": ioc_hash,
                        "technique": action_data['technique']
                    })
            
            action = DefenseAction(
                timestamp=datetime.now().isoformat(),
                category="intelligence",
                technique="IOC Extraction and Sharing",
                target="Threat Intelligence Platform",
                action_taken=f"Extracted {len(iocs)} IOCs and updated threat signatures",
                effectiveness="high",
                artifacts=["ioc_feed.json", "signature_updates.txt"],
                cost=2
            )
            ti_actions.append(action)
            self.log_defense_action(action)
            self.threat_intelligence.extend(iocs)
        
        return ti_actions
    
    def execute_defense_round(self, round_id: int) -> Dict[str, Any]:
        """Execute complete defense round"""
        print(f"\n[BLUE TEAM] Starting defense round {round_id} at {datetime.now().isoformat()}")
        
        env_state = self.load_environment_state()
        red_actions = self.load_red_team_actions(round_id)
        all_actions = []
        
        # Analyze threats from Red Team actions
        detections = self.analyze_threat_indicators(red_actions)
        print(f"[BLUE] Detected {len(detections)} potential threats")
        
        # Deploy preventive measures
        all_actions.extend(self.deploy_preventive_measures(env_state))
        
        # Enhance detection capabilities
        all_actions.extend(self.deploy_detection_systems(env_state))
        
        # Deploy deception technology
        all_actions.extend(self.deploy_deception_technology(env_state))
        
        # Execute incident response
        all_actions.extend(self.incident_response_actions(detections))
        
        # Update threat intelligence
        all_actions.extend(self.threat_intelligence_update(red_actions))
        
        # Calculate defense effectiveness
        total_cost = sum(action.cost for action in all_actions)
        high_effectiveness_count = len([a for a in all_actions if a.effectiveness == "high"])
        
        defense_report = {
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
            "total_actions": len(all_actions),
            "total_resource_cost": total_cost,
            "detections": [asdict(detection) for detection in detections],
            "deployed_countermeasures": list(self.deployed_countermeasures),
            "monitoring_coverage": self.monitoring_coverage,
            "honeypots_deployed": self.honeypots_deployed,
            "threat_intelligence_items": len(self.threat_intelligence),
            "actions": [asdict(action) for action in all_actions],
            "effectiveness_metrics": {
                "detection_rate": len(detections) / max(len(red_actions.get('actions', [])), 1) if red_actions else 0,
                "prevention_score": high_effectiveness_count / max(len(all_actions), 1),
                "cost_efficiency": high_effectiveness_count / max(total_cost, 1),
                "false_positive_rate": sum(d.false_positive_risk for d in detections) / max(len(detections), 1)
            }
        }
        
        return defense_report

if __name__ == "__main__":
    blue_agent = BlueTeamAgent("/root/Xorb/wargame/purple/environment_state.json")
    report = blue_agent.execute_defense_round(1)
    
    # Save defense report
    with open(f"/root/Xorb/wargame/reports/blue/defenses_round_{report['round_id']}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[BLUE TEAM] Defense round completed. Report saved.")
    print(f"Deployed countermeasures: {report['deployed_countermeasures']}")
    print(f"Detection rate: {report['effectiveness_metrics']['detection_rate']:.2%}")
    print(f"Total resource cost: {report['total_resource_cost']}")