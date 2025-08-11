#!/usr/bin/env python3
"""
Wargame Orchestrator
Manages the continuous Red vs Blue wargame with Purple Team synthetic environment
"""

import json
import random
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add the wargame directory to Python path
sys.path.append('/root/Xorb/wargame')

from red.red_team_agent import RedTeamAgent
from blue.blue_team_agent import BlueTeamAgent

@dataclass
class EnvironmentChange:
    timestamp: str
    change_type: str  # configuration, vulnerability, infrastructure, policy
    description: str
    impact: str  # low, medium, high
    affected_systems: List[str]

class WargameOrchestrator:
    def __init__(self):
        self.environment_state_path = "/root/Xorb/wargame/purple/environment_state.json"
        self.threat_model_path = "/root/Xorb/wargame/purple/threat_model.json"
        self.red_agent = RedTeamAgent(self.environment_state_path)
        self.blue_agent = BlueTeamAgent(self.environment_state_path)
        self.current_round = 0
        self.wargame_history = []
        self.environment_changes = []
        
    def load_environment_state(self) -> Dict:
        """Load current environment state"""
        with open(self.environment_state_path, 'r') as f:
            return json.load(f)
    
    def save_environment_state(self, state: Dict):
        """Save updated environment state"""
        with open(self.environment_state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def simulate_environment_changes(self, round_id: int) -> List[EnvironmentChange]:
        """Simulate realistic environment changes that happen in real organizations"""
        changes = []
        
        # Random probability of changes each round
        if random.random() < 0.3:  # 30% chance of infrastructure change
            change = EnvironmentChange(
                timestamp=datetime.now().isoformat(),
                change_type="infrastructure",
                description=f"IT deployed new application server in engineering VLAN",
                impact="medium",
                affected_systems=["engineering"]
            )
            changes.append(change)
        
        if random.random() < 0.2:  # 20% chance of new vulnerability
            vuln_types = ["unpatched_software", "misconfiguration", "weak_credentials", "exposed_service"]
            change = EnvironmentChange(
                timestamp=datetime.now().isoformat(),
                change_type="vulnerability",
                description=f"New {random.choice(vuln_types)} introduced in routine update",
                impact="high",
                affected_systems=["web_applications"]
            )
            changes.append(change)
        
        if random.random() < 0.4:  # 40% chance of configuration change
            change = EnvironmentChange(
                timestamp=datetime.now().isoformat(),
                change_type="configuration",
                description="Updated firewall rules to allow new business application",
                impact="medium",
                affected_systems=["network", "security"]
            )
            changes.append(change)
        
        if random.random() < 0.1:  # 10% chance of policy change
            change = EnvironmentChange(
                timestamp=datetime.now().isoformat(),
                change_type="policy",
                description="Updated password policy requiring 12+ character passwords",
                impact="low",
                affected_systems=["authentication"]
            )
            changes.append(change)
        
        return changes
    
    def apply_environment_changes(self, changes: List[EnvironmentChange]) -> Dict:
        """Apply environment changes to the Purple environment"""
        env_state = self.load_environment_state()
        
        for change in changes:
            if change.change_type == "vulnerability":
                # Add new vulnerability
                new_vuln = {
                    "id": f"VULN-{len(env_state['vulnerabilities']) + 1:03d}",
                    "type": "misconfiguration",
                    "location": "Finance Application",
                    "description": "Debug logging enabled exposing sensitive data",
                    "severity": "medium",
                    "discoverable": True
                }
                env_state['vulnerabilities'].append(new_vuln)
            
            elif change.change_type == "infrastructure":
                # Add new application
                new_app = {
                    "name": "Project Management Tool",
                    "url": "https://projects.meridiandynamics.com",
                    "type": "Node.js",
                    "version": "16.14.0",
                    "status": "internal",
                    "authentication": "LDAP",
                    "last_updated": datetime.now().strftime("%Y-%m-%d")
                }
                env_state['applications'].append(new_app)
        
        # Update simulation parameters
        env_state['simulation_parameters']['environment_changes'] += len(changes)
        env_state['simulation_parameters']['last_update'] = datetime.now().isoformat()
        
        self.save_environment_state(env_state)
        return env_state
    
    def update_purple_environment_post_round(self, red_report: Dict, blue_report: Dict) -> Dict:
        """Update Purple environment based on Red and Blue actions"""
        env_state = self.load_environment_state()
        
        # Track compromises
        if red_report.get('compromised_assets'):
            env_state['simulation_parameters']['successful_compromises'] += len(red_report['compromised_assets'])
        
        # Track detections
        if blue_report.get('detections'):
            env_state['simulation_parameters']['detected_attacks'] += len(blue_report['detections'])
        
        # Remove vulnerabilities that were patched by Blue Team
        patched_vulns = []
        for action in blue_report.get('actions', []):
            if action['category'] == 'prevention' and 'patch' in action['action_taken'].lower():
                if 'wordpress' in action['target'].lower():
                    patched_vulns.append('VULN-004')
            elif action['category'] == 'prevention' and 's3' in action['target'].lower():
                patched_vulns.append('VULN-003')
            elif action['category'] == 'prevention' and 'api' in action['target'].lower():
                patched_vulns.append('VULN-002')
        
        # Remove patched vulnerabilities
        env_state['vulnerabilities'] = [v for v in env_state['vulnerabilities'] 
                                       if v['id'] not in patched_vulns]
        
        # Update round counter
        env_state['simulation_parameters']['round'] += 1
        
        self.save_environment_state(env_state)
        return env_state
    
    def generate_round_summary(self, round_id: int, red_report: Dict, blue_report: Dict, 
                              env_changes: List[EnvironmentChange]) -> Dict:
        """Generate comprehensive round summary"""
        summary = {
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
            "duration": "5 minutes",  # Simulated round duration
            "red_team_performance": {
                "total_actions": red_report.get('total_actions', 0),
                "successful_actions": red_report.get('successful_actions', 0),
                "success_rate": red_report.get('successful_actions', 0) / max(red_report.get('total_actions', 1), 1),
                "compromised_assets": red_report.get('compromised_assets', []),
                "persistence_established": len(red_report.get('persistence_mechanisms', [])) > 0,
                "data_exfiltrated": any('exfiltration' in action.get('phase', '') for action in red_report.get('actions', []))
            },
            "blue_team_performance": {
                "total_actions": blue_report.get('total_actions', 0),
                "detections": len(blue_report.get('detections', [])),
                "detection_rate": blue_report.get('effectiveness_metrics', {}).get('detection_rate', 0),
                "countermeasures_deployed": len(blue_report.get('deployed_countermeasures', [])),
                "resource_cost": blue_report.get('total_resource_cost', 0),
                "prevention_score": blue_report.get('effectiveness_metrics', {}).get('prevention_score', 0)
            },
            "environment_evolution": {
                "changes_applied": len(env_changes),
                "new_vulnerabilities": len([c for c in env_changes if c.change_type == 'vulnerability']),
                "infrastructure_changes": len([c for c in env_changes if c.change_type == 'infrastructure']),
                "configuration_updates": len([c for c in env_changes if c.change_type == 'configuration'])
            },
            "security_posture": {
                "overall_risk": self._calculate_risk_score(red_report, blue_report),
                "attack_surface": self._calculate_attack_surface(),
                "defense_maturity": self._calculate_defense_maturity(blue_report)
            },
            "lessons_learned": self._extract_lessons_learned(red_report, blue_report),
            "next_round_recommendations": self._generate_recommendations(red_report, blue_report)
        }
        
        return summary
    
    def _calculate_risk_score(self, red_report: Dict, blue_report: Dict) -> str:
        """Calculate overall security risk score"""
        red_success_rate = red_report.get('successful_actions', 0) / max(red_report.get('total_actions', 1), 1)
        blue_detection_rate = blue_report.get('effectiveness_metrics', {}).get('detection_rate', 0)
        
        risk_score = (red_success_rate * 0.6) + ((1 - blue_detection_rate) * 0.4)
        
        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_attack_surface(self) -> str:
        """Calculate current attack surface"""
        env_state = self.load_environment_state()
        
        public_apps = len([app for app in env_state['applications'] if app['status'] in ['public', 'external']])
        vulnerabilities = len(env_state['vulnerabilities'])
        
        surface_score = (public_apps * 0.3) + (vulnerabilities * 0.7)
        
        if surface_score > 5:
            return "large"
        elif surface_score > 2:
            return "medium"
        else:
            return "small"
    
    def _calculate_defense_maturity(self, blue_report: Dict) -> str:
        """Calculate defense maturity level"""
        prevention_score = blue_report.get('effectiveness_metrics', {}).get('prevention_score', 0)
        detection_rate = blue_report.get('effectiveness_metrics', {}).get('detection_rate', 0)
        
        maturity_score = (prevention_score * 0.5) + (detection_rate * 0.5)
        
        if maturity_score > 0.8:
            return "advanced"
        elif maturity_score > 0.5:
            return "intermediate"
        else:
            return "basic"
    
    def _extract_lessons_learned(self, red_report: Dict, blue_report: Dict) -> List[str]:
        """Extract key lessons learned from the round"""
        lessons = []
        
        if red_report.get('successful_actions', 0) > red_report.get('total_actions', 1) * 0.7:
            lessons.append("High attack success rate indicates need for improved preventive controls")
        
        if blue_report.get('effectiveness_metrics', {}).get('detection_rate', 0) < 0.5:
            lessons.append("Low detection rate suggests gaps in monitoring coverage")
        
        if red_report.get('persistence_mechanisms'):
            lessons.append("Persistence established - need enhanced endpoint detection and response")
        
        if blue_report.get('effectiveness_metrics', {}).get('false_positive_rate', 0) > 0.3:
            lessons.append("High false positive rate may lead to alert fatigue")
        
        return lessons
    
    def _generate_recommendations(self, red_report: Dict, blue_report: Dict) -> List[str]:
        """Generate recommendations for next round"""
        recommendations = []
        
        compromised_assets = red_report.get('compromised_assets', [])
        if 'HR Portal' in compromised_assets:
            recommendations.append("Implement multi-factor authentication for administrative accounts")
        
        if 'AWS S3 Bucket' in compromised_assets:
            recommendations.append("Review and audit all cloud storage configurations")
        
        detection_rate = blue_report.get('effectiveness_metrics', {}).get('detection_rate', 0)
        if detection_rate < 0.6:
            recommendations.append("Deploy additional detection capabilities in blind spots")
        
        resource_cost = blue_report.get('total_resource_cost', 0)
        if resource_cost > 25:
            recommendations.append("Optimize defense spending for better cost-effectiveness")
        
        return recommendations
    
    def execute_wargame_round(self) -> Dict:
        """Execute a complete wargame round"""
        self.current_round += 1
        print(f"\n{'='*60}")
        print(f"WARGAME ROUND {self.current_round}")
        print(f"{'='*60}")
        
        # Phase 1: Environment changes (Purple Team evolution)
        print(f"\n[PURPLE] Simulating environment changes...")
        env_changes = self.simulate_environment_changes(self.current_round)
        if env_changes:
            self.apply_environment_changes(env_changes)
            print(f"[PURPLE] Applied {len(env_changes)} environment changes")
            for change in env_changes:
                print(f"  - {change.description}")
        else:
            print(f"[PURPLE] No environment changes this round")
        
        # Phase 2: Red Team attack
        print(f"\n[RED] Executing attack phase...")
        red_report = self.red_agent.execute_attack_round()
        
        # Save Red Team report
        os.makedirs("/root/Xorb/wargame/reports/red", exist_ok=True)
        with open(f"/root/Xorb/wargame/reports/red/attacks_round_{self.current_round}.json", 'w') as f:
            json.dump(red_report, f, indent=2)
        
        # Phase 3: Blue Team defense
        print(f"\n[BLUE] Executing defense phase...")
        blue_report = self.blue_agent.execute_defense_round(self.current_round)
        
        # Save Blue Team report
        os.makedirs("/root/Xorb/wargame/reports/blue", exist_ok=True)
        with open(f"/root/Xorb/wargame/reports/blue/defenses_round_{self.current_round}.json", 'w') as f:
            json.dump(blue_report, f, indent=2)
        
        # Phase 4: Purple Team assessment and environment update
        print(f"\n[PURPLE] Updating environment state...")
        updated_env = self.update_purple_environment_post_round(red_report, blue_report)
        
        # Phase 5: Generate round summary
        round_summary = self.generate_round_summary(self.current_round, red_report, blue_report, env_changes)
        
        # Save round summary
        os.makedirs("/root/Xorb/wargame/reports/purple", exist_ok=True)
        with open(f"/root/Xorb/wargame/reports/purple/round_{self.current_round}_summary.json", 'w') as f:
            json.dump(round_summary, f, indent=2)
        
        self.wargame_history.append(round_summary)
        
        # Print round results
        print(f"\n{'='*60}")
        print(f"ROUND {self.current_round} RESULTS")
        print(f"{'='*60}")
        print(f"Red Team: {red_report.get('successful_actions', 0)}/{red_report.get('total_actions', 0)} successful actions")
        print(f"Blue Team: {len(blue_report.get('detections', []))} detections, {len(blue_report.get('deployed_countermeasures', []))} countermeasures")
        print(f"Overall Risk: {round_summary['security_posture']['overall_risk'].upper()}")
        print(f"Defense Maturity: {round_summary['security_posture']['defense_maturity'].upper()}")
        
        if round_summary.get('lessons_learned'):
            print(f"\nKey Lessons:")
            for lesson in round_summary['lessons_learned']:
                print(f"  - {lesson}")
        
        if round_summary.get('next_round_recommendations'):
            print(f"\nRecommendations for Next Round:")
            for rec in round_summary['next_round_recommendations']:
                print(f"  - {rec}")
        
        return round_summary
    
    def run_continuous_wargame(self, max_rounds: int = 5, round_delay: int = 10):
        """Run continuous wargame for specified number of rounds"""
        print(f"Starting Continuous Red vs Blue Wargame")
        print(f"Target Organization: Meridian Dynamics Corp")
        print(f"Rounds: {max_rounds}, Delay: {round_delay}s between rounds")
        
        for round_num in range(max_rounds):
            try:
                round_summary = self.execute_wargame_round()
                
                if round_num < max_rounds - 1:
                    print(f"\nWaiting {round_delay} seconds before next round...")
                    time.sleep(round_delay)
                    
            except KeyboardInterrupt:
                print(f"\nWargame interrupted by user after {self.current_round} rounds")
                break
            except Exception as e:
                print(f"Error in round {self.current_round}: {e}")
                continue
        
        # Generate final summary
        self.generate_final_wargame_report()
    
    def generate_final_wargame_report(self):
        """Generate comprehensive final report"""
        final_report = {
            "wargame_summary": {
                "total_rounds": self.current_round,
                "duration": f"{self.current_round * 5} minutes (simulated)",
                "final_timestamp": datetime.now().isoformat()
            },
            "overall_metrics": {
                "total_red_actions": sum(r['red_team_performance']['total_actions'] for r in self.wargame_history),
                "total_successful_attacks": sum(r['red_team_performance']['successful_actions'] for r in self.wargame_history),
                "total_blue_actions": sum(r['blue_team_performance']['total_actions'] for r in self.wargame_history),
                "total_detections": sum(r['blue_team_performance']['detections'] for r in self.wargame_history),
                "environment_changes": sum(r['environment_evolution']['changes_applied'] for r in self.wargame_history)
            },
            "trend_analysis": {
                "attack_effectiveness_trend": [r['red_team_performance']['success_rate'] for r in self.wargame_history],
                "detection_rate_trend": [r['blue_team_performance']['detection_rate'] for r in self.wargame_history],
                "risk_level_trend": [r['security_posture']['overall_risk'] for r in self.wargame_history]
            },
            "final_environment_state": self.load_environment_state(),
            "round_summaries": self.wargame_history
        }
        
        with open("/root/Xorb/wargame/reports/purple/final_wargame_report.json", 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"FINAL WARGAME REPORT")
        print(f"{'='*60}")
        print(f"Total Rounds: {final_report['wargame_summary']['total_rounds']}")
        print(f"Total Red Actions: {final_report['overall_metrics']['total_red_actions']}")
        print(f"Total Blue Actions: {final_report['overall_metrics']['total_blue_actions']}")
        print(f"Total Detections: {final_report['overall_metrics']['total_detections']}")
        print(f"Environment Changes: {final_report['overall_metrics']['environment_changes']}")
        print(f"\nFinal report saved to: /root/Xorb/wargame/reports/purple/final_wargame_report.json")

if __name__ == "__main__":
    orchestrator = WargameOrchestrator()
    
    # Run the wargame
    try:
        orchestrator.run_continuous_wargame(max_rounds=3, round_delay=5)
    except Exception as e:
        print(f"Wargame error: {e}")
        orchestrator.generate_final_wargame_report()