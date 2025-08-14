#!/usr/bin/env python3
"""
Red Team Campaign Manager
Orchestrates complex multi-stage attack campaigns with adaptive planning
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class CampaignPhase(Enum):
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

class CampaignStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class CampaignObjective:
    id: str
    name: str
    description: str
    priority: int  # 1-10, 10 being highest
    mitre_tactics: List[str]
    success_criteria: Dict[str, Any]
    timeout_hours: int
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class AttackVector:
    id: str
    name: str
    technique: str
    mitre_id: str
    target_systems: List[str]
    prerequisites: List[str]
    payloads: List[str]
    c2_profiles: List[str]
    stealth_level: int  # 1-10, 10 being most stealthy
    success_probability: float
    detection_risk: float

@dataclass
class CampaignExecution:
    phase: CampaignPhase
    attack_vector: AttackVector
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    results: Dict[str, Any]
    artifacts: List[str]
    iocs_generated: List[str]

class OffensiveCampaignManager:
    def __init__(self, config_path: str = "campaign_config.json"):
        self.config_path = config_path
        self.active_campaigns: Dict[str, Dict] = {}
        self.campaign_templates: Dict[str, Dict] = {}
        self.execution_history: List[CampaignExecution] = []
        self.intelligence_feed = {}
        self.target_environments = {}
        
        # Load campaign templates
        self.load_campaign_templates()
        
    def load_campaign_templates(self):
        """Load pre-defined campaign templates"""
        self.campaign_templates = {
            "apt_simulation": {
                "name": "Advanced Persistent Threat Simulation",
                "description": "Long-term stealth campaign mimicking nation-state actors",
                "duration_days": 30,
                "phases": [
                    CampaignPhase.RECONNAISSANCE,
                    CampaignPhase.INITIAL_ACCESS,
                    CampaignPhase.PERSISTENCE,
                    CampaignPhase.PRIVILEGE_ESCALATION,
                    CampaignPhase.LATERAL_MOVEMENT,
                    CampaignPhase.COLLECTION,
                    CampaignPhase.EXFILTRATION
                ],
                "objectives": [
                    CampaignObjective(
                        id="obj_001",
                        name="Establish Persistent Access",
                        description="Gain and maintain long-term access to target network",
                        priority=10,
                        mitre_tactics=["TA0003", "TA0004"],
                        success_criteria={"persistent_access": True, "stealth_maintained": True},
                        timeout_hours=72
                    ),
                    CampaignObjective(
                        id="obj_002", 
                        name="Exfiltrate Sensitive Data",
                        description="Locate and exfiltrate high-value intellectual property",
                        priority=9,
                        mitre_tactics=["TA0009", "TA0010"],
                        success_criteria={"data_exfiltrated": True, "volume_gb": 10},
                        timeout_hours=48,
                        dependencies=["obj_001"]
                    )
                ]
            },
            
            "ransomware_simulation": {
                "name": "Ransomware Attack Simulation",
                "description": "Fast-moving encryption and extortion campaign",
                "duration_days": 3,
                "phases": [
                    CampaignPhase.RECONNAISSANCE,
                    CampaignPhase.INITIAL_ACCESS,
                    CampaignPhase.LATERAL_MOVEMENT,
                    CampaignPhase.IMPACT
                ],
                "objectives": [
                    CampaignObjective(
                        id="obj_003",
                        name="Encrypt Critical Systems",
                        description="Encrypt file systems and databases",
                        priority=10,
                        mitre_tactics=["TA0040"],
                        success_criteria={"systems_encrypted": 5, "ransom_note_deployed": True},
                        timeout_hours=24
                    )
                ]
            },
            
            "insider_threat_simulation": {
                "name": "Malicious Insider Simulation", 
                "description": "Abuse of legitimate access for malicious purposes",
                "duration_days": 14,
                "phases": [
                    CampaignPhase.COLLECTION,
                    CampaignPhase.EXFILTRATION,
                    CampaignPhase.IMPACT
                ],
                "objectives": [
                    CampaignObjective(
                        id="obj_004",
                        name="Data Theft via Legitimate Access",
                        description="Exfiltrate data using authorized credentials",
                        priority=8,
                        mitre_tactics=["TA0009", "TA0010"],
                        success_criteria={"legitimate_access_abused": True, "detection_avoided": True},
                        timeout_hours=168
                    )
                ]
            }
        }
    
    def create_campaign(self, template_name: str, target_environment: str, 
                       custom_objectives: List[CampaignObjective] = None) -> str:
        """Create new attack campaign from template"""
        
        if template_name not in self.campaign_templates:
            raise ValueError(f"Template {template_name} not found")
        
        campaign_id = str(uuid.uuid4())
        template = self.campaign_templates[template_name]
        
        campaign = {
            "id": campaign_id,
            "name": template["name"],
            "description": template["description"],
            "template": template_name,
            "target_environment": target_environment,
            "status": CampaignStatus.PLANNING.value,
            "created_time": datetime.now().isoformat(),
            "start_time": None,
            "end_time": None,
            "duration_days": template["duration_days"],
            "phases": [phase.value for phase in template["phases"]],
            "objectives": custom_objectives or template["objectives"],
            "current_phase": None,
            "current_objectives": [],
            "completed_objectives": [],
            "failed_objectives": [],
            "attack_vectors": [],
            "execution_log": [],
            "metrics": {
                "objectives_completed": 0,
                "attack_vectors_executed": 0,
                "systems_compromised": 0,
                "data_exfiltrated_gb": 0,
                "persistence_mechanisms": 0,
                "detection_events": 0,
                "stealth_score": 10.0
            }
        }
        
        self.active_campaigns[campaign_id] = campaign
        
        # Generate attack vectors for this campaign
        self.generate_attack_vectors(campaign_id)
        
        print(f"[CAMPAIGN] Created campaign {campaign_id}: {template['name']}")
        return campaign_id
    
    def generate_attack_vectors(self, campaign_id: str):
        """Generate specific attack vectors for campaign phases"""
        campaign = self.active_campaigns[campaign_id]
        attack_vectors = []
        
        # Phase-specific attack vector generation
        for phase in campaign["phases"]:
            if phase == CampaignPhase.RECONNAISSANCE.value:
                attack_vectors.extend([
                    AttackVector(
                        id=f"av_{len(attack_vectors)+1:03d}",
                        name="Passive DNS Enumeration",
                        technique="T1590.002",
                        mitre_id="T1590.002",
                        target_systems=["external_dns"],
                        prerequisites=[],
                        payloads=["dns_enum.py"],
                        c2_profiles=["passive_recon"],
                        stealth_level=9,
                        success_probability=0.95,
                        detection_risk=0.05
                    ),
                    AttackVector(
                        id=f"av_{len(attack_vectors)+2:03d}",
                        name="OSINT Collection",
                        technique="T1589",
                        mitre_id="T1589",
                        target_systems=["social_media", "public_websites"],
                        prerequisites=[],
                        payloads=["osint_crawler.py"],
                        c2_profiles=["tor_proxy"],
                        stealth_level=10,
                        success_probability=0.90,
                        detection_risk=0.02
                    )
                ])
            
            elif phase == CampaignPhase.INITIAL_ACCESS.value:
                attack_vectors.extend([
                    AttackVector(
                        id=f"av_{len(attack_vectors)+1:03d}",
                        name="Spear Phishing Email",
                        technique="T1566.001",
                        mitre_id="T1566.001",
                        target_systems=["email_users"],
                        prerequisites=["email_addresses"],
                        payloads=["phishing_payload.exe"],
                        c2_profiles=["https_beacon"],
                        stealth_level=7,
                        success_probability=0.15,
                        detection_risk=0.30
                    ),
                    AttackVector(
                        id=f"av_{len(attack_vectors)+2:03d}",
                        name="Web Application Exploit",
                        technique="T1190",
                        mitre_id="T1190",
                        target_systems=["web_applications"],
                        prerequisites=["vulnerability_scan"],
                        payloads=["web_shell.php"],
                        c2_profiles=["http_tunnel"],
                        stealth_level=6,
                        success_probability=0.70,
                        detection_risk=0.40
                    )
                ])
            
            elif phase == CampaignPhase.PERSISTENCE.value:
                attack_vectors.extend([
                    AttackVector(
                        id=f"av_{len(attack_vectors)+1:03d}",
                        name="Scheduled Task Creation",
                        technique="T1053.005",
                        mitre_id="T1053.005",
                        target_systems=["windows_hosts"],
                        prerequisites=["admin_access"],
                        payloads=["persistence_task.bat"],
                        c2_profiles=["scheduled_beacon"],
                        stealth_level=8,
                        success_probability=0.85,
                        detection_risk=0.25
                    ),
                    AttackVector(
                        id=f"av_{len(attack_vectors)+3:03d}",
                        name="Registry Modification",
                        technique="T1547.001",
                        mitre_id="T1547.001",
                        target_systems=["windows_hosts"],
                        prerequisites=["system_access"],
                        payloads=["registry_backdoor.ps1"],
                        c2_profiles=["powershell_beacon"],
                        stealth_level=7,
                        success_probability=0.80,
                        detection_risk=0.35
                    )
                ])
        
        campaign["attack_vectors"] = [asdict(av) for av in attack_vectors]
    
    async def execute_campaign(self, campaign_id: str):
        """Execute attack campaign with orchestrated phases"""
        if campaign_id not in self.active_campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.active_campaigns[campaign_id]
        campaign["status"] = CampaignStatus.ACTIVE.value
        campaign["start_time"] = datetime.now().isoformat()
        
        print(f"[CAMPAIGN] Starting execution of {campaign['name']} ({campaign_id})")
        
        try:
            for phase in campaign["phases"]:
                print(f"[CAMPAIGN] Entering phase: {phase}")
                campaign["current_phase"] = phase
                
                # Execute attack vectors for this phase
                phase_vectors = [av for av in campaign["attack_vectors"] 
                               if self.get_phase_for_technique(av["mitre_id"]) == phase]
                
                for vector_data in phase_vectors:
                    vector = AttackVector(**vector_data)
                    execution = await self.execute_attack_vector(campaign_id, vector)
                    campaign["execution_log"].append(asdict(execution))
                    
                    # Update campaign metrics
                    if execution.status == "success":
                        campaign["metrics"]["attack_vectors_executed"] += 1
                        if "system_compromised" in execution.results:
                            campaign["metrics"]["systems_compromised"] += execution.results["system_compromised"]
                    
                    # Check if phase objectives are met
                    if self.check_phase_objectives(campaign_id, phase):
                        print(f"[CAMPAIGN] Phase {phase} objectives completed")
                        break
                
                # Phase delay for stealth
                if phase != campaign["phases"][-1]:  # Don't delay after last phase
                    delay = self.calculate_phase_delay(campaign_id, phase)
                    print(f"[CAMPAIGN] Phase {phase} complete. Waiting {delay}s before next phase...")
                    await asyncio.sleep(delay)
            
            # Campaign completion
            campaign["status"] = CampaignStatus.COMPLETED.value
            campaign["end_time"] = datetime.now().isoformat()
            
            # Generate final campaign report
            await self.generate_campaign_report(campaign_id)
            
        except Exception as e:
            print(f"[CAMPAIGN] Campaign {campaign_id} failed: {e}")
            campaign["status"] = CampaignStatus.FAILED.value
            campaign["end_time"] = datetime.now().isoformat()
    
    async def execute_attack_vector(self, campaign_id: str, vector: AttackVector) -> CampaignExecution:
        """Execute individual attack vector"""
        start_time = datetime.now()
        print(f"[ATTACK] Executing {vector.name} ({vector.mitre_id})")
        
        # Simulate attack execution
        await asyncio.sleep(2)  # Simulated execution time
        
        # Determine success based on probability
        import random
        success = random.random() < vector.success_probability
        
        # Generate execution results
        execution = CampaignExecution(
            phase=CampaignPhase(self.get_phase_for_technique(vector.mitre_id)),
            attack_vector=vector,
            start_time=start_time,
            end_time=datetime.now(),
            status="success" if success else "failed",
            results={
                "success": success,
                "target_systems": vector.target_systems,
                "payloads_deployed": vector.payloads if success else [],
                "system_compromised": len(vector.target_systems) if success else 0,
                "stealth_maintained": random.random() > vector.detection_risk
            },
            artifacts=[
                f"payload_{vector.id}.bin",
                f"execution_log_{vector.id}.txt"
            ] if success else [],
            iocs_generated=[
                f"file_hash_{vector.id}",
                f"network_traffic_{vector.id}"
            ] if success else []
        )
        
        self.execution_history.append(execution)
        return execution
    
    def get_phase_for_technique(self, mitre_id: str) -> str:
        """Map MITRE technique to campaign phase"""
        technique_phase_map = {
            "T1590": CampaignPhase.RECONNAISSANCE.value,
            "T1589": CampaignPhase.RECONNAISSANCE.value,
            "T1566": CampaignPhase.INITIAL_ACCESS.value,
            "T1190": CampaignPhase.INITIAL_ACCESS.value,
            "T1053": CampaignPhase.PERSISTENCE.value,
            "T1547": CampaignPhase.PERSISTENCE.value,
            "T1078": CampaignPhase.CREDENTIAL_ACCESS.value,
            "T1021": CampaignPhase.LATERAL_MOVEMENT.value,
            "T1041": CampaignPhase.EXFILTRATION.value,
            "T1486": CampaignPhase.IMPACT.value
        }
        
        # Extract technique family (first 5 chars)
        technique_family = mitre_id[:5]
        return technique_phase_map.get(technique_family, CampaignPhase.EXECUTION.value)
    
    def check_phase_objectives(self, campaign_id: str, phase: str) -> bool:
        """Check if phase objectives are completed"""
        campaign = self.active_campaigns[campaign_id]
        
        # Simple objective checking logic
        successful_executions = [
            ex for ex in campaign["execution_log"]
            if ex["status"] == "success" and 
            self.get_phase_for_technique(ex["attack_vector"]["mitre_id"]) == phase
        ]
        
        # Basic completion criteria
        if phase == CampaignPhase.RECONNAISSANCE.value:
            return len(successful_executions) >= 1
        elif phase == CampaignPhase.INITIAL_ACCESS.value:
            return any(ex["results"]["system_compromised"] > 0 for ex in successful_executions)
        else:
            return len(successful_executions) >= 1
    
    def calculate_phase_delay(self, campaign_id: str, phase: str) -> int:
        """Calculate delay between phases for stealth"""
        campaign = self.active_campaigns[campaign_id]
        
        # Adaptive delay based on detection risk
        base_delay = 300  # 5 minutes
        stealth_multiplier = campaign["metrics"]["stealth_score"] / 10.0
        detection_events = campaign["metrics"]["detection_events"]
        
        # Increase delay if detections occurred
        delay = base_delay * stealth_multiplier * (1 + detection_events * 0.5)
        
        return int(min(delay, 3600))  # Max 1 hour delay
    
    async def generate_campaign_report(self, campaign_id: str):
        """Generate comprehensive campaign report"""
        campaign = self.active_campaigns[campaign_id]
        
        report = {
            "campaign_id": campaign_id,
            "campaign_name": campaign["name"],
            "execution_summary": {
                "status": campaign["status"],
                "start_time": campaign["start_time"],
                "end_time": campaign["end_time"],
                "duration_hours": self.calculate_duration_hours(campaign),
                "phases_completed": len([p for p in campaign["phases"] if self.check_phase_objectives(campaign_id, p)]),
                "total_phases": len(campaign["phases"])
            },
            "objectives_analysis": {
                "total_objectives": len(campaign["objectives"]),
                "completed_objectives": campaign["metrics"]["objectives_completed"],
                "success_rate": campaign["metrics"]["objectives_completed"] / len(campaign["objectives"])
            },
            "attack_metrics": campaign["metrics"],
            "execution_timeline": campaign["execution_log"],
            "iocs_generated": [
                ioc for ex in campaign["execution_log"] 
                for ioc in ex.get("iocs_generated", [])
            ],
            "recommendations": self.generate_recommendations(campaign_id)
        }
        
        # Save report
        with open(f"/root/Xorb/wargame-enterprise/red-segment/reports/campaign_{campaign_id}_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"[CAMPAIGN] Report generated for campaign {campaign_id}")
        return report
    
    def calculate_duration_hours(self, campaign: Dict) -> float:
        """Calculate campaign duration in hours"""
        if not campaign["start_time"] or not campaign["end_time"]:
            return 0.0
        
        start = datetime.fromisoformat(campaign["start_time"])
        end = datetime.fromisoformat(campaign["end_time"])
        return (end - start).total_seconds() / 3600
    
    def generate_recommendations(self, campaign_id: str) -> List[str]:
        """Generate post-campaign recommendations"""
        campaign = self.active_campaigns[campaign_id]
        recommendations = []
        
        success_rate = campaign["metrics"]["objectives_completed"] / len(campaign["objectives"])
        
        if success_rate < 0.5:
            recommendations.append("Campaign success rate low - review attack vector selection")
        
        if campaign["metrics"]["detection_events"] > 5:
            recommendations.append("High detection rate - improve stealth techniques")
        
        if campaign["metrics"]["systems_compromised"] < 3:
            recommendations.append("Limited system compromise - enhance lateral movement capabilities")
        
        return recommendations
    
    def pause_campaign(self, campaign_id: str):
        """Pause active campaign"""
        if campaign_id in self.active_campaigns:
            self.active_campaigns[campaign_id]["status"] = CampaignStatus.PAUSED.value
            print(f"[CAMPAIGN] Paused campaign {campaign_id}")
    
    def resume_campaign(self, campaign_id: str):
        """Resume paused campaign"""
        if campaign_id in self.active_campaigns:
            self.active_campaigns[campaign_id]["status"] = CampaignStatus.ACTIVE.value
            print(f"[CAMPAIGN] Resumed campaign {campaign_id}")
    
    def abort_campaign(self, campaign_id: str):
        """Abort campaign execution"""
        if campaign_id in self.active_campaigns:
            self.active_campaigns[campaign_id]["status"] = CampaignStatus.ABORTED.value
            self.active_campaigns[campaign_id]["end_time"] = datetime.now().isoformat()
            print(f"[CAMPAIGN] Aborted campaign {campaign_id}")

# Usage example and testing
async def main():
    """Example usage of the Campaign Manager"""
    
    # Initialize campaign manager
    campaign_mgr = OffensiveCampaignManager()
    
    # Create APT simulation campaign
    campaign_id = campaign_mgr.create_campaign(
        template_name="apt_simulation",
        target_environment="meridian_dynamics"
    )
    
    # Execute campaign
    await campaign_mgr.execute_campaign(campaign_id)
    
    print(f"Campaign {campaign_id} execution completed")

if __name__ == "__main__":
    asyncio.run(main())