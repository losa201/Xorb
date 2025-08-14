#!/usr/bin/env python3
"""
XORB PTaaS Cyber Range Orchestrator
Manages and coordinates Red vs Blue cyber range exercises
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import yaml
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExerciseMode(Enum):
    STAGING = "staging"
    LIVE = "live"
    TERMINATED = "terminated"

class CampaignStatus(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

class TeamType(Enum):
    RED = "red"
    BLUE = "blue"
    WHITE = "white"  # Facilitators/observers

@dataclass
class Team:
    """Represents a team in the cyber range exercise"""
    team_id: str
    team_type: TeamType
    name: str
    namespace: str
    members: List[str]
    capabilities: List[str]
    constraints: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self):
        return {
            "team_id": self.team_id,
            "team_type": self.team_type.value,
            "name": self.name,
            "namespace": self.namespace,
            "members": self.members,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class Target:
    """Represents a target system in the cyber range"""
    target_id: str
    name: str
    type: str  # web, internal, ot, database, etc.
    ip_address: str
    ports: List[int]
    services: List[str]
    vulnerabilities: List[str]
    difficulty: str  # easy, medium, hard
    namespace: str
    created_at: datetime
    
    def to_dict(self):
        return {
            "target_id": self.target_id,
            "name": self.name,
            "type": self.type,
            "ip_address": self.ip_address,
            "ports": self.ports,
            "services": self.services,
            "vulnerabilities": self.vulnerabilities,
            "difficulty": self.difficulty,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class Campaign:
    """Represents a cyber range exercise campaign"""
    campaign_id: str
    name: str
    description: str
    mode: ExerciseMode
    status: CampaignStatus
    teams: List[Team]
    targets: List[Target]
    scenario: str
    duration_hours: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "mode": self.mode.value,
            "status": self.status.value,
            "teams": [team.to_dict() for team in self.teams],
            "targets": [target.to_dict() for target in self.targets],
            "scenario": self.scenario,
            "duration_hours": self.duration_hours,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata or {}
        }

class CyberRangeOrchestrator:
    """Main orchestrator for cyber range exercises"""
    
    def __init__(self, config_path: str = "/app/config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.campaigns: Dict[str, Campaign] = {}
        self.current_campaign: Optional[Campaign] = None
        self.k8s_client = None
        self.kill_switch_active = False
        self.performance_metrics = {
            "campaigns_created": 0,
            "campaigns_completed": 0,
            "campaigns_failed": 0,
            "total_runtime_hours": 0.0,
            "attacks_blocked": 0,
            "attacks_allowed": 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "cyber_range": {
                "mode": "staging",
                "exercise_duration": "4h",
                "auto_reset": True,
                "kill_switch_enabled": True
            },
            "security": {
                "network_isolation": True,
                "traffic_monitoring": True,
                "malware_detection": True,
                "geographic_restrictions": True
            },
            "teams": {
                "red_team": {
                    "namespace": "cyber-range-red",
                    "max_concurrent_attacks": 10,
                    "rate_limiting": True
                },
                "blue_team": {
                    "namespace": "cyber-range-blue",
                    "monitoring_enabled": True,
                    "alert_thresholds": {
                        "critical": 5,
                        "warning": 20
                    }
                }
            },
            "targets": {
                "auto_restore": True,
                "snapshot_interval": "15m",
                "backup_retention": "7d"
            },
            "scenarios": {
                "available": [
                    "web_app_pentest",
                    "network_lateral_movement",
                    "apt_simulation",
                    "insider_threat",
                    "ransomware_defense"
                ],
                "default_scenario": "web_app_pentest"
            }
        }
    
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing XORB Cyber Range Orchestrator...")
        
        # Initialize Kubernetes client
        try:
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
                # Running in cluster
                config.load_incluster_config()
            else:
                # Running outside cluster
                config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Check kill switch status
        await self._check_kill_switch_status()
        
        # Load existing campaigns
        await self._load_existing_campaigns()
        
        logger.info("XORB Cyber Range Orchestrator initialized successfully")
    
    async def _check_kill_switch_status(self):
        """Check if kill switch is active"""
        try:
            if os.path.exists("/var/log/cyber-range/kill-switch-status.json"):
                with open("/var/log/cyber-range/kill-switch-status.json", 'r') as f:
                    status_data = json.load(f)
                    if status_data.get("status") == "kill_switch_active":
                        self.kill_switch_active = True
                        logger.warning("Kill switch is ACTIVE - operations restricted")
                    else:
                        self.kill_switch_active = False
            else:
                self.kill_switch_active = False
        except Exception as e:
            logger.error(f"Error checking kill switch status: {e}")
            self.kill_switch_active = False
    
    async def _load_existing_campaigns(self):
        """Load existing campaigns from persistent storage"""
        try:
            campaigns_dir = "/var/log/cyber-range/campaigns"
            if os.path.exists(campaigns_dir):
                for filename in os.listdir(campaigns_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(campaigns_dir, filename)
                        with open(filepath, 'r') as f:
                            campaign_data = json.load(f)
                            campaign = self._campaign_from_dict(campaign_data)
                            self.campaigns[campaign.campaign_id] = campaign
                            
                            # Set current campaign if it's running
                            if campaign.status == CampaignStatus.RUNNING:
                                self.current_campaign = campaign
                
                logger.info(f"Loaded {len(self.campaigns)} existing campaigns")
        except Exception as e:
            logger.error(f"Error loading existing campaigns: {e}")
    
    def _campaign_from_dict(self, data: Dict[str, Any]) -> Campaign:
        """Create Campaign object from dictionary"""
        teams = []
        for team_data in data.get("teams", []):
            team = Team(
                team_id=team_data["team_id"],
                team_type=TeamType(team_data["team_type"]),
                name=team_data["name"],
                namespace=team_data["namespace"],
                members=team_data["members"],
                capabilities=team_data["capabilities"],
                constraints=team_data["constraints"],
                created_at=datetime.fromisoformat(team_data["created_at"])
            )
            teams.append(team)
        
        targets = []
        for target_data in data.get("targets", []):
            target = Target(
                target_id=target_data["target_id"],
                name=target_data["name"],
                type=target_data["type"],
                ip_address=target_data["ip_address"],
                ports=target_data["ports"],
                services=target_data["services"],
                vulnerabilities=target_data["vulnerabilities"],
                difficulty=target_data["difficulty"],
                namespace=target_data["namespace"],
                created_at=datetime.fromisoformat(target_data["created_at"])
            )
            targets.append(target)
        
        return Campaign(
            campaign_id=data["campaign_id"],
            name=data["name"],
            description=data["description"],
            mode=ExerciseMode(data["mode"]),
            status=CampaignStatus(data["status"]),
            teams=teams,
            targets=targets,
            scenario=data["scenario"],
            duration_hours=data["duration_hours"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {})
        )
    
    async def create_campaign(self, 
                            name: str, 
                            description: str,
                            scenario: str,
                            duration_hours: int = 4,
                            mode: ExerciseMode = ExerciseMode.STAGING) -> Campaign:
        """Create a new cyber range campaign"""
        
        if self.kill_switch_active:
            raise Exception("Cannot create campaign while kill switch is active")
        
        logger.info(f"Creating new campaign: {name}")
        
        campaign_id = str(uuid.uuid4())
        
        # Create default teams
        teams = await self._create_default_teams()
        
        # Create targets based on scenario
        targets = await self._create_scenario_targets(scenario)
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name=name,
            description=description,
            mode=mode,
            status=CampaignStatus.CREATED,
            teams=teams,
            targets=targets,
            scenario=scenario,
            duration_hours=duration_hours,
            created_at=datetime.utcnow(),
            metadata={
                "orchestrator_version": "1.0.0",
                "created_by": "xorb-orchestrator",
                "auto_terminate": True
            }
        )
        
        self.campaigns[campaign_id] = campaign
        await self._save_campaign(campaign)
        
        self.performance_metrics["campaigns_created"] += 1
        
        logger.info(f"Campaign created successfully: {campaign_id}")
        return campaign
    
    async def _create_default_teams(self) -> List[Team]:
        """Create default red and blue teams"""
        teams = []
        
        # Red Team
        red_team = Team(
            team_id=str(uuid.uuid4()),
            team_type=TeamType.RED,
            name="Red Team Alpha",
            namespace="cyber-range-red",
            members=["red-operator-1", "red-operator-2"],
            capabilities=[
                "web_application_testing",
                "network_penetration",
                "social_engineering",
                "lateral_movement",
                "privilege_escalation"
            ],
            constraints={
                "max_concurrent_attacks": 10,
                "rate_limiting": True,
                "forbidden_targets": ["10.10.10.0/24", "10.30.0.0/24"],
                "allowed_hours": "09:00-17:00",
                "geographic_restrictions": True
            },
            created_at=datetime.utcnow()
        )
        teams.append(red_team)
        
        # Blue Team
        blue_team = Team(
            team_id=str(uuid.uuid4()),
            team_type=TeamType.BLUE,
            name="Blue Team Defense",
            namespace="cyber-range-blue",
            members=["blue-analyst-1", "blue-analyst-2", "blue-manager-1"],
            capabilities=[
                "siem_monitoring",
                "incident_response",
                "threat_hunting",
                "forensic_analysis",
                "malware_analysis"
            ],
            constraints={
                "monitoring_scope": "all_targets",
                "alert_thresholds": {
                    "critical": 5,
                    "high": 15,
                    "medium": 50
                },
                "response_time_sla": "15_minutes"
            },
            created_at=datetime.utcnow()
        )
        teams.append(blue_team)
        
        # White Team (Facilitators)
        white_team = Team(
            team_id=str(uuid.uuid4()),
            team_type=TeamType.WHITE,
            name="White Team Control",
            namespace="cyber-range-control",
            members=["facilitator-1", "observer-1"],
            capabilities=[
                "exercise_control",
                "scenario_injection",
                "performance_monitoring",
                "kill_switch_control"
            ],
            constraints={
                "full_access": True,
                "emergency_powers": True
            },
            created_at=datetime.utcnow()
        )
        teams.append(white_team)
        
        return teams
    
    async def _create_scenario_targets(self, scenario: str) -> List[Target]:
        """Create targets based on scenario"""
        targets = []
        
        if scenario == "web_app_pentest":
            # Web application targets
            web_target = Target(
                target_id=str(uuid.uuid4()),
                name="DVWA Web Application",
                type="web",
                ip_address="10.100.0.10",
                ports=[80, 443, 8080],
                services=["apache", "mysql", "php"],
                vulnerabilities=[
                    "sql_injection",
                    "xss_reflected",
                    "csrf",
                    "file_inclusion",
                    "weak_authentication"
                ],
                difficulty="medium",
                namespace="cyber-range-targets",
                created_at=datetime.utcnow()
            )
            targets.append(web_target)
            
            # Database target
            db_target = Target(
                target_id=str(uuid.uuid4()),
                name="MySQL Database Server",
                type="database",
                ip_address="10.100.0.11",
                ports=[3306],
                services=["mysql"],
                vulnerabilities=[
                    "weak_credentials",
                    "unencrypted_connections",
                    "privilege_escalation"
                ],
                difficulty="easy",
                namespace="cyber-range-targets",
                created_at=datetime.utcnow()
            )
            targets.append(db_target)
            
        elif scenario == "network_lateral_movement":
            # Multiple internal targets for lateral movement
            for i in range(3):
                target = Target(
                    target_id=str(uuid.uuid4()),
                    name=f"Internal Server {i+1}",
                    type="internal",
                    ip_address=f"10.110.0.{10+i}",
                    ports=[22, 135, 139, 445, 3389],
                    services=["ssh", "smb", "rdp", "winrm"],
                    vulnerabilities=[
                        "weak_smb_config",
                        "unpatched_services",
                        "shared_credentials"
                    ],
                    difficulty="medium",
                    namespace="cyber-range-targets",
                    created_at=datetime.utcnow()
                )
                targets.append(target)
                
        elif scenario == "apt_simulation":
            # Advanced Persistent Threat scenario targets
            targets.extend([
                Target(
                    target_id=str(uuid.uuid4()),
                    name="Executive Workstation",
                    type="workstation",
                    ip_address="10.110.0.20",
                    ports=[22, 3389, 5985],
                    services=["ssh", "rdp", "winrm"],
                    vulnerabilities=[
                        "spear_phishing_susceptible",
                        "outdated_software",
                        "weak_local_admin"
                    ],
                    difficulty="hard",
                    namespace="cyber-range-targets",
                    created_at=datetime.utcnow()
                ),
                Target(
                    target_id=str(uuid.uuid4()),
                    name="Domain Controller",
                    type="domain_controller",
                    ip_address="10.110.0.21",
                    ports=[53, 88, 135, 139, 389, 445, 636],
                    services=["dns", "kerberos", "ldap", "smb"],
                    vulnerabilities=[
                        "kerberoasting",
                        "dcsync_possible",
                        "weak_service_accounts"
                    ],
                    difficulty="hard",
                    namespace="cyber-range-targets",
                    created_at=datetime.utcnow()
                )
            ])
        
        return targets
    
    async def start_campaign(self, campaign_id: str) -> bool:
        """Start a cyber range campaign"""
        
        if self.kill_switch_active:
            raise Exception("Cannot start campaign while kill switch is active")
        
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign not found: {campaign_id}")
        
        campaign = self.campaigns[campaign_id]
        
        if campaign.status != CampaignStatus.CREATED:
            raise ValueError(f"Campaign cannot be started from status: {campaign.status}")
        
        logger.info(f"Starting campaign: {campaign.name} ({campaign_id})")
        
        try:
            # Update campaign status
            campaign.status = CampaignStatus.INITIALIZING
            campaign.started_at = datetime.utcnow()
            await self._save_campaign(campaign)
            
            # Deploy infrastructure
            await self._deploy_campaign_infrastructure(campaign)
            
            # Configure network policies based on mode
            await self._configure_network_policies(campaign)
            
            # Deploy targets
            await self._deploy_targets(campaign)
            
            # Deploy team infrastructure
            await self._deploy_team_infrastructure(campaign)
            
            # Start monitoring
            await self._start_monitoring(campaign)
            
            # Schedule auto-termination
            if campaign.metadata.get("auto_terminate", True):
                await self._schedule_auto_termination(campaign)
            
            # Update status to running
            campaign.status = CampaignStatus.RUNNING
            self.current_campaign = campaign
            await self._save_campaign(campaign)
            
            logger.info(f"Campaign started successfully: {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start campaign {campaign_id}: {e}")
            campaign.status = CampaignStatus.FAILED
            await self._save_campaign(campaign)
            raise
    
    async def _deploy_campaign_infrastructure(self, campaign: Campaign):
        """Deploy Kubernetes infrastructure for the campaign"""
        logger.info(f"Deploying infrastructure for campaign: {campaign.campaign_id}")
        
        # Create campaign-specific configmap
        await self._create_campaign_configmap(campaign)
        
        # Apply resource quotas if not already present
        await self._apply_resource_quotas()
        
        logger.info("Campaign infrastructure deployed")
    
    async def _create_campaign_configmap(self, campaign: Campaign):
        """Create a configmap with campaign configuration"""
        v1 = client.CoreV1Api(self.k8s_client)
        
        config_data = {
            "campaign.json": json.dumps(campaign.to_dict(), indent=2),
            "mode": campaign.mode.value,
            "scenario": campaign.scenario,
            "duration_hours": str(campaign.duration_hours)
        }
        
        configmap = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=f"campaign-{campaign.campaign_id[:8]}",
                namespace="cyber-range-control",
                labels={
                    "app.kubernetes.io/name": "cyber-range-campaign",
                    "cyber-range.xorb.io/campaign-id": campaign.campaign_id,
                    "cyber-range.xorb.io/mode": campaign.mode.value
                }
            ),
            data=config_data
        )
        
        try:
            v1.create_namespaced_config_map(
                namespace="cyber-range-control",
                body=configmap
            )
            logger.info(f"Created campaign configmap: campaign-{campaign.campaign_id[:8]}")
        except ApiException as e:
            if e.status == 409:  # Already exists
                v1.patch_namespaced_config_map(
                    name=f"campaign-{campaign.campaign_id[:8]}",
                    namespace="cyber-range-control",
                    body=configmap
                )
                logger.info(f"Updated campaign configmap: campaign-{campaign.campaign_id[:8]}")
            else:
                raise
    
    async def _apply_resource_quotas(self):
        """Apply resource quotas to namespaces if not already present"""
        v1 = client.CoreV1Api(self.k8s_client)
        
        quotas = {
            "cyber-range-red": {
                "requests.cpu": "8",
                "requests.memory": "16Gi",
                "limits.cpu": "16",
                "limits.memory": "32Gi"
            },
            "cyber-range-blue": {
                "requests.cpu": "6",
                "requests.memory": "12Gi", 
                "limits.cpu": "12",
                "limits.memory": "24Gi"
            },
            "cyber-range-targets": {
                "requests.cpu": "10",
                "requests.memory": "24Gi",
                "limits.cpu": "20", 
                "limits.memory": "48Gi"
            }
        }
        
        for namespace, resources in quotas.items():
            quota = client.V1ResourceQuota(
                metadata=client.V1ObjectMeta(
                    name=f"{namespace}-quota",
                    namespace=namespace
                ),
                spec=client.V1ResourceQuotaSpec(
                    hard=resources
                )
            )
            
            try:
                v1.create_namespaced_resource_quota(
                    namespace=namespace,
                    body=quota
                )
                logger.info(f"Created resource quota for {namespace}")
            except ApiException as e:
                if e.status == 409:  # Already exists
                    logger.debug(f"Resource quota already exists for {namespace}")
                else:
                    logger.error(f"Failed to create resource quota for {namespace}: {e}")
    
    async def _configure_network_policies(self, campaign: Campaign):
        """Configure network policies based on campaign mode"""
        logger.info(f"Configuring network policies for {campaign.mode.value} mode")
        
        # Switch to appropriate mode using mode-switch script
        mode_script = "/opt/cyber-range/scripts/mode-switch.sh"
        
        if os.path.exists(mode_script):
            import subprocess
            try:
                result = subprocess.run([
                    mode_script, campaign.mode.value
                ], capture_output=True, text=True, check=True)
                logger.info(f"Network policies configured for {campaign.mode.value} mode")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to configure network policies: {e.stderr}")
                raise
        else:
            logger.warning("Mode switch script not found, applying policies directly")
            await self._apply_mode_specific_policies(campaign)
    
    async def _apply_mode_specific_policies(self, campaign: Campaign):
        """Apply mode-specific network policies directly via Kubernetes API"""
        networking_v1 = client.NetworkingV1Api(self.k8s_client)
        
        if campaign.mode == ExerciseMode.STAGING:
            # Apply staging mode policy (blocks red team attacks)
            policy_body = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy", 
                "metadata": {
                    "name": "red-team-staging-mode",
                    "namespace": "cyber-range-red",
                    "labels": {
                        "cyber-range.xorb.io/policy-type": "attack-staging",
                        "cyber-range.xorb.io/mode": "staging",
                        "cyber-range.xorb.io/campaign-id": campaign.campaign_id
                    }
                },
                "spec": {
                    "podSelector": {},
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [
                        {
                            "from": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-control"
                                        }
                                    }
                                }
                            ]
                        }
                    ],
                    "egress": [
                        {
                            "to": [],
                            "ports": [
                                {"protocol": "UDP", "port": 53},
                                {"protocol": "TCP", "port": 53}
                            ]
                        },
                        {
                            "to": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-control"
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
        elif campaign.mode == ExerciseMode.LIVE:
            # Apply live mode policy (allows red team attacks)
            policy_body = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "red-team-live-mode",
                    "namespace": "cyber-range-red",
                    "labels": {
                        "cyber-range.xorb.io/policy-type": "attack-live",
                        "cyber-range.xorb.io/mode": "live",
                        "cyber-range.xorb.io/campaign-id": campaign.campaign_id
                    }
                },
                "spec": {
                    "podSelector": {},
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [
                        {
                            "from": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-control"
                                        }
                                    }
                                }
                            ]
                        },
                        {
                            "from": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-targets"
                                        }
                                    }
                                }
                            ],
                            "ports": [
                                {"protocol": "TCP", "port": 4444},
                                {"protocol": "TCP", "port": 1234}
                            ]
                        }
                    ],
                    "egress": [
                        {
                            "to": [],
                            "ports": [
                                {"protocol": "UDP", "port": 53},
                                {"protocol": "TCP", "port": 53}
                            ]
                        },
                        {
                            "to": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-control"
                                        }
                                    }
                                }
                            ]
                        },
                        {
                            "to": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "app.kubernetes.io/name": "cyber-range-targets"
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        
        try:
            networking_v1.create_namespaced_network_policy(
                namespace="cyber-range-red",
                body=policy_body
            )
            logger.info(f"Applied {campaign.mode.value} mode network policy")
        except ApiException as e:
            if e.status == 409:  # Already exists
                networking_v1.patch_namespaced_network_policy(
                    name=policy_body["metadata"]["name"],
                    namespace="cyber-range-red",
                    body=policy_body
                )
                logger.info(f"Updated {campaign.mode.value} mode network policy")
            else:
                raise
    
    async def _deploy_targets(self, campaign: Campaign):
        """Deploy target systems for the campaign"""
        logger.info(f"Deploying {len(campaign.targets)} targets")
        
        for target in campaign.targets:
            await self._deploy_single_target(target)
        
        logger.info("All targets deployed successfully")
    
    async def _deploy_single_target(self, target: Target):
        """Deploy a single target system"""
        apps_v1 = client.AppsV1Api(self.k8s_client)
        v1 = client.CoreV1Api(self.k8s_client)
        
        # Create deployment based on target type
        if target.type == "web":
            deployment = self._create_web_target_deployment(target)
        elif target.type == "database":
            deployment = self._create_database_target_deployment(target)
        elif target.type == "internal":
            deployment = self._create_internal_target_deployment(target)
        else:
            deployment = self._create_generic_target_deployment(target)
        
        try:
            apps_v1.create_namespaced_deployment(
                namespace=target.namespace,
                body=deployment
            )
            logger.info(f"Deployed target: {target.name}")
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Target already exists: {target.name}")
            else:
                raise
    
    def _create_web_target_deployment(self, target: Target) -> client.V1Deployment:
        """Create deployment for web application target"""
        return client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"target-{target.target_id[:8]}",
                namespace=target.namespace,
                labels={
                    "app": f"target-{target.target_id[:8]}",
                    "cyber-range.xorb.io/target-type": target.type,
                    "cyber-range.xorb.io/target-id": target.target_id,
                    "cyber-range.xorb.io/difficulty": target.difficulty
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"target-{target.target_id[:8]}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"target-{target.target_id[:8]}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="dvwa",
                                image="vulnerables/web-dvwa:latest",
                                ports=[
                                    client.V1ContainerPort(container_port=80)
                                ],
                                env=[
                                    client.V1EnvVar(name="MYSQL_DATABASE", value="dvwa"),
                                    client.V1EnvVar(name="MYSQL_USER", value="dvwa"),
                                    client.V1EnvVar(name="MYSQL_PASSWORD", value="p@ssw0rd")
                                ]
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_database_target_deployment(self, target: Target) -> client.V1Deployment:
        """Create deployment for database target"""
        return client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"target-{target.target_id[:8]}",
                namespace=target.namespace,
                labels={
                    "app": f"target-{target.target_id[:8]}",
                    "cyber-range.xorb.io/target-type": target.type,
                    "cyber-range.xorb.io/target-id": target.target_id
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"target-{target.target_id[:8]}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"target-{target.target_id[:8]}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="mysql",
                                image="mysql:5.7",
                                ports=[
                                    client.V1ContainerPort(container_port=3306)
                                ],
                                env=[
                                    client.V1EnvVar(name="MYSQL_ROOT_PASSWORD", value="weak_password"),
                                    client.V1EnvVar(name="MYSQL_DATABASE", value="testdb"),
                                    client.V1EnvVar(name="MYSQL_USER", value="testuser"),
                                    client.V1EnvVar(name="MYSQL_PASSWORD", value="test123")
                                ]
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_internal_target_deployment(self, target: Target) -> client.V1Deployment:
        """Create deployment for internal server target"""
        return client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"target-{target.target_id[:8]}",
                namespace=target.namespace,
                labels={
                    "app": f"target-{target.target_id[:8]}",
                    "cyber-range.xorb.io/target-type": target.type,
                    "cyber-range.xorb.io/target-id": target.target_id
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"target-{target.target_id[:8]}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"target-{target.target_id[:8]}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="ubuntu-server",
                                image="ubuntu:20.04",
                                command=["/bin/bash"],
                                args=["-c", "apt-get update && apt-get install -y openssh-server && service ssh start && sleep infinity"],
                                ports=[
                                    client.V1ContainerPort(container_port=22)
                                ]
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_generic_target_deployment(self, target: Target) -> client.V1Deployment:
        """Create deployment for generic target"""
        return client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"target-{target.target_id[:8]}",
                namespace=target.namespace,
                labels={
                    "app": f"target-{target.target_id[:8]}",
                    "cyber-range.xorb.io/target-type": target.type,
                    "cyber-range.xorb.io/target-id": target.target_id
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"target-{target.target_id[:8]}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"target-{target.target_id[:8]}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="target",
                                image="nginx:alpine",
                                ports=[
                                    client.V1ContainerPort(container_port=80)
                                ]
                            )
                        ]
                    )
                )
            )
        )
    
    async def _deploy_team_infrastructure(self, campaign: Campaign):
        """Deploy infrastructure for teams"""
        logger.info("Deploying team infrastructure")
        
        # Team infrastructure is handled by the base Kubernetes manifests
        # This method can be extended to deploy team-specific tools
        
        for team in campaign.teams:
            if team.team_type == TeamType.RED:
                await self._ensure_red_team_tools(team)
            elif team.team_type == TeamType.BLUE:
                await self._ensure_blue_team_tools(team)
    
    async def _ensure_red_team_tools(self, team: Team):
        """Ensure red team tools are available"""
        # Red team tools are deployed via the red-team.yaml manifest
        # This method can check if tools are running and restart if needed
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            deployments = apps_v1.list_namespaced_deployment(namespace=team.namespace)
            tool_count = len(deployments.items)
            logger.info(f"Red team has {tool_count} tools deployed")
        except ApiException as e:
            logger.error(f"Failed to check red team tools: {e}")
    
    async def _ensure_blue_team_tools(self, team: Team):
        """Ensure blue team tools are available"""
        # Blue team tools are deployed via the blue-team.yaml manifest
        # This method can check if SIEM and monitoring tools are running
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            deployments = apps_v1.list_namespaced_deployment(namespace=team.namespace)
            tool_count = len(deployments.items)
            logger.info(f"Blue team has {tool_count} monitoring tools deployed")
        except ApiException as e:
            logger.error(f"Failed to check blue team tools: {e}")
    
    async def _start_monitoring(self, campaign: Campaign):
        """Start monitoring for the campaign"""
        logger.info("Starting campaign monitoring")
        
        # Start background monitoring tasks
        asyncio.create_task(self._monitor_campaign_health(campaign))
        asyncio.create_task(self._monitor_attack_activity(campaign))
        asyncio.create_task(self._monitor_performance_metrics(campaign))
    
    async def _monitor_campaign_health(self, campaign: Campaign):
        """Monitor overall campaign health"""
        while campaign.status == CampaignStatus.RUNNING:
            try:
                # Check if kill switch is activated
                await self._check_kill_switch_status()
                if self.kill_switch_active:
                    logger.warning("Kill switch activated - terminating campaign")
                    await self.terminate_campaign(campaign.campaign_id, "kill_switch_activated")
                    break
                
                # Check pod health
                v1 = client.CoreV1Api(self.k8s_client)
                for namespace in ["cyber-range-red", "cyber-range-blue", "cyber-range-targets"]:
                    pods = v1.list_namespaced_pod(namespace=namespace)
                    failed_pods = [pod for pod in pods.items if pod.status.phase == "Failed"]
                    if failed_pods:
                        logger.warning(f"Found {len(failed_pods)} failed pods in {namespace}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in campaign health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_attack_activity(self, campaign: Campaign):
        """Monitor attack activity and update metrics"""
        while campaign.status == CampaignStatus.RUNNING:
            try:
                # This would integrate with actual network monitoring
                # For now, simulate some activity
                if campaign.mode == ExerciseMode.LIVE:
                    self.performance_metrics["attacks_allowed"] += 1
                else:
                    self.performance_metrics["attacks_blocked"] += 1
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in attack activity monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance_metrics(self, campaign: Campaign):
        """Monitor and update performance metrics"""
        while campaign.status == CampaignStatus.RUNNING:
            try:
                # Update runtime metrics
                if campaign.started_at:
                    runtime_hours = (datetime.utcnow() - campaign.started_at).total_seconds() / 3600
                    self.performance_metrics["total_runtime_hours"] = runtime_hours
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_auto_termination(self, campaign: Campaign):
        """Schedule automatic termination of campaign"""
        logger.info(f"Scheduling auto-termination in {campaign.duration_hours} hours")
        
        async def auto_terminate():
            await asyncio.sleep(campaign.duration_hours * 3600)
            if campaign.status == CampaignStatus.RUNNING:
                logger.info(f"Auto-terminating campaign: {campaign.campaign_id}")
                await self.terminate_campaign(campaign.campaign_id, "time_limit_exceeded")
        
        asyncio.create_task(auto_terminate())
    
    async def terminate_campaign(self, campaign_id: str, reason: str = "manual_termination") -> bool:
        """Terminate a running campaign"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign not found: {campaign_id}")
        
        campaign = self.campaigns[campaign_id]
        
        if campaign.status not in [CampaignStatus.RUNNING, CampaignStatus.PAUSED]:
            raise ValueError(f"Campaign cannot be terminated from status: {campaign.status}")
        
        logger.info(f"Terminating campaign: {campaign.name} ({campaign_id}) - Reason: {reason}")
        
        try:
            # Update campaign status
            campaign.status = CampaignStatus.TERMINATED
            campaign.completed_at = datetime.utcnow()
            campaign.metadata["termination_reason"] = reason
            
            # Clean up resources
            await self._cleanup_campaign_resources(campaign)
            
            # Update metrics
            if reason == "time_limit_exceeded" or reason == "manual_termination":
                self.performance_metrics["campaigns_completed"] += 1
            else:
                self.performance_metrics["campaigns_failed"] += 1
            
            # Save final state
            await self._save_campaign(campaign)
            
            # Clear current campaign if this was it
            if self.current_campaign and self.current_campaign.campaign_id == campaign_id:
                self.current_campaign = None
            
            logger.info(f"Campaign terminated successfully: {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate campaign {campaign_id}: {e}")
            campaign.status = CampaignStatus.FAILED
            await self._save_campaign(campaign)
            raise
    
    async def _cleanup_campaign_resources(self, campaign: Campaign):
        """Clean up Kubernetes resources for the campaign"""
        logger.info(f"Cleaning up resources for campaign: {campaign.campaign_id}")
        
        # Delete campaign configmap
        v1 = client.CoreV1Api(self.k8s_client)
        try:
            v1.delete_namespaced_config_map(
                name=f"campaign-{campaign.campaign_id[:8]}",
                namespace="cyber-range-control"
            )
        except ApiException as e:
            if e.status != 404:  # Ignore not found
                logger.error(f"Failed to delete campaign configmap: {e}")
        
        # Delete target deployments
        apps_v1 = client.AppsV1Api(self.k8s_client)
        for target in campaign.targets:
            try:
                apps_v1.delete_namespaced_deployment(
                    name=f"target-{target.target_id[:8]}",
                    namespace=target.namespace
                )
            except ApiException as e:
                if e.status != 404:  # Ignore not found
                    logger.error(f"Failed to delete target deployment: {e}")
        
        # Clean up network policies
        networking_v1 = client.NetworkingV1Api(self.k8s_client)
        for policy_name in [f"red-team-{campaign.mode.value}-mode"]:
            try:
                networking_v1.delete_namespaced_network_policy(
                    name=policy_name,
                    namespace="cyber-range-red"
                )
            except ApiException as e:
                if e.status != 404:  # Ignore not found
                    logger.error(f"Failed to delete network policy: {e}")
        
        logger.info("Resource cleanup completed")
    
    async def _save_campaign(self, campaign: Campaign):
        """Save campaign to persistent storage"""
        campaigns_dir = "/var/log/cyber-range/campaigns"
        os.makedirs(campaigns_dir, exist_ok=True)
        
        filepath = os.path.join(campaigns_dir, f"{campaign.campaign_id}.json")
        with open(filepath, 'w') as f:
            json.dump(campaign.to_dict(), f, indent=2)
    
    async def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get current status of a campaign"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign not found: {campaign_id}")
        
        campaign = self.campaigns[campaign_id]
        
        # Get real-time pod status
        pod_status = await self._get_pod_status_for_campaign(campaign)
        
        return {
            "campaign": campaign.to_dict(),
            "pod_status": pod_status,
            "metrics": self.performance_metrics,
            "kill_switch_active": self.kill_switch_active
        }
    
    async def _get_pod_status_for_campaign(self, campaign: Campaign) -> Dict[str, Any]:
        """Get pod status for campaign resources"""
        v1 = client.CoreV1Api(self.k8s_client)
        
        status = {}
        
        for namespace in ["cyber-range-red", "cyber-range-blue", "cyber-range-targets"]:
            try:
                pods = v1.list_namespaced_pod(namespace=namespace)
                status[namespace] = {
                    "total": len(pods.items),
                    "running": len([p for p in pods.items if p.status.phase == "Running"]),
                    "pending": len([p for p in pods.items if p.status.phase == "Pending"]),
                    "failed": len([p for p in pods.items if p.status.phase == "Failed"])
                }
            except ApiException as e:
                status[namespace] = {"error": str(e)}
        
        return status
    
    async def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all campaigns"""
        return [campaign.to_dict() for campaign in self.campaigns.values()]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "metrics": self.performance_metrics,
            "current_campaign": self.current_campaign.to_dict() if self.current_campaign else None,
            "total_campaigns": len(self.campaigns),
            "kill_switch_active": self.kill_switch_active,
            "timestamp": datetime.utcnow().isoformat()
        }

# FastAPI integration
async def create_orchestrator() -> CyberRangeOrchestrator:
    """Create and initialize the orchestrator"""
    orchestrator = CyberRangeOrchestrator()
    await orchestrator.initialize()
    return orchestrator

# Global orchestrator instance (for FastAPI integration)
_orchestrator_instance: Optional[CyberRangeOrchestrator] = None

async def get_orchestrator() -> CyberRangeOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = await create_orchestrator()
    return _orchestrator_instance

if __name__ == "__main__":
    async def main():
        """Main function for standalone execution"""
        orchestrator = await create_orchestrator()
        
        logger.info("XORB Cyber Range Orchestrator is running...")
        
        try:
            # Keep the orchestrator running
            while True:
                await asyncio.sleep(60)
                
                # Periodic health check
                if orchestrator.current_campaign:
                    logger.info(f"Current campaign: {orchestrator.current_campaign.name} ({orchestrator.current_campaign.status.value})")
                
        except KeyboardInterrupt:
            logger.info("Shutting down orchestrator...")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            sys.exit(1)
    
    asyncio.run(main())