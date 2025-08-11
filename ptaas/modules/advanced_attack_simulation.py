#!/usr/bin/env python3
"""
Advanced Attack Simulation Engine
Sophisticated penetration testing and red team simulation capabilities
"""

import asyncio
import logging
import json
import time
import random
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
import uuid
import subprocess
import tempfile
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'api', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'common'))

try:
    from services.base_service import SecurityService, ServiceHealth, ServiceStatus
except ImportError:
    SecurityService = None
    ServiceHealth = None 
    ServiceStatus = None

logger = logging.getLogger(__name__)

class AttackPhase(Enum):
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
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

class AttackTechnique(Enum):
    # Reconnaissance
    NETWORK_SCANNING = "T1595.001"
    DNS_ENUMERATION = "T1595.002"
    SOCIAL_MEDIA_RECON = "T1593.001"
    
    # Initial Access
    SPEAR_PHISHING = "T1566.001"
    DRIVE_BY_COMPROMISE = "T1189"
    EXPLOIT_PUBLIC_APPLICATION = "T1190"
    SUPPLY_CHAIN_COMPROMISE = "T1195"
    
    # Execution
    COMMAND_LINE_INTERFACE = "T1059.003"
    POWERSHELL = "T1059.001"
    SCRIPTING = "T1059.006"
    
    # Persistence
    SCHEDULED_TASK = "T1053.005"
    REGISTRY_RUN_KEYS = "T1547.001"
    STARTUP_FOLDER = "T1547.001"
    
    # Privilege Escalation
    UAC_BYPASS = "T1548.002"
    TOKEN_IMPERSONATION = "T1134.001"
    EXPLOITATION_FOR_PRIVESC = "T1068"
    
    # Defense Evasion
    OBFUSCATED_FILES = "T1027"
    PROCESS_INJECTION = "T1055"
    TIMESTOMP = "T1070.006"
    
    # Credential Access
    CREDENTIAL_DUMPING = "T1003"
    KEYLOGGING = "T1056.001"
    BRUTE_FORCE = "T1110"
    
    # Discovery
    SYSTEM_INFORMATION_DISCOVERY = "T1082"
    NETWORK_SERVICE_SCANNING = "T1046"
    PROCESS_DISCOVERY = "T1057"
    
    # Lateral Movement
    REMOTE_DESKTOP_PROTOCOL = "T1021.001"
    SMB_WINDOWS_ADMIN_SHARES = "T1021.002"
    PASS_THE_HASH = "T1550.002"
    
    # Collection
    DATA_FROM_LOCAL_SYSTEM = "T1005"
    SCREEN_CAPTURE = "T1113"
    CLIPBOARD_DATA = "T1115"
    
    # Command and Control
    APPLICATION_LAYER_PROTOCOL = "T1071"
    DNS_TUNNELING = "T1071.004"
    ENCRYPTED_CHANNEL = "T1573"
    
    # Exfiltration
    EXFILTRATION_OVER_C2 = "T1041"
    SCHEDULED_TRANSFER = "T1029"
    
    # Impact
    DATA_ENCRYPTED_FOR_IMPACT = "T1486"
    SERVICE_STOP = "T1489"
    DEFACEMENT = "T1491"

class SimulationComplexity(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    APT_LEVEL = "apt_level"

class SimulationEnvironment(Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    ISOLATED = "isolated"

@dataclass
class AttackStep:
    """Individual step in an attack simulation"""
    step_id: str
    phase: AttackPhase
    technique: AttackTechnique
    name: str
    description: str
    parameters: Dict[str, Any]
    prerequisites: List[str]
    success_criteria: List[str]
    detection_signatures: List[str]
    cleanup_actions: List[str]
    estimated_duration: int  # seconds
    risk_level: str = "medium"
    
@dataclass
class SimulationScenario:
    """Complete attack simulation scenario"""
    scenario_id: str
    name: str
    description: str
    attack_chain: List[AttackStep]
    complexity: SimulationComplexity
    target_environment: SimulationEnvironment
    objectives: List[str]
    constraints: List[str]
    safety_measures: List[str]
    estimated_duration: int  # total seconds
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class SimulationExecution:
    """Runtime execution of a simulation"""
    execution_id: str
    scenario_id: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    detected_steps: List[str] = None
    results: Dict[str, Any] = None
    logs: List[str] = None
    safety_triggered: bool = False
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        if self.detected_steps is None:
            self.detected_steps = []
        if self.results is None:
            self.results = {}
        if self.logs is None:
            self.logs = []

class AttackStepExecutor:
    """Executes individual attack steps"""
    
    def __init__(self):
        self.step_handlers = {}
        self.detection_monitors = []
        self._register_handlers()
    
    def _register_handlers(self):
        """Register attack step handlers"""
        # Reconnaissance handlers
        self.step_handlers[AttackTechnique.NETWORK_SCANNING] = self._execute_network_scanning
        self.step_handlers[AttackTechnique.DNS_ENUMERATION] = self._execute_dns_enumeration
        
        # Initial Access handlers
        self.step_handlers[AttackTechnique.SPEAR_PHISHING] = self._execute_spear_phishing
        self.step_handlers[AttackTechnique.EXPLOIT_PUBLIC_APPLICATION] = self._execute_web_exploit
        
        # Execution handlers
        self.step_handlers[AttackTechnique.COMMAND_LINE_INTERFACE] = self._execute_command_line
        self.step_handlers[AttackTechnique.POWERSHELL] = self._execute_powershell
        
        # Persistence handlers
        self.step_handlers[AttackTechnique.SCHEDULED_TASK] = self._execute_scheduled_task
        self.step_handlers[AttackTechnique.REGISTRY_RUN_KEYS] = self._execute_registry_persistence
        
        # Privilege Escalation handlers
        self.step_handlers[AttackTechnique.UAC_BYPASS] = self._execute_uac_bypass
        
        # Defense Evasion handlers
        self.step_handlers[AttackTechnique.OBFUSCATED_FILES] = self._execute_obfuscation
        self.step_handlers[AttackTechnique.PROCESS_INJECTION] = self._execute_process_injection
        
        # Credential Access handlers
        self.step_handlers[AttackTechnique.CREDENTIAL_DUMPING] = self._execute_credential_dumping
        
        # Discovery handlers
        self.step_handlers[AttackTechnique.SYSTEM_INFORMATION_DISCOVERY] = self._execute_system_discovery
        self.step_handlers[AttackTechnique.NETWORK_SERVICE_SCANNING] = self._execute_service_scanning
        
        # Lateral Movement handlers
        self.step_handlers[AttackTechnique.REMOTE_DESKTOP_PROTOCOL] = self._execute_rdp_lateral_movement
        self.step_handlers[AttackTechnique.SMB_WINDOWS_ADMIN_SHARES] = self._execute_smb_lateral_movement
        
        # Collection handlers
        self.step_handlers[AttackTechnique.DATA_FROM_LOCAL_SYSTEM] = self._execute_data_collection
        
        # Exfiltration handlers
        self.step_handlers[AttackTechnique.EXFILTRATION_OVER_C2] = self._execute_data_exfiltration
        
        # Impact handlers
        self.step_handlers[AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT] = self._execute_ransomware_simulation
    
    async def execute_step(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single attack step"""
        try:
            logger.info(f"Executing attack step: {step.name} ({step.technique.value})")
            
            # Check prerequisites
            if not await self._check_prerequisites(step, context):
                return {
                    "success": False,
                    "error": "Prerequisites not met",
                    "step_id": step.step_id
                }
            
            # Execute the step
            handler = self.step_handlers.get(step.technique)
            if not handler:
                return {
                    "success": False,
                    "error": f"No handler for technique {step.technique.value}",
                    "step_id": step.step_id
                }
            
            # Set timeout
            result = await asyncio.wait_for(
                handler(step, context),
                timeout=step.estimated_duration * 2  # Allow double the estimated time
            )
            
            # Check for detection
            detected = await self._check_detection(step, result, context)
            if detected:
                result["detected"] = True
                result["detection_details"] = detected
            
            result["step_id"] = step.step_id
            result["technique"] = step.technique.value
            result["phase"] = step.phase.value
            
            logger.info(f"Attack step {step.name} completed: {'detected' if detected else 'undetected'}")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Attack step {step.name} timed out")
            return {
                "success": False,
                "error": "Step execution timed out",
                "step_id": step.step_id,
                "timeout": True
            }
        except Exception as e:
            logger.error(f"Attack step {step.name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id
            }
    
    async def _check_prerequisites(self, step: AttackStep, context: Dict[str, Any]) -> bool:
        """Check if step prerequisites are met"""
        for prereq in step.prerequisites:
            if prereq not in context.get("completed_techniques", []):
                logger.warning(f"Prerequisite {prereq} not met for step {step.step_id}")
                return False
        return True
    
    async def _check_detection(self, step: AttackStep, result: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if the attack step was detected"""
        # Simulate detection based on step characteristics
        detection_probability = self._calculate_detection_probability(step, context)
        
        if random.random() < detection_probability:
            return {
                "detected_by": random.choice(["EDR", "SIEM", "Network_Monitor", "User_Report"]),
                "detection_time": datetime.utcnow().isoformat(),
                "confidence": random.uniform(0.7, 0.95),
                "signature_matched": random.choice(step.detection_signatures) if step.detection_signatures else None
            }
        
        return None
    
    def _calculate_detection_probability(self, step: AttackStep, context: Dict[str, Any]) -> float:
        """Calculate probability of detection for an attack step"""
        base_probability = 0.3  # Base 30% detection rate
        
        # Adjust based on phase
        phase_modifiers = {
            AttackPhase.RECONNAISSANCE: -0.1,
            AttackPhase.INITIAL_ACCESS: 0.1,
            AttackPhase.EXECUTION: 0.2,
            AttackPhase.PERSISTENCE: 0.15,
            AttackPhase.PRIVILEGE_ESCALATION: 0.25,
            AttackPhase.DEFENSE_EVASION: -0.2,
            AttackPhase.LATERAL_MOVEMENT: 0.3,
            AttackPhase.COLLECTION: 0.2,
            AttackPhase.EXFILTRATION: 0.4,
            AttackPhase.IMPACT: 0.5
        }
        
        probability = base_probability + phase_modifiers.get(step.phase, 0)
        
        # Adjust based on risk level
        if step.risk_level == "high":
            probability += 0.2
        elif step.risk_level == "low":
            probability -= 0.1
        
        # Adjust based on environment
        env = context.get("environment", SimulationEnvironment.ISOLATED)
        if env == SimulationEnvironment.PRODUCTION:
            probability += 0.2
        elif env == SimulationEnvironment.ISOLATED:
            probability -= 0.3
        
        return max(0.05, min(0.95, probability))  # Clamp between 5% and 95%
    
    # Attack step handlers (simulated implementations)
    
    async def _execute_network_scanning(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network scanning"""
        target_range = step.parameters.get("target_range", "192.168.1.0/24")
        ports = step.parameters.get("ports", [22, 80, 443, 445, 3389])
        
        await asyncio.sleep(2)  # Simulate scan time
        
        # Simulate discovered hosts
        discovered_hosts = []
        for i in range(random.randint(3, 10)):
            host_ip = f"192.168.1.{random.randint(10, 250)}"
            open_ports = random.sample(ports, random.randint(1, 3))
            discovered_hosts.append({
                "ip": host_ip,
                "open_ports": open_ports,
                "os_guess": random.choice(["Windows 10", "Windows Server 2019", "Ubuntu 20.04", "CentOS 7"])
            })
        
        return {
            "success": True,
            "discovered_hosts": discovered_hosts,
            "scan_duration": 120,
            "total_hosts": len(discovered_hosts)
        }
    
    async def _execute_dns_enumeration(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate DNS enumeration"""
        domain = step.parameters.get("domain", "target.com")
        
        await asyncio.sleep(1)
        
        subdomains = [
            f"www.{domain}",
            f"mail.{domain}",
            f"ftp.{domain}",
            f"admin.{domain}",
            f"dev.{domain}",
            f"staging.{domain}"
        ]
        
        return {
            "success": True,
            "domain": domain,
            "discovered_subdomains": random.sample(subdomains, random.randint(2, 5)),
            "dns_records": {
                "A": ["192.168.1.100", "192.168.1.101"],
                "MX": ["mail.target.com"],
                "NS": ["ns1.target.com", "ns2.target.com"]
            }
        }
    
    async def _execute_spear_phishing(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate spear phishing attack"""
        targets = step.parameters.get("targets", ["user1@target.com", "user2@target.com"])
        template = step.parameters.get("template", "credential_harvest")
        
        await asyncio.sleep(1)
        
        # Simulate email delivery and user interaction
        results = []
        for target in targets:
            interaction = random.choice(["opened", "clicked", "downloaded", "ignored"])
            results.append({
                "target": target,
                "delivered": True,
                "interaction": interaction,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        successful_compromises = [r for r in results if r["interaction"] in ["clicked", "downloaded"]]
        
        return {
            "success": len(successful_compromises) > 0,
            "emails_sent": len(targets),
            "successful_compromises": len(successful_compromises),
            "results": results,
            "credentials_harvested": len(successful_compromises) * random.randint(0, 2)
        }
    
    async def _execute_web_exploit(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web application exploitation"""
        target_url = step.parameters.get("target_url", "http://target.com/app")
        exploit_type = step.parameters.get("exploit_type", "sql_injection")
        
        await asyncio.sleep(1.5)
        
        # Simulate exploitation attempt
        success_rate = 0.7 if exploit_type == "sql_injection" else 0.5
        success = random.random() < success_rate
        
        if success:
            return {
                "success": True,
                "exploit_type": exploit_type,
                "target_url": target_url,
                "shell_obtained": True,
                "privileges": "www-data",
                "access_level": "limited"
            }
        else:
            return {
                "success": False,
                "exploit_type": exploit_type,
                "target_url": target_url,
                "error": "Exploitation failed - target may be patched"
            }
    
    async def _execute_command_line(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate command line execution"""
        commands = step.parameters.get("commands", ["whoami", "ipconfig", "tasklist"])
        
        await asyncio.sleep(0.5)
        
        # Simulate command execution
        results = []
        for cmd in commands:
            results.append({
                "command": cmd,
                "output": f"Simulated output for: {cmd}",
                "exit_code": 0,
                "execution_time": random.uniform(0.1, 2.0)
            })
        
        return {
            "success": True,
            "commands_executed": len(commands),
            "results": results,
            "shell_access": "maintained"
        }
    
    async def _execute_powershell(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate PowerShell execution"""
        script = step.parameters.get("script", "Get-Process")
        
        await asyncio.sleep(0.8)
        
        return {
            "success": True,
            "script_executed": script,
            "output_length": random.randint(100, 1000),
            "execution_policy_bypassed": True,
            "modules_loaded": ["Microsoft.PowerShell.Management", "Microsoft.PowerShell.Utility"]
        }
    
    async def _execute_scheduled_task(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate scheduled task creation for persistence"""
        task_name = step.parameters.get("task_name", "SystemUpdate")
        
        await asyncio.sleep(0.5)
        
        return {
            "success": True,
            "task_name": task_name,
            "task_created": True,
            "trigger": "daily",
            "persistence_established": True
        }
    
    async def _execute_registry_persistence(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate registry-based persistence"""
        registry_key = step.parameters.get("key", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run")
        
        await asyncio.sleep(0.3)
        
        return {
            "success": True,
            "registry_key": registry_key,
            "value_created": True,
            "persistence_mechanism": "autostart",
            "stealth_level": "medium"
        }
    
    async def _execute_uac_bypass(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate UAC bypass for privilege escalation"""
        method = step.parameters.get("method", "eventvwr_bypass")
        
        await asyncio.sleep(1)
        
        success = random.random() < 0.6  # 60% success rate
        
        if success:
            return {
                "success": True,
                "bypass_method": method,
                "elevated_privileges": True,
                "new_privilege_level": "Administrator"
            }
        else:
            return {
                "success": False,
                "bypass_method": method,
                "error": "UAC bypass failed - method may be patched"
            }
    
    async def _execute_obfuscation(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate file obfuscation for defense evasion"""
        file_type = step.parameters.get("file_type", "executable")
        
        await asyncio.sleep(0.7)
        
        return {
            "success": True,
            "file_type": file_type,
            "obfuscation_applied": True,
            "entropy_increased": True,
            "av_detection_reduced": True
        }
    
    async def _execute_process_injection(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate process injection"""
        target_process = step.parameters.get("target_process", "explorer.exe")
        
        await asyncio.sleep(0.8)
        
        return {
            "success": True,
            "target_process": target_process,
            "injection_successful": True,
            "method": "DLL injection",
            "stealth_achieved": True
        }
    
    async def _execute_credential_dumping(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate credential dumping"""
        method = step.parameters.get("method", "lsass_dump")
        
        await asyncio.sleep(1.2)
        
        # Simulate credential extraction
        credentials = []
        for i in range(random.randint(2, 8)):
            credentials.append({
                "username": f"user{i}",
                "domain": "CORPORATE",
                "hash": hashlib.md5(f"password{i}".encode()).hexdigest(),
                "type": random.choice(["NTLM", "Kerberos", "plaintext"])
            })
        
        return {
            "success": True,
            "method": method,
            "credentials_extracted": len(credentials),
            "credentials": credentials,
            "admin_credentials": random.randint(0, 2)
        }
    
    async def _execute_system_discovery(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate system information discovery"""
        await asyncio.sleep(0.4)
        
        return {
            "success": True,
            "os_version": "Windows 10 Pro",
            "architecture": "x64",
            "domain": "CORPORATE.LOCAL",
            "installed_software": ["Microsoft Office", "Adobe Reader", "Google Chrome"],
            "network_interfaces": ["Ethernet", "WiFi"],
            "users": ["Administrator", "user1", "user2", "service_account"]
        }
    
    async def _execute_service_scanning(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network service scanning"""
        await asyncio.sleep(1.5)
        
        services = []
        for port in [21, 22, 80, 135, 139, 443, 445, 3389]:
            if random.random() < 0.4:  # 40% chance service is running
                services.append({
                    "port": port,
                    "service": {21: "FTP", 22: "SSH", 80: "HTTP", 135: "RPC", 
                              139: "NetBIOS", 443: "HTTPS", 445: "SMB", 3389: "RDP"}[port],
                    "version": "Unknown",
                    "state": "open"
                })
        
        return {
            "success": True,
            "services_discovered": len(services),
            "services": services,
            "potential_vulnerabilities": random.randint(1, 3)
        }
    
    async def _execute_rdp_lateral_movement(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate lateral movement via RDP"""
        target_host = step.parameters.get("target_host", "192.168.1.50")
        
        await asyncio.sleep(2)
        
        success = random.random() < 0.5  # 50% success rate
        
        if success:
            return {
                "success": True,
                "target_host": target_host,
                "connection_established": True,
                "method": "RDP",
                "credentials_used": "dumped_admin_creds"
            }
        else:
            return {
                "success": False,
                "target_host": target_host,
                "error": "RDP connection failed - credentials invalid or RDP disabled"
            }
    
    async def _execute_smb_lateral_movement(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate lateral movement via SMB"""
        target_host = step.parameters.get("target_host", "192.168.1.51")
        
        await asyncio.sleep(1.5)
        
        return {
            "success": True,
            "target_host": target_host,
            "shares_accessed": ["C$", "ADMIN$"],
            "method": "SMB",
            "files_accessed": random.randint(5, 20)
        }
    
    async def _execute_data_collection(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data collection from local system"""
        file_types = step.parameters.get("file_types", ["*.doc", "*.pdf", "*.xlsx"])
        
        await asyncio.sleep(1)
        
        collected_files = []
        for file_type in file_types:
            for i in range(random.randint(1, 5)):
                collected_files.append({
                    "filename": f"document_{i}{file_type[1:]}",
                    "size": random.randint(1024, 10485760),  # 1KB to 10MB
                    "path": f"C:\\Users\\user1\\Documents\\document_{i}{file_type[1:]}",
                    "modified": datetime.utcnow().isoformat()
                })
        
        return {
            "success": True,
            "files_collected": len(collected_files),
            "total_size": sum(f["size"] for f in collected_files),
            "files": collected_files
        }
    
    async def _execute_data_exfiltration(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data exfiltration"""
        method = step.parameters.get("method", "https")
        
        await asyncio.sleep(2)
        
        return {
            "success": True,
            "method": method,
            "data_exfiltrated": True,
            "transfer_size": random.randint(1048576, 104857600),  # 1MB to 100MB
            "destination": "attacker-c2.com",
            "encryption": True
        }
    
    async def _execute_ransomware_simulation(self, step: AttackStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ransomware impact (SAFELY - no actual encryption)"""
        target_directories = step.parameters.get("directories", ["Documents", "Desktop"])
        
        await asyncio.sleep(0.5)  # Quick simulation
        
        # SAFETY: This is only a simulation - no actual files are encrypted
        return {
            "success": True,
            "simulation_only": True,
            "target_directories": target_directories,
            "files_that_would_be_encrypted": random.randint(50, 500),
            "ransom_note": "Your files have been encrypted (SIMULATION ONLY)",
            "safety_note": "No actual files were harmed in this simulation"
        }

class AdvancedAttackSimulationEngine(SecurityService if SecurityService else object):
    """Advanced attack simulation engine for sophisticated penetration testing"""
    
    def __init__(self, **kwargs):
        if SecurityService:
            super().__init__(**kwargs)
        
        self.scenarios: Dict[str, SimulationScenario] = {}
        self.active_simulations: Dict[str, SimulationExecution] = {}
        self.simulation_history: List[SimulationExecution] = []
        self.step_executor = AttackStepExecutor()
        self.safety_monitors: List[Callable] = []
        self.emergency_stop = False
        
        # Safety limits
        self.max_concurrent_simulations = 3
        self.max_simulation_duration = 7200  # 2 hours
        self.production_safeguards = True
        
        if hasattr(super(), 'logger'):
            self.logger = super().logger
        else:
            self.logger = logging.getLogger("AdvancedAttackSimulation")
    
    async def initialize(self) -> bool:
        """Initialize the attack simulation engine"""
        try:
            self.logger.info("Initializing Advanced Attack Simulation Engine...")
            
            # Load default scenarios
            await self._load_default_scenarios()
            
            # Initialize safety monitors
            self._initialize_safety_monitors()
            
            self.logger.info(f"Attack Simulation Engine initialized with {len(self.scenarios)} scenarios")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Attack Simulation Engine: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the attack simulation engine"""
        try:
            self.logger.info("Shutting down Attack Simulation Engine...")
            
            # Stop all active simulations
            for execution_id in list(self.active_simulations.keys()):
                await self.stop_simulation(execution_id, "System shutdown")
            
            self.logger.info("Attack Simulation Engine shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown Attack Simulation Engine: {e}")
            return False
    
    async def _load_default_scenarios(self):
        """Load default attack simulation scenarios"""
        
        # APT-style Advanced Persistent Threat scenario
        apt_scenario = SimulationScenario(
            scenario_id="apt_advanced_campaign",
            name="Advanced Persistent Threat Campaign",
            description="Sophisticated multi-stage APT attack simulation",
            attack_chain=[
                AttackStep(
                    step_id="recon_phase",
                    phase=AttackPhase.RECONNAISSANCE,
                    technique=AttackTechnique.NETWORK_SCANNING,
                    name="Network Reconnaissance",
                    description="Conduct stealth network reconnaissance",
                    parameters={"target_range": "10.0.0.0/24", "stealth": True},
                    prerequisites=[],
                    success_criteria=["Hosts discovered", "Services identified"],
                    detection_signatures=["Suspicious network scanning", "Port sweep detected"],
                    cleanup_actions=["Clear scan logs"],
                    estimated_duration=300
                ),
                AttackStep(
                    step_id="initial_compromise",
                    phase=AttackPhase.INITIAL_ACCESS,
                    technique=AttackTechnique.SPEAR_PHISHING,
                    name="Spear Phishing Campaign",
                    description="Targeted phishing attack against key personnel",
                    parameters={"targets": ["admin@target.com", "ceo@target.com"]},
                    prerequisites=["recon_phase"],
                    success_criteria=["Email delivered", "Credentials harvested"],
                    detection_signatures=["Suspicious email attachment", "Credential harvesting"],
                    cleanup_actions=["Remove phishing infrastructure"],
                    estimated_duration=600
                ),
                AttackStep(
                    step_id="establish_foothold",
                    phase=AttackPhase.EXECUTION,
                    technique=AttackTechnique.COMMAND_LINE_INTERFACE,
                    name="Establish Initial Foothold",
                    description="Execute commands to establish presence",
                    parameters={"commands": ["whoami", "net user", "systeminfo"]},
                    prerequisites=["initial_compromise"],
                    success_criteria=["Commands executed", "System information gathered"],
                    detection_signatures=["Suspicious command execution", "Process creation"],
                    cleanup_actions=["Clear command history"],
                    estimated_duration=180
                ),
                AttackStep(
                    step_id="persistence_mechanism",
                    phase=AttackPhase.PERSISTENCE,
                    technique=AttackTechnique.SCHEDULED_TASK,
                    name="Establish Persistence",
                    description="Create persistence mechanism",
                    parameters={"task_name": "WindowsUpdate"},
                    prerequisites=["establish_foothold"],
                    success_criteria=["Scheduled task created", "Persistence verified"],
                    detection_signatures=["Suspicious scheduled task", "Persistence mechanism"],
                    cleanup_actions=["Remove scheduled task"],
                    estimated_duration=120
                ),
                AttackStep(
                    step_id="privilege_escalation",
                    phase=AttackPhase.PRIVILEGE_ESCALATION,
                    technique=AttackTechnique.UAC_BYPASS,
                    name="Escalate Privileges",
                    description="Bypass UAC and gain elevated privileges",
                    parameters={"method": "eventvwr_bypass"},
                    prerequisites=["persistence_mechanism"],
                    success_criteria=["UAC bypassed", "Administrative privileges obtained"],
                    detection_signatures=["UAC bypass attempt", "Privilege escalation"],
                    cleanup_actions=["Restore privilege levels"],
                    estimated_duration=240
                ),
                AttackStep(
                    step_id="credential_harvesting",
                    phase=AttackPhase.CREDENTIAL_ACCESS,
                    technique=AttackTechnique.CREDENTIAL_DUMPING,
                    name="Harvest Credentials",
                    description="Extract credentials from system",
                    parameters={"method": "lsass_dump"},
                    prerequisites=["privilege_escalation"],
                    success_criteria=["Credentials extracted", "Domain accounts identified"],
                    detection_signatures=["LSASS access", "Credential dumping"],
                    cleanup_actions=["Clear credential artifacts"],
                    estimated_duration=300
                ),
                AttackStep(
                    step_id="lateral_movement",
                    phase=AttackPhase.LATERAL_MOVEMENT,
                    technique=AttackTechnique.REMOTE_DESKTOP_PROTOCOL,
                    name="Lateral Movement",
                    description="Move laterally to additional systems",
                    parameters={"target_host": "192.168.1.50"},
                    prerequisites=["credential_harvesting"],
                    success_criteria=["Connected to remote system", "Additional access gained"],
                    detection_signatures=["Lateral movement", "RDP connection"],
                    cleanup_actions=["Close remote connections"],
                    estimated_duration=360
                ),
                AttackStep(
                    step_id="data_collection",
                    phase=AttackPhase.COLLECTION,
                    technique=AttackTechnique.DATA_FROM_LOCAL_SYSTEM,
                    name="Collect Sensitive Data",
                    description="Identify and collect sensitive data",
                    parameters={"file_types": ["*.doc", "*.xlsx", "*.pdf"]},
                    prerequisites=["lateral_movement"],
                    success_criteria=["Sensitive files identified", "Data staged for exfiltration"],
                    detection_signatures=["Bulk file access", "Data staging"],
                    cleanup_actions=["Remove staged data"],
                    estimated_duration=480
                ),
                AttackStep(
                    step_id="exfiltration",
                    phase=AttackPhase.EXFILTRATION,
                    technique=AttackTechnique.EXFILTRATION_OVER_C2,
                    name="Exfiltrate Data",
                    description="Exfiltrate collected data",
                    parameters={"method": "https"},
                    prerequisites=["data_collection"],
                    success_criteria=["Data transmitted", "Exfiltration completed"],
                    detection_signatures=["Data exfiltration", "Suspicious network traffic"],
                    cleanup_actions=["Clear exfiltration traces"],
                    estimated_duration=600
                )
            ],
            complexity=SimulationComplexity.APT_LEVEL,
            target_environment=SimulationEnvironment.ISOLATED,
            objectives=[
                "Test detection capabilities against advanced threats",
                "Validate incident response procedures",
                "Assess data protection controls"
            ],
            constraints=[
                "No actual data exfiltration",
                "No permanent system changes",
                "Maintain business operations"
            ],
            safety_measures=[
                "Continuous monitoring",
                "Emergency stop capability",
                "Automated cleanup",
                "Isolated environment only"
            ],
            estimated_duration=3240,  # Total duration in seconds
            metadata={
                "threat_actor": "APT29",
                "attack_framework": "MITRE ATT&CK",
                "difficulty": "Expert",
                "industry_focus": "Enterprise"
            }
        )
        
        # Basic penetration test scenario
        basic_pentest_scenario = SimulationScenario(
            scenario_id="basic_pentest",
            name="Basic Penetration Test",
            description="Standard penetration testing workflow",
            attack_chain=[
                AttackStep(
                    step_id="network_scan",
                    phase=AttackPhase.RECONNAISSANCE,
                    technique=AttackTechnique.NETWORK_SCANNING,
                    name="Network Scanning",
                    description="Scan network for live hosts and services",
                    parameters={"target_range": "192.168.1.0/24"},
                    prerequisites=[],
                    success_criteria=["Active hosts found", "Open ports identified"],
                    detection_signatures=["Port scan detected"],
                    cleanup_actions=[],
                    estimated_duration=180
                ),
                AttackStep(
                    step_id="service_enum",
                    phase=AttackPhase.DISCOVERY,
                    technique=AttackTechnique.NETWORK_SERVICE_SCANNING,
                    name="Service Enumeration",
                    description="Enumerate services on discovered hosts",
                    parameters={"ports": [21, 22, 80, 443, 445]},
                    prerequisites=["network_scan"],
                    success_criteria=["Services identified", "Versions discovered"],
                    detection_signatures=["Service enumeration"],
                    cleanup_actions=[],
                    estimated_duration=240
                ),
                AttackStep(
                    step_id="web_exploit",
                    phase=AttackPhase.INITIAL_ACCESS,
                    technique=AttackTechnique.EXPLOIT_PUBLIC_APPLICATION,
                    name="Web Application Exploitation",
                    description="Attempt to exploit web applications",
                    parameters={"target_url": "http://target.com/app"},
                    prerequisites=["service_enum"],
                    success_criteria=["Shell access obtained"],
                    detection_signatures=["Web exploitation attempt"],
                    cleanup_actions=["Close shell sessions"],
                    estimated_duration=300
                )
            ],
            complexity=SimulationComplexity.BASIC,
            target_environment=SimulationEnvironment.STAGING,
            objectives=[
                "Identify basic security vulnerabilities",
                "Test perimeter defenses"
            ],
            constraints=[
                "Limited to staging environment",
                "No data modification"
            ],
            safety_measures=[
                "Staging environment only",
                "Limited scope"
            ],
            estimated_duration=720,
            metadata={
                "type": "Basic Penetration Test",
                "scope": "External",
                "methodology": "OWASP"
            }
        )
        
        # Ransomware simulation scenario
        ransomware_scenario = SimulationScenario(
            scenario_id="ransomware_simulation",
            name="Ransomware Attack Simulation",
            description="Simulated ransomware attack for incident response testing",
            attack_chain=[
                AttackStep(
                    step_id="email_delivery",
                    phase=AttackPhase.INITIAL_ACCESS,
                    technique=AttackTechnique.SPEAR_PHISHING,
                    name="Malicious Email Delivery",
                    description="Deliver ransomware via email",
                    parameters={"attachment_type": "macro_document"},
                    prerequisites=[],
                    success_criteria=["Email delivered", "Attachment opened"],
                    detection_signatures=["Malicious email", "Macro execution"],
                    cleanup_actions=["Remove email traces"],
                    estimated_duration=120
                ),
                AttackStep(
                    step_id="execution",
                    phase=AttackPhase.EXECUTION,
                    technique=AttackTechnique.COMMAND_LINE_INTERFACE,
                    name="Ransomware Execution",
                    description="Execute ransomware payload",
                    parameters={"payload_type": "simulated"},
                    prerequisites=["email_delivery"],
                    success_criteria=["Payload executed"],
                    detection_signatures=["Malicious process", "File encryption activity"],
                    cleanup_actions=["Terminate processes"],
                    estimated_duration=60
                ),
                AttackStep(
                    step_id="encryption_simulation",
                    phase=AttackPhase.IMPACT,
                    technique=AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT,
                    name="File Encryption Simulation",
                    description="Simulate file encryption (NO ACTUAL ENCRYPTION)",
                    parameters={"simulation_only": True},
                    prerequisites=["execution"],
                    success_criteria=["Encryption simulation completed"],
                    detection_signatures=["Mass file modification", "Ransom note"],
                    cleanup_actions=["Remove simulation artifacts"],
                    estimated_duration=180,
                    risk_level="high"
                )
            ],
            complexity=SimulationComplexity.INTERMEDIATE,
            target_environment=SimulationEnvironment.ISOLATED,
            objectives=[
                "Test ransomware detection capabilities",
                "Validate backup and recovery procedures",
                "Train incident response team"
            ],
            constraints=[
                "SIMULATION ONLY - no actual encryption",
                "Isolated environment required",
                "No business impact"
            ],
            safety_measures=[
                "No actual file encryption",
                "Isolated test environment",
                "Immediate rollback capability",
                "Continuous safety monitoring"
            ],
            estimated_duration=360,
            metadata={
                "simulation_type": "Ransomware",
                "safety_level": "Maximum",
                "training_purpose": True
            }
        )
        
        # Register scenarios
        self.scenarios[apt_scenario.scenario_id] = apt_scenario
        self.scenarios[basic_pentest_scenario.scenario_id] = basic_pentest_scenario
        self.scenarios[ransomware_scenario.scenario_id] = ransomware_scenario
        
        self.logger.info(f"Loaded {len(self.scenarios)} default attack scenarios")
    
    def _initialize_safety_monitors(self):
        """Initialize safety monitoring systems"""
        self.safety_monitors = [
            self._monitor_production_access,
            self._monitor_simulation_duration,
            self._monitor_system_impact,
            self._monitor_emergency_signals
        ]
    
    async def create_simulation_scenario(self, scenario: SimulationScenario) -> bool:
        """Create a new simulation scenario"""
        try:
            # Validate scenario
            if not self._validate_scenario(scenario):
                return False
            
            self.scenarios[scenario.scenario_id] = scenario
            self.logger.info(f"Created simulation scenario: {scenario.scenario_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create simulation scenario: {e}")
            return False
    
    def _validate_scenario(self, scenario: SimulationScenario) -> bool:
        """Validate simulation scenario"""
        try:
            # Check for required fields
            if not scenario.scenario_id or not scenario.name or not scenario.attack_chain:
                self.logger.error("Scenario missing required fields")
                return False
            
            # Validate attack chain
            step_ids = [step.step_id for step in scenario.attack_chain]
            if len(step_ids) != len(set(step_ids)):
                self.logger.error("Duplicate step IDs in attack chain")
                return False
            
            # Check prerequisites
            for step in scenario.attack_chain:
                for prereq in step.prerequisites:
                    if prereq not in step_ids:
                        self.logger.error(f"Step {step.step_id} has invalid prerequisite: {prereq}")
                        return False
            
            # Safety validation
            if scenario.target_environment == SimulationEnvironment.PRODUCTION and self.production_safeguards:
                self.logger.error("Production environment simulations require explicit safety override")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scenario validation failed: {e}")
            return False
    
    async def start_simulation(self, scenario_id: str, parameters: Dict[str, Any] = None) -> str:
        """Start an attack simulation"""
        try:
            if scenario_id not in self.scenarios:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            if len(self.active_simulations) >= self.max_concurrent_simulations:
                raise ValueError("Maximum concurrent simulations reached")
            
            execution_id = f"sim_{scenario_id}_{uuid.uuid4().hex[:8]}"
            scenario = self.scenarios[scenario_id]
            
            # Safety check
            if not await self._safety_check(scenario, parameters or {}):
                raise ValueError("Safety check failed - simulation not permitted")
            
            # Create simulation execution
            execution = SimulationExecution(
                execution_id=execution_id,
                scenario_id=scenario_id,
                status="starting"
            )
            
            self.active_simulations[execution_id] = execution
            
            # Start simulation in background
            asyncio.create_task(self._execute_simulation(execution_id, parameters or {}))
            
            self.logger.info(f"Started simulation: {execution_id}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to start simulation {scenario_id}: {e}")
            raise
    
    async def _safety_check(self, scenario: SimulationScenario, parameters: Dict[str, Any]) -> bool:
        """Perform comprehensive safety check before simulation"""
        try:
            # Check environment safety
            if scenario.target_environment == SimulationEnvironment.PRODUCTION:
                if not parameters.get("production_override", False):
                    self.logger.error("Production environment requires explicit override")
                    return False
            
            # Check emergency stop status
            if self.emergency_stop:
                self.logger.error("Emergency stop active - no simulations permitted")
                return False
            
            # Check for high-risk steps
            high_risk_steps = [step for step in scenario.attack_chain if step.risk_level == "high"]
            if high_risk_steps and scenario.target_environment != SimulationEnvironment.ISOLATED:
                self.logger.warning(f"High-risk steps detected: {[s.step_id for s in high_risk_steps]}")
                if not parameters.get("risk_acknowledged", False):
                    return False
            
            # Duration check
            if scenario.estimated_duration > self.max_simulation_duration:
                self.logger.error(f"Simulation duration {scenario.estimated_duration}s exceeds limit {self.max_simulation_duration}s")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    async def _execute_simulation(self, execution_id: str, parameters: Dict[str, Any]):
        """Execute a simulation scenario"""
        try:
            execution = self.active_simulations[execution_id]
            scenario = self.scenarios[execution.scenario_id]
            
            execution.status = "running"
            execution.start_time = datetime.utcnow()
            execution.logs.append(f"Simulation started at {execution.start_time}")
            
            # Context for step execution
            context = {
                "execution_id": execution_id,
                "environment": scenario.target_environment,
                "parameters": parameters,
                "completed_techniques": []
            }
            
            # Execute attack chain
            for step in scenario.attack_chain:
                try:
                    # Safety check before each step
                    if await self._check_safety_monitors():
                        execution.safety_triggered = True
                        execution.status = "stopped"
                        execution.logs.append("Safety monitor triggered - simulation stopped")
                        break
                    
                    execution.current_step = step.step_id
                    execution.logs.append(f"Starting step: {step.name}")
                    
                    # Execute step
                    step_result = await self.step_executor.execute_step(step, context)
                    
                    if step_result.get("success", False):
                        execution.completed_steps.append(step.step_id)
                        context["completed_techniques"].append(step.technique.value)
                        
                        if step_result.get("detected", False):
                            execution.detected_steps.append(step.step_id)
                            execution.logs.append(f"Step {step.name} detected by security controls")
                        else:
                            execution.logs.append(f"Step {step.name} completed undetected")
                    else:
                        execution.failed_steps.append(step.step_id)
                        execution.logs.append(f"Step {step.name} failed: {step_result.get('error', 'Unknown error')}")
                        
                        # Check if failure should stop simulation
                        if step.step_id in [s.prerequisites for s in scenario.attack_chain if s.prerequisites]:
                            execution.logs.append("Critical step failed - stopping simulation")
                            break
                    
                    # Store step result
                    execution.results[step.step_id] = step_result
                    
                except Exception as e:
                    execution.failed_steps.append(step.step_id)
                    execution.logs.append(f"Step {step.name} error: {str(e)}")
                    self.logger.error(f"Step execution failed: {e}")
            
            # Finalize execution
            execution.end_time = datetime.utcnow()
            execution.current_step = None
            
            # Determine final status
            if execution.safety_triggered:
                execution.status = "stopped"
            elif len(execution.failed_steps) > len(execution.completed_steps):
                execution.status = "failed"
            else:
                execution.status = "completed"
            
            # Cleanup
            await self._cleanup_simulation(execution_id)
            
            # Generate summary
            execution.results["summary"] = self._generate_simulation_summary(execution, scenario)
            
            # Move to history
            self.simulation_history.append(execution)
            del self.active_simulations[execution_id]
            
            # Limit history size
            if len(self.simulation_history) > 100:
                self.simulation_history = self.simulation_history[-50:]
            
            self.logger.info(f"Simulation {execution_id} completed with status: {execution.status}")
            
        except Exception as e:
            self.logger.error(f"Simulation execution failed: {e}")
            if execution_id in self.active_simulations:
                execution = self.active_simulations[execution_id]
                execution.status = "error"
                execution.end_time = datetime.utcnow()
                execution.logs.append(f"Simulation error: {str(e)}")
                self.simulation_history.append(execution)
                del self.active_simulations[execution_id]
    
    async def _check_safety_monitors(self) -> bool:
        """Check all safety monitors"""
        for monitor in self.safety_monitors:
            if await monitor():
                return True
        return False
    
    async def _monitor_production_access(self) -> bool:
        """Monitor for unauthorized production access"""
        # Implement production system monitoring
        return False
    
    async def _monitor_simulation_duration(self) -> bool:
        """Monitor simulation duration limits"""
        current_time = datetime.utcnow()
        for execution in self.active_simulations.values():
            if execution.start_time:
                duration = (current_time - execution.start_time).total_seconds()
                if duration > self.max_simulation_duration:
                    self.logger.warning(f"Simulation {execution.execution_id} exceeded duration limit")
                    return True
        return False
    
    async def _monitor_system_impact(self) -> bool:
        """Monitor for excessive system impact"""
        # Implement system resource monitoring
        return False
    
    async def _monitor_emergency_signals(self) -> bool:
        """Monitor for emergency stop signals"""
        return self.emergency_stop
    
    async def _cleanup_simulation(self, execution_id: str):
        """Cleanup after simulation execution"""
        try:
            execution = self.active_simulations[execution_id]
            scenario = self.scenarios[execution.scenario_id]
            
            execution.logs.append("Starting cleanup procedures")
            
            # Execute cleanup actions for completed steps
            for step in scenario.attack_chain:
                if step.step_id in execution.completed_steps:
                    for cleanup_action in step.cleanup_actions:
                        try:
                            # Execute cleanup action
                            execution.logs.append(f"Cleanup: {cleanup_action}")
                            await asyncio.sleep(0.1)  # Simulate cleanup time
                        except Exception as e:
                            execution.logs.append(f"Cleanup failed for {cleanup_action}: {str(e)}")
            
            execution.logs.append("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed for simulation {execution_id}: {e}")
    
    def _generate_simulation_summary(self, execution: SimulationExecution, scenario: SimulationScenario) -> Dict[str, Any]:
        """Generate simulation summary report"""
        duration = 0
        if execution.start_time and execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
        
        return {
            "scenario_name": scenario.name,
            "execution_duration": duration,
            "total_steps": len(scenario.attack_chain),
            "completed_steps": len(execution.completed_steps),
            "failed_steps": len(execution.failed_steps),
            "detected_steps": len(execution.detected_steps),
            "success_rate": len(execution.completed_steps) / len(scenario.attack_chain) if scenario.attack_chain else 0,
            "detection_rate": len(execution.detected_steps) / len(execution.completed_steps) if execution.completed_steps else 0,
            "phases_completed": list(set([
                step.phase.value for step in scenario.attack_chain 
                if step.step_id in execution.completed_steps
            ])),
            "techniques_used": list(set([
                step.technique.value for step in scenario.attack_chain 
                if step.step_id in execution.completed_steps
            ])),
            "safety_triggered": execution.safety_triggered,
            "objectives_met": self._assess_objectives(execution, scenario),
            "recommendations": self._generate_recommendations(execution, scenario)
        }
    
    def _assess_objectives(self, execution: SimulationExecution, scenario: SimulationScenario) -> List[str]:
        """Assess which scenario objectives were met"""
        # Simple assessment based on completion rate
        completion_rate = len(execution.completed_steps) / len(scenario.attack_chain)
        
        if completion_rate >= 0.8:
            return scenario.objectives
        elif completion_rate >= 0.5:
            return scenario.objectives[:len(scenario.objectives)//2]
        else:
            return []
    
    def _generate_recommendations(self, execution: SimulationExecution, scenario: SimulationScenario) -> List[str]:
        """Generate security recommendations based on simulation results"""
        recommendations = []
        
        # Detection recommendations
        undetected_steps = [
            step for step in scenario.attack_chain 
            if step.step_id in execution.completed_steps and step.step_id not in execution.detected_steps
        ]
        
        if undetected_steps:
            recommendations.append(f"Improve detection for {len(undetected_steps)} undetected attack techniques")
        
        # Phase-specific recommendations
        completed_phases = set([
            step.phase for step in scenario.attack_chain 
            if step.step_id in execution.completed_steps
        ])
        
        if AttackPhase.INITIAL_ACCESS in completed_phases:
            recommendations.append("Strengthen perimeter defenses and email security")
        
        if AttackPhase.PERSISTENCE in completed_phases:
            recommendations.append("Implement better persistence detection mechanisms")
        
        if AttackPhase.LATERAL_MOVEMENT in completed_phases:
            recommendations.append("Enhance network segmentation and monitoring")
        
        if AttackPhase.EXFILTRATION in completed_phases:
            recommendations.append("Improve data loss prevention controls")
        
        # General recommendations
        recommendations.extend([
            "Conduct regular attack simulation exercises",
            "Update incident response procedures based on findings",
            "Provide security awareness training for identified weaknesses"
        ])
        
        return recommendations
    
    async def stop_simulation(self, execution_id: str, reason: str = "User requested") -> bool:
        """Stop an active simulation"""
        try:
            if execution_id in self.active_simulations:
                execution = self.active_simulations[execution_id]
                execution.status = "stopped"
                execution.end_time = datetime.utcnow()
                execution.logs.append(f"Simulation stopped: {reason}")
                
                # Cleanup
                await self._cleanup_simulation(execution_id)
                
                # Move to history
                self.simulation_history.append(execution)
                del self.active_simulations[execution_id]
                
                self.logger.info(f"Stopped simulation {execution_id}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop simulation {execution_id}: {e}")
            return False
    
    def emergency_stop_all(self, reason: str = "Emergency stop activated"):
        """Emergency stop all simulations"""
        self.emergency_stop = True
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Stop all active simulations
        for execution_id in list(self.active_simulations.keys()):
            asyncio.create_task(self.stop_simulation(execution_id, reason))
    
    def clear_emergency_stop(self):
        """Clear emergency stop status"""
        self.emergency_stop = False
        self.logger.info("Emergency stop cleared")
    
    def get_simulation_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a simulation"""
        # Check active simulations
        if execution_id in self.active_simulations:
            execution = self.active_simulations[execution_id]
            return self._serialize_execution(execution)
        
        # Check history
        for execution in self.simulation_history:
            if execution.execution_id == execution_id:
                return self._serialize_execution(execution)
        
        return None
    
    def _serialize_execution(self, execution: SimulationExecution) -> Dict[str, Any]:
        """Serialize execution for API response"""
        return {
            "execution_id": execution.execution_id,
            "scenario_id": execution.scenario_id,
            "status": execution.status,
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "current_step": execution.current_step,
            "progress": {
                "completed_steps": len(execution.completed_steps),
                "failed_steps": len(execution.failed_steps),
                "detected_steps": len(execution.detected_steps),
                "total_steps": len(self.scenarios[execution.scenario_id].attack_chain)
            },
            "safety_triggered": execution.safety_triggered,
            "logs": execution.logs[-10:] if execution.logs else [],  # Last 10 log entries
            "results_summary": execution.results.get("summary", {})
        }
    
    def list_scenarios(self) -> List[Dict[str, Any]]:
        """List available simulation scenarios"""
        return [
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "complexity": scenario.complexity.value,
                "target_environment": scenario.target_environment.value,
                "estimated_duration": scenario.estimated_duration,
                "steps": len(scenario.attack_chain),
                "phases": list(set([step.phase.value for step in scenario.attack_chain])),
                "objectives": scenario.objectives,
                "safety_measures": scenario.safety_measures
            }
            for scenario in self.scenarios.values()
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation engine metrics"""
        total_simulations = len(self.simulation_history) + len(self.active_simulations)
        completed_simulations = len([s for s in self.simulation_history if s.status == "completed"])
        
        return {
            "total_simulations": total_simulations,
            "active_simulations": len(self.active_simulations),
            "completed_simulations": completed_simulations,
            "failed_simulations": len([s for s in self.simulation_history if s.status == "failed"]),
            "available_scenarios": len(self.scenarios),
            "emergency_stop_active": self.emergency_stop,
            "success_rate": completed_simulations / total_simulations if total_simulations > 0 else 0,
            "average_detection_rate": self._calculate_average_detection_rate()
        }
    
    def _calculate_average_detection_rate(self) -> float:
        """Calculate average detection rate across simulations"""
        detection_rates = []
        for execution in self.simulation_history:
            if execution.completed_steps:
                detection_rate = len(execution.detected_steps) / len(execution.completed_steps)
                detection_rates.append(detection_rate)
        
        return sum(detection_rates) / len(detection_rates) if detection_rates else 0.0
    
    async def health_check(self) -> 'ServiceHealth':
        """Perform health check"""
        try:
            checks = {
                "active_simulations": len(self.active_simulations),
                "available_scenarios": len(self.scenarios),
                "emergency_stop": self.emergency_stop,
                "step_executor_available": self.step_executor is not None,
                "safety_monitors": len(self.safety_monitors)
            }
            
            status = ServiceStatus.HEALTHY if ServiceStatus else "healthy"
            if self.emergency_stop:
                status = ServiceStatus.DEGRADED if ServiceStatus else "degraded"
            
            health = ServiceHealth(
                status=status,
                message="Attack Simulation Engine is operational",
                timestamp=datetime.utcnow(),
                checks=checks
            ) if ServiceHealth else {
                "status": status,
                "message": "Attack Simulation Engine is operational",
                "timestamp": datetime.utcnow(),
                "checks": checks
            }
            
            return health
            
        except Exception as e:
            status = ServiceStatus.UNHEALTHY if ServiceStatus else "unhealthy"
            return ServiceHealth(
                status=status,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            ) if ServiceHealth else {
                "status": status,
                "message": f"Health check failed: {e}",
                "timestamp": datetime.utcnow(),
                "checks": {"error": str(e)}
            }

# Global simulation engine instance
_simulation_engine: Optional[AdvancedAttackSimulationEngine] = None

async def get_simulation_engine() -> AdvancedAttackSimulationEngine:
    """Get global simulation engine instance"""
    global _simulation_engine
    
    if _simulation_engine is None:
        _simulation_engine = AdvancedAttackSimulationEngine()
        await _simulation_engine.initialize()
    
    return _simulation_engine

# Example usage
if __name__ == "__main__":
    async def demo():
        engine = AdvancedAttackSimulationEngine()
        await engine.initialize()
        
        # List available scenarios
        scenarios = engine.list_scenarios()
        print(f"Available scenarios: {len(scenarios)}")
        for scenario in scenarios:
            print(f"- {scenario['name']} ({scenario['complexity']})")
        
        # Start APT simulation
        execution_id = await engine.start_simulation("apt_advanced_campaign")
        print(f"\nStarted APT simulation: {execution_id}")
        
        # Monitor progress
        while True:
            status = engine.get_simulation_status(execution_id)
            if status:
                print(f"Status: {status['status']} - Steps: {status['progress']['completed_steps']}/{status['progress']['total_steps']}")
                if status['status'] in ['completed', 'failed', 'stopped']:
                    break
            await asyncio.sleep(2)
        
        # Get final results
        final_status = engine.get_simulation_status(execution_id)
        print(f"\nFinal Results:")
        print(f"Status: {final_status['status']}")
        print(f"Detection Rate: {final_status['results_summary'].get('detection_rate', 0):.2%}")
        print(f"Recommendations: {final_status['results_summary'].get('recommendations', [])}")
        
        # Get metrics
        metrics = engine.get_metrics()
        print(f"\nEngine Metrics: {json.dumps(metrics, indent=2)}")
        
        await engine.shutdown()
    
    asyncio.run(demo())