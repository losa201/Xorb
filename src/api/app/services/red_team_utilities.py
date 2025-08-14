"""
Red Team Utilities Module
Advanced utilities and helper functions for sophisticated red team operations

This module provides comprehensive utilities for:
- Attack technique analysis and optimization
- Defensive countermeasure generation
- Purple team collaboration features
- Training and education content generation
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import base64
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class TrainingDifficulty(Enum):
    """Training exercise difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DetectionRuleType(Enum):
    """Types of detection rules"""
    SIGMA = "sigma"
    YARA = "yara"
    SNORT = "snort"
    ELASTIC = "elastic"
    SPLUNK = "splunk"
    CUSTOM = "custom"


@dataclass
class DetectionRule:
    """Detection rule for red team techniques"""
    rule_id: str
    name: str
    description: str
    rule_type: DetectionRuleType
    technique_id: str
    rule_content: str
    confidence_level: float
    false_positive_rate: float
    coverage_areas: List[str]
    data_sources: List[str]
    tags: List[str]


@dataclass
class TrainingExercise:
    """Purple team training exercise"""
    exercise_id: str
    name: str
    description: str
    difficulty: TrainingDifficulty
    techniques_covered: List[str]
    learning_objectives: List[str]
    duration_minutes: int
    prerequisites: List[str]
    exercise_steps: List[Dict[str, Any]]
    success_criteria: List[str]
    defensive_focus_areas: List[str]


@dataclass
class ThreatHuntingQuery:
    """Threat hunting query for red team techniques"""
    query_id: str
    name: str
    description: str
    technique_id: str
    query_language: str  # KQL, SPL, SQL, etc.
    query_content: str
    data_sources: List[str]
    false_positive_likelihood: str
    hunting_hypothesis: str
    expected_results: str


class RedTeamUtilities:
    """Comprehensive utilities for red team operations"""

    def __init__(self):
        self.detection_rules_db: Dict[str, DetectionRule] = {}
        self.training_exercises_db: Dict[str, TrainingExercise] = {}
        self.hunting_queries_db: Dict[str, ThreatHuntingQuery] = {}
        self.technique_mappings: Dict[str, Dict[str, Any]] = {}

        # Statistical models for analysis
        self.technique_effectiveness_data: Dict[str, List[float]] = defaultdict(list)
        self.detection_success_rates: Dict[str, float] = {}
        self.evasion_success_rates: Dict[str, float] = {}

    async def initialize(self):
        """Initialize red team utilities"""
        try:
            await self._load_detection_rules_database()
            await self._load_training_exercises()
            await self._load_hunting_queries()
            await self._build_technique_mappings()

            logger.info("Red team utilities initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize red team utilities: {e}")

    async def _load_detection_rules_database(self):
        """Load comprehensive detection rules database"""
        try:
            # Sigma rules for common MITRE techniques
            sigma_rules = {
                'T1566.001': DetectionRule(
                    rule_id="sigma_t1566_001_001",
                    name="Spearphishing Attachment Detection",
                    description="Detects potential spearphishing attachments with suspicious characteristics",
                    rule_type=DetectionRuleType.SIGMA,
                    technique_id="T1566.001",
                    rule_content="""
title: Suspicious Email Attachment Execution
id: 12345678-1234-5678-9012-123456789012
status: experimental
description: Detects execution of suspicious email attachments
author: Red Team Utilities
date: 2025/01/01
references:
    - https://attack.mitre.org/techniques/T1566/001/
tags:
    - attack.initial_access
    - attack.t1566.001
logsource:
    product: windows
    service: sysmon
detection:
    selection:
        EventID: 1
        ParentImage|endswith:
            - '\\\\outlook.exe'
            - '\\\\thunderbird.exe'
        Image|endswith:
            - '*.exe'
            - '*.scr'
            - '*.bat'
            - '*.cmd'
            - '*.com'
            - '*.pif'
    condition: selection
falsepositives:
    - Legitimate software execution from email clients
level: medium
""",
                    confidence_level=0.8,
                    false_positive_rate=0.15,
                    coverage_areas=['email_security', 'endpoint_detection'],
                    data_sources=['Process Creation', 'Email Logs'],
                    tags=['initial_access', 'email_attack']
                ),

                'T1059.001': DetectionRule(
                    rule_id="sigma_t1059_001_001",
                    name="PowerShell Script Execution Detection",
                    description="Detects suspicious PowerShell script execution patterns",
                    rule_type=DetectionRuleType.SIGMA,
                    technique_id="T1059.001",
                    rule_content="""
title: Suspicious PowerShell Execution
id: 87654321-4321-8765-2109-876543210987
status: experimental
description: Detects suspicious PowerShell execution with encoded commands
author: Red Team Utilities
date: 2025/01/01
references:
    - https://attack.mitre.org/techniques/T1059/001/
tags:
    - attack.execution
    - attack.t1059.001
logsource:
    product: windows
    service: powershell
detection:
    selection:
        EventID: 4104
        ScriptBlockText|contains:
            - '-EncodedCommand'
            - '-enc'
            - 'FromBase64String'
            - 'IEX'
            - 'Invoke-Expression'
            - 'DownloadString'
            - 'WebClient'
    condition: selection
falsepositives:
    - Legitimate automation scripts
    - System administration tasks
level: high
""",
                    confidence_level=0.85,
                    false_positive_rate=0.12,
                    coverage_areas=['endpoint_detection', 'script_monitoring'],
                    data_sources=['PowerShell Logs', 'Process Creation'],
                    tags=['execution', 'powershell']
                ),

                'T1055': DetectionRule(
                    rule_id="sigma_t1055_001",
                    name="Process Injection Detection",
                    description="Detects various process injection techniques",
                    rule_type=DetectionRuleType.SIGMA,
                    technique_id="T1055",
                    rule_content="""
title: Process Injection Detection
id: 11111111-2222-3333-4444-555555555555
status: experimental
description: Detects process injection techniques
author: Red Team Utilities
date: 2025/01/01
references:
    - https://attack.mitre.org/techniques/T1055/
tags:
    - attack.privilege_escalation
    - attack.defense_evasion
    - attack.t1055
logsource:
    product: windows
    service: sysmon
detection:
    selection1:
        EventID: 8
        TargetImage|endswith:
            - '\\\\explorer.exe'
            - '\\\\svchost.exe'
            - '\\\\winlogon.exe'
    selection2:
        EventID: 10
        TargetImage|endswith:
            - '\\\\lsass.exe'
        GrantedAccess: '0x1010'
    condition: selection1 or selection2
falsepositives:
    - Legitimate software operations
    - Security software
level: high
""",
                    confidence_level=0.9,
                    false_positive_rate=0.08,
                    coverage_areas=['endpoint_detection', 'memory_analysis'],
                    data_sources=['Process Access', 'Process Creation'],
                    tags=['privilege_escalation', 'injection']
                )
            }

            # YARA rules for malware detection
            yara_rules = {
                'T1204.002': DetectionRule(
                    rule_id="yara_t1204_002_001",
                    name="Malicious User Execution Detection",
                    description="YARA rule for detecting malicious file execution",
                    rule_type=DetectionRuleType.YARA,
                    technique_id="T1204.002",
                    rule_content="""
rule Suspicious_User_Execution {
    meta:
        description = "Detects suspicious files likely to trick users into execution"
        author = "Red Team Utilities"
        date = "2025-01-01"
        technique_id = "T1204.002"

    strings:
        $icon_masquerade1 = { 00 00 01 00 01 00 20 20 00 00 01 00 08 00 } // ICO header
        $icon_masquerade2 = { 00 00 02 00 01 00 } // ICO header variant
        $double_extension1 = ".pdf.exe" ascii nocase
        $double_extension2 = ".doc.exe" ascii nocase
        $double_extension3 = ".jpg.exe" ascii nocase
        $social_eng1 = "URGENT" ascii nocase
        $social_eng2 = "CONFIDENTIAL" ascii nocase
        $social_eng3 = "INVOICE" ascii nocase

    condition:
        (($icon_masquerade1 or $icon_masquerade2) and
         ($double_extension1 or $double_extension2 or $double_extension3)) or
        (2 of ($social_eng*) and ($double_extension1 or $double_extension2 or $double_extension3))
}
""",
                    confidence_level=0.75,
                    false_positive_rate=0.2,
                    coverage_areas=['file_analysis', 'email_security'],
                    data_sources=['File System', 'Email Attachments'],
                    tags=['user_execution', 'social_engineering']
                )
            }

            # Combine all rules
            self.detection_rules_db.update(sigma_rules)
            self.detection_rules_db.update(yara_rules)

            logger.info(f"Loaded {len(self.detection_rules_db)} detection rules")

        except Exception as e:
            logger.error(f"Failed to load detection rules database: {e}")

    async def _load_training_exercises(self):
        """Load purple team training exercises"""
        try:
            exercises = {
                'phishing_simulation': TrainingExercise(
                    exercise_id="exercise_phishing_001",
                    name="Advanced Phishing Simulation Exercise",
                    description="Comprehensive phishing attack simulation with defensive response training",
                    difficulty=TrainingDifficulty.INTERMEDIATE,
                    techniques_covered=["T1566.001", "T1566.002", "T1204.002"],
                    learning_objectives=[
                        "Understand phishing attack vectors",
                        "Implement email security controls",
                        "Develop incident response procedures",
                        "Train users on phishing awareness"
                    ],
                    duration_minutes=120,
                    prerequisites=[
                        "Basic understanding of email security",
                        "Access to email security tools",
                        "Incident response team participation"
                    ],
                    exercise_steps=[
                        {
                            "step": 1,
                            "title": "Red Team: Reconnaissance",
                            "description": "Gather information about target organization's email infrastructure",
                            "duration_minutes": 15,
                            "role": "red_team",
                            "actions": [
                                "Research public email addresses",
                                "Identify email security controls",
                                "Analyze social media for context"
                            ]
                        },
                        {
                            "step": 2,
                            "title": "Red Team: Payload Creation",
                            "description": "Create sophisticated phishing payloads",
                            "duration_minutes": 30,
                            "role": "red_team",
                            "actions": [
                                "Design convincing email templates",
                                "Create safe payload for testing",
                                "Setup tracking mechanisms"
                            ]
                        },
                        {
                            "step": 3,
                            "title": "Red Team: Attack Execution",
                            "description": "Execute controlled phishing campaign",
                            "duration_minutes": 15,
                            "role": "red_team",
                            "actions": [
                                "Send phishing emails to test accounts",
                                "Monitor email delivery and interaction",
                                "Track user responses"
                            ]
                        },
                        {
                            "step": 4,
                            "title": "Blue Team: Detection and Response",
                            "description": "Detect and respond to phishing attack",
                            "duration_minutes": 45,
                            "role": "blue_team",
                            "actions": [
                                "Monitor email security alerts",
                                "Investigate suspicious emails",
                                "Implement containment measures",
                                "Notify affected users"
                            ]
                        },
                        {
                            "step": 5,
                            "title": "Purple Team: Analysis and Learning",
                            "description": "Joint analysis of attack and defense effectiveness",
                            "duration_minutes": 15,
                            "role": "purple_team",
                            "actions": [
                                "Review attack success rates",
                                "Analyze detection effectiveness",
                                "Identify improvement opportunities",
                                "Document lessons learned"
                            ]
                        }
                    ],
                    success_criteria=[
                        "Successful phishing payload delivery",
                        "Detection by email security controls",
                        "Appropriate incident response execution",
                        "Comprehensive post-incident analysis"
                    ],
                    defensive_focus_areas=[
                        "Email security configuration",
                        "User awareness training",
                        "Incident response procedures",
                        "Detection rule optimization"
                    ]
                ),

                'lateral_movement': TrainingExercise(
                    exercise_id="exercise_lateral_001",
                    name="Lateral Movement Detection Exercise",
                    description="Advanced lateral movement simulation with network monitoring",
                    difficulty=TrainingDifficulty.ADVANCED,
                    techniques_covered=["T1021.001", "T1021.002", "T1078", "T1550.002"],
                    learning_objectives=[
                        "Understand lateral movement techniques",
                        "Implement network segmentation",
                        "Develop monitoring strategies",
                        "Create detection rules"
                    ],
                    duration_minutes=180,
                    prerequisites=[
                        "Network security knowledge",
                        "Access to network monitoring tools",
                        "Understanding of Windows authentication"
                    ],
                    exercise_steps=[
                        {
                            "step": 1,
                            "title": "Environment Setup",
                            "description": "Prepare controlled network environment",
                            "duration_minutes": 30,
                            "role": "infrastructure",
                            "actions": [
                                "Setup isolated network segments",
                                "Configure monitoring tools",
                                "Prepare target systems"
                            ]
                        },
                        {
                            "step": 2,
                            "title": "Red Team: Initial Access",
                            "description": "Establish initial foothold in network",
                            "duration_minutes": 30,
                            "role": "red_team",
                            "actions": [
                                "Simulate initial compromise",
                                "Establish command and control",
                                "Begin reconnaissance"
                            ]
                        },
                        {
                            "step": 3,
                            "title": "Red Team: Credential Harvesting",
                            "description": "Harvest credentials for lateral movement",
                            "duration_minutes": 45,
                            "role": "red_team",
                            "actions": [
                                "Dump credentials from memory",
                                "Extract cached credentials",
                                "Crack password hashes"
                            ]
                        },
                        {
                            "step": 4,
                            "title": "Red Team: Lateral Movement",
                            "description": "Move laterally through the network",
                            "duration_minutes": 45,
                            "role": "red_team",
                            "actions": [
                                "Use RDP for lateral movement",
                                "Execute WMI commands remotely",
                                "Access file shares"
                            ]
                        },
                        {
                            "step": 5,
                            "title": "Blue Team: Detection and Hunting",
                            "description": "Hunt for lateral movement indicators",
                            "duration_minutes": 30,
                            "role": "blue_team",
                            "actions": [
                                "Monitor authentication logs",
                                "Analyze network traffic",
                                "Hunt for suspicious processes"
                            ]
                        }
                    ],
                    success_criteria=[
                        "Successful lateral movement execution",
                        "Detection of movement patterns",
                        "Effective incident containment",
                        "Creation of new detection rules"
                    ],
                    defensive_focus_areas=[
                        "Network segmentation",
                        "Privileged access management",
                        "Authentication monitoring",
                        "Behavioral analytics"
                    ]
                )
            }

            self.training_exercises_db.update(exercises)

            logger.info(f"Loaded {len(self.training_exercises_db)} training exercises")

        except Exception as e:
            logger.error(f"Failed to load training exercises: {e}")

    async def _load_hunting_queries(self):
        """Load threat hunting queries"""
        try:
            queries = {
                'powershell_hunting': ThreatHuntingQuery(
                    query_id="hunt_powershell_001",
                    name="Suspicious PowerShell Activity Hunt",
                    description="Hunt for suspicious PowerShell execution patterns",
                    technique_id="T1059.001",
                    query_language="KQL",
                    query_content="""
// Hunt for suspicious PowerShell activity
SecurityEvent
| where TimeGenerated > ago(7d)
| where EventID == 4688
| where Process has "powershell.exe"
| where CommandLine has_any ("IEX", "Invoke-Expression", "DownloadString",
                            "FromBase64String", "-EncodedCommand", "-enc")
| extend SuspiciousScore = case(
    CommandLine has "IEX" and CommandLine has "DownloadString", 3,
    CommandLine has "FromBase64String", 2,
    CommandLine has "-EncodedCommand", 2,
    1)
| where SuspiciousScore >= 2
| summarize Count = count(),
          UniqueCommands = dcount(CommandLine),
          FirstSeen = min(TimeGenerated),
          LastSeen = max(TimeGenerated)
          by Computer, Account, SuspiciousScore
| order by SuspiciousScore desc, Count desc
""",
                    data_sources=["Security Events", "Process Creation"],
                    false_positive_likelihood="Medium",
                    hunting_hypothesis="Adversaries use PowerShell for post-exploitation activities with obfuscated commands",
                    expected_results="Identify potential malicious PowerShell usage with encoded or suspicious commands"
                ),

                'lateral_movement_hunting': ThreatHuntingQuery(
                    query_id="hunt_lateral_001",
                    name="Lateral Movement Detection Hunt",
                    description="Hunt for lateral movement patterns using authentication logs",
                    technique_id="T1021.001",
                    query_language="KQL",
                    query_content="""
// Hunt for lateral movement patterns
SecurityEvent
| where TimeGenerated > ago(24h)
| where EventID in (4624, 4625)  // Successful and failed logons
| where LogonType in (3, 10)     // Network and RDP logons
| where Account !has "$"         // Exclude computer accounts
| summarize SuccessfulLogons = countif(EventID == 4624),
          FailedLogons = countif(EventID == 4625),
          UniqueWorkstations = dcount(WorkstationName),
          LogonTimes = make_list(TimeGenerated)
          by Account
| where UniqueWorkstations >= 5  // Account logged into 5+ systems
| extend RiskScore = case(
    UniqueWorkstations >= 10, 3,
    UniqueWorkstations >= 7, 2,
    1)
| order by RiskScore desc, UniqueWorkstations desc
""",
                    data_sources=["Authentication Logs", "Security Events"],
                    false_positive_likelihood="Low",
                    hunting_hypothesis="Lateral movement involves authentication to multiple systems in short timeframes",
                    expected_results="Identify accounts authenticating to unusually high numbers of systems"
                ),

                'persistence_hunting': ThreatHuntingQuery(
                    query_id="hunt_persistence_001",
                    name="Persistence Mechanism Hunt",
                    description="Hunt for common persistence mechanisms",
                    technique_id="T1547.001",
                    query_language="SPL",
                    query_content="""
index=windows source="WinEventLog:Microsoft-Windows-Sysmon/Operational"
EventCode=13
(TargetObject="*\\Run\\*" OR TargetObject="*\\RunOnce\\*" OR
 TargetObject="*\\Winlogon\\*" OR TargetObject="*\\Services\\*")
| eval persistence_type=case(
    match(TargetObject, ".*\\\\Run\\\\.*"), "Registry Run Key",
    match(TargetObject, ".*\\\\RunOnce\\\\.*"), "Registry RunOnce Key",
    match(TargetObject, ".*\\\\Winlogon\\\\.*"), "Winlogon Helper",
    match(TargetObject, ".*\\\\Services\\\\.*"), "Service Creation",
    "Unknown")
| stats count by host, persistence_type, TargetObject, Details
| where count > 1
| sort - count
""",
                    data_sources=["Sysmon", "Registry Events"],
                    false_positive_likelihood="Medium",
                    hunting_hypothesis="Adversaries establish persistence through registry modifications",
                    expected_results="Identify unusual registry modifications that could indicate persistence"
                )
            }

            self.hunting_queries_db.update(queries)

            logger.info(f"Loaded {len(self.hunting_queries_db)} hunting queries")

        except Exception as e:
            logger.error(f"Failed to load hunting queries: {e}")

    async def _build_technique_mappings(self):
        """Build comprehensive technique mappings"""
        try:
            # Map techniques to detection methods, countermeasures, and training
            self.technique_mappings = {
                'T1566.001': {
                    'technique_name': 'Spearphishing Attachment',
                    'detection_rules': ['sigma_t1566_001_001'],
                    'hunting_queries': [],
                    'training_exercises': ['exercise_phishing_001'],
                    'countermeasures': [
                        'Email security gateways',
                        'Attachment scanning',
                        'User awareness training',
                        'Email authentication (SPF, DKIM, DMARC)'
                    ],
                    'data_sources': [
                        'Email logs',
                        'File system monitoring',
                        'Process creation'
                    ],
                    'difficulty_to_detect': 'Medium',
                    'common_evasions': [
                        'File format obfuscation',
                        'Social engineering tactics',
                        'Zero-day exploits'
                    ]
                },
                'T1059.001': {
                    'technique_name': 'PowerShell',
                    'detection_rules': ['sigma_t1059_001_001'],
                    'hunting_queries': ['hunt_powershell_001'],
                    'training_exercises': [],
                    'countermeasures': [
                        'PowerShell logging',
                        'Execution policy enforcement',
                        'Script block logging',
                        'Application whitelisting'
                    ],
                    'data_sources': [
                        'PowerShell logs',
                        'Process creation',
                        'Script block text'
                    ],
                    'difficulty_to_detect': 'Medium',
                    'common_evasions': [
                        'Base64 encoding',
                        'String obfuscation',
                        'Invoke-Expression alternatives'
                    ]
                },
                'T1055': {
                    'technique_name': 'Process Injection',
                    'detection_rules': ['sigma_t1055_001'],
                    'hunting_queries': [],
                    'training_exercises': [],
                    'countermeasures': [
                        'Memory protection',
                        'Control Flow Guard',
                        'Behavioral analysis',
                        'API monitoring'
                    ],
                    'data_sources': [
                        'Process access',
                        'Process modification',
                        'Memory analysis'
                    ],
                    'difficulty_to_detect': 'High',
                    'common_evasions': [
                        'Process hollowing',
                        'DLL injection',
                        'Thread hijacking'
                    ]
                },
                'T1021.001': {
                    'technique_name': 'Remote Desktop Protocol',
                    'detection_rules': [],
                    'hunting_queries': ['hunt_lateral_001'],
                    'training_exercises': ['exercise_lateral_001'],
                    'countermeasures': [
                        'Network segmentation',
                        'Multi-factor authentication',
                        'RDP security hardening',
                        'Connection monitoring'
                    ],
                    'data_sources': [
                        'Authentication logs',
                        'Network traffic',
                        'Remote desktop logs'
                    ],
                    'difficulty_to_detect': 'Low',
                    'common_evasions': [
                        'Legitimate credential use',
                        'Off-hours access',
                        'Tunneling through other protocols'
                    ]
                }
            }

            logger.info(f"Built technique mappings for {len(self.technique_mappings)} techniques")

        except Exception as e:
            logger.error(f"Failed to build technique mappings: {e}")

    async def generate_detection_rule(self, technique_id: str, rule_type: DetectionRuleType = DetectionRuleType.SIGMA) -> Optional[DetectionRule]:
        """Generate detection rule for a specific technique"""
        try:
            if technique_id not in self.technique_mappings:
                logger.warning(f"No mapping found for technique {technique_id}")
                return None

            technique_info = self.technique_mappings[technique_id]

            # Check if rule already exists
            existing_rules = [
                rule_id for rule_id in technique_info.get('detection_rules', [])
                if self.detection_rules_db.get(rule_id, {}).rule_type == rule_type
            ]

            if existing_rules:
                return self.detection_rules_db[existing_rules[0]]

            # Generate new rule based on technique characteristics
            if rule_type == DetectionRuleType.SIGMA:
                return await self._generate_sigma_rule(technique_id, technique_info)
            elif rule_type == DetectionRuleType.YARA:
                return await self._generate_yara_rule(technique_id, technique_info)
            else:
                logger.warning(f"Rule generation not implemented for type {rule_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate detection rule for {technique_id}: {e}")
            return None

    async def _generate_sigma_rule(self, technique_id: str, technique_info: Dict[str, Any]) -> DetectionRule:
        """Generate Sigma detection rule"""
        try:
            technique_name = technique_info.get('technique_name', 'Unknown')
            data_sources = technique_info.get('data_sources', [])

            # Template-based rule generation
            rule_content = f"""
title: {technique_name} Detection
id: {hashlib.md5(f"{technique_id}_auto_generated".encode()).hexdigest()}
status: experimental
description: Auto-generated detection rule for {technique_name}
author: Red Team Utilities - Auto-generated
date: {datetime.now().strftime('%Y/%m/%d')}
references:
    - https://attack.mitre.org/techniques/{technique_id}/
tags:
    - attack.{technique_id.lower()}
logsource:
    product: windows
    service: security
detection:
    selection:
        # Auto-generated selection criteria based on technique
        EventID: [4688, 4689]  # Process creation/termination
    condition: selection
falsepositives:
    - Legitimate administrative activities
    - Normal software operations
level: medium
"""

            return DetectionRule(
                rule_id=f"auto_gen_{technique_id}_{datetime.now().strftime('%Y%m%d')}",
                name=f"{technique_name} Auto-Generated Detection",
                description=f"Auto-generated detection rule for MITRE technique {technique_id}",
                rule_type=DetectionRuleType.SIGMA,
                technique_id=technique_id,
                rule_content=rule_content.strip(),
                confidence_level=0.6,  # Lower confidence for auto-generated
                false_positive_rate=0.25,
                coverage_areas=['endpoint_detection'],
                data_sources=data_sources,
                tags=['auto_generated', technique_id.lower()]
            )

        except Exception as e:
            logger.error(f"Failed to generate Sigma rule: {e}")
            raise

    async def generate_training_exercise(self, technique_ids: List[str], difficulty: TrainingDifficulty = TrainingDifficulty.INTERMEDIATE) -> TrainingExercise:
        """Generate custom training exercise for specific techniques"""
        try:
            # Combine technique information
            covered_techniques = []
            learning_objectives = []
            defensive_focus_areas = set()

            for technique_id in technique_ids:
                if technique_id in self.technique_mappings:
                    technique_info = self.technique_mappings[technique_id]
                    covered_techniques.append(technique_id)
                    learning_objectives.append(f"Understand and detect {technique_info.get('technique_name', technique_id)}")
                    defensive_focus_areas.update(technique_info.get('countermeasures', []))

            # Generate exercise steps
            exercise_steps = [
                {
                    "step": 1,
                    "title": "Environment Preparation",
                    "description": "Setup controlled testing environment",
                    "duration_minutes": 20,
                    "role": "infrastructure",
                    "actions": [
                        "Configure monitoring tools",
                        "Prepare target systems",
                        "Setup logging and detection"
                    ]
                },
                {
                    "step": 2,
                    "title": "Red Team: Technique Execution",
                    "description": f"Execute techniques: {', '.join(covered_techniques)}",
                    "duration_minutes": 60,
                    "role": "red_team",
                    "actions": [
                        f"Implement technique {tid}" for tid in covered_techniques[:3]  # Limit for readability
                    ]
                },
                {
                    "step": 3,
                    "title": "Blue Team: Detection and Response",
                    "description": "Detect and respond to red team activities",
                    "duration_minutes": 45,
                    "role": "blue_team",
                    "actions": [
                        "Monitor for technique indicators",
                        "Analyze detection alerts",
                        "Implement response procedures"
                    ]
                },
                {
                    "step": 4,
                    "title": "Purple Team: Joint Analysis",
                    "description": "Collaborative analysis and improvement",
                    "duration_minutes": 30,
                    "role": "purple_team",
                    "actions": [
                        "Review detection effectiveness",
                        "Identify gaps and improvements",
                        "Document lessons learned"
                    ]
                }
            ]

            return TrainingExercise(
                exercise_id=f"auto_gen_exercise_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Multi-Technique Training: {', '.join(covered_techniques[:2])}{'...' if len(covered_techniques) > 2 else ''}",
                description=f"Custom training exercise covering {len(covered_techniques)} MITRE ATT&CK techniques",
                difficulty=difficulty,
                techniques_covered=covered_techniques,
                learning_objectives=learning_objectives,
                duration_minutes=155,  # Sum of step durations
                prerequisites=[
                    "Basic understanding of MITRE ATT&CK framework",
                    "Access to testing environment",
                    "Red and blue team participation"
                ],
                exercise_steps=exercise_steps,
                success_criteria=[
                    "Successful technique execution",
                    "Effective detection and response",
                    "Comprehensive analysis and documentation"
                ],
                defensive_focus_areas=list(defensive_focus_areas)[:5]  # Limit for readability
            )

        except Exception as e:
            logger.error(f"Failed to generate training exercise: {e}")
            raise

    async def analyze_technique_effectiveness(self, technique_id: str, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effectiveness of a red team technique"""
        try:
            analysis = {
                'technique_id': technique_id,
                'execution_timestamp': datetime.now().isoformat(),
                'success_rate': 0.0,
                'detection_rate': 0.0,
                'evasion_effectiveness': 0.0,
                'defensive_recommendations': [],
                'improvement_opportunities': []
            }

            # Calculate success rate
            if 'success_indicators' in execution_results:
                successful_attempts = len(execution_results['success_indicators'])
                total_attempts = execution_results.get('total_attempts', 1)
                analysis['success_rate'] = successful_attempts / total_attempts

            # Calculate detection rate
            if 'detection_events' in execution_results:
                detected_attempts = len(execution_results['detection_events'])
                total_attempts = execution_results.get('total_attempts', 1)
                analysis['detection_rate'] = detected_attempts / total_attempts

            # Calculate evasion effectiveness
            analysis['evasion_effectiveness'] = 1.0 - analysis['detection_rate']

            # Generate recommendations based on results
            if analysis['detection_rate'] < 0.5:
                analysis['defensive_recommendations'].append({
                    'type': 'detection_improvement',
                    'priority': 'high',
                    'description': f"Low detection rate ({analysis['detection_rate']:.2%}) for technique {technique_id}",
                    'recommendations': [
                        'Implement additional detection rules',
                        'Enhance monitoring coverage',
                        'Review data source collection'
                    ]
                })

            if analysis['success_rate'] > 0.8:
                analysis['improvement_opportunities'].append({
                    'type': 'prevention_enhancement',
                    'priority': 'medium',
                    'description': f"High success rate ({analysis['success_rate']:.2%}) indicates weak preventive controls",
                    'recommendations': [
                        'Implement preventive security controls',
                        'Harden system configurations',
                        'Review access controls'
                    ]
                })

            # Store historical data for trend analysis
            self.technique_effectiveness_data[technique_id].append(analysis['success_rate'])
            self.detection_success_rates[technique_id] = analysis['detection_rate']

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze technique effectiveness: {e}")
            return {'error': str(e)}

    async def generate_purple_team_report(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive purple team collaboration report"""
        try:
            report = {
                'report_id': str(uuid.uuid4()),
                'generation_timestamp': datetime.now().isoformat(),
                'executive_summary': {},
                'technical_analysis': {},
                'defensive_improvements': [],
                'training_recommendations': [],
                'metrics': {},
                'next_steps': []
            }

            # Executive Summary
            techniques_executed = len(operation_results.get('techniques_used', []))
            successful_techniques = len([t for t in operation_results.get('techniques_used', []) if t.get('success', False)])
            detected_techniques = len([t for t in operation_results.get('techniques_used', []) if t.get('detection_triggered', False)])

            report['executive_summary'] = {
                'operation_overview': f"Executed {techniques_executed} techniques with {successful_techniques} successes and {detected_techniques} detections",
                'key_findings': [
                    f"Success rate: {(successful_techniques/max(techniques_executed,1)):.1%}",
                    f"Detection rate: {(detected_techniques/max(techniques_executed,1)):.1%}",
                    f"Defensive gaps identified: {techniques_executed - detected_techniques}"
                ],
                'overall_security_posture': self._assess_security_posture(successful_techniques, detected_techniques, techniques_executed)
            }

            # Technical Analysis
            report['technical_analysis'] = {
                'technique_breakdown': operation_results.get('techniques_used', []),
                'detection_analysis': operation_results.get('detection_events', []),
                'evasion_analysis': [
                    t for t in operation_results.get('techniques_used', [])
                    if t.get('success', False) and not t.get('detection_triggered', False)
                ]
            }

            # Defensive Improvements
            undetected_techniques = [
                t for t in operation_results.get('techniques_used', [])
                if t.get('success', False) and not t.get('detection_triggered', False)
            ]

            for technique in undetected_techniques:
                technique_id = technique.get('technique_id', 'Unknown')
                if technique_id in self.technique_mappings:
                    mapping = self.technique_mappings[technique_id]
                    improvement = {
                        'technique_id': technique_id,
                        'technique_name': mapping.get('technique_name', 'Unknown'),
                        'priority': 'high',
                        'recommended_controls': mapping.get('countermeasures', []),
                        'detection_rules': mapping.get('detection_rules', []),
                        'implementation_effort': 'medium'
                    }
                    report['defensive_improvements'].append(improvement)

            # Training Recommendations
            report['training_recommendations'] = await self._generate_training_recommendations(operation_results)

            # Metrics
            report['metrics'] = {
                'techniques_executed': techniques_executed,
                'success_rate': successful_techniques / max(techniques_executed, 1),
                'detection_rate': detected_techniques / max(techniques_executed, 1),
                'mean_time_to_detection': self._calculate_mean_time_to_detection(operation_results),
                'false_positive_rate': self._estimate_false_positive_rate(operation_results)
            }

            # Next Steps
            report['next_steps'] = [
                'Implement high-priority defensive improvements',
                'Conduct targeted training on undetected techniques',
                'Schedule follow-up purple team exercise',
                'Review and update detection rules',
                'Enhance monitoring capabilities'
            ]

            return report

        except Exception as e:
            logger.error(f"Failed to generate purple team report: {e}")
            return {'error': str(e)}

    def _assess_security_posture(self, successful: int, detected: int, total: int) -> str:
        """Assess overall security posture based on metrics"""
        if total == 0:
            return "Unable to assess - no techniques executed"

        detection_rate = detected / total
        success_rate = successful / total

        if detection_rate >= 0.8 and success_rate <= 0.3:
            return "Strong - High detection rate with low attack success"
        elif detection_rate >= 0.6 and success_rate <= 0.5:
            return "Good - Adequate detection with moderate attack success"
        elif detection_rate >= 0.4 and success_rate <= 0.7:
            return "Fair - Some detection gaps and moderate vulnerability"
        else:
            return "Needs Improvement - Significant detection gaps and high attack success"

    async def _generate_training_recommendations(self, operation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate training recommendations based on operation results"""
        recommendations = []

        try:
            # Analyze which techniques were most successful
            successful_techniques = [
                t.get('technique_id') for t in operation_results.get('techniques_used', [])
                if t.get('success', False) and not t.get('detection_triggered', False)
            ]

            # Group by technique categories
            technique_categories = defaultdict(list)
            for technique_id in successful_techniques:
                if technique_id in self.technique_mappings:
                    # Categorize by MITRE tactic (simplified)
                    if technique_id.startswith('T1566'):
                        technique_categories['Initial Access'].append(technique_id)
                    elif technique_id.startswith('T1059'):
                        technique_categories['Execution'].append(technique_id)
                    elif technique_id.startswith('T1055'):
                        technique_categories['Defense Evasion'].append(technique_id)
                    else:
                        technique_categories['Other'].append(technique_id)

            # Generate category-specific recommendations
            for category, techniques in technique_categories.items():
                if len(techniques) >= 2:  # Focus on categories with multiple gaps
                    recommendation = {
                        'training_type': 'focused_workshop',
                        'category': category,
                        'priority': 'high',
                        'target_audience': ['security_analysts', 'incident_responders'],
                        'techniques_covered': techniques,
                        'estimated_duration_hours': 4,
                        'learning_objectives': [
                            f'Understand {category.lower()} techniques',
                            f'Implement detection for {category.lower()} attacks',
                            f'Develop response procedures for {category.lower()} incidents'
                        ]
                    }
                    recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate training recommendations: {e}")
            return []

    def _calculate_mean_time_to_detection(self, operation_results: Dict[str, Any]) -> Optional[float]:
        """Calculate mean time to detection for detected techniques"""
        try:
            detection_times = []

            for event in operation_results.get('detection_events', []):
                if 'timestamp' in event:
                    # In a real implementation, this would calculate the time difference
                    # between technique execution and detection
                    detection_times.append(300)  # Placeholder: 5 minutes

            if detection_times:
                return sum(detection_times) / len(detection_times)

            return None

        except Exception as e:
            logger.error(f"Failed to calculate mean time to detection: {e}")
            return None

    def _estimate_false_positive_rate(self, operation_results: Dict[str, Any]) -> float:
        """Estimate false positive rate based on detection events"""
        try:
            # In a real implementation, this would analyze actual vs expected detections
            # For now, return a reasonable estimate based on detection complexity

            detection_events = operation_results.get('detection_events', [])
            if not detection_events:
                return 0.0

            # Estimate based on number of detection events vs expected
            # This is a simplified calculation for demonstration
            return min(0.15, len(detection_events) * 0.02)  # Cap at 15%

        except Exception as e:
            logger.error(f"Failed to estimate false positive rate: {e}")
            return 0.0

    async def get_technique_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about red team techniques"""
        try:
            stats = {
                'total_techniques': len(self.technique_mappings),
                'detection_rules_available': len(self.detection_rules_db),
                'hunting_queries_available': len(self.hunting_queries_db),
                'training_exercises_available': len(self.training_exercises_db),
                'technique_effectiveness': {},
                'detection_coverage': {},
                'most_effective_techniques': [],
                'hardest_to_detect_techniques': []
            }

            # Calculate technique effectiveness statistics
            for technique_id, data in self.technique_effectiveness_data.items():
                if data:
                    stats['technique_effectiveness'][technique_id] = {
                        'mean_success_rate': np.mean(data),
                        'std_success_rate': np.std(data),
                        'execution_count': len(data)
                    }

            # Calculate detection coverage
            for technique_id, mapping in self.technique_mappings.items():
                detection_rules = len(mapping.get('detection_rules', []))
                hunting_queries = len(mapping.get('hunting_queries', []))
                training_exercises = len(mapping.get('training_exercises', []))

                coverage_score = (detection_rules * 0.4 + hunting_queries * 0.3 + training_exercises * 0.3)
                stats['detection_coverage'][technique_id] = {
                    'coverage_score': coverage_score,
                    'detection_rules': detection_rules,
                    'hunting_queries': hunting_queries,
                    'training_exercises': training_exercises
                }

            # Identify most effective and hardest to detect techniques
            if self.detection_success_rates:
                sorted_by_detection = sorted(
                    self.detection_success_rates.items(),
                    key=lambda x: x[1]
                )
                stats['hardest_to_detect_techniques'] = [
                    {'technique_id': tid, 'detection_rate': rate}
                    for tid, rate in sorted_by_detection[:5]
                ]

            return stats

        except Exception as e:
            logger.error(f"Failed to get technique statistics: {e}")
            return {'error': str(e)}


# Global utilities instance
_red_team_utilities: Optional[RedTeamUtilities] = None

async def get_red_team_utilities() -> RedTeamUtilities:
    """Get singleton instance of red team utilities"""
    global _red_team_utilities

    if _red_team_utilities is None:
        _red_team_utilities = RedTeamUtilities()
        await _red_team_utilities.initialize()

    return _red_team_utilities
