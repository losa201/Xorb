"""
MITRE ATT&CK Framework Integration - Production Implementation
Real-world threat intelligence using official MITRE ATT&CK framework data
"""

import asyncio
import json
import logging
import aiohttp
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class AttackTechnique:
    """MITRE ATT&CK Technique representation"""
    id: str  # T1566.001
    name: str
    description: str
    tactic: str
    platform: List[str]
    data_sources: List[str]
    detection: str
    mitigations: List[str]
    references: List[str]
    kill_chain_phases: List[str]
    x_mitre_version: str
    revoked: bool = False
    deprecated: bool = False

@dataclass
class AttackGroup:
    """MITRE ATT&CK Group/Actor representation"""
    id: str  # G0001
    name: str
    aliases: List[str]
    description: str
    techniques: List[str]
    software: List[str]
    references: List[str]
    associated_campaigns: List[str]

@dataclass
class AttackSoftware:
    """MITRE ATT&CK Software/Malware representation"""
    id: str  # S0001
    name: str
    type: str  # malware, tool
    description: str
    aliases: List[str]
    platforms: List[str]
    techniques: List[str]
    references: List[str]

@dataclass
class ThreatMapping:
    """Maps detected threats to MITRE ATT&CK techniques"""
    threat_id: str
    technique_ids: List[str]
    confidence: float
    evidence: List[str]
    timestamp: datetime
    source: str

class MITREAttackFramework:
    """Production MITRE ATT&CK Framework integration"""
    
    def __init__(self, data_path: str = "./data/mitre_attack"):
        self.data_path = Path(data_path)
        self.db_path = self.data_path / "mitre_attack.db"
        self.techniques: Dict[str, AttackTechnique] = {}
        self.groups: Dict[str, AttackGroup] = {}
        self.software: Dict[str, AttackSoftware] = {}
        self.tactics: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # MITRE ATT&CK official data sources
        self.data_sources = {
            "enterprise": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "mobile": "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json",
            "ics": "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"
        }
        
        # Technique detection patterns
        self.detection_patterns = {}
        self.threat_mappings: List[ThreatMapping] = []
        
    async def initialize(self) -> bool:
        """Initialize MITRE ATT&CK framework with latest data"""
        try:
            logger.info("Initializing MITRE ATT&CK Framework integration...")
            
            # Create data directory
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Initialize database
            await self._initialize_database()
            
            # Load or update MITRE data
            await self._load_mitre_data()
            
            # Load detection patterns
            await self._load_detection_patterns()
            
            # Validate data integrity
            await self._validate_data_integrity()
            
            logger.info(f"MITRE ATT&CK Framework initialized with {len(self.techniques)} techniques")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MITRE ATT&CK Framework: {e}")
            return False
    
    async def _initialize_database(self):
        """Initialize SQLite database for MITRE data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS techniques (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tactic TEXT,
                platform TEXT,
                data_sources TEXT,
                detection TEXT,
                mitigations TEXT,
                references TEXT,
                kill_chain_phases TEXT,
                version TEXT,
                revoked BOOLEAN DEFAULT FALSE,
                deprecated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                techniques TEXT,
                software TEXT,
                references TEXT,
                campaigns TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS software (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                aliases TEXT,
                platforms TEXT,
                techniques TEXT,
                references TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_mappings (
                id TEXT PRIMARY KEY,
                threat_id TEXT NOT NULL,
                technique_ids TEXT,
                confidence REAL,
                evidence TEXT,
                source TEXT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_techniques_tactic ON techniques(tactic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_techniques_platform ON techniques(platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_threat_mappings_threat_id ON threat_mappings(threat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_threat_mappings_timestamp ON threat_mappings(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info("MITRE ATT&CK database initialized")
    
    async def _load_mitre_data(self):
        """Load MITRE ATT&CK data from official sources"""
        for domain, url in self.data_sources.items():
            try:
                logger.info(f"Loading MITRE ATT&CK {domain} data...")
                
                # Check if we have recent cached data
                cache_file = self.data_path / f"{domain}_attack.json"
                if cache_file.exists():
                    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_age < timedelta(days=7):  # Use cache if less than 7 days old
                        logger.info(f"Using cached {domain} data")
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        await self._parse_mitre_data(data, domain)
                        continue
                
                # Download fresh data
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache the data
                        with open(cache_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        await self._parse_mitre_data(data, domain)
                        logger.info(f"Successfully loaded {domain} MITRE ATT&CK data")
                    else:
                        logger.error(f"Failed to download {domain} data: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"Error loading {domain} MITRE data: {e}")
                # Try to load from cache as fallback
                cache_file = self.data_path / f"{domain}_attack.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        await self._parse_mitre_data(data, domain)
                        logger.info(f"Loaded {domain} data from cache as fallback")
                    except Exception as cache_error:
                        logger.error(f"Failed to load {domain} cache: {cache_error}")
    
    async def _parse_mitre_data(self, data: Dict[str, Any], domain: str):
        """Parse MITRE ATT&CK JSON data"""
        objects = data.get("objects", [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        techniques_count = 0
        groups_count = 0
        software_count = 0
        
        for obj in objects:
            try:
                obj_type = obj.get("type", "")
                
                if obj_type == "attack-pattern":
                    technique = await self._parse_technique(obj, domain)
                    if technique:
                        self.techniques[technique.id] = technique
                        
                        # Store in database
                        cursor.execute("""
                            INSERT OR REPLACE INTO techniques 
                            (id, name, description, tactic, platform, data_sources, detection, mitigations, references, kill_chain_phases, version, revoked, deprecated, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            technique.id, technique.name, technique.description, technique.tactic,
                            json.dumps(technique.platform), json.dumps(technique.data_sources),
                            technique.detection, json.dumps(technique.mitigations),
                            json.dumps(technique.references), json.dumps(technique.kill_chain_phases),
                            technique.x_mitre_version, technique.revoked, technique.deprecated
                        ))
                        techniques_count += 1
                
                elif obj_type == "intrusion-set":
                    group = await self._parse_group(obj, domain)
                    if group:
                        self.groups[group.id] = group
                        
                        # Store in database
                        cursor.execute("""
                            INSERT OR REPLACE INTO groups 
                            (id, name, aliases, description, techniques, software, references, campaigns, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            group.id, group.name, json.dumps(group.aliases), group.description,
                            json.dumps(group.techniques), json.dumps(group.software),
                            json.dumps(group.references), json.dumps(group.associated_campaigns)
                        ))
                        groups_count += 1
                
                elif obj_type in ["malware", "tool"]:
                    software = await self._parse_software(obj, domain)
                    if software:
                        self.software[software.id] = software
                        
                        # Store in database
                        cursor.execute("""
                            INSERT OR REPLACE INTO software 
                            (id, name, type, description, aliases, platforms, techniques, references, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            software.id, software.name, software.type, software.description,
                            json.dumps(software.aliases), json.dumps(software.platforms),
                            json.dumps(software.techniques), json.dumps(software.references)
                        ))
                        software_count += 1
                        
            except Exception as e:
                logger.debug(f"Error parsing MITRE object {obj.get('id', 'unknown')}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Parsed {domain} domain: {techniques_count} techniques, {groups_count} groups, {software_count} software")
    
    async def _parse_technique(self, obj: Dict[str, Any], domain: str) -> Optional[AttackTechnique]:
        """Parse MITRE ATT&CK technique object"""
        try:
            # Extract external references to get technique ID
            external_refs = obj.get("external_references", [])
            technique_id = None
            references = []
            
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    technique_id = ref.get("external_id")
                references.append(ref.get("url", ""))
            
            if not technique_id:
                return None
            
            # Extract kill chain phases and tactics
            kill_chain_phases = []
            tactics = []
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    phase_name = phase.get("phase_name", "")
                    kill_chain_phases.append(phase_name)
                    tactics.append(phase_name.replace("-", "_"))
            
            # Extract platforms
            platforms = obj.get("x_mitre_platforms", [])
            
            # Extract data sources
            data_sources = []
            for data_component in obj.get("x_mitre_data_sources", []):
                data_sources.append(data_component)
            
            # Extract detection information
            detection = obj.get("x_mitre_detection", "")
            
            return AttackTechnique(
                id=technique_id,
                name=obj.get("name", ""),
                description=obj.get("description", ""),
                tactic=tactics[0] if tactics else "",
                platform=platforms,
                data_sources=data_sources,
                detection=detection,
                mitigations=[],  # Will be populated from relationship objects
                references=references,
                kill_chain_phases=kill_chain_phases,
                x_mitre_version=obj.get("x_mitre_version", ""),
                revoked=obj.get("revoked", False),
                deprecated=obj.get("x_mitre_deprecated", False)
            )
            
        except Exception as e:
            logger.debug(f"Error parsing technique: {e}")
            return None
    
    async def _parse_group(self, obj: Dict[str, Any], domain: str) -> Optional[AttackGroup]:
        """Parse MITRE ATT&CK group object"""
        try:
            # Extract external references to get group ID
            external_refs = obj.get("external_references", [])
            group_id = None
            references = []
            
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    group_id = ref.get("external_id")
                references.append(ref.get("url", ""))
            
            if not group_id:
                return None
            
            return AttackGroup(
                id=group_id,
                name=obj.get("name", ""),
                aliases=obj.get("aliases", []),
                description=obj.get("description", ""),
                techniques=[],  # Will be populated from relationship objects
                software=[],  # Will be populated from relationship objects
                references=references,
                associated_campaigns=[]
            )
            
        except Exception as e:
            logger.debug(f"Error parsing group: {e}")
            return None
    
    async def _parse_software(self, obj: Dict[str, Any], domain: str) -> Optional[AttackSoftware]:
        """Parse MITRE ATT&CK software object"""
        try:
            # Extract external references to get software ID
            external_refs = obj.get("external_references", [])
            software_id = None
            references = []
            
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    software_id = ref.get("external_id")
                references.append(ref.get("url", ""))
            
            if not software_id:
                return None
            
            return AttackSoftware(
                id=software_id,
                name=obj.get("name", ""),
                type=obj.get("type", ""),
                description=obj.get("description", ""),
                aliases=obj.get("x_mitre_aliases", []),
                platforms=obj.get("x_mitre_platforms", []),
                techniques=[],  # Will be populated from relationship objects
                references=references
            )
            
        except Exception as e:
            logger.debug(f"Error parsing software: {e}")
            return None
    
    async def _load_detection_patterns(self):
        """Load detection patterns for MITRE techniques"""
        self.detection_patterns = {
            # Initial Access
            "T1566.001": {  # Spearphishing Attachment
                "network_indicators": ["suspicious_attachment", "email_from_unknown"],
                "host_indicators": ["office_macro_execution", "suspicious_process_creation"],
                "log_sources": ["email_gateway", "endpoint_detection", "proxy_logs"]
            },
            "T1190": {  # Exploit Public-Facing Application
                "network_indicators": ["exploit_payload", "unusual_http_requests"],
                "host_indicators": ["web_shell_creation", "privilege_escalation"],
                "log_sources": ["web_access_logs", "application_logs", "endpoint_detection"]
            },
            
            # Execution
            "T1059.001": {  # PowerShell
                "network_indicators": ["powershell_download", "encoded_commands"],
                "host_indicators": ["powershell_execution", "script_block_logging"],
                "log_sources": ["windows_event_logs", "powershell_logs", "endpoint_detection"]
            },
            "T1059.003": {  # Windows Command Shell
                "network_indicators": ["cmd_remote_execution"],
                "host_indicators": ["cmd_suspicious_args", "batch_file_execution"],
                "log_sources": ["process_creation_logs", "command_line_auditing"]
            },
            
            # Persistence
            "T1053.005": {  # Scheduled Task
                "network_indicators": [],
                "host_indicators": ["task_creation", "suspicious_task_actions"],
                "log_sources": ["task_scheduler_logs", "windows_event_logs"]
            },
            "T1547.001": {  # Registry Run Keys / Startup Folder
                "network_indicators": [],
                "host_indicators": ["registry_modification", "startup_folder_changes"],
                "log_sources": ["registry_monitoring", "file_system_monitoring"]
            },
            
            # Privilege Escalation
            "T1068": {  # Exploitation for Privilege Escalation
                "network_indicators": ["exploit_kit_activity"],
                "host_indicators": ["privilege_escalation", "kernel_exploit"],
                "log_sources": ["endpoint_detection", "kernel_logs", "security_event_logs"]
            },
            
            # Defense Evasion
            "T1055": {  # Process Injection
                "network_indicators": [],
                "host_indicators": ["process_injection", "dll_injection", "memory_modification"],
                "log_sources": ["endpoint_detection", "memory_analysis", "api_monitoring"]
            },
            "T1027": {  # Obfuscated Files or Information
                "network_indicators": ["encoded_payload"],
                "host_indicators": ["obfuscated_files", "packed_executables"],
                "log_sources": ["file_analysis", "malware_sandbox", "endpoint_detection"]
            },
            
            # Credential Access
            "T1003": {  # OS Credential Dumping
                "network_indicators": [],
                "host_indicators": ["lsass_access", "sam_database_access", "credential_dumping"],
                "log_sources": ["endpoint_detection", "windows_event_logs", "privileged_access_monitoring"]
            },
            "T1110": {  # Brute Force
                "network_indicators": ["multiple_failed_logins", "password_spray"],
                "host_indicators": ["account_lockouts", "authentication_failures"],
                "log_sources": ["authentication_logs", "domain_controller_logs", "application_logs"]
            },
            
            # Discovery
            "T1083": {  # File and Directory Discovery
                "network_indicators": [],
                "host_indicators": ["directory_enumeration", "file_listing"],
                "log_sources": ["file_system_monitoring", "process_monitoring"]
            },
            "T1057": {  # Process Discovery
                "network_indicators": [],
                "host_indicators": ["process_enumeration", "tasklist_execution"],
                "log_sources": ["process_monitoring", "command_line_auditing"]
            },
            
            # Lateral Movement
            "T1021.001": {  # Remote Desktop Protocol
                "network_indicators": ["rdp_connections", "remote_desktop_traffic"],
                "host_indicators": ["rdp_logons", "terminal_services"],
                "log_sources": ["network_monitoring", "windows_event_logs", "authentication_logs"]
            },
            "T1021.002": {  # SMB/Windows Admin Shares
                "network_indicators": ["smb_traffic", "admin_share_access"],
                "host_indicators": ["network_share_access", "file_transfer"],
                "log_sources": ["network_monitoring", "file_share_logs", "endpoint_detection"]
            },
            
            # Collection
            "T1005": {  # Data from Local System
                "network_indicators": [],
                "host_indicators": ["file_access_patterns", "data_staging"],
                "log_sources": ["file_system_monitoring", "data_loss_prevention"]
            },
            "T1113": {  # Screen Capture
                "network_indicators": [],
                "host_indicators": ["screenshot_tools", "graphics_api_usage"],
                "log_sources": ["endpoint_detection", "api_monitoring"]
            },
            
            # Exfiltration
            "T1041": {  # Exfiltration Over C2 Channel
                "network_indicators": ["data_upload", "suspicious_outbound_traffic"],
                "host_indicators": ["large_file_access", "data_compression"],
                "log_sources": ["network_monitoring", "data_loss_prevention", "proxy_logs"]
            },
            "T1567": {  # Exfiltration Over Web Service
                "network_indicators": ["cloud_service_upload", "webmail_usage"],
                "host_indicators": ["browser_activity", "file_upload"],
                "log_sources": ["web_proxy_logs", "cloud_access_security_broker", "endpoint_detection"]
            },
            
            # Impact
            "T1486": {  # Data Encrypted for Impact
                "network_indicators": ["ransomware_communication"],
                "host_indicators": ["file_encryption", "ransom_note_creation", "bulk_file_deletion"],
                "log_sources": ["endpoint_detection", "file_system_monitoring", "backup_monitoring"]
            },
            "T1490": {  # Inhibit System Recovery
                "network_indicators": [],
                "host_indicators": ["backup_deletion", "shadow_copy_deletion", "recovery_disabling"],
                "log_sources": ["backup_logs", "system_event_logs", "endpoint_detection"]
            }
        }
        
        logger.info(f"Loaded detection patterns for {len(self.detection_patterns)} MITRE techniques")
    
    async def _validate_data_integrity(self):
        """Validate the integrity of loaded MITRE data"""
        issues = []
        
        # Check for minimum expected techniques
        if len(self.techniques) < 100:
            issues.append(f"Only {len(self.techniques)} techniques loaded (expected > 100)")
        
        # Check for critical techniques
        critical_techniques = ["T1566.001", "T1190", "T1059.001", "T1055", "T1003"]
        missing_critical = [t for t in critical_techniques if t not in self.techniques]
        if missing_critical:
            issues.append(f"Missing critical techniques: {missing_critical}")
        
        # Check for minimum expected groups
        if len(self.groups) < 50:
            issues.append(f"Only {len(self.groups)} groups loaded (expected > 50)")
        
        # Check for minimum expected software
        if len(self.software) < 200:
            issues.append(f"Only {len(self.software)} software loaded (expected > 200)")
        
        if issues:
            logger.warning(f"MITRE data validation issues: {issues}")
        else:
            logger.info("MITRE data validation passed")
    
    async def map_threat_to_techniques(self, threat_indicators: List[Dict[str, Any]], 
                                     context: Dict[str, Any] = None) -> ThreatMapping:
        """Map detected threats to MITRE ATT&CK techniques"""
        try:
            threat_id = str(uuid.uuid4())
            mapped_techniques = []
            evidence = []
            confidence_scores = []
            
            for indicator in threat_indicators:
                indicator_type = indicator.get("type", "")
                indicator_value = indicator.get("value", "")
                indicator_context = indicator.get("context", {})
                
                # Map based on indicator type and patterns
                techniques = await self._analyze_indicator_for_techniques(
                    indicator_type, indicator_value, indicator_context
                )
                
                for tech_id, confidence in techniques:
                    if tech_id not in mapped_techniques:
                        mapped_techniques.append(tech_id)
                        confidence_scores.append(confidence)
                        evidence.append(f"{indicator_type}:{indicator_value}")
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            mapping = ThreatMapping(
                threat_id=threat_id,
                technique_ids=mapped_techniques,
                confidence=overall_confidence,
                evidence=evidence,
                timestamp=datetime.now(),
                source="mitre_attack_integration"
            )
            
            # Store mapping
            await self._store_threat_mapping(mapping)
            
            logger.info(f"Mapped threat {threat_id} to {len(mapped_techniques)} MITRE techniques")
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping threat to techniques: {e}")
            return ThreatMapping(
                threat_id=str(uuid.uuid4()),
                technique_ids=[],
                confidence=0.0,
                evidence=[],
                timestamp=datetime.now(),
                source="error"
            )
    
    async def _analyze_indicator_for_techniques(self, indicator_type: str, 
                                              indicator_value: str, 
                                              context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Analyze indicator and map to MITRE techniques"""
        techniques = []
        
        try:
            # Network-based indicators
            if indicator_type == "ip-dst":
                if context.get("is_c2_server"):
                    techniques.append(("T1071", 0.8))  # Application Layer Protocol
                    techniques.append(("T1041", 0.7))  # Exfiltration Over C2 Channel
                
                if context.get("port") in [22, 3389]:
                    techniques.append(("T1021.001", 0.6))  # Remote Desktop Protocol
                    
            # File-based indicators
            elif indicator_type in ["md5", "sha1", "sha256"]:
                if context.get("file_type") == "executable":
                    techniques.append(("T1204.002", 0.7))  # Malicious File
                    
                if context.get("is_packed"):
                    techniques.append(("T1027", 0.8))  # Obfuscated Files
                    
                if context.get("has_persistence"):
                    techniques.append(("T1547.001", 0.6))  # Registry Run Keys
            
            # Email-based indicators
            elif indicator_type == "email":
                if context.get("has_attachment"):
                    techniques.append(("T1566.001", 0.8))  # Spearphishing Attachment
                    
                if context.get("phishing_indicators"):
                    techniques.append(("T1566.002", 0.7))  # Spearphishing Link
            
            # Domain/URL indicators
            elif indicator_type in ["domain", "url"]:
                if context.get("is_malicious_domain"):
                    techniques.append(("T1071.001", 0.7))  # Web Protocols
                    
                if context.get("is_phishing_site"):
                    techniques.append(("T1566.002", 0.8))  # Spearphishing Link
                    
                if context.get("serves_malware"):
                    techniques.append(("T1189", 0.6))  # Drive-by Compromise
            
            # Process/command indicators
            elif indicator_type == "process":
                if "powershell" in indicator_value.lower():
                    techniques.append(("T1059.001", 0.9))  # PowerShell
                    
                if "cmd" in indicator_value.lower():
                    techniques.append(("T1059.003", 0.8))  # Windows Command Shell
                    
                if context.get("is_injection"):
                    techniques.append(("T1055", 0.8))  # Process Injection
                    
                if context.get("privilege_escalation"):
                    techniques.append(("T1068", 0.7))  # Exploitation for Privilege Escalation
            
            # Registry indicators
            elif indicator_type == "registry":
                if "run" in indicator_value.lower():
                    techniques.append(("T1547.001", 0.8))  # Registry Run Keys
                    
                if context.get("persistence_mechanism"):
                    techniques.append(("T1112", 0.6))  # Modify Registry
            
            # Network protocol indicators
            elif indicator_type == "network_protocol":
                if indicator_value.lower() == "rdp":
                    techniques.append(("T1021.001", 0.8))  # Remote Desktop Protocol
                    
                elif indicator_value.lower() == "smb":
                    techniques.append(("T1021.002", 0.7))  # SMB/Windows Admin Shares
                    
                elif indicator_value.lower() == "ssh":
                    techniques.append(("T1021.004", 0.6))  # SSH
            
            # Behavioral indicators
            elif indicator_type == "behavior":
                if "credential_dumping" in indicator_value.lower():
                    techniques.append(("T1003", 0.9))  # OS Credential Dumping
                    
                if "lateral_movement" in indicator_value.lower():
                    techniques.append(("T1021", 0.7))  # Remote Services
                    
                if "data_exfiltration" in indicator_value.lower():
                    techniques.append(("T1041", 0.8))  # Exfiltration Over C2 Channel
                    
                if "file_encryption" in indicator_value.lower():
                    techniques.append(("T1486", 0.9))  # Data Encrypted for Impact
            
            return techniques
            
        except Exception as e:
            logger.error(f"Error analyzing indicator for techniques: {e}")
            return []
    
    async def _store_threat_mapping(self, mapping: ThreatMapping):
        """Store threat mapping in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO threat_mappings 
                (id, threat_id, technique_ids, confidence, evidence, source, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                mapping.threat_id,
                json.dumps(mapping.technique_ids),
                mapping.confidence,
                json.dumps(mapping.evidence),
                mapping.source,
                mapping.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.threat_mappings.append(mapping)
            
        except Exception as e:
            logger.error(f"Error storing threat mapping: {e}")
    
    async def get_technique_details(self, technique_id: str) -> Optional[AttackTechnique]:
        """Get detailed information about a specific technique"""
        return self.techniques.get(technique_id)
    
    async def get_techniques_by_tactic(self, tactic: str) -> List[AttackTechnique]:
        """Get all techniques for a specific tactic"""
        return [tech for tech in self.techniques.values() if tech.tactic == tactic]
    
    async def get_group_techniques(self, group_id: str) -> List[AttackTechnique]:
        """Get all techniques used by a specific threat group"""
        group = self.groups.get(group_id)
        if not group:
            return []
        
        return [self.techniques[tech_id] for tech_id in group.techniques if tech_id in self.techniques]
    
    async def search_techniques(self, query: str, limit: int = 10) -> List[AttackTechnique]:
        """Search techniques by name or description"""
        query_lower = query.lower()
        results = []
        
        for technique in self.techniques.values():
            if (query_lower in technique.name.lower() or 
                query_lower in technique.description.lower()):
                results.append(technique)
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def generate_detection_rules(self, technique_id: str) -> Dict[str, Any]:
        """Generate detection rules for a specific technique"""
        technique = self.techniques.get(technique_id)
        if not technique:
            return {}
        
        patterns = self.detection_patterns.get(technique_id, {})
        
        return {
            "technique_id": technique_id,
            "technique_name": technique.name,
            "detection_patterns": patterns,
            "sigma_rules": await self._generate_sigma_rules(technique_id, patterns),
            "splunk_queries": await self._generate_splunk_queries(technique_id, patterns),
            "elastic_queries": await self._generate_elastic_queries(technique_id, patterns),
            "mitre_detection": technique.detection
        }
    
    async def _generate_sigma_rules(self, technique_id: str, patterns: Dict[str, Any]) -> List[str]:
        """Generate Sigma detection rules"""
        rules = []
        
        # Basic template for Sigma rules
        if "host_indicators" in patterns:
            for indicator in patterns["host_indicators"]:
                rule = f"""
title: {self.techniques[technique_id].name} Detection
id: {str(uuid.uuid4())}
status: experimental
description: Detects {indicator} related to {technique_id}
references:
    - https://attack.mitre.org/techniques/{technique_id}/
tags:
    - attack.{self.techniques[technique_id].tactic}
    - attack.{technique_id.lower()}
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        - CommandLine|contains: '{indicator}'
    condition: selection
falsepositives:
    - Unknown
level: medium
"""
                rules.append(rule.strip())
        
        return rules
    
    async def _generate_splunk_queries(self, technique_id: str, patterns: Dict[str, Any]) -> List[str]:
        """Generate Splunk detection queries"""
        queries = []
        
        if "host_indicators" in patterns:
            for indicator in patterns["host_indicators"]:
                query = f"""
index=* sourcetype=wineventlog:security OR sourcetype=wineventlog:system
| search "*{indicator}*"
| eval technique_id="{technique_id}"
| eval technique_name="{self.techniques[technique_id].name}"
| stats count by host, user, technique_id
| where count > 1
"""
                queries.append(query.strip())
        
        return queries
    
    async def _generate_elastic_queries(self, technique_id: str, patterns: Dict[str, Any]) -> List[str]:
        """Generate Elasticsearch/EQL detection queries"""
        queries = []
        
        if "host_indicators" in patterns:
            for indicator in patterns["host_indicators"]:
                query = f"""
{{
  "query": {{
    "bool": {{
      "must": [
        {{"term": {{"technique.id": "{technique_id}"}}}},
        {{"wildcard": {{"process.command_line": "*{indicator}*"}}}}
      ]
    }}
  }},
  "aggs": {{
    "by_host": {{
      "terms": {{
        "field": "host.name"
      }}
    }}
  }}
}}
"""
                queries.append(query.strip())
        
        return queries
    
    async def get_attack_flow(self, technique_ids: List[str]) -> Dict[str, Any]:
        """Generate attack flow visualization data"""
        try:
            # Group techniques by tactic
            tactic_groups = {}
            for tech_id in technique_ids:
                technique = self.techniques.get(tech_id)
                if technique:
                    tactic = technique.tactic
                    if tactic not in tactic_groups:
                        tactic_groups[tactic] = []
                    tactic_groups[tactic].append(technique)
            
            # Define kill chain order
            kill_chain_order = [
                "reconnaissance", "resource_development", "initial_access",
                "execution", "persistence", "privilege_escalation",
                "defense_evasion", "credential_access", "discovery",
                "lateral_movement", "collection", "command_and_control",
                "exfiltration", "impact"
            ]
            
            # Create ordered flow
            flow_data = {
                "attack_flow": [],
                "total_techniques": len(technique_ids),
                "tactics_covered": len(tactic_groups),
                "complexity_score": len(tactic_groups) / len(kill_chain_order)
            }
            
            for tactic in kill_chain_order:
                if tactic in tactic_groups:
                    flow_data["attack_flow"].append({
                        "tactic": tactic,
                        "techniques": [
                            {
                                "id": tech.id,
                                "name": tech.name,
                                "platforms": tech.platform
                            } for tech in tactic_groups[tactic]
                        ]
                    })
            
            return flow_data
            
        except Exception as e:
            logger.error(f"Error generating attack flow: {e}")
            return {"attack_flow": [], "error": str(e)}
    
    async def get_threat_landscape_summary(self) -> Dict[str, Any]:
        """Get summary of current threat landscape based on MITRE data"""
        try:
            # Get recent threat mappings
            recent_mappings = [
                mapping for mapping in self.threat_mappings
                if mapping.timestamp > datetime.now() - timedelta(days=30)
            ]
            
            # Count techniques by tactic
            tactic_counts = {}
            for technique in self.techniques.values():
                tactic = technique.tactic
                tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
            
            # Most active groups (mock data based on recent intelligence)
            active_groups = ["APT1", "APT28", "APT29", "Lazarus Group", "FIN7"]
            
            # Top techniques (based on real-world prevalence)
            top_techniques = [
                "T1566.001",  # Spearphishing Attachment
                "T1059.001",  # PowerShell
                "T1055",      # Process Injection
                "T1003",      # OS Credential Dumping
                "T1021.001",  # Remote Desktop Protocol
                "T1486",      # Data Encrypted for Impact
                "T1190",      # Exploit Public-Facing Application
                "T1078",      # Valid Accounts
                "T1027",      # Obfuscated Files or Information
                "T1070.004"   # File Deletion
            ]
            
            return {
                "framework_version": "ATT&CK v12",
                "total_techniques": len(self.techniques),
                "total_groups": len(self.groups),
                "total_software": len(self.software),
                "techniques_by_tactic": tactic_counts,
                "recent_threat_mappings": len(recent_mappings),
                "most_active_groups": active_groups,
                "top_techniques": top_techniques,
                "last_updated": datetime.now().isoformat(),
                "coverage_summary": {
                    "enterprise": len([t for t in self.techniques.values() if "enterprise" in t.platform or "windows" in t.platform]),
                    "mobile": len([t for t in self.techniques.values() if any(p in ["android", "ios"] for p in t.platform)]),
                    "ics": len([t for t in self.techniques.values() if any(p in ["ics", "industrial"] for p in t.platform)])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating threat landscape summary: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown MITRE ATT&CK integration"""
        if self.session:
            await self.session.close()
        
        logger.info("MITRE ATT&CK Framework integration shutdown complete")

# Global instance management
_mitre_framework: Optional[MITREAttackFramework] = None

async def get_mitre_framework() -> MITREAttackFramework:
    """Get global MITRE ATT&CK framework instance"""
    global _mitre_framework
    
    if _mitre_framework is None:
        _mitre_framework = MITREAttackFramework()
        await _mitre_framework.initialize()
    
    return _mitre_framework