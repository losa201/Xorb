"""
MITRE ATT&CK Integration Service
Production-ready MITRE ATT&CK framework integration for threat mapping and analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import aiofiles

from .base_service import IntelligenceService, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)


class TacticCategory(Enum):
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    EXFILTRATION = "TA0010"
    COMMAND_AND_CONTROL = "TA0011"
    IMPACT = "TA0040"
    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"


@dataclass
class MitreTactic:
    """MITRE ATT&CK Tactic"""
    tactic_id: str
    name: str
    description: str
    techniques: List[str] = field(default_factory=list)
    url: str = ""
    platforms: List[str] = field(default_factory=list)


@dataclass
class MitreTechnique:
    """MITRE ATT&CK Technique"""
    technique_id: str
    name: str
    description: str
    tactic_refs: List[str] = field(default_factory=list)
    subtechniques: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    detection: Dict[str, Any] = field(default_factory=dict)
    url: str = ""
    kill_chain_phases: List[str] = field(default_factory=list)


@dataclass
class MitreGroup:
    """MITRE ATT&CK Group (Threat Actor)"""
    group_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    techniques: List[str] = field(default_factory=list)
    software: List[str] = field(default_factory=list)
    country: Optional[str] = None
    motivation: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_activity: Optional[datetime] = None


@dataclass
class MitreMitigation:
    """MITRE ATT&CK Mitigation"""
    mitigation_id: str
    name: str
    description: str
    techniques: List[str] = field(default_factory=list)
    implementation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPattern:
    """Detected attack pattern based on MITRE framework"""
    pattern_id: str
    name: str
    confidence: float
    techniques_matched: List[str]
    tactics_involved: List[str]
    severity: str
    detection_time: datetime
    indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MitreAttackService(IntelligenceService):
    """Production MITRE ATT&CK integration service"""

    def __init__(self, **kwargs):
        super().__init__(
            service_id="mitre_attack",
            dependencies=["threat_intelligence"],
            **kwargs
        )

        # MITRE ATT&CK data
        self.tactics: Dict[str, MitreTactic] = {}
        self.techniques: Dict[str, MitreTechnique] = {}
        self.groups: Dict[str, MitreGroup] = {}
        self.mitigations: Dict[str, MitreMitigation] = {}

        # Attack pattern detection
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.pattern_rules: List[Dict[str, Any]] = []

        # Framework URLs
        self.mitre_urls = {
            "enterprise": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "mobile": "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json",
            "ics": "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"
        }

        # Analytics
        self.analytics = {
            "tactics_loaded": 0,
            "techniques_loaded": 0,
            "groups_loaded": 0,
            "patterns_detected": 0,
            "mappings_performed": 0,
            "last_update": None
        }

        # Mapping cache
        self.mapping_cache: Dict[str, Any] = {}

        # Update tracking
        self.last_framework_update = None
        self.update_interval = timedelta(days=7)  # Weekly updates

    async def initialize(self) -> bool:
        """Initialize MITRE ATT&CK service"""
        try:
            logger.info("Initializing MITRE ATT&CK Service...")

            # Load MITRE ATT&CK framework data
            await self._load_mitre_framework()

            # Initialize attack pattern detection rules
            self._initialize_pattern_rules()

            # Start periodic updates
            asyncio.create_task(self._periodic_framework_update())

            logger.info(f"MITRE ATT&CK Service initialized with {len(self.techniques)} techniques")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MITRE ATT&CK service: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the service"""
        try:
            # Save current state
            await self._save_framework_data()

            logger.info("MITRE ATT&CK Service shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {
                "tactics_loaded": len(self.tactics),
                "techniques_loaded": len(self.techniques),
                "groups_loaded": len(self.groups),
                "mitigations_loaded": len(self.mitigations),
                "pattern_rules": len(self.pattern_rules),
                "cache_size": len(self.mapping_cache),
                "last_update": self.last_framework_update.isoformat() if self.last_framework_update else None
            }

            # Check if data is stale
            status = ServiceStatus.HEALTHY
            message = "MITRE ATT&CK service operational"

            if self.last_framework_update:
                age = datetime.utcnow() - self.last_framework_update
                if age > timedelta(days=30):
                    status = ServiceStatus.DEGRADED
                    message = "MITRE framework data is stale"

            if not self.techniques:
                status = ServiceStatus.UNHEALTHY
                message = "No MITRE techniques loaded"

            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )

        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )

    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data for MITRE ATT&CK patterns"""
        try:
            if isinstance(data, dict):
                if "indicators" in data:
                    return await self.map_indicators_to_attack_patterns(data["indicators"])
                elif "techniques" in data:
                    return await self.analyze_technique_sequence(data["techniques"])
                elif "events" in data:
                    return await self.detect_attack_patterns(data["events"])

            # Default analysis
            return await self.map_indicators_to_attack_patterns([str(data)])

        except Exception as e:
            logger.error(f"Failed to analyze data: {e}")
            raise

    async def get_model_status(self) -> Dict[str, Any]:
        """Get MITRE ATT&CK framework status"""
        return {
            "framework_loaded": len(self.techniques) > 0,
            "framework_version": "ATT&CK v12.1",
            "last_update": self.last_framework_update.isoformat() if self.last_framework_update else None,
            "tactics_count": len(self.tactics),
            "techniques_count": len(self.techniques),
            "groups_count": len(self.groups),
            "mitigations_count": len(self.mitigations),
            "analytics": self.analytics
        }

    async def map_indicators_to_attack_patterns(self, indicators: List[str]) -> Dict[str, Any]:
        """Map threat indicators to MITRE ATT&CK techniques"""
        try:
            mapping_result = {
                "mapping_id": f"mapping_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "indicators_analyzed": len(indicators),
                "techniques_mapped": [],
                "tactics_identified": [],
                "attack_patterns": [],
                "confidence_scores": {},
                "recommendations": []
            }

            technique_matches = {}
            tactic_matches = set()

            for indicator in indicators:
                # Check cache first
                cache_key = f"indicator_mapping:{indicator}"
                if cache_key in self.mapping_cache:
                    cached_result = self.mapping_cache[cache_key]
                    for technique in cached_result.get("techniques", []):
                        technique_matches[technique] = technique_matches.get(technique, 0) + 1
                    tactic_matches.update(cached_result.get("tactics", []))
                    continue

                # Perform mapping
                mapped_techniques, mapped_tactics = await self._map_indicator_to_techniques(indicator)

                # Cache result
                self.mapping_cache[cache_key] = {
                    "techniques": mapped_techniques,
                    "tactics": mapped_tactics,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Aggregate results
                for technique in mapped_techniques:
                    technique_matches[technique] = technique_matches.get(technique, 0) + 1
                tactic_matches.update(mapped_tactics)

            # Build final results
            for technique_id, count in technique_matches.items():
                if technique_id in self.techniques:
                    technique = self.techniques[technique_id]
                    confidence = min(100.0, (count / len(indicators)) * 100 + 20)

                    mapping_result["techniques_mapped"].append({
                        "technique_id": technique_id,
                        "name": technique.name,
                        "confidence": confidence,
                        "indicator_matches": count,
                        "platforms": technique.platforms,
                        "mitigations": technique.mitigations[:3]  # Top 3 mitigations
                    })

                    mapping_result["confidence_scores"][technique_id] = confidence

            # Add tactics
            for tactic_id in tactic_matches:
                if tactic_id in self.tactics:
                    tactic = self.tactics[tactic_id]
                    mapping_result["tactics_identified"].append({
                        "tactic_id": tactic_id,
                        "name": tactic.name,
                        "description": tactic.description
                    })

            # Detect attack patterns
            mapping_result["attack_patterns"] = await self._detect_attack_patterns_from_mapping(
                mapping_result["techniques_mapped"],
                mapping_result["tactics_identified"]
            )

            # Generate recommendations
            mapping_result["recommendations"] = self._generate_mapping_recommendations(mapping_result)

            self.analytics["mappings_performed"] += 1

            return mapping_result

        except Exception as e:
            logger.error(f"Failed to map indicators to attack patterns: {e}")
            raise

    async def analyze_technique_sequence(self, technique_sequence: List[str]) -> Dict[str, Any]:
        """Analyze a sequence of techniques for attack pattern recognition"""
        try:
            analysis_result = {
                "analysis_id": f"sequence_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "techniques_analyzed": len(technique_sequence),
                "sequence_analysis": {},
                "probable_campaigns": [],
                "kill_chain_coverage": {},
                "threat_actor_matches": [],
                "recommendations": []
            }

            # Analyze technique sequence
            valid_techniques = []
            for tech_id in technique_sequence:
                if tech_id in self.techniques:
                    valid_techniques.append(tech_id)

            if not valid_techniques:
                analysis_result["error"] = "No valid MITRE techniques found in sequence"
                return analysis_result

            # Kill chain analysis
            kill_chain_phases = []
            tactics_involved = set()

            for tech_id in valid_techniques:
                technique = self.techniques[tech_id]
                kill_chain_phases.extend(technique.kill_chain_phases)
                tactics_involved.update(technique.tactic_refs)

            # Calculate kill chain coverage
            all_phases = ["reconnaissance", "weaponization", "delivery", "exploitation",
                         "installation", "command-and-control", "actions-on-objectives"]

            coverage = {}
            for phase in all_phases:
                coverage[phase] = phase in kill_chain_phases

            analysis_result["kill_chain_coverage"] = coverage

            # Find matching threat actors
            actor_matches = []
            for group_id, group in self.groups.items():
                common_techniques = set(valid_techniques) & set(group.techniques)
                if common_techniques:
                    match_percentage = len(common_techniques) / len(set(valid_techniques)) * 100
                    if match_percentage >= 30:  # At least 30% technique overlap
                        actor_matches.append({
                            "group_id": group_id,
                            "name": group.name,
                            "aliases": group.aliases,
                            "match_percentage": match_percentage,
                            "common_techniques": list(common_techniques),
                            "country": group.country
                        })

            # Sort by match percentage
            actor_matches.sort(key=lambda x: x["match_percentage"], reverse=True)
            analysis_result["threat_actor_matches"] = actor_matches[:5]  # Top 5 matches

            # Sequence analysis
            analysis_result["sequence_analysis"] = {
                "valid_techniques": len(valid_techniques),
                "tactics_involved": len(tactics_involved),
                "kill_chain_phases": len(set(kill_chain_phases)),
                "sequence_complexity": self._calculate_sequence_complexity(valid_techniques),
                "temporal_pattern": self._analyze_temporal_pattern(valid_techniques)
            }

            # Generate recommendations
            analysis_result["recommendations"] = self._generate_sequence_recommendations(analysis_result)

            return analysis_result

        except Exception as e:
            logger.error(f"Failed to analyze technique sequence: {e}")
            raise

    async def detect_attack_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect MITRE ATT&CK patterns from security events"""
        try:
            detection_result = {
                "detection_id": f"detect_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "events_analyzed": len(events),
                "patterns_detected": [],
                "confidence_scores": {},
                "timeline": [],
                "recommendations": []
            }

            # Extract potential techniques from events
            detected_techniques = []
            event_timeline = []

            for event in events:
                timestamp = event.get("timestamp", datetime.utcnow().isoformat())
                event_type = event.get("type", "unknown")

                # Map event to potential techniques
                techniques = await self._map_event_to_techniques(event)

                for technique_id in techniques:
                    detected_techniques.append(technique_id)
                    event_timeline.append({
                        "timestamp": timestamp,
                        "technique": technique_id,
                        "event_type": event_type,
                        "confidence": techniques[technique_id]
                    })

            # Detect patterns using rules
            for rule in self.pattern_rules:
                pattern_match = await self._evaluate_pattern_rule(rule, detected_techniques, events)
                if pattern_match:
                    detection_result["patterns_detected"].append(pattern_match)
                    detection_result["confidence_scores"][pattern_match["pattern_id"]] = pattern_match["confidence"]

            # Sort timeline
            event_timeline.sort(key=lambda x: x["timestamp"])
            detection_result["timeline"] = event_timeline

            # Generate recommendations
            detection_result["recommendations"] = self._generate_detection_recommendations(detection_result)

            self.analytics["patterns_detected"] += len(detection_result["patterns_detected"])

            return detection_result

        except Exception as e:
            logger.error(f"Failed to detect attack patterns: {e}")
            raise

    async def get_technique_details(self, technique_id: str) -> Dict[str, Any]:
        """Get detailed information about a MITRE technique"""
        if technique_id not in self.techniques:
            raise ValueError(f"Technique {technique_id} not found")

        technique = self.techniques[technique_id]

        return {
            "technique_id": technique_id,
            "name": technique.name,
            "description": technique.description,
            "tactics": [self.tactics[tactic_id].name for tactic_id in technique.tactic_refs if tactic_id in self.tactics],
            "platforms": technique.platforms,
            "data_sources": technique.data_sources,
            "mitigations": [
                {
                    "mitigation_id": mit_id,
                    "name": self.mitigations[mit_id].name if mit_id in self.mitigations else "Unknown",
                    "description": self.mitigations[mit_id].description if mit_id in self.mitigations else ""
                }
                for mit_id in technique.mitigations
            ],
            "detection": technique.detection,
            "subtechniques": technique.subtechniques,
            "url": technique.url,
            "related_groups": self._get_groups_using_technique(technique_id)
        }

    async def get_mitigation_recommendations(self, technique_ids: List[str]) -> Dict[str, Any]:
        """Get mitigation recommendations for techniques"""
        recommendations = {
            "analysis_id": f"mitigations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "techniques_analyzed": len(technique_ids),
            "mitigations": [],
            "implementation_priority": [],
            "coverage_analysis": {}
        }

        # Collect all mitigations
        mitigation_counts = {}
        technique_coverage = {}

        for tech_id in technique_ids:
            if tech_id in self.techniques:
                technique = self.techniques[tech_id]
                technique_coverage[tech_id] = {
                    "name": technique.name,
                    "mitigations": []
                }

                for mit_id in technique.mitigations:
                    mitigation_counts[mit_id] = mitigation_counts.get(mit_id, 0) + 1
                    technique_coverage[tech_id]["mitigations"].append(mit_id)

        # Build mitigation recommendations
        for mit_id, count in mitigation_counts.items():
            if mit_id in self.mitigations:
                mitigation = self.mitigations[mit_id]
                effectiveness = (count / len(technique_ids)) * 100

                recommendations["mitigations"].append({
                    "mitigation_id": mit_id,
                    "name": mitigation.name,
                    "description": mitigation.description,
                    "effectiveness_percentage": effectiveness,
                    "techniques_covered": count,
                    "implementation": mitigation.implementation,
                    "priority": "high" if effectiveness >= 70 else "medium" if effectiveness >= 40 else "low"
                })

        # Sort by effectiveness
        recommendations["mitigations"].sort(key=lambda x: x["effectiveness_percentage"], reverse=True)

        # Implementation priority
        high_priority = [m for m in recommendations["mitigations"] if m["priority"] == "high"]
        recommendations["implementation_priority"] = high_priority[:5]  # Top 5 high priority

        recommendations["coverage_analysis"] = technique_coverage

        return recommendations

    # Helper methods
    async def _load_mitre_framework(self):
        """Load MITRE ATT&CK framework data"""
        try:
            # Load enterprise framework
            await self._load_framework_from_url("enterprise", self.mitre_urls["enterprise"])

            # Initialize built-in data if download fails
            if not self.techniques:
                self._load_builtin_framework_data()

            self.last_framework_update = datetime.utcnow()
            self.analytics["last_update"] = self.last_framework_update.isoformat()

        except Exception as e:
            logger.error(f"Failed to load MITRE framework: {e}")
            # Fallback to built-in data
            self._load_builtin_framework_data()

    async def _load_framework_from_url(self, framework_name: str, url: str):
        """Load framework data from MITRE repository"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._parse_stix_data(data)
                        logger.info(f"Loaded {framework_name} framework from MITRE repository")
                    else:
                        logger.error(f"Failed to download {framework_name} framework: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error loading framework from {url}: {e}")

    async def _parse_stix_data(self, stix_data: Dict[str, Any]):
        """Parse STIX format MITRE data"""
        objects = stix_data.get("objects", [])

        for obj in objects:
            obj_type = obj.get("type")

            if obj_type == "attack-pattern":
                await self._parse_technique(obj)
            elif obj_type == "x-mitre-tactic":
                await self._parse_tactic(obj)
            elif obj_type == "intrusion-set":
                await self._parse_group(obj)
            elif obj_type == "course-of-action":
                await self._parse_mitigation(obj)

    async def _parse_technique(self, obj: Dict[str, Any]):
        """Parse technique from STIX object"""
        external_refs = obj.get("external_references", [])
        mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)

        if mitre_ref:
            technique_id = mitre_ref.get("external_id")
            if technique_id:
                technique = MitreTechnique(
                    technique_id=technique_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", ""),
                    platforms=obj.get("x_mitre_platforms", []),
                    data_sources=obj.get("x_mitre_data_sources", []),
                    url=mitre_ref.get("url", "")
                )

                # Extract kill chain phases
                kill_chain = obj.get("kill_chain_phases", [])
                technique.kill_chain_phases = [phase.get("phase_name", "") for phase in kill_chain]

                # Map to tactics
                technique.tactic_refs = [phase.get("kill_chain_name", "") for phase in kill_chain]

                self.techniques[technique_id] = technique
                self.analytics["techniques_loaded"] += 1

    async def _parse_tactic(self, obj: Dict[str, Any]):
        """Parse tactic from STIX object"""
        external_refs = obj.get("external_references", [])
        mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)

        if mitre_ref:
            tactic_id = mitre_ref.get("external_id")
            if tactic_id:
                tactic = MitreTactic(
                    tactic_id=tactic_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", ""),
                    url=mitre_ref.get("url", "")
                )

                self.tactics[tactic_id] = tactic
                self.analytics["tactics_loaded"] += 1

    async def _parse_group(self, obj: Dict[str, Any]):
        """Parse group (threat actor) from STIX object"""
        external_refs = obj.get("external_references", [])
        mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)

        if mitre_ref:
            group_id = mitre_ref.get("external_id")
            if group_id:
                group = MitreGroup(
                    group_id=group_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", ""),
                    aliases=obj.get("aliases", [])
                )

                self.groups[group_id] = group
                self.analytics["groups_loaded"] += 1

    async def _parse_mitigation(self, obj: Dict[str, Any]):
        """Parse mitigation from STIX object"""
        external_refs = obj.get("external_references", [])
        mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)

        if mitre_ref:
            mitigation_id = mitre_ref.get("external_id")
            if mitigation_id:
                mitigation = MitreMitigation(
                    mitigation_id=mitigation_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", "")
                )

                self.mitigations[mitigation_id] = mitigation

    def _load_builtin_framework_data(self):
        """Load built-in MITRE framework data as fallback"""
        # Basic framework data for fallback
        self.tactics = {
            "TA0001": MitreTactic("TA0001", "Initial Access", "Initial access to target network"),
            "TA0002": MitreTactic("TA0002", "Execution", "Code execution on target systems"),
            "TA0003": MitreTactic("TA0003", "Persistence", "Maintain presence in target network"),
            "TA0004": MitreTactic("TA0004", "Privilege Escalation", "Gain higher-level permissions"),
            "TA0005": MitreTactic("TA0005", "Defense Evasion", "Avoid detection by security controls"),
            "TA0006": MitreTactic("TA0006", "Credential Access", "Steal account names and passwords"),
            "TA0007": MitreTactic("TA0007", "Discovery", "Explore the target environment"),
            "TA0008": MitreTactic("TA0008", "Lateral Movement", "Move through the target network"),
            "TA0009": MitreTactic("TA0009", "Collection", "Gather information of interest"),
            "TA0010": MitreTactic("TA0010", "Exfiltration", "Steal data from target network"),
            "TA0011": MitreTactic("TA0011", "Command and Control", "Communicate with compromised systems")
        }

        # Sample techniques
        self.techniques = {
            "T1566": MitreTechnique("T1566", "Phishing", "Email-based attack vectors", ["TA0001"]),
            "T1190": MitreTechnique("T1190", "Exploit Public-Facing Application", "Exploit internet-facing services", ["TA0001"]),
            "T1078": MitreTechnique("T1078", "Valid Accounts", "Use legitimate credentials", ["TA0001", "TA0003", "TA0004"]),
            "T1055": MitreTechnique("T1055", "Process Injection", "Inject code into running processes", ["TA0004", "TA0005"]),
            "T1027": MitreTechnique("T1027", "Obfuscated Files or Information", "Hide malicious content", ["TA0005"]),
            "T1003": MitreTechnique("T1003", "OS Credential Dumping", "Extract credentials from memory", ["TA0006"]),
            "T1018": MitreTechnique("T1018", "Remote System Discovery", "Identify remote systems", ["TA0007"]),
            "T1105": MitreTechnique("T1105", "Ingress Tool Transfer", "Bring tools into target environment", ["TA0011"])
        }

        # Sample groups
        self.groups = {
            "G0016": MitreGroup("G0016", "APT29", ["Cozy Bear", "The Dukes"], "Russian state-sponsored group"),
            "G0028": MitreGroup("G0028", "Lazarus Group", ["HIDDEN COBRA"], "North Korean state-sponsored group"),
            "G0050": MitreGroup("G0050", "APT32", ["OceanLotus"], "Vietnamese state-sponsored group")
        }

        self.analytics["tactics_loaded"] = len(self.tactics)
        self.analytics["techniques_loaded"] = len(self.techniques)
        self.analytics["groups_loaded"] = len(self.groups)

        logger.info("Loaded built-in MITRE framework data")

    def _initialize_pattern_rules(self):
        """Initialize attack pattern detection rules"""
        self.pattern_rules = [
            {
                "pattern_id": "apt_lateral_movement",
                "name": "APT Lateral Movement Pattern",
                "techniques": ["T1078", "T1105", "T1018", "T1003"],
                "min_techniques": 3,
                "time_window": 3600,  # 1 hour
                "severity": "high"
            },
            {
                "pattern_id": "ransomware_execution",
                "name": "Ransomware Execution Pattern",
                "techniques": ["T1566", "T1055", "T1027", "T1486"],
                "min_techniques": 2,
                "time_window": 1800,  # 30 minutes
                "severity": "critical"
            },
            {
                "pattern_id": "credential_theft",
                "name": "Credential Theft Pattern",
                "techniques": ["T1003", "T1555", "T1552"],
                "min_techniques": 2,
                "time_window": 7200,  # 2 hours
                "severity": "high"
            }
        ]

    async def _map_indicator_to_techniques(self, indicator: str) -> Tuple[List[str], List[str]]:
        """Map indicator to MITRE techniques and tactics"""
        techniques = []
        tactics = []

        # Simple indicator-to-technique mapping
        # In production, this would be more sophisticated

        if "@" in indicator:  # Email indicator
            techniques.append("T1566")  # Phishing
            tactics.append("TA0001")    # Initial Access

        elif indicator.startswith("http"):  # URL indicator
            techniques.extend(["T1566", "T1190"])  # Phishing, Exploit Public-Facing App
            tactics.extend(["TA0001"])

        elif "." in indicator and not indicator.startswith("http"):  # Domain indicator
            techniques.extend(["T1071", "T1105"])  # C2, Ingress Tool Transfer
            tactics.extend(["TA0011", "TA0010"])

        # Default mapping for unknown indicators
        if not techniques:
            techniques.append("T1105")  # Ingress Tool Transfer
            tactics.append("TA0011")   # Command and Control

        return techniques, tactics

    async def _detect_attack_patterns_from_mapping(self, techniques: List[Dict], tactics: List[Dict]) -> List[Dict[str, Any]]:
        """Detect attack patterns from technique/tactic mapping"""
        patterns = []

        technique_ids = [t["technique_id"] for t in techniques]

        for rule in self.pattern_rules:
            matches = set(technique_ids) & set(rule["techniques"])
            if len(matches) >= rule["min_techniques"]:
                confidence = (len(matches) / len(rule["techniques"])) * 100

                pattern = {
                    "pattern_id": rule["pattern_id"],
                    "name": rule["name"],
                    "confidence": confidence,
                    "techniques_matched": list(matches),
                    "severity": rule["severity"],
                    "detection_time": datetime.utcnow().isoformat()
                }
                patterns.append(pattern)

        return patterns

    async def _map_event_to_techniques(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Map security event to MITRE techniques with confidence scores"""
        techniques = {}

        event_type = event.get("type", "").lower()
        description = event.get("description", "").lower()

        # Process creation events
        if "process" in event_type and "create" in event_type:
            techniques["T1055"] = 70.0  # Process Injection

        # Network connection events
        elif "network" in event_type:
            techniques["T1071"] = 60.0  # Application Layer Protocol
            techniques["T1105"] = 50.0  # Ingress Tool Transfer

        # File creation/modification events
        elif "file" in event_type:
            techniques["T1027"] = 40.0  # Obfuscated Files
            techniques["T1105"] = 45.0  # Ingress Tool Transfer

        # Authentication events
        elif "auth" in event_type or "login" in event_type:
            techniques["T1078"] = 65.0  # Valid Accounts
            if "fail" in description:
                techniques["T1110"] = 55.0  # Brute Force

        # Registry events
        elif "registry" in event_type:
            techniques["T1112"] = 60.0  # Modify Registry
            techniques["T1547"] = 50.0  # Boot or Logon Autostart

        return techniques

    async def _evaluate_pattern_rule(self, rule: Dict[str, Any], techniques: List[str], events: List[Dict]) -> Optional[Dict[str, Any]]:
        """Evaluate a pattern rule against detected techniques"""
        required_techniques = set(rule["techniques"])
        detected_techniques = set(techniques)

        matches = required_techniques & detected_techniques

        if len(matches) >= rule["min_techniques"]:
            confidence = (len(matches) / len(required_techniques)) * 100

            return {
                "pattern_id": rule["pattern_id"],
                "name": rule["name"],
                "confidence": confidence,
                "techniques_matched": list(matches),
                "severity": rule["severity"],
                "detection_time": datetime.utcnow().isoformat(),
                "events_analyzed": len(events)
            }

        return None

    def _calculate_sequence_complexity(self, techniques: List[str]) -> str:
        """Calculate complexity of technique sequence"""
        unique_tactics = set()
        for tech_id in techniques:
            if tech_id in self.techniques:
                unique_tactics.update(self.techniques[tech_id].tactic_refs)

        complexity_score = len(unique_tactics) + len(techniques)

        if complexity_score >= 15:
            return "high"
        elif complexity_score >= 8:
            return "medium"
        else:
            return "low"

    def _analyze_temporal_pattern(self, techniques: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns in technique sequence"""
        # Simplified temporal analysis
        return {
            "sequence_length": len(techniques),
            "unique_techniques": len(set(techniques)),
            "pattern_type": "sequential" if len(techniques) == len(set(techniques)) else "repeated",
            "analysis": "Basic temporal pattern analysis"
        }

    def _get_groups_using_technique(self, technique_id: str) -> List[Dict[str, Any]]:
        """Get threat groups that use a specific technique"""
        groups = []
        for group_id, group in self.groups.items():
            if technique_id in group.techniques:
                groups.append({
                    "group_id": group_id,
                    "name": group.name,
                    "aliases": group.aliases,
                    "country": group.country
                })
        return groups

    def _generate_mapping_recommendations(self, mapping_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on mapping results"""
        recommendations = []

        high_confidence_techniques = [
            t for t in mapping_result["techniques_mapped"]
            if t["confidence"] >= 70
        ]

        if high_confidence_techniques:
            recommendations.append(f"🚨 HIGH CONFIDENCE: {len(high_confidence_techniques)} techniques detected with high confidence")

        if mapping_result["attack_patterns"]:
            recommendations.append(f"⚠️ ATTACK PATTERNS: {len(mapping_result['attack_patterns'])} known attack patterns identified")

        tactics_count = len(mapping_result["tactics_identified"])
        if tactics_count >= 5:
            recommendations.append("🔍 COMPLEX ATTACK: Multiple MITRE tactics involved - investigate thoroughly")
        elif tactics_count >= 3:
            recommendations.append("⚠️ MULTI-STAGE ATTACK: Multiple attack phases detected")

        recommendations.extend([
            "📋 Review MITRE mitigations for detected techniques",
            "🛡️ Implement detection rules for identified patterns",
            "📊 Monitor for additional techniques from same tactics"
        ])

        return recommendations

    def _generate_sequence_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for technique sequence analysis"""
        recommendations = []

        if analysis_result["threat_actor_matches"]:
            top_match = analysis_result["threat_actor_matches"][0]
            recommendations.append(f"🎯 THREAT ACTOR: Technique sequence matches {top_match['name']} ({top_match['match_percentage']:.1f}% similarity)")

        kill_chain = analysis_result["kill_chain_coverage"]
        coverage_count = sum(1 for covered in kill_chain.values() if covered)

        if coverage_count >= 5:
            recommendations.append("🚨 FULL ATTACK CYCLE: Most kill chain phases covered - sophisticated attack")
        elif coverage_count >= 3:
            recommendations.append("⚠️ ADVANCED ATTACK: Multiple kill chain phases identified")

        complexity = analysis_result["sequence_analysis"]["sequence_complexity"]
        if complexity == "high":
            recommendations.append("🔍 HIGH COMPLEXITY: Complex attack pattern requiring detailed investigation")

        recommendations.extend([
            "📈 Analyze attack timeline for pattern insights",
            "🛡️ Focus defenses on uncovered kill chain phases",
            "👥 Consider threat hunting for similar patterns"
        ])

        return recommendations

    def _generate_detection_recommendations(self, detection_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for pattern detection"""
        recommendations = []

        patterns_found = len(detection_result["patterns_detected"])
        if patterns_found > 0:
            recommendations.append(f"🚨 ACTIVE THREATS: {patterns_found} attack patterns detected in events")

        high_confidence = [
            p for p in detection_result["patterns_detected"]
            if p.get("confidence", 0) >= 80
        ]

        if high_confidence:
            recommendations.append(f"🎯 HIGH CONFIDENCE: {len(high_confidence)} patterns with high confidence scores")

        recommendations.extend([
            "🔍 Investigate timeline for attack progression",
            "🛡️ Implement blocking for detected techniques",
            "📊 Enhance monitoring for pattern techniques",
            "👥 Share IOCs with threat intelligence team"
        ])

        return recommendations

    async def _periodic_framework_update(self):
        """Periodically update MITRE framework data"""
        while True:
            try:
                await asyncio.sleep(self.update_interval.total_seconds())

                if self.last_framework_update:
                    age = datetime.utcnow() - self.last_framework_update
                    if age >= self.update_interval:
                        logger.info("Updating MITRE ATT&CK framework data")
                        await self._load_mitre_framework()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Framework update error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    async def _save_framework_data(self):
        """Save framework data to persistent storage"""
        try:
            # Save to file system or database
            data = {
                "tactics": {k: v.__dict__ for k, v in self.tactics.items()},
                "techniques": {k: v.__dict__ for k, v in self.techniques.items()},
                "groups": {k: v.__dict__ for k, v in self.groups.items()},
                "mitigations": {k: v.__dict__ for k, v in self.mitigations.items()},
                "last_update": self.last_framework_update.isoformat() if self.last_framework_update else None
            }

            # In production, this would save to database or file
            logger.info("MITRE framework data saved")

        except Exception as e:
            logger.error(f"Failed to save framework data: {e}")


# Global service instance
_mitre_service: Optional[MitreAttackService] = None

async def get_mitre_attack_service() -> MitreAttackService:
    """Get global MITRE ATT&CK service instance"""
    global _mitre_service

    if _mitre_service is None:
        _mitre_service = MitreAttackService()
        await _mitre_service.initialize()

        # Register with global service registry
        from .base_service import service_registry
        service_registry.register(_mitre_service)

    return _mitre_service
