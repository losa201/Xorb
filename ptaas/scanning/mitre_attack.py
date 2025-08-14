import json
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MITREAttackMapper:
    """Class to map vulnerabilities to MITRE ATT&CK techniques"""

    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the MITRE ATT&CK mapper

        Args:
            mapping_file: Path to custom mapping file. If None, uses default mappings.
        """
        self.attack_data = self._load_attack_data(mapping_file)
        self.technique_index = self._build_technique_index()
        self.tactic_index = self._build_tactic_index()

    def _load_attack_data(self, mapping_file: Optional[str] = None) -> Dict:
        """
        Load MITRE ATT&CK mapping data

        Returns:
            Dictionary containing MITRE ATT&CK mappings
        """
        try:
            if mapping_file and os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    return json.load(f)
            else:
                # Default mappings (simplified for demonstration)
                return {
                    "techniques": {
                        "T1190": {  # Exploit Public-Facing Application
                            "name": "Exploit Public-Facing Application",
                            "tactic": "initial_access",
                            "description": "Adversaries may exploit public-facing applications to initially access a system.",
                            "url": "https://attack.mitre.org/techniques/T1190/",
                            "related_cves": ["CVE-2021-44228", "CVE-2021-45046", "CVE-2020-1472"]
                        },
                        "T1071": {  # Application Layer Protocol
                            "name": "Application Layer Protocol",
                            "tactic": "command_and_control",
                            "description": "Adversaries may use application layer protocols for communication between compromised systems and C2 servers.",
                            "url": "https://attack.mitre.org/techniques/T1071/",
                            "related_cves": []
                        },
                        "T1059": {  # Command and Scripting Interpreter
                            "name": "Command and Scripting Interpreter",
                            "tactic": "execution",
                            "description": "Adversaries may use command and scripting interpreters to execute commands, scripts, or binaries.",
                            "url": "https://attack.mitre.org/techniques/T1059/",
                            "related_cves": []
                        },
                        "T1047": {  # Windows Management Instrumentation
                            "name": "Windows Management Instrumentation",
                            "tactic": "execution",
                            "description": "Adversaries may use Windows Management Instrumentation (WMI) to perform actions on remote systems.",
                            "url": "https://attack.mitre.org/techniques/T1047/",
                            "related_cves": []
                        },
                        "T1003": {  # Credential Dumping
                            "name": "Credential Dumping",
                            "tactic": "credential_access",
                            "description": "Adversaries may attempt to access credential material stored in various locations throughout a system.",
                            "url": "https://attack.mitre.org/techniques/T1003/",
                            "related_cves": []
                        },
                        "T1070": {  # Indicator Removal
                            "name": "Indicator Removal",
                            "tactic": "defense_evasion",
                            "description": "Adversaries may delete or modify artifacts of an intrusion to prevent detection.",
                            "url": "https://attack.mitre.org/techniques/T1070/",
                            "related_cves": []
                        },
                        "T1027": {  # Obfuscated Files or Information
                            "name": "Obfuscated Files or Information",
                            "tactic": "defense_evasion",
                            "description": "Adversaries may attempt to make an executable or document difficult to analyze by obfuscating its contents.",
                            "url": "https://attack.mitre.org/techniques/T1027/",
                            "related_cves": []
                        },
                        "T1055": {  # Process Injection
                            "name": "Process Injection",
                            "tactic": "execution",
                            "description": "Adversaries may inject code into processes to evade process-based defenses and hide execution.",
                            "url": "https://attack.mitre.org/techniques/T1055/",
                            "related_cves": []
                        },
                        "T1016": {  # System Network Configuration Discovery
                            "name": "System Network Configuration Discovery",
                            "tactic": "discovery",
                            "description": "Adversaries may look to discover information about the network configuration of systems they are accessing.",
                            "url": "https://attack.mitre.org/techniques/T1016/",
                            "related_cves": []
                        },
                        "T1082": {  # System Information Discovery
                            "name": "System Information Discovery",
                            "tactic": "discovery",
                            "description": "Adversaries may gather information about the system to identify potential weaknesses.",
                            "url": "https://attack.mitre.org/techniques/T1082/",
                            "related_cves": []
                        }
                    },
                    "tactics": {
                        "initial_access": {
                            "name": "Initial Access",
                            "description": "Techniques used to initially access a system.",
                            "url": "https://attack.mitre.org/tactics/TA0001/"
                        },
                        "execution": {
                            "name": "Execution",
                            "description": "Techniques used to execute arbitrary code.",
                            "url": "https://attack.mitre.org/tactics/TA0002/"
                        },
                        "persistence": {
                            "name": "Persistence",
                            "description": "Techniques used to maintain access to systems.",
                            "url": "https://attack.mitre.org/tactics/TA0003/"
                        },
                        "privilege_escalation": {
                            "name": "Privilege Escalation",
                            "description": "Techniques used to gain higher-level permissions.",
                            "url": "https://attack.mitre.org/tactics/TA0004/"
                        },
                        "defense_evasion": {
                            "name": "Defense Evasion",
                            "description": "Techniques used to avoid detection.",
                            "url": "https://attack.mitre.org/tactics/TA0005/"
                        },
                        "credential_access": {
                            "name": "Credential Access",
                            "description": "Techniques used to steal credentials.",
                            "url": "https://attack.mitre.org/tactics/TA0006/"
                        },
                        "discovery": {
                            "name": "Discovery",
                            "description": "Techniques used to gather information about systems.",
                            "url": "https://attack.mitre.org/tactics/TA0007/"
                        },
                        "lateral_movement": {
                            "name": "Lateral Movement",
                            "description": "Techniques used to move through networks.",
                            "url": "https://attack.mitre.org/tactics/TA0008/"
                        },
                        "collection": {
                            "name": "Collection",
                            "description": "Techniques used to gather data from systems.",
                            "url": "https://attack.mitre.org/tactics/TA0009/"
                        },
                        "command_and_control": {
                            "name": "Command and Control",
                            "description": "Techniques used to communicate with compromised systems.",
                            "url": "https://attack.mitre.org/tactics/TA0011/"
                        },
                        "exfiltration": {
                            "name": "Exfiltration",
                            "description": "Techniques used to steal data from systems.",
                            "url": "https://attack.mitre.org/tactics/TA0010/"
                        },
                        "impact": {
                            "name": "Impact",
                            "description": "Techniques used to disrupt systems.",
                            "url": "https://attack.mitre.org/tactics/TA0040/"
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error loading MITRE ATT&CK data: {str(e)}")
            return {"techniques": {}, "tactics": {}}

    def _build_technique_index(self) -> Dict[str, str]:
        """Build an index of CVEs to techniques"""
        index = {}
        for tech_id, tech_data in self.attack_data["techniques"].items():
            for cve in tech_data.get("related_cves", []):
                index[cve.lower()] = tech_id
        return index

    def _build_tactic_index(self) -> Dict[str, List[str]]:
        """Build an index of tactics to techniques"""
        index = {}
        for tech_id, tech_data in self.attack_data["techniques"].items():
            tactic = tech_data["tactic"]
            if tactic not in index:
                index[tactic] = []
            index[tactic].append(tech_id)
        return index

    def map_vulnerability(self, vulnerability: Dict) -> Dict:
        """
        Map a vulnerability to MITRE ATT&CK techniques

        Args:
            vulnerability: Dictionary containing vulnerability data

        Returns:
            Dictionary containing MITRE ATT&CK mappings
        """
        try:
            # Try to map by CVE first
            if vulnerability.get("references"):
                for ref in vulnerability["references"]:
                    if ref.startswith("CVE-"):
                        cve_id = ref.split(":")[0]  # Take just the CVE ID
                        if cve_id.lower() in self.technique_index:
                            tech_id = self.technique_index[cve_id.lower()]
                            return {
                                "technique_id": tech_id,
                                "technique_name": self.attack_data["techniques"][tech_id]["name"],
                                "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                                "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                                "url": self.attack_data["techniques"][tech_id]["url"]
                            }

            # If no CVE mapping, try to map by vulnerability title/description
            title = vulnerability.get("title", "").lower()
            description = vulnerability.get("description", "").lower()

            # Map to common techniques based on keywords
            if "exploit" in title or "public-facing" in description:
                tech_id = "T1190"
                return {
                    "technique_id": tech_id,
                    "technique_name": self.attack_data["techniques"][tech_id]["name"],
                    "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                    "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                    "url": self.attack_data["techniques"][tech_id]["url"]
                }
            elif "command" in title or "script" in description or "interpreter" in description:
                tech_id = "T1059"
                return {
                    "technique_id": tech_id,
                    "technique_name": self.attack_data["techniques"][tech_id]["name"],
                    "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                    "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                    "url": self.attack_data["techniques"][tech_id]["url"]
                }
            elif "credential" in title or "dump" in description:
                tech_id = "T1003"
                return {
                    "technique_id": tech_id,
                    "technique_name": self.attack_data["techniques"][tech_id]["name"],
                    "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                    "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                    "url": self.attack_data["techniques"][tech_id]["url"]
                }
            elif "obfusc" in title or "encode" in description:
                tech_id = "T1027"
                return {
                    "technique_id": tech_id,
                    "technique_name": self.attack_data["techniques"][tech_id]["name"],
                    "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                    "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                    "url": self.attack_data["techniques"][tech_id]["url"]
                }
            elif "network" in title or "configuration" in description:
                tech_id = "T1016"
                return {
                    "technique_id": tech_id,
                    "technique_name": self.attack_data["techniques"][tech_id]["name"],
                    "tactic": self.attack_data["techniques"][tech_id]["tactic"],
                    "tactic_name": self.attack_data["tactics"][self.attack_data["techniques"][tech_id]["tactic"]]["name"],
                    "url": self.attack_data["techniques"][tech_id]["url"]
                }

            # Default fallback
            return {
                "technique_id": None,
                "technique_name": None,
                "tactic": None,
                "tactic_name": None,
                "url": None
            }

        except Exception as e:
            logger.error(f"Error mapping vulnerability to MITRE ATT&CK: {str(e)}")
            return {
                "technique_id": None,
                "technique_name": None,
                "tactic": None,
                "tactic_name": None,
                "url": None
            }

    def get_technique_details(self, technique_id: str) -> Dict:
        """
        Get details about a specific MITRE ATT&CK technique

        Args:
            technique_id: MITRE ATT&CK technique ID (e.g., "T1190")

        Returns:
            Dictionary containing technique details
        """
        if technique_id in self.attack_data["techniques"]:
            return self.attack_data["techniques"][technique_id]
        return {}

    def get_tactic_details(self, tactic_id: str) -> Dict:
        """
        Get details about a specific MITRE ATT&CK tactic

        Args:
            tactic_id: MITRE ATT&CK tactic ID (e.g., "initial_access")

        Returns:
            Dictionary containing tactic details
        """
        if tactic_id in self.attack_data["tactics"]:
            return self.attack_data["tactics"][tactic_id]
        return {}

    def get_all_techniques(self) -> Dict:
        """Get all MITRE ATT&CK techniques"""
        return self.attack_data.get("techniques", {})

    def get_all_tactics(self) -> Dict:
        """Get all MITRE ATT&CK tactics"""
        return self.attack_data.get("tactics", {})

    def get_techniques_by_tactic(self, tactic_id: str) -> List[Dict]:
        """
        Get all techniques associated with a specific tactic

        Args:
            tactic_id: MITRE ATT&CK tactic ID (e.g., "initial_access")

        Returns:
            List of technique dictionaries
        """
        techniques = []
        if tactic_id in self.tactic_index:
            for tech_id in self.tactic_index[tactic_id]:
                tech_data = self.attack_data["techniques"].get(tech_id)
                if tech_data:
                    techniques.append(tech_data)
        return techniques

    def search_techniques(self, query: str) -> List[Dict]:
        """
        Search MITRE ATT&CK techniques by keyword

        Args:
            query: Search term

        Returns:
            List of matching technique dictionaries
        """
        query = query.lower()
        results = []

        for tech_id, tech_data in self.attack_data["techniques"].items():
            if (query in tech_data["name"].lower() or
                query in tech_data["description"].lower()):
                results.append(tech_data)

        return results

    def search_tactics(self, query: str) -> List[Dict]:
        """
        Search MITRE ATT&CK tactics by keyword

        Args:
            query: Search term

        Returns:
            List of matching tactic dictionaries
        """
        query = query.lower()
        results = []

        for tactic_id, tactic_data in self.attack_data["tactics"].items():
            if (query in tactic_data["name"].lower() or
                query in tactic_data["description"].lower()):
                results.append(tactic_data)

        return results

    def get_attack_chain(self, technique_ids: List[str]) -> Dict:
        """
        Generate an attack chain from a list of technique IDs

        Args:
            technique_ids: List of MITRE ATT&CK technique IDs

        Returns:
            Dictionary containing attack chain information
        """
        tactics = {}

        for tech_id in technique_ids:
            if tech_id in self.attack_data["techniques"]:
                tech_data = self.attack_data["techniques"][tech_id]
                tactic_id = tech_data["tactic"]

                if tactic_id not in tactics:
                    tactics[tactic_id] = {
                        "tactic": tactic_id,
                        "tactic_name": self.attack_data["tactics"][tactic_id]["name"],
                        "url": self.attack_data["tactics"][tactic_id]["url"],
                        "techniques": []
                    }

                tactics[tactic_id]["techniques"].append({
                    "technique_id": tech_id,
                    "technique_name": tech_data["name"],
                    "url": tech_data["url"]
                })

        # Sort tactics by their natural order in the attack lifecycle
        ordered_tactics = []
        for tactic_id in ["initial_access", "execution", "persistence",
                         "privilege_escalation", "defense_evasion",
                         "credential_access", "discovery", "lateral_movement",
                         "collection", "command_and_control", "exfiltration", "impact"]:
            if tactic_id in tactics:
                ordered_tactics.append(tactics[tactic_id])

        return {
            "attack_chain": ordered_tactics,
            "total_tactics": len(ordered_tactics),
            "total_techniques": sum(len(tactic["techniques"]) for tactic in ordered_tactics)
        }
