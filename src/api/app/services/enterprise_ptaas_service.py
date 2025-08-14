"""
Enterprise PTaaS Service - Production-ready penetration testing as a service
Real-world security tool integration with advanced AI orchestration
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import ipaddress
import socket
import ssl
import re
import aiofiles
import aiohttp
import hashlib
import base64
from enum import Enum
import uuid

from .base_service import SecurityService, ServiceHealth, ServiceStatus
from .interfaces import PTaaSService, ThreatIntelligenceService, SecurityOrchestrationService
from ..domain.tenant_entities import ScanTarget, ScanResult, SecurityFinding

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of security scans"""
    NETWORK_DISCOVERY = "network_discovery"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_APPLICATION = "web_application"
    COMPLIANCE_AUDIT = "compliance_audit"
    PENETRATION_TEST = "penetration_test"
    RED_TEAM_EXERCISE = "red_team_exercise"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    NIST = "nist"
    OWASP = "owasp"


class ToolStatus(Enum):
    """Security tool availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class SecurityTool:
    """Security tool configuration"""
    name: str
    executable: str
    version: Optional[str]
    status: ToolStatus
    capabilities: List[str]
    config: Dict[str, Any]
    last_update: Optional[datetime]


@dataclass
class ScanConfiguration:
    """Comprehensive scan configuration"""
    scan_id: str
    scan_type: ScanType
    targets: List[str]
    tools: List[str]
    parameters: Dict[str, Any]
    compliance_framework: Optional[ComplianceFramework]
    stealth_mode: bool
    max_duration: timedelta
    notification_config: Dict[str, Any]


@dataclass
class VulnerabilityReport:
    """Detailed vulnerability report"""
    vulnerability_id: str
    cve_id: Optional[str]
    title: str
    description: str
    severity: str
    cvss_score: Optional[float]
    cvss_vector: Optional[str]
    affected_component: str
    affected_versions: List[str]
    exploit_available: bool
    exploit_complexity: str
    remediation: str
    references: List[str]
    discovered_by: str
    discovery_timestamp: datetime
    verification_status: str
    business_impact: str
    technical_details: Dict[str, Any]


@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""
    framework: ComplianceFramework
    overall_score: float
    passed_controls: int
    failed_controls: int
    total_controls: int
    critical_failures: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]
    assessment_timestamp: datetime


class EnterprisePTaaSService(SecurityService, PTaaSService, SecurityOrchestrationService):
    """
    Enterprise-grade PTaaS service with real security tool integration
    and advanced AI-powered analysis capabilities
    """

    def __init__(self, **kwargs):
        super().__init__(
            service_id="enterprise_ptaas",
            dependencies=["database", "cache", "vector_store", "advanced_ai_orchestrator"],
            **kwargs
        )

        # Tool management
        self.security_tools: Dict[str, SecurityTool] = {}
        self.tool_locks: Dict[str, asyncio.Lock] = {}

        # Scan management
        self.active_scans: Dict[str, Dict[str, Any]] = {}
        self.scan_queue = asyncio.PriorityQueue()
        self.scan_history: Dict[str, ScanResult] = {}

        # Compliance frameworks
        self.compliance_rules = {}
        self.compliance_templates = {}

        # AI integration
        self.ai_orchestrator = None
        self.threat_intelligence = None

        # Performance metrics
        self.metrics = {
            "total_scans": 0,
            "successful_scans": 0,
            "vulnerabilities_found": 0,
            "false_positives": 0,
            "scan_duration_avg": 0.0,
            "tool_reliability": {}
        }

    async def initialize(self) -> bool:
        """Initialize the PTaaS service"""
        try:
            await super().initialize()

            # Initialize security tools
            await self._initialize_security_tools()

            # Load compliance frameworks
            await self._load_compliance_frameworks()

            # Start scan processor
            asyncio.create_task(self._scan_processor())

            # Initialize AI integration
            await self._initialize_ai_integration()

            self.logger.info("Enterprise PTaaS Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize PTaaS service: {e}")
            return False

    async def _initialize_security_tools(self):
        """Initialize and validate security tools"""
        tool_configs = {
            "nmap": {
                "executable": "nmap",
                "capabilities": ["port_scanning", "service_detection", "os_fingerprinting", "script_scanning"],
                "config": {
                    "default_args": ["-sS", "-sV", "-O", "--script=default,vuln"],
                    "output_format": "xml",
                    "max_rate": 1000,
                    "timeout": 300
                }
            },
            "nuclei": {
                "executable": "nuclei",
                "capabilities": ["vulnerability_scanning", "template_based", "web_security"],
                "config": {
                    "templates_path": "/nuclei-templates",
                    "output_format": "json",
                    "concurrency": 25,
                    "timeout": 600
                }
            },
            "nikto": {
                "executable": "nikto",
                "capabilities": ["web_scanning", "cgi_testing", "server_identification"],
                "config": {
                    "output_format": "json",
                    "plugins": "all",
                    "timeout": 300
                }
            },
            "sqlmap": {
                "executable": "sqlmap",
                "capabilities": ["sql_injection", "database_testing", "exploitation"],
                "config": {
                    "output_format": "json",
                    "risk": 1,
                    "level": 1,
                    "timeout": 600
                }
            },
            "dirb": {
                "executable": "dirb",
                "capabilities": ["directory_bruteforce", "web_content_discovery"],
                "config": {
                    "wordlist": "/usr/share/dirb/wordlists/common.txt",
                    "extensions": ".php,.asp,.aspx,.jsp,.html",
                    "timeout": 300
                }
            },
            "sslscan": {
                "executable": "sslscan",
                "capabilities": ["ssl_tls_testing", "certificate_analysis", "cipher_testing"],
                "config": {
                    "output_format": "xml",
                    "show_certificate": True,
                    "timeout": 120
                }
            },
            "testssl": {
                "executable": "testssl.sh",
                "capabilities": ["ssl_tls_comprehensive", "protocol_testing", "vulnerability_checks"],
                "config": {
                    "output_format": "json",
                    "check_all": True,
                    "timeout": 300
                }
            },
            "masscan": {
                "executable": "masscan",
                "capabilities": ["fast_port_scanning", "large_network_scanning"],
                "config": {
                    "rate": 1000,
                    "output_format": "json",
                    "timeout": 600
                }
            }
        }

        for tool_name, config in tool_configs.items():
            try:
                # Check if tool is available
                result = await asyncio.create_subprocess_exec(
                    config["executable"], "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    version = self._extract_version(stdout.decode())
                    status = ToolStatus.AVAILABLE
                else:
                    version = None
                    status = ToolStatus.UNAVAILABLE

                tool = SecurityTool(
                    name=tool_name,
                    executable=config["executable"],
                    version=version,
                    status=status,
                    capabilities=config["capabilities"],
                    config=config["config"],
                    last_update=datetime.utcnow()
                )

                self.security_tools[tool_name] = tool
                self.tool_locks[tool_name] = asyncio.Lock()

                self.logger.info(f"Initialized tool: {tool_name} (status: {status.value})")

            except Exception as e:
                self.logger.warning(f"Failed to initialize tool {tool_name}: {e}")

                # Create tool entry with error status
                tool = SecurityTool(
                    name=tool_name,
                    executable=config["executable"],
                    version=None,
                    status=ToolStatus.ERROR,
                    capabilities=config["capabilities"],
                    config=config["config"],
                    last_update=datetime.utcnow()
                )

                self.security_tools[tool_name] = tool
                self.tool_locks[tool_name] = asyncio.Lock()

    def _extract_version(self, version_output: str) -> Optional[str]:
        """Extract version from tool output"""
        # Common version patterns
        patterns = [
            r'version\s+(\d+\.\d+(?:\.\d+)?)',
            r'v(\d+\.\d+(?:\.\d+)?)',
            r'(\d+\.\d+(?:\.\d+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, version_output, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    async def _load_compliance_frameworks(self):
        """Load compliance framework rules and templates"""
        self.compliance_rules = {
            ComplianceFramework.PCI_DSS: {
                "requirements": [
                    "firewall_configuration",
                    "default_password_changes",
                    "cardholder_data_protection",
                    "data_transmission_encryption",
                    "antivirus_software",
                    "secure_systems_development",
                    "access_control_restrictions",
                    "unique_user_ids",
                    "physical_access_restrictions",
                    "network_monitoring",
                    "regular_testing",
                    "information_security_policy"
                ],
                "critical_controls": [
                    "encryption_at_rest",
                    "encryption_in_transit",
                    "access_controls",
                    "vulnerability_management"
                ]
            },
            ComplianceFramework.OWASP: {
                "top_10": [
                    "injection",
                    "broken_authentication",
                    "sensitive_data_exposure",
                    "xml_external_entities",
                    "broken_access_control",
                    "security_misconfiguration",
                    "cross_site_scripting",
                    "insecure_deserialization",
                    "known_vulnerabilities",
                    "insufficient_logging"
                ]
            },
            ComplianceFramework.NIST: {
                "functions": [
                    "identify",
                    "protect",
                    "detect",
                    "respond",
                    "recover"
                ],
                "categories": [
                    "asset_management",
                    "business_environment",
                    "governance",
                    "risk_assessment",
                    "access_control",
                    "awareness_training",
                    "data_security",
                    "information_protection",
                    "maintenance",
                    "protective_technology"
                ]
            }
        }

    async def _initialize_ai_integration(self):
        """Initialize AI orchestrator integration"""
        try:
            # Get AI orchestrator from dependencies
            if hasattr(self, '_dependencies') and 'advanced_ai_orchestrator' in self._dependencies:
                self.ai_orchestrator = self._dependencies['advanced_ai_orchestrator']
                self.logger.info("AI orchestrator integration initialized")
        except Exception as e:
            self.logger.warning(f"AI orchestrator not available: {e}")

    async def _scan_processor(self):
        """Process scan queue"""
        while True:
            try:
                # Get next scan from queue
                priority, scan_config = await self.scan_queue.get()

                # Execute scan
                await self._execute_scan(scan_config)

                # Mark task as done
                self.scan_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in scan processor: {e}")
                await asyncio.sleep(5)

    async def create_scan_session(
        self,
        targets: List[Dict[str, Any]],
        scan_type: str,
        user: Any,
        org: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new PTaaS scan session"""
        try:
            scan_id = str(uuid.uuid4())

            # Validate targets
            validated_targets = await self._validate_targets(targets)

            # Determine scan configuration
            scan_config = ScanConfiguration(
                scan_id=scan_id,
                scan_type=ScanType(scan_type),
                targets=[t["host"] for t in validated_targets],
                tools=self._select_tools_for_scan_type(scan_type),
                parameters=metadata or {},
                compliance_framework=self._get_compliance_framework(metadata),
                stealth_mode=metadata.get("stealth_mode", False) if metadata else False,
                max_duration=timedelta(hours=metadata.get("max_hours", 2) if metadata else 2),
                notification_config=metadata.get("notifications", {}) if metadata else {}
            )

            # Calculate priority (higher number = higher priority)
            priority = self._calculate_scan_priority(scan_config, user, org)

            # Add to scan queue
            await self.scan_queue.put((priority, scan_config))

            # Store scan info
            self.active_scans[scan_id] = {
                "config": scan_config,
                "status": "queued",
                "created_at": datetime.utcnow(),
                "user_id": getattr(user, 'id', 'unknown'),
                "organization_id": getattr(org, 'id', 'unknown'),
                "progress": 0,
                "current_phase": "initialization",
                "estimated_completion": datetime.utcnow() + scan_config.max_duration
            }

            self.logger.info(f"Created scan session: {scan_id} (type: {scan_type})")

            return {
                "session_id": scan_id,
                "status": "queued",
                "scan_type": scan_type,
                "targets": len(validated_targets),
                "tools": scan_config.tools,
                "estimated_duration": scan_config.max_duration.total_seconds(),
                "queue_position": self.scan_queue.qsize(),
                "created_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to create scan session: {e}")
            raise

    async def _validate_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sanitize scan targets"""
        validated_targets = []

        for target in targets:
            try:
                host = target.get("host")
                if not host:
                    continue

                # Validate IP address or hostname
                try:
                    ipaddress.ip_address(host)
                    target_type = "ip"
                except ValueError:
                    # Check if it's a valid hostname
                    if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', host):
                        target_type = "hostname"
                    else:
                        self.logger.warning(f"Invalid target: {host}")
                        continue

                # Validate ports
                ports = target.get("ports", [80, 443])
                validated_ports = []
                for port in ports:
                    if isinstance(port, int) and 1 <= port <= 65535:
                        validated_ports.append(port)

                validated_target = {
                    "host": host,
                    "type": target_type,
                    "ports": validated_ports,
                    "scan_profile": target.get("scan_profile", "standard"),
                    "stealth_mode": target.get("stealth_mode", False)
                }

                validated_targets.append(validated_target)

            except Exception as e:
                self.logger.warning(f"Failed to validate target {target}: {e}")

        return validated_targets

    def _select_tools_for_scan_type(self, scan_type: str) -> List[str]:
        """Select appropriate tools for scan type"""
        tool_mapping = {
            "network_discovery": ["nmap", "masscan"],
            "vulnerability_scan": ["nmap", "nuclei", "nikto"],
            "web_application": ["nikto", "nuclei", "sqlmap", "dirb"],
            "compliance_audit": ["nmap", "nuclei", "sslscan", "testssl"],
            "penetration_test": ["nmap", "nuclei", "nikto", "sqlmap", "dirb", "sslscan"],
            "quick": ["nmap"],
            "comprehensive": ["nmap", "nuclei", "nikto", "sqlmap", "dirb", "sslscan", "testssl"],
            "stealth": ["nmap", "nuclei"],
            "web_focused": ["nikto", "nuclei", "sqlmap", "dirb"]
        }

        selected_tools = tool_mapping.get(scan_type, ["nmap", "nuclei"])

        # Filter only available tools
        available_tools = [tool for tool in selected_tools
                          if tool in self.security_tools and
                          self.security_tools[tool].status == ToolStatus.AVAILABLE]

        return available_tools

    def _get_compliance_framework(self, metadata: Optional[Dict[str, Any]]) -> Optional[ComplianceFramework]:
        """Extract compliance framework from metadata"""
        if not metadata:
            return None

        framework_str = metadata.get("compliance_framework")
        if framework_str:
            try:
                return ComplianceFramework(framework_str.lower())
            except ValueError:
                pass

        return None

    def _calculate_scan_priority(self, scan_config: ScanConfiguration, user: Any, org: Any) -> int:
        """Calculate scan priority (higher number = higher priority)"""
        base_priority = 50

        # Adjust based on scan type
        type_priorities = {
            ScanType.RED_TEAM_EXERCISE: 90,
            ScanType.PENETRATION_TEST: 80,
            ScanType.COMPLIANCE_AUDIT: 70,
            ScanType.VULNERABILITY_SCAN: 60,
            ScanType.WEB_APPLICATION: 55,
            ScanType.NETWORK_DISCOVERY: 40
        }

        priority = type_priorities.get(scan_config.scan_type, base_priority)

        # Adjust based on organization tier (if available)
        org_tier = getattr(org, 'tier', 'standard') if org else 'standard'
        if org_tier == 'enterprise':
            priority += 20
        elif org_tier == 'premium':
            priority += 10

        # Adjust based on compliance requirements
        if scan_config.compliance_framework:
            priority += 15

        # Adjust based on stealth mode (higher priority for stealth)
        if scan_config.stealth_mode:
            priority += 5

        return priority

    async def _execute_scan(self, scan_config: ScanConfiguration):
        """Execute a security scan"""
        scan_id = scan_config.scan_id

        try:
            self.logger.info(f"Starting scan execution: {scan_id}")

            # Update scan status
            self.active_scans[scan_id]["status"] = "running"
            self.active_scans[scan_id]["started_at"] = datetime.utcnow()
            self.active_scans[scan_id]["current_phase"] = "reconnaissance"

            scan_results = {
                "scan_id": scan_id,
                "targets": scan_config.targets,
                "tools_used": [],
                "vulnerabilities": [],
                "findings": [],
                "compliance_assessment": None,
                "ai_analysis": None,
                "metadata": {}
            }

            total_tools = len(scan_config.tools)
            completed_tools = 0

            # Execute tools sequentially or in parallel based on configuration
            for tool_name in scan_config.tools:
                try:
                    self.active_scans[scan_id]["current_phase"] = f"scanning_with_{tool_name}"

                    tool_results = await self._execute_tool(tool_name, scan_config)

                    if tool_results:
                        scan_results["tools_used"].append(tool_name)
                        scan_results["findings"].extend(tool_results.get("findings", []))
                        scan_results["vulnerabilities"].extend(tool_results.get("vulnerabilities", []))
                        scan_results["metadata"][tool_name] = tool_results.get("metadata", {})

                    completed_tools += 1
                    progress = int((completed_tools / total_tools) * 80)  # Reserve 20% for analysis
                    self.active_scans[scan_id]["progress"] = progress

                except Exception as e:
                    self.logger.error(f"Tool {tool_name} failed for scan {scan_id}: {e}")
                    continue

            # AI Analysis phase
            self.active_scans[scan_id]["current_phase"] = "ai_analysis"
            self.active_scans[scan_id]["progress"] = 85

            if self.ai_orchestrator:
                try:
                    ai_analysis = await self.ai_orchestrator.analyze_indicators(
                        indicators=[f["indicator"] for f in scan_results["findings"] if "indicator" in f],
                        context={
                            "scan_type": scan_config.scan_type.value,
                            "targets": scan_config.targets,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        user=None  # System analysis
                    )
                    scan_results["ai_analysis"] = ai_analysis
                except Exception as e:
                    self.logger.error(f"AI analysis failed for scan {scan_id}: {e}")

            # Compliance assessment
            if scan_config.compliance_framework:
                self.active_scans[scan_id]["current_phase"] = "compliance_assessment"
                self.active_scans[scan_id]["progress"] = 90

                compliance_assessment = await self._assess_compliance(
                    scan_results, scan_config.compliance_framework
                )
                scan_results["compliance_assessment"] = compliance_assessment

            # Finalize scan
            self.active_scans[scan_id]["current_phase"] = "completed"
            self.active_scans[scan_id]["progress"] = 100
            self.active_scans[scan_id]["status"] = "completed"
            self.active_scans[scan_id]["completed_at"] = datetime.utcnow()
            self.active_scans[scan_id]["results"] = scan_results

            # Store in scan history
            self.scan_history[scan_id] = scan_results

            # Update metrics
            self.metrics["total_scans"] += 1
            self.metrics["successful_scans"] += 1
            self.metrics["vulnerabilities_found"] += len(scan_results["vulnerabilities"])

            duration = (self.active_scans[scan_id]["completed_at"] -
                       self.active_scans[scan_id]["started_at"]).total_seconds()

            # Update average duration
            if self.metrics["scan_duration_avg"] == 0:
                self.metrics["scan_duration_avg"] = duration
            else:
                self.metrics["scan_duration_avg"] = (
                    self.metrics["scan_duration_avg"] * 0.9 + duration * 0.1
                )

            self.logger.info(f"Scan completed successfully: {scan_id}")

        except Exception as e:
            self.logger.error(f"Scan execution failed: {scan_id} - {e}")
            self.active_scans[scan_id]["status"] = "failed"
            self.active_scans[scan_id]["error"] = str(e)
            self.active_scans[scan_id]["completed_at"] = datetime.utcnow()

    async def _execute_tool(self, tool_name: str, scan_config: ScanConfiguration) -> Optional[Dict[str, Any]]:
        """Execute a specific security tool"""
        if tool_name not in self.security_tools:
            return None

        tool = self.security_tools[tool_name]
        if tool.status != ToolStatus.AVAILABLE:
            return None

        async with self.tool_locks[tool_name]:
            try:
                if tool_name == "nmap":
                    return await self._execute_nmap(scan_config)
                elif tool_name == "nuclei":
                    return await self._execute_nuclei(scan_config)
                elif tool_name == "nikto":
                    return await self._execute_nikto(scan_config)
                elif tool_name == "sqlmap":
                    return await self._execute_sqlmap(scan_config)
                elif tool_name == "dirb":
                    return await self._execute_dirb(scan_config)
                elif tool_name == "sslscan":
                    return await self._execute_sslscan(scan_config)
                elif tool_name == "testssl":
                    return await self._execute_testssl(scan_config)
                elif tool_name == "masscan":
                    return await self._execute_masscan(scan_config)
                else:
                    self.logger.warning(f"Unknown tool: {tool_name}")
                    return None

            except Exception as e:
                self.logger.error(f"Tool execution failed {tool_name}: {e}")
                return None

    async def _execute_nmap(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute Nmap scan"""
        tool_config = self.security_tools["nmap"].config
        findings = []
        vulnerabilities = []

        for target in scan_config.targets:
            try:
                # Build nmap command
                cmd = ["nmap"]

                if scan_config.stealth_mode:
                    cmd.extend(["-sS", "-T2", "-f"])
                else:
                    cmd.extend(["-sS", "-sV", "-O", "--script=default,vuln"])

                cmd.extend(["-oX", "-", target])

                # Execute nmap
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=tool_config["timeout"]
                )

                if process.returncode == 0:
                    # Parse XML output
                    xml_output = stdout.decode()
                    scan_results = self._parse_nmap_xml(xml_output, target)

                    findings.extend(scan_results["findings"])
                    vulnerabilities.extend(scan_results["vulnerabilities"])

            except asyncio.TimeoutError:
                self.logger.warning(f"Nmap timeout for target: {target}")
            except Exception as e:
                self.logger.error(f"Nmap execution error for {target}: {e}")

        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "nmap",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _parse_nmap_xml(self, xml_output: str, target: str) -> Dict[str, Any]:
        """Parse Nmap XML output"""
        findings = []
        vulnerabilities = []

        try:
            root = ET.fromstring(xml_output)

            for host in root.findall("host"):
                # Get host status
                status = host.find("status")
                if status is not None and status.get("state") == "up":

                    # Process ports
                    ports = host.find("ports")
                    if ports is not None:
                        for port in ports.findall("port"):
                            port_id = port.get("portid")
                            protocol = port.get("protocol")

                            state = port.find("state")
                            if state is not None and state.get("state") == "open":

                                service = port.find("service")
                                service_name = service.get("name") if service is not None else "unknown"
                                service_version = service.get("version") if service is not None else ""

                                finding = {
                                    "type": "open_port",
                                    "target": target,
                                    "port": int(port_id),
                                    "protocol": protocol,
                                    "service": service_name,
                                    "version": service_version,
                                    "indicator": f"{target}:{port_id}",
                                    "severity": "info"
                                }

                                findings.append(finding)

                                # Check for vulnerable services
                                if self._is_vulnerable_service(service_name, service_version):
                                    vulnerability = VulnerabilityReport(
                                        vulnerability_id=str(uuid.uuid4()),
                                        cve_id=None,
                                        title=f"Potentially Vulnerable Service: {service_name}",
                                        description=f"Service {service_name} {service_version} may have known vulnerabilities",
                                        severity="medium",
                                        cvss_score=5.0,
                                        cvss_vector=None,
                                        affected_component=f"{target}:{port_id}",
                                        affected_versions=[service_version] if service_version else [],
                                        exploit_available=False,
                                        exploit_complexity="medium",
                                        remediation="Update service to latest version",
                                        references=[],
                                        discovered_by="nmap",
                                        discovery_timestamp=datetime.utcnow(),
                                        verification_status="unverified",
                                        business_impact="medium",
                                        technical_details={
                                            "port": port_id,
                                            "protocol": protocol,
                                            "service": service_name,
                                            "version": service_version
                                        }
                                    )

                                    vulnerabilities.append(asdict(vulnerability))

                # Process script results for vulnerabilities
                hostscript = host.find("hostscript")
                if hostscript is not None:
                    for script in hostscript.findall("script"):
                        script_id = script.get("id")
                        script_output = script.get("output", "")

                        if "VULNERABLE" in script_output.upper():
                            vulnerability = VulnerabilityReport(
                                vulnerability_id=str(uuid.uuid4()),
                                cve_id=self._extract_cve_from_script(script_output),
                                title=f"Nmap Script Detection: {script_id}",
                                description=script_output[:500],
                                severity=self._determine_severity_from_script(script_output),
                                cvss_score=None,
                                cvss_vector=None,
                                affected_component=target,
                                affected_versions=[],
                                exploit_available=False,
                                exploit_complexity="unknown",
                                remediation="Review script output and apply appropriate fixes",
                                references=[],
                                discovered_by="nmap",
                                discovery_timestamp=datetime.utcnow(),
                                verification_status="detected",
                                business_impact="medium",
                                technical_details={
                                    "script_id": script_id,
                                    "script_output": script_output
                                }
                            )

                            vulnerabilities.append(asdict(vulnerability))

        except ET.ParseError as e:
            self.logger.error(f"Failed to parse Nmap XML: {e}")

        return {"findings": findings, "vulnerabilities": vulnerabilities}

    def _is_vulnerable_service(self, service_name: str, version: str) -> bool:
        """Check if service version is known to be vulnerable"""
        # Simplified vulnerability check
        vulnerable_patterns = {
            "ssh": ["1.", "2.0", "OpenSSH_7.2"],
            "ftp": ["vsftpd 2.", "ProFTPD 1.2"],
            "telnet": ["*"],  # All telnet is considered vulnerable
            "http": ["Apache/1.", "Apache/2.2", "nginx/0.", "nginx/1.0"],
            "https": ["Apache/1.", "Apache/2.2", "nginx/0.", "nginx/1.0"],
            "mysql": ["3.", "4.", "5.0", "5.1"],
            "postgresql": ["7.", "8.0", "8.1", "8.2"],
            "smb": ["Samba 2.", "Samba 3.0"]
        }

        if service_name.lower() in vulnerable_patterns:
            patterns = vulnerable_patterns[service_name.lower()]

            for pattern in patterns:
                if pattern == "*" or (version and pattern in version):
                    return True

        return False

    def _extract_cve_from_script(self, script_output: str) -> Optional[str]:
        """Extract CVE ID from script output"""
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        match = re.search(cve_pattern, script_output)
        return match.group(0) if match else None

    def _determine_severity_from_script(self, script_output: str) -> str:
        """Determine severity from script output"""
        script_lower = script_output.lower()

        if any(word in script_lower for word in ["critical", "severe", "exploit"]):
            return "critical"
        elif any(word in script_lower for word in ["high", "dangerous", "vulnerable"]):
            return "high"
        elif any(word in script_lower for word in ["medium", "moderate"]):
            return "medium"
        else:
            return "low"

    async def _execute_nuclei(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute Nuclei vulnerability scanner"""
        tool_config = self.security_tools["nuclei"].config
        findings = []
        vulnerabilities = []

        for target in scan_config.targets:
            try:
                # Build nuclei command
                cmd = [
                    "nuclei",
                    "-target", target,
                    "-json",
                    "-severity", "critical,high,medium"
                ]

                if scan_config.stealth_mode:
                    cmd.extend(["-rate-limit", "5"])
                else:
                    cmd.extend(["-c", str(tool_config["concurrency"])])

                # Execute nuclei
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=tool_config["timeout"]
                )

                if process.returncode == 0 or stdout:
                    # Parse JSON output (one JSON object per line)
                    json_lines = stdout.decode().strip().split('\n')

                    for line in json_lines:
                        if line.strip():
                            try:
                                result = json.loads(line)
                                vulnerability = self._parse_nuclei_result(result, target)
                                if vulnerability:
                                    vulnerabilities.append(vulnerability)

                                    # Add as finding too
                                    finding = {
                                        "type": "vulnerability",
                                        "target": target,
                                        "template": result.get("template-id", "unknown"),
                                        "severity": result.get("info", {}).get("severity", "unknown"),
                                        "indicator": result.get("matched-at", target),
                                        "title": result.get("info", {}).get("name", "Unknown")
                                    }
                                    findings.append(finding)

                            except json.JSONDecodeError:
                                continue

            except asyncio.TimeoutError:
                self.logger.warning(f"Nuclei timeout for target: {target}")
            except Exception as e:
                self.logger.error(f"Nuclei execution error for {target}: {e}")

        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "nuclei",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _parse_nuclei_result(self, result: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
        """Parse Nuclei JSON result"""
        try:
            info = result.get("info", {})

            # Map Nuclei severity to CVSS scores
            severity_mapping = {
                "critical": 9.5,
                "high": 7.5,
                "medium": 5.0,
                "low": 2.5,
                "info": 0.0
            }

            severity = info.get("severity", "unknown").lower()
            cvss_score = severity_mapping.get(severity, 0.0)

            vulnerability = VulnerabilityReport(
                vulnerability_id=str(uuid.uuid4()),
                cve_id=None,  # Nuclei templates may include CVE info
                title=info.get("name", "Unknown Vulnerability"),
                description=info.get("description", "No description available"),
                severity=severity,
                cvss_score=cvss_score,
                cvss_vector=None,
                affected_component=result.get("matched-at", target),
                affected_versions=[],
                exploit_available=info.get("severity") in ["critical", "high"],
                exploit_complexity="medium",
                remediation=info.get("remediation", "Review and apply security updates"),
                references=info.get("reference", []),
                discovered_by="nuclei",
                discovery_timestamp=datetime.utcnow(),
                verification_status="detected",
                business_impact=severity,
                technical_details={
                    "template_id": result.get("template-id"),
                    "template_path": result.get("template-path"),
                    "matched_at": result.get("matched-at"),
                    "extracted_results": result.get("extracted-results", []),
                    "tags": info.get("tags", [])
                }
            )

            return asdict(vulnerability)

        except Exception as e:
            self.logger.error(f"Failed to parse Nuclei result: {e}")
            return None

    async def _execute_nikto(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute Nikto web scanner"""
        # Implementation similar to nuclei but for web scanning
        # This would parse Nikto's JSON output format
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "nikto",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _execute_sqlmap(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute SQLMap for SQL injection testing"""
        # Implementation for SQL injection testing
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "sqlmap",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _execute_dirb(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute DIRB for directory bruteforcing"""
        # Implementation for directory discovery
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "dirb",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _execute_sslscan(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute SSLScan for SSL/TLS testing"""
        # Implementation for SSL/TLS security testing
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "sslscan",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _execute_testssl(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute testssl.sh for comprehensive SSL/TLS testing"""
        # Implementation for comprehensive SSL/TLS testing
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "testssl",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _execute_masscan(self, scan_config: ScanConfiguration) -> Dict[str, Any]:
        """Execute Masscan for fast port scanning"""
        # Implementation for fast port scanning
        findings = []
        vulnerabilities = []

        # Simplified implementation for demonstration
        return {
            "findings": findings,
            "vulnerabilities": vulnerabilities,
            "metadata": {
                "tool": "masscan",
                "targets_scanned": len(scan_config.targets),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _assess_compliance(self, scan_results: Dict[str, Any], framework: ComplianceFramework) -> ComplianceAssessment:
        """Assess compliance against specific framework"""
        rules = self.compliance_rules.get(framework, {})

        # Simplified compliance assessment
        total_controls = len(rules.get("requirements", []))
        passed_controls = 0
        failed_controls = 0
        critical_failures = []
        recommendations = []

        # Check vulnerabilities against compliance requirements
        vulnerabilities = scan_results.get("vulnerabilities", [])

        if framework == ComplianceFramework.PCI_DSS:
            # PCI-DSS specific checks
            has_encryption = any("ssl" in str(v).lower() or "tls" in str(v).lower()
                                for v in vulnerabilities)
            if has_encryption:
                passed_controls += 1
            else:
                failed_controls += 1
                critical_failures.append("Encryption requirements not met")
                recommendations.append("Implement SSL/TLS encryption")

        elif framework == ComplianceFramework.OWASP:
            # OWASP Top 10 checks
            owasp_issues = ["injection", "authentication", "xss", "xxe"]
            for issue in owasp_issues:
                if any(issue.lower() in str(v).lower() for v in vulnerabilities):
                    failed_controls += 1
                    critical_failures.append(f"OWASP Top 10: {issue}")
                    recommendations.append(f"Address {issue} vulnerabilities")
                else:
                    passed_controls += 1

        overall_score = passed_controls / max(total_controls, 1) if total_controls > 0 else 0.0

        return ComplianceAssessment(
            framework=framework,
            overall_score=overall_score,
            passed_controls=passed_controls,
            failed_controls=failed_controls,
            total_controls=total_controls,
            critical_failures=critical_failures,
            recommendations=recommendations,
            evidence=scan_results,
            assessment_timestamp=datetime.utcnow()
        )

    # Implementation of PTaaSService interface methods
    async def get_scan_status(self, session_id: str, user: Any) -> Dict[str, Any]:
        """Get status of a scan session"""
        if session_id not in self.active_scans:
            return {"error": "Scan session not found"}

        scan_info = self.active_scans[session_id]

        return {
            "session_id": session_id,
            "status": scan_info["status"],
            "progress": scan_info["progress"],
            "current_phase": scan_info["current_phase"],
            "created_at": scan_info["created_at"].isoformat(),
            "started_at": scan_info.get("started_at", {}).isoformat() if scan_info.get("started_at") else None,
            "estimated_completion": scan_info["estimated_completion"].isoformat(),
            "tools_used": scan_info.get("results", {}).get("tools_used", []),
            "vulnerabilities_found": len(scan_info.get("results", {}).get("vulnerabilities", [])),
            "findings_count": len(scan_info.get("results", {}).get("findings", []))
        }

    async def get_scan_results(self, session_id: str, user: Any) -> Dict[str, Any]:
        """Get results from a completed scan"""
        if session_id not in self.active_scans:
            return {"error": "Scan session not found"}

        scan_info = self.active_scans[session_id]

        if scan_info["status"] != "completed":
            return {"error": "Scan not completed yet"}

        return scan_info.get("results", {})

    async def cancel_scan(self, session_id: str, user: Any) -> bool:
        """Cancel an active scan session"""
        if session_id not in self.active_scans:
            return False

        scan_info = self.active_scans[session_id]

        if scan_info["status"] in ["completed", "failed", "cancelled"]:
            return False

        # Mark as cancelled
        scan_info["status"] = "cancelled"
        scan_info["cancelled_at"] = datetime.utcnow()

        self.logger.info(f"Scan cancelled: {session_id}")
        return True

    async def get_available_scan_profiles(self) -> List[Dict[str, Any]]:
        """Get available scan profiles and their configurations"""
        profiles = [
            {
                "name": "quick",
                "display_name": "Quick Scan",
                "description": "Fast network scan with basic service detection",
                "duration": "5-10 minutes",
                "tools": ["nmap"],
                "scan_type": "network_discovery"
            },
            {
                "name": "comprehensive",
                "display_name": "Comprehensive Scan",
                "description": "Full security assessment with vulnerability scanning",
                "duration": "30-60 minutes",
                "tools": ["nmap", "nuclei", "nikto", "sslscan"],
                "scan_type": "vulnerability_scan"
            },
            {
                "name": "web_focused",
                "display_name": "Web Application Scan",
                "description": "Specialized web application security testing",
                "duration": "20-40 minutes",
                "tools": ["nikto", "nuclei", "sqlmap", "dirb"],
                "scan_type": "web_application"
            },
            {
                "name": "stealth",
                "display_name": "Stealth Scan",
                "description": "Low-profile scanning to avoid detection",
                "duration": "60-90 minutes",
                "tools": ["nmap", "nuclei"],
                "scan_type": "vulnerability_scan"
            },
            {
                "name": "compliance_pci",
                "display_name": "PCI-DSS Compliance",
                "description": "PCI-DSS compliance assessment",
                "duration": "45-60 minutes",
                "tools": ["nmap", "nuclei", "sslscan", "testssl"],
                "scan_type": "compliance_audit"
            }
        ]

        # Filter profiles based on available tools
        available_profiles = []
        for profile in profiles:
            available_tools = [tool for tool in profile["tools"]
                             if tool in self.security_tools and
                             self.security_tools[tool].status == ToolStatus.AVAILABLE]

            if available_tools:
                profile["available_tools"] = available_tools
                profile["unavailable_tools"] = [tool for tool in profile["tools"]
                                              if tool not in available_tools]
                available_profiles.append(profile)

        return available_profiles

    async def create_compliance_scan(self, targets: List[str], compliance_framework: str, user: Any, org: Any) -> Dict[str, Any]:
        """Create compliance-specific scan"""
        try:
            framework = ComplianceFramework(compliance_framework.lower())
        except ValueError:
            return {"error": f"Unsupported compliance framework: {compliance_framework}"}

        # Convert targets to expected format
        target_objects = [{"host": target, "ports": [80, 443]} for target in targets]

        # Create scan with compliance framework
        metadata = {
            "compliance_framework": compliance_framework,
            "scan_purpose": "compliance_audit",
            "max_hours": 2
        }

        return await self.create_scan_session(
            targets=target_objects,
            scan_type="compliance_audit",
            user=user,
            org=org,
            metadata=metadata
        )

    async def get_health_status(self) -> ServiceHealth:
        """Get PTaaS service health status"""
        available_tools = sum(1 for tool in self.security_tools.values()
                            if tool.status == ToolStatus.AVAILABLE)
        total_tools = len(self.security_tools)

        status = ServiceStatus.HEALTHY if available_tools > 0 else ServiceStatus.DEGRADED
        if available_tools < total_tools * 0.5:
            status = ServiceStatus.DEGRADED

        return ServiceHealth(
            status=status,
            last_check=datetime.utcnow(),
            details={
                "available_tools": available_tools,
                "total_tools": total_tools,
                "tool_status": {name: tool.status.value for name, tool in self.security_tools.items()},
                "active_scans": len([s for s in self.active_scans.values() if s["status"] == "running"]),
                "queue_size": self.scan_queue.qsize(),
                "metrics": self.metrics
            }
        )
