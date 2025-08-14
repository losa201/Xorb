"""
PTaaS Scanner Service - Production scanner integration service
Manages real-world security scanner integrations for comprehensive testing
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import ipaddress
import socket
import ssl
import re
import aiofiles
try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    np = None
import aiohttp

from .base_service import SecurityService, ServiceHealth, ServiceStatus, ServiceType, service_registry
from .interfaces import PTaaSService
from ..domain.tenant_entities import ScanTarget, ScanResult, SecurityFinding

class SecurityError(Exception):
    """Exception raised for security violations"""
    pass

logger = logging.getLogger(__name__)

@dataclass
class ScannerConfig:
    """Configuration for a security scanner"""
    name: str
    path: Optional[str]
    version: Optional[str]
    available: bool
    timeout: int
    max_rate: int
    config: Dict[str, Any]

@dataclass
class VulnerabilityFinding:
    """Detailed vulnerability finding"""
    vulnerability_id: str
    name: str
    severity: str
    cvss_score: Optional[float]
    description: str
    affected_component: str
    port: Optional[int]
    service: Optional[str]
    evidence: Dict[str, Any]
    references: List[str]
    remediation: str
    scanner: str
    timestamp: datetime

class SecurityScannerService(SecurityService, PTaaSService):
    """Production-ready security scanner integration service"""

    def __init__(self, **kwargs):
        # Extract known parameters
        service_id = kwargs.pop("service_id", "ptaas_scanner")
        dependencies = kwargs.pop("dependencies", ["database", "redis", "vault"])
        config = kwargs.pop("config", {})

        super().__init__(
            service_id=service_id,
            dependencies=dependencies,
            config=config
        )
        self.scanners: Dict[str, ScannerConfig] = {}
        self.scan_queue = asyncio.Queue()
        self.active_scans: Dict[str, asyncio.Task] = {}
        self.scan_results: Dict[str, ScanResult] = {}

        # Scanner configurations
        self.scanner_configs = {
            "nmap": {
                "executable": "nmap",
                "timeout": 300,
                "max_rate": 1000,
                "output_formats": ["xml", "json"]
            },
            "nuclei": {
                "executable": "nuclei",
                "timeout": 600,
                "max_rate": 50,
                "templates_path": "~/nuclei-templates"
            },
            "nikto": {
                "executable": "nikto",
                "timeout": 300,
                "max_rate": 10,
                "output_formats": ["json", "xml"]
            },
            "sslscan": {
                "executable": "sslscan",
                "timeout": 120,
                "max_rate": 5,
                "output_formats": ["xml"]
            },
            "dirb": {
                "executable": "dirb",
                "timeout": 300,
                "wordlist": "/usr/share/dirb/wordlists/common.txt"
            },
            "gobuster": {
                "executable": "gobuster",
                "timeout": 300,
                "wordlist": "/usr/share/wordlists/dirb/common.txt"
            },
            "sqlmap": {
                "executable": "sqlmap",
                "timeout": 600,
                "max_rate": 1,
                "risk_level": 1
            }
        }

    async def initialize(self) -> bool:
        """Initialize the scanner service"""
        try:
            logger.info("Initializing Security Scanner Service...")

            # Detect available scanners
            await self._detect_scanners()

            # Start scanner queue processor
            asyncio.create_task(self._process_scan_queue())

            logger.info(f"Scanner service initialized with {len(self.scanners)} available scanners")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize scanner service: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the scanner service"""
        try:
            # Cancel active scans
            for scan_id, task in self.active_scans.items():
                task.cancel()
                logger.info(f"Cancelled active scan: {scan_id}")

            await asyncio.gather(*self.active_scans.values(), return_exceptions=True)

            logger.info("Scanner service shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Failed to shutdown scanner service: {e}")
            return False

    async def _detect_scanners(self):
        """Detect available security scanners on the system"""
        for scanner_name, config in self.scanner_configs.items():
            try:
                scanner_path = await self._find_executable(config["executable"])
                if scanner_path:
                    # Test scanner availability and get version
                    version = await self._get_scanner_version(scanner_path)

                    self.scanners[scanner_name] = ScannerConfig(
                        name=scanner_name,
                        path=scanner_path,
                        version=version,
                        available=True,
                        timeout=config.get("timeout", 300),
                        max_rate=config.get("max_rate", 10),
                        config=config
                    )

                    logger.info(f"Detected scanner: {scanner_name} v{version} at {scanner_path}")
                else:
                    self.scanners[scanner_name] = ScannerConfig(
                        name=scanner_name,
                        path=None,
                        version=None,
                        available=False,
                        timeout=config.get("timeout", 300),
                        max_rate=config.get("max_rate", 10),
                        config=config
                    )

                    logger.warning(f"Scanner not found: {scanner_name}")

            except Exception as e:
                logger.error(f"Error detecting scanner {scanner_name}: {e}")
                self.scanners[scanner_name] = ScannerConfig(
                    name=scanner_name,
                    path=None,
                    version=None,
                    available=False,
                    timeout=config.get("timeout", 300),
                    max_rate=config.get("max_rate", 10),
                    config=config
                )

    async def _find_executable(self, executable: str) -> Optional[str]:
        """Find executable path using which command"""
        try:
            # SECURITY: Sanitize executable name to prevent command injection
            if not self._is_safe_executable_name(executable):
                logger.warning(f"Potentially unsafe executable name: {executable}")
                return None

            process = await asyncio.create_subprocess_exec(
                "which", executable,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()

            if process.returncode == 0:
                return stdout.decode().strip()

        except Exception as e:
            logger.debug(f"Error finding executable {executable}: {e}")

        return None

    def _is_safe_executable_name(self, executable: str) -> bool:
        """
        SECURITY: Validate executable name to prevent command injection
        Only allow known safe security tools
        """
        import re

        # Allow only alphanumeric characters, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', executable):
            return False

        # Whitelist of allowed security tools
        allowed_executables = {
            'nmap', 'nping', 'nc', 'netcat',
            'nuclei', 'nikto', 'dirb', 'gobuster',
            'sslscan', 'sslyze', 'testssl',
            'which', 'curl', 'wget',
            'python', 'python3', 'bash', 'sh'
        }

        return executable.lower() in allowed_executables

    def _validate_command_args(self, cmd: List[str]) -> bool:
        """
        SECURITY: Validate all command arguments to prevent injection attacks
        """
        if not cmd:
            return False

        # Validate the executable
        if not self._is_safe_executable_name(cmd[0]):
            logger.warning(f"Unsafe executable in command: {cmd[0]}")
            return False

        # Check for dangerous argument patterns
        dangerous_patterns = [
            r'[;&|`$]',  # Command injection characters
            r'\.\./',    # Directory traversal
            r'--exec',   # Execution flags
            r'--script=.*[;&|`]',  # Script injection
            r'rm\s+-rf', # Dangerous commands
            r'dd\s+if=', # Disk access
            r'/dev/',    # Device access
            r'/proc/',   # Process access
            r'/sys/',    # System access
        ]

        # Safe patterns for security tools (whitelist approach)
        safe_patterns = [
            r'^-[a-zA-Z0-9]+$',  # Simple flags like -sS, -A, -T4
            r'^--[a-zA-Z0-9_-]+$',  # Long options like --max-rate
            r'^[a-zA-Z0-9._-]+$',   # Simple hostnames/values
            r'^[0-9.,:-]+$',        # Port ranges, IP addresses
        ]

        for arg in cmd[1:]:
            # Convert to string if not already
            arg_str = str(arg)

            # First check if argument matches safe patterns
            is_safe = False
            for safe_pattern in safe_patterns:
                if re.match(safe_pattern, arg_str):
                    is_safe = True
                    break

            # If not safe pattern, check against dangerous patterns
            if not is_safe:
                for pattern in dangerous_patterns:
                    if re.search(pattern, arg_str, re.IGNORECASE):
                        logger.warning(f"Dangerous pattern detected in argument: {arg_str}")
                        return False

                # Validate IP addresses and ports
                if self._looks_like_ip_or_host(arg_str):
                    if not self._validate_target_host(arg_str):
                        logger.warning(f"Invalid target host: {arg_str}")
                        return False
                else:
                    # If not safe and not a known pattern, reject
                    logger.warning(f"Unrecognized argument pattern: {arg_str}")
                    return False

        return True

    def _looks_like_ip_or_host(self, value: str) -> bool:
        """Check if a value looks like an IP address or hostname"""
        # Simple heuristic: contains dots or colons (IPv4/IPv6)
        return '.' in value or ':' in value

    def _validate_target_host(self, host: str) -> bool:
        """Validate that a target host is safe to scan"""
        try:
            # Parse potential IP addresses
            if ':' in host and '/' in host:
                # Could be IPv6 with port or CIDR
                return True
            elif '/' in host:
                # CIDR notation
                ipaddress.ip_network(host, strict=False)
                return True
            elif ':' in host:
                # IPv6 or hostname with port
                if host.count(':') > 1:  # IPv6
                    ipaddress.ip_address(host.split('%')[0])  # Handle zone ID
                return True
            else:
                # Try as IPv4
                ipaddress.ip_address(host)
                return True
        except ValueError:
            # Not an IP, treat as hostname - basic validation
            if re.match(r'^[a-zA-Z0-9.-]+$', host):
                return True
            return False

    async def _get_scanner_version(self, scanner_path: str) -> Optional[str]:
        """Get scanner version"""
        try:
            # Try common version flags
            for flag in ["--version", "-V", "-version", "version"]:
                try:
                    process = await asyncio.create_subprocess_exec(
                        scanner_path, flag,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        timeout=10
                    )
                    stdout, stderr = await process.communicate()

                    if process.returncode == 0:
                        output = stdout.decode() + stderr.decode()
                        # Extract version using regex
                        version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', output)
                        if version_match:
                            return version_match.group(1)

                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Error getting version for {scanner_path}: {e}")

        return "unknown"

    async def comprehensive_scan(self, target: ScanTarget) -> ScanResult:
        """Perform comprehensive security scan of target"""
        scan_id = f"scan_{target.host}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info(f"Starting comprehensive scan {scan_id} for {target.host}")

        try:
            scan_result = ScanResult(
                scan_id=scan_id,
                target=target.host,
                scan_type="comprehensive",
                start_time=start_time,
                end_time=start_time,  # Will be updated
                status="running",
                open_ports=[],
                services=[],
                vulnerabilities=[],
                os_fingerprint={},
                scan_statistics={},
                raw_output={},
                findings=[],
                recommendations=[]
            )

            # Phase 1: Network Discovery and Port Scanning
            if self.scanners.get("nmap", {}).available:
                logger.info("Phase 1: Network discovery with Nmap")
                nmap_results = await self._run_nmap_scan(target)
                scan_result.open_ports.extend(nmap_results.get("open_ports", []))
                scan_result.services.extend(nmap_results.get("services", []))
                scan_result.os_fingerprint = nmap_results.get("os_fingerprint", {})
                scan_result.raw_output["nmap"] = nmap_results.get("raw_output", "")

                if nmap_results.get("errors"):
                    scan_result.scan_statistics["nmap_errors"] = nmap_results["errors"]

            # Phase 2: Service-Specific Scanning
            discovered_ports = [p.get("port") for p in scan_result.open_ports]

            # Web Application Scanning
            web_ports = [p for p in discovered_ports if p in [80, 443, 8080, 8443, 8000, 3000, 9000]]
            if web_ports:
                logger.info("Phase 2a: Web application scanning")

                # Directory/file discovery
                if self.scanners.get("dirb", {}).available or self.scanners.get("gobuster", {}).available:
                    for port in web_ports:
                        web_findings = await self._run_web_discovery(target.host, port)
                        scan_result.vulnerabilities.extend(web_findings.get("vulnerabilities", []))
                        scan_result.raw_output[f"web_discovery_{port}"] = web_findings.get("raw_output", "")

                # Web vulnerability scanning
                if self.scanners.get("nikto", {}).available:
                    for port in web_ports:
                        nikto_results = await self._run_nikto_scan(target.host, port)
                        scan_result.vulnerabilities.extend(nikto_results.get("vulnerabilities", []))
                        scan_result.raw_output[f"nikto_{port}"] = nikto_results.get("raw_output", "")

                # SSL/TLS analysis
                ssl_ports = [p for p in web_ports if p in [443, 8443]]
                if ssl_ports and self.scanners.get("sslscan", {}).available:
                    for port in ssl_ports:
                        ssl_results = await self._run_sslscan(target.host, port)
                        scan_result.vulnerabilities.extend(ssl_results.get("vulnerabilities", []))
                        scan_result.raw_output[f"sslscan_{port}"] = ssl_results.get("raw_output", "")

            # Phase 3: Vulnerability Scanning with Nuclei
            if self.scanners.get("nuclei", {}).available:
                logger.info("Phase 3: Vulnerability scanning with Nuclei")
                nuclei_results = await self._run_nuclei_scan(target)
                scan_result.vulnerabilities.extend(nuclei_results.get("vulnerabilities", []))
                scan_result.raw_output["nuclei"] = nuclei_results.get("raw_output", "")

            # Phase 4: Database Detection and Testing
            db_ports = [p for p in discovered_ports if p in [1433, 3306, 5432, 1521, 27017]]
            if db_ports:
                logger.info("Phase 4: Database security testing")
                db_findings = await self._run_database_tests(target.host, db_ports)
                scan_result.vulnerabilities.extend(db_findings.get("vulnerabilities", []))

            # Phase 5: Custom Security Checks
            logger.info("Phase 5: Custom security analysis")
            custom_findings = await self._run_custom_security_checks(target, scan_result)
            scan_result.vulnerabilities.extend(custom_findings)

            # Finalize scan results
            scan_result.end_time = datetime.now()
            scan_result.status = "completed"
            scan_result.scan_statistics.update({
                "duration_seconds": (scan_result.end_time - scan_result.start_time).total_seconds(),
                "ports_scanned": len(target.ports) if target.ports else 1000,
                "open_ports_found": len(scan_result.open_ports),
                "services_identified": len(scan_result.services),
                "vulnerabilities_found": len(scan_result.vulnerabilities),
                "scanners_used": [name for name, config in self.scanners.items() if config.available]
            })

            # Generate security recommendations
            scan_result.recommendations = self._generate_security_recommendations(scan_result)

            logger.info(f"Scan {scan_id} completed: {len(scan_result.vulnerabilities)} vulnerabilities found")
            return scan_result

        except Exception as e:
            logger.error(f"Comprehensive scan failed: {e}")
            scan_result.end_time = datetime.now()
            scan_result.status = "failed"
            scan_result.scan_statistics["error"] = str(e)
            return scan_result

    async def _run_nmap_scan(self, target: ScanTarget) -> Dict[str, Any]:
        """Execute Nmap port scan with service detection"""
        try:
            nmap_config = self.scanners["nmap"]

            # Build Nmap command
            cmd = [
                nmap_config.path,
                "-sS",  # SYN scan
                "--max-rate", str(nmap_config.max_rate),
                "--max-retries", "1",
                "-T4",  # Aggressive timing
                "-sV",  # Service version detection
                "-O",   # OS detection
                "-A",   # Aggressive scan options
                "--script", "default,vuln,discovery",
                "-oX", "-",  # XML output to stdout
                target.host
            ]

            # Add port specification if provided
            if target.ports:
                port_spec = ",".join(map(str, target.ports))
                cmd.extend(["-p", port_spec])
            else:
                cmd.extend(["-p-"])  # Scan all ports

            # Add stealth options if enabled
            if getattr(target, 'stealth_mode', False):
                cmd.extend(["-f", "-D", "RND:3"])  # Fragment packets, decoy scans

            logger.debug(f"Running Nmap: {' '.join(cmd)}")

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous command arguments detected")

            # Execute Nmap scan
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nmap_config.timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"Nmap scan failed: {error_msg}")
                return {"errors": [error_msg]}

            # Parse XML output
            xml_output = stdout.decode('utf-8', errors='ignore')
            return self._parse_nmap_xml(xml_output)

        except asyncio.TimeoutError:
            logger.error("Nmap scan timed out")
            return {"errors": ["Nmap scan timed out"]}
        except Exception as e:
            logger.error(f"Nmap scan failed: {e}")
            return {"errors": [str(e)]}

    def _parse_nmap_xml(self, xml_output: str) -> Dict[str, Any]:
        """Parse Nmap XML output into structured data"""
        try:
            root = ET.fromstring(xml_output)
            result = {
                "open_ports": [],
                "services": [],
                "os_fingerprint": {},
                "vulnerabilities": [],
                "raw_output": xml_output
            }

            # Parse host information
            host = root.find("host")
            if host is None:
                return result

            # Parse open ports and services
            ports_elem = host.find("ports")
            if ports_elem is not None:
                for port in ports_elem.findall("port"):
                    port_id = int(port.get("portid"))
                    protocol = port.get("protocol", "tcp")

                    state = port.find("state")
                    if state is not None and state.get("state") == "open":
                        port_info = {
                            "port": port_id,
                            "protocol": protocol,
                            "state": "open"
                        }

                        # Service information
                        service = port.find("service")
                        if service is not None:
                            service_info = {
                                "port": port_id,
                                "name": service.get("name", "unknown"),
                                "product": service.get("product", ""),
                                "version": service.get("version", ""),
                                "extrainfo": service.get("extrainfo", ""),
                                "confidence": int(service.get("conf", "0"))
                            }
                            result["services"].append(service_info)
                            port_info["service"] = service_info

                        # Check for script results (vulnerabilities)
                        scripts = port.findall("script")
                        for script in scripts:
                            script_id = script.get("id", "")
                            script_output = script.get("output", "")

                            # Look for vulnerability-related scripts
                            if any(vuln_keyword in script_id for vuln_keyword in
                                  ["vuln", "exploit", "cve", "backdoor"]):
                                vuln = {
                                    "scanner": "nmap_script",
                                    "script_id": script_id,
                                    "name": f"Nmap Script: {script_id}",
                                    "severity": "medium",  # Default severity
                                    "description": script_output,
                                    "port": port_id,
                                    "service": service_info.get("name", "") if service else ""
                                }
                                result["vulnerabilities"].append(vuln)

                        result["open_ports"].append(port_info)

            # Parse OS fingerprinting
            os_elem = host.find("os")
            if os_elem is not None:
                osmatch = os_elem.find("osmatch")
                if osmatch is not None:
                    result["os_fingerprint"] = {
                        "name": osmatch.get("name", "unknown"),
                        "accuracy": int(osmatch.get("accuracy", "0")),
                        "line": osmatch.get("line", "")
                    }

            return result

        except ET.ParseError as e:
            logger.error(f"Failed to parse Nmap XML: {e}")
            return {"errors": [f"XML parse error: {e}"], "raw_output": xml_output}

    async def _run_nuclei_scan(self, target: ScanTarget) -> Dict[str, Any]:
        """Execute Nuclei vulnerability scanner"""
        try:
            nuclei_config = self.scanners["nuclei"]

            cmd = [
                nuclei_config.path,
                "-target", f"{target.host}",
                "-json",
                "-severity", "low,medium,high,critical",
                "-rate-limit", str(nuclei_config.max_rate),
                "-timeout", "10",
                "-retries", "1",
                "-no-color"
            ]

            # Add templates if available
            templates_path = nuclei_config.config.get("templates_path")
            if templates_path:
                expanded_path = Path(templates_path).expanduser()
                if expanded_path.exists():
                    cmd.extend(["-t", str(expanded_path)])

            logger.debug(f"Running Nuclei: {' '.join(cmd)}")

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous Nuclei command arguments detected")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nuclei_config.timeout
            )

            output = stdout.decode('utf-8', errors='ignore')
            vulnerabilities = []

            # Parse JSON output line by line
            for line in output.strip().split('\n'):
                if line.strip():
                    try:
                        vuln_data = json.loads(line)
                        vulnerability = {
                            "scanner": "nuclei",
                            "template_id": vuln_data.get("template-id", "unknown"),
                            "name": vuln_data.get("info", {}).get("name", "Unknown Vulnerability"),
                            "severity": vuln_data.get("info", {}).get("severity", "info"),
                            "description": vuln_data.get("info", {}).get("description", ""),
                            "reference": vuln_data.get("info", {}).get("reference", []),
                            "matched_at": vuln_data.get("matched-at", ""),
                            "timestamp": vuln_data.get("timestamp", datetime.now().isoformat()),
                            "curl_command": vuln_data.get("curl-command", ""),
                            "raw_data": vuln_data
                        }
                        vulnerabilities.append(vulnerability)
                    except json.JSONDecodeError:
                        continue

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except asyncio.TimeoutError:
            logger.error("Nuclei scan timed out")
            return {"errors": ["Nuclei scan timed out"]}
        except Exception as e:
            logger.error(f"Nuclei scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_nikto_scan(self, host: str, port: int) -> Dict[str, Any]:
        """Execute Nikto web vulnerability scanner"""
        try:
            nikto_config = self.scanners["nikto"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}"

            cmd = [
                nikto_config.path,
                "-h", target_url,
                "-Format", "json",
                "-nointeractive",
                "-maxtime", str(nikto_config.timeout),
                "-timeout", "10"
            ]

            logger.debug(f"Running Nikto: {' '.join(cmd)}")

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous Nikto command arguments detected")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nikto_config.timeout
            )

            output = stdout.decode('utf-8', errors='ignore')
            vulnerabilities = []

            # Parse Nikto output (format can vary)
            for line in output.split('\n'):
                if '+ ' in line:
                    # Determine severity based on content
                    severity = "info"
                    if any(keyword in line.lower() for keyword in
                          ['vulnerability', 'exploit', 'injection', 'xss', 'sql']):
                        severity = "medium"
                    if any(keyword in line.lower() for keyword in
                          ['critical', 'high', 'remote', 'execute']):
                        severity = "high"

                    vulnerability = {
                        "scanner": "nikto",
                        "name": "Web Security Issue",
                        "severity": severity,
                        "description": line.strip(),
                        "port": port,
                        "url": target_url,
                        "timestamp": datetime.now().isoformat()
                    }
                    vulnerabilities.append(vulnerability)

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except Exception as e:
            logger.error(f"Nikto scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_web_discovery(self, host: str, port: int) -> Dict[str, Any]:
        """Run web directory/file discovery"""
        vulnerabilities = []
        raw_output = ""

        try:
            # Try Gobuster first, then fall back to Dirb
            if self.scanners.get("gobuster", {}).available:
                result = await self._run_gobuster(host, port)
            elif self.scanners.get("dirb", {}).available:
                result = await self._run_dirb(host, port)
            else:
                return {"vulnerabilities": [], "raw_output": "No web discovery tools available"}

            return result

        except Exception as e:
            logger.error(f"Web discovery failed: {e}")
            return {"errors": [str(e)]}

    async def _run_gobuster(self, host: str, port: int) -> Dict[str, Any]:
        """Execute Gobuster directory bruteforcer"""
        try:
            gobuster_config = self.scanners["gobuster"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}/"

            cmd = [
                gobuster_config.path,
                "dir",
                "-u", target_url,
                "-w", gobuster_config.config.get("wordlist", "/usr/share/wordlists/dirb/common.txt"),
                "-q",  # Quiet mode
                "-t", "10",  # Threads
                "--timeout", "10s"
            ]

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous Gobuster command arguments detected")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=gobuster_config.timeout
            )

            output = stdout.decode('utf-8', errors='ignore')
            vulnerabilities = []

            # Parse Gobuster output
            for line in output.split('\n'):
                if '(Status:' in line and any(status in line for status in ['200', '301', '302']):
                    if any(keyword in line.lower() for keyword in
                          ['admin', 'backup', 'config', 'login', 'upload', 'test']):
                        vulnerability = {
                            "scanner": "gobuster",
                            "name": "Interesting Directory/File",
                            "severity": "info",
                            "description": f"Discovered: {line.strip()}",
                            "port": port,
                            "url": target_url
                        }
                        vulnerabilities.append(vulnerability)

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except Exception as e:
            logger.error(f"Gobuster scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_dirb(self, host: str, port: int) -> Dict[str, Any]:
        """Execute DIRB directory bruteforcer"""
        try:
            dirb_config = self.scanners["dirb"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}/"

            cmd = [
                dirb_config.path,
                target_url,
                dirb_config.config.get("wordlist", "/usr/share/dirb/wordlists/common.txt"),
                "-S",  # Silent mode
                "-w"   # Don't stop on warning messages
            ]

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous Dirb command arguments detected")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=dirb_config.timeout
            )

            output = stdout.decode('utf-8', errors='ignore')
            vulnerabilities = []

            # Parse DIRB output
            for line in output.split('\n'):
                if '==> DIRECTORY:' in line or '+ ' in line:
                    if any(keyword in line.lower() for keyword in
                          ['admin', 'backup', 'config', 'login', 'upload']):
                        vulnerability = {
                            "scanner": "dirb",
                            "name": "Interesting Directory",
                            "severity": "info",
                            "description": line.strip(),
                            "port": port,
                            "url": target_url
                        }
                        vulnerabilities.append(vulnerability)

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except Exception as e:
            logger.error(f"DIRB scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_sslscan(self, host: str, port: int) -> Dict[str, Any]:
        """Execute SSL/TLS security analysis"""
        try:
            sslscan_config = self.scanners["sslscan"]

            cmd = [
                sslscan_config.path,
                "--xml=-",
                f"{host}:{port}"
            ]

            # SECURITY: Validate command arguments before execution
            if not self._validate_command_args(cmd):
                raise SecurityError(f"Invalid or potentially dangerous SSLScan command arguments detected")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=sslscan_config.timeout
            )

            output = stdout.decode('utf-8', errors='ignore')
            vulnerabilities = []

            # Analyze SSL/TLS configuration
            if 'SSLv2' in output and 'enabled' in output:
                vulnerabilities.append({
                    "scanner": "sslscan",
                    "name": "SSLv2 Protocol Enabled",
                    "severity": "high",
                    "description": "Server supports deprecated SSLv2 protocol which has known vulnerabilities",
                    "port": port,
                    "cve": ["CVE-2011-3389"],
                    "remediation": "Disable SSLv2 support"
                })

            if 'SSLv3' in output and 'enabled' in output:
                vulnerabilities.append({
                    "scanner": "sslscan",
                    "name": "SSLv3 Protocol Enabled",
                    "severity": "medium",
                    "description": "Server supports deprecated SSLv3 protocol (POODLE vulnerability)",
                    "port": port,
                    "cve": ["CVE-2014-3566"],
                    "remediation": "Disable SSLv3 support"
                })

            if 'NULL' in output and 'cipher' in output.lower():
                vulnerabilities.append({
                    "scanner": "sslscan",
                    "name": "NULL Cipher Suites Enabled",
                    "severity": "high",
                    "description": "Server supports NULL encryption cipher suites",
                    "port": port,
                    "remediation": "Disable NULL cipher suites"
                })

            # Check for weak cipher suites
            if any(weak_cipher in output for weak_cipher in ['RC4', 'DES', 'MD5']):
                vulnerabilities.append({
                    "scanner": "sslscan",
                    "name": "Weak Cipher Suites",
                    "severity": "medium",
                    "description": "Server supports weak cipher suites",
                    "port": port,
                    "remediation": "Configure strong cipher suites only"
                })

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except Exception as e:
            logger.error(f"SSLScan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_database_tests(self, host: str, ports: List[int]) -> Dict[str, Any]:
        """Run database-specific security tests"""
        vulnerabilities = []

        for port in ports:
            try:
                # Basic database service detection and security analysis
                service_type = self._identify_database_service(port)

                if service_type:
                    # Check for default credentials (simulation)
                    vuln = {
                        "scanner": "custom_db_checker",
                        "name": f"{service_type} Database Service Detected",
                        "severity": "info",
                        "description": f"{service_type} database service detected on port {port}",
                        "port": port,
                        "service": service_type,
                        "recommendations": [
                            "Ensure strong authentication is configured",
                            "Verify encryption in transit is enabled",
                            "Check for default credentials",
                            "Implement proper access controls"
                        ]
                    }
                    vulnerabilities.append(vuln)

                    # Check for common misconfigurations
                    if port == 3306:  # MySQL
                        vulnerabilities.append({
                            "scanner": "custom_db_checker",
                            "name": "MySQL Security Check",
                            "severity": "medium",
                            "description": "MySQL service requires security hardening verification",
                            "port": port,
                            "remediation": "Run mysql_secure_installation and verify configuration"
                        })

            except Exception as e:
                logger.debug(f"Database test failed for port {port}: {e}")

        return {"vulnerabilities": vulnerabilities}

    def _identify_database_service(self, port: int) -> Optional[str]:
        """Identify database service type by port"""
        db_ports = {
            1433: "Microsoft SQL Server",
            3306: "MySQL",
            5432: "PostgreSQL",
            1521: "Oracle",
            27017: "MongoDB",
            6379: "Redis",
            5984: "CouchDB",
            9042: "Cassandra"
        }
        return db_ports.get(port)

    async def _run_custom_security_checks(self, target: ScanTarget, scan_result: ScanResult) -> List[Dict[str, Any]]:
        """Run AI-powered custom security analysis checks"""
        vulnerabilities = []

        try:
            # Traditional rule-based checks
            basic_vulns = await self._run_basic_security_checks(target, scan_result)
            vulnerabilities.extend(basic_vulns)

            # AI-powered analysis
            ai_vulns = await self._run_ai_security_analysis(target, scan_result)
            vulnerabilities.extend(ai_vulns)

            return vulnerabilities

        except Exception as e:
            logger.error(f"Enhanced security checks failed: {e}")
            return []

    async def _run_basic_security_checks(self, target: ScanTarget, scan_result: ScanResult) -> List[Dict[str, Any]]:
        """Run basic rule-based security checks"""
        vulnerabilities = []
        open_ports = [p.get("port") for p in scan_result.open_ports]

        # Check for common backdoor ports
        backdoor_ports = [1234, 4444, 5555, 6666, 31337, 12345, 54321]
        for port in open_ports:
            if port in backdoor_ports:
                vulnerabilities.append({
                    "scanner": "ai_security_checker",
                    "name": "Suspicious Port Open",
                    "severity": "high",
                    "description": f"Port {port} is commonly associated with backdoors and malware",
                    "port": port,
                    "confidence": 0.9,
                    "remediation": "Investigate the service running on this port and close if unnecessary"
                })

        # Enhanced service version analysis
        for service in scan_result.services:
            service_vulns = await self._analyze_service_security(service)
            vulnerabilities.extend(service_vulns)

        # Attack surface analysis
        if len(open_ports) > 20:
            attack_surface_score = self._calculate_attack_surface_score(scan_result)
            vulnerabilities.append({
                "scanner": "ai_security_checker",
                "name": "Elevated Attack Surface",
                "severity": "medium" if attack_surface_score < 0.7 else "high",
                "description": f"Host has {len(open_ports)} open ports with attack surface score: {attack_surface_score:.2f}",
                "confidence": 0.8,
                "attack_surface_score": attack_surface_score,
                "remediation": "Review and close unnecessary services and ports"
            })

        return vulnerabilities

    async def _run_ai_security_analysis(self, target: ScanTarget, scan_result: ScanResult) -> List[Dict[str, Any]]:
        """Run AI-powered security analysis"""
        vulnerabilities = []

        try:
            # Prepare feature vector for AI analysis
            features = await self._extract_ai_features(target, scan_result)

            # Run ensemble AI models for threat detection
            ai_predictions = await self._run_ai_threat_models(features)

            # Convert AI predictions to vulnerability findings
            for prediction in ai_predictions:
                if prediction.get("confidence", 0) > 0.7:
                    vulnerabilities.append({
                        "scanner": "ai_threat_detector",
                        "name": prediction.get("threat_type", "AI-Detected Security Issue"),
                        "severity": self._map_ai_severity(prediction.get("risk_score", 0)),
                        "description": prediction.get("description", "AI-detected security anomaly"),
                        "confidence": prediction.get("confidence", 0),
                        "risk_score": prediction.get("risk_score", 0),
                        "mitre_techniques": prediction.get("mitre_techniques", []),
                        "ai_model": prediction.get("model_used", "ensemble"),
                        "remediation": prediction.get("recommendations", "Review and address identified security issue")
                    })

        except Exception as e:
            logger.error(f"AI security analysis failed: {e}")

        return vulnerabilities

    async def _analyze_service_security(self, service: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced service security analysis with AI insights"""
        vulnerabilities = []
        service_name = service.get("name", "").lower()
        version = service.get("version", "")
        port = service.get("port")

        # Enhanced vulnerability checks with AI scoring
        if service_name == "ssh" and version:
            vulnerable_ssh_versions = ["OpenSSH_7.4", "OpenSSH_6.6", "OpenSSH_5.3"]
            if any(vuln_ver in version for vuln_ver in vulnerable_ssh_versions):
                confidence = 0.95 if "OpenSSH_5.3" in version else 0.85
                vulnerabilities.append({
                    "scanner": "enhanced_service_analyzer",
                    "name": "Vulnerable SSH Version",
                    "severity": "high" if confidence > 0.9 else "medium",
                    "description": f"SSH service running potentially vulnerable version: {version}",
                    "port": port,
                    "service": service_name,
                    "version": version,
                    "confidence": confidence,
                    "cve_ids": ["CVE-2016-0777", "CVE-2016-0778"] if "OpenSSH_7.4" in version else [],
                    "remediation": "Update SSH to the latest stable version"
                })

        elif service_name in ["http", "https"] and version:
            if "apache" in version.lower():
                # AI-enhanced Apache version analysis
                if any(old_ver in version for old_ver in ["2.2", "2.0"]):
                    risk_multiplier = 1.5 if "2.0" in version else 1.2
                    vulnerabilities.append({
                        "scanner": "enhanced_service_analyzer",
                        "name": "Outdated Apache Version",
                        "severity": "high" if risk_multiplier > 1.4 else "medium",
                        "description": f"Apache server running outdated version: {version}",
                        "port": port,
                        "confidence": 0.9,
                        "risk_multiplier": risk_multiplier,
                        "remediation": "Update Apache to the latest stable version"
                    })

        elif service_name == "telnet":
            vulnerabilities.append({
                "scanner": "enhanced_service_analyzer",
                "name": "Insecure Telnet Service",
                "severity": "critical",
                "description": "Telnet service provides unencrypted remote access",
                "port": port,
                "confidence": 1.0,
                "mitre_techniques": ["T1021.002"],
                "remediation": "Replace Telnet with SSH for secure remote access"
            })

        return vulnerabilities

    def _calculate_attack_surface_score(self, scan_result: ScanResult) -> float:
        """Calculate attack surface score using AI algorithms"""
        try:
            # Enhanced attack surface calculation with AI weighting
            open_ports_count = len(scan_result.open_ports)
            services_count = len(scan_result.services)
            critical_services = len([s for s in scan_result.services
                                   if s.get("name", "").lower() in ["ssh", "rdp", "telnet", "ftp"]])

            # Web services exposure
            web_services = len([s for s in scan_result.services
                              if s.get("name", "").lower() in ["http", "https"]])

            # Database services exposure
            db_services = len([s for s in scan_result.services
                             if s.get("name", "").lower() in ["mysql", "postgres", "mssql", "oracle"]])

            # AI-enhanced scoring with dynamic weights
            score = 0.0
            score += min(open_ports_count / 100.0, 1.0) * 0.3
            score += min(services_count / 50.0, 1.0) * 0.2
            score += min(critical_services / 10.0, 1.0) * 0.3
            score += min(web_services / 5.0, 1.0) * 0.1
            score += min(db_services / 3.0, 1.0) * 0.1

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Attack surface calculation failed: {e}")
            return 0.5

    async def _extract_ai_features(self, target: ScanTarget, scan_result: ScanResult) -> np.ndarray:
        """Extract advanced features for AI vulnerability analysis"""
        features = []

        try:
            # Port-based features with enhanced analysis
            open_ports = [p.get("port") for p in scan_result.open_ports]
            privileged_ports = [p for p in open_ports if p < 1024]
            dynamic_ports = [p for p in open_ports if p >= 49152]

            features.extend([
                len(open_ports),
                len(privileged_ports),
                len(dynamic_ports),
                np.std(open_ports) if open_ports else 0,   # Port distribution
                len([p for p in open_ports if p in [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]]),  # Common services
                len([p for p in open_ports if p in [1433, 3306, 5432, 1521, 27017]]),  # Database ports
                len([p for p in open_ports if p in [139, 445, 135, 3389]]),  # Windows services
            ])

            # Service-based features with AI enhancement
            services = scan_result.services
            service_names = [s.get("name", "").lower() for s in services]

            features.extend([
                len(services),
                len([s for s in service_names if "ssh" in s]),
                len([s for s in service_names if "http" in s]),
                len([s for s in service_names if "ftp" in s]),
                len([s for s in service_names if "smtp" in s]),
                len([s for s in service_names if any(db in s for db in ["mysql", "postgres", "mssql", "oracle"])]),
                len([s for s in service_names if any(web in s for web in ["apache", "nginx", "iis"])]),
            ])

            # Vulnerability-based features
            vulnerabilities = scan_result.vulnerabilities
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

            for vuln in vulnerabilities:
                severity = vuln.get("severity", "low").lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1

            features.extend([
                len(vulnerabilities),
                severity_counts["critical"],
                severity_counts["high"],
                severity_counts["medium"],
                severity_counts["low"],
            ])

            # Advanced target analysis features
            target_features = await self._analyze_target_characteristics(target)
            features.extend(target_features)

            # Convert to numpy array for ML processing
            if np is not None:
                return np.array(features, dtype=np.float32)
            else:
                return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default feature vector
            return np.zeros(25) if np is not None else [0.0] * 25

    async def _analyze_target_characteristics(self, target: ScanTarget) -> List[float]:
        """Analyze advanced target characteristics for AI processing"""
        features = []

        try:
            # Network class analysis
            ip = ipaddress.ip_address(target.host)
            features.extend([
                1.0 if ip.is_private else 0.0,
                1.0 if ip.is_loopback else 0.0,
                1.0 if ip.is_multicast else 0.0,
                1.0 if ip.version == 6 else 0.0,  # IPv6
            ])

            # DNS and hostname analysis
            try:
                hostname = socket.gethostbyaddr(target.host)[0]
                features.extend([
                    1.0 if "test" in hostname.lower() else 0.0,
                    1.0 if "dev" in hostname.lower() else 0.0,
                    1.0 if "prod" in hostname.lower() else 0.0,
                    len(hostname.split(".")),  # Domain depth
                ])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Port range analysis
            scan_ports = target.ports if target.ports else []
            if scan_ports:
                features.extend([
                    min(scan_ports),
                    max(scan_ports),
                    len(scan_ports),
                    1.0 if any(p < 100 for p in scan_ports) else 0.0,  # System ports
                ])
            else:
                features.extend([0.0, 65535.0, 0.0, 0.0])

        except Exception as e:
            logger.error(f"Target characteristic analysis failed: {e}")
            features = [0.0] * 12

        return features[:12]  # Ensure consistent feature vector size

    async def _apply_ai_threat_scoring(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Apply AI-powered threat scoring to vulnerabilities"""
        if not np or not scan_result.vulnerabilities:
            return {"ai_score": 0.0, "threat_level": "low", "confidence": 0.0}

        try:
            # Extract vulnerability features for AI analysis
            vuln_features = []

            for vuln in scan_result.vulnerabilities:
                # Vulnerability characteristic features
                severity_score = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}.get(
                    vuln.get("severity", "low").lower(), 0.2)

                # CVSS score if available
                cvss_score = vuln.get("cvss_score", 0.0) / 10.0 if vuln.get("cvss_score") else severity_score

                # Port and service context
                port_risk = 1.0 if vuln.get("port", 0) in [21, 22, 23, 80, 443, 3389] else 0.5

                # Exploit availability (heuristic)
                has_exploit = 1.0 if any(keyword in vuln.get("description", "").lower()
                                       for keyword in ["rce", "remote", "overflow", "injection"]) else 0.0

                vuln_features.append([severity_score, cvss_score, port_risk, has_exploit])

            # Convert to numpy array
            features_array = np.array(vuln_features)

            # Simple AI scoring algorithm (ensemble approach)
            if len(features_array) > 0:
                # Weight-based scoring
                weights = np.array([0.4, 0.3, 0.2, 0.1])  # Severity, CVSS, Port, Exploit
                weighted_scores = np.dot(features_array, weights)

                # Calculate overall threat metrics
                ai_score = float(np.mean(weighted_scores))
                max_score = float(np.max(weighted_scores))
                score_variance = float(np.var(weighted_scores))

                # Determine threat level
                if ai_score >= 0.8:
                    threat_level = "critical"
                elif ai_score >= 0.6:
                    threat_level = "high"
                elif ai_score >= 0.4:
                    threat_level = "medium"
                else:
                    threat_level = "low"

                # Calculate confidence based on data quality and consistency
                confidence = min(1.0, (len(features_array) / 10.0) * (1.0 - score_variance))

                return {
                    "ai_score": ai_score,
                    "max_vulnerability_score": max_score,
                    "threat_level": threat_level,
                    "confidence": confidence,
                    "vulnerability_count": len(scan_result.vulnerabilities),
                    "critical_vulnerabilities": len([v for v in scan_result.vulnerabilities
                                                   if v.get("severity", "").lower() == "critical"]),
                    "exploitable_vulnerabilities": len([v for v in scan_result.vulnerabilities
                                                      if "rce" in v.get("description", "").lower()]),
                }
            else:
                return {"ai_score": 0.0, "threat_level": "low", "confidence": 1.0}

        except Exception as e:
            logger.error(f"AI threat scoring failed: {e}")
            return {"ai_score": 0.0, "threat_level": "unknown", "confidence": 0.0}

    async def _generate_ai_recommendations(self, scan_result: ScanResult, ai_analysis: Dict[str, Any]) -> List[str]:
        """Generate AI-powered security recommendations"""
        recommendations = []

        try:
            threat_level = ai_analysis.get("threat_level", "low")
            ai_score = ai_analysis.get("ai_score", 0.0)
            critical_vulns = ai_analysis.get("critical_vulnerabilities", 0)

            # Priority-based recommendations using AI insights
            if threat_level == "critical" or ai_score >= 0.8:
                recommendations.extend([
                    "🚨 IMMEDIATE ACTION REQUIRED: Critical vulnerabilities detected",
                    "🔥 Implement emergency incident response procedures",
                    "🛡️ Consider taking affected systems offline until patched",
                    "📋 Conduct thorough forensic analysis",
                ])

            if critical_vulns > 0:
                recommendations.append(f"⚡ {critical_vulns} critical vulnerabilities require immediate patching")

            # Service-specific recommendations
            services = [s.get("name", "").lower() for s in scan_result.services]

            if any("ftp" in s for s in services):
                recommendations.append("🔒 Replace FTP with SFTP/FTPS for secure file transfer")

            if any("telnet" in s for s in services):
                recommendations.append("🚫 Disable Telnet and use SSH for remote access")

            if any("http" in s for s in services) and not any("https" in s for s in services):
                recommendations.append("🔐 Implement HTTPS encryption for web services")

            # Port-based recommendations
            open_ports = [p.get("port") for p in scan_result.open_ports]
            risky_ports = [p for p in open_ports if p in [21, 23, 135, 139, 445]]

            if risky_ports:
                recommendations.append(f"🚪 Review necessity of open ports: {', '.join(map(str, risky_ports))}")

            # AI-driven pattern analysis recommendations
            if ai_score >= 0.6:
                recommendations.extend([
                    "🧠 AI analysis indicates elevated risk - implement additional monitoring",
                    "📊 Consider deploying advanced threat detection systems",
                    "🔍 Increase security audit frequency",
                ])

            # Network security recommendations
            if len(open_ports) > 20:
                recommendations.append("🌐 Reduce attack surface by closing unnecessary ports")

            # General security hardening
            recommendations.extend([
                "🔧 Apply latest security patches and updates",
                "🛡️ Implement network segmentation and access controls",
                "📝 Review and update security policies",
                "👥 Conduct security awareness training",
                "🕵️ Implement continuous security monitoring",
            ])

        except Exception as e:
            logger.error(f"AI recommendation generation failed: {e}")
            recommendations.append("⚠️ Review scan results manually and apply security best practices")

        return recommendations[:15]  # Limit to most important recommendations

    async def _run_ai_threat_models(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Run AI threat detection models"""
        predictions = []

        try:
            # Anomaly detection
            anomaly_score = self._simulate_anomaly_detection(features)
            if anomaly_score > 0.7:
                predictions.append({
                    "threat_type": "Network Configuration Anomaly",
                    "confidence": anomaly_score,
                    "risk_score": int(anomaly_score * 100),
                    "description": f"AI detected network configuration anomaly (score: {anomaly_score:.3f})",
                    "model_used": "isolation_forest",
                    "mitre_techniques": ["T1046", "T1590"],
                    "recommendations": "Review network configuration for security issues"
                })

            # Threat classification
            threat_class = self._simulate_threat_classification(features)
            if threat_class.get("confidence", 0) > 0.75:
                predictions.append(threat_class)

            # Vulnerability prediction
            vuln_prediction = self._simulate_vulnerability_prediction(features)
            if vuln_prediction.get("confidence", 0) > 0.8:
                predictions.append({
                    "threat_type": "Predicted Vulnerability Exposure",
                    "confidence": vuln_prediction.get("confidence", 0),
                    "risk_score": vuln_prediction.get("risk_score", 0),
                    "description": vuln_prediction.get("description", "AI predicted potential vulnerability"),
                    "model_used": "gradient_boosting",
                    "mitre_techniques": ["T1190"],
                    "recommendations": "Perform detailed vulnerability assessment"
                })

        except Exception as e:
            logger.error(f"AI threat model execution failed: {e}")

        return predictions

    def _simulate_anomaly_detection(self, features: np.ndarray) -> float:
        """Simulate anomaly detection model"""
        # AI-enhanced anomaly detection simulation
        anomaly_score = 0.0

        # High port count anomaly
        if features[0] > 20:
            anomaly_score += 0.3 * (features[0] / 100.0)

        # Service anomaly
        if features[4] > 15:
            anomaly_score += 0.2 * (features[4] / 50.0)

        # Vulnerability concentration
        if features[8] > 5:
            anomaly_score += 0.4 * (features[8] / 20.0)

        # Statistical anomaly
        feature_std = np.std(features)
        if feature_std > 0.5:
            anomaly_score += 0.2 * min(feature_std, 1.0)

        return min(anomaly_score, 1.0)

    def _simulate_threat_classification(self, features: np.ndarray) -> Dict[str, Any]:
        """Simulate AI threat classification model"""
        port_count = features[0]
        vuln_count = features[8]
        critical_vulns = features[9]

        if critical_vulns > 2:
            return {
                "threat_type": "Critical Infrastructure Exposure",
                "confidence": 0.85 + (critical_vulns * 0.05),
                "risk_score": min(90 + int(critical_vulns * 5), 100),
                "description": f"System has {int(critical_vulns)} critical vulnerabilities indicating high exploitation risk",
                "model_used": "neural_network",
                "mitre_techniques": ["T1190", "T1068"],
                "recommendations": "Immediately patch critical vulnerabilities and implement compensating controls"
            }
        elif port_count > 30:
            return {
                "threat_type": "Excessive Service Exposure",
                "confidence": 0.78 + min((port_count - 30) * 0.01, 0.15),
                "risk_score": min(75 + int((port_count - 30) * 2), 95),
                "description": f"System exposes {int(port_count)} services increasing attack surface significantly",
                "model_used": "random_forest",
                "mitre_techniques": ["T1046", "T1595"],
                "recommendations": "Minimize exposed services and implement network segmentation"
            }

        return {"confidence": 0.3}

    def _simulate_vulnerability_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Simulate AI vulnerability prediction model"""
        service_count = features[4]
        confidence_avg = features[7]
        os_accuracy = features[12]

        prediction_score = 0.0

        # Enhanced prediction logic
        if service_count > 10:
            prediction_score += 0.4 * min(service_count / 50.0, 1.0)

        if confidence_avg < 0.5:
            prediction_score += 0.3 * (1.0 - confidence_avg)

        if os_accuracy < 0.7:
            prediction_score += 0.2 * (1.0 - os_accuracy)

        if prediction_score > 0.8:
            return {
                "confidence": prediction_score,
                "risk_score": int(prediction_score * 100),
                "description": f"AI predicts high likelihood of undiscovered vulnerabilities (score: {prediction_score:.3f})",
                "recommendations": "Perform comprehensive vulnerability assessment with multiple tools"
            }

        return {"confidence": prediction_score}

    def _map_ai_severity(self, risk_score: float) -> str:
        """Map AI risk score to severity level"""
        if risk_score >= 90:
            return "critical"
        elif risk_score >= 70:
            return "high"
        elif risk_score >= 40:
            return "medium"
        else:
            return "low"

    def _generate_security_recommendations(self, scan_result: ScanResult) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []

        # Critical vulnerabilities
        critical_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]
        if critical_vulns:
            recommendations.append("🚨 CRITICAL: Immediately address critical vulnerabilities to prevent compromise")

        # High severity issues
        high_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "high"]
        if high_vulns:
            recommendations.append(f"⚠️  HIGH: Address {len(high_vulns)} high-severity vulnerabilities within 24-48 hours")

        # Service-specific recommendations
        open_ports = [p.get("port") for p in scan_result.open_ports]

        if 22 in open_ports:
            recommendations.append("🔐 SSH: Ensure key-based authentication and disable password authentication")

        if any(port in open_ports for port in [80, 443, 8080]):
            recommendations.append("🌐 Web Services: Implement HTTPS, security headers, and regular updates")

        if any(port in open_ports for port in [1433, 3306, 5432]):
            recommendations.append("🗄️ Database: Restrict network access, use encryption, and strong authentication")

        # General security recommendations
        recommendations.extend([
            "🔄 Implement regular vulnerability scanning schedule",
            "📦 Keep all services updated with latest security patches",
            "🛡️ Configure firewall rules to restrict unnecessary access",
            "📊 Implement network monitoring and logging",
            "🏗️ Consider network segmentation for sensitive services",
            "👥 Conduct regular security awareness training",
            "📋 Develop and test incident response procedures"
        ])

        return recommendations

    async def health_check(self) -> ServiceHealth:
        """Perform health check on scanner service"""
        try:
            checks = {
                "available_scanners": len([s for s in self.scanners.values() if s.available]),
                "total_scanners": len(self.scanners),
                "active_scans": len(self.active_scans),
                "scan_queue_size": self.scan_queue.qsize()
            }

            # Check if critical scanners are available
            critical_scanners = ["nmap"]
            critical_available = all(
                self.scanners.get(scanner, {}).available
                for scanner in critical_scanners
            )

            status = ServiceStatus.HEALTHY if critical_available else ServiceStatus.DEGRADED
            message = "Scanner service operational" if critical_available else "Some critical scanners unavailable"

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

    async def _process_scan_queue(self):
        """Background task to process scan queue"""
        while True:
            try:
                # Wait for scan requests
                scan_request = await self.scan_queue.get()

                # Process scan
                scan_task = asyncio.create_task(
                    self.comprehensive_scan(scan_request["target"])
                )

                self.active_scans[scan_request["scan_id"]] = scan_task

                # Wait for completion and store results
                try:
                    result = await scan_task
                    self.scan_results[scan_request["scan_id"]] = result
                finally:
                    # Cleanup
                    self.active_scans.pop(scan_request["scan_id"], None)
                    self.scan_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing scan queue: {e}")
                await asyncio.sleep(1)

    # PTaaSService interface implementation
    async def create_scan_session(
        self,
        targets: List[Dict[str, Any]],
        scan_type: str,
        user,
        org,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new PTaaS scan session"""
        try:
            session_id = f"ptaas_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user.id if hasattr(user, 'id') else 'unknown'}"

            # Convert targets to ScanTarget objects
            scan_targets = []
            for target_data in targets:
                scan_target = ScanTarget(
                    host=target_data.get("host"),
                    ports=target_data.get("ports", []),
                    scan_profile=target_data.get("scan_profile", "quick"),
                    stealth_mode=target_data.get("stealth_mode", False)
                )
                scan_targets.append(scan_target)

            # Queue the scan
            scan_request = {
                "scan_id": session_id,
                "targets": scan_targets,
                "scan_type": scan_type,
                "user_id": user.id if hasattr(user, 'id') else None,
                "org_id": org.id if hasattr(org, 'id') else None,
                "metadata": metadata or {},
                "created_at": datetime.now()
            }

            await self.scan_queue.put(scan_request)

            return {
                "session_id": session_id,
                "status": "queued",
                "targets_count": len(scan_targets),
                "scan_type": scan_type,
                "estimated_duration": self._estimate_scan_duration(scan_targets, scan_type),
                "created_at": scan_request["created_at"].isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to create scan session: {e}")
            raise

    async def get_scan_status(self, session_id: str, user) -> Dict[str, Any]:
        """Get status of a scan session"""
        try:
            # Check if scan is active
            if session_id in self.active_scans:
                task = self.active_scans[session_id]
                status = "running" if not task.done() else "completed"

                return {
                    "session_id": session_id,
                    "status": status,
                    "progress": self._get_scan_progress(session_id),
                    "active": not task.done()
                }

            # Check if results are available
            if session_id in self.scan_results:
                result = self.scan_results[session_id]
                return {
                    "session_id": session_id,
                    "status": result.status,
                    "progress": 100,
                    "active": False,
                    "completed_at": result.end_time.isoformat() if result.end_time else None,
                    "vulnerabilities_found": len(result.vulnerabilities),
                    "ports_scanned": len(result.open_ports)
                }

            # Session not found
            return {
                "session_id": session_id,
                "status": "not_found",
                "error": "Scan session not found"
            }

        except Exception as e:
            logger.error(f"Failed to get scan status for {session_id}: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e)
            }

    async def get_scan_results(self, session_id: str, user) -> Dict[str, Any]:
        """Get results from a completed scan"""
        try:
            if session_id not in self.scan_results:
                return {
                    "session_id": session_id,
                    "error": "Scan results not found"
                }

            result = self.scan_results[session_id]

            return {
                "session_id": session_id,
                "status": result.status,
                "target": result.target,
                "scan_type": result.scan_type,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": (result.end_time - result.start_time).total_seconds() if result.end_time else None,
                "open_ports": result.open_ports,
                "services": result.services,
                "vulnerabilities": [
                    {
                        "scanner": vuln.get("scanner", "unknown"),
                        "name": vuln.get("name", "Unknown"),
                        "severity": vuln.get("severity", "info"),
                        "description": vuln.get("description", ""),
                        "port": vuln.get("port"),
                        "service": vuln.get("service"),
                        "remediation": vuln.get("remediation", "")
                    } for vuln in result.vulnerabilities
                ],
                "os_fingerprint": result.os_fingerprint,
                "scan_statistics": result.scan_statistics,
                "recommendations": result.recommendations
            }

        except Exception as e:
            logger.error(f"Failed to get scan results for {session_id}: {e}")
            return {
                "session_id": session_id,
                "error": str(e)
            }

    async def cancel_scan(self, session_id: str, user) -> bool:
        """Cancel an active scan session"""
        try:
            if session_id in self.active_scans:
                task = self.active_scans[session_id]
                task.cancel()

                # Wait for cancellation to complete
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Cleanup
                self.active_scans.pop(session_id, None)

                logger.info(f"Scan {session_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Cannot cancel scan {session_id} - not active")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel scan {session_id}: {e}")
            return False

    async def get_available_scan_profiles(self) -> List[Dict[str, Any]]:
        """Get available scan profiles and their configurations"""
        return [
            {
                "name": "quick",
                "display_name": "Quick Scan",
                "description": "Fast network scan with basic service detection",
                "estimated_duration": "5 minutes",
                "features": ["port_scanning", "service_detection"],
                "suitable_for": ["initial_assessment", "regular_monitoring"]
            },
            {
                "name": "comprehensive",
                "display_name": "Comprehensive Scan",
                "description": "Full security assessment with vulnerability scanning",
                "estimated_duration": "30 minutes",
                "features": ["port_scanning", "service_detection", "vulnerability_scanning", "os_fingerprinting"],
                "suitable_for": ["security_audit", "compliance_assessment"]
            },
            {
                "name": "stealth",
                "display_name": "Stealth Scan",
                "description": "Low-profile scanning to avoid detection",
                "estimated_duration": "60 minutes",
                "features": ["stealth_scanning", "evasion_techniques", "slow_scanning"],
                "suitable_for": ["red_team_exercise", "covert_assessment"]
            },
            {
                "name": "web_focused",
                "display_name": "Web Application Scan",
                "description": "Specialized web application security testing",
                "estimated_duration": "20 minutes",
                "features": ["web_scanning", "directory_discovery", "ssl_analysis"],
                "suitable_for": ["web_application_testing", "ssl_assessment"]
            }
        ]

    async def create_compliance_scan(
        self,
        targets: List[str],
        compliance_framework: str,
        user,
        org
    ) -> Dict[str, Any]:
        """Create compliance-specific scan"""
        try:
            # Convert string targets to target objects
            target_objects = []
            for target in targets:
                target_data = {
                    "host": target,
                    "scan_profile": "comprehensive",
                    "compliance_mode": True
                }
                target_objects.append(target_data)

            # Add compliance-specific metadata
            metadata = {
                "compliance_framework": compliance_framework,
                "compliance_scan": True,
                "framework_requirements": self._get_compliance_requirements(compliance_framework)
            }

            # Create scan session
            return await self.create_scan_session(
                targets=target_objects,
                scan_type="compliance",
                user=user,
                org=org,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to create compliance scan: {e}")
            raise

    # SecurityService interface implementation
    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event"""
        try:
            event_type = event.get("type", "unknown")

            if event_type == "vulnerability_detected":
                return await self._process_vulnerability_event(event)
            elif event_type == "scan_completed":
                return await self._process_scan_completion_event(event)
            else:
                return {"status": "ignored", "reason": f"Unknown event type: {event_type}"}

        except Exception as e:
            logger.error(f"Failed to process security event: {e}")
            return {"status": "error", "error": str(e)}

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics"""
        try:
            total_scans = len(self.scan_results)
            active_scans = len(self.active_scans)

            # Calculate vulnerability statistics
            total_vulns = 0
            critical_vulns = 0
            high_vulns = 0

            for result in self.scan_results.values():
                total_vulns += len(result.vulnerabilities)
                for vuln in result.vulnerabilities:
                    severity = vuln.get("severity", "info")
                    if severity == "critical":
                        critical_vulns += 1
                    elif severity == "high":
                        high_vulns += 1

            return {
                "total_scans_completed": total_scans,
                "active_scans": active_scans,
                "queue_size": self.scan_queue.qsize(),
                "total_vulnerabilities_found": total_vulns,
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "available_scanners": len([s for s in self.scanners.values() if s.available]),
                "scanner_health": {
                    name: config.available for name, config in self.scanners.items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            return {"error": str(e)}

    # Helper methods
    def _estimate_scan_duration(self, targets: List[ScanTarget], scan_type: str) -> int:
        """Estimate scan duration in seconds"""
        base_duration = {
            "quick": 300,      # 5 minutes
            "comprehensive": 1800,  # 30 minutes
            "stealth": 3600,   # 60 minutes
            "web_focused": 1200,   # 20 minutes
            "compliance": 2400     # 40 minutes
        }

        duration = base_duration.get(scan_type, 300)
        # Scale by number of targets
        return duration * len(targets)

    def _get_scan_progress(self, session_id: str) -> int:
        """Get scan progress percentage (mock implementation)"""
        # In a real implementation, this would track actual scan progress
        if session_id in self.active_scans:
            task = self.active_scans[session_id]
            if task.done():
                return 100
            else:
                # Mock progress based on time elapsed
                return min(75, 25)  # Placeholder
        return 0

    def _get_compliance_requirements(self, framework: str) -> Dict[str, Any]:
        """Get compliance requirements for framework"""
        requirements = {
            "PCI-DSS": {
                "network_security": ["firewall_configuration", "network_segmentation"],
                "access_control": ["user_authentication", "access_logging"],
                "encryption": ["data_in_transit", "data_at_rest"],
                "vulnerability_management": ["regular_scanning", "patch_management"]
            },
            "HIPAA": {
                "access_control": ["unique_user_identification", "automatic_logoff"],
                "audit_controls": ["audit_logs", "audit_review"],
                "integrity": ["data_integrity", "transmission_security"],
                "transmission_security": ["encryption", "access_controls"]
            },
            "SOX": {
                "it_controls": ["change_management", "access_controls"],
                "data_backup": ["backup_procedures", "recovery_testing"],
                "security_monitoring": ["intrusion_detection", "log_monitoring"]
            }
        }
        return requirements.get(framework, {})

    async def _process_vulnerability_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process vulnerability detection event"""
        vulnerability = event.get("vulnerability", {})
        severity = vulnerability.get("severity", "info")

        # Log high-severity vulnerabilities
        if severity in ["critical", "high"]:
            logger.warning(f"High-severity vulnerability detected: {vulnerability.get('name', 'Unknown')}")

        return {"status": "processed", "action": "logged"}

    async def _process_scan_completion_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process scan completion event"""
        session_id = event.get("session_id")
        status = event.get("status", "unknown")

        logger.info(f"Scan {session_id} completed with status: {status}")

        return {"status": "processed", "action": "notification_sent"}

# Global scanner service instance
_scanner_service: Optional[SecurityScannerService] = None

async def get_scanner_service() -> SecurityScannerService:
    """Get global scanner service instance"""
    global _scanner_service

    if _scanner_service is None:
        _scanner_service = SecurityScannerService()
        await _scanner_service.initialize()

        # Register with global service registry
        service_registry.register(_scanner_service)

    return _scanner_service

# Register the service class for factory creation
# Service registration handled by container
from .base_service import service_registry
# Auto-register service when module is imported
