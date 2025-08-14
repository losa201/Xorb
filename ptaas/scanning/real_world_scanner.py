"""
Real-World Security Scanner Integration
Integrates with production security scanning tools for comprehensive testing
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
import aiohttp
import shlex

logger = logging.getLogger(__name__)

@dataclass
class ScanTarget:
    """Security scan target definition"""
    host: str
    ports: List[int]
    scan_type: str = "comprehensive"
    timeout: int = 300
    stealth_mode: bool = False
    authorized: bool = True

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

@dataclass
class ComprehensiveScanResult:
    """Complete scan result with all findings"""
    scan_id: str
    target: str
    scan_type: str
    start_time: datetime
    end_time: datetime
    status: str
    open_ports: List[Dict[str, Any]]
    services: List[Dict[str, Any]]
    vulnerabilities: List[VulnerabilityFinding]
    os_fingerprint: Dict[str, Any]
    scan_statistics: Dict[str, Any]
    raw_output: Dict[str, str]
    findings: List[Dict[str, Any]]
    recommendations: List[str]

class RealWorldSecurityScanner:
    """Production-ready security scanner with real tool integration"""

    def __init__(self):
        self.scanners = {}
        self.scan_results = {}
        self.logger = logging.getLogger(__name__)

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
            }
        }

    async def initialize(self) -> bool:
        """Initialize the scanner and detect available tools"""
        try:
            self.logger.info("Initializing Real-World Security Scanner...")

            # Detect available scanners
            await self._detect_scanners()

            self.logger.info(f"Scanner initialized with {len(self.scanners)} available tools")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize scanner: {e}")
            return False

    async def _detect_scanners(self):
        """Detect available security scanners on the system"""
        for scanner_name, config in self.scanner_configs.items():
            try:
                scanner_path = await self._find_executable(config["executable"])
                if scanner_path:
                    version = await self._get_scanner_version(scanner_path)

                    self.scanners[scanner_name] = {
                        "path": scanner_path,
                        "version": version,
                        "available": True,
                        "config": config
                    }

                    self.logger.info(f"Detected scanner: {scanner_name} v{version} at {scanner_path}")
                else:
                    self.scanners[scanner_name] = {
                        "path": None,
                        "version": None,
                        "available": False,
                        "config": config
                    }

                    self.logger.warning(f"Scanner not found: {scanner_name}")

            except Exception as e:
                self.logger.error(f"Error detecting scanner {scanner_name}: {e}")
                self.scanners[scanner_name] = {
                    "path": None,
                    "version": None,
                    "available": False,
                    "config": config
                }

    async def _find_executable(self, executable: str) -> Optional[str]:
        """Find executable path using which command"""
        try:
            process = await asyncio.create_subprocess_exec(
                "which", executable,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()

            if process.returncode == 0:
                return stdout.decode().strip()

        except Exception as e:
            self.logger.debug(f"Error finding executable {executable}: {e}")

        return None

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
            self.logger.debug(f"Error getting version for {scanner_path}: {e}")

        return "unknown"

    async def comprehensive_scan(self, target: ScanTarget) -> ComprehensiveScanResult:
        """Perform comprehensive security scan using multiple tools"""
        scan_id = f"scan_{target.host}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        self.logger.info(f"Starting comprehensive scan {scan_id} for {target.host}")

        try:
            scan_result = ComprehensiveScanResult(
                scan_id=scan_id,
                target=target.host,
                scan_type=target.scan_type,
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

            # Phase 1: Network Discovery and Port Scanning with Nmap
            if self.scanners.get("nmap", {}).get("available"):
                self.logger.info("Phase 1: Network discovery with Nmap")
                nmap_results = await self._run_nmap_scan(target)
                scan_result.open_ports.extend(nmap_results.get("open_ports", []))
                scan_result.services.extend(nmap_results.get("services", []))
                scan_result.os_fingerprint = nmap_results.get("os_fingerprint", {})
                scan_result.raw_output["nmap"] = nmap_results.get("raw_output", "")

                if nmap_results.get("errors"):
                    scan_result.scan_statistics["nmap_errors"] = nmap_results["errors"]

            # Phase 2: Vulnerability Scanning with Nuclei
            if self.scanners.get("nuclei", {}).get("available"):
                self.logger.info("Phase 2: Vulnerability scanning with Nuclei")
                nuclei_results = await self._run_nuclei_scan(target)
                scan_result.vulnerabilities.extend(self._convert_nuclei_findings(nuclei_results.get("vulnerabilities", [])))
                scan_result.raw_output["nuclei"] = nuclei_results.get("raw_output", "")

            # Phase 3: Web Application Scanning
            web_ports = [p for p in scan_result.open_ports if p.get("port") in [80, 443, 8080, 8443, 8000, 3000]]
            if web_ports:
                self.logger.info("Phase 3: Web application scanning")

                # Web vulnerability scanning with Nikto
                if self.scanners.get("nikto", {}).get("available"):
                    for port_info in web_ports:
                        port = port_info.get("port")
                        nikto_results = await self._run_nikto_scan(target.host, port)
                        scan_result.vulnerabilities.extend(self._convert_nikto_findings(nikto_results.get("vulnerabilities", []), port))
                        scan_result.raw_output[f"nikto_{port}"] = nikto_results.get("raw_output", "")

                # Directory/file discovery
                if self.scanners.get("gobuster", {}).get("available") or self.scanners.get("dirb", {}).get("available"):
                    for port_info in web_ports:
                        port = port_info.get("port")
                        web_findings = await self._run_web_discovery(target.host, port)
                        scan_result.vulnerabilities.extend(self._convert_web_findings(web_findings.get("vulnerabilities", []), port))
                        scan_result.raw_output[f"web_discovery_{port}"] = web_findings.get("raw_output", "")

                # SSL/TLS analysis for HTTPS ports
                ssl_ports = [p for p in web_ports if p.get("port") in [443, 8443]]
                if ssl_ports and self.scanners.get("sslscan", {}).get("available"):
                    for port_info in ssl_ports:
                        port = port_info.get("port")
                        ssl_results = await self._run_sslscan(target.host, port)
                        scan_result.vulnerabilities.extend(self._convert_ssl_findings(ssl_results.get("vulnerabilities", []), port))
                        scan_result.raw_output[f"sslscan_{port}"] = ssl_results.get("raw_output", "")

            # Phase 4: Custom Security Analysis
            self.logger.info("Phase 4: Custom security analysis")
            custom_findings = await self._run_custom_security_checks(target, scan_result)
            scan_result.vulnerabilities.extend(custom_findings)

            # Phase 5: Generate final results
            scan_result.end_time = datetime.now()
            scan_result.status = "completed"
            scan_result.scan_statistics.update({
                "duration_seconds": (scan_result.end_time - scan_result.start_time).total_seconds(),
                "ports_scanned": len(target.ports) if target.ports else 1000,
                "open_ports_found": len(scan_result.open_ports),
                "services_identified": len(scan_result.services),
                "vulnerabilities_found": len(scan_result.vulnerabilities),
                "scanners_used": [name for name, config in self.scanners.items() if config.get("available")]
            })

            # Generate security recommendations
            scan_result.recommendations = self._generate_security_recommendations(scan_result)

            self.logger.info(f"Scan {scan_id} completed: {len(scan_result.vulnerabilities)} vulnerabilities found")
            return scan_result

        except Exception as e:
            self.logger.error(f"Comprehensive scan failed: {e}")
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
                nmap_config["path"],
                "-sS",  # SYN scan
                "--max-rate", str(nmap_config["config"]["max_rate"]),
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
            if target.stealth_mode:
                cmd.extend(["-f", "-D", "RND:3"])  # Fragment packets, decoy scans

            self.logger.debug(f"Running Nmap: {' '.join(cmd)}")

            # Execute Nmap scan
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nmap_config["config"]["timeout"]
            )

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"Nmap scan failed: {error_msg}")
                return {"errors": [error_msg]}

            # Parse XML output
            xml_output = stdout.decode('utf-8', errors='ignore')
            return self._parse_nmap_xml(xml_output)

        except asyncio.TimeoutError:
            self.logger.error("Nmap scan timed out")
            return {"errors": ["Nmap scan timed out"]}
        except Exception as e:
            self.logger.error(f"Nmap scan failed: {e}")
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
            self.logger.error(f"Failed to parse Nmap XML: {e}")
            return {"errors": [f"XML parse error: {e}"], "raw_output": xml_output}

    async def _run_nuclei_scan(self, target: ScanTarget) -> Dict[str, Any]:
        """Execute Nuclei vulnerability scanner"""
        try:
            nuclei_config = self.scanners["nuclei"]

            cmd = [
                nuclei_config["path"],
                "-target", f"{target.host}",
                "-json",
                "-severity", "low,medium,high,critical",
                "-rate-limit", str(nuclei_config["config"]["max_rate"]),
                "-timeout", "10",
                "-retries", "1",
                "-no-color"
            ]

            # Add templates if available
            templates_path = nuclei_config["config"].get("templates_path")
            if templates_path:
                expanded_path = Path(templates_path).expanduser()
                if expanded_path.exists():
                    cmd.extend(["-t", str(expanded_path)])

            self.logger.debug(f"Running Nuclei: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nuclei_config["config"]["timeout"]
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
            self.logger.error("Nuclei scan timed out")
            return {"errors": ["Nuclei scan timed out"]}
        except Exception as e:
            self.logger.error(f"Nuclei scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_nikto_scan(self, host: str, port: int) -> Dict[str, Any]:
        """Execute Nikto web vulnerability scanner"""
        try:
            nikto_config = self.scanners["nikto"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}"

            cmd = [
                nikto_config["path"],
                "-h", target_url,
                "-Format", "json",
                "-nointeractive",
                "-maxtime", str(nikto_config["config"]["timeout"]),
                "-timeout", "10"
            ]

            self.logger.debug(f"Running Nikto: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=nikto_config["config"]["timeout"]
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
            self.logger.error(f"Nikto scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_web_discovery(self, host: str, port: int) -> Dict[str, Any]:
        """Run web directory/file discovery"""
        vulnerabilities = []
        raw_output = ""

        try:
            # Try Gobuster first, then fall back to Dirb
            if self.scanners.get("gobuster", {}).get("available"):
                result = await self._run_gobuster(host, port)
            elif self.scanners.get("dirb", {}).get("available"):
                result = await self._run_dirb(host, port)
            else:
                return {"vulnerabilities": [], "raw_output": "No web discovery tools available"}

            return result

        except Exception as e:
            self.logger.error(f"Web discovery failed: {e}")
            return {"errors": [str(e)]}

    async def _run_gobuster(self, host: str, port: int) -> Dict[str, Any]:
        """Execute Gobuster directory bruteforcer"""
        try:
            gobuster_config = self.scanners["gobuster"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}/"

            cmd = [
                gobuster_config["path"],
                "dir",
                "-u", target_url,
                "-w", gobuster_config["config"].get("wordlist", "/usr/share/wordlists/dirb/common.txt"),
                "-q",  # Quiet mode
                "-t", "10",  # Threads
                "--timeout", "10s"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=gobuster_config["config"]["timeout"]
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
            self.logger.error(f"Gobuster scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_dirb(self, host: str, port: int) -> Dict[str, Any]:
        """Execute DIRB directory bruteforcer"""
        try:
            dirb_config = self.scanners["dirb"]

            protocol = "https" if port in [443, 8443] else "http"
            target_url = f"{protocol}://{host}:{port}/"

            cmd = [
                dirb_config["path"],
                target_url,
                dirb_config["config"].get("wordlist", "/usr/share/dirb/wordlists/common.txt"),
                "-S",  # Silent mode
                "-w"   # Don't stop on warning messages
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=dirb_config["config"]["timeout"]
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
            self.logger.error(f"DIRB scan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_sslscan(self, host: str, port: int) -> Dict[str, Any]:
        """Execute SSL/TLS security analysis"""
        try:
            sslscan_config = self.scanners["sslscan"]

            cmd = [
                sslscan_config["path"],
                "--xml=-",
                f"{host}:{port}"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=sslscan_config["config"]["timeout"]
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

            return {
                "vulnerabilities": vulnerabilities,
                "raw_output": output
            }

        except Exception as e:
            self.logger.error(f"SSLScan failed: {e}")
            return {"errors": [str(e)]}

    async def _run_custom_security_checks(self, target: ScanTarget, scan_result: ComprehensiveScanResult) -> List[VulnerabilityFinding]:
        """Run custom security analysis checks"""
        vulnerabilities = []

        try:
            # Check for suspicious port combinations
            open_ports = [p.get("port") for p in scan_result.open_ports]

            # Check for common backdoor ports
            backdoor_ports = [1234, 4444, 5555, 6666, 31337, 12345, 54321]
            for port in open_ports:
                if port in backdoor_ports:
                    vuln = VulnerabilityFinding(
                        vulnerability_id=f"custom_backdoor_{port}",
                        name="Suspicious Port Open",
                        severity="high",
                        cvss_score=7.5,
                        description=f"Port {port} is commonly associated with backdoors and malware",
                        affected_component=f"{target.host}:{port}",
                        port=port,
                        service=None,
                        evidence={"port": port, "reason": "backdoor_port"},
                        references=["https://www.speedguide.net/port.php"],
                        remediation="Investigate the service running on this port and close if unnecessary",
                        scanner="custom_security_checker",
                        timestamp=datetime.now()
                    )
                    vulnerabilities.append(vuln)

            # Check service versions for known vulnerabilities
            for service in scan_result.services:
                service_name = service.get("name", "").lower()
                version = service.get("version", "")
                port = service.get("port")

                # SSH version checks
                if service_name == "ssh" and version:
                    vulnerable_ssh_versions = ["OpenSSH_7.4", "OpenSSH_6.6", "OpenSSH_5.3"]
                    if any(vuln_ver in version for vuln_ver in vulnerable_ssh_versions):
                        vuln = VulnerabilityFinding(
                            vulnerability_id=f"ssh_vuln_{port}",
                            name="Vulnerable SSH Version",
                            severity="medium",
                            cvss_score=5.3,
                            description=f"SSH service running potentially vulnerable version: {version}",
                            affected_component=f"{target.host}:{port}",
                            port=port,
                            service=service_name,
                            evidence={"version": version, "service": service_name},
                            references=["https://www.openssh.com/security.html"],
                            remediation="Update SSH to the latest stable version",
                            scanner="custom_security_checker",
                            timestamp=datetime.now()
                        )
                        vulnerabilities.append(vuln)

            return vulnerabilities

        except Exception as e:
            self.logger.error(f"Custom security checks failed: {e}")
            return []

    def _convert_nuclei_findings(self, nuclei_vulns: List[Dict[str, Any]]) -> List[VulnerabilityFinding]:
        """Convert Nuclei findings to VulnerabilityFinding objects"""
        findings = []

        for vuln in nuclei_vulns:
            finding = VulnerabilityFinding(
                vulnerability_id=vuln.get("template_id", "unknown"),
                name=vuln.get("name", "Unknown Vulnerability"),
                severity=vuln.get("severity", "info"),
                cvss_score=self._severity_to_cvss(vuln.get("severity", "info")),
                description=vuln.get("description", ""),
                affected_component=vuln.get("matched_at", ""),
                port=None,
                service=None,
                evidence=vuln.get("raw_data", {}),
                references=vuln.get("reference", []),
                remediation="Review Nuclei template documentation for specific remediation steps",
                scanner="nuclei",
                timestamp=datetime.now()
            )
            findings.append(finding)

        return findings

    def _convert_nikto_findings(self, nikto_vulns: List[Dict[str, Any]], port: int) -> List[VulnerabilityFinding]:
        """Convert Nikto findings to VulnerabilityFinding objects"""
        findings = []

        for vuln in nikto_vulns:
            finding = VulnerabilityFinding(
                vulnerability_id=f"nikto_{port}_{hash(vuln.get('description', ''))}",
                name=vuln.get("name", "Web Security Issue"),
                severity=vuln.get("severity", "info"),
                cvss_score=self._severity_to_cvss(vuln.get("severity", "info")),
                description=vuln.get("description", ""),
                affected_component=vuln.get("url", ""),
                port=port,
                service="http",
                evidence={"nikto_finding": vuln.get("description", "")},
                references=["https://cirt.net/Nikto2"],
                remediation="Review web application security configuration",
                scanner="nikto",
                timestamp=datetime.now()
            )
            findings.append(finding)

        return findings

    def _convert_web_findings(self, web_vulns: List[Dict[str, Any]], port: int) -> List[VulnerabilityFinding]:
        """Convert web discovery findings to VulnerabilityFinding objects"""
        findings = []

        for vuln in web_vulns:
            finding = VulnerabilityFinding(
                vulnerability_id=f"web_discovery_{port}_{hash(vuln.get('description', ''))}",
                name=vuln.get("name", "Web Discovery Finding"),
                severity=vuln.get("severity", "info"),
                cvss_score=self._severity_to_cvss(vuln.get("severity", "info")),
                description=vuln.get("description", ""),
                affected_component=vuln.get("url", ""),
                port=port,
                service="http",
                evidence={"discovery_finding": vuln.get("description", "")},
                references=["https://tools.kali.org/web-applications/"],
                remediation="Review exposed directories and files for sensitive information",
                scanner=vuln.get("scanner", "web_discovery"),
                timestamp=datetime.now()
            )
            findings.append(finding)

        return findings

    def _convert_ssl_findings(self, ssl_vulns: List[Dict[str, Any]], port: int) -> List[VulnerabilityFinding]:
        """Convert SSL findings to VulnerabilityFinding objects"""
        findings = []

        for vuln in ssl_vulns:
            finding = VulnerabilityFinding(
                vulnerability_id=f"ssl_{port}_{vuln.get('name', '').replace(' ', '_').lower()}",
                name=vuln.get("name", "SSL/TLS Security Issue"),
                severity=vuln.get("severity", "medium"),
                cvss_score=self._severity_to_cvss(vuln.get("severity", "medium")),
                description=vuln.get("description", ""),
                affected_component=f"SSL/TLS on port {port}",
                port=port,
                service="ssl/tls",
                evidence={"ssl_issue": vuln.get("description", "")},
                references=vuln.get("cve", []),
                remediation=vuln.get("remediation", "Review SSL/TLS configuration"),
                scanner="sslscan",
                timestamp=datetime.now()
            )
            findings.append(finding)

        return findings

    def _severity_to_cvss(self, severity: str) -> float:
        """Convert severity string to CVSS score"""
        severity_map = {
            "critical": 9.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5,
            "info": 0.0
        }
        return severity_map.get(severity.lower(), 5.0)

    def _generate_security_recommendations(self, scan_result: ComprehensiveScanResult) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []

        # Critical vulnerabilities
        critical_vulns = [v for v in scan_result.vulnerabilities if v.severity == "critical"]
        if critical_vulns:
            recommendations.append("🚨 CRITICAL: Immediately address critical vulnerabilities to prevent compromise")

        # High severity issues
        high_vulns = [v for v in scan_result.vulnerabilities if v.severity == "high"]
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
            "🏗️ Consider network segmentation for sensitive services"
        ])

        return recommendations

# Factory function for getting scanner instance
_scanner_instance: Optional[RealWorldSecurityScanner] = None

async def get_scanner() -> RealWorldSecurityScanner:
    """Get global scanner instance"""
    global _scanner_instance

    if _scanner_instance is None:
        _scanner_instance = RealWorldSecurityScanner()
        await _scanner_instance.initialize()

    return _scanner_instance
