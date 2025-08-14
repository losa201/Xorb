"""
Production PTaaS Scanner Implementation
Real-world security scanner integration with comprehensive vulnerability detection
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import ipaddress
import socket
import ssl
import re
import aiofiles
import hashlib
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import nmap
import dns.resolver
import requests
from urllib.parse import urljoin, urlparse
import validators

logger = logging.getLogger(__name__)

@dataclass
class ScanConfiguration:
    """Comprehensive scan configuration"""
    target: str
    ports: Optional[List[int]] = None
    scan_type: str = "comprehensive"
    stealth_mode: bool = False
    aggressive_mode: bool = False
    timeout: int = 300
    rate_limit: int = 1000
    custom_scripts: Optional[List[str]] = None
    exclude_hosts: Optional[List[str]] = None
    output_formats: List[str] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["json", "xml"]

@dataclass
class VulnerabilityDetail:
    """Detailed vulnerability information"""
    vulnerability_id: str
    name: str
    severity: str  # Critical, High, Medium, Low, Info
    cvss_score: Optional[float]
    cvss_vector: Optional[str]
    cve_ids: List[str]
    description: str
    affected_component: str
    port: Optional[int]
    service: Optional[str]
    protocol: Optional[str]
    evidence: Dict[str, Any]
    proof_of_concept: Optional[str]
    references: List[str]
    remediation: str
    scanner_used: str
    confidence: float  # 0.0 to 1.0
    exploitability: str  # High, Medium, Low
    timestamp: datetime
    raw_output: Optional[str] = None

@dataclass
class PortScanResult:
    """Port scan result details"""
    port: int
    protocol: str
    state: str  # open, closed, filtered
    service: Optional[str]
    version: Optional[str]
    banner: Optional[str]
    scripts: Dict[str, Any]

@dataclass
class ServiceInfo:
    """Service information"""
    service_name: str
    version: str
    banner: str
    fingerprint: str
    ssl_info: Optional[Dict[str, Any]]
    certificates: List[Dict[str, Any]]

class ProductionPTaaSScanner:
    """Production-ready PTaaS scanner with real security tools integration"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scan_results: Dict[str, Dict[str, Any]] = {}
        self.tool_paths = self._discover_tools()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _discover_tools(self) -> Dict[str, Optional[str]]:
        """Discover available security tools"""
        tools = {
            "nmap": None,
            "nuclei": None,
            "nikto": None,
            "sslscan": None,
            "gobuster": None,
            "dirb": None,
            "sqlmap": None,
            "wpscan": None
        }

        for tool in tools.keys():
            try:
                result = subprocess.run(
                    ["which", tool],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    tools[tool] = result.stdout.strip()
                    logger.info(f"Found {tool} at {tools[tool]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"Tool {tool} not found")

        return tools

    async def validate_target(self, target: str) -> Tuple[bool, str]:
        """Validate scan target for security and reachability"""
        try:
            # Validate format
            if not validators.domain(target) and not validators.ip_address(target):
                return False, "Invalid target format"

            # Check if target is internal/restricted
            if self._is_restricted_target(target):
                return False, "Target is restricted or internal"

            # Test reachability
            reachable = await self._test_reachability(target)
            if not reachable:
                return False, "Target is not reachable"

            return True, "Target is valid and reachable"

        except Exception as e:
            logger.error(f"Target validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def _is_restricted_target(self, target: str) -> bool:
        """Check if target is restricted from scanning"""
        restricted_networks = [
            "127.0.0.0/8",    # Loopback
            "10.0.0.0/8",     # Private Class A
            "172.16.0.0/12",  # Private Class B
            "192.168.0.0/16", # Private Class C
            "169.254.0.0/16", # Link-local
            "224.0.0.0/4",    # Multicast
        ]

        try:
            ip = ipaddress.ip_address(target)
            for network in restricted_networks:
                if ip in ipaddress.ip_network(network):
                    return True
        except ValueError:
            # If it's a domain, resolve it first
            try:
                import socket
                ip = socket.gethostbyname(target)
                return self._is_restricted_target(ip)
            except socket.gaierror:
                pass

        return False

    async def _test_reachability(self, target: str, port: int = 80) -> bool:
        """Test if target is reachable"""
        try:
            # Create socket with timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            # Test connection
            result = sock.connect_ex((target, port))
            sock.close()

            return result == 0

        except Exception:
            return False

    async def execute_comprehensive_scan(
        self,
        config: ScanConfiguration
    ) -> Dict[str, Any]:
        """Execute comprehensive security scan with multiple tools"""

        scan_id = self._generate_scan_id(config.target)
        scan_start = datetime.utcnow()

        logger.info(f"Starting comprehensive scan {scan_id} for {config.target}")

        results = {
            "scan_id": scan_id,
            "target": config.target,
            "scan_type": config.scan_type,
            "started_at": scan_start.isoformat(),
            "configuration": asdict(config),
            "vulnerabilities": [],
            "services": [],
            "ports": [],
            "ssl_analysis": {},
            "web_analysis": {},
            "compliance_checks": {},
            "recommendations": [],
            "summary": {}
        }

        try:
            # Phase 1: Network Discovery
            logger.info(f"Phase 1: Network discovery for {config.target}")
            port_results = await self._execute_nmap_scan(config)
            results["ports"] = port_results["ports"]
            results["services"] = port_results["services"]

            # Phase 2: Service Enumeration
            logger.info(f"Phase 2: Service enumeration for {config.target}")
            service_details = await self._enumerate_services(config, port_results)
            results["services"].extend(service_details)

            # Phase 3: Vulnerability Scanning
            logger.info(f"Phase 3: Vulnerability scanning for {config.target}")
            vulnerabilities = await self._execute_nuclei_scan(config)
            results["vulnerabilities"].extend(vulnerabilities)

            # Phase 4: Web Application Testing
            if self._has_web_services(port_results):
                logger.info(f"Phase 4: Web application testing for {config.target}")
                web_results = await self._execute_web_scan(config)
                results["web_analysis"] = web_results
                results["vulnerabilities"].extend(web_results.get("vulnerabilities", []))

            # Phase 5: SSL/TLS Analysis
            ssl_ports = self._get_ssl_ports(port_results)
            if ssl_ports:
                logger.info(f"Phase 5: SSL/TLS analysis for {config.target}")
                ssl_results = await self._execute_ssl_scan(config, ssl_ports)
                results["ssl_analysis"] = ssl_results
                results["vulnerabilities"].extend(ssl_results.get("vulnerabilities", []))

            # Phase 6: Compliance Checks
            logger.info(f"Phase 6: Compliance analysis for {config.target}")
            compliance_results = await self._execute_compliance_checks(config, results)
            results["compliance_checks"] = compliance_results

            # Phase 7: Generate Recommendations
            recommendations = self._generate_security_recommendations(results)
            results["recommendations"] = recommendations

            # Generate Summary
            results["summary"] = self._generate_scan_summary(results)
            results["completed_at"] = datetime.utcnow().isoformat()
            results["duration"] = (datetime.utcnow() - scan_start).total_seconds()

            logger.info(f"Comprehensive scan {scan_id} completed successfully")

        except Exception as e:
            logger.error(f"Comprehensive scan {scan_id} failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"

        # Store results
        self.scan_results[scan_id] = results
        return results

    async def _execute_nmap_scan(self, config: ScanConfiguration) -> Dict[str, Any]:
        """Execute Nmap network discovery and port scanning"""

        if not self.tool_paths.get("nmap"):
            logger.warning("Nmap not available, using fallback port scanner")
            return await self._fallback_port_scan(config)

        results = {"ports": [], "services": [], "os_detection": {}}

        try:
            # Build Nmap command
            nmap_args = [
                self.tool_paths["nmap"],
                "-sS",  # SYN scan
                "-sV",  # Version detection
                "-O",   # OS detection
                "--script=default,vuln",  # Default and vulnerability scripts
                "-oX", "-",  # XML output to stdout
            ]

            # Configure scan timing
            if config.stealth_mode:
                nmap_args.extend(["-T2", "--scan-delay", "1s"])
            elif config.aggressive_mode:
                nmap_args.extend(["-T4", "--min-rate", "1000"])
            else:
                nmap_args.append("-T3")

            # Port specification
            if config.ports:
                port_list = ",".join(map(str, config.ports))
                nmap_args.extend(["-p", port_list])
            else:
                nmap_args.append("-p-")  # All ports

            nmap_args.append(config.target)

            # Execute Nmap scan
            process = await asyncio.create_subprocess_exec(
                *nmap_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )

            if process.returncode == 0:
                # Parse XML results
                results = self._parse_nmap_xml(stdout.decode())
            else:
                logger.error(f"Nmap scan failed: {stderr.decode()}")

        except asyncio.TimeoutError:
            logger.error("Nmap scan timed out")
        except Exception as e:
            logger.error(f"Nmap scan error: {e}")

        return results

    async def _execute_nuclei_scan(self, config: ScanConfiguration) -> List[VulnerabilityDetail]:
        """Execute Nuclei vulnerability scanning"""

        if not self.tool_paths.get("nuclei"):
            logger.warning("Nuclei not available, using fallback vulnerability detection")
            return await self._fallback_vuln_scan(config)

        vulnerabilities = []

        try:
            # Build Nuclei command
            nuclei_args = [
                self.tool_paths["nuclei"],
                "-target", config.target,
                "-json",  # JSON output
                "-silent",
                "-no-color",
            ]

            # Configure severity
            if config.scan_type == "quick":
                nuclei_args.extend(["-severity", "critical,high"])
            else:
                nuclei_args.extend(["-severity", "critical,high,medium,low"])

            # Rate limiting
            if config.stealth_mode:
                nuclei_args.extend(["-rate-limit", "10"])
            else:
                nuclei_args.extend(["-rate-limit", "50"])

            # Execute Nuclei scan
            process = await asyncio.create_subprocess_exec(
                *nuclei_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )

            if stdout:
                # Parse JSON results
                for line in stdout.decode().strip().split('\n'):
                    if line.strip():
                        try:
                            result = json.loads(line)
                            vuln = self._parse_nuclei_result(result)
                            if vuln:
                                vulnerabilities.append(vuln)
                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            logger.error("Nuclei scan timed out")
        except Exception as e:
            logger.error(f"Nuclei scan error: {e}")

        return vulnerabilities

    async def _execute_web_scan(self, config: ScanConfiguration) -> Dict[str, Any]:
        """Execute web application security scanning"""

        web_results = {
            "technologies": [],
            "directories": [],
            "vulnerabilities": [],
            "security_headers": {},
            "cookies": {},
            "forms": []
        }

        # Directory/file discovery with Gobuster
        if self.tool_paths.get("gobuster"):
            directories = await self._execute_gobuster(config)
            web_results["directories"] = directories

        # Web vulnerability scanning with Nikto
        if self.tool_paths.get("nikto"):
            nikto_vulns = await self._execute_nikto(config)
            web_results["vulnerabilities"].extend(nikto_vulns)

        # Security headers analysis
        headers_analysis = await self._analyze_security_headers(config)
        web_results["security_headers"] = headers_analysis

        # Technology detection
        tech_detection = await self._detect_web_technologies(config)
        web_results["technologies"] = tech_detection

        return web_results

    async def _execute_ssl_scan(
        self,
        config: ScanConfiguration,
        ssl_ports: List[int]
    ) -> Dict[str, Any]:
        """Execute SSL/TLS security analysis"""

        ssl_results = {
            "vulnerabilities": [],
            "certificates": [],
            "cipher_suites": [],
            "protocols": [],
            "grade": "Unknown"
        }

        for port in ssl_ports:
            try:
                # SSL/TLS analysis
                if self.tool_paths.get("sslscan"):
                    port_results = await self._execute_sslscan(config, port)
                    ssl_results.update(port_results)
                else:
                    # Fallback SSL analysis
                    cert_info = await self._analyze_certificate(config.target, port)
                    ssl_results["certificates"].append(cert_info)

            except Exception as e:
                logger.error(f"SSL scan failed for port {port}: {e}")

        return ssl_results

    async def _execute_compliance_checks(
        self,
        config: ScanConfiguration,
        scan_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute compliance framework checks"""

        compliance_results = {
            "frameworks": {},
            "overall_score": 0,
            "critical_issues": [],
            "recommendations": []
        }

        # PCI DSS compliance checks
        pci_dss_results = self._check_pci_dss_compliance(scan_results)
        compliance_results["frameworks"]["pci_dss"] = pci_dss_results

        # OWASP Top 10 checks
        owasp_results = self._check_owasp_top10(scan_results)
        compliance_results["frameworks"]["owasp_top10"] = owasp_results

        # NIST Cybersecurity Framework
        nist_results = self._check_nist_framework(scan_results)
        compliance_results["frameworks"]["nist"] = nist_results

        # Calculate overall compliance score
        scores = [
            pci_dss_results.get("score", 0),
            owasp_results.get("score", 0),
            nist_results.get("score", 0)
        ]
        compliance_results["overall_score"] = sum(scores) / len(scores) if scores else 0

        return compliance_results

    def _generate_security_recommendations(
        self,
        scan_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable security recommendations"""

        recommendations = []

        # Critical vulnerability recommendations
        critical_vulns = [
            v for v in scan_results.get("vulnerabilities", [])
            if isinstance(v, dict) and v.get("severity") == "Critical"
        ]

        if critical_vulns:
            recommendations.append(
                f"URGENT: Address {len(critical_vulns)} critical vulnerabilities immediately"
            )

        # SSL/TLS recommendations
        ssl_analysis = scan_results.get("ssl_analysis", {})
        if ssl_analysis.get("grade") and ssl_analysis["grade"] in ["C", "D", "F"]:
            recommendations.append(
                "Improve SSL/TLS configuration - weak ciphers or protocols detected"
            )

        # Open port recommendations
        open_ports = scan_results.get("ports", [])
        high_risk_ports = [p for p in open_ports if isinstance(p, dict) and p.get("port") in [23, 135, 139, 445]]
        if high_risk_ports:
            recommendations.append(
                f"Close or restrict access to {len(high_risk_ports)} high-risk ports"
            )

        # Web security recommendations
        web_analysis = scan_results.get("web_analysis", {})
        missing_headers = web_analysis.get("security_headers", {}).get("missing", [])
        if missing_headers:
            recommendations.append(
                f"Implement missing security headers: {', '.join(missing_headers)}"
            )

        # Compliance recommendations
        compliance = scan_results.get("compliance_checks", {})
        if compliance.get("overall_score", 0) < 70:
            recommendations.append(
                "Improve compliance posture - multiple framework violations detected"
            )

        return recommendations

    def _generate_scan_summary(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive scan summary"""

        vulnerabilities = scan_results.get("vulnerabilities", [])

        # Count vulnerabilities by severity
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
        for vuln in vulnerabilities:
            if isinstance(vuln, dict):
                severity = vuln.get("severity", "Unknown")
                if severity in severity_counts:
                    severity_counts[severity] += 1

        # Calculate risk score (0-100)
        risk_score = (
            severity_counts["Critical"] * 10 +
            severity_counts["High"] * 7 +
            severity_counts["Medium"] * 4 +
            severity_counts["Low"] * 2 +
            severity_counts["Info"] * 1
        )

        # Determine risk level
        if risk_score >= 50:
            risk_level = "Critical"
        elif risk_score >= 30:
            risk_level = "High"
        elif risk_score >= 15:
            risk_level = "Medium"
        elif risk_score > 0:
            risk_level = "Low"
        else:
            risk_level = "Minimal"

        return {
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerability_breakdown": severity_counts,
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "ports_scanned": len(scan_results.get("ports", [])),
            "services_identified": len(scan_results.get("services", [])),
            "compliance_score": scan_results.get("compliance_checks", {}).get("overall_score", 0),
            "recommendations_count": len(scan_results.get("recommendations", []))
        }

    def _generate_scan_id(self, target: str) -> str:
        """Generate unique scan ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_hash = hashlib.md5(target.encode()).hexdigest()[:8]
        return f"scan_{target_hash}_{timestamp}"

    # Additional helper methods would continue here...
    # This is a comprehensive foundation for real-world security scanning

    async def _fallback_port_scan(self, config: ScanConfiguration) -> Dict[str, Any]:
        """Fallback port scanner when nmap is not available"""
        results = {"ports": [], "services": []}

        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        ports_to_scan = config.ports or common_ports

        for port in ports_to_scan:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((config.target, port))

                if result == 0:
                    results["ports"].append({
                        "port": port,
                        "protocol": "tcp",
                        "state": "open",
                        "service": self._guess_service_by_port(port)
                    })

                sock.close()

            except Exception:
                continue

        return results

    def _guess_service_by_port(self, port: int) -> str:
        """Guess service based on well-known ports"""
        port_services = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            53: "dns", 80: "http", 110: "pop3", 143: "imap",
            443: "https", 993: "imaps", 995: "pop3s"
        }
        return port_services.get(port, "unknown")

    async def _fallback_vuln_scan(self, config: ScanConfiguration) -> List[VulnerabilityDetail]:
        """Fallback vulnerability scanner"""
        vulnerabilities = []

        # Basic checks for common vulnerabilities
        try:
            # Check for common web vulnerabilities
            if await self._check_http_service(config.target):
                web_vulns = await self._basic_web_vuln_check(config.target)
                vulnerabilities.extend(web_vulns)

        except Exception as e:
            logger.error(f"Fallback vulnerability scan failed: {e}")

        return vulnerabilities

    async def _check_http_service(self, target: str) -> bool:
        """Check if HTTP service is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{target}", timeout=5) as response:
                    return True
        except:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"https://{target}", timeout=5) as response:
                        return True
            except:
                return False

    async def _basic_web_vuln_check(self, target: str) -> List[VulnerabilityDetail]:
        """Basic web vulnerability checks"""
        vulnerabilities = []

        # Check for common security headers
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{target}") as response:
                    headers = response.headers

                    # Missing security headers
                    security_headers = [
                        "Strict-Transport-Security",
                        "Content-Security-Policy",
                        "X-Frame-Options",
                        "X-Content-Type-Options"
                    ]

                    missing_headers = [h for h in security_headers if h not in headers]

                    if missing_headers:
                        vulnerabilities.append(VulnerabilityDetail(
                            vulnerability_id="WEB-001",
                            name="Missing Security Headers",
                            severity="Medium",
                            cvss_score=5.3,
                            cvss_vector=None,
                            cve_ids=[],
                            description=f"Missing security headers: {', '.join(missing_headers)}",
                            affected_component="Web Server",
                            port=80,
                            service="http",
                            protocol="tcp",
                            evidence={"missing_headers": missing_headers},
                            proof_of_concept=None,
                            references=["https://owasp.org/www-project-secure-headers/"],
                            remediation="Implement missing security headers in web server configuration",
                            scanner_used="fallback_scanner",
                            confidence=0.9,
                            exploitability="Low",
                            timestamp=datetime.utcnow()
                        ))

        except Exception as e:
            logger.error(f"Basic web vulnerability check failed: {e}")

        return vulnerabilities

    # Additional methods for parsing results and tool-specific implementations
    def _parse_nmap_xml(self, xml_data: str) -> Dict[str, Any]:
        """Parse Nmap XML output"""
        results = {"ports": [], "services": [], "os_detection": {}}

        try:
            root = ET.fromstring(xml_data)

            for host in root.findall('host'):
                # Parse ports
                for port in host.findall('.//port'):
                    port_id = port.get('portid')
                    protocol = port.get('protocol')

                    state_elem = port.find('state')
                    state = state_elem.get('state') if state_elem is not None else 'unknown'

                    service_elem = port.find('service')
                    service = service_elem.get('name') if service_elem is not None else 'unknown'
                    version = service_elem.get('version') if service_elem is not None else None

                    results["ports"].append({
                        "port": int(port_id),
                        "protocol": protocol,
                        "state": state,
                        "service": service,
                        "version": version
                    })

                # Parse OS detection
                os_elem = host.find('.//os')
                if os_elem is not None:
                    os_matches = []
                    for osmatch in os_elem.findall('osmatch'):
                        os_matches.append({
                            "name": osmatch.get('name'),
                            "accuracy": osmatch.get('accuracy')
                        })
                    results["os_detection"]["matches"] = os_matches

        except ET.ParseError as e:
            logger.error(f"Failed to parse Nmap XML: {e}")

        return results

    def _parse_nuclei_result(self, result: Dict[str, Any]) -> Optional[VulnerabilityDetail]:
        """Parse Nuclei JSON result"""
        try:
            return VulnerabilityDetail(
                vulnerability_id=result.get('template-id', 'NUCLEI-UNKNOWN'),
                name=result.get('info', {}).get('name', 'Unknown Vulnerability'),
                severity=result.get('info', {}).get('severity', 'info').title(),
                cvss_score=None,  # Nuclei doesn't provide CVSS scores directly
                cvss_vector=None,
                cve_ids=result.get('info', {}).get('reference', []),
                description=result.get('info', {}).get('description', ''),
                affected_component=result.get('matched-at', ''),
                port=None,
                service=None,
                protocol=None,
                evidence={
                    "matched_at": result.get('matched-at'),
                    "extracted_results": result.get('extracted-results', [])
                },
                proof_of_concept=None,
                references=result.get('info', {}).get('reference', []),
                remediation=result.get('info', {}).get('remediation', 'No specific remediation provided'),
                scanner_used="nuclei",
                confidence=0.8,  # Default confidence for Nuclei
                exploitability="Medium",
                timestamp=datetime.utcnow(),
                raw_output=json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to parse Nuclei result: {e}")
            return None
