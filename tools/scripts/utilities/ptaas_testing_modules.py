#!/usr/bin/env python3
"""
XORB PTaaS Automated Testing Modules
Advanced penetration testing agents with specialized capabilities
"""

import asyncio
import json
import logging
import time
import subprocess
import docker
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid
import aiohttp
import nmap
import requests
from urllib.parse import urljoin, urlparse
import socket
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    timestamp: datetime
    agent_id: str
    target: str
    scan_type: str
    status: str
    findings: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float

class BasePTaaSAgent(ABC):
    """Base class for all PTaaS testing agents"""

    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = "ready"
        self.current_target = None
        self.performance_metrics = {
            "scans_completed": 0,
            "findings_discovered": 0,
            "average_scan_time": 0.0,
            "success_rate": 0.0
        }

    @abstractmethod
    async def execute_scan(self, target: str, scan_options: Dict[str, Any]) -> ScanResult:
        """Execute a security scan against the target"""
        pass

    def update_metrics(self, scan_result: ScanResult):
        """Update agent performance metrics"""
        self.performance_metrics["scans_completed"] += 1
        self.performance_metrics["findings_discovered"] += len(scan_result.findings)

        # Update average scan time
        total_time = (self.performance_metrics["average_scan_time"] *
                     (self.performance_metrics["scans_completed"] - 1) +
                     scan_result.execution_time)
        self.performance_metrics["average_scan_time"] = total_time / self.performance_metrics["scans_completed"]

        # Update success rate
        successful_scans = sum(1 for _ in range(self.performance_metrics["scans_completed"])
                              if scan_result.status == "completed")
        self.performance_metrics["success_rate"] = successful_scans / self.performance_metrics["scans_completed"]

class NetworkReconnaissanceAgent(BasePTaaSAgent):
    """Advanced network reconnaissance and discovery agent"""

    def __init__(self):
        super().__init__(
            agent_id="AGENT-NETWORK-RECON-001",
            name="Advanced Network Reconnaissance Agent",
            capabilities=["network_scanning", "port_scanning", "service_discovery", "os_fingerprinting"]
        )
        self.nmap_scanner = nmap.PortScanner()

    async def execute_scan(self, target: str, scan_options: Dict[str, Any]) -> ScanResult:
        """Execute comprehensive network reconnaissance"""
        start_time = time.time()
        findings = []

        try:
            self.status = "scanning"
            self.current_target = target

            # Extract IP/domain from target
            target_host = self._extract_host(target)

            # Port scan
            port_findings = await self._port_scan(target_host, scan_options)
            findings.extend(port_findings)

            # Service detection
            service_findings = await self._service_detection(target_host, scan_options)
            findings.extend(service_findings)

            # OS fingerprinting
            os_findings = await self._os_fingerprinting(target_host, scan_options)
            findings.extend(os_findings)

            # DNS enumeration
            dns_findings = await self._dns_enumeration(target_host, scan_options)
            findings.extend(dns_findings)

            execution_time = time.time() - start_time

            scan_result = ScanResult(
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                target=target,
                scan_type="network_reconnaissance",
                status="completed",
                findings=findings,
                metadata={
                    "scan_options": scan_options,
                    "target_host": target_host,
                    "scan_techniques": ["port_scan", "service_detection", "os_fingerprinting", "dns_enum"]
                },
                execution_time=execution_time
            )

            self.update_metrics(scan_result)
            return scan_result

        except Exception as e:
            logger.error(f"Network reconnaissance failed for {target}: {e}")
            return ScanResult(
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                target=target,
                scan_type="network_reconnaissance",
                status="failed",
                findings=[],
                metadata={"error": str(e)},
                execution_time=time.time() - start_time
            )
        finally:
            self.status = "ready"
            self.current_target = None

    def _extract_host(self, target: str) -> str:
        """Extract hostname/IP from target URL or address"""
        if target.startswith(('http://', 'https://')):
            return urlparse(target).netloc.split(':')[0]
        return target.split(':')[0]

    async def _port_scan(self, target_host: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform comprehensive port scanning"""
        findings = []

        try:
            # Define port ranges
            top_ports = "1-1000" if scan_options.get("quick_scan") else "1-65535"

            # Nmap scan
            scan_result = self.nmap_scanner.scan(target_host, top_ports, arguments='-sS -sV -O --script vuln')

            if target_host in scan_result['scan']:
                host_info = scan_result['scan'][target_host]

                # Open ports
                if 'tcp' in host_info:
                    open_ports = []
                    for port, port_info in host_info['tcp'].items():
                        if port_info['state'] == 'open':
                            open_ports.append({
                                "port": port,
                                "service": port_info.get('name', 'unknown'),
                                "version": port_info.get('version', 'unknown'),
                                "product": port_info.get('product', 'unknown')
                            })

                    if open_ports:
                        findings.append({
                            "type": "open_ports",
                            "severity": "informational",
                            "title": f"Open Ports Discovered on {target_host}",
                            "description": f"Found {len(open_ports)} open ports during reconnaissance",
                            "details": {
                                "open_ports": open_ports,
                                "total_ports_scanned": len(host_info['tcp'])
                            }
                        })

        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            # Fallback to basic socket scanning
            findings.extend(await self._basic_port_scan(target_host))

        return findings

    async def _basic_port_scan(self, target_host: str) -> List[Dict[str, Any]]:
        """Basic socket-based port scanning as fallback"""
        findings = []
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
        open_ports = []

        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((target_host, port))
                if result == 0:
                    open_ports.append({"port": port, "service": "unknown", "version": "unknown"})
                sock.close()
            except:
                pass

        if open_ports:
            findings.append({
                "type": "open_ports",
                "severity": "informational",
                "title": f"Open Ports Discovered (Basic Scan)",
                "description": f"Found {len(open_ports)} open ports using basic scanning",
                "details": {"open_ports": open_ports}
            })

        return findings

    async def _service_detection(self, target_host: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect services running on open ports"""
        findings = []

        # Check common web services
        web_ports = [80, 443, 8080, 8443, 8000, 3000]
        for port in web_ports:
            try:
                protocol = "https" if port in [443, 8443] else "http"
                url = f"{protocol}://{target_host}:{port}"

                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(url) as response:
                        server_header = response.headers.get('Server', 'unknown')

                        findings.append({
                            "type": "web_service",
                            "severity": "informational",
                            "title": f"Web Service Detected on Port {port}",
                            "description": f"HTTP/HTTPS service running on {target_host}:{port}",
                            "details": {
                                "url": url,
                                "status_code": response.status,
                                "server": server_header,
                                "headers": dict(response.headers)
                            }
                        })
            except:
                pass

        return findings

    async def _os_fingerprinting(self, target_host: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Attempt to fingerprint the target operating system"""
        findings = []

        try:
            # Simple TTL-based OS detection
            response = subprocess.run(['ping', '-c', '1', target_host],
                                    capture_output=True, text=True, timeout=5)

            if response.returncode == 0:
                output = response.stdout
                if 'ttl=' in output.lower():
                    ttl_str = output.lower().split('ttl=')[1].split()[0]
                    ttl = int(ttl_str)

                    # Basic TTL-based OS detection
                    os_guess = "Unknown"
                    if ttl <= 64:
                        os_guess = "Linux/Unix"
                    elif ttl <= 128:
                        os_guess = "Windows"
                    elif ttl <= 255:
                        os_guess = "Cisco/Network Device"

                    findings.append({
                        "type": "os_fingerprint",
                        "severity": "informational",
                        "title": "Operating System Fingerprint",
                        "description": f"Estimated OS: {os_guess}",
                        "details": {
                            "ttl": ttl,
                            "estimated_os": os_guess,
                            "confidence": "low",
                            "method": "TTL analysis"
                        }
                    })

        except Exception as e:
            logger.debug(f"OS fingerprinting failed: {e}")

        return findings

    async def _dns_enumeration(self, target_host: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform DNS enumeration and subdomain discovery"""
        findings = []

        try:
            # Common subdomain prefixes
            subdomains = ['www', 'mail', 'ftp', 'admin', 'api', 'dev', 'staging', 'test']
            found_subdomains = []

            for subdomain in subdomains:
                try:
                    full_domain = f"{subdomain}.{target_host}"
                    socket.gethostbyname(full_domain)
                    found_subdomains.append(full_domain)
                except:
                    pass

            if found_subdomains:
                findings.append({
                    "type": "subdomain_discovery",
                    "severity": "informational",
                    "title": "Subdomains Discovered",
                    "description": f"Found {len(found_subdomains)} subdomains for {target_host}",
                    "details": {
                        "subdomains": found_subdomains,
                        "base_domain": target_host
                    }
                })

        except Exception as e:
            logger.debug(f"DNS enumeration failed: {e}")

        return findings

class WebApplicationAgent(BasePTaaSAgent):
    """Advanced web application security testing agent"""

    def __init__(self):
        super().__init__(
            agent_id="AGENT-WEB-APP-002",
            name="Advanced Web Application Security Agent",
            capabilities=["web_scanning", "sql_injection", "xss_testing", "directory_traversal", "authentication_bypass"]
        )

    async def execute_scan(self, target: str, scan_options: Dict[str, Any]) -> ScanResult:
        """Execute comprehensive web application security testing"""
        start_time = time.time()
        findings = []

        try:
            self.status = "scanning"
            self.current_target = target

            # Web crawler and directory discovery
            crawler_findings = await self._web_crawler(target, scan_options)
            findings.extend(crawler_findings)

            # SQL injection testing
            sqli_findings = await self._sql_injection_test(target, scan_options)
            findings.extend(sqli_findings)

            # XSS testing
            xss_findings = await self._xss_testing(target, scan_options)
            findings.extend(xss_findings)

            # Directory traversal testing
            traversal_findings = await self._directory_traversal_test(target, scan_options)
            findings.extend(traversal_findings)

            # SSL/TLS security testing
            ssl_findings = await self._ssl_security_test(target, scan_options)
            findings.extend(ssl_findings)

            execution_time = time.time() - start_time

            scan_result = ScanResult(
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                target=target,
                scan_type="web_application_security",
                status="completed",
                findings=findings,
                metadata={
                    "scan_options": scan_options,
                    "test_types": ["crawler", "sqli", "xss", "directory_traversal", "ssl"]
                },
                execution_time=execution_time
            )

            self.update_metrics(scan_result)
            return scan_result

        except Exception as e:
            logger.error(f"Web application scan failed for {target}: {e}")
            return ScanResult(
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                target=target,
                scan_type="web_application_security",
                status="failed",
                findings=[],
                metadata={"error": str(e)},
                execution_time=time.time() - start_time
            )
        finally:
            self.status = "ready"
            self.current_target = None

    async def _web_crawler(self, target: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crawl web application and discover endpoints"""
        findings = []
        discovered_paths = []

        try:
            # Common web paths and files
            common_paths = [
                '/admin', '/admin.php', '/administrator', '/login', '/login.php',
                '/config', '/config.php', '/wp-admin', '/wp-config.php',
                '/backup', '/backup.zip', '/database.sql', '/robots.txt',
                '/sitemap.xml', '/.git', '/.env', '/debug', '/test'
            ]

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for path in common_paths[:10]:  # Limit for demo
                    try:
                        url = urljoin(target, path)
                        async with session.get(url) as response:
                            if response.status == 200:
                                discovered_paths.append({
                                    "path": path,
                                    "url": url,
                                    "status_code": response.status,
                                    "content_length": len(await response.text())
                                })
                    except:
                        pass

            if discovered_paths:
                findings.append({
                    "type": "directory_discovery",
                    "severity": "informational",
                    "title": "Web Directories and Files Discovered",
                    "description": f"Found {len(discovered_paths)} accessible paths",
                    "details": {
                        "discovered_paths": discovered_paths,
                        "target_url": target
                    }
                })

        except Exception as e:
            logger.debug(f"Web crawler failed: {e}")

        return findings

    async def _sql_injection_test(self, target: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test for SQL injection vulnerabilities"""
        findings = []

        try:
            # Common SQL injection payloads
            sqli_payloads = [
                "' OR '1'='1",
                "' OR 1=1--",
                "'; DROP TABLE users;--",
                "' UNION SELECT 1,2,3--",
                "admin'--"
            ]

            # Test parameters
            test_params = ['id', 'user', 'username', 'email', 'search', 'q']

            vulnerable_endpoints = []

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for param in test_params[:3]:  # Limit for demo
                    for payload in sqli_payloads[:2]:  # Limit for demo
                        try:
                            test_url = f"{target}?{param}={payload}"
                            async with session.get(test_url) as response:
                                response_text = await response.text()

                                # Check for SQL error indicators
                                sql_errors = [
                                    'sql syntax', 'mysql error', 'warning: mysql',
                                    'ora-01756', 'microsoft jet database',
                                    'sqlite error', 'postgresql error'
                                ]

                                if any(error in response_text.lower() for error in sql_errors):
                                    vulnerable_endpoints.append({
                                        "url": test_url,
                                        "parameter": param,
                                        "payload": payload,
                                        "evidence": "SQL error in response"
                                    })
                        except:
                            pass

            if vulnerable_endpoints:
                findings.append({
                    "type": "sql_injection",
                    "severity": "high",
                    "title": "SQL Injection Vulnerability Detected",
                    "description": f"Found {len(vulnerable_endpoints)} potentially vulnerable endpoints",
                    "details": {
                        "vulnerable_endpoints": vulnerable_endpoints,
                        "impact": "Potential database access and data exposure",
                        "recommendation": "Implement parameterized queries and input validation"
                    }
                })

        except Exception as e:
            logger.debug(f"SQL injection test failed: {e}")

        return findings

    async def _xss_testing(self, target: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test for Cross-Site Scripting vulnerabilities"""
        findings = []

        try:
            # XSS payloads
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "'><script>alert('XSS')</script>"
            ]

            test_params = ['search', 'q', 'query', 'comment', 'message']
            vulnerable_endpoints = []

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for param in test_params[:2]:  # Limit for demo
                    for payload in xss_payloads[:2]:  # Limit for demo
                        try:
                            test_url = f"{target}?{param}={payload}"
                            async with session.get(test_url) as response:
                                response_text = await response.text()

                                # Check if payload is reflected in response
                                if payload in response_text:
                                    vulnerable_endpoints.append({
                                        "url": test_url,
                                        "parameter": param,
                                        "payload": payload,
                                        "type": "reflected"
                                    })
                        except:
                            pass

            if vulnerable_endpoints:
                findings.append({
                    "type": "xss_vulnerability",
                    "severity": "medium",
                    "title": "Cross-Site Scripting (XSS) Vulnerability",
                    "description": f"Found {len(vulnerable_endpoints)} XSS vulnerabilities",
                    "details": {
                        "vulnerable_endpoints": vulnerable_endpoints,
                        "impact": "Session hijacking and malicious script execution",
                        "recommendation": "Implement proper input encoding and validation"
                    }
                })

        except Exception as e:
            logger.debug(f"XSS testing failed: {e}")

        return findings

    async def _directory_traversal_test(self, target: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test for directory traversal vulnerabilities"""
        findings = []

        try:
            # Directory traversal payloads
            traversal_payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "../../../../etc/shadow",
                "../../../etc/hosts"
            ]

            test_params = ['file', 'path', 'include', 'page', 'doc']
            vulnerable_endpoints = []

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for param in test_params[:2]:  # Limit for demo
                    for payload in traversal_payloads[:2]:  # Limit for demo
                        try:
                            test_url = f"{target}?{param}={payload}"
                            async with session.get(test_url) as response:
                                response_text = await response.text()

                                # Check for system file indicators
                                system_indicators = [
                                    'root:x:', 'bin/bash', '127.0.0.1',
                                    '[boot loader]', 'localhost'
                                ]

                                if any(indicator in response_text for indicator in system_indicators):
                                    vulnerable_endpoints.append({
                                        "url": test_url,
                                        "parameter": param,
                                        "payload": payload,
                                        "evidence": "System file content detected"
                                    })
                        except:
                            pass

            if vulnerable_endpoints:
                findings.append({
                    "type": "directory_traversal",
                    "severity": "high",
                    "title": "Directory Traversal Vulnerability",
                    "description": f"Found {len(vulnerable_endpoints)} directory traversal vulnerabilities",
                    "details": {
                        "vulnerable_endpoints": vulnerable_endpoints,
                        "impact": "Access to system files and sensitive information",
                        "recommendation": "Implement proper path validation and file access controls"
                    }
                })

        except Exception as e:
            logger.debug(f"Directory traversal test failed: {e}")

        return findings

    async def _ssl_security_test(self, target: str, scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test SSL/TLS security configuration"""
        findings = []

        try:
            if not target.startswith('https://'):
                return findings

            parsed_url = urlparse(target)
            hostname = parsed_url.netloc.split(':')[0]
            port = parsed_url.port or 443

            # Test SSL/TLS connection
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()

                    # Check certificate validity
                    if cert:
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days

                        if days_until_expiry < 30:
                            findings.append({
                                "type": "ssl_certificate_expiry",
                                "severity": "medium",
                                "title": "SSL Certificate Expiring Soon",
                                "description": f"Certificate expires in {days_until_expiry} days",
                                "details": {
                                    "expiry_date": cert['notAfter'],
                                    "days_remaining": days_until_expiry,
                                    "subject": cert.get('subject', [])
                                }
                            })

                    # Check cipher strength
                    if cipher and len(cipher) >= 3:
                        cipher_name = cipher[0]
                        if any(weak in cipher_name.lower() for weak in ['rc4', 'des', 'md5']):
                            findings.append({
                                "type": "weak_ssl_cipher",
                                "severity": "medium",
                                "title": "Weak SSL/TLS Cipher Detected",
                                "description": f"Weak cipher in use: {cipher_name}",
                                "details": {
                                    "cipher": cipher_name,
                                    "recommendation": "Use strong ciphers like AES-256-GCM"
                                }
                            })

        except Exception as e:
            logger.debug(f"SSL security test failed: {e}")

        return findings

class PTaaSTestingModules:
    """Container for all PTaaS testing modules"""

    def __init__(self):
        self.agents = {
            "network_recon": NetworkReconnaissanceAgent(),
            "web_app": WebApplicationAgent()
        }
        self.active_scans = {}

    async def execute_test(self, agent_type: str, target: str, scan_options: Dict[str, Any] = None) -> ScanResult:
        """Execute a test using the specified agent"""
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        scan_options = scan_options or {}
        agent = self.agents[agent_type]

        scan_id = str(uuid.uuid4())
        self.active_scans[scan_id] = {
            "agent_type": agent_type,
            "target": target,
            "start_time": datetime.now(),
            "status": "running"
        }

        try:
            result = await agent.execute_scan(target, scan_options)
            self.active_scans[scan_id]["status"] = "completed"
            self.active_scans[scan_id]["result"] = result
            return result
        except Exception as e:
            self.active_scans[scan_id]["status"] = "failed"
            self.active_scans[scan_id]["error"] = str(e)
            raise

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        metrics = {}
        for agent_type, agent in self.agents.items():
            metrics[agent_type] = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "capabilities": agent.capabilities,
                "status": agent.status,
                "metrics": agent.performance_metrics
            }
        return metrics

    def get_active_scans(self) -> Dict[str, Any]:
        """Get information about active scans"""
        return self.active_scans

# Example usage and testing
if __name__ == "__main__":
    async def test_modules():
        """Test the PTaaS testing modules"""
        modules = PTaaSTestingModules()

        # Test network reconnaissance
        print("Testing Network Reconnaissance Agent...")
        network_result = await modules.execute_test(
            "network_recon",
            "example.com",
            {"quick_scan": True}
        )
        print(f"Network scan completed: {len(network_result.findings)} findings")

        # Test web application scanning
        print("\nTesting Web Application Agent...")
        web_result = await modules.execute_test(
            "web_app",
            "https://httpbin.org",
            {"aggressive_scan": False}
        )
        print(f"Web scan completed: {len(web_result.findings)} findings")

        # Display metrics
        print("\nAgent Metrics:")
        metrics = modules.get_agent_metrics()
        for agent_type, agent_metrics in metrics.items():
            print(f"{agent_type}: {agent_metrics['metrics']}")

    # Run tests
    asyncio.run(test_modules())
