"""
Production Security Scanner Integration
Real-world integration with industry-standard security tools
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re
import shutil
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class ScannerResult:
    """Standardized scanner result"""
    scanner_name: str
    target: str
    vulnerabilities: List[Dict[str, Any]]
    services: List[Dict[str, Any]]
    scan_duration: float
    scan_timestamp: datetime
    raw_output: str
    exit_code: int

@dataclass
class ScanConfiguration:
    """Scanner configuration"""
    target: str
    scan_type: str
    stealth_mode: bool
    timeout_seconds: int
    output_format: str
    custom_args: List[str]

class SecurityScannerService:
    """Production-ready security scanner integration service"""
    
    def __init__(self):
        self.available_scanners = {}
        self.scan_profiles = {
            'quick': {
                'timeout': 300,
                'stealth': False,
                'tools': ['nmap_quick', 'nuclei_high']
            },
            'comprehensive': {
                'timeout': 1800,
                'stealth': False,
                'tools': ['nmap_comprehensive', 'nuclei_all', 'nikto', 'sslscan']
            },
            'stealth': {
                'timeout': 3600,
                'stealth': True,
                'tools': ['nmap_stealth', 'nuclei_stealth']
            }
        }
        
        # Initialize available scanners
        self._detect_available_scanners()
    
    def _detect_available_scanners(self):
        """Detect which security scanners are available on the system"""
        scanners_to_check = {
            'nmap': ['nmap', '--version'],
            'nuclei': ['nuclei', '-version'],
            'nikto': ['nikto', '-Version'],
            'sslscan': ['sslscan', '--version'],
            'dirb': ['dirb'],
            'gobuster': ['gobuster', 'version']
        }
        
        for scanner_name, check_command in scanners_to_check.items():
            try:
                result = subprocess.run(
                    check_command,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = self._extract_version(scanner_name, result.stdout)
                    self.available_scanners[scanner_name] = {
                        'available': True,
                        'version': version,
                        'path': shutil.which(scanner_name)
                    }
                    logger.info(f"Detected {scanner_name} v{version}")
                else:
                    self.available_scanners[scanner_name] = {'available': False}
            except Exception as e:
                logger.warning(f"Scanner {scanner_name} not available: {e}")
                self.available_scanners[scanner_name] = {'available': False}
    
    def _extract_version(self, scanner_name: str, output: str) -> str:
        """Extract version from scanner output"""
        version_patterns = {
            'nmap': r'Nmap version (\d+\.\d+)',
            'nuclei': r'v(\d+\.\d+\.\d+)',
            'nikto': r'Nikto ver\. (\d+\.\d+\.\d+)',
            'sslscan': r'sslscan version (\d+\.\d+)',
        }
        
        pattern = version_patterns.get(scanner_name, r'(\d+\.\d+\.\d+)')
        match = re.search(pattern, output)
        return match.group(1) if match else 'unknown'
    
    async def execute_scan(self, targets: List[str], scan_type: str = 'comprehensive') -> Dict[str, Any]:
        """Execute comprehensive security scan"""
        try:
            profile = self.scan_profiles.get(scan_type, self.scan_profiles['comprehensive'])
            scan_results = []
            
            for target in targets:
                logger.info(f"Starting {scan_type} scan on {target}")
                
                # Execute scans based on profile
                target_results = []
                for tool_config in profile['tools']:
                    if await self._is_scanner_available(tool_config):
                        result = await self._execute_scanner(
                            tool_config, 
                            target,
                            profile['timeout'],
                            profile['stealth']
                        )
                        if result:
                            target_results.append(result)
                
                scan_results.extend(target_results)
            
            # Aggregate and correlate results
            aggregated_results = await self._aggregate_results(scan_results)
            
            return {
                'scan_type': scan_type,
                'targets': targets,
                'results': aggregated_results,
                'scanners_used': [r.scanner_name for r in scan_results],
                'total_vulnerabilities': sum(len(r.vulnerabilities) for r in scan_results),
                'scan_duration': sum(r.scan_duration for r in scan_results),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scan execution failed: {e}")
            raise
    
    async def _is_scanner_available(self, tool_config: str) -> bool:
        """Check if scanner is available"""
        scanner_name = tool_config.split('_')[0]
        return self.available_scanners.get(scanner_name, {}).get('available', False)
    
    async def _execute_scanner(
        self, 
        tool_config: str, 
        target: str,
        timeout: int,
        stealth: bool
    ) -> Optional[ScannerResult]:
        """Execute individual scanner"""
        try:
            scanner_name = tool_config.split('_')[0]
            scan_variant = tool_config.split('_', 1)[1] if '_' in tool_config else 'default'
            
            # Build scanner command
            command = await self._build_scanner_command(
                scanner_name, 
                scan_variant, 
                target, 
                stealth
            )
            
            if not command:
                logger.warning(f"Could not build command for {tool_config}")
                return None
            
            # Execute scanner
            start_time = datetime.utcnow()
            result = await self._run_scanner_command(command, timeout)
            end_time = datetime.utcnow()
            
            # Parse results
            vulnerabilities, services = await self._parse_scanner_output(
                scanner_name, 
                result['stdout']
            )
            
            return ScannerResult(
                scanner_name=scanner_name,
                target=target,
                vulnerabilities=vulnerabilities,
                services=services,
                scan_duration=(end_time - start_time).total_seconds(),
                scan_timestamp=start_time,
                raw_output=result['stdout'],
                exit_code=result['returncode']
            )
            
        except Exception as e:
            logger.error(f"Scanner execution failed for {tool_config}: {e}")
            return None
    
    async def _build_scanner_command(
        self, 
        scanner_name: str, 
        variant: str, 
        target: str,
        stealth: bool
    ) -> Optional[List[str]]:
        """Build scanner command based on configuration"""
        scanner_path = self.available_scanners[scanner_name]['path']
        
        if scanner_name == 'nmap':
            return await self._build_nmap_command(scanner_path, variant, target, stealth)
        elif scanner_name == 'nuclei':
            return await self._build_nuclei_command(scanner_path, variant, target, stealth)
        elif scanner_name == 'nikto':
            return await self._build_nikto_command(scanner_path, target, stealth)
        elif scanner_name == 'sslscan':
            return await self._build_sslscan_command(scanner_path, target)
        elif scanner_name == 'dirb':
            return await self._build_dirb_command(scanner_path, target, stealth)
        
        return None
    
    async def _build_nmap_command(
        self, 
        path: str, 
        variant: str, 
        target: str, 
        stealth: bool
    ) -> List[str]:
        """Build Nmap command with production-ready options"""
        base_cmd = [path]
        
        if variant == 'quick':
            base_cmd.extend(['-sS', '-T4', '--top-ports', '1000'])
        elif variant == 'comprehensive':
            base_cmd.extend([
                '-sS', '-sV', '-O', '-A',
                '--script=default,vuln,exploit',
                '--version-intensity=9',
                '-T4'
            ])
        elif variant == 'stealth':
            base_cmd.extend([
                '-sS', '-T2', '--scan-delay', '1s',
                '--max-retries', '1', '-f'
            ])
        
        if stealth:
            base_cmd.extend(['-f', '--scan-delay', '2s', '-T2'])
        
        # Output in XML format for better parsing
        base_cmd.extend(['-oX', '-', target])
        
        return base_cmd
    
    async def _build_nuclei_command(
        self, 
        path: str, 
        variant: str, 
        target: str, 
        stealth: bool
    ) -> List[str]:
        """Build Nuclei command with production templates"""
        base_cmd = [path, '-u', target, '-json']
        
        if variant == 'high':
            base_cmd.extend(['-severity', 'critical,high'])
        elif variant == 'all':
            base_cmd.extend(['-severity', 'critical,high,medium,low'])
        elif variant == 'stealth':
            base_cmd.extend(['-severity', 'critical,high', '-rate-limit', '10'])
        
        if stealth:
            base_cmd.extend(['-rate-limit', '5', '-timeout', '30'])
        else:
            base_cmd.extend(['-rate-limit', '150', '-timeout', '15'])
        
        # Add common template categories
        base_cmd.extend(['-t', 'cves/', '-t', 'vulnerabilities/', '-t', 'exposures/'])
        
        return base_cmd
    
    async def _build_nikto_command(self, path: str, target: str, stealth: bool) -> List[str]:
        """Build Nikto command for web vulnerability scanning"""
        base_cmd = [path, '-h', target, '-Format', 'json']
        
        if stealth:
            base_cmd.extend(['-Tuning', 'x', '-Pause', '2'])
        else:
            base_cmd.extend(['-Tuning', '123456789ab'])
        
        return base_cmd
    
    async def _build_sslscan_command(self, path: str, target: str) -> List[str]:
        """Build SSLScan command for SSL/TLS analysis"""
        # Extract hostname and port if specified
        if ':' in target:
            host, port = target.split(':', 1)
        else:
            host, port = target, '443'
        
        return [path, '--xml=-', f"{host}:{port}"]
    
    async def _build_dirb_command(self, path: str, target: str, stealth: bool) -> List[str]:
        """Build DIRB command for directory enumeration"""
        base_cmd = [path, f"http://{target}/", '-o', '-']
        
        if stealth:
            base_cmd.extend(['-z', '2000'])  # 2 second delay
        
        return base_cmd
    
    async def _run_scanner_command(self, command: List[str], timeout: int) -> Dict[str, Any]:
        """Run scanner command with proper error handling"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore'),
                'returncode': process.returncode
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Scanner command timed out after {timeout} seconds")
            if process:
                process.kill()
            raise
        except Exception as e:
            logger.error(f"Scanner command execution failed: {e}")
            raise
    
    async def _parse_scanner_output(
        self, 
        scanner_name: str, 
        output: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse scanner output into structured data"""
        try:
            if scanner_name == 'nmap':
                return await self._parse_nmap_output(output)
            elif scanner_name == 'nuclei':
                return await self._parse_nuclei_output(output)
            elif scanner_name == 'nikto':
                return await self._parse_nikto_output(output)
            elif scanner_name == 'sslscan':
                return await self._parse_sslscan_output(output)
            elif scanner_name == 'dirb':
                return await self._parse_dirb_output(output)
            
            return [], []
            
        except Exception as e:
            logger.error(f"Failed to parse {scanner_name} output: {e}")
            return [], []
    
    async def _parse_nmap_output(self, xml_output: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse Nmap XML output"""
        vulnerabilities = []
        services = []
        
        try:
            root = ET.fromstring(xml_output)
            
            for host in root.findall('host'):
                host_ip = host.find('address').get('addr')
                
                # Parse ports and services
                for port in host.findall('.//port'):
                    port_id = port.get('portid')
                    protocol = port.get('protocol')
                    
                    state_elem = port.find('state')
                    state = state_elem.get('state') if state_elem is not None else 'unknown'
                    
                    service_elem = port.find('service')
                    if service_elem is not None:
                        service_info = {
                            'host': host_ip,
                            'port': int(port_id),
                            'protocol': protocol,
                            'state': state,
                            'service': service_elem.get('name', 'unknown'),
                            'version': service_elem.get('version', ''),
                            'product': service_elem.get('product', ''),
                        }
                        services.append(service_info)
                
                # Parse script results for vulnerabilities
                for script in host.findall('.//script'):
                    script_id = script.get('id')
                    if 'vuln' in script_id or 'exploit' in script_id:
                        vulnerability = {
                            'host': host_ip,
                            'type': 'nmap_script',
                            'script_id': script_id,
                            'output': script.get('output', ''),
                            'severity': self._determine_nmap_severity(script_id),
                            'source': 'nmap'
                        }
                        vulnerabilities.append(vulnerability)
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse Nmap XML: {e}")
        
        return vulnerabilities, services
    
    async def _parse_nuclei_output(self, json_output: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse Nuclei JSON output"""
        vulnerabilities = []
        
        for line in json_output.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                vulnerability = {
                    'host': result.get('host', ''),
                    'template_id': result.get('template-id', ''),
                    'name': result.get('info', {}).get('name', ''),
                    'severity': result.get('info', {}).get('severity', 'info'),
                    'description': result.get('info', {}).get('description', ''),
                    'reference': result.get('info', {}).get('reference', []),
                    'matcher_name': result.get('matcher-name', ''),
                    'url': result.get('url', ''),
                    'source': 'nuclei'
                }
                vulnerabilities.append(vulnerability)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Nuclei JSON line: {e}")
        
        return vulnerabilities, []
    
    async def _parse_nikto_output(self, json_output: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse Nikto JSON output"""
        vulnerabilities = []
        
        try:
            # Nikto output format can vary, handle both single object and array
            if json_output.strip().startswith('['):
                results = json.loads(json_output)
            else:
                results = [json.loads(json_output)]
            
            for result in results:
                vulnerabilities_data = result.get('vulnerabilities', [])
                for vuln in vulnerabilities_data:
                    vulnerability = {
                        'host': result.get('host', ''),
                        'port': result.get('port', 80),
                        'id': vuln.get('id', ''),
                        'method': vuln.get('method', ''),
                        'uri': vuln.get('uri', ''),
                        'message': vuln.get('msg', ''),
                        'severity': 'medium',  # Nikto doesn't provide severity
                        'source': 'nikto'
                    }
                    vulnerabilities.append(vulnerability)
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Nikto JSON: {e}")
        
        return vulnerabilities, []
    
    async def _parse_sslscan_output(self, xml_output: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse SSLScan XML output"""
        vulnerabilities = []
        services = []
        
        try:
            root = ET.fromstring(xml_output)
            
            # Parse SSL/TLS configuration
            for target in root.findall('target'):
                host = target.get('host')
                port = target.get('port')
                
                # Check for weak ciphers
                for cipher in target.findall('.//cipher'):
                    cipher_name = cipher.get('cipher')
                    strength = cipher.get('bits')
                    
                    if cipher_name and ('NULL' in cipher_name or 'RC4' in cipher_name or 'DES' in cipher_name):
                        vulnerability = {
                            'host': host,
                            'port': int(port) if port else 443,
                            'type': 'weak_cipher',
                            'cipher': cipher_name,
                            'strength': strength,
                            'severity': 'high' if 'NULL' in cipher_name else 'medium',
                            'source': 'sslscan'
                        }
                        vulnerabilities.append(vulnerability)
                
                # Check protocol versions
                for protocol in target.findall('.//protocol'):
                    version = protocol.get('version')
                    enabled = protocol.get('enabled')
                    
                    if enabled == '1' and version in ['SSLv2', 'SSLv3', 'TLSv1']:
                        vulnerability = {
                            'host': host,
                            'port': int(port) if port else 443,
                            'type': 'weak_protocol',
                            'protocol': version,
                            'severity': 'high' if version in ['SSLv2', 'SSLv3'] else 'medium',
                            'source': 'sslscan'
                        }
                        vulnerabilities.append(vulnerability)
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse SSLScan XML: {e}")
        
        return vulnerabilities, services
    
    async def _parse_dirb_output(self, output: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse DIRB output"""
        vulnerabilities = []
        
        # DIRB outputs discovered directories/files
        lines = output.split('\n')
        for line in lines:
            if line.startswith('+ '):
                # Extract URL from DIRB output
                match = re.search(r'\+ (https?://[^\s]+)', line)
                if match:
                    url = match.group(1)
                    vulnerability = {
                        'type': 'directory_listing',
                        'url': url,
                        'description': 'Directory or file discovered',
                        'severity': 'info',
                        'source': 'dirb'
                    }
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities, []
    
    def _determine_nmap_severity(self, script_id: str) -> str:
        """Determine severity based on Nmap script ID"""
        high_severity_scripts = [
            'ftp-anon', 'http-sql-injection', 'ssh-hostkey',
            'ssl-poodle', 'ssl-heartbleed', 'smb-vuln-ms17-010'
        ]
        
        if any(high_script in script_id for high_script in high_severity_scripts):
            return 'high'
        elif 'vuln' in script_id:
            return 'medium'
        else:
            return 'low'
    
    async def _aggregate_results(self, scan_results: List[ScannerResult]) -> Dict[str, Any]:
        """Aggregate and correlate scan results"""
        all_vulnerabilities = []
        all_services = []
        
        for result in scan_results:
            all_vulnerabilities.extend(result.vulnerabilities)
            all_services.extend(result.services)
        
        # Group vulnerabilities by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        for vuln in all_vulnerabilities:
            severity = vuln.get('severity', 'info')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Identify unique services
        unique_services = {}
        for service in all_services:
            key = f"{service['host']}:{service['port']}"
            if key not in unique_services:
                unique_services[key] = service
        
        return {
            'vulnerabilities': all_vulnerabilities,
            'services': list(unique_services.values()),
            'vulnerability_summary': severity_counts,
            'total_vulnerabilities': len(all_vulnerabilities),
            'total_services': len(unique_services),
            'high_risk_findings': [v for v in all_vulnerabilities if v.get('severity') in ['critical', 'high']],
            'recommendations': self._generate_recommendations(all_vulnerabilities, list(unique_services.values()))
        }
    
    def _generate_recommendations(
        self, 
        vulnerabilities: List[Dict[str, Any]], 
        services: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on findings"""
        recommendations = []
        
        # Check for critical vulnerabilities
        critical_vulns = [v for v in vulnerabilities if v.get('severity') == 'critical']
        if critical_vulns:
            recommendations.append("CRITICAL: Immediately patch critical vulnerabilities identified")
        
        # Check for weak SSL/TLS
        weak_ssl = [v for v in vulnerabilities if v.get('type') in ['weak_cipher', 'weak_protocol']]
        if weak_ssl:
            recommendations.append("Update SSL/TLS configuration to disable weak ciphers and protocols")
        
        # Check for exposed services
        risky_services = [s for s in services if s.get('service') in ['ftp', 'telnet', 'rsh', 'rlogin']]
        if risky_services:
            recommendations.append("Disable or secure legacy network services (FTP, Telnet, etc.)")
        
        # Check for SQL injection
        sqli_vulns = [v for v in vulnerabilities if 'sql' in v.get('template_id', '').lower()]
        if sqli_vulns:
            recommendations.append("Implement parameterized queries to prevent SQL injection")
        
        # General recommendations
        recommendations.extend([
            "Implement network segmentation and least privilege access",
            "Enable logging and monitoring for security events",
            "Conduct regular vulnerability assessments",
            "Maintain an incident response plan"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def get_scanner_status(self) -> Dict[str, Any]:
        """Get status of all available scanners"""
        return {
            'available_scanners': self.available_scanners,
            'scan_profiles': list(self.scan_profiles.keys()),
            'total_scanners': len([s for s in self.available_scanners.values() if s.get('available')])
        }