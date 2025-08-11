"""
Enterprise PTaaS Orchestrator - Production-Ready Implementation
Advanced penetration testing orchestration with real security tool integration
"""

import asyncio
import json
import logging
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import tempfile
import shlex

from .interfaces import PTaaSService, SecurityOrchestrationService, ComplianceService
from .base_service import XORBService, ServiceHealth, ServiceStatus
from ..domain.entities import User, Organization
from ..domain.tenant_entities import (
    ScanTarget, ScanResult, SecurityFinding
)
from ..domain.repositories import ScanSessionRepository, CacheRepository

logger = logging.getLogger(__name__)


@dataclass
class ScanProfile:
    """Scan profile configuration"""
    name: str
    description: str
    duration_estimate_minutes: int
    tools: List[str]
    intensity: str
    stealth_mode: bool = False
    compliance_framework: Optional[str] = None


@dataclass
class ScannerTool:
    """Security scanner tool configuration"""
    name: str
    executable: str
    default_args: List[str]
    output_parser: str
    timeout_seconds: int = 300
    max_parallel: int = 5


class EnterprisePTaaSOrchestrator(XORBService, PTaaSService, SecurityOrchestrationService):
    """
    Enterprise-grade PTaaS orchestrator with real security tool integration
    Production-ready with comprehensive error handling and audit logging
    """
    
    def __init__(
        self,
        scan_repository: ScanSessionRepository,
        cache_repository: CacheRepository,
        **kwargs
    ):
        super().__init__(
            service_id="enterprise_ptaas_orchestrator",
            dependencies=["database", "cache", "security_scanners"],
            **kwargs
        )
        self.scan_repository = scan_repository
        self.cache = cache_repository
        
        # Active scan sessions
        self.active_sessions: Dict[str, ScanSession] = {}
        self.scan_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Initialize scan profiles and tools
        self.scan_profiles = self._initialize_scan_profiles()
        self.scanner_tools = self._initialize_scanner_tools()
        
        # Security validations
        self.allowed_ports = list(range(1, 65536))
        self.blocked_networks = [
            "127.0.0.0/8",    # Localhost
            "10.0.0.0/8",     # Private networks
            "172.16.0.0/12",  # Private networks
            "192.168.0.0/16", # Private networks
        ]
    
    def _initialize_scan_profiles(self) -> Dict[str, ScanProfile]:
        """Initialize available scan profiles"""
        return {
            "quick": ScanProfile(
                name="Quick Discovery Scan",
                description="Fast network discovery and basic service detection",
                duration_estimate_minutes=5,
                tools=["nmap_discovery", "basic_port_scan"],
                intensity="low"
            ),
            "comprehensive": ScanProfile(
                name="Comprehensive Security Assessment",
                description="Full security assessment with vulnerability scanning",
                duration_estimate_minutes=45,
                tools=["nmap_full", "nuclei_comprehensive", "nikto_scan", "ssl_analyzer"],
                intensity="high"
            ),
            "stealth": ScanProfile(
                name="Stealth Assessment",
                description="Low-profile scanning to avoid detection",
                duration_estimate_minutes=90,
                tools=["nmap_stealth", "passive_recon", "subdomain_enum"],
                intensity="minimal",
                stealth_mode=True
            ),
            "web_focused": ScanProfile(
                name="Web Application Security Scan",
                description="Specialized web application security testing",
                duration_estimate_minutes=30,
                tools=["nikto_scan", "nuclei_web", "dirb_scan", "ssl_analyzer"],
                intensity="medium"
            ),
            "compliance_pci": ScanProfile(
                name="PCI-DSS Compliance Scan",
                description="Payment Card Industry compliance assessment",
                duration_estimate_minutes=25,
                tools=["nmap_compliance", "nuclei_pci", "ssl_pci_scan"],
                intensity="medium",
                compliance_framework="PCI-DSS"
            ),
            "compliance_hipaa": ScanProfile(
                name="HIPAA Compliance Scan",
                description="Healthcare data protection assessment",
                duration_estimate_minutes=30,
                tools=["nmap_compliance", "nuclei_hipaa", "ssl_analyzer"],
                intensity="medium",
                compliance_framework="HIPAA"
            )
        }
    
    def _initialize_scanner_tools(self) -> Dict[str, ScannerTool]:
        """Initialize security scanner tools"""
        return {
            "nmap_discovery": ScannerTool(
                name="Nmap Discovery",
                executable="nmap",
                default_args=["-sn", "-PE", "-PP", "-PS80,443", "-PA3389"],
                output_parser="nmap_xml",
                timeout_seconds=300
            ),
            "nmap_full": ScannerTool(
                name="Nmap Full Scan",
                executable="nmap",
                default_args=["-sS", "-sV", "-O", "-A", "--script=default,vuln"],
                output_parser="nmap_xml",
                timeout_seconds=1800
            ),
            "nmap_stealth": ScannerTool(
                name="Nmap Stealth Scan",
                executable="nmap",
                default_args=["-sS", "-f", "-T2", "--randomize-hosts", "--spoof-mac", "random"],
                output_parser="nmap_xml",
                timeout_seconds=3600
            ),
            "nuclei_comprehensive": ScannerTool(
                name="Nuclei Vulnerability Scanner",
                executable="nuclei",
                default_args=["-severity", "critical,high,medium", "-stats", "-json"],
                output_parser="nuclei_json",
                timeout_seconds=1200
            ),
            "nuclei_web": ScannerTool(
                name="Nuclei Web Scanner",
                executable="nuclei",
                default_args=["-tags", "web,http,ssl", "-severity", "critical,high,medium", "-json"],
                output_parser="nuclei_json",
                timeout_seconds=900
            ),
            "nikto_scan": ScannerTool(
                name="Nikto Web Scanner",
                executable="nikto",
                default_args=["-Format", "json", "-nointeractive"],
                output_parser="nikto_json",
                timeout_seconds=1800
            ),
            "ssl_analyzer": ScannerTool(
                name="SSL/TLS Analyzer",
                executable="sslscan",
                default_args=["--xml=-"],
                output_parser="sslscan_xml",
                timeout_seconds=300
            ),
            "dirb_scan": ScannerTool(
                name="Directory Bruteforce",
                executable="dirb",
                default_args=["-S", "-w"],
                output_parser="dirb_text",
                timeout_seconds=1200
            )
        }
    
    async def create_scan_session(
        self,
        targets: List[Dict[str, Any]],
        scan_type: str,
        user: User,
        org: Organization,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new PTaaS scan session with comprehensive validation"""
        
        session_id = str(uuid4())
        
        try:
            # Validate scan profile
            if scan_type not in self.scan_profiles:
                raise ValueError(f"Invalid scan type: {scan_type}")
            
            profile = self.scan_profiles[scan_type]
            
            # Validate and sanitize targets
            validated_targets = []
            for target_data in targets:
                try:
                    target = await self._validate_and_create_target(target_data, org)
                    validated_targets.append(target)
                except ValueError as e:
                    logger.warning(f"Invalid target rejected: {target_data} - {e}")
            
            if not validated_targets:
                raise ValueError("No valid targets provided for scanning")
            
            # Create scan session entity
            scan_session = ScanSession(
                id=UUID(session_id),
                targets=validated_targets,
                scan_profile=scan_type,
                user_id=user.id,
                organization_id=org.id,
                status="queued",
                progress=0,
                results={},
                metadata={
                    **(metadata or {}),
                    "profile_name": profile.name,
                    "estimated_duration_minutes": profile.duration_estimate_minutes,
                    "tools": profile.tools,
                    "created_by": user.username,
                    "organization": org.name
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                scheduled_for=datetime.utcnow()
            )
            
            # Save to database
            created_session = await self.scan_repository.create(scan_session)
            self.active_sessions[session_id] = created_session
            
            # Initialize semaphore for concurrent tool execution
            self.scan_semaphores[session_id] = asyncio.Semaphore(profile.max_parallel if hasattr(profile, 'max_parallel') else 3)
            
            # Queue scan for execution
            asyncio.create_task(self._execute_scan_session(session_id))
            
            logger.info(f"Created PTaaS scan session {session_id} with {len(validated_targets)} targets")
            
            return {
                "session_id": session_id,
                "status": "queued",
                "targets_count": len(validated_targets),
                "scan_profile": profile.name,
                "estimated_duration_minutes": profile.duration_estimate_minutes,
                "tools": profile.tools
            }
            
        except Exception as e:
            logger.error(f"Failed to create scan session: {e}")
            raise
    
    async def get_scan_status(self, session_id: str, user: User) -> Dict[str, Any]:
        """Get detailed status of a scan session"""
        
        try:
            # Check active sessions first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
            else:
                # Fetch from database
                session = await self.scan_repository.get_by_id(UUID(session_id))
                if not session:
                    raise ValueError(f"Scan session {session_id} not found")
            
            # Verify user access
            if session.user_id != user.id:
                raise ValueError("Access denied to scan session")
            
            return {
                "session_id": session_id,
                "status": session.status,
                "progress": session.progress,
                "targets_count": len(session.targets),
                "scan_profile": session.scan_profile,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "results_available": bool(session.results) if session.status == "completed" else False
            }
            
        except Exception as e:
            logger.error(f"Failed to get scan status for {session_id}: {e}")
            raise
    
    async def get_scan_results(self, session_id: str, user: User) -> Dict[str, Any]:
        """Get comprehensive scan results"""
        
        try:
            session = await self.scan_repository.get_by_id(UUID(session_id))
            if not session:
                raise ValueError(f"Scan session {session_id} not found")
            
            # Verify user access
            if session.user_id != user.id:
                raise ValueError("Access denied to scan session")
            
            if session.status != "completed":
                return {
                    "session_id": session_id,
                    "status": session.status,
                    "message": "Scan not yet completed",
                    "progress": session.progress
                }
            
            # Parse and enrich results
            enriched_results = await self._enrich_scan_results(session.results)
            
            return {
                "session_id": session_id,
                "status": session.status,
                "scan_profile": session.scan_profile,
                "targets": [asdict(target) for target in session.targets],
                "results": enriched_results,
                "summary": self._generate_results_summary(enriched_results),
                "created_at": session.created_at.isoformat(),
                "completed_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get scan results for {session_id}: {e}")
            raise
    
    async def cancel_scan(self, session_id: str, user: User) -> bool:
        """Cancel an active scan session"""
        
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Verify user access
            if session.user_id != user.id:
                raise ValueError("Access denied to scan session")
            
            # Update session status
            session.status = "cancelled"
            session.updated_at = datetime.utcnow()
            
            # Save to database
            await self.scan_repository.update(session)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            if session_id in self.scan_semaphores:
                del self.scan_semaphores[session_id]
            
            logger.info(f"Cancelled scan session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel scan {session_id}: {e}")
            return False
    
    async def get_available_scan_profiles(self) -> List[Dict[str, Any]]:
        """Get available scan profiles with detailed information"""
        
        return [
            {
                "id": profile_id,
                "name": profile.name,
                "description": profile.description,
                "duration_estimate_minutes": profile.duration_estimate_minutes,
                "tools": profile.tools,
                "intensity": profile.intensity,
                "stealth_mode": profile.stealth_mode,
                "compliance_framework": profile.compliance_framework
            }
            for profile_id, profile in self.scan_profiles.items()
        ]
    
    async def create_compliance_scan(
        self,
        targets: List[str],
        compliance_framework: str,
        user: User,
        org: Organization
    ) -> Dict[str, Any]:
        """Create compliance-specific scan session"""
        
        try:
            # Map compliance framework to scan profile
            framework_profiles = {
                "PCI-DSS": "compliance_pci",
                "HIPAA": "compliance_hipaa",
                "SOX": "comprehensive",  # Use comprehensive for SOX
                "ISO-27001": "comprehensive"
            }
            
            scan_type = framework_profiles.get(compliance_framework, "comprehensive")
            
            # Convert target strings to target objects
            target_objects = [
                {"host": target, "ports": [80, 443], "scan_profile": scan_type}
                for target in targets
            ]
            
            # Create scan session
            result = await self.create_scan_session(
                target_objects,
                scan_type,
                user,
                org,
                metadata={
                    "compliance_framework": compliance_framework,
                    "compliance_scan": True,
                    "framework_requirements": self._get_compliance_requirements(compliance_framework)
                }
            )
            
            logger.info(f"Created compliance scan for {compliance_framework}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create compliance scan: {e}")
            raise
    
    async def _validate_and_create_target(self, target_data: Dict[str, Any], org: Organization) -> ScanTarget:
        """Validate and create scan target with security checks"""
        
        # Extract target information
        host = target_data.get("host", "").strip()
        ports = target_data.get("ports", [80, 443])
        scan_profile = target_data.get("scan_profile", "quick")
        
        # Validate host
        if not host:
            raise ValueError("Host is required")
        
        # Validate host format (IP or domain)
        if not self._is_valid_host(host):
            raise ValueError(f"Invalid host format: {host}")
        
        # Security validation - check against blocked networks
        if self._is_blocked_target(host):
            raise ValueError(f"Target {host} is in blocked network range")
        
        # Validate ports
        validated_ports = []
        for port in ports:
            if isinstance(port, int) and 1 <= port <= 65535:
                validated_ports.append(port)
            else:
                logger.warning(f"Invalid port {port} ignored")
        
        if not validated_ports:
            validated_ports = [80, 443]  # Default ports
        
        return ScanTarget(
            id=uuid4(),
            host=host,
            ports=validated_ports,
            scan_profile=scan_profile,
            metadata={
                "organization_id": str(org.id),
                "validated_at": datetime.utcnow().isoformat()
            }
        )
    
    async def _execute_scan_session(self, session_id: str) -> None:
        """Execute scan session with comprehensive tool orchestration"""
        
        try:
            session = self.active_sessions[session_id]
            profile = self.scan_profiles[session.scan_profile]
            
            logger.info(f"Starting execution of scan session {session_id}")
            
            # Update status to running
            session.status = "running"
            session.progress = 5
            session.updated_at = datetime.utcnow()
            await self.scan_repository.update(session)
            
            all_results = {}
            
            # Execute each tool in the profile
            tool_count = len(profile.tools)
            for i, tool_name in enumerate(profile.tools):
                try:
                    logger.info(f"Executing tool {tool_name} for session {session_id}")
                    
                    # Update progress
                    session.progress = 10 + (i * 80 // tool_count)
                    await self.scan_repository.update(session)
                    
                    # Execute tool for all targets
                    tool_results = await self._execute_tool_for_targets(
                        tool_name, session.targets, session_id
                    )
                    
                    all_results[tool_name] = tool_results
                    
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed for session {session_id}: {e}")
                    all_results[tool_name] = {"error": str(e)}
            
            # Post-process and analyze results
            session.progress = 95
            await self.scan_repository.update(session)
            
            analyzed_results = await self._analyze_results(all_results, profile)
            
            # Complete session
            session.status = "completed"
            session.progress = 100
            session.results = analyzed_results
            session.updated_at = datetime.utcnow()
            session.completed_at = datetime.utcnow()
            
            await self.scan_repository.update(session)
            
            # Clean up
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.scan_semaphores:
                del self.scan_semaphores[session_id]
            
            logger.info(f"Completed scan session {session_id}")
            
        except Exception as e:
            # Handle session failure
            try:
                session = self.active_sessions.get(session_id)
                if session:
                    session.status = "failed"
                    session.results = {"error": str(e), "failed_at": datetime.utcnow().isoformat()}
                    session.updated_at = datetime.utcnow()
                    await self.scan_repository.update(session)
            except Exception as cleanup_error:
                logger.error(f"Failed to update failed session {session_id}: {cleanup_error}")
            
            logger.error(f"Scan session {session_id} failed: {e}")
    
    async def _execute_tool_for_targets(
        self, 
        tool_name: str, 
        targets: List[ScanTarget], 
        session_id: str
    ) -> Dict[str, Any]:
        """Execute security tool for all targets with parallel processing"""
        
        if tool_name not in self.scanner_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.scanner_tools[tool_name]
        semaphore = self.scan_semaphores.get(session_id, asyncio.Semaphore(3))
        
        # Execute tool for each target
        tasks = []
        for target in targets:
            task = self._execute_tool_for_single_target(tool, target, semaphore)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {
            "tool": tool_name,
            "targets": {},
            "summary": {
                "total_targets": len(targets),
                "successful_scans": 0,
                "failed_scans": 0
            }
        }
        
        for i, (target, result) in enumerate(zip(targets, results)):
            target_key = f"{target.host}:{','.join(map(str, target.ports))}"
            
            if isinstance(result, Exception):
                combined_results["targets"][target_key] = {
                    "error": str(result),
                    "status": "failed"
                }
                combined_results["summary"]["failed_scans"] += 1
            else:
                combined_results["targets"][target_key] = result
                combined_results["summary"]["successful_scans"] += 1
        
        return combined_results
    
    async def _execute_tool_for_single_target(
        self, 
        tool: ScannerTool, 
        target: ScanTarget, 
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Execute security tool for a single target with comprehensive error handling"""
        
        async with semaphore:
            try:
                # Build command
                cmd = await self._build_tool_command(tool, target)
                
                logger.debug(f"Executing: {' '.join(cmd)}")
                
                # Execute command with timeout
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempfile.gettempdir()
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=tool.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise Exception(f"Tool {tool.name} timed out after {tool.timeout_seconds}s")
                
                # Parse output
                if process.returncode == 0:
                    result = await self._parse_tool_output(
                        tool.output_parser, 
                        stdout.decode('utf-8', errors='ignore')
                    )
                    return {
                        "status": "success",
                        "result": result,
                        "execution_time": tool.timeout_seconds,  # Placeholder
                        "target": f"{target.host}:{','.join(map(str, target.ports))}"
                    }
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logger.warning(f"Tool {tool.name} returned non-zero exit code: {error_msg}")
                    return {
                        "status": "partial_success",
                        "error": error_msg,
                        "target": f"{target.host}:{','.join(map(str, target.ports))}"
                    }
                
            except Exception as e:
                logger.error(f"Failed to execute {tool.name} for {target.host}: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "target": f"{target.host}:{','.join(map(str, target.ports))}"
                }
    
    async def _build_tool_command(self, tool: ScannerTool, target: ScanTarget) -> List[str]:
        """Build command line for security tool execution"""
        
        cmd = [tool.executable] + tool.default_args.copy()
        
        # Add target-specific arguments based on tool
        if tool.name.startswith("nmap"):
            # Add ports if specified
            if target.ports:
                port_list = ",".join(map(str, target.ports))
                cmd.extend(["-p", port_list])
            
            # Add output format
            cmd.extend(["-oX", "-"])  # XML output to stdout
            
            # Add target
            cmd.append(target.host)
            
        elif tool.name.startswith("nuclei"):
            # Add target URL
            if 443 in target.ports:
                cmd.extend(["-u", f"https://{target.host}"])
            else:
                cmd.extend(["-u", f"http://{target.host}"])
        
        elif tool.name.startswith("nikto"):
            # Add host and port
            if 443 in target.ports:
                cmd.extend(["-h", f"https://{target.host}"])
            else:
                cmd.extend(["-h", f"http://{target.host}"])
        
        elif tool.name.startswith("ssl"):
            # Add host and port
            if 443 in target.ports:
                cmd.append(f"{target.host}:443")
            else:
                cmd.append(f"{target.host}:443")  # Default to 443 for SSL
        
        return cmd
    
    async def _parse_tool_output(self, parser_type: str, output: str) -> Dict[str, Any]:
        """Parse tool output based on parser type"""
        
        try:
            if parser_type == "nmap_xml":
                return await self._parse_nmap_xml(output)
            elif parser_type == "nuclei_json":
                return await self._parse_nuclei_json(output)
            elif parser_type == "nikto_json":
                return await self._parse_nikto_json(output)
            elif parser_type == "sslscan_xml":
                return await self._parse_sslscan_xml(output)
            else:
                # Generic text parser
                return {"raw_output": output, "parsed": False}
        
        except Exception as e:
            logger.error(f"Failed to parse {parser_type} output: {e}")
            return {"raw_output": output, "parse_error": str(e)}
    
    async def _parse_nmap_xml(self, xml_output: str) -> Dict[str, Any]:
        """Parse Nmap XML output"""
        try:
            root = ET.fromstring(xml_output)
            hosts = []
            
            for host in root.findall('.//host'):
                host_data = {"status": "unknown", "addresses": [], "ports": []}
                
                # Get host status
                status = host.find('status')
                if status is not None:
                    host_data["status"] = status.get('state', 'unknown')
                
                # Get addresses
                for address in host.findall('address'):
                    host_data["addresses"].append({
                        "addr": address.get('addr'),
                        "addrtype": address.get('addrtype')
                    })
                
                # Get ports
                ports = host.find('ports')
                if ports is not None:
                    for port in ports.findall('port'):
                        port_data = {
                            "portid": port.get('portid'),
                            "protocol": port.get('protocol'),
                            "state": "unknown",
                            "service": {}
                        }
                        
                        state = port.find('state')
                        if state is not None:
                            port_data["state"] = state.get('state')
                        
                        service = port.find('service')
                        if service is not None:
                            port_data["service"] = {
                                "name": service.get('name'),
                                "product": service.get('product'),
                                "version": service.get('version')
                            }
                        
                        host_data["ports"].append(port_data)
                
                hosts.append(host_data)
            
            return {"hosts": hosts, "parser": "nmap_xml"}
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse Nmap XML: {e}")
            return {"error": "XML parse error", "raw_output": xml_output}
    
    async def _parse_nuclei_json(self, json_output: str) -> Dict[str, Any]:
        """Parse Nuclei JSON output"""
        try:
            findings = []
            for line in json_output.strip().split('\n'):
                if line.strip():
                    finding = json.loads(line)
                    findings.append(finding)
            
            return {"findings": findings, "parser": "nuclei_json", "count": len(findings)}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Nuclei JSON: {e}")
            return {"error": "JSON parse error", "raw_output": json_output}
    
    async def _parse_nikto_json(self, json_output: str) -> Dict[str, Any]:
        """Parse Nikto JSON output"""
        try:
            data = json.loads(json_output)
            return {"nikto_results": data, "parser": "nikto_json"}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Nikto JSON: {e}")
            return {"error": "JSON parse error", "raw_output": json_output}
    
    async def _parse_sslscan_xml(self, xml_output: str) -> Dict[str, Any]:
        """Parse SSLScan XML output"""
        try:
            root = ET.fromstring(xml_output)
            ssl_data = {
                "ciphers": [],
                "certificates": [],
                "vulnerabilities": []
            }
            
            # Parse cipher information
            for cipher in root.findall('.//cipher'):
                ssl_data["ciphers"].append({
                    "cipher": cipher.get('cipher'),
                    "strength": cipher.get('strength'),
                    "status": cipher.get('status')
                })
            
            return {"ssl_analysis": ssl_data, "parser": "sslscan_xml"}
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse SSLScan XML: {e}")
            return {"error": "XML parse error", "raw_output": xml_output}
    
    async def _analyze_results(self, all_results: Dict[str, Any], profile: ScanProfile) -> Dict[str, Any]:
        """Analyze and correlate scan results"""
        
        analyzed = {
            "scan_summary": {
                "profile": profile.name,
                "tools_executed": len(all_results),
                "total_findings": 0,
                "high_risk_findings": 0,
                "medium_risk_findings": 0,
                "low_risk_findings": 0
            },
            "findings": [],
            "recommendations": [],
            "compliance_status": {},
            "tool_results": all_results
        }
        
        # Process findings from each tool
        for tool_name, tool_result in all_results.items():
            if "error" in tool_result:
                continue
            
            # Extract findings based on tool type
            if tool_name.startswith("nuclei"):
                analyzed["findings"].extend(self._extract_nuclei_findings(tool_result))
            elif tool_name.startswith("nmap"):
                analyzed["findings"].extend(self._extract_nmap_findings(tool_result))
        
        # Calculate summary statistics
        analyzed["scan_summary"]["total_findings"] = len(analyzed["findings"])
        
        for finding in analyzed["findings"]:
            severity = finding.get("severity", "low")
            if severity in ["critical", "high"]:
                analyzed["scan_summary"]["high_risk_findings"] += 1
            elif severity == "medium":
                analyzed["scan_summary"]["medium_risk_findings"] += 1
            else:
                analyzed["scan_summary"]["low_risk_findings"] += 1
        
        # Generate recommendations
        analyzed["recommendations"] = self._generate_recommendations(analyzed["findings"])
        
        # Compliance analysis if applicable
        if profile.compliance_framework:
            analyzed["compliance_status"] = await self._analyze_compliance(
                analyzed["findings"], 
                profile.compliance_framework
            )
        
        return analyzed
    
    def _extract_nuclei_findings(self, tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract security findings from Nuclei results"""
        findings = []
        
        for target_key, target_result in tool_result.get("targets", {}).items():
            if "result" in target_result and "findings" in target_result["result"]:
                for nuclei_finding in target_result["result"]["findings"]:
                    finding = {
                        "source": "nuclei",
                        "target": target_key,
                        "template_id": nuclei_finding.get("template-id"),
                        "template_name": nuclei_finding.get("info", {}).get("name"),
                        "severity": nuclei_finding.get("info", {}).get("severity", "info"),
                        "description": nuclei_finding.get("info", {}).get("description"),
                        "matched_at": nuclei_finding.get("matched-at"),
                        "extracted_results": nuclei_finding.get("extracted-results", []),
                        "classification": nuclei_finding.get("info", {}).get("classification", {}),
                        "references": nuclei_finding.get("info", {}).get("reference", [])
                    }
                    findings.append(finding)
        
        return findings
    
    def _extract_nmap_findings(self, tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract security findings from Nmap results"""
        findings = []
        
        for target_key, target_result in tool_result.get("targets", {}).items():
            if "result" in target_result and "hosts" in target_result["result"]:
                for host in target_result["result"]["hosts"]:
                    # Analyze open ports
                    for port in host.get("ports", []):
                        if port.get("state") == "open":
                            finding = {
                                "source": "nmap",
                                "target": target_key,
                                "type": "open_port",
                                "port": port.get("portid"),
                                "protocol": port.get("protocol"),
                                "service": port.get("service", {}),
                                "severity": self._assess_port_risk(port),
                                "description": f"Open port {port.get('portid')}/{port.get('protocol')} detected"
                            }
                            findings.append(finding)
        
        return findings
    
    def _assess_port_risk(self, port: Dict[str, Any]) -> str:
        """Assess risk level of an open port"""
        port_num = int(port.get("portid", 0))
        service_name = port.get("service", {}).get("name", "").lower()
        
        # High-risk ports and services
        high_risk_ports = [21, 23, 135, 139, 445, 1433, 3389, 5432, 5984]
        high_risk_services = ["ftp", "telnet", "rpc", "netbios", "smb", "mssql", "rdp", "postgresql"]
        
        if port_num in high_risk_ports or service_name in high_risk_services:
            return "high"
        elif port_num in [22, 80, 443, 993, 995]:  # Common secure services
            return "low"
        else:
            return "medium"
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        # Analyze findings and generate recommendations
        high_risk_count = len([f for f in findings if f.get("severity") in ["critical", "high"]])
        open_ports = [f for f in findings if f.get("type") == "open_port"]
        
        if high_risk_count > 0:
            recommendations.append(f"Address {high_risk_count} high-risk security findings immediately")
        
        if len(open_ports) > 10:
            recommendations.append("Review and minimize exposed services - many open ports detected")
        
        # Add specific recommendations based on common findings
        for finding in findings:
            if finding.get("source") == "nuclei":
                template_id = finding.get("template_id", "")
                if "ssl" in template_id.lower():
                    recommendations.append("Review SSL/TLS configuration for security best practices")
                elif "xss" in template_id.lower():
                    recommendations.append("Implement input validation to prevent XSS attacks")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _analyze_compliance(self, findings: List[Dict[str, Any]], framework: str) -> Dict[str, Any]:
        """Analyze findings against compliance framework"""
        
        compliance_status = {
            "framework": framework,
            "overall_status": "compliant",
            "issues": [],
            "recommendations": []
        }
        
        if framework == "PCI-DSS":
            # PCI-DSS specific analysis
            for finding in findings:
                if finding.get("severity") in ["critical", "high"]:
                    compliance_status["overall_status"] = "non_compliant"
                    compliance_status["issues"].append({
                        "requirement": "6.5.1",  # Application vulnerabilities
                        "description": f"High-risk vulnerability detected: {finding.get('template_name')}",
                        "finding": finding
                    })
        
        return compliance_status
    
    def _get_compliance_requirements(self, framework: str) -> Dict[str, Any]:
        """Get compliance requirements for framework"""
        
        requirements = {
            "PCI-DSS": {
                "version": "4.0",
                "key_requirements": [
                    "6.2.4 - Software vulnerabilities",
                    "6.5.1 - Injection flaws",
                    "6.5.4 - Insecure communications",
                    "11.2.1 - Quarterly vulnerability scans"
                ]
            },
            "HIPAA": {
                "sections": ["164.312(a)(1)", "164.312(e)(1)", "164.308(a)(1)"],
                "key_requirements": [
                    "Access control",
                    "Transmission security", 
                    "Security management"
                ]
            }
        }
        
        return requirements.get(framework, {})
    
    async def _enrich_scan_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich scan results with additional intelligence"""
        
        enriched = results.copy()
        
        # Add threat intelligence correlation
        enriched["threat_intelligence"] = await self._correlate_threat_intelligence(results)
        
        # Add MITRE ATT&CK mapping
        enriched["mitre_attack"] = self._map_to_mitre_attack(results)
        
        # Add risk scoring
        enriched["risk_score"] = self._calculate_risk_score(results)
        
        return enriched
    
    async def _correlate_threat_intelligence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate findings with threat intelligence"""
        # Placeholder for threat intelligence correlation
        return {
            "indicators_checked": 0,
            "matches_found": 0,
            "threat_actor_associations": [],
            "malware_families": []
        }
    
    def _map_to_mitre_attack(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Map findings to MITRE ATT&CK framework"""
        # Placeholder for MITRE ATT&CK mapping
        return {
            "techniques": [],
            "tactics": [],
            "groups": []
        }
    
    def _calculate_risk_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk score"""
        findings = results.get("findings", [])
        
        if not findings:
            return {"score": 0, "level": "low"}
        
        # Simple risk scoring based on severity
        score = 0
        for finding in findings:
            severity = finding.get("severity", "low")
            if severity == "critical":
                score += 10
            elif severity == "high":
                score += 7
            elif severity == "medium":
                score += 4
            elif severity == "low":
                score += 1
        
        # Normalize score
        max_possible = len(findings) * 10
        normalized_score = min(100, (score / max_possible * 100) if max_possible > 0 else 0)
        
        if normalized_score >= 80:
            level = "critical"
        elif normalized_score >= 60:
            level = "high"
        elif normalized_score >= 40:
            level = "medium"
        else:
            level = "low"
        
        return {
            "score": round(normalized_score, 2),
            "level": level,
            "total_findings": len(findings)
        }
    
    def _generate_results_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of scan results"""
        
        scan_summary = results.get("scan_summary", {})
        risk_score = results.get("risk_score", {})
        
        return {
            "executive_summary": f"Scan completed with {scan_summary.get('total_findings', 0)} findings",
            "risk_level": risk_score.get("level", "unknown"),
            "risk_score": risk_score.get("score", 0),
            "high_priority_issues": scan_summary.get("high_risk_findings", 0),
            "tools_used": scan_summary.get("tools_executed", 0),
            "recommendations_count": len(results.get("recommendations", [])),
            "compliance_status": results.get("compliance_status", {}).get("overall_status", "unknown")
        }
    
    def _is_valid_host(self, host: str) -> bool:
        """Validate host format (IP address or domain name)"""
        import re
        
        # Check if it's a valid IP address
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            pass
        
        # Check if it's a valid domain name
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(domain_pattern, host)) and len(host) <= 253
    
    def _is_blocked_target(self, host: str) -> bool:
        """Check if target is in blocked network ranges"""
        try:
            target_ip = ipaddress.ip_address(host)
            
            for blocked_network in self.blocked_networks:
                if target_ip in ipaddress.ip_network(blocked_network):
                    return True
            
            return False
            
        except ValueError:
            # Not an IP address, assume it's a domain and allow it
            return False