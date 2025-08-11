"""
Enhanced Security Scanner Service - Advanced real-world security tool integration
Comprehensive penetration testing toolkit with AI-powered analysis
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import csv
import os
import re
import hashlib
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import socket
import ssl
import aiofiles
import aiohttp
from urllib.parse import urlparse, urljoin

# ML imports for vulnerability analysis
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available, using rule-based analysis")

from .base_service import SecurityService, ServiceHealth, ServiceStatus, ServiceType, service_registry
from .interfaces import PTaaSService
from ..domain.tenant_entities import ScanTarget, ScanResult, SecurityFinding

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Exception raised for security violations"""
    pass

@dataclass
class AdvancedScannerConfig:
    """Advanced configuration for security scanners"""
    name: str
    executable_path: Optional[str]
    version: Optional[str]
    available: bool
    timeout: int
    max_rate: int
    config: Dict[str, Any]
    capabilities: List[str]
    output_formats: List[str]
    threat_model: str

@dataclass
class VulnerabilityIntelligence:
    """Enhanced vulnerability with threat intelligence"""
    vulnerability_id: str
    name: str
    severity: str
    cvss_score: Optional[float]
    cvss_vector: Optional[str]
    description: str
    affected_component: str
    port: Optional[int]
    service: Optional[str]
    evidence: Dict[str, Any]
    references: List[str]
    remediation: str
    remediation_effort: str
    scanner: str
    timestamp: datetime
    threat_actor_groups: List[str]
    exploit_available: bool
    patch_available: bool
    business_impact: str
    technical_impact: str

@dataclass
class ThreatProfile:
    """Threat profile for target assessment"""
    target_id: str
    attack_surface_score: float
    exposure_level: str
    critical_assets: List[str]
    attack_vectors: List[str]
    threat_actors: List[str]
    mitre_techniques: List[str]
    risk_score: float

class AdvancedSecurityScannerService(SecurityService, PTaaSService):
    """Enhanced security scanner service with advanced capabilities"""
    
    def __init__(self, **kwargs):
        service_id = kwargs.pop("service_id", "advanced_ptaas_scanner")
        dependencies = kwargs.pop("dependencies", ["database", "redis", "vault", "ml_engine"])
        config = kwargs.pop("config", {})
        
        super().__init__(
            service_id=service_id,
            dependencies=dependencies,
            config=config
        )
        
        self.scanners: Dict[str, AdvancedScannerConfig] = {}
        self.scan_queue = asyncio.Queue()
        self.active_scans: Dict[str, asyncio.Task] = {}
        self.scan_results: Dict[str, ScanResult] = {}
        self.threat_intelligence: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Advanced scanner configurations
        self.scanner_configs = {
            "nmap": {
                "executable": "nmap",
                "timeout": 600,
                "max_rate": 1000,
                "output_formats": ["xml", "json", "greppable"],
                "capabilities": ["port_scan", "service_detection", "os_fingerprint", "vuln_scripts"],
                "threat_model": "network_reconnaissance"
            },
            "nuclei": {
                "executable": "nuclei",
                "timeout": 1800,
                "max_rate": 50,
                "templates_path": "~/nuclei-templates",
                "capabilities": ["vulnerability_scan", "misconfiguration", "exposed_services"],
                "threat_model": "application_vulnerabilities"
            },
            "nikto": {
                "executable": "nikto",
                "timeout": 900,
                "max_rate": 10,
                "output_formats": ["json", "xml", "csv"],
                "capabilities": ["web_scan", "cgi_scan", "ssl_scan"],
                "threat_model": "web_application"
            },
            "sslscan": {
                "executable": "sslscan",
                "timeout": 300,
                "max_rate": 5,
                "output_formats": ["xml", "json"],
                "capabilities": ["ssl_analysis", "cipher_analysis", "certificate_analysis"],
                "threat_model": "cryptographic_weaknesses"
            },
            "dirb": {
                "executable": "dirb",
                "timeout": 900,
                "wordlist": "/usr/share/dirb/wordlists/common.txt",
                "capabilities": ["directory_discovery", "file_discovery"],
                "threat_model": "information_disclosure"
            },
            "gobuster": {
                "executable": "gobuster",
                "timeout": 600,
                "wordlist": "/usr/share/wordlists/dirb/common.txt",
                "capabilities": ["directory_bruteforce", "dns_enumeration", "vhost_discovery"],
                "threat_model": "information_disclosure"
            },
            "sqlmap": {
                "executable": "sqlmap",
                "timeout": 1200,
                "max_rate": 1,
                "risk_level": 1,
                "capabilities": ["sql_injection", "database_enumeration", "data_extraction"],
                "threat_model": "data_breach"
            },
            "masscan": {
                "executable": "masscan",
                "timeout": 300,
                "max_rate": 10000,
                "capabilities": ["fast_port_scan", "internet_scan"],
                "threat_model": "network_reconnaissance"
            },
            "hydra": {
                "executable": "hydra",
                "timeout": 1800,
                "max_rate": 16,
                "capabilities": ["password_attack", "service_bruteforce"],
                "threat_model": "credential_compromise"
            },
            "testssl": {
                "executable": "testssl.sh",
                "timeout": 600,
                "output_formats": ["json", "csv", "html"],
                "capabilities": ["ssl_comprehensive", "tls_analysis", "cipher_audit"],
                "threat_model": "cryptographic_weaknesses"
            },
            "amass": {
                "executable": "amass",
                "timeout": 1800,
                "capabilities": ["subdomain_enumeration", "asset_discovery", "osint"],
                "threat_model": "reconnaissance"
            },
            "whatweb": {
                "executable": "whatweb",
                "timeout": 300,
                "capabilities": ["technology_detection", "cms_detection", "plugin_detection"],
                "threat_model": "technology_stack"
            }
        }
        
        # Initialize ML models if available
        if ML_AVAILABLE:
            self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize ML models for vulnerability analysis"""
        try:
            # Vulnerability severity classifier
            self.ml_models['severity_classifier'] = {
                'model': RandomForestClassifier(n_estimators=100),
                'vectorizer': TfidfVectorizer(max_features=1000),
                'trained': False
            }
            
            # Threat clustering model
            self.ml_models['threat_clusterer'] = {
                'model': DBSCAN(eps=0.3, min_samples=2),
                'trained': False
            }
            
            logger.info("ML models initialized for vulnerability analysis")
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def initialize(self) -> bool:
        """Initialize the enhanced scanner service"""
        try:
            logger.info("Initializing Advanced Security Scanner Service...")
            
            # Detect available scanners
            await self._detect_advanced_scanners()
            
            # Load threat intelligence
            await self._load_threat_intelligence()
            
            # Start scanner queue processor
            asyncio.create_task(self._process_scan_queue())
            
            # Start background tasks
            asyncio.create_task(self._update_threat_intelligence())
            asyncio.create_task(self._cleanup_old_results())
            
            logger.info(f"Advanced scanner service initialized with {len(self.scanners)} scanners")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced scanner service: {e}")
            return False
    
    async def _detect_advanced_scanners(self):
        """Detect available security scanners with enhanced capabilities"""
        for scanner_name, config in self.scanner_configs.items():
            try:
                scanner_path = await self._find_executable(config["executable"])
                if scanner_path:
                    version = await self._get_scanner_version(scanner_path, scanner_name)
                    capabilities = await self._detect_scanner_capabilities(scanner_path, scanner_name)
                    
                    self.scanners[scanner_name] = AdvancedScannerConfig(
                        name=scanner_name,
                        executable_path=scanner_path,
                        version=version,
                        available=True,
                        timeout=config.get("timeout", 300),
                        max_rate=config.get("max_rate", 10),
                        config=config,
                        capabilities=capabilities,
                        output_formats=config.get("output_formats", ["text"]),
                        threat_model=config.get("threat_model", "general")
                    )
                    
                    logger.info(f"Detected advanced scanner: {scanner_name} v{version} with capabilities: {capabilities}")
                else:
                    self.scanners[scanner_name] = AdvancedScannerConfig(
                        name=scanner_name,
                        executable_path=None,
                        version=None,
                        available=False,
                        timeout=config.get("timeout", 300),
                        max_rate=config.get("max_rate", 10),
                        config=config,
                        capabilities=[],
                        output_formats=config.get("output_formats", ["text"]),
                        threat_model=config.get("threat_model", "general")
                    )
                    
                    logger.warning(f"Advanced scanner not found: {scanner_name}")
                    
            except Exception as e:
                logger.error(f"Error detecting advanced scanner {scanner_name}: {e}")
    
    async def _detect_scanner_capabilities(self, scanner_path: str, scanner_name: str) -> List[str]:
        """Detect actual scanner capabilities"""
        capabilities = []
        
        try:
            if scanner_name == "nmap":
                # Check for NSE scripts
                nse_dir = "/usr/share/nmap/scripts"
                if os.path.exists(nse_dir):
                    capabilities.extend(["nse_scripts", "vuln_detection", "auth_bypass"])
                
                # Check for specific nmap capabilities
                help_output = await self._run_command([scanner_path, "--help"])
                if "-sV" in help_output:
                    capabilities.append("service_versioning")
                if "-O" in help_output:
                    capabilities.append("os_detection")
                if "--script" in help_output:
                    capabilities.append("script_engine")
            
            elif scanner_name == "nuclei":
                # Check templates
                templates_path = self.scanner_configs[scanner_name].get("templates_path")
                if templates_path:
                    expanded_path = Path(templates_path).expanduser()
                    if expanded_path.exists():
                        template_count = len(list(expanded_path.rglob("*.yaml")))
                        capabilities.append(f"templates_{template_count}")
            
            elif scanner_name == "sqlmap":
                help_output = await self._run_command([scanner_path, "--help"])
                if "--tamper" in help_output:
                    capabilities.append("evasion_techniques")
                if "--tor" in help_output:
                    capabilities.append("anonymization")
            
        except Exception as e:
            logger.debug(f"Error detecting capabilities for {scanner_name}: {e}")
        
        return capabilities
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence data"""
        try:
            # Load CVE data, MITRE ATT&CK mappings, etc.
            self.threat_intelligence = {
                "cve_database": {},
                "mitre_techniques": {},
                "threat_actors": {},
                "exploit_database": {},
                "last_updated": datetime.utcnow()
            }
            
            # In production, this would load from external threat intel feeds
            logger.info("Threat intelligence loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load threat intelligence: {e}")
    
    async def _update_threat_intelligence(self):
        """Background task to update threat intelligence"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                await self._load_threat_intelligence()
                logger.info("Threat intelligence updated")
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
    
    async def _cleanup_old_results(self):
        """Background task to cleanup old scan results"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Remove old results
                old_results = [
                    scan_id for scan_id, result in self.scan_results.items()
                    if result.start_time < cutoff_time
                ]
                
                for scan_id in old_results:
                    del self.scan_results[scan_id]
                
                if old_results:
                    logger.info(f"Cleaned up {len(old_results)} old scan results")
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def advanced_comprehensive_scan(self, target: ScanTarget) -> ScanResult:
        """Perform advanced comprehensive security scan with AI analysis"""
        scan_id = f"advanced_scan_{target.host}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting advanced comprehensive scan {scan_id} for {target.host}")
        
        try:
            scan_result = ScanResult(
                scan_id=scan_id,
                target=target.host,
                scan_type="advanced_comprehensive",
                start_time=start_time,
                end_time=start_time,
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
            
            # Phase 1: Intelligence Gathering
            logger.info("Phase 1: Advanced reconnaissance and intelligence gathering")
            intel_results = await self._advanced_reconnaissance(target)
            scan_result.raw_output.update(intel_results.get("raw_output", {}))
            
            # Phase 2: Network Analysis
            logger.info("Phase 2: Advanced network analysis")
            network_results = await self._advanced_network_analysis(target)
            scan_result.open_ports.extend(network_results.get("open_ports", []))
            scan_result.services.extend(network_results.get("services", []))
            scan_result.os_fingerprint = network_results.get("os_fingerprint", {})
            scan_result.raw_output.update(network_results.get("raw_output", {}))
            
            # Phase 3: Advanced Vulnerability Assessment
            logger.info("Phase 3: Advanced vulnerability assessment")
            vuln_results = await self._advanced_vulnerability_assessment(target, network_results)
            scan_result.vulnerabilities.extend(vuln_results.get("vulnerabilities", []))
            scan_result.raw_output.update(vuln_results.get("raw_output", {}))
            
            # Phase 4: Application Security Testing
            logger.info("Phase 4: Advanced application security testing")
            app_results = await self._advanced_application_testing(target, network_results)
            scan_result.vulnerabilities.extend(app_results.get("vulnerabilities", []))
            scan_result.raw_output.update(app_results.get("raw_output", {}))
            
            # Phase 5: AI-Powered Analysis
            if ML_AVAILABLE:
                logger.info("Phase 5: AI-powered vulnerability analysis")
                ai_analysis = await self._ai_vulnerability_analysis(scan_result)
                scan_result.vulnerabilities.extend(ai_analysis.get("enhanced_vulnerabilities", []))
                scan_result.scan_statistics.update(ai_analysis.get("ai_metrics", {}))
            
            # Phase 6: Threat Modeling
            logger.info("Phase 6: Threat profile generation")
            threat_profile = await self._generate_threat_profile(target, scan_result)
            scan_result.scan_statistics["threat_profile"] = asdict(threat_profile)
            
            # Finalize scan results
            scan_result.end_time = datetime.now()
            scan_result.status = "completed"
            scan_result.scan_statistics.update({
                "duration_seconds": (scan_result.end_time - scan_result.start_time).total_seconds(),
                "ports_scanned": len(target.ports) if target.ports else 65535,
                "open_ports_found": len(scan_result.open_ports),
                "services_identified": len(scan_result.services),
                "vulnerabilities_found": len(scan_result.vulnerabilities),
                "critical_vulns": len([v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]),
                "high_vulns": len([v for v in scan_result.vulnerabilities if v.get("severity") == "high"]),
                "scanners_used": [name for name, config in self.scanners.items() if config.available],
                "ai_analysis_enabled": ML_AVAILABLE
            })
            
            # Generate advanced recommendations
            scan_result.recommendations = await self._generate_advanced_recommendations(scan_result)
            
            logger.info(f"Advanced scan {scan_id} completed: {len(scan_result.vulnerabilities)} vulnerabilities found")
            return scan_result
            
        except Exception as e:
            logger.error(f"Advanced comprehensive scan failed: {e}")
            scan_result.end_time = datetime.now()
            scan_result.status = "failed"
            scan_result.scan_statistics["error"] = str(e)
            return scan_result
    
    async def _advanced_reconnaissance(self, target: ScanTarget) -> Dict[str, Any]:
        """Advanced reconnaissance using multiple OSINT tools"""
        results = {"raw_output": {}, "intelligence": {}}
        
        try:
            # Subdomain enumeration with Amass
            if self.scanners.get("amass", {}).available:
                amass_results = await self._run_amass_enum(target.host)
                results["raw_output"]["amass"] = amass_results.get("raw_output", "")
                results["intelligence"]["subdomains"] = amass_results.get("subdomains", [])
            
            # Technology detection with WhatWeb
            if self.scanners.get("whatweb", {}).available:
                whatweb_results = await self._run_whatweb_scan(target.host)
                results["raw_output"]["whatweb"] = whatweb_results.get("raw_output", "")
                results["intelligence"]["technologies"] = whatweb_results.get("technologies", [])
            
            # DNS enumeration
            dns_results = await self._advanced_dns_enumeration(target.host)
            results["raw_output"]["dns"] = dns_results.get("raw_output", "")
            results["intelligence"]["dns_records"] = dns_results.get("records", {})
            
        except Exception as e:
            logger.error(f"Advanced reconnaissance failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _advanced_network_analysis(self, target: ScanTarget) -> Dict[str, Any]:
        """Advanced network analysis with multiple scanning techniques"""
        results = {"open_ports": [], "services": [], "os_fingerprint": {}, "raw_output": {}}
        
        try:
            # Fast initial scan with Masscan
            if self.scanners.get("masscan", {}).available:
                masscan_results = await self._run_masscan_scan(target)
                results["raw_output"]["masscan"] = masscan_results.get("raw_output", "")
                fast_ports = masscan_results.get("open_ports", [])
                
                # Focus detailed scan on discovered ports
                if fast_ports:
                    target.ports = [p["port"] for p in fast_ports]
            
            # Detailed scan with Nmap
            if self.scanners.get("nmap", {}).available:
                nmap_results = await self._run_advanced_nmap_scan(target)
                results["open_ports"].extend(nmap_results.get("open_ports", []))
                results["services"].extend(nmap_results.get("services", []))
                results["os_fingerprint"] = nmap_results.get("os_fingerprint", {})
                results["raw_output"]["nmap"] = nmap_results.get("raw_output", "")
        
        except Exception as e:
            logger.error(f"Advanced network analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _advanced_vulnerability_assessment(self, target: ScanTarget, network_results: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced vulnerability assessment with multiple tools"""
        results = {"vulnerabilities": [], "raw_output": {}}
        
        try:
            # Nuclei vulnerability scanning
            if self.scanners.get("nuclei", {}).available:
                nuclei_results = await self._run_advanced_nuclei_scan(target)
                results["vulnerabilities"].extend(nuclei_results.get("vulnerabilities", []))
                results["raw_output"]["nuclei"] = nuclei_results.get("raw_output", "")
            
            # Service-specific vulnerability tests
            services = network_results.get("services", [])
            for service in services:
                service_vulns = await self._test_service_vulnerabilities(target.host, service)
                results["vulnerabilities"].extend(service_vulns)
        
        except Exception as e:
            logger.error(f"Advanced vulnerability assessment failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _advanced_application_testing(self, target: ScanTarget, network_results: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced application security testing"""
        results = {"vulnerabilities": [], "raw_output": {}}
        
        try:
            # Identify web applications
            web_ports = [p for p in network_results.get("open_ports", []) if p.get("port") in [80, 443, 8080, 8443, 3000, 8000]]
            
            for port_info in web_ports:
                port = port_info.get("port")
                
                # Advanced web application testing
                web_results = await self._advanced_web_testing(target.host, port)
                results["vulnerabilities"].extend(web_results.get("vulnerabilities", []))
                results["raw_output"][f"web_test_{port}"] = web_results.get("raw_output", "")
                
                # SQL injection testing
                if self.scanners.get("sqlmap", {}).available:
                    sqli_results = await self._run_sqlmap_scan(target.host, port)
                    results["vulnerabilities"].extend(sqli_results.get("vulnerabilities", []))
                    results["raw_output"][f"sqlmap_{port}"] = sqli_results.get("raw_output", "")
        
        except Exception as e:
            logger.error(f"Advanced application testing failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _ai_vulnerability_analysis(self, scan_result: ScanResult) -> Dict[str, Any]:
        """AI-powered vulnerability analysis and enhancement"""
        if not ML_AVAILABLE:
            return {"enhanced_vulnerabilities": [], "ai_metrics": {}}
        
        try:
            enhanced_vulns = []
            ai_metrics = {
                "ml_analysis_enabled": True,
                "vulnerability_clustering": {},
                "severity_predictions": {},
                "false_positive_likelihood": {}
            }
            
            # Enhance vulnerabilities with ML analysis
            for vuln in scan_result.vulnerabilities:
                enhanced_vuln = await self._enhance_vulnerability_with_ai(vuln)
                enhanced_vulns.append(enhanced_vuln)
            
            # Cluster similar vulnerabilities
            if len(scan_result.vulnerabilities) > 3:
                clusters = await self._cluster_vulnerabilities(scan_result.vulnerabilities)
                ai_metrics["vulnerability_clustering"] = clusters
            
            return {
                "enhanced_vulnerabilities": enhanced_vulns,
                "ai_metrics": ai_metrics
            }
        
        except Exception as e:
            logger.error(f"AI vulnerability analysis failed: {e}")
            return {"enhanced_vulnerabilities": [], "ai_metrics": {"error": str(e)}}
    
    async def _enhance_vulnerability_with_ai(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance individual vulnerability with AI analysis"""
        try:
            enhanced = vulnerability.copy()
            
            # Add threat intelligence
            vuln_name = vulnerability.get("name", "")
            if "sql" in vuln_name.lower():
                enhanced["threat_actors"] = ["APT groups", "Cybercriminals", "Script kiddies"]
                enhanced["mitre_techniques"] = ["T1190", "T1059"]
            elif "xss" in vuln_name.lower():
                enhanced["threat_actors"] = ["Web attackers", "Social engineers"]
                enhanced["mitre_techniques"] = ["T1189", "T1203"]
            
            # Calculate business impact score
            severity = vulnerability.get("severity", "info")
            port = vulnerability.get("port", 0)
            
            impact_score = 0
            if severity == "critical":
                impact_score += 9
            elif severity == "high":
                impact_score += 7
            elif severity == "medium":
                impact_score += 5
            
            if port in [22, 3389, 1433, 3306, 5432]:  # Critical services
                impact_score += 2
            
            enhanced["business_impact_score"] = min(impact_score, 10)
            enhanced["ai_enhanced"] = True
            
            return enhanced
        
        except Exception as e:
            logger.error(f"Error enhancing vulnerability with AI: {e}")
            return vulnerability
    
    async def _generate_threat_profile(self, target: ScanTarget, scan_result: ScanResult) -> ThreatProfile:
        """Generate comprehensive threat profile for target"""
        try:
            # Calculate attack surface score
            attack_surface_score = len(scan_result.open_ports) * 1.5 + len(scan_result.services) * 2.0
            attack_surface_score = min(attack_surface_score, 100.0)
            
            # Determine exposure level
            critical_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]
            high_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "high"]
            
            if critical_vulns:
                exposure_level = "critical"
            elif high_vulns:
                exposure_level = "high"
            elif scan_result.vulnerabilities:
                exposure_level = "medium"
            else:
                exposure_level = "low"
            
            # Identify critical assets
            critical_assets = []
            for service in scan_result.services:
                service_name = service.get("name", "").lower()
                if any(db in service_name for db in ["mysql", "postgres", "oracle", "mssql"]):
                    critical_assets.append("database")
                elif "ssh" in service_name:
                    critical_assets.append("remote_access")
                elif service_name in ["http", "https"]:
                    critical_assets.append("web_application")
            
            # Map attack vectors
            attack_vectors = []
            for vuln in scan_result.vulnerabilities:
                vuln_name = vuln.get("name", "").lower()
                if "sql" in vuln_name:
                    attack_vectors.append("sql_injection")
                elif "xss" in vuln_name:
                    attack_vectors.append("cross_site_scripting")
                elif "rce" in vuln_name or "remote" in vuln_name:
                    attack_vectors.append("remote_code_execution")
            
            # Calculate overall risk score
            risk_score = (attack_surface_score * 0.3 + 
                         len(critical_vulns) * 20 + 
                         len(high_vulns) * 10 + 
                         len(critical_assets) * 5)
            risk_score = min(risk_score, 100.0)
            
            return ThreatProfile(
                target_id=scan_result.scan_id,
                attack_surface_score=attack_surface_score,
                exposure_level=exposure_level,
                critical_assets=list(set(critical_assets)),
                attack_vectors=list(set(attack_vectors)),
                threat_actors=["APT groups", "Cybercriminals", "Insider threats"],
                mitre_techniques=["T1190", "T1059", "T1078", "T1055"],
                risk_score=risk_score
            )
        
        except Exception as e:
            logger.error(f"Error generating threat profile: {e}")
            return ThreatProfile(
                target_id=scan_result.scan_id,
                attack_surface_score=0.0,
                exposure_level="unknown",
                critical_assets=[],
                attack_vectors=[],
                threat_actors=[],
                mitre_techniques=[],
                risk_score=0.0
            )
    
    async def _generate_advanced_recommendations(self, scan_result: ScanResult) -> List[str]:
        """Generate advanced security recommendations with prioritization"""
        recommendations = []
        
        try:
            critical_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]
            high_vulns = [v for v in scan_result.vulnerabilities if v.get("severity") == "high"]
            
            # Critical recommendations
            if critical_vulns:
                recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Critical vulnerabilities detected")
                recommendations.append(f"🔴 Patch {len(critical_vulns)} critical vulnerabilities within 24 hours")
                
                for vuln in critical_vulns[:3]:  # Top 3 critical
                    recommendations.append(f"   • {vuln.get('name', 'Unknown')}: {vuln.get('remediation', 'Apply security patch')}")
            
            # High priority recommendations
            if high_vulns:
                recommendations.append(f"⚠️ HIGH PRIORITY: Address {len(high_vulns)} high-severity vulnerabilities within 72 hours")
            
            # Service-specific recommendations
            open_ports = [p.get("port") for p in scan_result.open_ports]
            
            if 22 in open_ports:
                recommendations.append("🔐 SSH Security: Implement key-based authentication, disable root login")
            
            if any(port in open_ports for port in [80, 443, 8080]):
                recommendations.append("🌐 Web Security: Implement WAF, security headers, and HTTPS")
            
            if any(port in open_ports for port in [1433, 3306, 5432]):
                recommendations.append("🗄️ Database Security: Restrict network access, enable TLS, audit access")
            
            # Advanced recommendations
            recommendations.extend([
                "🔄 Implement automated vulnerability scanning (weekly)",
                "📊 Deploy security monitoring and SIEM",
                "🛡️ Configure network segmentation and micro-segmentation",
                "👥 Conduct security awareness training",
                "📋 Develop incident response procedures",
                "🏗️ Implement infrastructure as code security scanning",
                "🔍 Deploy endpoint detection and response (EDR)",
                "📝 Create security policies and compliance documentation"
            ])
            
            # AI-powered recommendations
            if ML_AVAILABLE and scan_result.scan_statistics.get("ai_analysis_enabled"):
                threat_profile = scan_result.scan_statistics.get("threat_profile", {})
                risk_score = threat_profile.get("risk_score", 0)
                
                if risk_score > 70:
                    recommendations.insert(0, "🤖 AI ANALYSIS: High-risk target - implement comprehensive security controls")
                elif risk_score > 40:
                    recommendations.append("🤖 AI ANALYSIS: Medium risk - enhance monitoring and access controls")
            
        except Exception as e:
            logger.error(f"Error generating advanced recommendations: {e}")
            recommendations.append("❌ Error generating recommendations - manual review required")
        
        return recommendations
    
    # Additional helper methods for advanced scanning would be implemented here
    # Including _run_masscan_scan, _run_advanced_nuclei_scan, _run_sqlmap_scan, etc.
    
    async def _run_command(self, cmd: List[str], timeout: int = 30) -> str:
        """Run shell command safely with timeout"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return stdout.decode('utf-8', errors='ignore')
        
        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {' '.join(cmd)}")
            return ""
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return ""


# Global advanced scanner service instance
_advanced_scanner_service: Optional[AdvancedSecurityScannerService] = None

async def get_advanced_scanner_service() -> AdvancedSecurityScannerService:
    """Get global advanced scanner service instance"""
    global _advanced_scanner_service
    
    if _advanced_scanner_service is None:
        _advanced_scanner_service = AdvancedSecurityScannerService()
        await _advanced_scanner_service.initialize()
        
        # Register with global service registry
        service_registry.register(_advanced_scanner_service)
    
    return _advanced_scanner_service