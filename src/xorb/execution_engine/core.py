import asyncio
import logging
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
from dataclasses import dataclass, field
import concurrent.futures

import aioredis
import aiohttp
from aiohttp import web
import aiofiles
from playwright.async_api import async_playwright, Browser, Page
import requests
from pydantic import BaseModel, Field

from xorb.shared.epyc_execution_config import EPYCExecutionConfig
from xorb.shared.execution_enums import ScanType, ExploitType, EvidenceType
from xorb.shared.execution_models import ScanResult, ExploitResult, Evidence, StealthConfig
from xorb.shared.models import ScanResultModel, ExploitResultModel, EvidenceModel
from xorb.database.database import AsyncSessionLocal
from xorb.database.repositories import ScanRepository, ExploitRepository, EvidenceRepository
from sqlalchemy.ext.asyncio import AsyncSession
from .scanner import MultiEngineScanner
from .stealth_web_engine import StealthWebEngine

# Unified Execution Engine
class UnifiedExecutionEngine:
    def __init__(self, db_session: AsyncSession):
        self.app = web.Application()
        self.redis = None
        self.db_session = db_session
        self.scanner = None
        self.web_engine = None
        self.active_operations = {}
        self.evidence_store = {}
        self.logger = logging.getLogger(__name__)
        self.scan_repo = ScanRepository(db_session)
        self.exploit_repo = ExploitRepository(db_session)
        self.evidence_repo = EvidenceRepository(db_session)
        
    async def initialize(self):
        """Initialize the execution engine."""
        # Redis connection
        self.redis = aioredis.from_url("redis://redis:6379")
        
        # Initialize components
        self.scanner = MultiEngineScanner(self.redis)
        self.web_engine = StealthWebEngine()
        await self.web_engine.initialize()
        
        # Setup routes
        self.setup_routes()
        
        self.logger.info("Unified Execution Engine initialized")
    
    def setup_routes(self):
        """Setup API routes."""
        # Scanning operations
        self.app.router.add_post('/scan/nmap', self.run_nmap_scan)
        self.app.router.add_post('/scan/nuclei', self.run_nuclei_scan)
        self.app.router.add_post('/scan/zap', self.run_zap_scan)
        self.app.router.add_post('/scan/web', self.run_web_scan)
        
        # Exploitation operations
        self.app.router.add_post('/exploit/web', self.run_web_exploit)
        self.app.router.add_post('/exploit/network', self.run_network_exploit)
        
        # Evidence management
        self.app.router.add_get('/evidence', self.list_evidence)
        self.app.router.add_get('/evidence/{evidence_id}', self.get_evidence)
        self.app.router.add_post('/evidence', self.store_evidence)
        
        # Operation management
        self.app.router.add_get('/operations', self.list_operations)
        self.app.router.add_get('/operations/{operation_id}', self.get_operation)
        
        # Health and status
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.get_status)
    
    async def run_nmap_scan(self, request: web.Request):
        """Run Nmap scan endpoint with strategic database integration."""
        try:
            data = await request.json()
            target = data['target']
            scan_type = data.get('scan_type', 'comprehensive')
            
            # Create database model for persistence
            scan_model = ScanResultModel(
                target_id=data.get('target_id', target),
                scan_type=ScanType.PORT_SCAN.value,
                status="running",
                start_time=datetime.utcnow(),
                metadata={
                    'target': target,
                    'scan_type_requested': scan_type,
                    'epyc_optimized': True
                }
            )
            
            # Store initial scan record in database
            scan_record = await self.scan_repo.create_scan(scan_model)
            
            # Also cache in Redis for fast access during scan
            scan_pydantic = ScanResult(
                id=scan_record.id,
                target_id=scan_record.target_id,
                scan_type=ScanType.PORT_SCAN,
                status="running"
            )
            await self.redis.setex(f"scan:{scan_record.id}", 7200, scan_pydantic.json())
            
            # Run scan with EPYC optimization
            nmap_results = await self.scanner.run_nmap_scan(target, scan_type)
            
            # Update scan record with results
            scan_record.end_time = datetime.utcnow()
            scan_record.duration = (scan_record.end_time - scan_record.start_time).total_seconds()
            scan_record.results = nmap_results
            scan_record.status = "completed" if 'error' not in nmap_results else "failed"
            
            # Extract and analyze findings with enhanced intelligence
            findings = []
            if 'hosts' in nmap_results:
                for host in nmap_results['hosts']:
                    for port in host.get('ports', []):
                        if port.get('state') == 'open':
                            finding = {
                                'type': 'open_port',
                                'port': port.get('portid'),
                                'protocol': port.get('protocol'),
                                'service': port.get('service', {}),
                                'severity': self._calculate_port_severity(port),
                                'risk_score': self._calculate_risk_score(port),
                                'timestamp': datetime.utcnow().isoformat()
                            }
                            findings.append(finding)
            
            scan_record.findings = findings
            scan_record.metadata.update({
                'findings_count': len(findings),
                'high_risk_ports': len([f for f in findings if f.get('severity') == 'high']),
                'completion_time': datetime.utcnow().isoformat()
            })
            
            # Persist to database
            await self.scan_repo.update_scan(scan_record)
            
            # Update Redis cache
            scan_pydantic.status = scan_record.status
            scan_pydantic.end_time = scan_record.end_time
            scan_pydantic.duration = scan_record.duration
            scan_pydantic.results = scan_record.results
            scan_pydantic.findings = scan_record.findings
            await self.redis.setex(f"scan:{scan_record.id}", 7200, scan_pydantic.json())
            
            return web.json_response({
                'scan_id': scan_record.id,
                'status': scan_record.status,
                'duration': scan_record.duration,
                'findings_count': len(scan_record.findings),
                'high_risk_findings': scan_record.metadata.get('high_risk_ports', 0),
                'results': nmap_results,
                'intelligence_enhanced': True
            })
            
        except Exception as e:
            self.logger.error(f"Strategic Nmap scan error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    def _calculate_port_severity(self, port: Dict[str, Any]) -> str:
        """Calculate port severity with intelligence."""
        port_num = int(port.get('portid', 0))
        service = port.get('service', {})
        
        # High-risk services and ports
        high_risk_ports = [21, 22, 23, 25, 53, 80, 135, 139, 443, 445, 993, 995, 1433, 3389, 5432, 5900]
        critical_services = ['ssh', 'ftp', 'telnet', 'smtp', 'http', 'https', 'smb', 'rdp', 'vnc']
        
        if port_num in high_risk_ports or service.get('name', '').lower() in critical_services:
            return 'high'
        elif port_num < 1024:  # Well-known ports
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_score(self, port: Dict[str, Any]) -> float:
        """Calculate numerical risk score for ML analysis."""
        port_num = int(port.get('portid', 0))
        service = port.get('service', {})
        
        base_score = 1.0
        if port_num in [22, 23, 3389]:  # Remote access
            base_score = 9.0
        elif port_num in [80, 443]:  # Web services
            base_score = 7.0
        elif port_num in [21, 25, 53]:  # Network services
            base_score = 6.0
        elif port_num < 1024:
            base_score = 4.0
        
        # Version-based risk adjustment
        version = service.get('version', '')
        if 'outdated' in version.lower() or any(year in version for year in ['2019', '2020', '2021']):
            base_score += 2.0
            
        return min(10.0, base_score)
    
    async def run_web_scan(self, request: web.Request):
        """Run strategic stealth web scan with enhanced intelligence."""
        try:
            data = await request.json()
            target = data['target']
            
            # Configure advanced stealth mode with EPYC optimization
            stealth_config = StealthConfig(
                mode=data.get('stealth_mode', 'normal'),
                user_agent=data.get('user_agent', EPYCExecutionConfig.USER_AGENTS[0]),
                delay_range=tuple(data.get('delay_range', [1.0, 3.0]))
            )
            
            # Run intelligent stealth browse with enhanced analysis
            evidence_data = await self.web_engine.stealth_browse(target, stealth_config)
            
            # Create database model for evidence persistence
            evidence_model = EvidenceModel(
                evidence_type=EvidenceType.SCREENSHOT.value,
                target_id=data.get('target_id', target),
                content=json.dumps(evidence_data),
                file_path=evidence_data.get('screenshot'),
                metadata={
                    'url': target,
                    'stealth_config': stealth_config.dict(),
                    'forms_found': len(evidence_data.get('forms', [])),
                    'links_found': len(evidence_data.get('links', [])),
                    'screenshot_path': evidence_data.get('screenshot'),
                    'duration': evidence_data.get('duration', 0),
                    'intelligence_analysis': self._analyze_web_intelligence(evidence_data),
                    'epyc_optimized': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Store evidence in database
            evidence_record = await self.evidence_repo.create_evidence(evidence_model)
            
            # Also cache in Redis for fast access
            evidence_pydantic = Evidence(
                id=evidence_record.id,
                evidence_type=EvidenceType.SCREENSHOT,
                target_id=evidence_record.target_id,
                content=evidence_record.content,
                file_path=evidence_record.file_path,
                metadata=evidence_record.metadata
            )
            await self.redis.setex(f"evidence:{evidence_record.id}", 86400, evidence_pydantic.json())
            
            # Generate intelligence summary
            intelligence_summary = evidence_record.metadata.get('intelligence_analysis', {})
            
            return web.json_response({
                'evidence_id': evidence_record.id,
                'target': target,
                'forms_found': len(evidence_data.get('forms', [])),
                'links_found': len(evidence_data.get('links', [])),
                'screenshot_path': evidence_data.get('screenshot'),
                'duration': evidence_data.get('duration', 0),
                'intelligence_summary': intelligence_summary,
                'risk_indicators': intelligence_summary.get('risk_indicators', []),
                'stealth_effectiveness': intelligence_summary.get('stealth_score', 0),
                'intelligence_enhanced': True
            })
            
        except Exception as e:
            self.logger.error(f"Strategic web scan error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    def _analyze_web_intelligence(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intelligent analysis of web evidence."""
        analysis = {
            'risk_indicators': [],
            'attack_surface': {},
            'stealth_score': 10.0,
            'vulnerability_hints': [],
            'intelligence_confidence': 0.8
        }
        
        forms = evidence_data.get('forms', [])
        links = evidence_data.get('links', [])
        
        # Analyze forms for security risks
        for form in forms:
            form_analysis = self._analyze_form_security(form)
            if form_analysis['risk_level'] > 5:
                analysis['risk_indicators'].append({
                    'type': 'insecure_form',
                    'details': form_analysis,
                    'severity': 'medium' if form_analysis['risk_level'] < 8 else 'high'
                })
        
        # Analyze links for attack vectors
        for link in links:
            link_analysis = self._analyze_link_security(link)
            if link_analysis['suspicious']:
                analysis['vulnerability_hints'].append({
                    'type': 'suspicious_link',
                    'url': link,
                    'analysis': link_analysis
                })
        
        # Calculate attack surface
        analysis['attack_surface'] = {
            'form_inputs': len(forms),
            'external_links': len([l for l in links if self._is_external_link(l)]),
            'javascript_usage': 'high' if len(links) > 50 else 'medium',
            'estimated_complexity': self._estimate_app_complexity(evidence_data)
        }
        
        return analysis
    
    def _analyze_form_security(self, form: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze form for security vulnerabilities."""
        risk_score = 0
        issues = []
        
        method = form.get('method', '').upper()
        action = form.get('action', '')
        
        # Check for insecure practices
        if method == 'GET' and any('password' in inp.get('name', '').lower() for inp in form.get('inputs', [])):
            risk_score += 8
            issues.append('password_in_get')
        
        if not action.startswith('https://'):
            risk_score += 3
            issues.append('insecure_action')
        
        # Check for CSRF protection
        csrf_protected = any('csrf' in inp.get('name', '').lower() or 'token' in inp.get('name', '').lower() 
                           for inp in form.get('inputs', []))
        if not csrf_protected:
            risk_score += 4
            issues.append('no_csrf_protection')
        
        return {
            'risk_level': risk_score,
            'issues': issues,
            'method': method,
            'csrf_protected': csrf_protected
        }
    
    def _analyze_link_security(self, link: str) -> Dict[str, Any]:
        """Analyze link for security implications."""
        analysis = {
            'suspicious': False,
            'external': self._is_external_link(link),
            'risk_factors': []
        }
        
        # Check for suspicious patterns
        suspicious_patterns = ['admin', 'config', 'debug', 'test', 'dev', 'backup']
        if any(pattern in link.lower() for pattern in suspicious_patterns):
            analysis['suspicious'] = True
            analysis['risk_factors'].append('suspicious_path')
        
        # Check for file uploads or downloads
        if any(ext in link.lower() for ext in ['.zip', '.tar', '.sql', '.log', '.config']):
            analysis['suspicious'] = True
            analysis['risk_factors'].append('sensitive_file')
        
        return analysis
    
    def _is_external_link(self, link: str) -> bool:
        """Check if link is external."""
        return link.startswith('http') and not any(domain in link for domain in ['localhost', '127.0.0.1'])
    
    def _estimate_app_complexity(self, evidence_data: Dict[str, Any]) -> str:
        """Estimate application complexity for attack planning."""
        forms_count = len(evidence_data.get('forms', []))
        links_count = len(evidence_data.get('links', []))
        
        if forms_count > 10 or links_count > 100:
            return 'high'
        elif forms_count > 3 or links_count > 20:
            return 'medium'
        else:
            return 'low'
    
    async def health_check(self, request: web.Request):
        """Health check endpoint."""
        browser_status = self.web_engine.browser is not None
        
        return web.json_response({
            'status': 'healthy',
            'service': 'execution-engine',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'scanner': True,
                'web_engine': browser_status,
                'redis': await self._check_redis_health()
            },
            'active_operations': len(self.active_operations)
        })
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
