"""
Xorb PTaaS Scanner Service
Integrates Nuclei, ZAP, and Trivy for comprehensive security scanning
Optimized for AMD EPYC single-node deployment
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
import nats
from nats.js import JetStreamContext
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("xorb.scanner")

# Prometheus metrics
SCANS_TOTAL = Counter('xorb_scanner_scans_total', 'Total number of scans performed', ['tool', 'status'])
SCAN_DURATION = Histogram('xorb_scanner_scan_duration_seconds', 'Time spent scanning', ['tool'])
VULNERABILITIES_FOUND = Counter('xorb_scanner_vulnerabilities_total', 'Vulnerabilities found', ['tool', 'severity'])
ACTIVE_SCANS = Gauge('xorb_scanner_active_scans', 'Number of active scans')

class SecurityScanner:
    """Main scanner service orchestrating multiple security tools"""
    
    def __init__(self):
        self.nats_client = None
        self.js = None
        self.tools = {
            'nuclei': NucleiScanner(),
            'zap': ZAPScanner(),
            'trivy': TrivyScanner()
        }
        self.workspace = Path("/tmp/xorb/scanner_data")
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize NATS connection and JetStream"""
        try:
            self.nats_client = await nats.connect(
                servers=[os.getenv("NATS_URL", "nats://localhost:4222")],
                name="xorb-scanner"
            )
            self.js = self.nats_client.jetstream()
            
            # Subscribe to scan requests
            await self.js.subscribe(
                "scans.request",
                cb=self.handle_scan_request,
                queue="scanner-workers"
            )
            
            logger.info("Scanner service initialized", nats_connected=True)
            
        except Exception as e:
            logger.error("Failed to initialize scanner", error=str(e))
            raise
    
    async def handle_scan_request(self, msg):
        """Handle incoming scan requests from NATS"""
        try:
            scan_request = json.loads(msg.data.decode())
            scan_id = scan_request.get('scan_id')
            target = scan_request.get('target')
            scan_type = scan_request.get('type', 'comprehensive')
            
            logger.info("Processing scan request", 
                       scan_id=scan_id, target=target, type=scan_type)
            
            ACTIVE_SCANS.inc()
            
            # Perform scan
            results = await self.perform_scan(scan_request)
            
            # Publish results
            await self.publish_results(scan_id, results)
            
            # Acknowledge message
            await msg.ack()
            
            ACTIVE_SCANS.dec()
            SCANS_TOTAL.labels(tool='combined', status='success').inc()
            
        except Exception as e:
            logger.error("Scan request failed", error=str(e))
            SCANS_TOTAL.labels(tool='combined', status='error').inc()
            await msg.nak()
        finally:
            ACTIVE_SCANS.dec()
    
    async def perform_scan(self, scan_request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive security scan"""
        target = scan_request.get('target')
        scan_type = scan_request.get('type', 'comprehensive')
        tools = scan_request.get('tools', ['nuclei', 'zap', 'trivy'])
        
        results = {
            'scan_id': scan_request.get('scan_id'),
            'target': target,
            'timestamp': datetime.utcnow().isoformat(),
            'tools': {},
            'summary': {
                'total_vulnerabilities': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            }
        }
        
        # Run scans concurrently for better performance on EPYC
        scan_tasks = []
        for tool_name in tools:
            if tool_name in self.tools:
                task = asyncio.create_task(
                    self.run_tool_scan(tool_name, target, scan_request)
                )
                scan_tasks.append((tool_name, task))
        
        # Wait for all scans to complete
        for tool_name, task in scan_tasks:
            try:
                tool_result = await task
                results['tools'][tool_name] = tool_result
                
                # Update summary
                if 'vulnerabilities' in tool_result:
                    for vuln in tool_result['vulnerabilities']:
                        severity = vuln.get('severity', 'info').lower()
                        results['summary']['total_vulnerabilities'] += 1
                        if severity in results['summary']:
                            results['summary'][severity] += 1
                        
                        VULNERABILITIES_FOUND.labels(
                            tool=tool_name, 
                            severity=severity
                        ).inc()
                        
            except Exception as e:
                logger.error("Tool scan failed", tool=tool_name, error=str(e))
                results['tools'][tool_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    async def run_tool_scan(self, tool_name: str, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual security tool scan"""
        tool = self.tools[tool_name]
        
        with SCAN_DURATION.labels(tool=tool_name).time():
            try:
                result = await tool.scan(target, config)
                SCANS_TOTAL.labels(tool=tool_name, status='success').inc()
                return result
            except Exception as e:
                SCANS_TOTAL.labels(tool=tool_name, status='error').inc()
                raise
    
    async def publish_results(self, scan_id: str, results: Dict[str, Any]):
        """Publish scan results to NATS for triage service"""
        try:
            await self.js.publish(
                "scans.results",
                json.dumps(results).encode(),
                headers={'scan_id': scan_id}
            )
            logger.info("Scan results published", scan_id=scan_id)
        except Exception as e:
            logger.error("Failed to publish results", scan_id=scan_id, error=str(e))

class NucleiScanner:
    """Nuclei vulnerability scanner integration"""
    
    async def scan(self, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Nuclei scan"""
        logger.info("Starting Nuclei scan", target=target)
        
        cmd = [
            "nuclei",
            "-u", target,
            "-j",  # JSON output
            "-silent",
            "-nc",  # No color
            "-rate-limit", "100",  # EPYC can handle higher rates
            "-c", "32",  # Concurrency optimized for EPYC
        ]
        
        # Add template filters if specified
        if 'nuclei_templates' in config:
            cmd.extend(["-t", ",".join(config['nuclei_templates'])])
        else:
            cmd.extend(["-t", "cves,vulnerabilities,exposures"])
        
        try:
            result = await self._run_command(cmd)
            return self._parse_nuclei_output(result)
        except Exception as e:
            logger.error("Nuclei scan failed", target=target, error=str(e))
            raise
    
    async def _run_command(self, cmd: List[str]) -> str:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode()}")
        
        return stdout.decode()
    
    def _parse_nuclei_output(self, output: str) -> Dict[str, Any]:
        """Parse Nuclei JSON output"""
        vulnerabilities = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            try:
                vuln = json.loads(line)
                vulnerabilities.append({
                    'id': vuln.get('template-id'),
                    'name': vuln.get('info', {}).get('name'),
                    'severity': vuln.get('info', {}).get('severity', 'info'),
                    'description': vuln.get('info', {}).get('description'),
                    'url': vuln.get('matched-at'),
                    'timestamp': vuln.get('timestamp'),
                    'tool': 'nuclei'
                })
            except json.JSONDecodeError:
                continue
        
        return {
            'status': 'completed',
            'vulnerabilities': vulnerabilities,
            'tool': 'nuclei',
            'timestamp': datetime.utcnow().isoformat()
        }

class ZAPScanner:
    """OWASP ZAP scanner integration"""
    
    async def scan(self, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run ZAP scan"""
        logger.info("Starting ZAP scan", target=target)
        
        # Start ZAP in daemon mode
        zap_process = await asyncio.create_subprocess_exec(
            "zap.sh", "-daemon", "-port", "8090", "-config", "api.disablekey=true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        
        # Wait for ZAP to start
        await asyncio.sleep(10)
        
        try:
            # Configure ZAP and run scan
            async with aiohttp.ClientSession() as session:
                # Spider the target
                await self._zap_spider(session, target)
                
                # Active scan
                await self._zap_active_scan(session, target)
                
                # Get results
                results = await self._zap_get_results(session)
                
            return results
            
        finally:
            # Clean up ZAP process
            zap_process.terminate()
            await zap_process.wait()
    
    async def _zap_spider(self, session: aiohttp.ClientSession, target: str):
        """Spider target with ZAP"""
        async with session.get(f"http://localhost:8090/JSON/spider/action/scan/", 
                             params={'url': target}) as resp:
            result = await resp.json()
            scan_id = result['scan']
            
        # Wait for spider to complete
        while True:
            async with session.get(f"http://localhost:8090/JSON/spider/view/status/",
                                 params={'scanId': scan_id}) as resp:
                status = await resp.json()
                if int(status['status']) >= 100:
                    break
            await asyncio.sleep(2)
    
    async def _zap_active_scan(self, session: aiohttp.ClientSession, target: str):
        """Run active scan with ZAP"""
        async with session.get(f"http://localhost:8090/JSON/ascan/action/scan/",
                             params={'url': target}) as resp:
            result = await resp.json()
            scan_id = result['scan']
            
        # Wait for active scan to complete
        while True:
            async with session.get(f"http://localhost:8090/JSON/ascan/view/status/",
                                 params={'scanId': scan_id}) as resp:
                status = await resp.json()
                if int(status['status']) >= 100:
                    break
            await asyncio.sleep(5)
    
    async def _zap_get_results(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Get scan results from ZAP"""
        async with session.get("http://localhost:8090/JSON/core/view/alerts/") as resp:
            alerts = await resp.json()
            
        vulnerabilities = []
        for alert in alerts['alerts']:
            vulnerabilities.append({
                'id': alert.get('pluginId'),
                'name': alert.get('name'),
                'severity': self._map_zap_severity(alert.get('risk')),
                'description': alert.get('description'),
                'url': alert.get('url'),
                'solution': alert.get('solution'),
                'tool': 'zap'
            })
        
        return {
            'status': 'completed',
            'vulnerabilities': vulnerabilities,
            'tool': 'zap',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _map_zap_severity(self, risk: str) -> str:
        """Map ZAP risk levels to standard severity"""
        mapping = {
            'High': 'high',
            'Medium': 'medium',
            'Low': 'low',
            'Informational': 'info'
        }
        return mapping.get(risk, 'info')

class TrivyScanner:
    """Trivy vulnerability scanner for container images"""
    
    async def scan(self, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Trivy scan"""
        logger.info("Starting Trivy scan", target=target)
        
        # Determine scan type based on target
        scan_type = "image" if ":" in target else "fs"
        
        cmd = [
            "trivy",
            scan_type,
            "--format", "json",
            "--quiet",
            target
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Trivy scan failed: {stderr.decode()}")
            
            return self._parse_trivy_output(stdout.decode())
            
        except Exception as e:
            logger.error("Trivy scan failed", target=target, error=str(e))
            raise
    
    def _parse_trivy_output(self, output: str) -> Dict[str, Any]:
        """Parse Trivy JSON output"""
        try:
            trivy_result = json.loads(output)
            vulnerabilities = []
            
            for result in trivy_result.get('Results', []):
                for vuln in result.get('Vulnerabilities', []):
                    vulnerabilities.append({
                        'id': vuln.get('VulnerabilityID'),
                        'name': vuln.get('Title'),
                        'severity': vuln.get('Severity', 'info').lower(),
                        'description': vuln.get('Description'),
                        'package': vuln.get('PkgName'),
                        'version': vuln.get('InstalledVersion'),
                        'fixed_version': vuln.get('FixedVersion'),
                        'tool': 'trivy'
                    })
            
            return {
                'status': 'completed',
                'vulnerabilities': vulnerabilities,
                'tool': 'trivy',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Trivy output", error=str(e))
            return {
                'status': 'error',
                'error': f"Failed to parse output: {str(e)}",
                'tool': 'trivy'
            }

async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "scanner"}

async def main():
    """Main service entry point"""
    # Start Prometheus metrics server
    start_http_server(8004)
    
    # Initialize scanner
    scanner = SecurityScanner()
    await scanner.initialize()
    
    logger.info("Xorb Scanner service started", 
               epyc_optimized=True, 
               workspace=str(scanner.workspace))
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down scanner service")
    finally:
        await scanner.nats_client.close()

if __name__ == "__main__":
    asyncio.run(main())