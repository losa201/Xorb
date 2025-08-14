import asyncio
import logging
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import concurrent.futures

import aioredis
import aiofiles

from xorb.shared.epyc_execution_config import EPYCExecutionConfig
from xorb.shared.execution_enums import ScanType
from xorb.shared.execution_models import ScanResult

# Multi-Engine Scanner
class MultiEngineScanner:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.active_scans = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=EPYCExecutionConfig.MAX_CONCURRENT_SCANS
        )
        self.logger = logging.getLogger(__name__)

    async def run_nmap_scan(self, target: str, scan_type: str = "comprehensive") -> Dict[str, Any]:
        """Run Nmap scan with EPYC optimization."""
        try:
            # Build Nmap command based on scan type
            if scan_type == "discovery":
                cmd = [
                    "nmap", "-sn", "-T4", "--min-parallelism", "16",
                    "--max-parallelism", "32", target
                ]
            elif scan_type == "port_scan":
                cmd = [
                    "nmap", "-sS", "-T4", "-p-", "--min-parallelism", "16",
                    "--max-parallelism", "32", "--max-retries", "2", target
                ]
            elif scan_type == "comprehensive":
                cmd = [
                    "nmap", "-sS", "-sV", "-sC", "-T4", "-p-",
                    "--min-parallelism", "16", "--max-parallelism", "32",
                    "--script=vuln", "-oX", "-", target
                ]
            else:
                cmd = ["nmap", "-sS", "-T4", target]

            # Run scan in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_subprocess,
                cmd
            )

            # Parse XML output if available
            if result.get('stdout') and result['stdout'].startswith('<?xml'):
                return self._parse_nmap_xml(result['stdout'])

            return {
                'raw_output': result.get('stdout', ''),
                'error': result.get('stderr', ''),
                'return_code': result.get('return_code', 0)
            }

        except Exception as e:
            self.logger.error(f"Nmap scan error: {e}")
            return {'error': str(e)}

    async def run_nuclei_scan(self, target: str, templates: List[str] = None) -> Dict[str, Any]:
        """Run Nuclei vulnerability scan."""
        try:
            cmd = [
                "nuclei", "-u", target, "-j", "-c", str(EPYCExecutionConfig.NUCLEI_CONCURRENCY),
                "-rate-limit", "50", "-timeout", "10"
            ]

            if templates:
                for template in templates:
                    cmd.extend(["-t", template])
            else:
                cmd.extend(["-t", "cves/", "-t", "vulnerabilities/"])

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_subprocess,
                cmd
            )

            # Parse JSON output
            findings = []
            if result.get('stdout'):
                for line in result['stdout'].strip().split('\n'):
                    if line.strip():
                        try:
                            finding = json.loads(line)
                            findings.append(finding)
                        except json.JSONDecodeError:
                            continue

            return {
                'findings': findings,
                'total_findings': len(findings),
                'raw_output': result.get('stdout', ''),
                'error': result.get('stderr', '')
            }

        except Exception as e:
            self.logger.error(f"Nuclei scan error: {e}")
            return {'error': str(e)}

    async def run_zap_scan(self, target: str, scan_type: str = "baseline") -> Dict[str, Any]:
        """Run OWASP ZAP web application scan."""
        try:
            # ZAP baseline scan command
            cmd = [
                "zap-baseline.py", "-t", target, "-J", "/tmp/zap-report.json",
                "-m", "5"  # 5 minute timeout
            ]

            if scan_type == "full":
                cmd.extend(["-a", "-d"])  # Full scan with AJAX spider

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_subprocess,
                cmd
            )

            # Read ZAP JSON report
            zap_results = {}
            try:
                async with aiofiles.open('/tmp/zap-report.json', 'r') as f:
                    zap_content = await f.read()
                    zap_results = json.loads(zap_content)
            except Exception:
                pass

            return {
                'zap_results': zap_results,
                'raw_output': result.get('stdout', ''),
                'error': result.get('stderr', '')
            }

        except Exception as e:
            self.logger.error(f"ZAP scan error: {e}")
            return {'error': str(e)}

    def _run_subprocess(self, cmd: List[str]) -> Dict[str, Any]:
        """Run subprocess and return results."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Command timeout', 'return_code': -1}
        except Exception as e:
            return {'error': str(e), 'return_code': -1}

    def _parse_nmap_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse Nmap XML output."""
        try:
            root = ET.fromstring(xml_content)
            results = {
                'scan_summary': {},
                'hosts': []
            }

            # Parse scan summary
            for runstats in root.findall('runstats'):
                for finished in runstats.findall('finished'):
                    results['scan_summary']['elapsed'] = finished.get('elapsed')
                    results['scan_summary']['summary'] = finished.get('summary')

            # Parse hosts
            for host in root.findall('host'):
                host_info = {
                    'addresses': [],
                    'ports': [],
                    'os': {},
                    'scripts': []
                }

                # Get addresses
                for address in host.findall('address'):
                    host_info['addresses'].append({
                        'addr': address.get('addr'),
                        'addrtype': address.get('addrtype')
                    })

                # Get ports
                for ports in host.findall('ports'):
                    for port in ports.findall('port'):
                        port_info = {
                            'portid': port.get('portid'),
                            'protocol': port.get('protocol'),
                            'state': port.find('state').get('state') if port.find('state') is not None else 'unknown'
                        }

                        # Get service info
                        service = port.find('service')
                        if service is not None:
                            port_info['service'] = {
                                'name': service.get('name'),
                                'product': service.get('product'),
                                'version': service.get('version')
                            }

                        host_info['ports'].append(port_info)

                results['hosts'].append(host_info)

            return results
        except Exception as e:
            return {'error': f'XML parsing error: {e}', 'raw_xml': xml_content}
