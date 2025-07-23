#!/usr/bin/env python3

import asyncio
import logging
import json
import subprocess
import aiohttp
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentTask, AgentResult
from .playwright_agent import PlaywrightAgent


class ScanIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


@dataclass
class AgentLoadMetrics:
    agent_id: str
    current_tasks: int
    success_rate: float
    avg_response_time: float
    cpu_usage: float
    memory_usage: float
    error_count: int
    last_updated: datetime


class ZAPAgent(BaseAgent):
    """OWASP ZAP integration for automated web security scanning"""
    
    def __init__(self, zap_proxy_port: int = 8080, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        self.zap_proxy_port = zap_proxy_port
        self.zap_api_key = config.get('zap_api_key', '') if config else ''
        self.zap_base_url = f"http://127.0.0.1:{zap_proxy_port}"
        self.session = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.VULNERABILITY_SCANNER

    def _initialize_capabilities(self):
        self.capabilities = [
            AgentCapability(
                name="passive_scan",
                description="Passive vulnerability scanning via proxy",
                required_tools=["zap"]
            ),
            AgentCapability(
                name="active_scan",
                description="Active vulnerability scanning with attacks",
                required_tools=["zap"]
            ),
            AgentCapability(
                name="spider_crawl",
                description="Web application crawling and mapping",
                required_tools=["zap"]
            ),
            AgentCapability(
                name="authentication",
                description="Authenticated scanning with session management",
                required_tools=["zap"]
            ),
            AgentCapability(
                name="ajax_spider",
                description="AJAX-aware web crawling",
                required_tools=["zap"]
            )
        ]

    async def _on_start(self):
        """Initialize ZAP connection"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        )
        
        # Test ZAP connection
        if await self._test_zap_connection():
            self.logger.info("Connected to ZAP successfully")
        else:
            self.logger.error("Failed to connect to ZAP")

    async def _on_stop(self):
        """Cleanup ZAP connection"""
        if self.session:
            await self.session.close()

    async def _test_zap_connection(self) -> bool:
        """Test connection to ZAP"""
        try:
            async with self.session.get(f"{self.zap_base_url}/JSON/core/view/version/") as response:
                if response.status == 200:
                    data = await response.json()
                    version = data.get('version', 'unknown')
                    self.logger.info(f"ZAP version: {version}")
                    return True
        except Exception as e:
            self.logger.error(f"ZAP connection test failed: {e}")
        
        return False

    async def _execute_task(self, task: AgentTask) -> AgentResult:
        if task.task_type == "passive_scan":
            return await self._passive_scan(task)
        elif task.task_type == "active_scan":
            return await self._active_scan(task)
        elif task.task_type == "spider_crawl":
            return await self._spider_crawl(task)
        elif task.task_type == "ajax_spider":
            return await self._ajax_spider(task)
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[f"Unknown task type: {task.task_type}"]
            )

    async def _passive_scan(self, task: AgentTask) -> AgentResult:
        """Perform passive vulnerability scan"""
        target = task.target
        
        try:
            # Access the URL to trigger passive scanning
            async with self.session.get(f"{self.zap_base_url}/JSON/core/action/accessUrl/",
                                       params={'url': target}) as response:
                if response.status != 200:
                    return AgentResult(
                        task_id=task.task_id,
                        success=False,
                        errors=[f"Failed to access target: {response.status}"]
                    )

            # Wait for passive scan to complete
            await asyncio.sleep(10)

            # Get alerts
            async with self.session.get(f"{self.zap_base_url}/JSON/core/view/alerts/",
                                       params={'baseurl': target}) as response:
                if response.status != 200:
                    return AgentResult(
                        task_id=task.task_id,
                        success=False,
                        errors=[f"Failed to get alerts: {response.status}"]
                    )

                data = await response.json()
                alerts = data.get('alerts', [])

                # Convert alerts to findings
                findings = []
                for alert in alerts:
                    finding = {
                        "title": alert.get('alert', 'Unknown vulnerability'),
                        "description": alert.get('description', ''),
                        "severity": self._map_zap_risk(alert.get('risk', 'Low')),
                        "confidence": alert.get('confidence', 'Medium'),
                        "url": alert.get('url', target),
                        "param": alert.get('param', ''),
                        "evidence": alert.get('evidence', ''),
                        "solution": alert.get('solution', ''),
                        "reference": alert.get('reference', ''),
                        "cwe_id": alert.get('cweid', ''),
                        "wasc_id": alert.get('wascid', '')
                    }
                    findings.append(finding)

                return AgentResult(
                    task_id=task.task_id,
                    success=True,
                    findings=findings,
                    confidence=0.8,
                    metadata={
                        "scan_type": "passive",
                        "alerts_count": len(alerts),
                        "target": target
                    }
                )

        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _active_scan(self, task: AgentTask) -> AgentResult:
        """Perform active vulnerability scan"""
        target = task.target
        intensity = task.parameters.get('intensity', ScanIntensity.MEDIUM)
        
        try:
            # Start active scan
            scan_params = {
                'url': target,
                'recurse': 'true',
                'inScopeOnly': 'false'
            }
            
            # Adjust scan policy based on intensity
            if intensity == ScanIntensity.LOW:
                scan_params['scanPolicyName'] = 'Light'
            elif intensity == ScanIntensity.HIGH:
                scan_params['scanPolicyName'] = 'Full'
            elif intensity == ScanIntensity.AGGRESSIVE:
                scan_params['scanPolicyName'] = 'Full'
                scan_params['method'] = 'POST'

            async with self.session.get(f"{self.zap_base_url}/JSON/ascan/action/scan/",
                                       params=scan_params) as response:
                if response.status != 200:
                    return AgentResult(
                        task_id=task.task_id,
                        success=False,
                        errors=[f"Failed to start active scan: {response.status}"]
                    )

                data = await response.json()
                scan_id = data.get('scan')

            # Wait for scan to complete
            while True:
                async with self.session.get(f"{self.zap_base_url}/JSON/ascan/view/status/",
                                           params={'scanId': scan_id}) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = int(data.get('status', 0))
                        
                        if status >= 100:  # Scan complete
                            break
                        
                        self.logger.debug(f"Active scan progress: {status}%")
                        await asyncio.sleep(10)
                    else:
                        break

            # Get scan results
            return await self._get_scan_results(task.task_id, target, "active")

        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _spider_crawl(self, task: AgentTask) -> AgentResult:
        """Crawl website using ZAP spider"""
        target = task.target
        max_depth = task.parameters.get('max_depth', 5)
        
        try:
            # Start spider
            async with self.session.get(f"{self.zap_base_url}/JSON/spider/action/scan/",
                                       params={
                                           'url': target,
                                           'maxChildren': '10',
                                           'recurse': 'true',
                                           'contextName': '',
                                           'subtreeOnly': 'false'
                                       }) as response:
                if response.status != 200:
                    return AgentResult(
                        task_id=task.task_id,
                        success=False,
                        errors=[f"Failed to start spider: {response.status}"]
                    )

                data = await response.json()
                scan_id = data.get('scan')

            # Wait for spider to complete
            while True:
                async with self.session.get(f"{self.zap_base_url}/JSON/spider/view/status/",
                                           params={'scanId': scan_id}) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = int(data.get('status', 0))
                        
                        if status >= 100:
                            break
                        
                        self.logger.debug(f"Spider progress: {status}%")
                        await asyncio.sleep(5)
                    else:
                        break

            # Get discovered URLs
            async with self.session.get(f"{self.zap_base_url}/JSON/spider/view/results/",
                                       params={'scanId': scan_id}) as response:
                if response.status == 200:
                    data = await response.json()
                    urls = data.get('results', [])
                    
                    return AgentResult(
                        task_id=task.task_id,
                        success=True,
                        data={
                            "discovered_urls": urls,
                            "url_count": len(urls)
                        },
                        confidence=0.9,
                        metadata={
                            "scan_type": "spider",
                            "target": target,
                            "max_depth": max_depth
                        }
                    )

        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    def _map_zap_risk(self, zap_risk: str) -> str:
        """Map ZAP risk levels to standard severity"""
        risk_mapping = {
            'High': 'high',
            'Medium': 'medium',
            'Low': 'low',
            'Informational': 'info'
        }
        return risk_mapping.get(zap_risk, 'low')

    async def _get_scan_results(self, task_id: str, target: str, scan_type: str) -> AgentResult:
        """Get results from completed scan"""
        try:
            async with self.session.get(f"{self.zap_base_url}/JSON/core/view/alerts/",
                                       params={'baseurl': target}) as response:
                if response.status == 200:
                    data = await response.json()
                    alerts = data.get('alerts', [])
                    
                    findings = []
                    for alert in alerts:
                        finding = {
                            "title": alert.get('alert', 'Unknown'),
                            "severity": self._map_zap_risk(alert.get('risk', 'Low')),
                            "description": alert.get('description', ''),
                            "url": alert.get('url', target),
                            "solution": alert.get('solution', ''),
                            "reference": alert.get('reference', '')
                        }
                        findings.append(finding)
                    
                    return AgentResult(
                        task_id=task_id,
                        success=True,
                        findings=findings,
                        confidence=0.85,
                        metadata={"scan_type": scan_type, "target": target}
                    )
        
        except Exception as e:
            return AgentResult(
                task_id=task_id,
                success=False,
                errors=[str(e)]
            )


class NucleiAgent(BaseAgent):
    """Nuclei-powered vulnerability scanner"""
    
    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        self.nuclei_path = config.get('nuclei_path', 'nuclei') if config else 'nuclei'
        self.templates_path = config.get('templates_path', '') if config else ''

    @property
    def agent_type(self) -> AgentType:
        return AgentType.VULNERABILITY_SCANNER

    def _initialize_capabilities(self):
        self.capabilities = [
            AgentCapability(
                name="cve_scan",
                description="Scan for known CVE vulnerabilities",
                required_tools=["nuclei"]
            ),
            AgentCapability(
                name="tech_detect",
                description="Technology detection and fingerprinting",
                required_tools=["nuclei"]
            ),
            AgentCapability(
                name="misconfig_scan",
                description="Configuration vulnerability scanning",
                required_tools=["nuclei"]
            ),
            AgentCapability(
                name="fuzzing_scan",
                description="Fuzzing-based vulnerability detection",
                required_tools=["nuclei"]
            ),
            AgentCapability(
                name="dns_scan",
                description="DNS-based vulnerability scanning",
                required_tools=["nuclei"]
            )
        ]

    async def _execute_task(self, task: AgentTask) -> AgentResult:
        if task.task_type == "cve_scan":
            return await self._cve_scan(task)
        elif task.task_type == "tech_detect":
            return await self._tech_detect(task)
        elif task.task_type == "misconfig_scan":
            return await self._misconfig_scan(task)
        elif task.task_type == "fuzzing_scan":
            return await self._fuzzing_scan(task)
        elif task.task_type == "dns_scan":
            return await self._dns_scan(task)
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[f"Unknown task type: {task.task_type}"]
            )

    async def _cve_scan(self, task: AgentTask) -> AgentResult:
        """Scan for CVE vulnerabilities"""
        target = task.target
        severity = task.parameters.get('severity', ['high', 'critical'])
        
        cmd = [
            self.nuclei_path,
            '-u', target,
            '-t', 'cves/',
            '-severity', ','.join(severity),
            '-json',
            '-silent',
            '-no-color'
        ]
        
        return await self._run_nuclei_command(task.task_id, cmd, "cve_scan")

    async def _tech_detect(self, task: AgentTask) -> AgentResult:
        """Technology detection scan"""
        target = task.target
        
        cmd = [
            self.nuclei_path,
            '-u', target,
            '-t', 'technologies/',
            '-json',
            '-silent',
            '-no-color'
        ]
        
        return await self._run_nuclei_command(task.task_id, cmd, "tech_detect")

    async def _misconfig_scan(self, task: AgentTask) -> AgentResult:
        """Misconfiguration scan"""
        target = task.target
        
        cmd = [
            self.nuclei_path,
            '-u', target,
            '-t', 'misconfiguration/',
            '-json',
            '-silent',
            '-no-color'
        ]
        
        return await self._run_nuclei_command(task.task_id, cmd, "misconfig_scan")

    async def _fuzzing_scan(self, task: AgentTask) -> AgentResult:
        """Fuzzing-based scan"""
        target = task.target
        
        cmd = [
            self.nuclei_path,
            '-u', target,
            '-t', 'fuzzing/',
            '-json',
            '-silent',
            '-no-color'
        ]
        
        return await self._run_nuclei_command(task.task_id, cmd, "fuzzing_scan")

    async def _dns_scan(self, task: AgentTask) -> AgentResult:
        """DNS-based scan"""
        target = task.target
        
        cmd = [
            self.nuclei_path,
            '-u', target,
            '-t', 'dns/',
            '-json',
            '-silent',
            '-no-color'
        ]
        
        return await self._run_nuclei_command(task.task_id, cmd, "dns_scan")

    async def _run_nuclei_command(self, task_id: str, cmd: List[str], scan_type: str) -> AgentResult:
        """Run Nuclei command and parse results"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            findings = []
            errors = []
            
            if stderr:
                stderr_text = stderr.decode()
                if "WARN" not in stderr_text and "INFO" not in stderr_text:
                    errors.append(stderr_text)
            
            if stdout:
                for line in stdout.decode().strip().split('\n'):
                    if not line.strip():
                        continue
                        
                    try:
                        result = json.loads(line)
                        
                        # Extract finding information
                        info = result.get('info', {})
                        finding = {
                            "title": info.get('name', 'Unknown vulnerability'),
                            "severity": info.get('severity', 'info'),
                            "description": info.get('description', ''),
                            "template_id": result.get('template-id', ''),
                            "matched_at": result.get('matched-at', ''),
                            "extracted_results": result.get('extracted-results', []),
                            "curl_command": result.get('curl-command', ''),
                            "tags": info.get('tags', []),
                            "reference": info.get('reference', []),
                            "classification": info.get('classification', {}),
                            "metadata": info.get('metadata', {})
                        }
                        findings.append(finding)
                        
                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
            
            return AgentResult(
                task_id=task_id,
                success=process.returncode == 0,
                findings=findings,
                errors=errors,
                confidence=0.9,
                metadata={
                    "scan_type": scan_type,
                    "nuclei_exit_code": process.returncode,
                    "findings_count": len(findings)
                }
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task_id,
                success=False,
                errors=[str(e)]
            )


class StealthPlaywrightAgent(PlaywrightAgent):
    """Enhanced Playwright agent with stealth capabilities and proxy rotation"""
    
    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        self.proxy_list = config.get('proxy_list', []) if config else []
        self.current_proxy_index = 0
        self.user_agent_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]

    async def _on_start(self):
        """Enhanced initialization with stealth features"""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            
            # Launch browser with stealth options
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions',
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--disable-background-networking',
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-backgrounding-occluded-windows'
                ]
            )
            
            # Create context with stealth settings
            context_options = {
                'user_agent': self._get_random_user_agent(),
                'viewport': {'width': 1920, 'height': 1080},
                'ignore_https_errors': True,
                'extra_http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                }
            }
            
            # Add proxy if available
            if self.proxy_list:
                proxy = self._get_current_proxy()
                context_options['proxy'] = proxy
            
            self.context = await self.browser.new_context(**context_options)
            
            # Add stealth scripts to all pages
            await self.context.add_init_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                // Mock plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Mock chrome object
                window.chrome = {
                    runtime: {},
                };
                
                // Mock permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            self.page = await self.context.new_page()
            await self.page.set_default_timeout(self.timeout)
            
            # Setup additional stealth measures
            await self._setup_stealth_measures()
            
            self.logger.info("Stealth Playwright agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stealth agent: {e}")
            raise

    async def _setup_stealth_measures(self):
        """Setup additional stealth measures"""
        # Random viewport sizes
        viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1536, 'height': 864},
            {'width': 1440, 'height': 900}
        ]
        
        import random
        viewport = random.choice(viewports)
        await self.page.set_viewport_size(viewport['width'], viewport['height'])

    def _get_random_user_agent(self) -> str:
        """Get random user agent"""
        import random
        return random.choice(self.user_agent_list)

    def _get_current_proxy(self) -> Dict[str, str]:
        """Get current proxy configuration"""
        if not self.proxy_list:
            return None
        
        proxy = self.proxy_list[self.current_proxy_index]
        return {
            'server': f"http://{proxy['host']}:{proxy['port']}",
            'username': proxy.get('username'),
            'password': proxy.get('password')
        }

    async def rotate_proxy(self):
        """Rotate to next proxy"""
        if not self.proxy_list:
            return
        
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        
        # Create new context with new proxy
        await self.context.close()
        
        context_options = {
            'user_agent': self._get_random_user_agent(),
            'viewport': {'width': 1920, 'height': 1080},
            'ignore_https_errors': True,
            'proxy': self._get_current_proxy()
        }
        
        self.context = await self.browser.new_context(**context_options)
        self.page = await self.context.new_page()
        
        await self._setup_stealth_measures()
        self.logger.info(f"Rotated to proxy {self.current_proxy_index}")

    async def _execute_task(self, task: AgentTask) -> AgentResult:
        """Execute task with automatic proxy rotation on errors"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                return await super()._execute_task(task)
            
            except Exception as e:
                self.logger.warning(f"Task failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # Rotate proxy and retry
                    await self.rotate_proxy()
                    await asyncio.sleep(2)  # Brief delay
                else:
                    return AgentResult(
                        task_id=task.task_id,
                        success=False,
                        errors=[f"Task failed after {max_retries} attempts: {str(e)}"]
                    )


class AgentLoadBalancer:
    """Intelligent load balancer for managing agent assignments"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_metrics: Dict[str, AgentLoadMetrics] = {}
        self.task_history: Dict[str, List[AgentResult]] = {}
        self.logger = logging.getLogger(__name__)

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the load balancer"""
        self.agents[agent.agent_id] = agent
        self.agent_metrics[agent.agent_id] = AgentLoadMetrics(
            agent_id=agent.agent_id,
            current_tasks=0,
            success_rate=0.5,
            avg_response_time=60.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            error_count=0,
            last_updated=datetime.utcnow()
        )
        self.task_history[agent.agent_id] = []
        
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_metrics[agent_id]
            del self.task_history[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")

    async def assign_task(self, task: AgentTask) -> Optional[str]:
        """Assign task to the best available agent"""
        suitable_agents = self._get_suitable_agents(task.task_type)
        
        if not suitable_agents:
            self.logger.warning(f"No suitable agents for task type: {task.task_type}")
            return None
        
        # Score agents based on multiple factors
        scored_agents = []
        for agent_id in suitable_agents:
            score = await self._calculate_agent_score(agent_id, task)
            scored_agents.append((agent_id, score))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select best agent
        best_agent_id = scored_agents[0][0]
        best_agent = self.agents[best_agent_id]
        
        # Assign task
        success = await best_agent.add_task(task)
        
        if success:
            # Update metrics
            self.agent_metrics[best_agent_id].current_tasks += 1
            self.logger.info(f"Assigned task {task.task_id} to agent {best_agent_id}")
            return best_agent_id
        else:
            self.logger.error(f"Failed to assign task to agent {best_agent_id}")
            return None

    def _get_suitable_agents(self, task_type: str) -> List[str]:
        """Get agents that can handle the task type"""
        suitable = []
        
        for agent_id, agent in self.agents.items():
            if agent.has_capability(task_type) or self._agent_supports_task_type(agent, task_type):
                suitable.append(agent_id)
        
        return suitable

    def _agent_supports_task_type(self, agent: BaseAgent, task_type: str) -> bool:
        """Check if agent supports task type based on agent type and capabilities"""
        agent_task_mapping = {
            'web_crawl': [AgentType.WEB_CRAWLER],
            'form_analysis': [AgentType.WEB_CRAWLER],
            'passive_scan': [AgentType.VULNERABILITY_SCANNER],
            'active_scan': [AgentType.VULNERABILITY_SCANNER],
            'cve_scan': [AgentType.VULNERABILITY_SCANNER],
            'tech_detect': [AgentType.VULNERABILITY_SCANNER],
            'spider_crawl': [AgentType.WEB_CRAWLER, AgentType.VULNERABILITY_SCANNER],
            'screenshot': [AgentType.WEB_CRAWLER],
            'dom_analysis': [AgentType.WEB_CRAWLER]
        }
        
        supported_types = agent_task_mapping.get(task_type, [])
        return agent.agent_type in supported_types

    async def _calculate_agent_score(self, agent_id: str, task: AgentTask) -> float:
        """Calculate score for agent assignment"""
        metrics = self.agent_metrics[agent_id]
        
        # Base score factors
        load_factor = max(0.1, 1.0 - (metrics.current_tasks / 10.0))  # Prefer less loaded agents
        success_factor = metrics.success_rate
        speed_factor = max(0.1, 1.0 - (metrics.avg_response_time / 300.0))  # Prefer faster agents
        reliability_factor = max(0.1, 1.0 - (metrics.error_count / 100.0))  # Prefer reliable agents
        
        # Task-specific factors
        capability_factor = await self._get_capability_factor(agent_id, task.task_type)
        
        # Weighted score
        score = (
            load_factor * 0.3 +
            success_factor * 0.25 +
            speed_factor * 0.2 +
            reliability_factor * 0.15 +
            capability_factor * 0.1
        )
        
        return score

    async def _get_capability_factor(self, agent_id: str, task_type: str) -> float:
        """Get capability-specific factor for agent"""
        agent = self.agents[agent_id]
        
        # Check if agent has specific capability
        for capability in agent.capabilities:
            if capability.name == task_type:
                return capability.success_rate if capability.success_rate > 0 else 0.5
        
        return 0.5  # Default factor for generic capabilities

    async def update_agent_metrics(self, agent_id: str, task_result: AgentResult):
        """Update agent performance metrics based on task result"""
        if agent_id not in self.agent_metrics:
            return
        
        metrics = self.agent_metrics[agent_id]
        
        # Update current tasks count
        metrics.current_tasks = max(0, metrics.current_tasks - 1)
        
        # Update success rate
        history = self.task_history[agent_id]
        history.append(task_result)
        
        # Keep last 100 results
        if len(history) > 100:
            history = history[-100:]
            self.task_history[agent_id] = history
        
        # Calculate success rate
        successful_tasks = sum(1 for result in history if result.success)
        metrics.success_rate = successful_tasks / len(history) if history else 0.5
        
        # Update average response time
        response_times = [r.execution_time for r in history if r.execution_time > 0]
        metrics.avg_response_time = sum(response_times) / len(response_times) if response_times else 60.0
        
        # Update error count
        if not task_result.success:
            metrics.error_count += 1
        
        metrics.last_updated = datetime.utcnow()
        
        # Update agent capability stats
        agent = self.agents[agent_id]
        await agent.update_capability_stats(
            task_result.metadata.get('scan_type', 'unknown'),
            task_result.success,
            task_result.execution_time
        )

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        active_agents = len([a for a in self.agents.values() if a.status.value == 'idle' or a.status.value == 'running'])
        total_tasks = sum(metrics.current_tasks for metrics in self.agent_metrics.values())
        avg_success_rate = sum(metrics.success_rate for metrics in self.agent_metrics.values()) / len(self.agent_metrics) if self.agent_metrics else 0
        
        agent_types = {}
        for agent in self.agents.values():
            agent_type = agent.agent_type.value
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_agents,
            'total_current_tasks': total_tasks,
            'average_success_rate': avg_success_rate,
            'agent_types': agent_types,
            'agents_by_performance': [
                {
                    'agent_id': agent_id,
                    'agent_type': self.agents[agent_id].agent_type.value,
                    'success_rate': metrics.success_rate,
                    'current_tasks': metrics.current_tasks,
                    'avg_response_time': metrics.avg_response_time
                }
                for agent_id, metrics in self.agent_metrics.items()
            ]
        }


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_multi_engine_agents():
        """Demo the multi-engine agent system"""
        
        # Create load balancer
        load_balancer = AgentLoadBalancer()
        
        # Create agents
        zap_agent = ZAPAgent()
        nuclei_agent = NucleiAgent()
        stealth_agent = StealthPlaywrightAgent()
        
        # Register agents
        load_balancer.register_agent(zap_agent)
        load_balancer.register_agent(nuclei_agent)
        load_balancer.register_agent(stealth_agent)
        
        # Start agents
        await zap_agent.start()
        await nuclei_agent.start()
        await stealth_agent.start()
        
        print("Multi-engine agent system demo started")
        
        # Create test tasks
        test_tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type="cve_scan",
                target="https://example.com"
            ),
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type="web_crawl",
                target="https://example.com"
            ),
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type="passive_scan",
                target="https://example.com"
            )
        ]
        
        # Assign tasks
        for task in test_tasks:
            assigned_agent = await load_balancer.assign_task(task)
            if assigned_agent:
                print(f"Task {task.task_id} ({task.task_type}) assigned to {assigned_agent}")
            else:
                print(f"Failed to assign task {task.task_id}")
        
        # Wait a bit for processing
        await asyncio.sleep(5)
        
        # Show stats
        stats = load_balancer.get_load_balancer_stats()
        print(f"\nLoad Balancer Stats:")
        print(f"  Total Agents: {stats['total_agents']}")
        print(f"  Active Agents: {stats['active_agents']}")
        print(f"  Current Tasks: {stats['total_current_tasks']}")
        print(f"  Average Success Rate: {stats['average_success_rate']:.2f}")
        
        # Stop agents
        await zap_agent.stop()
        await nuclei_agent.stop()
        await stealth_agent.stop()
        
        print("Multi-engine agent demo completed")
    
    if "--demo" in sys.argv:
        asyncio.run(demo_multi_engine_agents())
    else:
        print("XORB Multi-Engine Agents")
        print("Usage: python multi_engine_agents.py --demo")