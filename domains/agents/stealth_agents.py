#!/usr/bin/env python3

import asyncio
import random
import time
import logging
import json
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    from playwright_stealth import stealth_async
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright stealth not available. Install with: pip install playwright playwright-stealth")

try:
    import httpx
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("HTTP libraries not available. Install with: pip install httpx requests")

from .base_agent import BaseAgent, AgentTask, AgentResult, AgentCapability


@dataclass
class StealthConfig:
    """Configuration for stealth operations."""
    user_agent_rotation: bool = True
    proxy_rotation: bool = True
    request_delay_min: float = 1.0
    request_delay_max: float = 5.0
    header_randomization: bool = True
    javascript_evasion: bool = True
    fingerprint_randomization: bool = True
    viewport_randomization: bool = True
    timezone_randomization: bool = True
    language_randomization: bool = True


@dataclass
class ProxyConfig:
    """Proxy configuration for anonymity."""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: str = 'http'  # 'http', 'socks5'
    country: Optional[str] = None
    reliability_score: float = 1.0


class UserAgentRotator:
    """Intelligent user agent rotation system."""
    
    def __init__(self):
        self.user_agents = [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
            
            # Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
            
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        ]
        
        self.mobile_user_agents = [
            # Mobile Chrome
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Android 14; Mobile; rv:120.0) Gecko/120.0 Firefox/120.0",
            "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        ]
        
        self.usage_stats = {ua: 0 for ua in self.user_agents + self.mobile_user_agents}
    
    def get_random_user_agent(self, mobile: bool = False) -> str:
        """Get a random user agent with usage balancing."""
        available_agents = self.mobile_user_agents if mobile else self.user_agents
        
        # Prefer less-used user agents
        sorted_agents = sorted(available_agents, key=lambda ua: self.usage_stats[ua])
        
        # Select from top 50% least used
        selection_pool = sorted_agents[:len(sorted_agents)//2 + 1]
        selected = random.choice(selection_pool)
        
        self.usage_stats[selected] += 1
        return selected


class ProxyRotator:
    """Intelligent proxy rotation system."""
    
    def __init__(self):
        self.proxies: List[ProxyConfig] = []
        self.current_proxy_index = 0
        self.proxy_performance = {}
        self.logger = logging.getLogger(__name__)
    
    def add_proxy(self, proxy: ProxyConfig):
        """Add a proxy to the rotation pool."""
        self.proxies.append(proxy)
        proxy_key = f"{proxy.host}:{proxy.port}"
        self.proxy_performance[proxy_key] = {
            'success_count': 0,
            'failure_count': 0,
            'avg_response_time': 0.0,
            'last_used': None
        }
    
    def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get the next proxy in rotation."""
        if not self.proxies:
            return None
        
        # Sort proxies by performance score
        sorted_proxies = sorted(
            self.proxies,
            key=lambda p: self._calculate_proxy_score(p),
            reverse=True
        )
        
        # Select from top performers
        selection_pool = sorted_proxies[:max(1, len(sorted_proxies)//2)]
        return random.choice(selection_pool)
    
    def _calculate_proxy_score(self, proxy: ProxyConfig) -> float:
        """Calculate proxy performance score."""
        proxy_key = f"{proxy.host}:{proxy.port}"
        stats = self.proxy_performance.get(proxy_key, {})
        
        success_rate = 0.5  # Default
        if stats.get('success_count', 0) + stats.get('failure_count', 0) > 0:
            success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])
        
        # Factor in response time (lower is better)
        response_time_factor = 1.0
        if stats.get('avg_response_time', 0) > 0:
            response_time_factor = 1.0 / (1.0 + stats['avg_response_time'] / 10.0)
        
        # Factor in how recently used (prefer less recently used)
        recency_factor = 1.0
        if stats.get('last_used'):
            hours_since_use = (datetime.utcnow() - stats['last_used']).total_seconds() / 3600
            recency_factor = min(2.0, 1.0 + hours_since_use / 24.0)
        
        return proxy.reliability_score * success_rate * response_time_factor * recency_factor
    
    def record_proxy_result(self, proxy: ProxyConfig, success: bool, response_time: float):
        """Record proxy performance metrics."""
        proxy_key = f"{proxy.host}:{proxy.port}"
        stats = self.proxy_performance[proxy_key]
        
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
        
        # Update average response time
        total_requests = stats['success_count'] + stats['failure_count']
        if total_requests > 1:
            stats['avg_response_time'] = (
                (stats['avg_response_time'] * (total_requests - 1) + response_time) / total_requests
            )
        else:
            stats['avg_response_time'] = response_time
        
        stats['last_used'] = datetime.utcnow()


class AntiDetectionEngine:
    """Advanced anti-detection techniques."""
    
    def __init__(self, config: StealthConfig):
        self.config = config
        self.user_agent_rotator = UserAgentRotator()
        self.proxy_rotator = ProxyRotator()
        self.logger = logging.getLogger(__name__)
        
        # Add some example proxies (in production, load from config)
        self._initialize_default_proxies()
    
    def _initialize_default_proxies(self):
        """Initialize with some default proxy configurations."""
        # In production, these would be loaded from configuration
        default_proxies = [
            ProxyConfig("proxy1.example.com", 8080, reliability_score=0.9),
            ProxyConfig("proxy2.example.com", 8080, reliability_score=0.8),
            ProxyConfig("proxy3.example.com", 8080, reliability_score=0.7),
        ]
        
        for proxy in default_proxies:
            self.proxy_rotator.add_proxy(proxy)
    
    async def apply_stealth_measures(self, page: Page):
        """Apply comprehensive stealth measures to a Playwright page."""
        if not PLAYWRIGHT_AVAILABLE:
            return
        
        try:
            # Apply playwright-stealth
            await stealth_async(page)
            
            # Randomize viewport
            if self.config.viewport_randomization:
                await self._randomize_viewport(page)
            
            # Randomize timezone
            if self.config.timezone_randomization:
                await self._randomize_timezone(page)
            
            # Randomize language
            if self.config.language_randomization:
                await self._randomize_language(page)
            
            # Override WebGL fingerprinting
            await self._override_webgl_fingerprinting(page)
            
            # Override canvas fingerprinting
            await self._override_canvas_fingerprinting(page)
            
            # Override audio fingerprinting
            await self._override_audio_fingerprinting(page)
            
        except Exception as e:
            self.logger.warning(f"Failed to apply some stealth measures: {e}")
    
    async def _randomize_viewport(self, page: Page):
        """Randomize browser viewport."""
        viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1536, "height": 864},
            {"width": 1440, "height": 900},
            {"width": 1280, "height": 720},
        ]
        
        viewport = random.choice(viewports)
        await page.set_viewport_size(**viewport)
    
    async def _randomize_timezone(self, page: Page):
        """Randomize browser timezone."""
        timezones = [
            "America/New_York",
            "America/Los_Angeles", 
            "Europe/London",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        
        timezone = random.choice(timezones)
        await page.emulate_timezone(timezone)
    
    async def _randomize_language(self, page: Page):
        """Randomize browser language."""
        languages = ["en-US", "en-GB", "de-DE", "fr-FR", "es-ES"]
        language = random.choice(languages)
        await page.set_extra_http_headers({"Accept-Language": f"{language},en;q=0.9"})
    
    async def _override_webgl_fingerprinting(self, page: Page):
        """Override WebGL fingerprinting."""
        await page.add_init_script("""
            Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
                value: function(type, attributes) {
                    if (type === 'webgl' || type === 'webgl2') {
                        const context = HTMLCanvasElement.prototype.getContext.call(this, type, attributes);
                        if (context) {
                            // Spoof WebGL parameters
                            const getParameter = context.getParameter;
                            context.getParameter = function(parameter) {
                                if (parameter === context.RENDERER) {
                                    return 'Intel Iris OpenGL Engine';
                                }
                                if (parameter === context.VENDOR) {
                                    return 'Intel Inc.';
                                }
                                return getParameter.call(this, parameter);
                            };
                        }
                        return context;
                    }
                    return HTMLCanvasElement.prototype.getContext.call(this, type, attributes);
                }
            });
        """)
    
    async def _override_canvas_fingerprinting(self, page: Page):
        """Override canvas fingerprinting."""
        await page.add_init_script("""
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function(...args) {
                // Add slight noise to canvas data
                const ctx = this.getContext('2d');
                if (ctx) {
                    const imageData = ctx.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        if (Math.random() < 0.001) {
                            imageData.data[i] = Math.floor(Math.random() * 256);
                        }
                    }
                    ctx.putImageData(imageData, 0, 0);
                }
                return originalToDataURL.apply(this, args);
            };
        """)
    
    async def _override_audio_fingerprinting(self, page: Page):
        """Override audio context fingerprinting."""
        await page.add_init_script("""
            const audioContext = window.AudioContext || window.webkitAudioContext;
            if (audioContext) {
                const originalCreateAnalyser = audioContext.prototype.createAnalyser;
                audioContext.prototype.createAnalyser = function() {
                    const analyser = originalCreateAnalyser.call(this);
                    const originalGetFrequencyData = analyser.getFrequencyData;
                    analyser.getFrequencyData = function(array) {
                        originalGetFrequencyData.call(this, array);
                        // Add noise to frequency data
                        for (let i = 0; i < array.length; i++) {
                            array[i] = array[i] + (Math.random() - 0.5) * 0.1;
                        }
                    };
                    return analyser;
                };
            }
        """)
    
    def get_random_headers(self, mobile: bool = False) -> Dict[str, str]:
        """Generate randomized HTTP headers."""
        user_agent = self.user_agent_rotator.get_random_user_agent(mobile)
        
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice([
                "en-US,en;q=0.5",
                "en-GB,en;q=0.9",
                "de-DE,de;q=0.9,en;q=0.8",
                "fr-FR,fr;q=0.9,en;q=0.8"
            ]),
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": random.choice(["1", "0"]),
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Randomly add optional headers
        if random.random() < 0.7:
            headers["Cache-Control"] = random.choice(["no-cache", "max-age=0"])
        
        if random.random() < 0.3:
            headers["Sec-Fetch-Dest"] = random.choice(["document", "empty"])
            headers["Sec-Fetch-Mode"] = random.choice(["navigate", "cors"])
            headers["Sec-Fetch-Site"] = random.choice(["none", "same-origin"])
        
        return headers
    
    async def add_random_delay(self):
        """Add randomized delay between requests."""
        delay = random.uniform(self.config.request_delay_min, self.config.request_delay_max)
        await asyncio.sleep(delay)


class StealthPlaywrightAgent(BaseAgent):
    """Advanced stealth browser agent with comprehensive anti-detection."""
    
    def __init__(self, agent_id: str, stealth_config: Optional[StealthConfig] = None):
        capabilities = [
            AgentCapability.WEB_CRAWLING,
            AgentCapability.VULNERABILITY_SCANNING,
            AgentCapability.RECONNAISSANCE
        ]
        
        super().__init__(agent_id, "stealth_playwright", capabilities)
        
        self.stealth_config = stealth_config or StealthConfig()
        self.anti_detection = AntiDetectionEngine(self.stealth_config)
        self.browser = None
        self.contexts = []
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the stealth browser agent."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available. Install with: pip install playwright playwright-stealth")
        
        await super().initialize()
        
        try:
            # Launch browser with stealth settings
            playwright = await async_playwright().start()
            
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-ipc-flooding-protection",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-background-timer-throttling",
                    "--force-color-profile=srgb",
                    "--disable-web-security",
                    "--disable-features=TranslateUI",
                ]
            )
            
            self.logger.info("Stealth Playwright agent initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stealth agent: {e}")
            raise
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a task with stealth measures."""
        self.logger.info(f"Executing stealth task: {task.task_type}")
        
        start_time = time.time()
        
        try:
            # Create isolated browser context
            context = await self._create_stealth_context()
            
            if task.task_type == "web_crawl":
                result = await self._stealth_web_crawl(context, task)
            elif task.task_type == "vulnerability_scan":
                result = await self._stealth_vulnerability_scan(context, task)
            elif task.task_type == "reconnaissance":
                result = await self._stealth_reconnaissance(context, task)
            else:
                result = AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    success=False,
                    error="Unsupported task type for stealth agent",
                    execution_time=time.time() - start_time
                )
            
            # Clean up context
            await context.close()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stealth task execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _create_stealth_context(self) -> BrowserContext:
        """Create a browser context with stealth settings."""
        # Get random user agent
        user_agent = self.anti_detection.user_agent_rotator.get_random_user_agent()
        
        # Get proxy if available
        proxy = self.anti_detection.proxy_rotator.get_next_proxy()
        proxy_config = None
        if proxy:
            proxy_config = {
                "server": f"{proxy.proxy_type}://{proxy.host}:{proxy.port}"
            }
            if proxy.username and proxy.password:
                proxy_config["username"] = proxy.username
                proxy_config["password"] = proxy.password
        
        # Create context with stealth settings
        context = await self.browser.new_context(
            user_agent=user_agent,
            proxy=proxy_config,
            java_script_enabled=True,
            accept_downloads=False,
            ignore_https_errors=True,
            extra_http_headers=self.anti_detection.get_random_headers()
        )
        
        self.contexts.append(context)
        return context
    
    async def _stealth_web_crawl(self, context: BrowserContext, task: AgentTask) -> AgentResult:
        """Perform stealth web crawling."""
        target_url = task.target
        findings = []
        
        try:
            page = await context.new_page()
            
            # Apply stealth measures
            await self.anti_detection.apply_stealth_measures(page)
            
            # Add random delay
            await self.anti_detection.add_random_delay()
            
            # Navigate to target
            response = await page.goto(target_url, wait_until="networkidle", timeout=30000)
            
            if response:
                # Extract page information
                title = await page.title()
                url = page.url
                
                # Find forms
                forms = await page.query_selector_all("form")
                form_data = []
                for form in forms:
                    action = await form.get_attribute("action")
                    method = await form.get_attribute("method") or "GET"
                    
                    # Find inputs
                    inputs = await form.query_selector_all("input")
                    input_data = []
                    for inp in inputs:
                        name = await inp.get_attribute("name")
                        input_type = await inp.get_attribute("type") or "text"
                        if name:
                            input_data.append({"name": name, "type": input_type})
                    
                    form_data.append({
                        "action": action,
                        "method": method.upper(),
                        "inputs": input_data
                    })
                
                # Find links
                links = await page.query_selector_all("a[href]")
                link_data = []
                for link in links[:20]:  # Limit to avoid overwhelming
                    href = await link.get_attribute("href")
                    text = await link.inner_text()
                    if href:
                        link_data.append({"href": href, "text": text[:100]})
                
                # Check for interesting technologies
                tech_indicators = await self._detect_technologies(page)
                
                findings.append({
                    "type": "web_crawl_result",
                    "url": url,
                    "title": title,
                    "forms": form_data,
                    "links": link_data,
                    "technologies": tech_indicators,
                    "response_status": response.status
                })
            
            await page.close()
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=True,
                findings=findings,
                execution_time=time.time() - task.created_at.timestamp()
            )
            
        except Exception as e:
            self.logger.error(f"Stealth web crawl failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - task.created_at.timestamp()
            )
    
    async def _detect_technologies(self, page: Page) -> List[Dict[str, Any]]:
        """Detect technologies used by the target."""
        technologies = []
        
        try:
            # Check for common frameworks in page source
            content = await page.content()
            
            tech_patterns = {
                "WordPress": ["wp-content", "wp-includes", "wordpress"],
                "Django": ["csrfmiddlewaretoken", "django"],
                "React": ["react", "_react"],
                "Angular": ["ng-", "angular"],
                "Vue.js": ["vue", "v-"],
                "jQuery": ["jquery", "jQuery"],
                "Bootstrap": ["bootstrap"],
                "Laravel": ["laravel_session", "_token"],
                "Express.js": ["express"],
                "ASP.NET": ["viewstate", "__VIEWSTATE"]
            }
            
            content_lower = content.lower()
            for tech, patterns in tech_patterns.items():
                if any(pattern.lower() in content_lower for pattern in patterns):
                    technologies.append({
                        "name": tech,
                        "confidence": 0.7,
                        "detection_method": "content_analysis"
                    })
            
            # Check response headers
            headers = await page.evaluate("() => { return Object.fromEntries(performance.getEntries()[0].responseStart ? [] : Object.entries(document.head.querySelector('meta[http-equiv]') || {})); }")
            
            if headers:
                if 'server' in str(headers).lower():
                    technologies.append({
                        "name": "Server Technology",
                        "value": str(headers),
                        "confidence": 0.9,
                        "detection_method": "headers"
                    })
            
        except Exception as e:
            self.logger.warning(f"Technology detection failed: {e}")
        
        return technologies
    
    async def _stealth_vulnerability_scan(self, context: BrowserContext, task: AgentTask) -> AgentResult:
        """Perform stealth vulnerability scanning."""
        # Implementation would include specific vulnerability checks
        # This is a simplified version focusing on common web vulnerabilities
        
        findings = []
        target_url = task.target
        
        try:
            page = await context.new_page()
            await self.anti_detection.apply_stealth_measures(page)
            
            # Test for SQL injection in forms
            await page.goto(target_url)
            forms = await page.query_selector_all("form")
            
            for form in forms:
                # Simple SQL injection test
                inputs = await form.query_selector_all("input[type='text'], input[type='search']")
                for inp in inputs:
                    test_payload = "' OR '1'='1"
                    await inp.fill(test_payload)
                    
                    # Submit and check response
                    await form.submit_form()
                    await page.wait_for_load_state("networkidle")
                    
                    content = await page.content()
                    if any(indicator in content.lower() for indicator in ["sql error", "mysql", "sqlite", "postgresql"]):
                        findings.append({
                            "type": "potential_sql_injection",
                            "severity": "high",
                            "url": page.url,
                            "description": "Potential SQL injection vulnerability detected",
                            "payload": test_payload
                        })
            
            await page.close()
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=True,
                findings=findings,
                execution_time=time.time() - task.created_at.timestamp()
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - task.created_at.timestamp()
            )
    
    async def _stealth_reconnaissance(self, context: BrowserContext, task: AgentTask) -> AgentResult:
        """Perform stealth reconnaissance."""
        findings = []
        target_url = task.target
        
        try:
            page = await context.new_page()
            await self.anti_detection.apply_stealth_measures(page)
            
            # Visit target and gather information
            response = await page.goto(target_url)
            
            # Check robots.txt
            robots_url = f"{target_url.rstrip('/')}/robots.txt"
            try:
                robots_response = await page.goto(robots_url)
                if robots_response.status == 200:
                    robots_content = await page.content()
                    findings.append({
                        "type": "robots_txt",
                        "content": robots_content,
                        "url": robots_url
                    })
            except:
                pass
            
            # Check sitemap.xml
            sitemap_url = f"{target_url.rstrip('/')}/sitemap.xml"
            try:
                sitemap_response = await page.goto(sitemap_url)
                if sitemap_response.status == 200:
                    findings.append({
                        "type": "sitemap_found",
                        "url": sitemap_url
                    })
            except:
                pass
            
            # Check common admin paths
            admin_paths = ["/admin", "/administrator", "/wp-admin", "/login", "/dashboard"]
            for path in admin_paths:
                try:
                    admin_url = f"{target_url.rstrip('/')}{path}"
                    admin_response = await page.goto(admin_url)
                    if admin_response.status == 200:
                        findings.append({
                            "type": "admin_panel_found",
                            "url": admin_url,
                            "status": admin_response.status
                        })
                    await self.anti_detection.add_random_delay()
                except:
                    continue
            
            await page.close()
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=True,
                findings=findings,
                execution_time=time.time() - task.created_at.timestamp()
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - task.created_at.timestamp()
            )
    
    async def cleanup(self):
        """Clean up browser resources."""
        try:
            # Close all contexts
            for context in self.contexts:
                await context.close()
            
            # Close browser
            if self.browser:
                await self.browser.close()
            
            await super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class StealthHTTPAgent(BaseAgent):
    """HTTP-based stealth agent with advanced evasion techniques."""
    
    def __init__(self, agent_id: str, stealth_config: Optional[StealthConfig] = None):
        capabilities = [
            AgentCapability.VULNERABILITY_SCANNING,
            AgentCapability.RECONNAISSANCE
        ]
        
        super().__init__(agent_id, "stealth_http", capabilities)
        
        self.stealth_config = stealth_config or StealthConfig()
        self.anti_detection = AntiDetectionEngine(self.stealth_config)
        self.session = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the stealth HTTP agent."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("HTTP libraries not available. Install with: pip install httpx requests")
        
        await super().initialize()
        
        # Create session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger.info("Stealth HTTP agent initialized")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute HTTP-based stealth task."""
        if task.task_type == "http_scan":
            return await self._stealth_http_scan(task)
        elif task.task_type == "directory_enumeration":
            return await self._stealth_directory_enum(task)
        else:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error="Unsupported task type for HTTP stealth agent"
            )
    
    async def _stealth_http_scan(self, task: AgentTask) -> AgentResult:
        """Perform stealth HTTP scanning."""
        findings = []
        target_url = task.target
        
        try:
            # Test various HTTP methods
            methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
            
            for method in methods:
                try:
                    headers = self.anti_detection.get_random_headers()
                    
                    # Get proxy if available
                    proxy = self.anti_detection.proxy_rotator.get_next_proxy()
                    proxies = None
                    if proxy:
                        proxy_url = f"{proxy.proxy_type}://{proxy.host}:{proxy.port}"
                        if proxy.username and proxy.password:
                            proxy_url = f"{proxy.proxy_type}://{proxy.username}:{proxy.password}@{proxy.host}:{proxy.port}"
                        proxies = {"http": proxy_url, "https": proxy_url}
                    
                    response = self.session.request(
                        method,
                        target_url,
                        headers=headers,
                        proxies=proxies,
                        timeout=10,
                        allow_redirects=False
                    )
                    
                    findings.append({
                        "type": "http_method_response",
                        "method": method,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content_length": len(response.content)
                    })
                    
                    # Add delay between requests
                    await self.anti_detection.add_random_delay()
                    
                except Exception as e:
                    self.logger.debug(f"HTTP method {method} failed: {e}")
                    continue
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=True,
                findings=findings,
                execution_time=time.time() - task.created_at.timestamp()
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - task.created_at.timestamp()
            )
    
    async def _stealth_directory_enum(self, task: AgentTask) -> AgentResult:
        """Perform stealth directory enumeration."""
        findings = []
        target_url = task.target.rstrip('/')
        
        # Common directories to check
        directories = [
            "/admin", "/administrator", "/wp-admin", "/phpmyadmin",
            "/backup", "/backups", "/test", "/tests", "/dev",
            "/api", "/v1", "/v2", "/docs", "/documentation",
            "/config", "/conf", "/tmp", "/temp", "/uploads",
            "/assets", "/static", "/public", "/private"
        ]
        
        try:
            for directory in directories:
                try:
                    url = f"{target_url}{directory}"
                    headers = self.anti_detection.get_random_headers()
                    
                    response = self.session.get(
                        url,
                        headers=headers,
                        timeout=5,
                        allow_redirects=False
                    )
                    
                    if response.status_code in [200, 301, 302, 403]:
                        findings.append({
                            "type": "directory_found",
                            "url": url,
                            "status_code": response.status_code,
                            "content_length": len(response.content)
                        })
                    
                    # Add delay between requests
                    await self.anti_detection.add_random_delay()
                    
                except Exception as e:
                    self.logger.debug(f"Directory check for {directory} failed: {e}")
                    continue
            
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=True,
                findings=findings,
                execution_time=time.time() - task.created_at.timestamp()
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - task.created_at.timestamp()
            )
    
    async def cleanup(self):
        """Clean up HTTP session."""
        if self.session:
            self.session.close()
        await super().cleanup()