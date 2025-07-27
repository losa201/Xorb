#!/usr/bin/env python3

import asyncio
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError

from agents.base_agent import BaseAgent, AgentType, AgentCapability, AgentTask, AgentResult
from knowledge_fabric.atom import KnowledgeAtom, AtomType, Source


class PlaywrightAgent(BaseAgent):
    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
        self.user_agent = self.config.get('user_agent', 
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
        self.timeout = self.config.get('timeout', 30000)
        self.headless = self.config.get('headless', True)
        
        self.visited_urls: set = set()
        self.discovered_endpoints: List[Dict[str, Any]] = []
        self.form_data: List[Dict[str, Any]] = []

    @property
    def agent_type(self) -> AgentType:
        return AgentType.WEB_CRAWLER

    def _initialize_capabilities(self):
        self.capabilities = [
            AgentCapability(
                name="web_crawling",
                description="Navigate and crawl web applications",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="form_discovery",
                description="Discover and analyze web forms",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="endpoint_discovery",
                description="Discover API endpoints and URLs",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="dom_analysis",
                description="Analyze DOM structure and content",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="javascript_execution",
                description="Execute JavaScript in browser context",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="screenshot_capture",
                description="Capture screenshots of web pages",
                required_tools=["playwright"]
            ),
            AgentCapability(
                name="cookie_analysis",
                description="Analyze cookies and session management",
                required_tools=["playwright"]
            )
        ]

    async def _on_start(self):
        try:
            self.playwright = await async_playwright().start()
            
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            self.context = await self.browser.new_context(
                user_agent=self.user_agent,
                viewport=self.viewport,
                ignore_https_errors=True,
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
            
            self.page = await self.context.new_page()
            await self.page.set_default_timeout(self.timeout)
            
            # Setup request/response interceptors
            await self._setup_interceptors()
            
            self.logger.info("Playwright browser initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise

    async def _on_stop(self):
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            self.logger.info("Playwright browser closed")
            
        except Exception as e:
            self.logger.error(f"Error closing browser: {e}")

    async def _execute_task(self, task: AgentTask) -> AgentResult:
        if not await self._validate_task(task):
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=["Task validation failed"]
            )
        
        try:
            if task.task_type == "web_crawl":
                return await self._crawl_website(task)
            elif task.task_type == "form_analysis":
                return await self._analyze_forms(task)
            elif task.task_type == "endpoint_discovery":
                return await self._discover_endpoints(task)
            elif task.task_type == "dom_analysis":
                return await self._analyze_dom(task)
            elif task.task_type == "screenshot":
                return await self._capture_screenshot(task)
            elif task.task_type == "javascript_execution":
                return await self._execute_javascript(task)
            else:
                return AgentResult(
                    task_id=task.task_id,
                    success=False,
                    errors=[f"Unknown task type: {task.task_type}"]
                )
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _crawl_website(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        max_depth = task.parameters.get('max_depth', 3)
        max_pages = task.parameters.get('max_pages', 50)
        
        crawl_results = {
            "visited_urls": [],
            "discovered_urls": [],
            "forms": [],
            "endpoints": [],
            "technologies": [],
            "cookies": [],
            "errors": []
        }
        
        findings = []
        
        try:
            await self._crawl_recursive(target_url, 0, max_depth, max_pages, crawl_results)
            
            # Generate findings based on crawl results
            if crawl_results["forms"]:
                findings.append({
                    "type": "forms_discovered",
                    "severity": "info",
                    "count": len(crawl_results["forms"]),
                    "description": f"Discovered {len(crawl_results['forms'])} forms"
                })
            
            if crawl_results["endpoints"]:
                findings.append({
                    "type": "endpoints_discovered", 
                    "severity": "info",
                    "count": len(crawl_results["endpoints"]),
                    "description": f"Discovered {len(crawl_results['endpoints'])} API endpoints"
                })
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data=crawl_results,
                findings=findings,
                confidence=0.8
            )
            
        except Exception as e:
            crawl_results["errors"].append(str(e))
            return AgentResult(
                task_id=task.task_id,
                success=False,
                data=crawl_results,
                errors=[str(e)]
            )

    async def _crawl_recursive(self, url: str, depth: int, max_depth: int, max_pages: int, results: Dict[str, Any]):
        if depth > max_depth or len(results["visited_urls"]) >= max_pages:
            return
        
        if url in self.visited_urls:
            return
        
        try:
            self.logger.debug(f"Crawling {url} at depth {depth}")
            
            response = await self.page.goto(url, wait_until='domcontentloaded')
            if not response or response.status >= 400:
                results["errors"].append(f"Failed to load {url}: {response.status if response else 'No response'}")
                return
            
            self.visited_urls.add(url)
            results["visited_urls"].append({
                "url": url,
                "status": response.status,
                "title": await self.page.title(),
                "depth": depth
            })
            
            # Analyze current page
            await self._analyze_current_page(url, results)
            
            # Find links for next level crawling
            if depth < max_depth:
                links = await self._extract_links()
                for link in links[:10]:  # Limit links per page
                    if self._should_crawl_url(link, url):
                        results["discovered_urls"].append(link)
                        await self._crawl_recursive(link, depth + 1, max_depth, max_pages, results)
            
            await asyncio.sleep(1)  # Rate limiting
            
        except PlaywrightTimeoutError:
            results["errors"].append(f"Timeout loading {url}")
        except Exception as e:
            results["errors"].append(f"Error crawling {url}: {str(e)}")

    async def _analyze_forms(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        
        try:
            await self.page.goto(target_url, wait_until='domcontentloaded')
            
            forms_data = await self.page.evaluate("""
                () => {
                    const forms = Array.from(document.forms);
                    return forms.map(form => {
                        const inputs = Array.from(form.elements).map(input => ({
                            name: input.name,
                            type: input.type,
                            id: input.id,
                            className: input.className,
                            required: input.required,
                            placeholder: input.placeholder
                        }));
                        
                        return {
                            action: form.action,
                            method: form.method,
                            enctype: form.enctype,
                            id: form.id,
                            className: form.className,
                            inputs: inputs,
                            inputCount: inputs.length
                        };
                    });
                }
            """)
            
            findings = []
            for i, form in enumerate(forms_data):
                if form["method"].lower() == "get" and any(inp["type"] == "password" for inp in form["inputs"]):
                    findings.append({
                        "type": "sensitive_data_in_get",
                        "severity": "medium",
                        "description": "Password field in GET form",
                        "form_index": i
                    })
                
                if not form["action"].startswith("https://") and any(inp["type"] == "password" for inp in form["inputs"]):
                    findings.append({
                        "type": "insecure_form_submission",
                        "severity": "high", 
                        "description": "Password form not submitted over HTTPS",
                        "form_index": i
                    })
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data={"forms": forms_data},
                findings=findings,
                confidence=0.9
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _discover_endpoints(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        discovered_endpoints = []
        
        try:
            await self.page.goto(target_url, wait_until='domcontentloaded')
            
            # Extract API endpoints from JavaScript
            js_endpoints = await self.page.evaluate("""
                () => {
                    const endpoints = [];
                    const scripts = Array.from(document.scripts);
                    
                    for (const script of scripts) {
                        if (script.src) {
                            endpoints.push({type: 'script', url: script.src});
                        }
                        if (script.textContent) {
                            const apiMatches = script.textContent.match(/['"](\/api\/[^'"]+)['"]|['"](https?:\/\/[^'"]*\/api\/[^'"]+)['"]/g);
                            if (apiMatches) {
                                apiMatches.forEach(match => {
                                    const url = match.replace(/['"]/g, '');
                                    endpoints.push({type: 'api', url: url});
                                });
                            }
                        }
                    }
                    
                    return endpoints;
                }
            """)
            
            discovered_endpoints.extend(js_endpoints)
            
            # Extract from network requests (from interceptor)
            discovered_endpoints.extend(self.discovered_endpoints)
            
            # Remove duplicates
            unique_endpoints = []
            seen_urls = set()
            
            for endpoint in discovered_endpoints:
                if endpoint["url"] not in seen_urls:
                    unique_endpoints.append(endpoint)
                    seen_urls.add(endpoint["url"])
            
            findings = []
            for endpoint in unique_endpoints:
                if "/api/" in endpoint["url"] or endpoint["type"] == "api":
                    findings.append({
                        "type": "api_endpoint",
                        "severity": "info",
                        "url": endpoint["url"],
                        "description": "Discovered API endpoint"
                    })
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data={"endpoints": unique_endpoints},
                findings=findings,
                confidence=0.7
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _analyze_dom(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        
        try:
            await self.page.goto(target_url, wait_until='domcontentloaded')
            
            dom_analysis = await self.page.evaluate("""
                () => {
                    const analysis = {
                        title: document.title,
                        meta: {},
                        technologies: [],
                        security_headers: {},
                        comments: [],
                        external_resources: []
                    };
                    
                    // Meta tags
                    const metas = Array.from(document.querySelectorAll('meta'));
                    metas.forEach(meta => {
                        if (meta.name) {
                            analysis.meta[meta.name] = meta.content;
                        }
                    });
                    
                    // Technology detection
                    if (window.jQuery) analysis.technologies.push('jQuery');
                    if (window.React) analysis.technologies.push('React');
                    if (window.Vue) analysis.technologies.push('Vue.js');
                    if (window.angular) analysis.technologies.push('Angular');
                    
                    // External resources
                    const scripts = Array.from(document.scripts);
                    const links = Array.from(document.links);
                    
                    scripts.forEach(script => {
                        if (script.src && !script.src.startsWith(window.location.origin)) {
                            analysis.external_resources.push({type: 'script', url: script.src});
                        }
                    });
                    
                    links.forEach(link => {
                        if (link.href && !link.href.startsWith(window.location.origin)) {
                            analysis.external_resources.push({type: 'link', url: link.href});
                        }
                    });
                    
                    // HTML comments
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_COMMENT,
                        null,
                        false
                    );
                    
                    let comment;
                    while (comment = walker.nextNode()) {
                        analysis.comments.push(comment.textContent.trim());
                    }
                    
                    return analysis;
                }
            """)
            
            findings = []
            
            # Check for sensitive information in comments
            for comment in dom_analysis.get("comments", []):
                if re.search(r'(password|key|secret|token|api)', comment, re.IGNORECASE):
                    findings.append({
                        "type": "sensitive_comment",
                        "severity": "medium",
                        "comment": comment,
                        "description": "Potentially sensitive information in HTML comment"
                    })
            
            # Check for external resources
            external_count = len(dom_analysis.get("external_resources", []))
            if external_count > 10:
                findings.append({
                    "type": "many_external_resources",
                    "severity": "low",
                    "count": external_count,
                    "description": f"Large number of external resources ({external_count})"
                })
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data=dom_analysis,
                findings=findings,
                confidence=0.8
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _capture_screenshot(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        filename = task.parameters.get('filename', f'screenshot_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.png')
        
        try:
            await self.page.goto(target_url, wait_until='domcontentloaded')
            
            screenshot_path = f"./screenshots/{filename}"
            await self.page.screenshot(path=screenshot_path, full_page=True)
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data={
                    "screenshot_path": screenshot_path,
                    "url": target_url,
                    "timestamp": datetime.utcnow().isoformat()
                },
                confidence=1.0
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _execute_javascript(self, task: AgentTask) -> AgentResult:
        target_url = task.target
        javascript_code = task.parameters.get('code', '')
        
        if not javascript_code:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=["No JavaScript code provided"]
            )
        
        try:
            await self.page.goto(target_url, wait_until='domcontentloaded')
            
            result = await self.page.evaluate(javascript_code)
            
            return AgentResult(
                task_id=task.task_id,
                success=True,
                data={
                    "result": result,
                    "code": javascript_code,
                    "url": target_url
                },
                confidence=0.9
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)]
            )

    async def _setup_interceptors(self):
        async def handle_request(request):
            self.discovered_endpoints.append({
                "type": "request",
                "url": request.url,
                "method": request.method
            })

        async def handle_response(response):
            # Log interesting responses
            if response.url.endswith('.js') or '/api/' in response.url:
                self.discovered_endpoints.append({
                    "type": "response",
                    "url": response.url,
                    "status": response.status
                })

        self.page.on("request", handle_request)
        self.page.on("response", handle_response)

    async def _analyze_current_page(self, url: str, results: Dict[str, Any]):
        # Analyze forms
        forms = await self.page.query_selector_all("form")
        for form in forms:
            form_data = await form.evaluate("""
                form => ({
                    action: form.action,
                    method: form.method,
                    inputs: Array.from(form.elements).map(el => ({
                        name: el.name,
                        type: el.type
                    }))
                })
            """)
            results["forms"].append(form_data)
        
        # Get cookies
        cookies = await self.context.cookies()
        results["cookies"].extend(cookies)
        
        # Technology detection
        technologies = await self.page.evaluate("""
            () => {
                const tech = [];
                if (window.jQuery) tech.push('jQuery');
                if (window.React) tech.push('React');
                if (window.Vue) tech.push('Vue.js');
                if (window.angular) tech.push('Angular');
                return tech;
            }
        """)
        results["technologies"].extend(technologies)

    async def _extract_links(self) -> List[str]:
        links = await self.page.evaluate("""
            () => {
                const links = Array.from(document.links);
                return links.map(link => link.href).filter(href => href);
            }
        """)
        return links

    def _should_crawl_url(self, url: str, base_url: str) -> bool:
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            # Only crawl same domain
            if parsed_url.netloc != parsed_base.netloc:
                return False
            
            # Skip common file extensions
            skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.ico', '.xml', '.txt']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip already visited
            if url in self.visited_urls:
                return False
            
            return True
            
        except Exception:
            return False