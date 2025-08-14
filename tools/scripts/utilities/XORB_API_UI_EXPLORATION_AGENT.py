#!/usr/bin/env python3
"""
🕸️ XORB API & UI Exploration Agent (UIXplorer)
Intelligent application flow discovery and boundary testing

This agent learns user flows from OpenAPI specs, HAR logs, Postman collections,
and browser automation to test privilege boundaries and hidden functions.
"""

import asyncio
import json
import logging
import aiohttp
import yaml
import re
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import urllib.parse
import secrets
import time

# Browser automation imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    # Create placeholder classes for type hints
    class Page:
        pass
    class Browser:
        pass
    class BrowserContext:
        pass
    logger.warning("Playwright not available - browser automation disabled")

class ExplorationMethod(Enum):
    OPENAPI_SPEC = "openapi_spec"
    HAR_ANALYSIS = "har_analysis"
    POSTMAN_COLLECTION = "postman_collection"
    BROWSER_AUTOMATION = "browser_automation"
    API_FUZZING = "api_fuzzing"
    SITEMAP_CRAWLING = "sitemap_crawling"

class UserRole(Enum):
    ANONYMOUS = "anonymous"
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    API_CLIENT = "api_client"
    SERVICE_ACCOUNT = "service_account"

class FlowType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    BUSINESS_LOGIC = "business_logic"
    ADMIN_FUNCTION = "admin_function"
    API_ENDPOINT = "api_endpoint"
    FILE_UPLOAD = "file_upload"
    PAYMENT_FLOW = "payment_flow"

@dataclass
class UserFlow:
    flow_id: str
    flow_type: FlowType
    user_role: UserRole
    steps: List[Dict[str, Any]]
    endpoints: List[str]
    parameters: Dict[str, Any]
    authentication_required: bool
    privilege_level: int
    discovered_by: ExplorationMethod
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PrivilegeBoundary:
    boundary_id: str
    source_role: UserRole
    target_role: UserRole
    protected_endpoints: List[str]
    bypass_attempts: List[Dict[str, Any]]
    violation_detected: bool
    risk_level: str

@dataclass
class HiddenFunction:
    function_id: str
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    discovered_method: ExplorationMethod
    access_level: str
    functionality: str
    risk_assessment: str

class XORBAPIUIExplorationAgent:
    """API & UI Exploration Agent for comprehensive application mapping"""

    def __init__(self):
        self.agent_id = f"UIXPLORER-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.discovered_flows = {}
        self.privilege_boundaries = {}
        self.hidden_functions = {}
        self.session_tokens = {}

        # Browser automation
        self.browser = None
        self.browser_context = None

        # Exploration configuration
        self.max_depth = 5
        self.request_delay = 1.0
        self.stealth_mode = True

        logger.info(f"🕸️ API & UI Exploration Agent initialized - ID: {self.agent_id}")

    async def explore_from_openapi_spec(self, spec_url: str) -> List[UserFlow]:
        """Explore application flows from OpenAPI specification"""
        try:
            flows = []

            async with aiohttp.ClientSession() as session:
                async with session.get(spec_url) as response:
                    if response.content_type == 'application/json':
                        spec_data = await response.json()
                    else:
                        spec_text = await response.text()
                        spec_data = yaml.safe_load(spec_text)

            # Parse OpenAPI specification
            base_url = self._extract_base_url(spec_data)
            paths = spec_data.get('paths', {})

            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                        flow = await self._create_flow_from_openapi(
                            path, method.upper(), details, base_url
                        )
                        if flow:
                            flows.append(flow)
                            self.discovered_flows[flow.flow_id] = flow

            logger.info(f"🕸️ Discovered {len(flows)} flows from OpenAPI spec")
            return flows

        except Exception as e:
            logger.error(f"❌ OpenAPI exploration error: {e}")
            return []

    async def explore_from_har_file(self, har_path: str) -> List[UserFlow]:
        """Explore application flows from HAR (HTTP Archive) file"""
        try:
            flows = []

            with open(har_path, 'r') as f:
                har_data = json.load(f)

            entries = har_data.get('log', {}).get('entries', [])

            # Group requests by session/user flow
            flow_sessions = self._group_har_entries_by_session(entries)

            for session_id, session_entries in flow_sessions.items():
                flow = await self._create_flow_from_har_session(session_id, session_entries)
                if flow:
                    flows.append(flow)
                    self.discovered_flows[flow.flow_id] = flow

            logger.info(f"🕸️ Discovered {len(flows)} flows from HAR file")
            return flows

        except Exception as e:
            logger.error(f"❌ HAR exploration error: {e}")
            return []

    async def explore_from_postman_collection(self, collection_path: str) -> List[UserFlow]:
        """Explore application flows from Postman collection"""
        try:
            flows = []

            with open(collection_path, 'r') as f:
                collection_data = json.load(f)

            # Parse Postman collection structure
            items = collection_data.get('item', [])

            for item in items:
                flow = await self._create_flow_from_postman_item(item)
                if flow:
                    flows.append(flow)
                    self.discovered_flows[flow.flow_id] = flow

            logger.info(f"🕸️ Discovered {len(flows)} flows from Postman collection")
            return flows

        except Exception as e:
            logger.error(f"❌ Postman exploration error: {e}")
            return []

    async def explore_with_browser_automation(self, target_url: str, credentials: Dict[str, Any] = None) -> List[UserFlow]:
        """Explore application using browser automation"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("⚠️ Browser automation unavailable - Playwright not installed")
            return []

        try:
            flows = []

            async with async_playwright() as p:
                # Launch browser in stealth mode
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-web-security']
                )

                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )

                page = await context.new_page()

                # Enable request interception
                captured_requests = []
                page.on('request', lambda request: captured_requests.append({
                    'url': request.url,
                    'method': request.method,
                    'headers': request.headers,
                    'post_data': request.post_data
                }))

                # Navigate and explore
                await page.goto(target_url)
                await self._perform_browser_exploration(page, credentials)

                # Create flows from captured requests
                flows = await self._create_flows_from_browser_requests(captured_requests)

                await browser.close()

            logger.info(f"🕸️ Discovered {len(flows)} flows via browser automation")
            return flows

        except Exception as e:
            logger.error(f"❌ Browser exploration error: {e}")
            return []

    async def test_privilege_boundaries(self, flows: List[UserFlow]) -> List[PrivilegeBoundary]:
        """Test privilege boundaries between different user roles"""
        try:
            boundaries = []

            # Group flows by privilege level
            privilege_groups = {}
            for flow in flows:
                level = flow.privilege_level
                if level not in privilege_groups:
                    privilege_groups[level] = []
                privilege_groups[level].append(flow)

            # Test cross-privilege access
            for low_level, low_flows in privilege_groups.items():
                for high_level, high_flows in privilege_groups.items():
                    if high_level > low_level:
                        boundary = await self._test_privilege_escalation(
                            low_flows, high_flows, low_level, high_level
                        )
                        if boundary:
                            boundaries.append(boundary)
                            self.privilege_boundaries[boundary.boundary_id] = boundary

            logger.info(f"🛡️ Tested {len(boundaries)} privilege boundaries")
            return boundaries

        except Exception as e:
            logger.error(f"❌ Privilege boundary testing error: {e}")
            return []

    async def discover_hidden_functions(self, base_url: str, known_endpoints: List[str]) -> List[HiddenFunction]:
        """Discover hidden functions and admin endpoints"""
        try:
            hidden_functions = []

            # Common admin/hidden paths
            hidden_paths = [
                '/admin', '/administrator', '/panel', '/dashboard', '/console',
                '/api/admin', '/api/internal', '/api/v2', '/api/debug',
                '/management', '/config', '/settings', '/status', '/health',
                '/debug', '/test', '/dev', '/staging', '/backup',
                '/.env', '/config.json', '/api-docs', '/swagger.json'
            ]

            # HTTP methods to test
            methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']

            async with aiohttp.ClientSession() as session:
                for path in hidden_paths:
                    for method in methods:
                        try:
                            url = f"{base_url}{path}"

                            async with session.request(method, url, timeout=5) as response:
                                if response.status not in [404, 405, 403]:
                                    hidden_function = HiddenFunction(
                                        function_id=f"HIDDEN-{hashlib.sha256(url.encode()).hexdigest()[:8]}",
                                        endpoint=path,
                                        method=method,
                                        parameters={},
                                        discovered_method=ExplorationMethod.API_FUZZING,
                                        access_level=self._assess_access_level(response.status),
                                        functionality=self._assess_functionality(path, response),
                                        risk_assessment=self._assess_risk_level(path, response.status)
                                    )

                                    hidden_functions.append(hidden_function)
                                    self.hidden_functions[hidden_function.function_id] = hidden_function

                            # Stealth delay
                            if self.stealth_mode:
                                await asyncio.sleep(self.request_delay)

                        except:
                            continue

            logger.info(f"🔍 Discovered {len(hidden_functions)} hidden functions")
            return hidden_functions

        except Exception as e:
            logger.error(f"❌ Hidden function discovery error: {e}")
            return []

    async def test_session_handling(self, flows: List[UserFlow]) -> Dict[str, Any]:
        """Test session handling and authentication flaws"""
        try:
            session_tests = {
                "session_fixation": [],
                "session_hijacking": [],
                "weak_tokens": [],
                "csrf_vulnerabilities": [],
                "broken_authentication": []
            }

            async with aiohttp.ClientSession() as session:
                for flow in flows:
                    if flow.authentication_required:
                        # Test session fixation
                        fixation_result = await self._test_session_fixation(session, flow)
                        if fixation_result:
                            session_tests["session_fixation"].append(fixation_result)

                        # Test weak tokens
                        token_result = await self._test_weak_tokens(session, flow)
                        if token_result:
                            session_tests["weak_tokens"].append(token_result)

                        # Test CSRF
                        csrf_result = await self._test_csrf_protection(session, flow)
                        if csrf_result:
                            session_tests["csrf_vulnerabilities"].append(csrf_result)

            total_issues = sum(len(issues) for issues in session_tests.values())
            logger.info(f"🔐 Found {total_issues} session handling issues")

            return session_tests

        except Exception as e:
            logger.error(f"❌ Session testing error: {e}")
            return {}

    async def _create_flow_from_openapi(self, path: str, method: str, details: Dict[str, Any], base_url: str) -> Optional[UserFlow]:
        """Create user flow from OpenAPI endpoint definition"""
        try:
            flow_id = f"API-{hashlib.sha256(f'{method}{path}'.encode()).hexdigest()[:8]}"

            # Determine flow type
            flow_type = self._classify_flow_type(path, method, details)

            # Determine required role
            user_role = self._determine_required_role(details)

            # Extract parameters
            parameters = {}
            for param in details.get('parameters', []):
                parameters[param['name']] = {
                    'type': param.get('schema', {}).get('type', 'string'),
                    'required': param.get('required', False),
                    'location': param.get('in', 'query')
                }

            # Create flow steps
            steps = [{
                'step': 1,
                'action': f'{method} {path}',
                'endpoint': f'{base_url}{path}',
                'parameters': parameters
            }]

            flow = UserFlow(
                flow_id=flow_id,
                flow_type=flow_type,
                user_role=user_role,
                steps=steps,
                endpoints=[f'{base_url}{path}'],
                parameters=parameters,
                authentication_required=self._requires_authentication(details),
                privilege_level=self._calculate_privilege_level(user_role),
                discovered_by=ExplorationMethod.OPENAPI_SPEC
            )

            return flow

        except Exception as e:
            logger.error(f"❌ OpenAPI flow creation error: {e}")
            return None

    async def _perform_browser_exploration(self, page: Page, credentials: Dict[str, Any] = None):
        """Perform automated browser exploration"""
        try:
            # Login if credentials provided
            if credentials:
                await self._perform_login(page, credentials)

            # Discover and click navigation elements
            await self._explore_navigation(page)

            # Fill and submit forms
            await self._explore_forms(page)

            # Test JavaScript interactions
            await self._explore_javascript_functions(page)

        except Exception as e:
            logger.error(f"❌ Browser exploration error: {e}")

    async def _perform_login(self, page: Page, credentials: Dict[str, Any]):
        """Perform login with provided credentials"""
        try:
            # Look for common login form selectors
            login_selectors = [
                'input[type="email"]', 'input[name="email"]', 'input[name="username"]',
                'input[id="email"]', 'input[id="username"]', '#login_email', '#login_username'
            ]

            password_selectors = [
                'input[type="password"]', 'input[name="password"]',
                'input[id="password"]', '#login_password'
            ]

            # Fill login form
            for selector in login_selectors:
                try:
                    await page.fill(selector, credentials.get('username', ''))
                    break
                except:
                    continue

            for selector in password_selectors:
                try:
                    await page.fill(selector, credentials.get('password', ''))
                    break
                except:
                    continue

            # Submit form
            submit_selectors = [
                'button[type="submit"]', 'input[type="submit"]',
                'button:has-text("Login")', 'button:has-text("Sign In")'
            ]

            for selector in submit_selectors:
                try:
                    await page.click(selector)
                    await page.wait_for_load_state()
                    break
                except:
                    continue

        except Exception as e:
            logger.error(f"❌ Login error: {e}")

    async def _explore_navigation(self, page: Page):
        """Explore navigation elements"""
        try:
            # Find navigation links
            nav_links = await page.query_selector_all('a[href]')

            for link in nav_links[:10]:  # Limit exploration
                try:
                    href = await link.get_attribute('href')
                    if href and not href.startswith('javascript:') and not href.startswith('#'):
                        await link.click()
                        await page.wait_for_load_state()
                        await page.go_back()
                        await page.wait_for_load_state()
                except:
                    continue

        except Exception as e:
            logger.error(f"❌ Navigation exploration error: {e}")

    def _classify_flow_type(self, path: str, method: str, details: Dict[str, Any]) -> FlowType:
        """Classify flow type based on path and method"""
        path_lower = path.lower()

        if 'auth' in path_lower or 'login' in path_lower:
            return FlowType.AUTHENTICATION
        elif 'admin' in path_lower or 'manage' in path_lower:
            return FlowType.ADMIN_FUNCTION
        elif 'upload' in path_lower or 'file' in path_lower:
            return FlowType.FILE_UPLOAD
        elif 'pay' in path_lower or 'billing' in path_lower:
            return FlowType.PAYMENT_FLOW
        elif method in ['GET', 'POST', 'PUT', 'DELETE']:
            return FlowType.API_ENDPOINT
        else:
            return FlowType.BUSINESS_LOGIC

    def _determine_required_role(self, details: Dict[str, Any]) -> UserRole:
        """Determine required user role from OpenAPI details"""
        security = details.get('security', [])

        if not security:
            return UserRole.ANONYMOUS

        # Check for admin requirements
        tags = details.get('tags', [])
        if any('admin' in tag.lower() for tag in tags):
            return UserRole.ADMIN

        return UserRole.USER

    def _calculate_privilege_level(self, user_role: UserRole) -> int:
        """Calculate numeric privilege level"""
        privilege_map = {
            UserRole.ANONYMOUS: 0,
            UserRole.USER: 1,
            UserRole.API_CLIENT: 2,
            UserRole.ADMIN: 3,
            UserRole.SUPER_ADMIN: 4,
            UserRole.SERVICE_ACCOUNT: 5
        }

        return privilege_map.get(user_role, 0)

    async def get_exploration_summary(self) -> Dict[str, Any]:
        """Get comprehensive exploration summary"""
        try:
            summary = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "discovered_flows": len(self.discovered_flows),
                "privilege_boundaries": len(self.privilege_boundaries),
                "hidden_functions": len(self.hidden_functions),
                "flow_types": self._analyze_flow_types(),
                "user_roles": self._analyze_user_roles(),
                "risk_assessment": await self._calculate_risk_assessment(),
                "exploration_coverage": await self._calculate_coverage()
            }

            return summary

        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")
            return {}

    def _analyze_flow_types(self) -> Dict[str, int]:
        """Analyze distribution of flow types"""
        flow_types = {}
        for flow in self.discovered_flows.values():
            flow_type = flow.flow_type.value
            flow_types[flow_type] = flow_types.get(flow_type, 0) + 1
        return flow_types

    def _analyze_user_roles(self) -> Dict[str, int]:
        """Analyze distribution of user roles"""
        user_roles = {}
        for flow in self.discovered_flows.values():
            role = flow.user_role.value
            user_roles[role] = user_roles.get(role, 0) + 1
        return user_roles

    async def _calculate_risk_assessment(self) -> Dict[str, Any]:
        """Calculate overall risk assessment"""
        high_risk_functions = [f for f in self.hidden_functions.values()
                              if f.risk_assessment == 'high']

        privilege_violations = [b for b in self.privilege_boundaries.values()
                               if b.violation_detected]

        return {
            "high_risk_hidden_functions": len(high_risk_functions),
            "privilege_violations": len(privilege_violations),
            "total_attack_surface": len(self.discovered_flows),
            "risk_score": min(10, len(high_risk_functions) + len(privilege_violations))
        }

async def main():
    """Demonstrate XORB API & UI Exploration Agent"""
    logger.info("🕸️ Starting API & UI Exploration demonstration")

    agent = XORBAPIUIExplorationAgent()

    # Sample explorations
    # Note: These would normally use real URLs and files

    # Simulate OpenAPI exploration
    logger.info("🔍 Simulating OpenAPI exploration...")
    sample_openapi_flows = []

    # Simulate hidden function discovery
    logger.info("🔍 Discovering hidden functions...")
    hidden_functions = await agent.discover_hidden_functions("https://example.com", [])

    # Get exploration summary
    summary = await agent.get_exploration_summary()

    logger.info("🕸️ Exploration demonstration complete")
    logger.info(f"📊 Total flows discovered: {summary.get('discovered_flows', 0)}")
    logger.info(f"🔍 Hidden functions found: {summary.get('hidden_functions', 0)}")
    logger.info(f"🛡️ Privilege boundaries tested: {summary.get('privilege_boundaries', 0)}")

    return {
        "agent_id": agent.agent_id,
        "flows_discovered": summary.get('discovered_flows', 0),
        "hidden_functions": summary.get('hidden_functions', 0),
        "privilege_boundaries": summary.get('privilege_boundaries', 0),
        "risk_score": summary.get('risk_assessment', {}).get('risk_score', 0)
    }

if __name__ == "__main__":
    asyncio.run(main())
