import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from playwright.async_api import async_playwright, Browser, Page

from xorb.shared.epyc_execution_config import EPYCExecutionConfig
from xorb.shared.execution_models import StealthConfig

# Stealth Web Engine
class StealthWebEngine:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize Playwright browser."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--memory-pressure-off',
                    '--max_old_space_size=2048'
                ]
            )
            self.logger.info("Stealth Web Engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize web engine: {e}")

    async def stealth_browse(self, url: str, config: StealthConfig) -> Dict[str, Any]:
        """Perform stealth web browsing and data collection."""
        if not self.browser:
            await self.initialize()

        try:
            context = await self.browser.new_context(
                user_agent=config.user_agent,
                extra_http_headers=config.request_headers,
                ignore_https_errors=True
            )

            page = await context.new_page()

            # Apply stealth techniques
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)

            # Browse with delays
            start_time = time.time()

            response = await page.goto(url, wait_until='networkidle')

            # Random delay based on config
            import random
            delay = random.uniform(*config.delay_range)
            await asyncio.sleep(delay)

            # Collect evidence
            evidence = {
                'url': url,
                'status_code': response.status if response else None,
                'title': await page.title(),
                'content_length': len(await page.content()),
                'screenshot': None,
                'forms': [],
                'links': [],
                'scripts': [],
                'cookies': await context.cookies()
            }

            # Take screenshot
            screenshot_path = f"/tmp/screenshot_{uuid4().hex}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            evidence['screenshot'] = screenshot_path

            # Extract forms
            forms = await page.query_selector_all('form')
            for form in forms:
                form_data = await form.evaluate("""form => ({
                    action: form.action,
                    method: form.method,
                    inputs: Array.from(form.querySelectorAll('input')).map(input => ({
                        name: input.name,
                        type: input.type,
                        value: input.value
                    }))
                })""")
                evidence['forms'].append(form_data)

            # Extract links
            links = await page.query_selector_all('a[href]')
            for link in links[:50]:  # Limit to 50 links
                href = await link.get_attribute('href')
                text = await link.inner_text()
                evidence['links'].append({'href': href, 'text': text[:100]})

            # Extract script sources
            scripts = await page.query_selector_all('script[src]')
            for script in scripts:
                src = await script.get_attribute('src')
                evidence['scripts'].append(src)

            await context.close()

            evidence['duration'] = time.time() - start_time
            return evidence

        except Exception as e:
            self.logger.error(f"Stealth browsing error for {url}: {e}")
            return {'error': str(e), 'url': url}
