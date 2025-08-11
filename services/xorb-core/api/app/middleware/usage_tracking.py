"""
Usage Tracking Middleware
Automatically tracks API usage for billing purposes
"""

import asyncio
import logging
import time

import aiohttp
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track API usage for billing"""

    def __init__(self, app, billing_service_url: str = "http://billing:8006"):
        super().__init__(app)
        self.billing_service_url = billing_service_url
        self.session = None

    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def extract_organization_id(self, request: Request) -> str | None:
        """Extract organization ID from request"""
        # Check JWT token for organization claim
        if hasattr(request.state, 'user') and request.state.user:
            return request.state.user.get('organization_id')

        # Check headers
        org_id = request.headers.get('X-Organization-ID')
        if org_id:
            return org_id

        # Check query parameters
        org_id = request.query_params.get('organization_id')
        if org_id:
            return org_id

        return None

    async def record_api_usage(self, organization_id: str, endpoint: str, method: str):
        """Record API usage to billing service"""
        try:
            session = await self.get_session()

            # Record the API call
            async with session.post(
                f"{self.billing_service_url}/api/v1/billing/usage",
                params={
                    "organization_id": organization_id,
                    "metric_name": "api_calls",
                    "value": 1
                },
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to record API usage: {response.status}")

        except TimeoutError:
            logger.warning("Timeout recording API usage")
        except Exception as e:
            logger.error(f"Error recording API usage: {e}")

    async def record_scan_usage(self, organization_id: str):
        """Record scan usage to billing service"""
        try:
            session = await self.get_session()

            async with session.post(
                f"{self.billing_service_url}/api/v1/billing/usage",
                params={
                    "organization_id": organization_id,
                    "metric_name": "scans",
                    "value": 1
                },
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to record scan usage: {response.status}")

        except TimeoutError:
            logger.warning("Timeout recording scan usage")
        except Exception as e:
            logger.error(f"Error recording scan usage: {e}")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and track usage"""
        start_time = time.time()

        # Extract organization ID
        organization_id = await self.extract_organization_id(request)

        # Process the request
        response = await call_next(request)

        # Track usage if organization ID is available and request was successful
        if organization_id and response.status_code < 400:
            # Track API usage for all requests
            asyncio.create_task(
                self.record_api_usage(
                    organization_id,
                    request.url.path,
                    request.method
                )
            )

            # Track specific resource usage
            if request.url.path.startswith('/api/v1/scans') and request.method == 'POST':
                asyncio.create_task(
                    self.record_scan_usage(organization_id)
                )

        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class UsageLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce usage limits"""

    def __init__(self, app, billing_service_url: str = "http://billing:8006"):
        super().__init__(app)
        self.billing_service_url = billing_service_url
        self.session = None

    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def check_usage_limits(self, organization_id: str, request_type: str) -> bool:
        """Check if request is within usage limits"""
        try:
            session = await self.get_session()

            # Get current usage dashboard
            async with session.get(
                f"{self.billing_service_url}/api/v1/billing/usage/{organization_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    usage_data = await response.json()

                    # Check specific limits based on request type
                    if request_type == "api_call":
                        api_usage = usage_data.get("usage", {}).get("api_calls", {})
                        limit = api_usage.get("limit")
                        current = api_usage.get("current_usage", 0)

                        if limit != "unlimited" and current >= limit:
                            return False

                    elif request_type == "scan":
                        scan_usage = usage_data.get("usage", {}).get("scans", {})
                        limit = scan_usage.get("limit")
                        current = scan_usage.get("current_usage", 0)

                        if limit != "unlimited" and current >= limit:
                            return False

                    return True
                else:
                    # If we can't check limits, allow the request
                    logger.warning(f"Failed to check usage limits: {response.status}")
                    return True

        except Exception as e:
            logger.error(f"Error checking usage limits: {e}")
            # If we can't check limits, allow the request
            return True

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check usage limits before processing request"""
        # Extract organization ID
        organization_id = None
        if hasattr(request.state, 'user') and request.state.user:
            organization_id = request.state.user.get('organization_id')

        if not organization_id:
            organization_id = request.headers.get('X-Organization-ID')

        # Check limits for specific endpoints
        if organization_id:
            request_type = None

            # Determine request type
            if request.url.path.startswith('/api/v1/scans') and request.method == 'POST':
                request_type = "scan"
            elif request.url.path.startswith('/api/v1/'):
                request_type = "api_call"

            if request_type:
                # Check if request is within limits
                within_limits = await self.check_usage_limits(organization_id, request_type)

                if not within_limits:
                    return Response(
                        content='{"error": "Usage limit exceeded for your current tier"}',
                        status_code=429,
                        headers={"Content-Type": "application/json"}
                    )

        # Process the request if within limits
        return await call_next(request)

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
