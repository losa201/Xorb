"""
Feature Flag Middleware
Automatically injects feature flag context into requests
"""

import logging

import aiohttp
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class FeatureFlagMiddleware(BaseHTTPMiddleware):
    """Middleware to inject feature flags into request context"""

    def __init__(self, app, feature_service_url: str = "http://feature-flags:8007"):
        super().__init__(app)
        self.feature_service_url = feature_service_url
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_organization_features(self, organization_id: str) -> dict:
        """Get feature flags for organization"""

        # Check cache first
        cache_key = f"features:{organization_id}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        try:
            session = await self.get_session()

            async with session.get(
                f"{self.feature_service_url}/api/v1/features/{organization_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Cache the result
                    self.cache[cache_key] = (data, time.time())
                    return data
                else:
                    logger.warning(f"Feature service returned {response.status}")

        except TimeoutError:
            logger.warning("Feature service timeout")
        except Exception as e:
            logger.error(f"Feature service error: {e}")

        # Return default features on error
        return {
            "organization_id": organization_id,
            "tier": "growth",
            "features": {}
        }

    async def extract_organization_id(self, request: Request) -> str | None:
        """Extract organization ID from request"""

        # Check JWT token
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

    async def dispatch(self, request: Request, call_next):
        """Inject feature flags into request context"""

        # Extract organization ID
        organization_id = await self.extract_organization_id(request)

        if organization_id:
            # Get feature flags
            features = await self.get_organization_features(organization_id)

            # Inject into request state
            request.state.features = features
            request.state.tier = features.get('tier', 'growth')

            # Add feature flag helper functions
            def is_feature_enabled(feature_key: str) -> bool:
                """Check if a feature is enabled"""
                return features.get('features', {}).get(feature_key, {}).get('enabled', False)

            def get_feature_value(feature_key: str, default=None):
                """Get feature flag value"""
                feature_data = features.get('features', {}).get(feature_key, {})
                return feature_data.get('value', default)

            def get_feature_variant(feature_key: str) -> str | None:
                """Get A/B test variant"""
                return features.get('features', {}).get(feature_key, {}).get('variant')

            request.state.is_feature_enabled = is_feature_enabled
            request.state.get_feature_value = get_feature_value
            request.state.get_feature_variant = get_feature_variant

        # Process request
        response = await call_next(request)

        # Add feature context to response headers (for debugging)
        if organization_id and hasattr(request.state, 'tier'):
            response.headers["X-Tier"] = request.state.tier
            response.headers["X-Features-Loaded"] = "true"

        return response

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class FeatureGateMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce feature gates on endpoints"""

    def __init__(self, app):
        super().__init__(app)

        # Define feature gates for specific endpoints
        self.feature_gates = {
            "/api/v1/scans/advanced": "advanced_scanning",
            "/api/v1/analytics": "advanced_analytics",
            "/api/v1/custom-rules": "custom_rules",
            "/api/v1/integrations/sso": "sso_integration",
            "/api/v1/white-label": "white_label"
        }

        # Define tier-based endpoint restrictions
        self.tier_restrictions = {
            "/api/v1/analytics": ["elite", "enterprise"],
            "/api/v1/custom-rules": ["elite", "enterprise"],
            "/api/v1/integrations/sso": ["enterprise"],
            "/api/v1/white-label": ["enterprise"]
        }

    async def dispatch(self, request: Request, call_next):
        """Enforce feature gates"""

        # Check if endpoint requires feature gate
        endpoint = request.url.path
        required_feature = self.feature_gates.get(endpoint)
        required_tiers = self.tier_restrictions.get(endpoint)

        if required_feature or required_tiers:
            # Check if we have feature context
            if not hasattr(request.state, 'features'):
                return self._create_error_response("Feature context not available", 500)

            # Check tier restriction
            if required_tiers:
                current_tier = getattr(request.state, 'tier', 'growth')
                if current_tier not in required_tiers:
                    return self._create_error_response(
                        f"This feature requires {' or '.join(required_tiers)} tier", 403
                    )

            # Check feature flag
            if required_feature:
                if not hasattr(request.state, 'is_feature_enabled'):
                    return self._create_error_response("Feature flags not loaded", 500)

                if not request.state.is_feature_enabled(required_feature):
                    return self._create_error_response(
                        f"Feature '{required_feature}' is not enabled", 403
                    )

        # Process request
        return await call_next(request)

    def _create_error_response(self, message: str, status_code: int):
        """Create error response"""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "code": "FEATURE_GATE_ERROR"
            }
        )
