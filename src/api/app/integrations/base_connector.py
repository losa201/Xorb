"""
Base Integration Connector Infrastructure
Clean architecture base classes for enterprise integrations
"""

import asyncio
import ssl
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of enterprise integrations"""
    SIEM = "siem"
    SOAR = "soar"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    FIREWALL = "firewall"
    EDR = "edr"
    IDENTITY_PROVIDER = "identity_provider"
    TICKETING = "ticketing"
    NOTIFICATION = "notification"
    THREAT_INTELLIGENCE = "threat_intelligence"
    COMPLIANCE = "compliance"


class AuthenticationType(Enum):
    """Authentication methods for integrations"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    CERTIFICATE = "certificate"
    SAML = "saml"
    JWT = "jwt"
    CUSTOM = "custom"


@dataclass
class IntegrationCredentials:
    """Credentials for external system integration"""
    auth_type: AuthenticationType
    credentials: Dict[str, str]
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    certificate_path: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


@dataclass
class IntegrationEndpoint:
    """External system endpoint configuration"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    base_url: str
    credentials: IntegrationCredentials
    custom_headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_per_minute: int = 60
    enabled: bool = True
    last_successful_call: Optional[datetime] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationRequest:
    """Request to external system"""
    endpoint_id: str
    method: str
    path: str
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    custom_headers: Optional[Dict[str, str]] = None
    timeout_override: Optional[int] = None


@dataclass
class IntegrationResponse:
    """Response from external system"""
    success: bool
    status_code: int
    data: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0


class BaseConnector(ABC):
    """
    Abstract base class for all integration connectors.
    Implements common functionality for authentication, request handling, and error management.
    """
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        self._circuit_breaker = CircuitBreakerState()
        self._rate_limiter = RateLimiterState()
    
    async def initialize(self) -> None:
        """Initialize the connector"""
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ssl=self._create_ssl_context()
            )
            
            timeout = aiohttp.ClientTimeout(total=self.endpoint.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_default_headers()
            )
    
    async def close(self) -> None:
        """Close the connector and cleanup resources"""
        if self._session:
            await self._session.close()
            self._session = None
    
    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests - implemented by concrete connectors"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the integration is healthy - implemented by concrete connectors"""
        pass
    
    async def make_authenticated_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """
        Make authenticated request with circuit breaker, rate limiting, and retry logic
        """
        if not self.endpoint.enabled:
            return IntegrationResponse(
                success=False,
                status_code=503,
                error="Integration endpoint disabled"
            )
        
        # Check circuit breaker
        if not self._circuit_breaker.can_proceed():
            return IntegrationResponse(
                success=False,
                status_code=503,
                error="Circuit breaker open - too many failures"
            )
        
        # Check rate limiting
        if not self._rate_limiter.can_proceed(self.endpoint.rate_limit_per_minute):
            return IntegrationResponse(
                success=False,
                status_code=429,
                error="Rate limit exceeded"
            )
        
        # Initialize session if needed
        await self.initialize()
        
        # Prepare request
        url = urljoin(self.endpoint.base_url, request.path)
        headers = {**self.endpoint.custom_headers}
        
        if request.custom_headers:
            headers.update(request.custom_headers)
        
        # Add authentication headers
        auth_headers = await self._prepare_auth_headers()
        headers.update(auth_headers)
        
        timeout = request.timeout_override or self.endpoint.timeout
        start_time = datetime.utcnow()
        
        # Retry loop
        for attempt in range(self.endpoint.retry_attempts):
            try:
                async with self._session.request(
                    method=request.method,
                    url=url,
                    json=request.data,
                    params=request.params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    try:
                        response_data = await response.json()
                    except:
                        response_data = await response.text()
                    
                    success = 200 <= response.status < 300
                    
                    if success:
                        self.endpoint.last_successful_call = datetime.utcnow()
                        self.endpoint.consecutive_failures = 0
                        self._circuit_breaker.record_success()
                        self._rate_limiter.record_request()
                    else:
                        self.endpoint.consecutive_failures += 1
                        self._circuit_breaker.record_failure()
                        
                        # Don't retry on client errors (4xx)
                        if 400 <= response.status < 500 and attempt == 0:
                            break
                    
                    response_obj = IntegrationResponse(
                        success=success,
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        response_time=response_time,
                        retry_count=attempt
                    )
                    
                    if success or 400 <= response.status < 500:
                        return response_obj
                    
                    # Log retry attempt
                    if attempt < self.endpoint.retry_attempts - 1:
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}): {response.status}. Retrying..."
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return response_obj
            
            except asyncio.TimeoutError:
                self.endpoint.consecutive_failures += 1
                self._circuit_breaker.record_failure()
                
                if attempt < self.endpoint.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return IntegrationResponse(
                        success=False,
                        status_code=408,
                        error="Request timeout",
                        response_time=(datetime.utcnow() - start_time).total_seconds(),
                        retry_count=attempt
                    )
            
            except Exception as e:
                self.endpoint.consecutive_failures += 1
                self._circuit_breaker.record_failure()
                
                if attempt < self.endpoint.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return IntegrationResponse(
                        success=False,
                        status_code=500,
                        error=str(e),
                        response_time=(datetime.utcnow() - start_time).total_seconds(),
                        retry_count=attempt
                    )
        
        # Should never reach here, but safety fallback
        return IntegrationResponse(
            success=False,
            status_code=500,
            error="Maximum retry attempts exceeded"
        )
    
    async def _prepare_auth_headers(self) -> Dict[str, str]:
        """Prepare authentication headers based on credential type"""
        headers = {}
        credentials = self.endpoint.credentials
        
        if credentials.auth_type == AuthenticationType.API_KEY:
            api_key = credentials.credentials.get("api_key")
            key_header = credentials.credentials.get("key_header", "X-API-Key")
            if api_key:
                headers[key_header] = api_key
        
        elif credentials.auth_type == AuthenticationType.BASIC_AUTH:
            username = credentials.credentials.get("username")
            password = credentials.credentials.get("password")
            if username and password:
                import base64
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_string}"
        
        elif credentials.auth_type == AuthenticationType.JWT:
            token = credentials.credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif credentials.auth_type == AuthenticationType.OAUTH2:
            # Handle OAuth2 token refresh if needed
            token = await self._get_oauth2_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        return headers
    
    async def _get_oauth2_token(self) -> Optional[str]:
        """Get OAuth2 token, refreshing if necessary"""
        credentials = self.endpoint.credentials
        
        # Check if current token is expired
        if credentials.is_expired() and credentials.refresh_token:
            await self._refresh_oauth2_token()
        
        return credentials.credentials.get("access_token")
    
    async def _refresh_oauth2_token(self) -> bool:
        """Refresh OAuth2 token - can be overridden by specific connectors"""
        # Base implementation - specific connectors should override this
        logger.warning(f"OAuth2 token refresh not implemented for {self.endpoint.name}")
        return False
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context()
        
        # For production, should verify certificates
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Add custom certificate if provided
        if self.endpoint.credentials.certificate_path:
            try:
                context.load_cert_chain(self.endpoint.credentials.certificate_path)
            except Exception as e:
                logger.error(f"Failed to load certificate: {e}")
        
        return context
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status information"""
        return {
            "endpoint_id": self.endpoint.integration_id,
            "name": self.endpoint.name,
            "type": self.endpoint.integration_type.value,
            "enabled": self.endpoint.enabled,
            "last_successful_call": self.endpoint.last_successful_call.isoformat() if self.endpoint.last_successful_call else None,
            "consecutive_failures": self.endpoint.consecutive_failures,
            "circuit_breaker_state": self._circuit_breaker.state,
            "rate_limit_usage": len(self._rate_limiter.requests),
            "rate_limit_per_minute": self.endpoint.rate_limit_per_minute
        }


class CircuitBreakerState:
    """Circuit breaker implementation for integration fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_proceed(self) -> bool:
        """Check if requests can proceed"""
        current_time = datetime.utcnow()
        
        if self.state == "open":
            # Check if we should transition to half-open
            if (self.last_failure_time and 
                (current_time - self.last_failure_time).total_seconds() > self.recovery_timeout):
                self.state = "half-open"
                return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful request"""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed request"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")


class RateLimiterState:
    """Rate limiter implementation for integration throttling"""
    
    def __init__(self):
        self.requests: List[datetime] = []
    
    def can_proceed(self, limit_per_minute: int) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=1)
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
        
        return len(self.requests) < limit_per_minute
    
    def record_request(self):
        """Record a new request"""
        self.requests.append(datetime.utcnow())