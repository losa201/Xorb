"""
Enterprise Integration Connector
Seamless integration with customer security stacks and third-party tools
"""

import asyncio
import logging
import json
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import base64
import hashlib
import hmac

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


class SplunkConnector:
    """Splunk SIEM integration"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        
    async def send_event(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Send security event to Splunk"""
        splunk_event = {
            "time": event_data.get("timestamp", datetime.utcnow().timestamp()),
            "event": event_data,
            "source": "xorb_platform",
            "sourcetype": "xorb:security_event"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/services/collector/event",
            data=splunk_event,
            custom_headers={
                "Authorization": f"Splunk {self.endpoint.credentials.credentials.get('token')}",
                "Content-Type": "application/json"
            }
        )
        
        return await EnterpriseConnector().make_request(request)
    
    async def search_events(self, query: str, earliest_time: str = "-1h") -> IntegrationResponse:
        """Search Splunk for events"""
        search_data = {
            "search": f"search {query}",
            "earliest_time": earliest_time,
            "output_mode": "json"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/services/search/jobs",
            data=search_data
        )
        
        return await EnterpriseConnector().make_request(request)


class QRadarConnector:
    """IBM QRadar SIEM integration"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        
    async def send_event(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Send event to QRadar"""
        qradar_event = {
            "events": [{
                "qid": 28250004,  # Custom rule QID
                "message": json.dumps(event_data),
                "properties": {
                    "sourceip": event_data.get("source_ip"),
                    "destinationip": event_data.get("destination_ip"),
                    "username": event_data.get("user"),
                    "magnitude": self._severity_to_magnitude(event_data.get("severity"))
                }
            }]
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/siem/events",
            data=qradar_event,
            custom_headers={
                "SEC": self.endpoint.credentials.credentials.get("sec_token"),
                "Version": "12.0"
            }
        )
        
        return await EnterpriseConnector().make_request(request)
    
    def _severity_to_magnitude(self, severity: str) -> int:
        """Convert severity to QRadar magnitude"""
        mapping = {
            "critical": 10,
            "high": 8,
            "medium": 5,
            "low": 3,
            "info": 1
        }
        return mapping.get(severity.lower() if severity else "", 5)


class ServiceNowConnector:
    """ServiceNow ITSM integration"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        
    async def create_incident(self, incident_data: Dict[str, Any]) -> IntegrationResponse:
        """Create incident in ServiceNow"""
        snow_incident = {
            "short_description": incident_data.get("title", "XORB Security Alert"),
            "description": incident_data.get("description"),
            "urgency": self._severity_to_urgency(incident_data.get("severity")),
            "impact": self._severity_to_impact(incident_data.get("severity")),
            "category": "Security",
            "subcategory": "Malware/Virus",
            "assignment_group": incident_data.get("assignment_group", "Security Operations"),
            "caller_id": incident_data.get("caller", "xorb.platform"),
            "work_notes": f"Generated by XORB Platform at {datetime.utcnow().isoformat()}"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/now/table/incident",
            data=snow_incident
        )
        
        return await EnterpriseConnector().make_request(request)
    
    def _severity_to_urgency(self, severity: str) -> str:
        """Convert severity to ServiceNow urgency"""
        mapping = {
            "critical": "1 - High",
            "high": "2 - Medium", 
            "medium": "3 - Low",
            "low": "3 - Low"
        }
        return mapping.get(severity.lower() if severity else "", "3 - Low")
    
    def _severity_to_impact(self, severity: str) -> str:
        """Convert severity to ServiceNow impact"""
        mapping = {
            "critical": "1 - High",
            "high": "2 - Medium",
            "medium": "3 - Low", 
            "low": "3 - Low"
        }
        return mapping.get(severity.lower() if severity else "", "3 - Low")


class PhantomConnector:
    """Phantom SOAR integration"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        
    async def create_container(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Create container (case) in Phantom"""
        container = {
            "name": event_data.get("title", "XORB Security Event"),
            "description": event_data.get("description"),
            "label": "events",
            "severity": event_data.get("severity", "medium"),
            "sensitivity": "amber",
            "status": "new",
            "tags": event_data.get("tags", []),
            "source_data_identifier": event_data.get("event_id")
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/rest/container",
            data=container
        )
        
        return await EnterpriseConnector().make_request(request)
    
    async def run_playbook(self, container_id: int, playbook_id: int) -> IntegrationResponse:
        """Run automated playbook in Phantom"""
        playbook_data = {
            "container_id": container_id,
            "playbook_id": playbook_id,
            "scope": "all"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/rest/playbook_run",
            data=playbook_data
        )
        
        return await EnterpriseConnector().make_request(request)


class CrowdStrikeConnector:
    """CrowdStrike Falcon EDR integration"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        
    async def get_detections(self, filter_query: str = None) -> IntegrationResponse:
        """Get detections from CrowdStrike"""
        params = {
            "limit": "100",
            "sort": "first_behavior|desc"
        }
        
        if filter_query:
            params["filter"] = filter_query
            
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/detects/queries/detects/v1",
            params=params,
            custom_headers={
                "Authorization": f"Bearer {await self._get_access_token()}"
            }
        )
        
        return await EnterpriseConnector().make_request(request)
    
    async def isolate_host(self, device_id: str) -> IntegrationResponse:
        """Isolate host in CrowdStrike"""
        isolation_data = {
            "ids": [device_id],
            "action_parameters": [
                {
                    "name": "isolation_type",
                    "value": "normal"
                }
            ]
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/devices/entities/devices-actions/v2",
            data=isolation_data,
            custom_headers={
                "Authorization": f"Bearer {await self._get_access_token()}"
            }
        )
        
        return await EnterpriseConnector().make_request(request)
    
    async def _get_access_token(self) -> str:
        """Get OAuth2 access token for CrowdStrike"""
        # Implementation would handle OAuth2 flow
        return self.endpoint.credentials.credentials.get("access_token", "")


class EnterpriseConnector:
    """Main enterprise integration manager"""
    
    def __init__(self):
        self.endpoints: Dict[str, IntegrationEndpoint] = {}
        self.connectors = {
            "splunk": SplunkConnector,
            "qradar": QRadarConnector,
            "servicenow": ServiceNowConnector,
            "phantom": PhantomConnector,
            "crowdstrike": CrowdStrikeConnector
        }
        self.rate_limiters = {}
        self.circuit_breakers = {}
        
    def register_endpoint(self, endpoint: IntegrationEndpoint):
        """Register integration endpoint"""
        self.endpoints[endpoint.integration_id] = endpoint
        self.rate_limiters[endpoint.integration_id] = []
        self.circuit_breakers[endpoint.integration_id] = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half-open
        }
        logger.info(f"Registered integration endpoint: {endpoint.name}")
    
    async def make_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Make authenticated request to external system"""
        endpoint = self.endpoints.get(request.endpoint_id)
        if not endpoint:
            return IntegrationResponse(
                success=False,
                status_code=404,
                error="Integration endpoint not found"
            )
        
        if not endpoint.enabled:
            return IntegrationResponse(
                success=False,
                status_code=503,
                error="Integration endpoint disabled"
            )
        
        # Check circuit breaker
        if not self._check_circuit_breaker(endpoint.integration_id):
            return IntegrationResponse(
                success=False,
                status_code=503,
                error="Circuit breaker open - too many failures"
            )
        
        # Check rate limiting
        if not self._check_rate_limit(endpoint.integration_id, endpoint.rate_limit_per_minute):
            return IntegrationResponse(
                success=False,
                status_code=429,
                error="Rate limit exceeded"
            )
        
        # Prepare request
        url = urljoin(endpoint.base_url, request.path)
        headers = {**endpoint.custom_headers}
        
        if request.custom_headers:
            headers.update(request.custom_headers)
        
        # Add authentication
        headers.update(self._prepare_auth_headers(endpoint.credentials))
        
        timeout = request.timeout_override or endpoint.timeout
        
        start_time = datetime.utcnow()
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=aiohttp.TCPConnector(ssl=self._create_ssl_context())
            ) as session:
                
                async with session.request(
                    method=request.method,
                    url=url,
                    json=request.data,
                    params=request.params,
                    headers=headers
                ) as response:
                    
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    try:
                        response_data = await response.json()
                    except:
                        response_data = await response.text()
                    
                    success = 200 <= response.status < 300
                    
                    if success:
                        endpoint.last_successful_call = datetime.utcnow()
                        endpoint.consecutive_failures = 0
                        self._record_success(endpoint.integration_id)
                    else:
                        endpoint.consecutive_failures += 1
                        self._record_failure(endpoint.integration_id)
                    
                    return IntegrationResponse(
                        success=success,
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        response_time=response_time
                    )
        
        except asyncio.TimeoutError:
            endpoint.consecutive_failures += 1
            self._record_failure(endpoint.integration_id)
            return IntegrationResponse(
                success=False,
                status_code=408,
                error="Request timeout",
                response_time=(datetime.utcnow() - start_time).total_seconds()
            )
        
        except Exception as e:
            endpoint.consecutive_failures += 1
            self._record_failure(endpoint.integration_id)
            return IntegrationResponse(
                success=False,
                status_code=500,
                error=str(e),
                response_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _prepare_auth_headers(self, credentials: IntegrationCredentials) -> Dict[str, str]:
        """Prepare authentication headers"""
        headers = {}
        
        if credentials.auth_type == AuthenticationType.API_KEY:
            api_key = credentials.credentials.get("api_key")
            key_header = credentials.credentials.get("key_header", "X-API-Key")
            if api_key:
                headers[key_header] = api_key
        
        elif credentials.auth_type == AuthenticationType.BASIC_AUTH:
            username = credentials.credentials.get("username")
            password = credentials.credentials.get("password")
            if username and password:
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_string}"
        
        elif credentials.auth_type == AuthenticationType.JWT:
            token = credentials.credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        return headers
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    def _check_rate_limit(self, endpoint_id: str, limit_per_minute: int) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.utcnow()
        rate_limiter = self.rate_limiters[endpoint_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - timedelta(minutes=1)
        self.rate_limiters[endpoint_id] = [
            timestamp for timestamp in rate_limiter
            if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.rate_limiters[endpoint_id]) < limit_per_minute:
            self.rate_limiters[endpoint_id].append(current_time)
            return True
        
        return False
    
    def _check_circuit_breaker(self, endpoint_id: str) -> bool:
        """Check circuit breaker state"""
        breaker = self.circuit_breakers[endpoint_id]
        current_time = datetime.utcnow()
        
        if breaker["state"] == "open":
            # Check if we should transition to half-open
            if breaker["last_failure"]:
                time_since_failure = (current_time - breaker["last_failure"]).total_seconds()
                if time_since_failure > 300:  # 5 minutes
                    breaker["state"] = "half-open"
                    return True
            return False
        
        return True
    
    def _record_success(self, endpoint_id: str):
        """Record successful request"""
        breaker = self.circuit_breakers[endpoint_id]
        breaker["failures"] = 0
        breaker["state"] = "closed"
    
    def _record_failure(self, endpoint_id: str):
        """Record failed request"""
        breaker = self.circuit_breakers[endpoint_id]
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit breaker after 5 consecutive failures
        if breaker["failures"] >= 5:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for endpoint {endpoint_id}")
    
    async def send_to_siem(self, event_data: Dict[str, Any], siem_type: str = "splunk") -> List[IntegrationResponse]:
        """Send event to configured SIEM systems"""
        responses = []
        
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.integration_type == IntegrationType.SIEM and endpoint.enabled:
                connector_class = self.connectors.get(siem_type.lower())
                if connector_class:
                    connector = connector_class(endpoint)
                    response = await connector.send_event(event_data)
                    responses.append(response)
        
        return responses
    
    async def create_incident(self, incident_data: Dict[str, Any]) -> List[IntegrationResponse]:
        """Create incident in ticketing systems"""
        responses = []
        
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.integration_type == IntegrationType.TICKETING and endpoint.enabled:
                if "servicenow" in endpoint.integration_id.lower():
                    connector = ServiceNowConnector(endpoint)
                    response = await connector.create_incident(incident_data)
                    responses.append(response)
        
        return responses
    
    async def trigger_soar_playbook(self, event_data: Dict[str, Any]) -> List[IntegrationResponse]:
        """Trigger SOAR playbook execution"""
        responses = []
        
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.integration_type == IntegrationType.SOAR and endpoint.enabled:
                if "phantom" in endpoint.integration_id.lower():
                    connector = PhantomConnector(endpoint)
                    # Create container first
                    container_response = await connector.create_container(event_data)
                    responses.append(container_response)
                    
                    # If container created successfully, run playbook
                    if container_response.success:
                        container_id = container_response.data.get("id")
                        playbook_id = endpoint.metadata.get("default_playbook_id")
                        if container_id and playbook_id:
                            playbook_response = await connector.run_playbook(container_id, playbook_id)
                            responses.append(playbook_response)
        
        return responses
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {}
        
        for endpoint_id, endpoint in self.endpoints.items():
            breaker = self.circuit_breakers[endpoint_id]
            rate_limiter = self.rate_limiters[endpoint_id]
            
            status[endpoint_id] = {
                "name": endpoint.name,
                "type": endpoint.integration_type.value,
                "enabled": endpoint.enabled,
                "last_successful_call": endpoint.last_successful_call.isoformat() if endpoint.last_successful_call else None,
                "consecutive_failures": endpoint.consecutive_failures,
                "circuit_breaker_state": breaker["state"],
                "current_rate_limit_usage": len(rate_limiter),
                "rate_limit_per_minute": endpoint.rate_limit_per_minute
            }
        
        return status


# Global enterprise connector instance
enterprise_connector: Optional[EnterpriseConnector] = None


def get_enterprise_connector() -> EnterpriseConnector:
    """Get global enterprise connector"""
    global enterprise_connector
    if enterprise_connector is None:
        enterprise_connector = EnterpriseConnector()
    return enterprise_connector


async def initialize_default_integrations():
    """Initialize common enterprise integrations"""
    connector = get_enterprise_connector()
    
    # Example Splunk integration
    splunk_endpoint = IntegrationEndpoint(
        integration_id="splunk_primary",
        name="Primary Splunk SIEM",
        integration_type=IntegrationType.SIEM,
        base_url="https://splunk.company.com:8088",
        credentials=IntegrationCredentials(
            auth_type=AuthenticationType.API_KEY,
            credentials={
                "token": "your-splunk-hec-token",
                "key_header": "Authorization"
            }
        )
    )
    
    # Example ServiceNow integration
    servicenow_endpoint = IntegrationEndpoint(
        integration_id="servicenow_itsm",
        name="ServiceNow ITSM",
        integration_type=IntegrationType.TICKETING,
        base_url="https://company.service-now.com",
        credentials=IntegrationCredentials(
            auth_type=AuthenticationType.BASIC_AUTH,
            credentials={
                "username": "xorb_integration",
                "password": "your-password"
            }
        )
    )
    
    connector.register_endpoint(splunk_endpoint)
    connector.register_endpoint(servicenow_endpoint)
    
    logger.info("Initialized default enterprise integrations")