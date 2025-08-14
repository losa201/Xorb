"""
Enterprise Integration Connectors for XORB Platform
Strategic integrations with enterprise security platforms and tools
"""

import asyncio
import json
import logging
import aiohttp
import xmltodict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hmac
import hashlib
import base64
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class IntegrationCredentials:
    """Integration credentials configuration"""
    platform: str
    auth_type: str  # api_key, oauth2, basic, token, certificate
    credentials: Dict[str, Any]
    endpoint_base: str
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class SecurityEvent:
    """Standardized security event structure"""
    event_id: str
    timestamp: datetime
    source: str
    event_type: str
    severity: str
    title: str
    description: str
    indicators: List[str]
    metadata: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class IntegrationResult:
    """Integration operation result"""
    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None
    response_time_ms: Optional[float] = None


class BaseIntegrationConnector(ABC):
    """Base class for enterprise integration connectors"""

    def __init__(self, credentials: IntegrationCredentials):
        self.credentials = credentials
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_health_check = None
        self.is_healthy = False

    async def initialize(self):
        """Initialize the connector"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.credentials.timeout_seconds)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_default_headers()
        )

        # Perform initial health check
        await self.health_check()

    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the integration is healthy"""
        pass

    @abstractmethod
    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Send security event to the platform"""
        pass

    @abstractmethod
    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query events from the platform"""
        pass

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> IntegrationResult:
        """Make HTTP request with retry logic"""

        url = f"{self.credentials.endpoint_base.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = self._get_default_headers()
        if headers:
            request_headers.update(headers)

        start_time = asyncio.get_event_loop().time()

        for attempt in range(self.credentials.retry_attempts):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers
                ) as response:
                    response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                    if response.status < 400:
                        response_data = await response.json() if response.content_type == 'application/json' else await response.text()

                        return IntegrationResult(
                            success=True,
                            message=f"Request successful: {response.status}",
                            data=response_data,
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(f"Request failed (attempt {attempt + 1}): {response.status} - {error_text}")

                        if attempt == self.credentials.retry_attempts - 1:
                            return IntegrationResult(
                                success=False,
                                message=f"Request failed: {response.status}",
                                error_code=str(response.status),
                                response_time_ms=response_time
                            )

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")

                if attempt == self.credentials.retry_attempts - 1:
                    return IntegrationResult(
                        success=False,
                        message=f"Request error: {str(e)}",
                        error_code="connection_error"
                    )

                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

        return IntegrationResult(
            success=False,
            message="Maximum retry attempts exceeded"
        )


class SplunkConnector(BaseIntegrationConnector):
    """Splunk SIEM integration connector"""

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Splunk {self.credentials.credentials['token']}",
            "Content-Type": "application/json"
        }

    async def health_check(self) -> bool:
        """Check Splunk connectivity"""
        try:
            result = await self._make_request("GET", "services/server/info")
            self.is_healthy = result.success
            self.last_health_check = datetime.utcnow()
            return result.success
        except Exception as e:
            logger.error(f"Splunk health check failed: {e}")
            self.is_healthy = False
            return False

    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Send event to Splunk via HTTP Event Collector"""

        splunk_event = {
            "time": event.timestamp.timestamp(),
            "sourcetype": "xorb:security",
            "source": "xorb_platform",
            "index": self.credentials.credentials.get("index", "main"),
            "event": {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity,
                "title": event.title,
                "description": event.description,
                "indicators": event.indicators,
                "metadata": event.metadata,
                "source_system": event.source
            }
        }

        return await self._make_request(
            "POST",
            "services/collector/event",
            data=splunk_event
        )

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query events from Splunk"""

        # Build search query
        search_query = f'search index=* earliest="{start_time.isoformat()}" latest="{end_time.isoformat()}"'

        if filters:
            for key, value in filters.items():
                search_query += f' {key}="{value}"'

        search_data = {
            "search": search_query,
            "output_mode": "json",
            "count": filters.get("limit", 1000) if filters else 1000
        }

        return await self._make_request(
            "POST",
            "services/search/jobs/oneshot",
            data=search_data
        )


class QRadarConnector(BaseIntegrationConnector):
    """IBM QRadar SIEM integration connector"""

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "SEC": self.credentials.credentials['sec_token'],
            "Content-Type": "application/json",
            "Version": "13.0"
        }

    async def health_check(self) -> bool:
        """Check QRadar connectivity"""
        try:
            result = await self._make_request("GET", "api/help/versions")
            self.is_healthy = result.success
            self.last_health_check = datetime.utcnow()
            return result.success
        except Exception as e:
            logger.error(f"QRadar health check failed: {e}")
            self.is_healthy = False
            return False

    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Send event to QRadar via custom property"""

        # Create custom event in QRadar format
        qradar_event = {
            "properties": {
                "xorb_event_id": event.event_id,
                "xorb_event_type": event.event_type,
                "xorb_severity": event.severity,
                "xorb_title": event.title,
                "xorb_description": event.description,
                "xorb_indicators": ",".join(event.indicators),
                "xorb_source": event.source
            },
            "timestamp": int(event.timestamp.timestamp() * 1000)
        }

        return await self._make_request(
            "POST",
            "api/siem/offenses",
            data=qradar_event
        )

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query events from QRadar"""

        params = {
            "filter": f"start_time >= {int(start_time.timestamp() * 1000)} and start_time <= {int(end_time.timestamp() * 1000)}",
            "sort": "-start_time",
            "limit": filters.get("limit", 1000) if filters else 1000
        }

        return await self._make_request("GET", "api/siem/offenses", params=params)


class SentinelConnector(BaseIntegrationConnector):
    """Microsoft Sentinel integration connector"""

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.credentials['access_token']}",
            "Content-Type": "application/json"
        }

    async def health_check(self) -> bool:
        """Check Sentinel connectivity"""
        try:
            # Check workspace availability
            workspace_id = self.credentials.credentials['workspace_id']
            result = await self._make_request("GET", f"subscriptions/{workspace_id}/providers/Microsoft.OperationalInsights/workspaces")
            self.is_healthy = result.success
            self.last_health_check = datetime.utcnow()
            return result.success
        except Exception as e:
            logger.error(f"Sentinel health check failed: {e}")
            self.is_healthy = False
            return False

    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Send event to Sentinel via Log Analytics API"""

        sentinel_event = [{
            "TimeGenerated": event.timestamp.isoformat(),
            "EventId": event.event_id,
            "EventType": event.event_type,
            "Severity": event.severity,
            "Title": event.title,
            "Description": event.description,
            "Indicators": json.dumps(event.indicators),
            "Metadata": json.dumps(event.metadata),
            "SourceSystem": event.source
        }]

        # Log Analytics Data Collector API
        workspace_id = self.credentials.credentials['workspace_id']
        shared_key = self.credentials.credentials['shared_key']
        log_type = "XORBSecurityEvents"

        # Build signature
        json_data = json.dumps(sentinel_event)
        timestamp = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        string_to_hash = f"POST\n{len(json_data)}\napplication/json\nx-ms-date:{timestamp}\n/api/logs"
        bytes_to_hash = bytes(string_to_hash, 'UTF-8')
        decoded_key = base64.b64decode(shared_key)
        encoded_hash = base64.b64encode(hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest())
        authorization = f"SharedKey {workspace_id}:{encoded_hash.decode()}"

        headers = {
            "Authorization": authorization,
            "Log-Type": log_type,
            "x-ms-date": timestamp,
            "time-generated-field": "TimeGenerated"
        }

        return await self._make_request(
            "POST",
            f"api/logs?api-version=2016-04-01",
            data=json_data,
            headers=headers
        )

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query events from Sentinel"""

        # KQL query for Sentinel
        query = f"""
        XORBSecurityEvents_CL
        | where TimeGenerated between (datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
        | order by TimeGenerated desc
        | limit {filters.get("limit", 1000) if filters else 1000}
        """

        query_data = {"query": query}
        workspace_id = self.credentials.credentials['workspace_id']

        return await self._make_request(
            "POST",
            f"v1/workspaces/{workspace_id}/query",
            data=query_data
        )


class ServiceNowConnector(BaseIntegrationConnector):
    """ServiceNow ITSM integration connector"""

    def _get_default_headers(self) -> Dict[str, str]:
        credentials = self.credentials.credentials
        auth_string = f"{credentials['username']}:{credentials['password']}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        return {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def health_check(self) -> bool:
        """Check ServiceNow connectivity"""
        try:
            result = await self._make_request("GET", "api/now/table/sys_user?sysparm_limit=1")
            self.is_healthy = result.success
            self.last_health_check = datetime.utcnow()
            return result.success
        except Exception as e:
            logger.error(f"ServiceNow health check failed: {e}")
            self.is_healthy = False
            return False

    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Create incident in ServiceNow"""

        # Map severity to ServiceNow impact/urgency
        severity_mapping = {
            "critical": {"impact": "1", "urgency": "1"},
            "high": {"impact": "2", "urgency": "2"},
            "medium": {"impact": "3", "urgency": "3"},
            "low": {"impact": "3", "urgency": "3"}
        }

        severity_config = severity_mapping.get(event.severity.lower(), {"impact": "3", "urgency": "3"})

        incident_data = {
            "short_description": f"XORB Security Alert: {event.title}",
            "description": f"{event.description}\n\nIndicators: {', '.join(event.indicators)}\n\nEvent ID: {event.event_id}",
            "impact": severity_config["impact"],
            "urgency": severity_config["urgency"],
            "category": "Security",
            "subcategory": "Security Incident",
            "caller_id": self.credentials.credentials.get("caller_id", "admin"),
            "assignment_group": self.credentials.credentials.get("assignment_group", "Security Team"),
            "work_notes": f"Security event detected by XORB Platform\nSource: {event.source}\nTimestamp: {event.timestamp.isoformat()}"
        }

        return await self._make_request(
            "POST",
            "api/now/table/incident",
            data=incident_data
        )

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query incidents from ServiceNow"""

        params = {
            "sysparm_query": f"sys_created_on>={start_time.isoformat()}^sys_created_on<={end_time.isoformat()}^category=Security",
            "sysparm_limit": filters.get("limit", 1000) if filters else 1000,
            "sysparm_order_by": "sys_created_on"
        }

        return await self._make_request("GET", "api/now/table/incident", params=params)


class SlackConnector(BaseIntegrationConnector):
    """Slack integration connector for notifications"""

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.credentials['bot_token']}",
            "Content-Type": "application/json"
        }

    async def health_check(self) -> bool:
        """Check Slack connectivity"""
        try:
            result = await self._make_request("GET", "api/auth.test")
            self.is_healthy = result.success
            self.last_health_check = datetime.utcnow()
            return result.success
        except Exception as e:
            logger.error(f"Slack health check failed: {e}")
            self.is_healthy = False
            return False

    async def send_event(self, event: SecurityEvent) -> IntegrationResult:
        """Send notification to Slack"""

        # Map severity to colors
        color_mapping = {
            "critical": "#FF0000",
            "high": "#FF8C00",
            "medium": "#FFD700",
            "low": "#32CD32"
        }

        color = color_mapping.get(event.severity.lower(), "#808080")

        # Build Slack message
        message = {
            "channel": self.credentials.credentials['channel'],
            "attachments": [{
                "color": color,
                "title": f"🚨 Security Alert: {event.title}",
                "title_link": f"https://xorb_platform.com/events/{event.event_id}",
                "text": event.description,
                "fields": [
                    {
                        "title": "Severity",
                        "value": event.severity.upper(),
                        "short": True
                    },
                    {
                        "title": "Source",
                        "value": event.source,
                        "short": True
                    },
                    {
                        "title": "Event Type",
                        "value": event.event_type,
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": event.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True
                    }
                ],
                "footer": "XORB Security Platform",
                "ts": int(event.timestamp.timestamp())
            }]
        }

        if event.indicators:
            message["attachments"][0]["fields"].append({
                "title": "Indicators",
                "value": ", ".join(event.indicators[:5]),  # Limit to first 5
                "short": False
            })

        return await self._make_request("POST", "api/chat.postMessage", data=message)

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Query messages from Slack (not typically used for events)"""

        params = {
            "channel": self.credentials.credentials['channel'],
            "oldest": str(int(start_time.timestamp())),
            "latest": str(int(end_time.timestamp())),
            "limit": filters.get("limit", 100) if filters else 100
        }

        return await self._make_request("GET", "api/conversations.history", params=params)


class EnterpriseIntegrationManager:
    """
    Manages enterprise integrations for XORB Platform
    Provides unified interface for multiple enterprise platforms
    """

    def __init__(self):
        self.connectors: Dict[str, BaseIntegrationConnector] = {}
        self.event_routing: Dict[str, List[str]] = {}  # event_type -> list of connector names
        self.health_status: Dict[str, Dict[str, Any]] = {}

    async def register_connector(self, name: str, connector: BaseIntegrationConnector):
        """Register an integration connector"""

        await connector.initialize()
        self.connectors[name] = connector

        # Initial health check
        await self.update_health_status(name)

        logger.info(f"Registered integration connector: {name}")

    async def unregister_connector(self, name: str):
        """Unregister an integration connector"""

        if name in self.connectors:
            await self.connectors[name].close()
            del self.connectors[name]

            if name in self.health_status:
                del self.health_status[name]

            logger.info(f"Unregistered integration connector: {name}")

    def configure_event_routing(self, event_type: str, connector_names: List[str]):
        """Configure which connectors receive specific event types"""

        self.event_routing[event_type] = connector_names
        logger.info(f"Configured routing for {event_type}: {connector_names}")

    async def send_security_event(self, event: SecurityEvent) -> Dict[str, IntegrationResult]:
        """Send security event to appropriate connectors"""

        results = {}

        # Get connectors for this event type
        target_connectors = self.event_routing.get(event.event_type, [])

        # If no specific routing, send to all healthy connectors
        if not target_connectors:
            target_connectors = [name for name, connector in self.connectors.items()
                               if self.health_status.get(name, {}).get("is_healthy", False)]

        # Send to each target connector
        tasks = []
        connector_names = []

        for connector_name in target_connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]
                tasks.append(connector.send_event(event))
                connector_names.append(connector_name)

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(task_results):
                connector_name = connector_names[i]
                if isinstance(result, Exception):
                    results[connector_name] = IntegrationResult(
                        success=False,
                        message=f"Exception: {str(result)}"
                    )
                else:
                    results[connector_name] = result

        return results

    async def query_events_from_all(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, IntegrationResult]:
        """Query events from all connectors"""

        results = {}
        tasks = []
        connector_names = []

        for name, connector in self.connectors.items():
            if self.health_status.get(name, {}).get("is_healthy", False):
                tasks.append(connector.query_events(start_time, end_time, filters))
                connector_names.append(name)

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(task_results):
                connector_name = connector_names[i]
                if isinstance(result, Exception):
                    results[connector_name] = IntegrationResult(
                        success=False,
                        message=f"Exception: {str(result)}"
                    )
                else:
                    results[connector_name] = result

        return results

    async def update_health_status(self, connector_name: str):
        """Update health status for a specific connector"""

        if connector_name not in self.connectors:
            return

        connector = self.connectors[connector_name]
        is_healthy = await connector.health_check()

        self.health_status[connector_name] = {
            "is_healthy": is_healthy,
            "last_check": datetime.utcnow().isoformat(),
            "platform": connector.credentials.platform
        }

    async def update_all_health_status(self):
        """Update health status for all connectors"""

        tasks = []
        for connector_name in self.connectors.keys():
            tasks.append(self.update_health_status(connector_name))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all connectors"""
        return self.health_status.copy()

    def get_available_connectors(self) -> List[str]:
        """Get list of registered connector names"""
        return list(self.connectors.keys())

    def get_healthy_connectors(self) -> List[str]:
        """Get list of healthy connector names"""
        return [name for name, status in self.health_status.items()
                if status.get("is_healthy", False)]

    async def shutdown(self):
        """Shutdown all connectors"""

        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.connectors.clear()
        self.health_status.clear()

        logger.info("Enterprise integration manager shutdown completed")
