"""
SIEM Integration Connectors
Focused connectors for SIEM platforms (Splunk, QRadar, Sentinel)
"""

import json
import logging
import base64
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Any, Optional
from .base_connector import (
    BaseConnector,
    IntegrationRequest,
    IntegrationResponse,
    AuthenticationType
)

logger = logging.getLogger(__name__)


class SplunkConnector(BaseConnector):
    """Splunk SIEM integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Splunk requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _prepare_auth_headers(self) -> Dict[str, str]:
        """Prepare Splunk-specific authentication headers"""
        headers = {}
        credentials = self.endpoint.credentials
        
        if credentials.auth_type == AuthenticationType.API_KEY:
            token = credentials.credentials.get("token")
            if token:
                headers["Authorization"] = f"Splunk {token}"
        else:
            # Fall back to base implementation
            headers.update(await super()._prepare_auth_headers())
        
        return headers
    
    async def health_check(self) -> bool:
        """Check Splunk connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/services/server/info"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Splunk health check failed: {e}")
            return False
    
    async def send_event(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Send security event to Splunk via HTTP Event Collector"""
        splunk_event = {
            "time": event_data.get("timestamp", datetime.utcnow().timestamp()),
            "event": event_data,
            "source": "xorb_platform",
            "sourcetype": "xorb:security_event",
            "index": self.endpoint.metadata.get("index", "main")
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/services/collector/event",
            data=splunk_event
        )
        
        return await self.make_authenticated_request(request)
    
    async def search_events(
        self, 
        query: str, 
        earliest_time: str = "-1h",
        latest_time: str = "now"
    ) -> IntegrationResponse:
        """Search Splunk for events"""
        search_data = {
            "search": f"search {query}",
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "output_mode": "json",
            "count": self.endpoint.metadata.get("search_limit", 1000)
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/services/search/jobs/oneshot",
            data=search_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def create_saved_search(
        self, 
        name: str, 
        search_query: str, 
        schedule: Optional[str] = None
    ) -> IntegrationResponse:
        """Create a saved search in Splunk"""
        saved_search_data = {
            "name": name,
            "search": search_query,
            "is_scheduled": 1 if schedule else 0
        }
        
        if schedule:
            saved_search_data["cron_schedule"] = schedule
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/services/saved/searches",
            data=saved_search_data
        )
        
        return await self.make_authenticated_request(request)


class QRadarConnector(BaseConnector):
    """IBM QRadar SIEM integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for QRadar requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Version": "13.0"
        }
    
    async def _prepare_auth_headers(self) -> Dict[str, str]:
        """Prepare QRadar-specific authentication headers"""
        headers = {}
        credentials = self.endpoint.credentials
        
        if credentials.auth_type == AuthenticationType.API_KEY:
            sec_token = credentials.credentials.get("sec_token")
            if sec_token:
                headers["SEC"] = sec_token
        else:
            # Fall back to base implementation
            headers.update(await super()._prepare_auth_headers())
        
        return headers
    
    async def health_check(self) -> bool:
        """Check QRadar connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/api/help/versions"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"QRadar health check failed: {e}")
            return False
    
    async def send_event(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Send event to QRadar"""
        qradar_event = {
            "events": [{
                "qid": self.endpoint.metadata.get("custom_qid", 28250004),
                "message": json.dumps(event_data),
                "properties": {
                    "sourceip": event_data.get("source_ip"),
                    "destinationip": event_data.get("destination_ip"),
                    "username": event_data.get("user"),
                    "magnitude": self._severity_to_magnitude(event_data.get("severity", "medium"))
                }
            }]
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/siem/events",
            data=qradar_event
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_offenses(
        self, 
        start_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResponse:
        """Get offenses from QRadar"""
        params = {
            "sort": "-start_time",
            "limit": filters.get("limit", 100) if filters else 100
        }
        
        if start_time:
            params["filter"] = f"start_time >= {int(start_time.timestamp() * 1000)}"
        
        if filters and "status" in filters:
            status_filter = f"status = '{filters['status']}'"
            if "filter" in params:
                params["filter"] += f" and {status_filter}"
            else:
                params["filter"] = status_filter
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/api/siem/offenses",
            params=params
        )
        
        return await self.make_authenticated_request(request)
    
    async def close_offense(self, offense_id: int, reason: str) -> IntegrationResponse:
        """Close an offense in QRadar"""
        close_data = {
            "closing_reason_id": self.endpoint.metadata.get("default_closing_reason_id", 1),
            "status": "CLOSED"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/siem/offenses/{offense_id}",
            data=close_data
        )
        
        return await self.make_authenticated_request(request)
    
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


class SentinelConnector(BaseConnector):
    """Microsoft Sentinel SIEM integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Sentinel requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Sentinel connectivity and health"""
        try:
            workspace_id = self.endpoint.credentials.credentials.get("workspace_id")
            if not workspace_id:
                return False
            
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path=f"/v1/workspaces/{workspace_id}/metadata"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Sentinel health check failed: {e}")
            return False
    
    async def send_event(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Send event to Sentinel via Log Analytics Data Collector API"""
        workspace_id = self.endpoint.credentials.credentials.get("workspace_id")
        shared_key = self.endpoint.credentials.credentials.get("shared_key")
        
        if not workspace_id or not shared_key:
            return IntegrationResponse(
                success=False,
                status_code=400,
                error="Missing workspace_id or shared_key"
            )
        
        # Prepare event for Log Analytics
        sentinel_event = [{
            "TimeGenerated": event_data.get("timestamp", datetime.utcnow()).isoformat(),
            "EventId": event_data.get("event_id"),
            "EventType": event_data.get("event_type"),
            "Severity": event_data.get("severity"),
            "Title": event_data.get("title"),
            "Description": event_data.get("description"),
            "Indicators": json.dumps(event_data.get("indicators", [])),
            "Metadata": json.dumps(event_data.get("metadata", {})),
            "SourceSystem": event_data.get("source", "XORB")
        }]
        
        # Build authentication signature
        json_data = json.dumps(sentinel_event)
        timestamp = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        string_to_hash = f"POST\n{len(json_data)}\napplication/json\nx-ms-date:{timestamp}\n/api/logs"
        bytes_to_hash = bytes(string_to_hash, 'UTF-8')
        decoded_key = base64.b64decode(shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        )
        authorization = f"SharedKey {workspace_id}:{encoded_hash.decode()}"
        
        log_type = self.endpoint.metadata.get("log_type", "XORBSecurityEvents")
        
        headers = {
            "Authorization": authorization,
            "Log-Type": log_type,
            "x-ms-date": timestamp,
            "time-generated-field": "TimeGenerated"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/logs?api-version=2016-04-01",
            data=json_data,
            custom_headers=headers
        )
        
        return await self.make_authenticated_request(request)
    
    async def query_events(
        self, 
        kql_query: str,
        timespan: Optional[str] = None
    ) -> IntegrationResponse:
        """Query events from Sentinel using KQL"""
        workspace_id = self.endpoint.credentials.credentials.get("workspace_id")
        
        if not workspace_id:
            return IntegrationResponse(
                success=False,
                status_code=400,
                error="Missing workspace_id"
            )
        
        query_data = {"query": kql_query}
        
        if timespan:
            query_data["timespan"] = timespan
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/v1/workspaces/{workspace_id}/query",
            data=query_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def create_incident(
        self, 
        title: str, 
        description: str, 
        severity: str = "Medium"
    ) -> IntegrationResponse:
        """Create an incident in Sentinel"""
        workspace_id = self.endpoint.credentials.credentials.get("workspace_id")
        resource_group = self.endpoint.credentials.credentials.get("resource_group")
        subscription_id = self.endpoint.credentials.credentials.get("subscription_id")
        
        if not all([workspace_id, resource_group, subscription_id]):
            return IntegrationResponse(
                success=False,
                status_code=400,
                error="Missing required Azure credentials"
            )
        
        incident_data = {
            "properties": {
                "title": title,
                "description": description,
                "severity": severity,
                "status": "New",
                "owner": {
                    "objectId": self.endpoint.metadata.get("default_owner_id")
                }
            }
        }
        
        path = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights/workspaces/{workspace_id}/providers/Microsoft.SecurityInsights/incidents"
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PUT",
            path=path,
            data=incident_data,
            custom_headers={"api-version": "2021-10-01"}
        )
        
        return await self.make_authenticated_request(request)