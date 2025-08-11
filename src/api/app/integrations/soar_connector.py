"""
SOAR Integration Connectors
Focused connectors for SOAR platforms (Phantom, Demisto, Swimlane)
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .base_connector import (
    BaseConnector,
    IntegrationRequest,
    IntegrationResponse,
    AuthenticationType
)

logger = logging.getLogger(__name__)


class PhantomConnector(BaseConnector):
    """Phantom SOAR integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Phantom requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Phantom connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/rest/system_info"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Phantom health check failed: {e}")
            return False
    
    async def create_container(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Create container (case) in Phantom"""
        container = {
            "name": event_data.get("title", "XORB Security Event"),
            "description": event_data.get("description", "Security event from XORB Platform"),
            "label": self.endpoint.metadata.get("default_label", "events"),
            "severity": self._map_severity(event_data.get("severity", "medium")),
            "sensitivity": self.endpoint.metadata.get("default_sensitivity", "amber"),
            "status": "new",
            "tags": event_data.get("tags", []),
            "source_data_identifier": event_data.get("event_id"),
            "data": {
                "source_system": event_data.get("source", "XORB"),
                "event_type": event_data.get("event_type"),
                "indicators": event_data.get("indicators", []),
                "metadata": event_data.get("metadata", {})
            }
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/rest/container",
            data=container
        )
        
        return await self.make_authenticated_request(request)
    
    async def add_artifacts(
        self, 
        container_id: int, 
        artifacts: List[Dict[str, Any]]
    ) -> IntegrationResponse:
        """Add artifacts to a container"""
        phantom_artifacts = []
        
        for artifact in artifacts:
            phantom_artifact = {
                "container_id": container_id,
                "name": artifact.get("name", "XORB Artifact"),
                "label": artifact.get("label", "indicator"),
                "source_data_identifier": artifact.get("identifier"),
                "cef": artifact.get("cef_data", {}),
                "data": artifact.get("data", {}),
                "tags": artifact.get("tags", [])
            }
            phantom_artifacts.append(phantom_artifact)
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/rest/artifact",
            data=phantom_artifacts
        )
        
        return await self.make_authenticated_request(request)
    
    async def run_playbook(
        self, 
        container_id: int, 
        playbook_id: Optional[int] = None,
        playbook_name: Optional[str] = None
    ) -> IntegrationResponse:
        """Run automated playbook in Phantom"""
        if not playbook_id and not playbook_name:
            playbook_id = self.endpoint.metadata.get("default_playbook_id")
        
        playbook_data = {
            "container_id": container_id,
            "scope": "all"
        }
        
        if playbook_id:
            playbook_data["playbook_id"] = playbook_id
        elif playbook_name:
            playbook_data["playbook"] = playbook_name
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/rest/playbook_run",
            data=playbook_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_container_status(self, container_id: int) -> IntegrationResponse:
        """Get container status and details"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/rest/container/{container_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def update_container_status(
        self, 
        container_id: int, 
        status: str,
        resolution: Optional[str] = None
    ) -> IntegrationResponse:
        """Update container status"""
        update_data = {"status": status}
        
        if resolution:
            update_data["resolution"] = resolution
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/rest/container/{container_id}",
            data=update_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_available_playbooks(self) -> IntegrationResponse:
        """Get list of available playbooks"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/rest/playbook",
            params={"page_size": 100}
        )
        
        return await self.make_authenticated_request(request)
    
    def _map_severity(self, xorb_severity: str) -> str:
        """Map XORB severity to Phantom severity"""
        mapping = {
            "critical": "high",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "info": "low"
        }
        return mapping.get(xorb_severity.lower(), "medium")


class DemistoConnector(BaseConnector):
    """Demisto (Cortex XSOAR) integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Demisto requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Demisto connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/about"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Demisto health check failed: {e}")
            return False
    
    async def create_incident(self, event_data: Dict[str, Any]) -> IntegrationResponse:
        """Create incident in Demisto"""
        incident = {
            "name": event_data.get("title", "XORB Security Incident"),
            "details": event_data.get("description", "Security incident from XORB Platform"),
            "type": self.endpoint.metadata.get("incident_type", "Unclassified"),
            "severity": self._map_severity_to_numeric(event_data.get("severity", "medium")),
            "labels": self._create_labels(event_data),
            "customFields": {
                "xorbsource": event_data.get("source", "XORB"),
                "xorbeventid": event_data.get("event_id"),
                "xorbeventtype": event_data.get("event_type"),
                "xorbindicators": json.dumps(event_data.get("indicators", [])),
                "xorbmetadata": json.dumps(event_data.get("metadata", {}))
            }
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/incident",
            data=incident
        )
        
        return await self.make_authenticated_request(request)
    
    async def run_playbook(
        self, 
        incident_id: str, 
        playbook_id: str
    ) -> IntegrationResponse:
        """Run playbook on incident"""
        playbook_data = {
            "playbookId": playbook_id,
            "incidentId": incident_id
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/playbook/execute",
            data=playbook_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_incident_status(self, incident_id: str) -> IntegrationResponse:
        """Get incident status and details"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/incident/{incident_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def add_evidence(
        self, 
        incident_id: str, 
        evidence_data: Dict[str, Any]
    ) -> IntegrationResponse:
        """Add evidence to incident"""
        evidence = {
            "incidentId": incident_id,
            "description": evidence_data.get("description"),
            "tags": evidence_data.get("tags", []),
            "data": evidence_data.get("data", {}),
            "entryType": evidence_data.get("entry_type", "note")
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/entry",
            data=evidence
        )
        
        return await self.make_authenticated_request(request)
    
    async def close_incident(
        self, 
        incident_id: str, 
        close_reason: str,
        close_notes: Optional[str] = None
    ) -> IntegrationResponse:
        """Close incident with reason"""
        close_data = {
            "status": 2,  # Closed
            "closeReason": close_reason
        }
        
        if close_notes:
            close_data["closeNotes"] = close_notes
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/incident/{incident_id}",
            data=close_data
        )
        
        return await self.make_authenticated_request(request)
    
    def _map_severity_to_numeric(self, xorb_severity: str) -> int:
        """Map XORB severity to Demisto numeric severity"""
        mapping = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "info": 0
        }
        return mapping.get(xorb_severity.lower(), 2)
    
    def _create_labels(self, event_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create Demisto labels from event data"""
        labels = [
            {"type": "Source", "value": event_data.get("source", "XORB")},
            {"type": "EventType", "value": event_data.get("event_type", "Unknown")}
        ]
        
        # Add indicator labels
        for indicator in event_data.get("indicators", []):
            labels.append({"type": "Indicator", "value": indicator})
        
        # Add custom labels from metadata
        for key, value in event_data.get("metadata", {}).items():
            if isinstance(value, str):
                labels.append({"type": key, "value": value})
        
        return labels


class SwimlaneConnector(BaseConnector):
    """Swimlane SOAR integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Swimlane requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Swimlane connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/api/user/profile"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Swimlane health check failed: {e}")
            return False
    
    async def create_record(
        self, 
        app_id: str, 
        event_data: Dict[str, Any]
    ) -> IntegrationResponse:
        """Create record in Swimlane application"""
        record = {
            "applicationId": app_id,
            "values": {
                "Title": event_data.get("title", "XORB Security Event"),
                "Description": event_data.get("description", "Security event from XORB Platform"),
                "Severity": event_data.get("severity", "medium").title(),
                "Source": event_data.get("source", "XORB"),
                "Event ID": event_data.get("event_id"),
                "Event Type": event_data.get("event_type"),
                "Indicators": "\n".join(event_data.get("indicators", [])),
                "Timestamp": event_data.get("timestamp", datetime.utcnow()).isoformat()
            }
        }
        
        # Add custom fields from metadata
        for key, value in event_data.get("metadata", {}).items():
            if isinstance(value, (str, int, float, bool)):
                record["values"][key] = value
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/app/{}/record".format(app_id),
            data=record
        )
        
        return await self.make_authenticated_request(request)
    
    async def update_record(
        self, 
        app_id: str, 
        record_id: str, 
        updates: Dict[str, Any]
    ) -> IntegrationResponse:
        """Update record in Swimlane"""
        update_data = {
            "values": updates
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PUT",
            path=f"/api/app/{app_id}/record/{record_id}",
            data=update_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def execute_workflow(
        self, 
        app_id: str, 
        record_id: str, 
        workflow_id: str
    ) -> IntegrationResponse:
        """Execute workflow on record"""
        workflow_data = {
            "workflowId": workflow_id,
            "recordId": record_id
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/app/{app_id}/workflow/execute",
            data=workflow_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_record(self, app_id: str, record_id: str) -> IntegrationResponse:
        """Get record details"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/api/app/{app_id}/record/{record_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def search_records(
        self, 
        app_id: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> IntegrationResponse:
        """Search records in application"""
        search_data = {
            "applicationId": app_id,
            "limit": filters.get("limit", 100) if filters else 100
        }
        
        if filters and "conditions" in filters:
            search_data["filters"] = filters["conditions"]
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/search",
            data=search_data
        )
        
        return await self.make_authenticated_request(request)