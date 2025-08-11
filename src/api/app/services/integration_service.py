"""
Integration Service - Clean Architecture Service Layer
Orchestrates enterprise integrations using focused connectors
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..integrations.connector_factory import get_connector_manager, ConnectorManager
from ..integrations.base_connector import (
    IntegrationEndpoint,
    IntegrationRequest,
    IntegrationResponse,
    IntegrationType
)
from .interfaces import SecurityIntegrationService

logger = logging.getLogger(__name__)


class ProductionIntegrationService(SecurityIntegrationService):
    """
    Production integration service implementing clean architecture patterns.
    Manages enterprise security integrations through focused connectors.
    """
    
    def __init__(self, connector_manager: Optional[ConnectorManager] = None):
        self.connector_manager = connector_manager or get_connector_manager()
        self._event_routing: Dict[str, List[str]] = {}
    
    async def initialize(self) -> None:
        """Initialize the integration service"""
        logger.info("Initializing Production Integration Service")
        # Service is ready - connectors are registered on-demand
    
    async def register_integration(self, endpoint: IntegrationEndpoint) -> bool:
        """Register a new integration endpoint"""
        try:
            success = await self.connector_manager.register_endpoint(endpoint)
            
            if success:
                logger.info(f"Successfully registered integration: {endpoint.name}")
            else:
                logger.error(f"Failed to register integration: {endpoint.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error registering integration {endpoint.name}: {e}")
            return False
    
    async def send_security_event(
        self,
        event_data: Dict[str, Any],
        target_types: Optional[List[IntegrationType]] = None
    ) -> Dict[str, IntegrationResponse]:
        """Send security event to appropriate integrations"""
        try:
            results = {}
            
            # If no specific types requested, send to all healthy connectors
            if not target_types:
                target_types = [
                    IntegrationType.SIEM,
                    IntegrationType.SOAR,
                    IntegrationType.NOTIFICATION
                ]
            
            for integration_type in target_types:
                connectors = self.connector_manager.get_connectors_by_type(integration_type)
                
                for endpoint_id, connector in connectors.items():
                    try:
                        # Prepare integration-specific request
                        request = IntegrationRequest(
                            endpoint_id=endpoint_id,
                            method="POST",
                            path=self._get_event_path(integration_type),
                            data=event_data
                        )
                        
                        response = await connector.make_authenticated_request(request)
                        results[endpoint_id] = response
                        
                        if response.success:
                            logger.info(f"Successfully sent event to {endpoint_id}")
                        else:
                            logger.warning(f"Failed to send event to {endpoint_id}: {response.error}")
                            
                    except Exception as e:
                        logger.error(f"Error sending event to {endpoint_id}: {e}")
                        results[endpoint_id] = IntegrationResponse(
                            success=False,
                            status_code=500,
                            error=str(e)
                        )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in send_security_event: {e}")
            return {}
    
    async def create_incident(
        self,
        incident_data: Dict[str, Any]
    ) -> Dict[str, IntegrationResponse]:
        """Create incident in ticketing systems"""
        try:
            results = {}
            connectors = self.connector_manager.get_connectors_by_type(IntegrationType.TICKETING)
            
            for endpoint_id, connector in connectors.items():
                try:
                    # Use connector-specific incident creation method
                    if hasattr(connector, 'create_incident'):
                        response = await connector.create_incident(incident_data)
                    elif hasattr(connector, 'send_event'):
                        response = await connector.send_event(incident_data)
                    else:
                        # Fallback to generic request
                        request = IntegrationRequest(
                            endpoint_id=endpoint_id,
                            method="POST",
                            path="/api/incident",
                            data=incident_data
                        )
                        response = await connector.make_authenticated_request(request)
                    
                    results[endpoint_id] = response
                    
                    if response.success:
                        logger.info(f"Successfully created incident in {endpoint_id}")
                    else:
                        logger.warning(f"Failed to create incident in {endpoint_id}: {response.error}")
                        
                except Exception as e:
                    logger.error(f"Error creating incident in {endpoint_id}: {e}")
                    results[endpoint_id] = IntegrationResponse(
                        success=False,
                        status_code=500,
                        error=str(e)
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in create_incident: {e}")
            return {}
    
    async def block_threat_indicators(
        self,
        indicators: List[str],
        threat_type: str = "ip",
        reason: str = "XORB Threat Detection"
    ) -> Dict[str, IntegrationResponse]:
        """Block threat indicators on security devices"""
        try:
            results = {}
            
            # Get firewall and EDR connectors
            target_types = [IntegrationType.FIREWALL, IntegrationType.EDR]
            
            for integration_type in target_types:
                connectors = self.connector_manager.get_connectors_by_type(integration_type)
                
                for endpoint_id, connector in connectors.items():
                    try:
                        for indicator in indicators:
                            if threat_type == "ip" and hasattr(connector, 'block_ip'):
                                response = await connector.block_ip(indicator, reason)
                            elif hasattr(connector, 'block_indicator'):
                                response = await connector.block_indicator(indicator, threat_type, reason)
                            else:
                                # Generic blocking request
                                block_data = {
                                    "indicator": indicator,
                                    "type": threat_type,
                                    "action": "block",
                                    "reason": reason
                                }
                                
                                request = IntegrationRequest(
                                    endpoint_id=endpoint_id,
                                    method="POST",
                                    path="/api/block",
                                    data=block_data
                                )
                                response = await connector.make_authenticated_request(request)
                            
                            results[f"{endpoint_id}_{indicator}"] = response
                            
                            if response.success:
                                logger.info(f"Successfully blocked {indicator} on {endpoint_id}")
                            else:
                                logger.warning(f"Failed to block {indicator} on {endpoint_id}: {response.error}")
                                
                    except Exception as e:
                        logger.error(f"Error blocking indicators on {endpoint_id}: {e}")
                        results[endpoint_id] = IntegrationResponse(
                            success=False,
                            status_code=500,
                            error=str(e)
                        )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in block_threat_indicators: {e}")
            return {}
    
    async def trigger_response_automation(
        self,
        event_data: Dict[str, Any],
        playbook_name: Optional[str] = None
    ) -> Dict[str, IntegrationResponse]:
        """Trigger automated response in SOAR platforms"""
        try:
            results = {}
            connectors = self.connector_manager.get_connectors_by_type(IntegrationType.SOAR)
            
            for endpoint_id, connector in connectors.items():
                try:
                    # First create container/incident
                    if hasattr(connector, 'create_container'):
                        container_response = await connector.create_container(event_data)
                        results[f"{endpoint_id}_container"] = container_response
                        
                        if container_response.success and hasattr(connector, 'run_playbook'):
                            # Extract container ID and run playbook
                            container_id = container_response.data.get("id")
                            if container_id:
                                playbook_response = await connector.run_playbook(
                                    container_id,
                                    playbook_name=playbook_name
                                )
                                results[f"{endpoint_id}_playbook"] = playbook_response
                    
                    elif hasattr(connector, 'create_incident'):
                        incident_response = await connector.create_incident(event_data)
                        results[f"{endpoint_id}_incident"] = incident_response
                        
                    else:
                        # Generic automation trigger
                        automation_data = {
                            "event": event_data,
                            "playbook": playbook_name,
                            "trigger": "xorb_security_event"
                        }
                        
                        request = IntegrationRequest(
                            endpoint_id=endpoint_id,
                            method="POST",
                            path="/api/automation/trigger",
                            data=automation_data
                        )
                        response = await connector.make_authenticated_request(request)
                        results[endpoint_id] = response
                        
                except Exception as e:
                    logger.error(f"Error triggering automation on {endpoint_id}: {e}")
                    results[endpoint_id] = IntegrationResponse(
                        success=False,
                        status_code=500,
                        error=str(e)
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in trigger_response_automation: {e}")
            return {}
    
    async def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        try:
            # Get connector status
            connector_status = self.connector_manager.get_connector_status()
            
            # Run health checks
            health_results = await self.connector_manager.health_check_all()
            
            # Combine results
            for endpoint_id in connector_status:
                connector_status[endpoint_id]["health_check"] = health_results.get(endpoint_id, False)
                connector_status[endpoint_id]["last_health_check"] = datetime.utcnow().isoformat()
            
            return connector_status
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {}
    
    async def test_integration(self, endpoint_id: str) -> IntegrationResponse:
        """Test a specific integration"""
        try:
            connector = self.connector_manager.get_connector(endpoint_id)
            
            if not connector:
                return IntegrationResponse(
                    success=False,
                    status_code=404,
                    error="Integration not found"
                )
            
            # Run health check
            is_healthy = await connector.health_check()
            
            return IntegrationResponse(
                success=is_healthy,
                status_code=200 if is_healthy else 503,
                data={"health_check": is_healthy, "endpoint_id": endpoint_id}
            )
            
        except Exception as e:
            logger.error(f"Error testing integration {endpoint_id}: {e}")
            return IntegrationResponse(
                success=False,
                status_code=500,
                error=str(e)
            )
    
    def _get_event_path(self, integration_type: IntegrationType) -> str:
        """Get appropriate API path for event based on integration type"""
        path_mapping = {
            IntegrationType.SIEM: "/api/events",
            IntegrationType.SOAR: "/api/incidents",
            IntegrationType.TICKETING: "/api/incidents",
            IntegrationType.NOTIFICATION: "/api/notifications",
            IntegrationType.THREAT_INTELLIGENCE: "/api/indicators"
        }
        
        return path_mapping.get(integration_type, "/api/events")
    
    async def shutdown(self):
        """Shutdown integration service"""
        logger.info("Shutting down integration service...")
        await self.connector_manager.shutdown_all()
        logger.info("Integration service shutdown complete")