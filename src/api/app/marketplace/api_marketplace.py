"""
XORB API Marketplace
Enables third-party integrations and partner ecosystem
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of marketplace integrations"""
    SIEM = "siem"
    TICKETING = "ticketing"
    NOTIFICATION = "notification"
    THREAT_INTEL = "threat_intelligence"
    VULNERABILITY = "vulnerability_scanner"
    COMPLIANCE = "compliance"
    AUTHENTICATION = "authentication"
    MONITORING = "monitoring"


class IntegrationStatus(Enum):
    """Integration deployment status"""
    AVAILABLE = "available"
    INSTALLED = "installed"
    CONFIGURED = "configured"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class Integration:
    """Marketplace integration definition"""
    integration_id: str
    name: str
    description: str
    vendor: str
    version: str
    type: IntegrationType
    status: IntegrationStatus
    logo_url: str
    documentation_url: str
    api_endpoints: List[str]
    required_scopes: List[str]
    pricing_model: str
    configuration_schema: Dict[str, Any]
    webhooks: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class IntegrationInstance:
    """Deployed integration instance"""
    instance_id: str
    integration_id: str
    tenant_id: str
    name: str
    configuration: Dict[str, Any]
    status: IntegrationStatus
    last_sync: Optional[datetime]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class APIMarketplace:
    """XORB API Marketplace management"""
    
    def __init__(self):
        self.integrations = self._load_marketplace_integrations()
        self.instances = {}  # tenant_id -> List[IntegrationInstance]
        
    def _load_marketplace_integrations(self) -> List[Integration]:
        """Load available marketplace integrations"""
        return [
            # SIEM Integrations
            Integration(
                integration_id="splunk-enterprise",
                name="Splunk Enterprise",
                description="Connect XORB threat intelligence to Splunk SIEM",
                vendor="Splunk Inc.",
                version="1.2.0",
                type=IntegrationType.SIEM,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/splunk.png",
                documentation_url="https://docs.xorb.io/integrations/splunk",
                api_endpoints=["/api/v1/siem/splunk/events", "/api/v1/siem/splunk/alerts"],
                required_scopes=["siem:write", "threat_intel:read"],
                pricing_model="enterprise",
                configuration_schema={
                    "splunk_host": {"type": "string", "required": True},
                    "splunk_token": {"type": "string", "required": True, "sensitive": True},
                    "index_name": {"type": "string", "default": "xorb_security"},
                    "batch_size": {"type": "integer", "default": 100, "min": 1, "max": 1000}
                },
                webhooks=[
                    {"event": "threat_detected", "url_pattern": "/webhook/splunk/threat"}
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            Integration(
                integration_id="qradar-siem",
                name="IBM QRadar SIEM",
                description="Stream security events to IBM QRadar",
                vendor="IBM",
                version="2.1.0",
                type=IntegrationType.SIEM,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/qradar.png",
                documentation_url="https://docs.xorb.io/integrations/qradar",
                api_endpoints=["/api/v1/siem/qradar/events"],
                required_scopes=["siem:write", "events:read"],
                pricing_model="enterprise",
                configuration_schema={
                    "qradar_host": {"type": "string", "required": True},
                    "api_token": {"type": "string", "required": True, "sensitive": True},
                    "source_type": {"type": "string", "default": "XORB Security Platform"}
                },
                webhooks=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            # Ticketing Systems
            Integration(
                integration_id="jira-service-desk",
                name="Jira Service Management",
                description="Create security incidents in Jira automatically",
                vendor="Atlassian",
                version="3.0.1",
                type=IntegrationType.TICKETING,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/jira.png",
                documentation_url="https://docs.xorb.io/integrations/jira",
                api_endpoints=["/api/v1/ticketing/jira/incidents"],
                required_scopes=["incidents:write", "tickets:create"],
                pricing_model="free",
                configuration_schema={
                    "jira_url": {"type": "string", "required": True},
                    "username": {"type": "string", "required": True},
                    "api_token": {"type": "string", "required": True, "sensitive": True},
                    "project_key": {"type": "string", "required": True},
                    "issue_type": {"type": "string", "default": "Security Incident"}
                },
                webhooks=[
                    {"event": "incident_created", "url_pattern": "/webhook/jira/incident"}
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            Integration(
                integration_id="servicenow-itsm",
                name="ServiceNow ITSM",
                description="Integrate with ServiceNow for incident management",
                vendor="ServiceNow",
                version="1.5.0",
                type=IntegrationType.TICKETING,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/servicenow.png",
                documentation_url="https://docs.xorb.io/integrations/servicenow",
                api_endpoints=["/api/v1/ticketing/servicenow/incidents"],
                required_scopes=["incidents:write", "tickets:create"],
                pricing_model="enterprise",
                configuration_schema={
                    "instance_url": {"type": "string", "required": True},
                    "username": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True, "sensitive": True},
                    "assignment_group": {"type": "string", "default": "Security Team"}
                },
                webhooks=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            # Threat Intelligence
            Integration(
                integration_id="misp-threat-intel",
                name="MISP Threat Intelligence",
                description="Share and receive threat intelligence via MISP",
                vendor="MISP Project",
                version="2.4.0",
                type=IntegrationType.THREAT_INTEL,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/misp.png",
                documentation_url="https://docs.xorb.io/integrations/misp",
                api_endpoints=["/api/v1/threat-intel/misp/indicators"],
                required_scopes=["threat_intel:read", "threat_intel:write"],
                pricing_model="free",
                configuration_schema={
                    "misp_url": {"type": "string", "required": True},
                    "api_key": {"type": "string", "required": True, "sensitive": True},
                    "sharing_group": {"type": "string", "default": "Security Community"}
                },
                webhooks=[
                    {"event": "new_ioc", "url_pattern": "/webhook/misp/indicator"}
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            # Notifications
            Integration(
                integration_id="slack-notifications",
                name="Slack Notifications",
                description="Send security alerts to Slack channels",
                vendor="Slack Technologies",
                version="1.8.0",
                type=IntegrationType.NOTIFICATION,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/slack.png",
                documentation_url="https://docs.xorb.io/integrations/slack",
                api_endpoints=["/api/v1/notifications/slack/alerts"],
                required_scopes=["notifications:write"],
                pricing_model="free",
                configuration_schema={
                    "webhook_url": {"type": "string", "required": True, "sensitive": True},
                    "channel": {"type": "string", "default": "#security-alerts"},
                    "alert_levels": {"type": "array", "default": ["high", "critical"]}
                },
                webhooks=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            
            Integration(
                integration_id="pagerduty-alerts",
                name="PagerDuty Incident Management",
                description="Create PagerDuty incidents for critical security events",
                vendor="PagerDuty",
                version="2.0.0",
                type=IntegrationType.NOTIFICATION,
                status=IntegrationStatus.AVAILABLE,
                logo_url="https://marketplace.xorb.io/logos/pagerduty.png",
                documentation_url="https://docs.xorb.io/integrations/pagerduty",
                api_endpoints=["/api/v1/notifications/pagerduty/incidents"],
                required_scopes=["notifications:write", "incidents:create"],
                pricing_model="professional",
                configuration_schema={
                    "routing_key": {"type": "string", "required": True, "sensitive": True},
                    "escalation_policy": {"type": "string", "required": True},
                    "severity_mapping": {"type": "object", "default": {"critical": "critical", "high": "error"}}
                },
                webhooks=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        
    async def get_available_integrations(self, type_filter: Optional[IntegrationType] = None) -> List[Dict[str, Any]]:
        """Get available marketplace integrations"""
        integrations = self.integrations
        
        if type_filter:
            integrations = [i for i in integrations if i.type == type_filter]
            
        return [self._serialize_integration(integration) for integration in integrations]
        
    async def get_integration_details(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific integration"""
        integration = next((i for i in self.integrations if i.integration_id == integration_id), None)
        
        if not integration:
            return None
            
        details = self._serialize_integration(integration)
        
        # Add additional details
        details.update({
            "installation_guide": await self._get_installation_guide(integration_id),
            "compatibility": await self._get_compatibility_info(integration_id),
            "reviews": await self._get_integration_reviews(integration_id),
            "support_contact": await self._get_support_contact(integration_id)
        })
        
        return details
        
    async def install_integration(
        self, 
        tenant_id: str, 
        integration_id: str, 
        configuration: Dict[str, Any],
        instance_name: str
    ) -> Dict[str, Any]:
        """Install an integration for a tenant"""
        
        # Find the integration
        integration = next((i for i in self.integrations if i.integration_id == integration_id), None)
        if not integration:
            raise ValueError(f"Integration not found: {integration_id}")
            
        # Validate configuration
        validation_result = await self._validate_configuration(integration, configuration)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid configuration: {validation_result['errors']}")
            
        # Create integration instance
        instance = IntegrationInstance(
            instance_id=str(uuid.uuid4()),
            integration_id=integration_id,
            tenant_id=tenant_id,
            name=instance_name,
            configuration=configuration,
            status=IntegrationStatus.INSTALLED,
            last_sync=None,
            metrics={"events_sent": 0, "errors": 0, "last_error": None},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store instance
        if tenant_id not in self.instances:
            self.instances[tenant_id] = []
        self.instances[tenant_id].append(instance)
        
        # Initialize the integration
        try:
            await self._initialize_integration(instance)
            instance.status = IntegrationStatus.CONFIGURED
        except Exception as e:
            instance.status = IntegrationStatus.ERROR
            instance.metrics["last_error"] = str(e)
            logger.error(f"Failed to initialize integration {integration_id}: {e}")
            
        return asdict(instance)
        
    async def configure_integration(
        self, 
        tenant_id: str, 
        instance_id: str, 
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure an installed integration"""
        
        instance = await self._get_integration_instance(tenant_id, instance_id)
        if not instance:
            raise ValueError(f"Integration instance not found: {instance_id}")
            
        integration = next((i for i in self.integrations if i.integration_id == instance.integration_id), None)
        
        # Validate new configuration
        validation_result = await self._validate_configuration(integration, configuration)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid configuration: {validation_result['errors']}")
            
        # Update configuration
        instance.configuration.update(configuration)
        instance.updated_at = datetime.utcnow()
        instance.status = IntegrationStatus.CONFIGURED
        
        # Test the configuration
        try:
            await self._test_integration(instance)
            instance.status = IntegrationStatus.ACTIVE
        except Exception as e:
            instance.status = IntegrationStatus.ERROR
            instance.metrics["last_error"] = str(e)
            
        return asdict(instance)
        
    async def get_tenant_integrations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all integrations for a tenant"""
        tenant_instances = self.instances.get(tenant_id, [])
        
        result = []
        for instance in tenant_instances:
            integration = next((i for i in self.integrations if i.integration_id == instance.integration_id), None)
            
            instance_data = asdict(instance)
            if integration:
                instance_data["integration_name"] = integration.name
                instance_data["integration_type"] = integration.type.value
                instance_data["vendor"] = integration.vendor
                
            result.append(instance_data)
            
        return result
        
    async def send_data_to_integration(
        self, 
        tenant_id: str, 
        instance_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Send data to an external integration"""
        
        instance = await self._get_integration_instance(tenant_id, instance_id)
        if not instance or instance.status != IntegrationStatus.ACTIVE:
            return False
            
        integration = next((i for i in self.integrations if i.integration_id == instance.integration_id), None)
        if not integration:
            return False
            
        try:
            # Route to appropriate integration handler
            if integration.type == IntegrationType.SIEM:
                success = await self._send_to_siem(instance, data)
            elif integration.type == IntegrationType.TICKETING:
                success = await self._send_to_ticketing(instance, data)
            elif integration.type == IntegrationType.NOTIFICATION:
                success = await self._send_to_notification(instance, data)
            elif integration.type == IntegrationType.THREAT_INTEL:
                success = await self._send_to_threat_intel(instance, data)
            else:
                success = await self._send_generic(instance, data)
                
            # Update metrics
            if success:
                instance.metrics["events_sent"] += 1
            else:
                instance.metrics["errors"] += 1
                
            instance.last_sync = datetime.utcnow()
            return success
            
        except Exception as e:
            instance.metrics["errors"] += 1
            instance.metrics["last_error"] = str(e)
            logger.error(f"Failed to send data to integration {instance_id}: {e}")
            return False
            
    def _serialize_integration(self, integration: Integration) -> Dict[str, Any]:
        """Serialize integration for API response"""
        data = asdict(integration)
        data["type"] = integration.type.value
        data["status"] = integration.status.value
        return data
        
    async def _validate_configuration(self, integration: Integration, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integration configuration"""
        errors = []
        
        for field, schema in integration.configuration_schema.items():
            if schema.get("required", False) and field not in configuration:
                errors.append(f"Required field missing: {field}")
                continue
                
            if field in configuration:
                value = configuration[field]
                field_type = schema.get("type")
                
                if field_type == "string" and not isinstance(value, str):
                    errors.append(f"Field {field} must be a string")
                elif field_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field {field} must be an integer")
                elif field_type == "array" and not isinstance(value, list):
                    errors.append(f"Field {field} must be an array")
                    
                # Check min/max for integers
                if field_type == "integer" and isinstance(value, int):
                    if "min" in schema and value < schema["min"]:
                        errors.append(f"Field {field} must be >= {schema['min']}")
                    if "max" in schema and value > schema["max"]:
                        errors.append(f"Field {field} must be <= {schema['max']}")
                        
        return {"valid": len(errors) == 0, "errors": errors}
        
    async def _get_integration_instance(self, tenant_id: str, instance_id: str) -> Optional[IntegrationInstance]:
        """Get integration instance by ID"""
        tenant_instances = self.instances.get(tenant_id, [])
        return next((i for i in tenant_instances if i.instance_id == instance_id), None)
        
    async def _initialize_integration(self, instance: IntegrationInstance):
        """Initialize a newly installed integration"""
        # Test connectivity and setup
        await self._test_integration(instance)
        
    async def _test_integration(self, instance: IntegrationInstance):
        """Test integration connectivity"""
        integration = next((i for i in self.integrations if i.integration_id == instance.integration_id), None)
        
        if integration.type == IntegrationType.SIEM:
            await self._test_siem_connection(instance)
        elif integration.type == IntegrationType.TICKETING:
            await self._test_ticketing_connection(instance)
        elif integration.type == IntegrationType.NOTIFICATION:
            await self._test_notification_connection(instance)
            
    async def _test_siem_connection(self, instance: IntegrationInstance):
        """Test SIEM integration connection"""
        # Simulate connection test
        await asyncio.sleep(0.1)
        
    async def _test_ticketing_connection(self, instance: IntegrationInstance):
        """Test ticketing system connection"""
        # Simulate connection test
        await asyncio.sleep(0.1)
        
    async def _test_notification_connection(self, instance: IntegrationInstance):
        """Test notification system connection"""
        # Simulate connection test
        await asyncio.sleep(0.1)
        
    async def _send_to_siem(self, instance: IntegrationInstance, data: Dict[str, Any]) -> bool:
        """Send data to SIEM system"""
        # Implementation would depend on specific SIEM API
        logger.info(f"Sending data to SIEM {instance.integration_id}")
        return True
        
    async def _send_to_ticketing(self, instance: IntegrationInstance, data: Dict[str, Any]) -> bool:
        """Send data to ticketing system"""
        # Implementation would depend on specific ticketing API
        logger.info(f"Creating ticket in {instance.integration_id}")
        return True
        
    async def _send_to_notification(self, instance: IntegrationInstance, data: Dict[str, Any]) -> bool:
        """Send notification"""
        # Implementation would depend on specific notification API
        logger.info(f"Sending notification via {instance.integration_id}")
        return True
        
    async def _send_to_threat_intel(self, instance: IntegrationInstance, data: Dict[str, Any]) -> bool:
        """Send data to threat intelligence platform"""
        logger.info(f"Sharing threat intel via {instance.integration_id}")
        return True
        
    async def _send_generic(self, instance: IntegrationInstance, data: Dict[str, Any]) -> bool:
        """Send data to generic integration"""
        logger.info(f"Sending data to {instance.integration_id}")
        return True
        
    async def _get_installation_guide(self, integration_id: str) -> Dict[str, Any]:
        """Get installation guide for integration"""
        return {
            "steps": [
                "Configure API credentials in your external system",
                "Install the integration in XORB marketplace",
                "Configure connection parameters",
                "Test the integration",
                "Activate data flow"
            ],
            "prerequisites": ["Admin access to external system", "API access enabled"],
            "estimated_time": "15-30 minutes"
        }
        
    async def _get_compatibility_info(self, integration_id: str) -> Dict[str, Any]:
        """Get compatibility information"""
        return {
            "supported_versions": ["Latest", "LTS"],
            "minimum_requirements": ["API access", "Network connectivity"],
            "tested_environments": ["Cloud", "On-premise", "Hybrid"]
        }
        
    async def _get_integration_reviews(self, integration_id: str) -> List[Dict[str, Any]]:
        """Get integration reviews and ratings"""
        return [
            {
                "rating": 4.5,
                "review": "Great integration, works seamlessly with our SIEM",
                "author": "Security Team Lead",
                "date": "2025-01-15"
            }
        ]
        
    async def _get_support_contact(self, integration_id: str) -> Dict[str, Any]:
        """Get support contact information"""
        return {
            "email": "support@xorb.io",
            "documentation": "https://docs.xorb.io/integrations",
            "community": "https://community.xorb.io"
        }


# Global marketplace instance
api_marketplace = APIMarketplace()


async def get_marketplace_integrations(type_filter: str = None) -> List[Dict[str, Any]]:
    """API endpoint to get marketplace integrations"""
    filter_type = IntegrationType(type_filter) if type_filter else None
    return await api_marketplace.get_available_integrations(filter_type)


async def install_marketplace_integration(
    tenant_id: str, 
    integration_id: str, 
    configuration: Dict[str, Any],
    instance_name: str
) -> Dict[str, Any]:
    """API endpoint to install integration"""
    return await api_marketplace.install_integration(tenant_id, integration_id, configuration, instance_name)