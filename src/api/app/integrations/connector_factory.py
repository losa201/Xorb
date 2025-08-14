"""
Connector Factory - Clean Architecture Factory Pattern
Creates and manages focused integration connectors based on type and configuration
"""

import logging
from typing import Dict, Type, Optional, Any
from .base_connector import BaseConnector, IntegrationEndpoint, IntegrationType
from .siem_connector import SplunkConnector, QRadarConnector, SentinelConnector
from .soar_connector import PhantomConnector, DemistoConnector, SwimlaneConnector
from .firewall_connector import PaloAltoConnector, FortiGateConnector, CheckPointConnector
from .identity_connector import ActiveDirectoryConnector, OktaConnector, AzureADConnector

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """
    Factory class for creating integration connectors.
    Implements Factory Pattern for clean separation of concerns.
    """
    
    # Registry of available connector classes
    _connector_registry: Dict[str, Type[BaseConnector]] = {
        # SIEM Connectors
        "splunk": SplunkConnector,
        "qradar": QRadarConnector,
        "sentinel": SentinelConnector,
        "microsoft_sentinel": SentinelConnector,
        "azure_sentinel": SentinelConnector,
        
        # SOAR Connectors
        "phantom": PhantomConnector,
        "splunk_phantom": PhantomConnector,
        "demisto": DemistoConnector,
        "cortex_xsoar": DemistoConnector,
        "palo_alto_cortex": DemistoConnector,
        "swimlane": SwimlaneConnector,
        
        # Firewall Connectors
        "palo_alto": PaloAltoConnector,
        "paloalto": PaloAltoConnector,
        "pa_firewall": PaloAltoConnector,
        "fortigate": FortiGateConnector,
        "fortinet": FortiGateConnector,
        "checkpoint": CheckPointConnector,
        "check_point": CheckPointConnector,
        
        # Identity Provider Connectors
        "active_directory": ActiveDirectoryConnector,
        "ad": ActiveDirectoryConnector,
        "microsoft_ad": ActiveDirectoryConnector,
        "okta": OktaConnector,
        "azure_ad": AzureADConnector,
        "azure_active_directory": AzureADConnector,
        "microsoft_graph": AzureADConnector,
    }
    
    @classmethod
    def register_connector(cls, name: str, connector_class: Type[BaseConnector]) -> None:
        """Register a new connector class"""
        cls._connector_registry[name.lower()] = connector_class
        logger.info(f"Registered connector: {name} -> {connector_class.__name__}")
    
    @classmethod
    def get_available_connectors(cls) -> Dict[str, str]:
        """Get list of available connector types"""
        return {name: connector_class.__name__ for name, connector_class in cls._connector_registry.items()}
    
    @classmethod
    def create_connector(cls, endpoint: IntegrationEndpoint) -> Optional[BaseConnector]:
        """
        Create a connector instance based on endpoint configuration.
        
        Args:
            endpoint: Integration endpoint configuration
            
        Returns:
            Configured connector instance or None if type not supported
        """
        # Determine connector type from metadata or endpoint name
        connector_type = cls._determine_connector_type(endpoint)
        
        if not connector_type:
            logger.error(f"Unable to determine connector type for endpoint: {endpoint.name}")
            return None
        
        connector_class = cls._connector_registry.get(connector_type.lower())
        
        if not connector_class:
            logger.error(f"No connector class found for type: {connector_type}")
            return None
        
        try:
            # Create and configure connector
            connector = connector_class(endpoint)
            logger.info(f"Created connector: {connector_class.__name__} for {endpoint.name}")
            return connector
            
        except Exception as e:
            logger.error(f"Failed to create connector for {endpoint.name}: {e}")
            return None
    
    @classmethod
    def create_connector_by_type(cls, connector_type: str, endpoint: IntegrationEndpoint) -> Optional[BaseConnector]:
        """
        Create a connector instance by explicit type.
        
        Args:
            connector_type: Specific connector type to create
            endpoint: Integration endpoint configuration
            
        Returns:
            Configured connector instance or None if type not supported
        """
        connector_class = cls._connector_registry.get(connector_type.lower())
        
        if not connector_class:
            logger.error(f"No connector class found for type: {connector_type}")
            return None
        
        try:
            connector = connector_class(endpoint)
            logger.info(f"Created connector: {connector_class.__name__} for {endpoint.name}")
            return connector
            
        except Exception as e:
            logger.error(f"Failed to create connector {connector_type} for {endpoint.name}: {e}")
            return None
    
    @classmethod
    def _determine_connector_type(cls, endpoint: IntegrationEndpoint) -> Optional[str]:
        """
        Determine connector type from endpoint configuration.
        
        Args:
            endpoint: Integration endpoint configuration
            
        Returns:
            Connector type string or None if cannot be determined
        """
        # 1. Check explicit connector_type in metadata
        if "connector_type" in endpoint.metadata:
            return endpoint.metadata["connector_type"]
        
        # 2. Check platform in metadata
        if "platform" in endpoint.metadata:
            return endpoint.metadata["platform"]
        
        # 3. Infer from endpoint name or base URL
        name_lower = endpoint.name.lower()
        base_url_lower = endpoint.base_url.lower()
        
        # SIEM platforms
        if any(keyword in name_lower for keyword in ["splunk"]):
            return "splunk"
        elif any(keyword in name_lower for keyword in ["qradar", "ibm"]):
            return "qradar"
        elif any(keyword in name_lower for keyword in ["sentinel", "azure"]):
            return "sentinel"
        
        # SOAR platforms
        elif any(keyword in name_lower for keyword in ["phantom"]):
            return "phantom"
        elif any(keyword in name_lower for keyword in ["demisto", "cortex", "xsoar"]):
            return "demisto"
        elif any(keyword in name_lower for keyword in ["swimlane"]):
            return "swimlane"
        
        # Firewall platforms
        elif any(keyword in name_lower for keyword in ["palo", "alto", "pa-"]):
            return "palo_alto"
        elif any(keyword in name_lower for keyword in ["forti", "fortinet"]):
            return "fortigate"
        elif any(keyword in name_lower for keyword in ["checkpoint", "check_point"]):
            return "checkpoint"
        
        # Identity providers
        elif any(keyword in name_lower for keyword in ["active_directory", "microsoft_ad", "ad"]):
            return "active_directory"
        elif any(keyword in name_lower for keyword in ["okta"]):
            return "okta"
        elif any(keyword in name_lower for keyword in ["azure_ad", "graph"]):
            return "azure_ad"
        
        # Check base URL patterns
        elif "splunk" in base_url_lower:
            return "splunk"
        elif "qradar" in base_url_lower:
            return "qradar"
        elif "microsoftonline" in base_url_lower or "graph.microsoft" in base_url_lower:
            return "azure_ad"
        elif "okta" in base_url_lower:
            return "okta"
        
        # 4. Fallback to integration type mapping
        integration_type_mapping = {
            IntegrationType.SIEM: "splunk",  # Default SIEM
            IntegrationType.SOAR: "phantom",  # Default SOAR
            IntegrationType.FIREWALL: "palo_alto",  # Default Firewall
            IntegrationType.IDENTITY_PROVIDER: "active_directory",  # Default IdP
        }
        
        return integration_type_mapping.get(endpoint.integration_type)


class ConnectorManager:
    """
    Manages the lifecycle of integration connectors.
    Provides high-level interface for connector operations.
    """
    
    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self.factory = ConnectorFactory()
    
    async def register_endpoint(self, endpoint: IntegrationEndpoint) -> bool:
        """
        Register and initialize an integration endpoint.
        
        Args:
            endpoint: Integration endpoint configuration
            
        Returns:
            True if successfully registered, False otherwise
        """
        try:
            # Create connector
            connector = self.factory.create_connector(endpoint)
            
            if not connector:
                logger.error(f"Failed to create connector for endpoint: {endpoint.name}")
                return False
            
            # Initialize connector
            await connector.initialize()
            
            # Test health
            is_healthy = await connector.health_check()
            if not is_healthy:
                logger.warning(f"Health check failed for endpoint: {endpoint.name}")
                # Continue anyway - might be temporary
            
            # Store connector
            self.connectors[endpoint.integration_id] = connector
            
            logger.info(f"Successfully registered endpoint: {endpoint.name} ({endpoint.integration_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register endpoint {endpoint.name}: {e}")
            return False
    
    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister and cleanup an integration endpoint.
        
        Args:
            endpoint_id: Integration endpoint ID
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        try:
            if endpoint_id not in self.connectors:
                logger.warning(f"Endpoint not found: {endpoint_id}")
                return False
            
            connector = self.connectors[endpoint_id]
            await connector.close()
            
            del self.connectors[endpoint_id]
            
            logger.info(f"Successfully unregistered endpoint: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister endpoint {endpoint_id}: {e}")
            return False
    
    def get_connector(self, endpoint_id: str) -> Optional[BaseConnector]:
        """Get connector by endpoint ID"""
        return self.connectors.get(endpoint_id)
    
    def get_connectors_by_type(self, integration_type: IntegrationType) -> Dict[str, BaseConnector]:
        """Get all connectors of a specific type"""
        return {
            endpoint_id: connector
            for endpoint_id, connector in self.connectors.items()
            if connector.endpoint.integration_type == integration_type
        }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all connectors"""
        results = {}
        
        for endpoint_id, connector in self.connectors.items():
            try:
                is_healthy = await connector.health_check()
                results[endpoint_id] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {endpoint_id}: {e}")
                results[endpoint_id] = False
        
        return results
    
    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all connectors"""
        return {
            endpoint_id: connector.get_status()
            for endpoint_id, connector in self.connectors.items()
        }
    
    async def shutdown_all(self):
        """Shutdown all connectors"""
        logger.info("Shutting down all connectors...")
        
        for endpoint_id in list(self.connectors.keys()):
            await self.unregister_endpoint(endpoint_id)
        
        logger.info("All connectors shut down successfully")


# Global connector manager instance
_connector_manager: Optional[ConnectorManager] = None


def get_connector_manager() -> ConnectorManager:
    """Get global connector manager instance"""
    global _connector_manager
    if _connector_manager is None:
        _connector_manager = ConnectorManager()
    return _connector_manager