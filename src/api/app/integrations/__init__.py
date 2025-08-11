"""
Enterprise Integration Package - Clean Architecture
Focused connectors for enterprise security platforms
"""

from .base_connector import (
    BaseConnector,
    IntegrationEndpoint,
    IntegrationCredentials,
    IntegrationRequest,
    IntegrationResponse,
    IntegrationType,
    AuthenticationType
)

from .connector_factory import (
    ConnectorFactory,
    ConnectorManager,
    get_connector_manager
)

from .siem_connector import (
    SplunkConnector,
    QRadarConnector,
    SentinelConnector
)

from .soar_connector import (
    PhantomConnector,
    DemistoConnector,
    SwimlaneConnector
)

from .firewall_connector import (
    PaloAltoConnector,
    FortiGateConnector,
    CheckPointConnector
)

from .identity_connector import (
    ActiveDirectoryConnector,
    OktaConnector,
    AzureADConnector
)

__all__ = [
    # Base infrastructure
    'BaseConnector',
    'IntegrationEndpoint',
    'IntegrationCredentials', 
    'IntegrationRequest',
    'IntegrationResponse',
    'IntegrationType',
    'AuthenticationType',
    
    # Factory and management
    'ConnectorFactory',
    'ConnectorManager',
    'get_connector_manager',
    
    # SIEM connectors
    'SplunkConnector',
    'QRadarConnector', 
    'SentinelConnector',
    
    # SOAR connectors
    'PhantomConnector',
    'DemistoConnector',
    'SwimlaneConnector',
    
    # Firewall connectors
    'PaloAltoConnector',
    'FortiGateConnector',
    'CheckPointConnector',
    
    # Identity connectors
    'ActiveDirectoryConnector',
    'OktaConnector',
    'AzureADConnector'
]