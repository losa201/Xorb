"""
XORB Orchestration Module

This module provides orchestration capabilities for the XORB ecosystem including
dynamic resource management, scaling, coordination, and distributed campaign management.
"""

from .dynamic_resource_manager import (
    ResourceQuota,
    ScalingPolicy,
    DynamicResourceManager,
    LocalResourceProvider,
    KubernetesResourceProvider,
    CloudResourceProvider,
    create_development_policy,
    create_production_policy,
    create_staging_policy
)

from .distributed_campaign_coordinator import (
    NodeInfo,
    NodeRole,
    CampaignState,
    CoordinationMessage,
    CoordinationMessageType,
    DistributedTask,
    DistributedCampaign,
    ConsensusAlgorithm,
    DistributedCampaignCoordinator,
    distributed_coordinator,
    initialize_distributed_coordination,
    shutdown_distributed_coordination,
    get_distributed_coordinator
)

__all__ = [
    "ResourceQuota",
    "ScalingPolicy", 
    "DynamicResourceManager",
    "LocalResourceProvider",
    "KubernetesResourceProvider",
    "CloudResourceProvider",
    "create_development_policy",
    "create_production_policy",
    "create_staging_policy",
    "NodeInfo",
    "NodeRole",
    "CampaignState",
    "CoordinationMessage",
    "CoordinationMessageType",
    "DistributedTask",
    "DistributedCampaign",
    "ConsensusAlgorithm",
    "DistributedCampaignCoordinator",
    "distributed_coordinator",
    "initialize_distributed_coordination",
    "shutdown_distributed_coordination",
    "get_distributed_coordinator"
]

__version__ = "2.0.0"