"""
XORB Orchestration Module

This module provides orchestration capabilities for the XORB ecosystem including
dynamic resource management, scaling, coordination, and distributed campaign management.
"""

from .distributed_campaign_coordinator import (
    CampaignState,
    ConsensusAlgorithm,
    CoordinationMessage,
    CoordinationMessageType,
    DistributedCampaign,
    DistributedCampaignCoordinator,
    DistributedTask,
    NodeInfo,
    NodeRole,
    distributed_coordinator,
    get_distributed_coordinator,
    initialize_distributed_coordination,
    shutdown_distributed_coordination,
)
from .dynamic_resource_manager import (
    DynamicResourceManager,
    KubernetesResourceProvider,
    LocalResourceProvider,
    ResourceQuota,
    ScalingPolicy,
    create_development_policy,
    create_epyc_optimized_policy,
)

__all__ = [
    "ResourceQuota",
    "ScalingPolicy",
    "DynamicResourceManager",
    "LocalResourceProvider",
    "KubernetesResourceProvider",
    "create_development_policy",
    "create_epyc_optimized_policy",
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
