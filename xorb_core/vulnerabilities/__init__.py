"""
XORB Vulnerability Management Module

This module provides comprehensive vulnerability lifecycle management including
discovery, classification, prioritization, remediation tracking, and automated
workflow orchestration.
"""

from .vulnerability_lifecycle_manager import (
    Vulnerability,
    VulnerabilityState,
    VulnerabilitySeverity,
    VulnerabilityCategory,
    RemediationPriority,
    VulnerabilityEvidence,
    RemediationStep,
    VulnerabilityLifecycleManager,
    IVulnerabilitySource,
    IRemediationProvider,
    AutomatedRemediationProvider,
    WorkflowRule,
    vulnerability_manager,
    initialize_vulnerability_management,
    shutdown_vulnerability_management,
    get_vulnerability_manager
)

__all__ = [
    "Vulnerability",
    "VulnerabilityState",
    "VulnerabilitySeverity", 
    "VulnerabilityCategory",
    "RemediationPriority",
    "VulnerabilityEvidence",
    "RemediationStep",
    "VulnerabilityLifecycleManager",
    "IVulnerabilitySource",
    "IRemediationProvider", 
    "AutomatedRemediationProvider",
    "WorkflowRule",
    "vulnerability_manager",
    "initialize_vulnerability_management",
    "shutdown_vulnerability_management",
    "get_vulnerability_manager"
]

__version__ = "2.0.0"