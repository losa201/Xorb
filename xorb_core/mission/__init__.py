#!/usr/bin/env python3
"""
XORB Mission Execution Module v9.0

This package provides autonomous mission execution capabilities:
- External platform engagement (bounty platforms, compliance frameworks)
- Adaptive mission orchestration and real-time adaptation
- Secure external intelligence APIs
- Autonomous remediation and self-healing
- Comprehensive audit trails and governance
"""

from .adaptive_mission_engine import (
    AdaptationAction,
    AdaptiveMissionEngine,
    MissionObjective,
    MissionPlan,
    MissionType,
)
from .audit_trail_system import (
    AuditEvent,
    AuditEventType,
    AuditTrailSystem,
    ComplianceReport,
    OverrideRequest,
)
from .autonomous_bounty_engagement import (
    AutonomousBountyEngagement,
    BountyMission,
    BountyPlatform,
    BountyProgram,
    VulnerabilitySubmission,
)
from .autonomous_remediation_agents import (
    AutonomousRemediationAgent,
    AutonomousRemediationSystem,
    RemediationAction,
    RemediationPlan,
    RemediationType,
)
from .compliance_platform_integration import (
    ComplianceAssessment,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceFramework,
    CompliancePlatformIntegration,
)
from .external_intelligence_api import (
    APICredentials,
    APIEndpoint,
    ExternalIntelligenceAPI,
    IntelligenceProduct,
)

__all__ = [
    # Bounty engagement
    'AutonomousBountyEngagement',
    'BountyPlatform',
    'BountyProgram',
    'BountyMission',
    'VulnerabilitySubmission',

    # Compliance integration
    'CompliancePlatformIntegration',
    'ComplianceFramework',
    'ComplianceAssessment',
    'ComplianceEvidence',
    'ComplianceControl',

    # Mission engine
    'AdaptiveMissionEngine',
    'MissionType',
    'MissionPlan',
    'MissionObjective',
    'AdaptationAction',

    # External API
    'ExternalIntelligenceAPI',
    'APICredentials',
    'APIEndpoint',
    'IntelligenceProduct',

    # Remediation agents
    'AutonomousRemediationAgent',
    'AutonomousRemediationSystem',
    'RemediationType',
    'RemediationPlan',
    'RemediationAction',

    # Audit trail
    'AuditTrailSystem',
    'AuditEvent',
    'AuditEventType',
    'OverrideRequest',
    'ComplianceReport'
]

__version__ = '9.0.0'
__author__ = 'XORB Development Team'
__description__ = 'Autonomous Mission Execution and External Engagement System'
