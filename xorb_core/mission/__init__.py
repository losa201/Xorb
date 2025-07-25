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

from .autonomous_bounty_engagement import (
    AutonomousBountyEngagement,
    BountyPlatform,
    BountyProgram,
    BountyMission,
    VulnerabilitySubmission
)

from .compliance_platform_integration import (
    CompliancePlatformIntegration,
    ComplianceFramework,
    ComplianceAssessment,
    ComplianceEvidence,
    ComplianceControl
)

from .adaptive_mission_engine import (
    AdaptiveMissionEngine,
    MissionType,
    MissionPlan,
    MissionObjective,
    AdaptationAction
)

from .external_intelligence_api import (
    ExternalIntelligenceAPI,
    APICredentials,
    APIEndpoint,
    IntelligenceProduct
)

from .autonomous_remediation_agents import (
    AutonomousRemediationAgent,
    AutonomousRemediationSystem,
    RemediationType,
    RemediationPlan,
    RemediationAction
)

from .audit_trail_system import (
    AuditTrailSystem,
    AuditEvent,
    AuditEventType,
    OverrideRequest,
    ComplianceReport
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