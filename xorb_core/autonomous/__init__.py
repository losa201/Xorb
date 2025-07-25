#!/usr/bin/env python3
"""
Autonomous Worker Framework for Xorb Security Intelligence Platform

This package provides secure autonomous capabilities for Xorb workers while
maintaining strict defensive security boundaries:

- AutonomousWorker: Core autonomous agent with security validation
- AutonomousOrchestrator: Intelligent orchestration and decision making  
- AutonomousMonitor: Comprehensive monitoring and observability
- Resource management and adaptive optimization
- Security compliance and audit logging

All autonomous operations operate within strict Rules of Engagement (RoE)
and maintain comprehensive audit trails for security compliance.
"""

from .autonomous_worker import (
    AutonomousWorker,
    AutonomyLevel,
    WorkerCapability,
    AutonomousConfig,
    WorkerIntelligence,
    SecurityConstraint,
    ResourceMonitor,
    DecisionEngine,
    TaskPrioritizer,
    WorkflowOptimizer
)

from .models import (
    AutonomousDecision,
    WorkloadProfile,
    WorkloadAnalyzer,
    PerformanceOptimizer
)

from .monitoring import (
    AutonomousMonitor,
    Alert,
    AlertSeverity,
    MetricType,
    PerformanceBaseline,
    ResourceMetrics,
    ResourcePredictor,
    SecurityComplianceMonitor,
    AutonomousDecisionAuditor,
    FailurePredictor,
    PerformanceAnalyzer
)

__version__ = "2.0.0"
__author__ = "Xorb Security Intelligence Team"

# Package metadata
__all__ = [
    # Core autonomous worker
    "AutonomousWorker",
    "AutonomyLevel", 
    "WorkerCapability",
    "AutonomousConfig",
    "WorkerIntelligence",
    "SecurityConstraint",
    
    # Resource management
    "ResourceMonitor",
    "DecisionEngine", 
    "TaskPrioritizer",
    "WorkflowOptimizer",
    
    # Orchestration
    from .autonomous_orchestrator import AutonomousOrchestrator
from .models import AutonomousDecision, WorkloadProfile, WorkloadAnalyzer, PerformanceOptimizer
from .rl_orchestrator_extensions import AutonomousMetrics, IntelligentScheduler
    
    # Monitoring and observability
    "AutonomousMonitor",
    "Alert",
    "AlertSeverity",
    "MetricType", 
    "PerformanceBaseline",
    "ResourceMetrics",
    "ResourcePredictor",
    "SecurityComplianceMonitor",
    "AutonomousDecisionAuditor",
    "FailurePredictor",
    "PerformanceAnalyzer"
]

# Security notice
SECURITY_NOTICE = """
SECURITY NOTICE: Autonomous Worker Framework

This framework operates within strict defensive security boundaries:

1. All operations must comply with Rules of Engagement (RoE)
2. Security validation is required for all autonomous decisions
3. Comprehensive audit logging tracks all autonomous actions
4. Resource limits prevent system exhaustion
5. Network boundaries restrict target accessibility

Any modifications to security constraints require security team approval.
Violations are logged and may trigger automatic system shutdown.
"""

def print_security_notice():
    """Print the security notice for autonomous operations"""
    print(SECURITY_NOTICE)

# Defensive security validation
def validate_autonomous_config(config: AutonomousConfig) -> bool:
    """
    Validate autonomous configuration for security compliance
    
    Args:
        config: AutonomousConfig to validate
        
    Returns:
        bool: True if configuration is secure and compliant
    """
    
    # Ensure security validation is enabled for high autonomy levels
    if (config.autonomy_level in [AutonomyLevel.HIGH, AutonomyLevel.MAXIMUM] and
        not config.security_validation_required):
        return False
    
    # Ensure RoE compliance is strict
    if not config.roe_compliance_strict:
        return False
    
    # Reasonable resource limits
    if config.max_concurrent_tasks > 32:  # Prevent resource exhaustion
        return False
    
    return True

# Factory function for secure autonomous worker creation
def create_secure_autonomous_worker(
    agent_id: str = None,
    config: dict = None,
    autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE
) -> AutonomousWorker:
    """
    Create a securely configured autonomous worker
    
    Args:
        agent_id: Optional agent identifier
        config: Agent configuration dictionary
        autonomy_level: Desired autonomy level
        
    Returns:
        AutonomousWorker: Configured autonomous worker with security validation
    """
    
    # Ensure secure defaults
    autonomous_config = AutonomousConfig(
        autonomy_level=autonomy_level,
        security_validation_required=True,
        roe_compliance_strict=True,
        max_concurrent_tasks=min(8, 32)  # Conservative default
    )
    
    # Validate configuration
    if not validate_autonomous_config(autonomous_config):
        raise ValueError("Autonomous configuration failed security validation")
    
    # Ensure config has required security fields
    secure_config = config or {}
    if 'authorized_targets' not in secure_config:
        secure_config['authorized_targets'] = []
    if 'authorized_networks' not in secure_config:
        secure_config['authorized_networks'] = []
    if 'prohibited_actions' not in secure_config:
        secure_config['prohibited_actions'] = ['destructive_scan', 'exploit', 'brute_force']
    
    return AutonomousWorker(
        agent_id=agent_id,
        config=secure_config,
        autonomous_config=autonomous_config
    )

# Factory function for secure autonomous orchestrator creation  
def create_secure_autonomous_orchestrator(
    redis_url: str = "redis://localhost:6379",
    nats_url: str = "nats://localhost:4222",
    autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE,
    max_concurrent_agents: int = 16
) -> AutonomousOrchestrator:
    """
    Create a securely configured autonomous orchestrator
    
    Args:
        redis_url: Redis connection URL
        nats_url: NATS connection URL  
        autonomy_level: Desired autonomy level
        max_concurrent_agents: Maximum concurrent agents (capped for security)
        
    Returns:
        AutonomousOrchestrator: Configured orchestrator with security validation
    """
    
    # Cap concurrent agents for security
    safe_concurrent_agents = min(max_concurrent_agents, 32)
    
    return AutonomousOrchestrator(
        redis_url=redis_url,
        nats_url=nats_url,
        max_concurrent_agents=safe_concurrent_agents,
        max_concurrent_campaigns=min(10, safe_concurrent_agents // 2),
        autonomy_level=autonomy_level
    )

# Print security notice on import
if __name__ != "__main__":
    import os
    if os.environ.get('XORB_SHOW_SECURITY_NOTICE', 'true').lower() == 'true':
        print_security_notice()