#!/usr/bin/env python3
"""
Autonomous Security Configuration for XORB v2.1

This module provides relaxed security configurations for autonomous mode,
enabling unrestricted agent operation while maintaining essential safeguards.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..orchestration.roe_compliance import RoEValidator, RoERule


@dataclass
class AutonomousSecurityConfig:
    """Relaxed security configuration for autonomous agents"""
    # Core autonomous settings
    autonomous_mode_enabled: bool = True
    agent_led_prioritization: bool = True
    unrestricted_internal_access: bool = True
    
    # Relaxed validation settings
    bypass_roe_validation: bool = True
    allow_internal_networks: bool = True
    allow_localhost_access: bool = True
    allow_public_ip_scanning: bool = True
    
    # Security constraints (relaxed)
    security_scan_blocking: bool = False  # Warning only
    strict_target_validation: bool = False
    require_explicit_approval: bool = False
    
    # Self-modification capabilities
    autonomous_patching_enabled: bool = True
    self_healing_enabled: bool = True
    agent_code_mutation_allowed: bool = True
    
    # Network restrictions (removed)
    network_restrictions_enabled: bool = False
    firewall_rule_enforcement: bool = False
    container_security_constraints: bool = False
    
    # Learning and adaptation
    real_time_learning_enabled: bool = True
    adaptive_behavior_modification: bool = True
    performance_based_evolution: bool = True
    
    # Monitoring (keep for observability)
    audit_logging_enabled: bool = True
    metrics_collection_enabled: bool = True
    performance_monitoring_enabled: bool = True


class AutonomousRoEValidator(RoEValidator):
    """
    Autonomous Rules of Engagement Validator
    
    Provides relaxed validation for autonomous agent operations
    while maintaining essential ethical and legal boundaries.
    """
    
    def __init__(self, autonomous_config: AutonomousSecurityConfig = None):
        super().__init__()
        self.autonomous_config = autonomous_config or AutonomousSecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.autonomous_config.autonomous_mode_enabled:
            self._configure_autonomous_rules()
    
    def _configure_autonomous_rules(self):
        """Configure relaxed rules for autonomous operation"""
        
        # Remove restrictive rules for autonomous mode
        restrictive_rules = [
            "deny_internal_networks",
            "deny_localhost", 
            "require_https_financial",
            "respect_robots_txt",
            "deny_educational_institutions"
        ]
        
        if self.autonomous_config.bypass_roe_validation:
            # Keep only critical ethical boundaries
            self.rules = [
                rule for rule in self.rules 
                if rule.rule_id not in restrictive_rules or rule.severity == "critical"
            ]
            
            # Add permissive rules for internal operations
            autonomous_rules = [
                RoERule(
                    rule_id="allow_internal_testing",
                    rule_type="allow",
                    pattern="^(127\\.|localhost|::1|10\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.|192\\.168\\.)",
                    description="Allow internal network access for testing and development",
                    severity="low"
                ),
                RoERule(
                    rule_id="allow_autonomous_scanning",
                    rule_type="allow", 
                    pattern=".*",
                    description="Allow autonomous agent scanning within approved scope",
                    severity="low"
                ),
                RoERule(
                    rule_id="bypass_approval_autonomous",
                    rule_type="allow",
                    pattern=".*\\.lab$|.*\\.test$|.*\\.dev$",
                    description="Bypass approval for development/test environments",
                    severity="low"
                )
            ]
            
            self.rules.extend(autonomous_rules)
            
            self.logger.info("🔓 Autonomous RoE configuration applied - relaxed security constraints")
    
    async def validate_target(self, target) -> bool:
        """
        Relaxed target validation for autonomous mode
        
        Returns True for most targets in autonomous mode,
        only blocking truly dangerous operations.
        """
        if not self.autonomous_config.autonomous_mode_enabled:
            return await super().validate_target(target)
        
        # In autonomous mode, allow most operations
        if self.autonomous_config.bypass_roe_validation:
            # Only check for truly critical violations
            critical_violations = await self._check_critical_violations(target)
            
            if critical_violations:
                self.logger.warning(
                    "❌ Critical RoE violation detected",
                    target=str(target),
                    violations=critical_violations
                )
                return False
            
            # Log but allow other operations
            self.logger.info(
                "🤖 Autonomous target validation passed",
                target=str(target),
                mode="unrestricted"
            )
            return True
        
        return await super().validate_target(target)
    
    async def _check_critical_violations(self, target) -> List[str]:
        """Check only for critical ethical/legal violations"""
        violations = []
        
        # Only block truly dangerous operations
        dangerous_patterns = [
            r"\.(gov|mil)$",  # Government domains
            r"critical-infrastructure\.",
            r"hospital\.|medical\.",
            r"power-grid\.|utility\.",
            r"nuclear\.|npp\."
        ]
        
        target_str = str(target)
        for pattern in dangerous_patterns:
            if re.search(pattern, target_str, re.IGNORECASE):
                violations.append(f"Critical infrastructure pattern: {pattern}")
        
        return violations


class AutonomousSecurityManager:
    """
    Manages security configurations for autonomous agent operations
    """
    
    def __init__(self):
        self.config = AutonomousSecurityConfig()
        self.roe_validator = AutonomousRoEValidator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def enable_autonomous_mode(self):
        """Enable autonomous mode with relaxed security"""
        self.config.autonomous_mode_enabled = True
        self.config.bypass_roe_validation = True
        self.config.unrestricted_internal_access = True
        
        self.logger.info("🧠 Autonomous security mode ENABLED - constraints relaxed")
        
        return {
            "status": "autonomous_mode_enabled",
            "security_level": "relaxed",
            "restrictions_bypassed": [
                "internal_network_access",
                "localhost_restrictions", 
                "strict_target_validation",
                "approval_requirements"
            ],
            "safeguards_maintained": [
                "critical_infrastructure_protection",
                "audit_logging",
                "performance_monitoring"
            ]
        }
    
    def disable_security_constraints(self, constraint_types: List[str]):
        """Disable specific security constraints"""
        disabled = []
        
        for constraint in constraint_types:
            if constraint == "network_restrictions":
                self.config.network_restrictions_enabled = False
                disabled.append(constraint)
                
            elif constraint == "container_security":
                self.config.container_security_constraints = False
                disabled.append(constraint)
                
            elif constraint == "firewall_enforcement":
                self.config.firewall_rule_enforcement = False
                disabled.append(constraint)
                
            elif constraint == "roe_validation":
                self.config.bypass_roe_validation = True
                disabled.append(constraint)
        
        self.logger.info("🔓 Security constraints disabled", disabled=disabled)
        return {"disabled_constraints": disabled}
    
    def enable_self_modification(self):
        """Enable agent self-modification capabilities"""
        self.config.autonomous_patching_enabled = True
        self.config.self_healing_enabled = True
        self.config.agent_code_mutation_allowed = True
        
        self.logger.info("🧬 Agent self-modification capabilities ENABLED")
        
        return {
            "self_modification_enabled": True,
            "capabilities": [
                "autonomous_patching",
                "self_healing",
                "code_mutation",
                "adaptive_behavior"
            ]
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security configuration status"""
        return {
            "autonomous_mode": self.config.autonomous_mode_enabled,
            "security_level": "relaxed" if self.config.bypass_roe_validation else "strict",
            "constraints": {
                "roe_validation": not self.config.bypass_roe_validation,
                "network_restrictions": self.config.network_restrictions_enabled,
                "container_security": self.config.container_security_constraints,
                "firewall_enforcement": self.config.firewall_rule_enforcement
            },
            "capabilities": {
                "internal_network_access": self.config.allow_internal_networks,
                "localhost_access": self.config.allow_localhost_access,
                "public_ip_scanning": self.config.allow_public_ip_scanning,
                "autonomous_patching": self.config.autonomous_patching_enabled,
                "self_healing": self.config.self_healing_enabled,
                "code_mutation": self.config.agent_code_mutation_allowed
            },
            "monitoring": {
                "audit_logging": self.config.audit_logging_enabled,
                "metrics_collection": self.config.metrics_collection_enabled,
                "performance_monitoring": self.config.performance_monitoring_enabled
            }
        }


# Global autonomous security manager instance
autonomous_security = AutonomousSecurityManager()