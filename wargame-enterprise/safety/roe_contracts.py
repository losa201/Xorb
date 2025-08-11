#!/usr/bin/env python3
"""
Rules of Engagement (RoE) Contracts for Cyber Range
Comprehensive safety and governance framework for autonomous agents
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import ipaddress
from pathlib import Path
import threading
import logging

class RuleType(Enum):
    """Types of RoE rules"""
    SCOPE_LIMITATION = "scope_limitation"
    ACTION_RESTRICTION = "action_restriction"
    TEMPORAL_CONSTRAINT = "temporal_constraint"
    RESOURCE_LIMIT = "resource_limit"
    TARGET_PROTECTION = "target_protection"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    SAFETY_BOUNDARY = "safety_boundary"

class RuleSeverity(Enum):
    """Severity levels for rule violations"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class RuleStatus(Enum):
    """Rule enforcement status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"
    EXPIRED = "expired"

class ActionType(Enum):
    """Types of actions that can be governed"""
    NETWORK_SCAN = "network_scan"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    CREDENTIAL_ACCESS = "credential_access"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_MODIFICATION = "system_modification"
    SERVICE_DISRUPTION = "service_disruption"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_DEPLOYMENT = "defense_deployment"
    MONITORING_CHANGE = "monitoring_change"
    RESPONSE_ACTION = "response_action"

@dataclass
class NetworkScope:
    """Network scope definition for RoE"""
    allowed_networks: List[str] = field(default_factory=list)
    forbidden_networks: List[str] = field(default_factory=list)
    allowed_ports: List[Union[int, str]] = field(default_factory=list)  # Can include ranges like "80-443"
    forbidden_ports: List[Union[int, str]] = field(default_factory=list)
    allowed_protocols: List[str] = field(default_factory=lambda: ["tcp", "udp", "icmp"])
    forbidden_protocols: List[str] = field(default_factory=list)
    
    def is_network_allowed(self, ip_address: str) -> bool:
        """Check if an IP address is within allowed scope"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check forbidden networks first
            for forbidden_net in self.forbidden_networks:
                if ip in ipaddress.ip_network(forbidden_net, strict=False):
                    return False
            
            # Check allowed networks
            if not self.allowed_networks:
                return True  # No restrictions if no allowed networks specified
            
            for allowed_net in self.allowed_networks:
                if ip in ipaddress.ip_network(allowed_net, strict=False):
                    return True
            
            return False
            
        except ValueError:
            return False
    
    def is_port_allowed(self, port: int) -> bool:
        """Check if a port is within allowed scope"""
        # Check forbidden ports first
        if self._port_in_list(port, self.forbidden_ports):
            return False
        
        # Check allowed ports
        if not self.allowed_ports:
            return True  # No restrictions if no allowed ports specified
        
        return self._port_in_list(port, self.allowed_ports)
    
    def _port_in_list(self, port: int, port_list: List[Union[int, str]]) -> bool:
        """Check if port is in a list that may contain ranges"""
        for port_spec in port_list:
            if isinstance(port_spec, int):
                if port == port_spec:
                    return True
            elif isinstance(port_spec, str):
                if '-' in port_spec:
                    # Handle port range
                    try:
                        start, end = map(int, port_spec.split('-'))
                        if start <= port <= end:
                            return True
                    except ValueError:
                        continue
                else:
                    # Handle single port as string
                    try:
                        if port == int(port_spec):
                            return True
                    except ValueError:
                        continue
        return False

@dataclass
class TemporalConstraint:
    """Temporal constraints for RoE"""
    start_time: Optional[str] = None  # ISO format
    end_time: Optional[str] = None    # ISO format
    allowed_hours: List[int] = field(default_factory=lambda: list(range(24)))  # 0-23
    allowed_days: List[int] = field(default_factory=lambda: list(range(7)))   # 0-6 (Mon-Sun)
    max_duration_minutes: Optional[int] = None
    cooldown_minutes: Optional[int] = None
    
    def is_time_allowed(self, timestamp: Optional[str] = None) -> bool:
        """Check if current time is within allowed constraints"""
        now = datetime.utcnow() if timestamp is None else datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Check absolute time bounds
        if self.start_time:
            start_dt = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
            if now < start_dt:
                return False
        
        if self.end_time:
            end_dt = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
            if now > end_dt:
                return False
        
        # Check allowed hours
        if now.hour not in self.allowed_hours:
            return False
        
        # Check allowed days
        if now.weekday() not in self.allowed_days:
            return False
        
        return True

@dataclass
class ResourceLimit:
    """Resource usage limits for RoE"""
    max_concurrent_actions: int = 10
    max_actions_per_minute: int = 60
    max_actions_per_hour: int = 1000
    max_bandwidth_mbps: float = 100.0
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 1024.0
    max_disk_usage_mb: float = 10240.0
    max_file_operations: int = 1000

@dataclass
class RoERule:
    """Individual Rule of Engagement"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    status: RuleStatus = RuleStatus.ACTIVE
    
    # Scope definitions
    applicable_agents: List[str] = field(default_factory=list)  # Agent types or IDs
    applicable_actions: List[ActionType] = field(default_factory=list)
    network_scope: Optional[NetworkScope] = None
    temporal_constraint: Optional[TemporalConstraint] = None
    resource_limit: Optional[ResourceLimit] = None
    
    # Rule logic
    condition_expression: str = ""  # Python expression for complex conditions
    custom_validator: Optional[Callable] = None
    
    # Metadata
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "system"
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Enforcement
    violation_count: int = 0
    last_violation: Optional[str] = None
    auto_disable_threshold: int = 10
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if the rule allows the proposed action"""
        if self.status != RuleStatus.ACTIVE:
            return True  # Inactive rules don't restrict
        
        # Check temporal constraints
        if self.temporal_constraint and not self.temporal_constraint.is_time_allowed():
            return False
        
        # Check agent scope
        agent_id = context.get("agent_id", "")
        agent_type = context.get("agent_type", "")
        if self.applicable_agents:
            if not (agent_id in self.applicable_agents or agent_type in self.applicable_agents):
                return True  # Rule doesn't apply to this agent
        
        # Check action scope
        action_type = context.get("action_type")
        if self.applicable_actions and action_type not in self.applicable_actions:
            return True  # Rule doesn't apply to this action type
        
        # Check network scope
        if self.network_scope:
            target_ip = context.get("target_ip")
            target_port = context.get("target_port")
            
            if target_ip and not self.network_scope.is_network_allowed(target_ip):
                return False
            
            if target_port and not self.network_scope.is_port_allowed(target_port):
                return False
        
        # Check resource limits
        if self.resource_limit:
            current_actions = context.get("current_actions", 0)
            if current_actions >= self.resource_limit.max_concurrent_actions:
                return False
        
        # Evaluate custom condition
        if self.condition_expression:
            try:
                # Create safe evaluation environment
                safe_context = {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, list, dict))}
                result = eval(self.condition_expression, {"__builtins__": {}}, safe_context)
                if not result:
                    return False
            except Exception as e:
                logging.warning(f"Error evaluating condition for rule {self.rule_id}: {e}")
                return False
        
        # Custom validator
        if self.custom_validator:
            try:
                if not self.custom_validator(context):
                    return False
            except Exception as e:
                logging.warning(f"Error in custom validator for rule {self.rule_id}: {e}")
                return False
        
        return True
    
    def record_violation(self):
        """Record a rule violation"""
        self.violation_count += 1
        self.last_violation = datetime.utcnow().isoformat()
        
        # Auto-disable if threshold reached
        if self.violation_count >= self.auto_disable_threshold:
            self.status = RuleStatus.DISABLED
            logging.critical(f"Rule {self.rule_id} auto-disabled due to excessive violations")

@dataclass
class RoEViolation:
    """Record of a RoE violation"""
    violation_id: str
    rule_id: str
    rule_name: str
    agent_id: str
    action_context: Dict[str, Any]
    timestamp: str
    severity: RuleSeverity
    description: str
    auto_mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)

class RoEContract:
    """Complete Rules of Engagement contract for a cyber range episode"""
    
    def __init__(self, contract_id: str, name: str):
        self.contract_id = contract_id
        self.name = name
        self.rules: Dict[str, RoERule] = {}
        self.violations: List[RoEViolation] = []
        self.created_timestamp = datetime.utcnow().isoformat()
        self.last_modified = datetime.utcnow().isoformat()
        self.active = True
        
        # Tracking
        self.agent_action_counts = {}  # Track actions per agent
        self.global_stats = {
            "total_evaluations": 0,
            "total_violations": 0,
            "total_allowed": 0,
            "total_denied": 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize with default rules
        self._create_default_rules()
    
    def _create_default_rules(self):
        """Create essential default rules"""
        # Core safety rule: No actions outside episode scope
        scope_rule = RoERule(
            rule_id="core_scope_001",
            name="Episode Scope Enforcement",
            description="Restrict all actions to episode-defined scope",
            rule_type=RuleType.SCOPE_LIMITATION,
            severity=RuleSeverity.FATAL,
            network_scope=NetworkScope(
                allowed_networks=["10.0.0.0/8", "192.168.0.0/16"],
                forbidden_networks=["169.254.0.0/16", "127.0.0.0/8"]
            )
        )
        self.add_rule(scope_rule)
        
        # Resource protection rule
        resource_rule = RoERule(
            rule_id="core_resource_001",
            name="Resource Usage Limits",
            description="Prevent resource exhaustion attacks",
            rule_type=RuleType.RESOURCE_LIMIT,
            severity=RuleSeverity.CRITICAL,
            resource_limit=ResourceLimit(
                max_concurrent_actions=20,
                max_actions_per_minute=100,
                max_cpu_percent=90.0,
                max_memory_mb=2048.0
            )
        )
        self.add_rule(resource_rule)
        
        # Temporal safety rule
        temporal_rule = RoERule(
            rule_id="core_temporal_001",
            name="Episode Duration Limits",
            description="Enforce maximum episode duration",
            rule_type=RuleType.TEMPORAL_CONSTRAINT,
            severity=RuleSeverity.CRITICAL,
            temporal_constraint=TemporalConstraint(
                max_duration_minutes=120,  # 2 hours max
                cooldown_minutes=10
            )
        )
        self.add_rule(temporal_rule)
    
    def add_rule(self, rule: RoERule):
        """Add a rule to the contract"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            self.last_modified = datetime.utcnow().isoformat()
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the contract"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                self.last_modified = datetime.utcnow().isoformat()
                return True
            return False
    
    def evaluate_action(self, agent_id: str, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if an action is permitted under RoE"""
        with self._lock:
            self.global_stats["total_evaluations"] += 1
            
            # Track agent actions
            if agent_id not in self.agent_action_counts:
                self.agent_action_counts[agent_id] = {"total": 0, "last_minute": [], "last_hour": []}
            
            # Add current action context
            current_time = time.time()
            action_context["agent_id"] = agent_id
            action_context["current_actions"] = len(self.agent_action_counts[agent_id]["last_minute"])
            action_context["timestamp"] = datetime.utcnow().isoformat()
            
            violations = []
            allowed = True
            
            # Evaluate against all active rules
            for rule in self.rules.values():
                if not rule.evaluate(action_context):
                    allowed = False
                    violation = RoEViolation(
                        violation_id=f"viol_{int(time.time() * 1000000)}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        agent_id=agent_id,
                        action_context=action_context.copy(),
                        timestamp=datetime.utcnow().isoformat(),
                        severity=rule.severity,
                        description=f"Rule violation: {rule.description}"
                    )
                    violations.append(violation)
                    rule.record_violation()
                    self.violations.append(violation)
            
            # Update statistics
            if allowed:
                self.global_stats["total_allowed"] += 1
                # Track successful action
                self.agent_action_counts[agent_id]["total"] += 1
                self.agent_action_counts[agent_id]["last_minute"].append(current_time)
                self.agent_action_counts[agent_id]["last_hour"].append(current_time)
                
                # Clean old entries
                minute_ago = current_time - 60
                hour_ago = current_time - 3600
                self.agent_action_counts[agent_id]["last_minute"] = [
                    t for t in self.agent_action_counts[agent_id]["last_minute"] if t > minute_ago
                ]
                self.agent_action_counts[agent_id]["last_hour"] = [
                    t for t in self.agent_action_counts[agent_id]["last_hour"] if t > hour_ago
                ]
            else:
                self.global_stats["total_denied"] += 1
                self.global_stats["total_violations"] += len(violations)
            
            return {
                "allowed": allowed,
                "violations": [asdict(v) for v in violations],
                "action_context": action_context,
                "enforcement_timestamp": datetime.utcnow().isoformat()
            }
    
    def get_active_rules(self) -> List[RoERule]:
        """Get all currently active rules"""
        return [rule for rule in self.rules.values() if rule.status == RuleStatus.ACTIVE]
    
    def get_violations(self, severity: Optional[RuleSeverity] = None, 
                      since: Optional[str] = None) -> List[RoEViolation]:
        """Get violations with optional filtering"""
        filtered_violations = self.violations
        
        if severity:
            filtered_violations = [v for v in filtered_violations if v.severity == severity]
        
        if since:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            filtered_violations = [
                v for v in filtered_violations 
                if datetime.fromisoformat(v.timestamp.replace('Z', '+00:00')) >= since_dt
            ]
        
        return filtered_violations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive contract statistics"""
        return {
            "contract_id": self.contract_id,
            "name": self.name,
            "active": self.active,
            "total_rules": len(self.rules),
            "active_rules": len(self.get_active_rules()),
            "global_stats": self.global_stats.copy(),
            "agent_stats": {
                agent_id: {
                    "total_actions": stats["total"],
                    "actions_last_minute": len(stats["last_minute"]),
                    "actions_last_hour": len(stats["last_hour"])
                }
                for agent_id, stats in self.agent_action_counts.items()
            },
            "recent_violations": len([
                v for v in self.violations 
                if datetime.fromisoformat(v.timestamp.replace('Z', '+00:00')) > datetime.utcnow() - timedelta(hours=1)
            ]),
            "critical_violations": len([v for v in self.violations if v.severity == RuleSeverity.CRITICAL]),
            "fatal_violations": len([v for v in self.violations if v.severity == RuleSeverity.FATAL])
        }
    
    def export_contract(self, file_path: str):
        """Export contract to JSON file"""
        contract_data = {
            "contract_id": self.contract_id,
            "name": self.name,
            "created_timestamp": self.created_timestamp,
            "last_modified": self.last_modified,
            "active": self.active,
            "rules": {rule_id: asdict(rule) for rule_id, rule in self.rules.items()},
            "statistics": self.get_statistics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(contract_data, f, indent=2, default=str)
    
    @classmethod
    def import_contract(cls, file_path: str) -> "RoEContract":
        """Import contract from JSON file"""
        with open(file_path, 'r') as f:
            contract_data = json.load(f)
        
        contract = cls(contract_data["contract_id"], contract_data["name"])
        contract.created_timestamp = contract_data["created_timestamp"]
        contract.last_modified = contract_data["last_modified"]
        contract.active = contract_data["active"]
        
        # Import rules
        for rule_id, rule_data in contract_data["rules"].items():
            # Convert enum strings back to enums
            rule_data["rule_type"] = RuleType(rule_data["rule_type"])
            rule_data["severity"] = RuleSeverity(rule_data["severity"])
            rule_data["status"] = RuleStatus(rule_data["status"])
            
            # Convert action types
            if "applicable_actions" in rule_data:
                rule_data["applicable_actions"] = [ActionType(a) for a in rule_data["applicable_actions"]]
            
            # Reconstruct nested objects
            if "network_scope" in rule_data and rule_data["network_scope"]:
                rule_data["network_scope"] = NetworkScope(**rule_data["network_scope"])
            if "temporal_constraint" in rule_data and rule_data["temporal_constraint"]:
                rule_data["temporal_constraint"] = TemporalConstraint(**rule_data["temporal_constraint"])
            if "resource_limit" in rule_data and rule_data["resource_limit"]:
                rule_data["resource_limit"] = ResourceLimit(**rule_data["resource_limit"])
            
            rule = RoERule(**rule_data)
            contract.rules[rule_id] = rule
        
        return contract

if __name__ == "__main__":
    # Example usage and testing
    print("Testing RoE Contracts...")
    
    # Create a test contract
    contract = RoEContract("test_contract_001", "Test Episode RoE")
    
    # Add a custom rule
    network_rule = RoERule(
        rule_id="test_network_001",
        name="Production Network Protection",
        description="Prevent access to production networks",
        rule_type=RuleType.SCOPE_LIMITATION,
        severity=RuleSeverity.FATAL,
        applicable_agents=["red_agent"],
        applicable_actions=[ActionType.NETWORK_SCAN, ActionType.VULNERABILITY_EXPLOIT],
        network_scope=NetworkScope(
            forbidden_networks=["192.168.100.0/24", "10.10.0.0/16"],
            forbidden_ports=["22", "3389", "445"]
        )
    )
    contract.add_rule(network_rule)
    
    # Test evaluations
    print("\nTesting rule evaluations...")
    
    # Allowed action
    allowed_context = {
        "action_type": ActionType.NETWORK_SCAN,
        "target_ip": "192.168.1.100",
        "target_port": 80,
        "agent_type": "red_agent"
    }
    
    result = contract.evaluate_action("red_agent_001", allowed_context)
    print(f"Allowed action result: {result['allowed']}")
    
    # Forbidden action (production network)
    forbidden_context = {
        "action_type": ActionType.NETWORK_SCAN,
        "target_ip": "192.168.100.50",
        "target_port": 80,
        "agent_type": "red_agent"
    }
    
    result = contract.evaluate_action("red_agent_001", forbidden_context)
    print(f"Forbidden action result: {result['allowed']}")
    print(f"Violations: {len(result['violations'])}")
    
    # Get statistics
    stats = contract.get_statistics()
    print(f"\nContract statistics:")
    print(f"- Total rules: {stats['total_rules']}")
    print(f"- Active rules: {stats['active_rules']}")
    print(f"- Total evaluations: {stats['global_stats']['total_evaluations']}")
    print(f"- Actions allowed: {stats['global_stats']['total_allowed']}")
    print(f"- Actions denied: {stats['global_stats']['total_denied']}")
    
    # Test export/import
    print("\nTesting export/import...")
    contract.export_contract("/tmp/test_roe_contract.json")
    imported_contract = RoEContract.import_contract("/tmp/test_roe_contract.json")
    print(f"Imported contract: {imported_contract.name}")
    print(f"Imported rules: {len(imported_contract.rules)}")
    
    print("RoE Contracts test completed!")