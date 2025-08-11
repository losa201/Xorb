import logging
import sys
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from abc import ABC, abstractmethod

# Add parent directory to path for service imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api', 'app'))
try:
    from services.base_service import SecurityService, ServiceHealth, ServiceStatus
except ImportError:
    # Fallback for when base service is not available
    SecurityService = None
    ServiceHealth = None
    ServiceStatus = None

logger = logging.getLogger("NetworkMicrosegmentation")

class SecurityPolicy:
    """Represents a network security policy"""
    def __init__(self, 
                 policy_id: str,
                 name: str,
                 description: str,
                 rules: List[Dict[str, Any]],
                 priority: int = 100,
                 enabled: bool = True):
        self.policy_id = policy_id
        self.name = name
        self.description = description
        self.rules = rules
        self.priority = priority
        self.enabled = enabled
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def update_rules(self, new_rules: List[Dict[str, Any]]):
        """Update the policy rules"""
        self.rules = new_rules
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "rules": self.rules,
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class NetworkSegment:
    """Represents a network segment with security controls"""
    def __init__(self, 
                 segment_id: str,
                 name: str,
                 description: str,
                 assets: List[Dict[str, Any]],
                 policies: List[SecurityPolicy],
                 metadata: Dict[str, Any] = None):
        self.segment_id = segment_id
        self.name = name
        self.description = description
        self.assets = assets
        self.policies = policies
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        
    def add_asset(self, asset: Dict[str, Any]) -> None:
        """Add an asset to the network segment"""
        self.assets.append(asset)
        
    def remove_asset(self, asset_id: str) -> bool:
        """Remove an asset from the network segment"""
        initial_count = len(self.assets)
        self.assets = [a for a in self.assets if a.get("asset_id") != asset_id]
        return len(self.assets) < initial_count
        
    def apply_policy(self, policy: SecurityPolicy) -> None:
        """Apply a security policy to the network segment"""
        # Check if policy already exists
        existing_policy = next((p for p in self.policies 
                              if p.policy_id == policy.policy_id), None)
        if existing_policy:
            # Update existing policy
            existing_policy.update_rules(policy.rules)
        else:
            # Add new policy
            self.policies.append(policy)
        
    def get_applicable_policies(self, context: Dict[str, Any]) -> List[SecurityPolicy]:
        """Get applicable policies based on context"""
        applicable_policies = []
        for policy in self.policies:
            if policy.enabled and self._context_matches_policy(context, policy):
                applicable_policies.append(policy)
        return applicable_policies
        
    def _context_matches_policy(self, context: Dict[str, Any], 
                               policy: SecurityPolicy) -> bool:
        """Check if context matches policy criteria"""
        # This is a simplified implementation - in a real system this would be more complex
        if not policy.rules:
            return True
            
        for rule in policy.rules:
            rule_type = rule.get("type")
            if rule_type == "time_based":
                if not self._check_time_rule(rule, context):
                    return False
            elif rule_type == "user_role":
                if not self._check_user_role_rule(rule, context):
                    return False
            elif rule_type == "device_type":
                if not self._check_device_type_rule(rule, context):
                    return False
            elif rule_type == "traffic_pattern":
                if not self._check_traffic_pattern_rule(rule, context):
                    return False
        return True
        
    def _check_time_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check time-based rule against context"""
        current_time = datetime.now().time()
        start_time = datetime.strptime(rule.get("start_time", "00:00"), "%H:%M").time()
        end_time = datetime.strptime(rule.get("end_time", "23:59"), "%H:%M").time()
        
        return start_time <= current_time <= end_time
        
    def _check_user_role_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check user role rule against context"""
        user_role = context.get("user", {}).get("role")
        allowed_roles = rule.get("allowed_roles", [])
        return not allowed_roles or user_role in allowed_roles
        
    def _check_device_type_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check device type rule against context"""
        device_type = context.get("device", {}).get("type")
        allowed_types = rule.get("allowed_types", [])
        return not allowed_types or device_type in allowed_types
        
    def _check_traffic_pattern_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check traffic pattern rule against context"""
        traffic_type = context.get("traffic", {}).get("type")
        allowed_types = rule.get("allowed_types", [])
        return not allowed_types or traffic_type in allowed_types

class PolicyEngine(SecurityService if SecurityService else object):
    """Engine for managing and enforcing security policies"""
    def __init__(self, **kwargs):
        if SecurityService:
            super().__init__(**kwargs)
        self.policies: Dict[str, SecurityPolicy] = {}
        self.segments: Dict[str, NetworkSegment] = {}
        self.policy_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize logger
        if hasattr(super(), 'logger'):
            self.logger = super().logger
        else:
            self.logger = logging.getLogger("NetworkMicrosegmentation")
        
    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a new security policy"""
        if policy.policy_id in self.policies:
            raise ValueError(f"Policy {policy.policy_id} already exists")
        self.policies[policy.policy_id] = policy
        self.policy_history[policy.policy_id] = [{
            "timestamp": datetime.now().isoformat(),
            "action": "created",
            "policy": policy.to_dict()
        }]
        
    def update_policy(self, policy_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing security policy"""
        if policy_id not in self.policies:
            return False
            
        # For simplicity, we'll just update the rules
        # In a real implementation, we'd handle other fields too
        if "rules" in update_data:
            self.policies[policy_id].update_rules(update_data["rules"])
            self.policy_history[policy_id].append({
                "timestamp": datetime.now().isoformat(),
                "action": "updated",
                "policy": self.policies[policy_id].to_dict()
            })
            return True
        return False
        
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get a security policy by ID"""
        return self.policies.get(policy_id)
        
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a security policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            del self.policy_history[policy_id]
            return True
        return False
        
    def add_segment(self, segment: NetworkSegment) -> None:
        """Add a new network segment"""
        if segment.segment_id in self.segments:
            raise ValueError(f"Segment {segment.segment_id} already exists")
        self.segments[segment.segment_id] = segment
        
    def get_segment(self, segment_id: str) -> Optional[NetworkSegment]:
        """Get a network segment by ID"""
        return self.segments.get(segment_id)
        
    def apply_policies_to_segment(self, segment_id: str, policy_ids: List[str]) -> bool:
        """Apply multiple policies to a network segment"""
        if segment_id not in self.segments:
            return False
            
        segment = self.segments[segment_id]
        for policy_id in policy_ids:
            if policy_id in self.policies:
                segment.apply_policy(self.policies[policy_id])
        return True
        
    def evaluate_context(self, segment_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate context against policies in a segment"""
        if segment_id not in self.segments:
            return {
                "allowed": False,
                "reason": "Segment not found",
                "policies_evaluated": []
            }
            
        segment = self.segments[segment_id]
        applicable_policies = segment.get_applicable_policies(context)
        
        # If no applicable policies, default to allow
        if not applicable_policies:
            return {
                "allowed": True,
                "reason": "No applicable policies",
                "policies_evaluated": []
            }
            
        # Check each policy - if any policy denies, the action is denied
        denied = False
        evaluated_policies = []
        
        for policy in applicable_policies:
            evaluated_policies.append({
                "policy_id": policy.policy_id,
                "name": policy.name,
                "result": "allowed"
            })
            
            # In a real implementation, we would check the policy rules
            # against the context in more detail
            if policy.priority > 50:  # High priority policies can override
                denied = True
                evaluated_policies[-1]["result"] = "denied"
                
        return {
            "allowed": not denied,
            "reason": "Policy evaluation" + (" denied" if denied else " allowed"),
            "policies_evaluated": evaluated_policies
        }
        
    def get_policy_history(self, policy_id: str) -> List[Dict[str, Any]]:
        """Get policy history"""
        return self.policy_history.get(policy_id, [])

# Zero-Trust Network Micro-Segmentation Implementation
class ZeroTrustNetwork:
    """Main class implementing zero-trust network micro-segmentation"""
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.segment_history: Dict[str, List[Dict[str, Any]]] = {}
        self.default_policy = SecurityPolicy(
            policy_id="default_policy",
            name="Default Policy",
            description="Default security policy applied to all segments",
            rules=[],
            priority=100,
            enabled=True
        )
        
    def create_segment(self, segment: NetworkSegment) -> None:
        """Create a new network segment"""
        # Ensure all segments have the default policy applied
        segment.apply_policy(self.default_policy)
        self.policy_engine.add_segment(segment)
        
        # Record in history
        if segment.segment_id not in self.segment_history:
            self.segment_history[segment.segment_id] = []
            
        self.segment_history[segment.segment_id].append({
            "timestamp": datetime.now().isoformat(),
            "action": "created",
            "segment": segment.__dict__  # Simplified for demo
        })
        
    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a new security policy"""
        self.policy_engine.add_policy(policy)
        
    def update_policy(self, policy_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing security policy"""
        return self.policy_engine.update_policy(policy_id, update_data)
        
    def apply_policy_to_segment(self, segment_id: str, policy_id: str) -> bool:
        """Apply a policy to a network segment"""
        return self.policy_engine.apply_policies_to_segment(segment_id, [policy_id])
        
    def evaluate_access(self, segment_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate access request to a network segment"""
        result = self.policy_engine.evaluate_context(segment_id, context)
        
        # Record evaluation in segment history
        if segment_id in self.segment_history:
            self.segment_history[segment_id].append({
                "timestamp": datetime.now().isoformat(),
                "action": "access_evaluation",
                "context": context,
                "result": result
            })
            
        return result
        
    def get_segment_policies(self, segment_id: str) -> List[SecurityPolicy]:
        """Get all policies applied to a segment"""
        segment = self.policy_engine.get_segment(segment_id)
        return segment.policies if segment else []
        
    def get_segment_history(self, segment_id: str) -> List[Dict[str, Any]]:
        """Get history for a network segment"""
        return self.segment_history.get(segment_id, [])
        
    def create_policy_from_template(self, template_type: str, parameters: Dict[str, Any]) -> SecurityPolicy:
        """Create a policy from a template"""
        # In a real implementation, this would use policy templates
        # For now, create a basic policy
        policy_id = f"policy_{hashlib.md5(f'{template_type}_{datetime.now().isoformat()}'.encode()).hexdigest()}"
        
        # Different template types could create different rule sets
        rules = []
        if template_type == "pci_dss":
            rules = [{
                "type": "data_classification",
                "required": ["pci_data"]
            }]
        elif template_type == "hipaa":
            rules = [{
                "type": "data_classification",
                "required": ["phi_data"]
            }]
        
        return SecurityPolicy(
            policy_id=policy_id,
            name=f"{template_type.upper()} Policy",
            description=f"{template_type.upper()} compliance policy",
            rules=rules,
            priority=50,
            enabled=True
        )
    
    async def initialize(self) -> bool:
        """Initialize the policy engine"""
        try:
            self.logger.info("PolicyEngine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PolicyEngine: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the policy engine"""
        try:
            # Clear sensitive data
            self.policies.clear()
            self.segments.clear()
            self.policy_history.clear()
            self.logger.info("PolicyEngine shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown PolicyEngine: {e}")
            return False
    
    async def health_check(self) -> 'ServiceHealth':
        """Perform health check"""
        try:
            checks = {
                "policies_count": len(self.policies),
                "segments_count": len(self.segments),
                "policy_history_count": len(self.policy_history),
                "active_policies": len([p for p in self.policies.values() if p.enabled])
            }
            
            status = ServiceStatus.HEALTHY if ServiceStatus else "healthy"
            health = ServiceHealth(
                status=status,
                message="PolicyEngine is operational",
                timestamp=datetime.utcnow(),
                checks=checks
            ) if ServiceHealth else {
                "status": status,
                "message": "PolicyEngine is operational",
                "timestamp": datetime.utcnow(),
                "checks": checks
            }
            
            return health
        except Exception as e:
            status = ServiceStatus.UNHEALTHY if ServiceStatus else "unhealthy"
            return ServiceHealth(
                status=status,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            ) if ServiceHealth else {
                "status": status,
                "message": f"Health check failed: {e}",
                "timestamp": datetime.utcnow(),
                "checks": {"error": str(e)}
            }
    
    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event for policy enforcement"""
        try:
            source_ip = event.get("source_ip")
            dest_ip = event.get("dest_ip")
            
            if not source_ip or not dest_ip:
                return {"error": "Missing source_ip or dest_ip in event"}
            
            # Create enforcement context
            context = {
                "source": {"ip": source_ip},
                "destination": {"ip": dest_ip},
                "traffic": event.get("traffic", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check policy enforcement
            enforcement_results = []
            for policy in self.policies.values():
                if policy.enabled:
                    result = self.enforce_policy(policy.policy_id, context)
                    enforcement_results.append({
                        "policy_id": policy.policy_id,
                        "policy_name": policy.name,
                        "allowed": result.get("allowed", False),
                        "rule_matches": result.get("rule_matches", [])
                    })
            
            return {
                "event_processed": True,
                "enforcement_results": enforcement_results,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to process security event: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics"""
        try:
            total_policies = len(self.policies)
            active_policies = len([p for p in self.policies.values() if p.enabled])
            total_segments = len(self.segments)
            
            # Calculate policy effectiveness (simplified)
            high_priority_policies = len([p for p in self.policies.values() if p.priority < 50])
            
            return {
                "total_policies": total_policies,
                "active_policies": active_policies,
                "total_segments": total_segments,
                "high_priority_policies": high_priority_policies,
                "policy_effectiveness_rate": active_policies / total_policies if total_policies > 0 else 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get security metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

# Example usage
if __name__ == "__main__":
    # Initialize zero-trust network
    ztn = ZeroTrustNetwork()
    
    # Create a network segment for financial data
    financial_segment = NetworkSegment(
        segment_id="fin_segment_001",
        name="Financial Data Segment",
        description="Contains sensitive financial information",
        assets=[{
            "asset_id": "db_fin_001",
            "type": "database",
            "sensitivity": "high"
        }],
        policies=[],
        metadata={
            "compliance": "pci_dss"
        }
    )
    
    # Create and apply PCI DSS policy
    pci_policy = ztn.create_policy_from_template("pci_dss", {})
    ztn.add_policy(pci_policy)
    ztn.apply_policy_to_segment(financial_segment.segment_id, pci_policy.policy_id)
    
    # Create and add another policy
    custom_policy = SecurityPolicy(
        policy_id="custom_001",
        name="Custom Security Policy",
        description="Custom rules for financial segment",
        rules=[{
            "type": "time_based",
            "start_time": "08:00",
            "end_time": "18:00"
        }, {
            "type": "user_role",
            "allowed_roles": ["finance", "admin"]
        }],
        priority=75,
        enabled=True
    )
    
    ztn.add_policy(custom_policy)
    ztn.apply_policy_to_segment(financial_segment.segment_id, custom_policy.policy_id)
    
    # Evaluate access during business hours
    access_context = {
        "user": {
            "id": "user_123",
            "role": "finance"
        },
        "device": {
            "id": "device_456",
            "type": "laptop"
        },
        "time": datetime.now().isoformat()
    }
    
    result = ztn.evaluate_access(financial_segment.segment_id, access_context)
    print("\nAccess Evaluation Result:")
    print(json.dumps(result, indent=2))
    
    # Try access with unauthorized role
    access_context["user"]["role"] = "guest"
    result = ztn.evaluate_access(financial_segment.segment_id, access_context)
    print("\nAccess Evaluation Result (Unauthorized):")
    print(json.dumps(result, indent=2))
    
    # Get segment history
    history = ztn.get_segment_history(financial_segment.segment_id)
    print("\nSegment History:")
    print(json.dumps(history, indent=2))
    
    print("\nZero-trust network micro-segmentation system initialized successfully!")
    