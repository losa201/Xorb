"""
Capability Registry for Red/Blue Agent Framework

Manages environment-specific technique allow/deny lists with dynamic capability loading.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types for capability restrictions"""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    CYBER_RANGE = "cyber_range"


class TechniqueCategory(Enum):
    """MITRE ATT&CK inspired technique categories"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    
    # Blue team categories
    DETECTION = "detection"
    ANALYSIS = "analysis"
    MITIGATION = "mitigation"
    RECOVERY = "recovery"
    THREAT_HUNTING = "threat_hunting"


@dataclass
class TechniqueParameter:
    """Parameter definition for a technique"""
    name: str
    type: str  # string, int, float, bool, list
    required: bool = True
    default: Any = None
    constraints: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class TechniqueDefinition:
    """Definition of an attack/defense technique"""
    id: str
    name: str
    category: TechniqueCategory
    description: str
    mitre_id: Optional[str] = None
    parameters: List[TechniqueParameter] = None
    dependencies: List[str] = None  # Other technique IDs this depends on
    platforms: List[str] = None  # windows, linux, macos, docker, kubernetes
    risk_level: str = "medium"  # low, medium, high, critical
    stealth_level: str = "medium"  # low, medium, high
    detection_difficulty: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.dependencies is None:
            self.dependencies = []
        if self.platforms is None:
            self.platforms = ["linux", "windows"]


@dataclass
class EnvironmentPolicy:
    """Policy defining what techniques are allowed in an environment"""
    environment: Environment
    allowed_categories: Set[TechniqueCategory]
    denied_techniques: Set[str]  # Specific technique IDs to deny
    allowed_techniques: Set[str]  # Specific technique IDs to allow (overrides category)
    max_risk_level: str = "high"  # maximum risk level allowed
    max_concurrent_agents: int = 10
    sandbox_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sandbox_constraints is None:
            self.sandbox_constraints = {}


class CapabilityRegistry:
    """
    Central registry for managing agent capabilities and environment policies.
    
    Features:
    - Environment-specific technique allow/deny
    - Dynamic capability loading from JSON manifests
    - Redis-backed caching for performance
    - Policy validation and enforcement
    - Technique dependency resolution
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, config_path: Optional[Path] = None):
        self.redis_client = redis_client
        self.config_path = config_path or Path(__file__).parent.parent / "configs"
        self.techniques: Dict[str, TechniqueDefinition] = {}
        self.policies: Dict[Environment, EnvironmentPolicy] = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
    async def initialize(self):
        """Initialize the capability registry"""
        logger.info("Initializing Capability Registry...")
        
        # Load technique definitions from JSON manifests
        await self._load_technique_manifests()
        
        # Load environment policies
        await self._load_environment_policies()
        
        # Validate technique dependencies
        await self._validate_dependencies()
        
        logger.info(f"Loaded {len(self.techniques)} techniques and {len(self.policies)} environment policies")
        
    async def _load_technique_manifests(self):
        """Load technique definitions from JSON manifest files"""
        manifests_dir = self.config_path / "techniques"
        if not manifests_dir.exists():
            logger.warning(f"Techniques directory not found: {manifests_dir}")
            return
            
        for manifest_file in manifests_dir.glob("*.json"):
            try:
                with open(manifest_file) as f:
                    manifest_data = json.load(f)
                    
                for technique_data in manifest_data.get("techniques", []):
                    technique = self._parse_technique_definition(technique_data)
                    self.techniques[technique.id] = technique
                    
                logger.info(f"Loaded techniques from {manifest_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load technique manifest {manifest_file}: {e}")
                
    async def _load_environment_policies(self):
        """Load environment-specific policies"""
        policies_file = self.config_path / "environment_policies.json"
        if not policies_file.exists():
            logger.warning(f"Environment policies file not found: {policies_file}")
            return
            
        try:
            with open(policies_file) as f:
                policies_data = json.load(f)
                
            for env_name, policy_data in policies_data.items():
                try:
                    env = Environment(env_name)
                    policy = self._parse_environment_policy(env, policy_data)
                    self.policies[env] = policy
                except ValueError:
                    logger.warning(f"Unknown environment: {env_name}")
                    
        except Exception as e:
            logger.error(f"Failed to load environment policies: {e}")
            
    def _parse_technique_definition(self, data: Dict[str, Any]) -> TechniqueDefinition:
        """Parse technique definition from JSON data"""
        parameters = []
        for param_data in data.get("parameters", []):
            param = TechniqueParameter(**param_data)
            parameters.append(param)
            
        return TechniqueDefinition(
            id=data["id"],
            name=data["name"],
            category=TechniqueCategory(data["category"]),
            description=data["description"],
            mitre_id=data.get("mitre_id"),
            parameters=parameters,
            dependencies=data.get("dependencies", []),
            platforms=data.get("platforms", ["linux", "windows"]),
            risk_level=data.get("risk_level", "medium"),
            stealth_level=data.get("stealth_level", "medium"),
            detection_difficulty=data.get("detection_difficulty", "medium")
        )
        
    def _parse_environment_policy(self, env: Environment, data: Dict[str, Any]) -> EnvironmentPolicy:
        """Parse environment policy from JSON data"""
        allowed_categories = {TechniqueCategory(cat) for cat in data.get("allowed_categories", [])}
        denied_techniques = set(data.get("denied_techniques", []))
        allowed_techniques = set(data.get("allowed_techniques", []))
        
        return EnvironmentPolicy(
            environment=env,
            allowed_categories=allowed_categories,
            denied_techniques=denied_techniques,
            allowed_techniques=allowed_techniques,
            max_risk_level=data.get("max_risk_level", "high"),
            max_concurrent_agents=data.get("max_concurrent_agents", 10),
            sandbox_constraints=data.get("sandbox_constraints", {})
        )
        
    async def _validate_dependencies(self):
        """Validate that all technique dependencies exist"""
        for technique_id, technique in self.techniques.items():
            for dep_id in technique.dependencies:
                if dep_id not in self.techniques:
                    logger.warning(f"Technique {technique_id} depends on unknown technique {dep_id}")
                    
    async def is_technique_allowed(self, technique_id: str, environment: Environment) -> bool:
        """Check if a technique is allowed in the given environment"""
        # Check cache first
        cache_key = f"technique_allowed:{technique_id}:{environment.value}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached is not None:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
                
        # Calculate permission
        allowed = await self._calculate_technique_permission(technique_id, environment)
        
        # Cache result
        if self.redis_client:
            try:
                await self.redis_client.setex(cache_key, self._cache_ttl, json.dumps(allowed))
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
                
        return allowed
        
    async def _calculate_technique_permission(self, technique_id: str, environment: Environment) -> bool:
        """Calculate if technique is allowed based on policies"""
        technique = self.techniques.get(technique_id)
        if not technique:
            logger.warning(f"Unknown technique: {technique_id}")
            return False
            
        policy = self.policies.get(environment)
        if not policy:
            logger.warning(f"No policy defined for environment: {environment}")
            return False
            
        # Check explicit allow list (overrides everything)
        if technique_id in policy.allowed_techniques:
            return True
            
        # Check explicit deny list
        if technique_id in policy.denied_techniques:
            return False
            
        # Check category permissions
        if technique.category not in policy.allowed_categories:
            return False
            
        # Check risk level
        risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if risk_levels.get(technique.risk_level, 2) > risk_levels.get(policy.max_risk_level, 3):
            return False
            
        return True
        
    async def get_allowed_techniques(self, environment: Environment, category: Optional[TechniqueCategory] = None) -> List[TechniqueDefinition]:
        """Get all techniques allowed in the environment, optionally filtered by category"""
        allowed_techniques = []
        
        for technique_id, technique in self.techniques.items():
            if category and technique.category != category:
                continue
                
            if await self.is_technique_allowed(technique_id, environment):
                allowed_techniques.append(technique)
                
        return allowed_techniques
        
    async def resolve_technique_dependencies(self, technique_id: str, environment: Environment) -> List[str]:
        """Resolve all dependencies for a technique in the given environment"""
        technique = self.techniques.get(technique_id)
        if not technique:
            return []
            
        resolved_deps = []
        to_resolve = [technique_id]
        resolved = set()
        
        while to_resolve:
            current_id = to_resolve.pop(0)
            if current_id in resolved:
                continue
                
            current_technique = self.techniques.get(current_id)
            if not current_technique:
                continue
                
            # Check if technique is allowed
            if not await self.is_technique_allowed(current_id, environment):
                logger.warning(f"Dependency {current_id} not allowed in {environment}")
                continue
                
            resolved.add(current_id)
            if current_id != technique_id:  # Don't include the original technique
                resolved_deps.append(current_id)
                
            # Add dependencies to resolve list
            for dep_id in current_technique.dependencies:
                if dep_id not in resolved:
                    to_resolve.append(dep_id)
                    
        return resolved_deps
        
    async def get_technique_parameters(self, technique_id: str) -> List[TechniqueParameter]:
        """Get parameter definitions for a technique"""
        technique = self.techniques.get(technique_id)
        return technique.parameters if technique else []
        
    async def validate_technique_parameters(self, technique_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize technique parameters"""
        technique = self.techniques.get(technique_id)
        if not technique:
            raise ValueError(f"Unknown technique: {technique_id}")
            
        validated_params = {}
        errors = []
        
        # Check required parameters
        for param_def in technique.parameters:
            if param_def.required and param_def.name not in parameters:
                if param_def.default is not None:
                    validated_params[param_def.name] = param_def.default
                else:
                    errors.append(f"Required parameter missing: {param_def.name}")
                continue
                
            if param_def.name not in parameters:
                continue
                
            value = parameters[param_def.name]
            
            # Type validation
            try:
                validated_value = self._validate_parameter_type(value, param_def.type)
                
                # Constraint validation
                if param_def.constraints:
                    validated_value = self._validate_parameter_constraints(validated_value, param_def.constraints)
                    
                validated_params[param_def.name] = validated_value
                
            except ValueError as e:
                errors.append(f"Parameter {param_def.name}: {e}")
                
        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
            
        return validated_params
        
    def _validate_parameter_type(self, value: Any, expected_type: str) -> Any:
        """Validate and convert parameter type"""
        type_mapping = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if not expected_python_type:
            raise ValueError(f"Unknown parameter type: {expected_type}")
            
        if not isinstance(value, expected_python_type):
            try:
                return expected_python_type(value)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert {type(value).__name__} to {expected_type}")
                
        return value
        
    def _validate_parameter_constraints(self, value: Any, constraints: Dict[str, Any]) -> Any:
        """Validate parameter constraints"""
        if "min" in constraints and value < constraints["min"]:
            raise ValueError(f"Value {value} below minimum {constraints['min']}")
            
        if "max" in constraints and value > constraints["max"]:
            raise ValueError(f"Value {value} above maximum {constraints['max']}")
            
        if "choices" in constraints and value not in constraints["choices"]:
            raise ValueError(f"Value {value} not in allowed choices: {constraints['choices']}")
            
        if "pattern" in constraints:
            import re
            if not re.match(constraints["pattern"], str(value)):
                raise ValueError(f"Value {value} does not match pattern {constraints['pattern']}")
                
        return value
        
    async def export_capabilities(self, environment: Environment) -> Dict[str, Any]:
        """Export all capabilities for an environment as JSON"""
        allowed_techniques = await self.get_allowed_techniques(environment)
        
        return {
            "environment": environment.value,
            "timestamp": datetime.utcnow().isoformat(),
            "total_techniques": len(self.techniques),
            "allowed_techniques": len(allowed_techniques),
            "policy": asdict(self.policies.get(environment, {})),
            "techniques": [asdict(tech) for tech in allowed_techniques]
        }
        
    async def update_policy(self, environment: Environment, policy_updates: Dict[str, Any]):
        """Dynamically update environment policy"""
        current_policy = self.policies.get(environment)
        if not current_policy:
            raise ValueError(f"No policy exists for environment: {environment}")
            
        # Update policy fields
        for field, value in policy_updates.items():
            if hasattr(current_policy, field):
                setattr(current_policy, field, value)
                
        # Clear related cache entries
        if self.redis_client:
            try:
                pattern = f"technique_allowed:*:{environment.value}"
                async for key in self.redis_client.scan_iter(match=pattern):
                    await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
                
        logger.info(f"Updated policy for environment {environment}")
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_techniques": len(self.techniques),
            "categories": {},
            "risk_levels": {},
            "environments": {}
        }
        
        # Count by category
        for technique in self.techniques.values():
            cat = technique.category.value
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            
            risk = technique.risk_level
            stats["risk_levels"][risk] = stats["risk_levels"].get(risk, 0) + 1
            
        # Count allowed techniques per environment
        for env, policy in self.policies.items():
            allowed = await self.get_allowed_techniques(env)
            stats["environments"][env.value] = {
                "total_allowed": len(allowed),
                "max_concurrent_agents": policy.max_concurrent_agents,
                "max_risk_level": policy.max_risk_level
            }
            
        return stats