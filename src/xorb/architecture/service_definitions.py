#!/usr/bin/env python3
"""
XORB Pristine Microservices Architecture - Service Definitions
Domain-Driven Service Boundaries with EPYC Optimization
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import datetime

class ServiceTier(Enum):
    CORE = "core"           # Critical business logic
    DOMAIN = "domain"       # Domain-specific services  
    PLATFORM = "platform"   # Infrastructure services
    EDGE = "edge"          # External-facing services

class ResourceProfile(Enum):
    CPU_INTENSIVE = "cpu_intensive"     # High CPU, low memory
    MEMORY_INTENSIVE = "memory_intensive" # High memory, moderate CPU
    IO_INTENSIVE = "io_intensive"       # High I/O, moderate CPU/memory
    BALANCED = "balanced"              # Balanced resource usage
    LIGHTWEIGHT = "lightweight"       # Minimal resources

class CommunicationPattern(Enum):
    SYNCHRONOUS = "sync"      # HTTP/gRPC request-response
    ASYNCHRONOUS = "async"    # Event-driven messaging
    HYBRID = "hybrid"         # Both sync and async
    STREAMING = "streaming"   # Real-time data streams

@dataclass
class EPYCOptimization:
    """EPYC-specific optimization configuration."""
    ccx_affinity: Optional[int] = None          # Core Complex preference
    numa_node: Optional[int] = None             # NUMA node binding
    l3_cache_sensitivity: bool = False          # Cache-sensitive workload
    thermal_policy: str = "balanced"            # conservative/balanced/performance
    core_allocation: Optional[List[int]] = None # Specific core assignment
    memory_policy: str = "interleave"          # NUMA memory policy

@dataclass  
class ServiceDefinition:
    """Enhanced service definition with architectural metadata."""
    name: str
    tier: ServiceTier
    domain: str
    resource_profile: ResourceProfile
    communication_patterns: Set[CommunicationPattern]
    dependencies: Set[str] = field(default_factory=set)
    data_stores: Set[str] = field(default_factory=set)
    exposed_apis: Set[str] = field(default_factory=set)
    consumed_events: Set[str] = field(default_factory=set)
    published_events: Set[str] = field(default_factory=set)
    epyc_optimization: EPYCOptimization = field(default_factory=EPYCOptimization)
    security_level: str = "standard"           # minimal/standard/high/critical
    scalability_pattern: str = "horizontal"   # horizontal/vertical/static
    observability_level: str = "standard"     # minimal/standard/enhanced/deep
    circuit_breaker_enabled: bool = True
    rate_limiting_enabled: bool = True
    caching_strategy: str = "none"            # none/local/distributed/hybrid

class XORBArchitecture:
    """XORB Pristine Microservices Architecture Definition."""
    
    def __init__(self):
        self.services = self._define_service_architecture()
        self.service_mesh_config = self._define_service_mesh()
        self.deployment_topology = self._define_deployment_topology()
    
    def _define_service_architecture(self) -> Dict[str, ServiceDefinition]:
        """Define the complete service architecture with optimal boundaries."""
        
        services = {}
        
        # ==============================
        # CORE TIER - Critical Business Logic
        # ==============================
        
        services["campaign-orchestrator"] = ServiceDefinition(
            name="campaign-orchestrator",
            tier=ServiceTier.CORE,
            domain="campaign_management",
            resource_profile=ResourceProfile.CPU_INTENSIVE,
            communication_patterns={CommunicationPattern.HYBRID},
            dependencies={"target-registry", "agent-lifecycle", "evidence-collector"},
            data_stores={"postgres-primary", "redis-cache"},
            exposed_apis={"campaign-management-api", "orchestration-api"},
            consumed_events={"target.discovered", "agent.completed", "evidence.collected"},
            published_events={"campaign.started", "campaign.completed", "campaign.failed"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=0,  # First CCX for critical operations
                numa_node=0,
                l3_cache_sensitivity=True,
                thermal_policy="performance",
                core_allocation=[0, 1, 2, 3]
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            observability_level="deep",
            caching_strategy="hybrid"
        )
        
        services["target-registry"] = ServiceDefinition(
            name="target-registry",
            tier=ServiceTier.CORE,
            domain="target_management",
            resource_profile=ResourceProfile.BALANCED,
            communication_patterns={CommunicationPattern.SYNCHRONOUS, CommunicationPattern.ASYNCHRONOUS},
            dependencies={"vulnerability-scanner", "threat-intelligence"},
            data_stores={"postgres-primary", "vector-store"},
            exposed_apis={"target-registry-api", "discovery-api"},
            consumed_events={"scan.completed", "threat.detected"},
            published_events={"target.discovered", "target.updated", "target.enriched"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=1,
                numa_node=0,
                l3_cache_sensitivity=False,
                thermal_policy="balanced"
            ),
            security_level="high",
            scalability_pattern="horizontal",
            caching_strategy="distributed"
        )
        
        services["agent-lifecycle"] = ServiceDefinition(
            name="agent-lifecycle",
            tier=ServiceTier.CORE,
            domain="agent_management",
            resource_profile=ResourceProfile.CPU_INTENSIVE,
            communication_patterns={CommunicationPattern.HYBRID},
            dependencies={"stealth-manager", "execution-engine", "ai-gateway"},
            data_stores={"postgres-primary", "time-series-db"},
            exposed_apis={"agent-lifecycle-api", "execution-control-api"},
            consumed_events={"execution.requested", "stealth.updated"},
            published_events={"agent.spawned", "agent.completed", "agent.failed"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=2,
                numa_node=0,
                l3_cache_sensitivity=True,
                thermal_policy="performance",
                core_allocation=[8, 9, 10, 11]
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            observability_level="deep"
        )
        
        services["evidence-collector"] = ServiceDefinition(
            name="evidence-collector",
            tier=ServiceTier.CORE,
            domain="evidence_management",
            resource_profile=ResourceProfile.IO_INTENSIVE,
            communication_patterns={CommunicationPattern.STREAMING, CommunicationPattern.ASYNCHRONOUS},
            dependencies={"forensics-analyzer", "ai-gateway"},
            data_stores={"postgres-primary", "object-store", "search-engine"},
            exposed_apis={"evidence-api", "forensics-api"},
            consumed_events={"scan.evidence", "exploit.evidence", "analysis.evidence"},
            published_events={"evidence.collected", "evidence.analyzed", "evidence.enriched"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=3,
                numa_node=1,  # Different NUMA node for I/O intensive
                l3_cache_sensitivity=False,
                thermal_policy="balanced"
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            caching_strategy="local"
        )
        
        # ==============================
        # DOMAIN TIER - Domain-Specific Services
        # ==============================
        
        services["vulnerability-scanner"] = ServiceDefinition(
            name="vulnerability-scanner",
            tier=ServiceTier.DOMAIN,
            domain="security_scanning",
            resource_profile=ResourceProfile.CPU_INTENSIVE,
            communication_patterns={CommunicationPattern.ASYNCHRONOUS},
            dependencies={"ai-gateway", "threat-intelligence"},
            data_stores={"postgres-primary", "redis-cache"},
            exposed_apis={"scanner-api", "vulnerability-api"},
            consumed_events={"target.discovered", "scan.requested"},
            published_events={"scan.started", "scan.completed", "vulnerability.found"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=4,
                numa_node=1,
                l3_cache_sensitivity=True,
                thermal_policy="performance",
                core_allocation=[16, 17, 18, 19]
            ),
            security_level="high",
            scalability_pattern="horizontal",
            observability_level="enhanced"
        )
        
        services["exploitation-engine"] = ServiceDefinition(
            name="exploitation-engine",
            tier=ServiceTier.DOMAIN,
            domain="security_exploitation",
            resource_profile=ResourceProfile.CPU_INTENSIVE,
            communication_patterns={CommunicationPattern.HYBRID},
            dependencies={"payload-generator", "stealth-manager", "ai-gateway"},
            data_stores={"postgres-primary", "secret-store"},
            exposed_apis={"exploitation-api", "payload-api"},
            consumed_events={"vulnerability.found", "exploit.requested"},
            published_events={"exploit.started", "exploit.completed", "exploit.failed"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=5,
                numa_node=1,
                l3_cache_sensitivity=True,
                thermal_policy="performance",
                core_allocation=[20, 21, 22, 23]
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            observability_level="deep"
        )
        
        services["stealth-manager"] = ServiceDefinition(
            name="stealth-manager",
            tier=ServiceTier.DOMAIN,
            domain="operational_security",
            resource_profile=ResourceProfile.BALANCED,
            communication_patterns={CommunicationPattern.SYNCHRONOUS},
            dependencies={"ai-gateway", "threat-intelligence"},
            data_stores={"postgres-primary", "redis-cache"},
            exposed_apis={"stealth-api", "opsec-api"},
            consumed_events={"detection.risk", "activity.anomaly"},
            published_events={"stealth.updated", "opsec.alert", "evasion.applied"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=6,
                numa_node=1,
                thermal_policy="balanced"
            ),
            security_level="critical",
            scalability_pattern="vertical",
            caching_strategy="distributed"
        )
        
        services["ai-gateway"] = ServiceDefinition(
            name="ai-gateway",
            tier=ServiceTier.DOMAIN,
            domain="artificial_intelligence",
            resource_profile=ResourceProfile.MEMORY_INTENSIVE,
            communication_patterns={CommunicationPattern.SYNCHRONOUS},
            dependencies={"model-registry", "inference-pipeline"},
            data_stores={"redis-cache", "vector-store"},
            exposed_apis={"ai-api", "llm-api", "ml-api"},
            consumed_events={"model.updated", "inference.requested"},
            published_events={"inference.completed", "model.performance"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=7,
                numa_node=0,  # High memory access patterns
                l3_cache_sensitivity=True,
                thermal_policy="performance",
                memory_policy="preferred"
            ),
            security_level="high",
            scalability_pattern="horizontal",
            observability_level="enhanced",
            caching_strategy="hybrid"
        )
        
        services["threat-intelligence"] = ServiceDefinition(
            name="threat-intelligence",
            tier=ServiceTier.DOMAIN,
            domain="intelligence_analysis",
            resource_profile=ResourceProfile.BALANCED,
            communication_patterns={CommunicationPattern.STREAMING, CommunicationPattern.ASYNCHRONOUS},
            dependencies={"ai-gateway", "external-feeds"},
            data_stores={"postgres-primary", "vector-store", "search-engine"},
            exposed_apis={"threat-intel-api", "indicators-api"},
            consumed_events={"external.feed", "analysis.request"},
            published_events={"threat.detected", "indicator.updated", "intel.enriched"},
            epyc_optimization=EPYCOptimization(
                ccx_affinity=0,  # Share with core services
                numa_node=0,
                thermal_policy="balanced"
            ),
            security_level="high",
            scalability_pattern="horizontal",
            caching_strategy="distributed"
        )
        
        # ==============================
        # PLATFORM TIER - Infrastructure Services
        # ==============================
        
        services["api-gateway"] = ServiceDefinition(
            name="api-gateway",
            tier=ServiceTier.PLATFORM,
            domain="platform_infrastructure",
            resource_profile=ResourceProfile.IO_INTENSIVE,
            communication_patterns={CommunicationPattern.SYNCHRONOUS},
            dependencies={"auth-service", "rate-limiter", "metrics-collector"},
            data_stores={"redis-cache"},
            exposed_apis={"public-api", "partner-api"},
            consumed_events={"auth.validated", "rate.limited"},
            published_events={"request.received", "request.completed"},
            epyc_optimization=EPYCOptimization(
                numa_node=0,
                thermal_policy="conservative",  # Handle request spikes
                core_allocation=[4, 5, 6, 7]   # Dedicated cores for I/O
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            observability_level="enhanced",
            caching_strategy="local"
        )
        
        services["auth-service"] = ServiceDefinition(
            name="auth-service",
            tier=ServiceTier.PLATFORM,
            domain="platform_infrastructure",
            resource_profile=ResourceProfile.LIGHTWEIGHT,
            communication_patterns={CommunicationPattern.SYNCHRONOUS},
            dependencies={"user-registry", "session-store"},
            data_stores={"postgres-primary", "redis-cache"},
            exposed_apis={"auth-api", "identity-api"},
            consumed_events={"user.login", "session.expired"},
            published_events={"auth.validated", "auth.failed", "session.created"},
            epyc_optimization=EPYCOptimization(
                thermal_policy="conservative",
                memory_policy="interleave"
            ),
            security_level="critical",
            scalability_pattern="horizontal",
            caching_strategy="distributed"
        )
        
        services["metrics-collector"] = ServiceDefinition(
            name="metrics-collector",
            tier=ServiceTier.PLATFORM,
            domain="observability",
            resource_profile=ResourceProfile.IO_INTENSIVE,
            communication_patterns={CommunicationPattern.STREAMING},
            dependencies={"time-series-db", "alert-manager"},
            data_stores={"time-series-db", "postgres-analytics"},
            exposed_apis={"metrics-api", "health-api"},
            consumed_events={"metric.reported", "alert.triggered"},
            published_events={"metric.aggregated", "threshold.exceeded"},
            epyc_optimization=EPYCOptimization(
                numa_node=1,  # Separate from core services
                thermal_policy="conservative"
            ),
            security_level="standard",
            scalability_pattern="horizontal",
            caching_strategy="local"
        )
        
        # ==============================
        # EDGE TIER - External-Facing Services
        # ==============================
        
        services["web-interface"] = ServiceDefinition(
            name="web-interface",
            tier=ServiceTier.EDGE,
            domain="user_interface",
            resource_profile=ResourceProfile.LIGHTWEIGHT,
            communication_patterns={CommunicationPattern.SYNCHRONOUS},
            dependencies={"api-gateway", "asset-service"},
            data_stores={"redis-cache"},
            exposed_apis={"web-ui", "dashboard-api"},
            epyc_optimization=EPYCOptimization(
                thermal_policy="conservative"
            ),
            security_level="high",
            scalability_pattern="horizontal",
            caching_strategy="local"
        )
        
        services["external-integrations"] = ServiceDefinition(
            name="external-integrations",
            tier=ServiceTier.EDGE,
            domain="external_connectivity",
            resource_profile=ResourceProfile.IO_INTENSIVE,
            communication_patterns={CommunicationPattern.HYBRID},
            dependencies={"api-gateway", "auth-service"},
            data_stores={"postgres-primary", "redis-cache"},
            exposed_apis={"webhook-api", "integration-api"},
            consumed_events={"integration.configured", "webhook.received"},
            published_events={"external.data", "integration.status"},
            epyc_optimization=EPYCOptimization(
                numa_node=1,
                thermal_policy="conservative"
            ),
            security_level="high",
            scalability_pattern="horizontal"
        )
        
        return services
    
    def _define_service_mesh(self) -> Dict[str, any]:
        """Define service mesh configuration for optimal communication."""
        return {
            "mesh_provider": "istio",
            "traffic_management": {
                "load_balancing": "round_robin",
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout": "30s",
                    "min_request_amount": 20
                },
                "retry_policy": {
                    "attempts": 3,
                    "per_try_timeout": "2s",
                    "retry_on": ["5xx", "gateway-error", "connect-failure"]
                },
                "timeout": {
                    "default": "30s",
                    "ai_services": "120s",
                    "scanner_services": "300s"
                }
            },
            "security": {
                "mtls": {
                    "mode": "STRICT"
                },
                "authorization": {
                    "default_policy": "DENY",
                    "rules_based_on": "service_accounts"
                }
            },
            "observability": {
                "tracing": {
                    "sampling_rate": 0.1,
                    "jaeger_endpoint": "jaeger-collector:14268"
                },
                "metrics": {
                    "prometheus_endpoint": "prometheus:9090"
                }
            }
        }
    
    def _define_deployment_topology(self) -> Dict[str, any]:
        """Define EPYC-optimized deployment topology."""
        return {
            "node_affinity": {
                "core_services": {
                    "required": ["node.kubernetes.io/cpu-family=EPYC"],
                    "preferred": ["node.kubernetes.io/numa-topology=dual"]
                },
                "ai_services": {
                    "required": ["node.kubernetes.io/cpu-family=EPYC"],
                    "preferred": ["node.kubernetes.io/cache-size=large"]
                }
            },
            "pod_anti_affinity": {
                "critical_services": "required",
                "standard_services": "preferred"
            },
            "resource_quotas": {
                "core_tier": {
                    "cpu_limit": "16000m",
                    "memory_limit": "32Gi",
                    "numa_nodes": [0]
                },
                "domain_tier": {
                    "cpu_limit": "12000m", 
                    "memory_limit": "24Gi",
                    "numa_nodes": [0, 1]
                },
                "platform_tier": {
                    "cpu_limit": "8000m",
                    "memory_limit": "16Gi", 
                    "numa_nodes": [1]
                }
            },
            "scaling_policies": {
                "cpu_threshold": 60,  # Conservative for EPYC
                "memory_threshold": 70,
                "scale_up_cooldown": "300s",
                "scale_down_cooldown": "600s"
            }
        }
    
    def get_service_dependencies(self, service_name: str) -> Set[str]:
        """Get all dependencies for a service."""
        if service_name not in self.services:
            return set()
        return self.services[service_name].dependencies
    
    def get_services_by_tier(self, tier: ServiceTier) -> List[ServiceDefinition]:
        """Get all services in a specific tier."""
        return [svc for svc in self.services.values() if svc.tier == tier]
    
    def get_services_by_domain(self, domain: str) -> List[ServiceDefinition]:
        """Get all services in a specific domain."""
        return [svc for svc in self.services.values() if svc.domain == domain]
    
    def validate_architecture(self) -> List[str]:
        """Validate the architecture for common anti-patterns."""
        issues = []
        
        # Check for circular dependencies
        for service_name, service in self.services.items():
            if self._has_circular_dependency(service_name, service.dependencies):
                issues.append(f"Circular dependency detected for {service_name}")
        
        # Check for single points of failure
        for service_name, service in self.services.items():
            dependents = self._get_dependents(service_name)
            if len(dependents) > 5 and service.scalability_pattern == "static":
                issues.append(f"Potential SPOF: {service_name} has many dependents but static scaling")
        
        # Check EPYC optimization conflicts
        core_allocations = {}
        for service_name, service in self.services.items():
            if service.epyc_optimization.core_allocation:
                for core in service.epyc_optimization.core_allocation:
                    if core in core_allocations:
                        issues.append(f"Core allocation conflict: {service_name} and {core_allocations[core]} both use core {core}")
                    core_allocations[core] = service_name
        
        return issues
    
    def _has_circular_dependency(self, service_name: str, dependencies: Set[str], visited: Set[str] = None) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()
        
        if service_name in visited:
            return True
        
        visited.add(service_name)
        
        for dep in dependencies:
            if dep in self.services:
                if self._has_circular_dependency(dep, self.services[dep].dependencies, visited.copy()):
                    return True
        
        return False
    
    def _get_dependents(self, service_name: str) -> Set[str]:
        """Get all services that depend on the given service."""
        dependents = set()
        for svc_name, svc in self.services.items():
            if service_name in svc.dependencies:
                dependents.add(svc_name)
        return dependents

# Global architecture instance
XORB_ARCHITECTURE = XORBArchitecture()