#!/usr/bin/env python3
"""
XORB Microservices Deployment System
Advanced microservices deployment with dependency management, health checks, and service mesh integration
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger('XORBMicroservices')

class ServiceStatus(Enum):
    """Microservice deployment status"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    UPDATING = "updating"
    STOPPED = "stopped"
    SCALING = "scaling"

class DeploymentStrategy(Enum):
    """Deployment strategies"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"

class HealthCheckType(Enum):
    """Health check types"""
    HTTP = "http"
    TCP = "tcp"
    EXEC = "exec"
    GRPC = "grpc"

@dataclass
class HealthCheck:
    """Health check configuration"""
    type: HealthCheckType
    path: str = "/"
    port: int = 8080
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1
    command: List[str] = field(default_factory=list)

@dataclass
class ResourceRequirements:
    """Resource requirements for microservices"""
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    storage_request: str = "1Gi"
    ephemeral_storage_limit: str = "2Gi"

@dataclass
class AutoscalingConfig:
    """Autoscaling configuration"""
    enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    target_requests_per_second: int = 100
    scale_up_stabilization_window: int = 60
    scale_down_stabilization_window: int = 300

@dataclass
class ServiceMeshConfig:
    """Service mesh configuration"""
    enabled: bool = False
    inject_sidecar: bool = True
    enable_mtls: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    circuit_breaker_enabled: bool = True

@dataclass
class SecurityConfig:
    """Security configuration for microservices"""
    run_as_non_root: bool = True
    run_as_user: int = 1000
    run_as_group: int = 2000
    fs_group: int = 2000
    read_only_root_filesystem: bool = True
    allow_privilege_escalation: bool = False
    drop_capabilities: List[str] = field(default_factory=lambda: ["ALL"])
    add_capabilities: List[str] = field(default_factory=list)
    seccomp_profile: str = "RuntimeDefault"
    apparmor_profile: str = "runtime/default"

@dataclass
class MicroserviceConfig:
    """Comprehensive microservice configuration"""
    name: str
    image: str
    version: str = "latest"
    port: int = 8080
    replicas: int = 1
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: List[str] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    volume_mounts: List[Dict[str, Any]] = field(default_factory=list)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    health_check: HealthCheck = field(default_factory=lambda: HealthCheck(HealthCheckType.HTTP))
    readiness_check: Optional[HealthCheck] = None
    liveness_check: Optional[HealthCheck] = None
    startup_check: Optional[HealthCheck] = None
    autoscaling: AutoscalingConfig = field(default_factory=AutoscalingConfig)
    service_mesh: ServiceMeshConfig = field(default_factory=ServiceMeshConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Dict[str, Any] = field(default_factory=dict)
    service_account: str = ""
    enable_prometheus_scraping: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    enable_jaeger_tracing: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"

@dataclass
class MicroserviceDeployment:
    """Microservice deployment state"""
    config: MicroserviceConfig
    status: ServiceStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    deployment_manifest: str = ""
    service_manifest: str = ""
    current_replicas: int = 0
    ready_replicas: int = 0
    updated_replicas: int = 0
    endpoints: Dict[str, str] = field(default_factory=dict)
    health_status: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    pods: List[Dict[str, Any]] = field(default_factory=list)

class XORBMicroservicesDeployment:
    """Advanced microservices deployment system"""
    
    def __init__(self, namespace: str = "xorb-services", platform: str = "kubernetes"):
        self.namespace = namespace
        self.platform = platform
        self.services: Dict[str, MicroserviceDeployment] = {}
        self.deployment_id = f"microservices_{int(time.time())}"
        
        # Initialize service configurations
        self._initialize_microservice_configs()
        
        logger.info(f"🚀 Microservices deployment system initialized")
        logger.info(f"📋 Namespace: {self.namespace}")
        logger.info(f"🎯 Platform: {self.platform}")
        logger.info(f"🆔 Deployment ID: {self.deployment_id}")
    
    def _initialize_microservice_configs(self):
        """Initialize microservice configurations"""
        
        # API Gateway Service
        api_gateway_config = MicroserviceConfig(
            name="xorb-api-gateway",
            image="xorb/api-gateway",
            version="2.0.0",
            port=8080,
            replicas=3,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            dependencies=["postgresql-primary", "redis-cluster", "vault"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8080",
                "METRICS_PORT": "9090",
                "DATABASE_URL": "postgresql://xorb_user:${POSTGRES_PASSWORD}@postgresql-primary:5432/xorb",
                "REDIS_URL": "redis://redis-cluster:6379",
                "VAULT_URL": "http://vault:8200"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="200m",
                cpu_limit="1000m",
                memory_request="256Mi",
                memory_limit="1Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=2,
                max_replicas=20,
                target_cpu_utilization=70
            ),
            service_mesh=ServiceMeshConfig(
                enabled=True,
                inject_sidecar=True,
                enable_mtls=True
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8080,
                initial_delay_seconds=30
            )
        )
        
        self.services["xorb-api-gateway"] = MicroserviceDeployment(
            config=api_gateway_config,
            status=ServiceStatus.PENDING
        )
        
        # Neural Orchestrator Service
        orchestrator_config = MicroserviceConfig(
            name="xorb-neural-orchestrator",
            image="xorb/neural-orchestrator",
            version="2.0.0",
            port=8003,
            replicas=2,
            dependencies=["postgresql-primary", "redis-cluster", "neo4j-cluster"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8003",
                "DATABASE_URL": "postgresql://xorb_user:${POSTGRES_PASSWORD}@postgresql-primary:5432/xorb",
                "REDIS_URL": "redis://redis-cluster:6379",
                "NEO4J_URL": "bolt://neo4j-cluster:7687"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=1,
                max_replicas=10,
                target_cpu_utilization=75
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8003,
                initial_delay_seconds=60
            )
        )
        
        self.services["xorb-neural-orchestrator"] = MicroserviceDeployment(
            config=orchestrator_config,
            status=ServiceStatus.PENDING
        )
        
        # Learning Service
        learning_config = MicroserviceConfig(
            name="xorb-learning-service",
            image="xorb/learning-service",
            version="2.0.0",
            port=8004,
            replicas=2,
            dependencies=["postgresql-primary", "redis-cluster", "elasticsearch-cluster"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8004",
                "DATABASE_URL": "postgresql://xorb_user:${POSTGRES_PASSWORD}@postgresql-primary:5432/xorb",
                "REDIS_URL": "redis://redis-cluster:6379",
                "ELASTICSEARCH_URL": "http://elasticsearch-cluster:9200"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="1000m",
                cpu_limit="4000m",
                memory_request="2Gi",
                memory_limit="8Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=1,
                max_replicas=8,
                target_cpu_utilization=80
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8004,
                initial_delay_seconds=90
            )
        )
        
        self.services["xorb-learning-service"] = MicroserviceDeployment(
            config=learning_config,
            status=ServiceStatus.PENDING
        )
        
        # Threat Detection Service
        threat_detection_config = MicroserviceConfig(
            name="xorb-threat-detection",
            image="xorb/threat-detection",
            version="2.0.0",
            port=8005,
            replicas=3,
            dependencies=["elasticsearch-cluster", "redis-cluster"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8005",
                "ELASTICSEARCH_URL": "http://elasticsearch-cluster:9200",
                "REDIS_URL": "redis://redis-cluster:6379"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=2,
                max_replicas=15,
                target_cpu_utilization=70
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8005
            )
        )
        
        self.services["xorb-threat-detection"] = MicroserviceDeployment(
            config=threat_detection_config,
            status=ServiceStatus.PENDING
        )
        
        # Worker Pool Service
        worker_pool_config = MicroserviceConfig(
            name="xorb-worker-pool",
            image="xorb/worker",
            version="2.0.0",
            port=8001,
            replicas=5,
            dependencies=["postgresql-primary", "redis-cluster", "neo4j-cluster"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8001",
                "DATABASE_URL": "postgresql://xorb_user:${POSTGRES_PASSWORD}@postgresql-primary:5432/xorb",
                "REDIS_URL": "redis://redis-cluster:6379",
                "NEO4J_URL": "bolt://neo4j-cluster:7687"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="300m",
                cpu_limit="1000m",
                memory_request="512Mi",
                memory_limit="2Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=3,
                max_replicas=50,
                target_cpu_utilization=75
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8001
            )
        )
        
        self.services["xorb-worker-pool"] = MicroserviceDeployment(
            config=worker_pool_config,
            status=ServiceStatus.PENDING
        )
        
        # Analytics Engine
        analytics_config = MicroserviceConfig(
            name="xorb-analytics-engine",
            image="xorb/analytics",
            version="2.0.0",
            port=8006,
            replicas=2,
            dependencies=["elasticsearch-cluster", "postgresql-primary"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8006",
                "ELASTICSEARCH_URL": "http://elasticsearch-cluster:9200",
                "DATABASE_URL": "postgresql://xorb_user:password@postgresql-primary:5432/xorb"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=1,
                max_replicas=8,
                target_cpu_utilization=70
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8006
            )
        )
        
        self.services["xorb-analytics-engine"] = MicroserviceDeployment(
            config=analytics_config,
            status=ServiceStatus.PENDING
        )
        
        # Notification Service
        notification_config = MicroserviceConfig(
            name="xorb-notification-service",
            image="xorb/notifications",
            version="2.0.0",
            port=8007,
            replicas=2,
            dependencies=["redis-cluster", "postgresql-primary"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8007",
                "REDIS_URL": "redis://redis-cluster:6379",
                "DATABASE_URL": "postgresql://xorb_user:password@postgresql-primary:5432/xorb"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="200m",
                cpu_limit="500m",
                memory_request="256Mi",
                memory_limit="1Gi"
            ),
            autoscaling=AutoscalingConfig(
                enabled=True,
                min_replicas=1,
                max_replicas=10,
                target_cpu_utilization=70
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8007
            )
        )
        
        self.services["xorb-notification-service"] = MicroserviceDeployment(
            config=notification_config,
            status=ServiceStatus.PENDING
        )
        
        # Backup Service
        backup_config = MicroserviceConfig(
            name="xorb-backup-service",
            image="xorb/backup",
            version="2.0.0",
            port=8008,
            replicas=1,
            dependencies=["postgresql-primary", "redis-cluster", "neo4j-cluster"],
            environment_variables={
                "LOG_LEVEL": "INFO",
                "PORT": "8008",
                "DATABASE_URL": "postgresql://xorb_user:${POSTGRES_PASSWORD}@postgresql-primary:5432/xorb",
                "REDIS_URL": "redis://redis-cluster:6379",
                "NEO4J_URL": "bolt://neo4j-cluster:7687"
            },
            resource_requirements=ResourceRequirements(
                cpu_request="200m",
                cpu_limit="1000m",
                memory_request="512Mi",
                memory_limit="2Gi"
            ),
            health_check=HealthCheck(
                type=HealthCheckType.HTTP,
                path="/health",
                port=8008
            )
        )
        
        self.services["xorb-backup-service"] = MicroserviceDeployment(
            config=backup_config,
            status=ServiceStatus.PENDING
        )
        
        logger.info(f"🔧 Initialized {len(self.services)} microservice configurations")
    
    async def deploy_microservices(self) -> bool:
        """Deploy all microservices with dependency management"""
        try:
            logger.info("🚀 Starting microservices deployment")
            
            # Create namespace
            if self.platform == "kubernetes":
                await self._create_kubernetes_namespace()
            
            # Create deployment plan based on dependencies
            deployment_plan = self._create_deployment_plan()
            
            # Execute deployment phases
            for phase_num, services_in_phase in enumerate(deployment_plan, 1):
                logger.info(f"📋 Deploying Phase {phase_num}: {[s.config.name for s in services_in_phase]}")
                
                # Deploy services in parallel within phase
                tasks = []
                for service_deployment in services_in_phase:
                    task = asyncio.create_task(self._deploy_microservice(service_deployment))
                    tasks.append(task)
                
                # Wait for all services in phase to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                failed_services = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception) or not result:
                        failed_services.append(services_in_phase[i])
                
                if failed_services:
                    logger.error(f"❌ Phase {phase_num} failed for services: {[s.config.name for s in failed_services]}")
                    return False
                
                logger.info(f"✅ Phase {phase_num} completed successfully")
            
            # Setup service discovery and networking
            await self._setup_service_discovery()
            
            # Setup monitoring for all services
            await self._setup_microservices_monitoring()
            
            logger.info("🎉 Microservices deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Microservices deployment failed: {e}")
            return False
    
    def _create_deployment_plan(self) -> List[List[MicroserviceDeployment]]:
        """Create deployment plan based on service dependencies"""
        
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for service_name, service_deployment in self.services.items():
            dependencies = service_deployment.config.dependencies
            # Filter dependencies to only include other microservices
            service_dependencies = [dep for dep in dependencies if dep in self.services]
            graph[service_name] = service_dependencies
            in_degree[service_name] = len(service_dependencies)
        
        # Topological sort
        phases = []
        remaining = set(self.services.keys())
        
        while remaining:
            # Find services with no dependencies
            ready = [name for name in remaining if in_degree[name] == 0]
            
            if not ready:
                # Handle circular dependencies
                ready = [min(remaining)]
                logger.warning(f"⚠️  Circular dependency detected, forcing deployment of {ready[0]}")
            
            # Create phase with ready services
            phase_services = [self.services[name] for name in ready]
            phases.append(phase_services)
            
            # Remove deployed services and update dependencies
            for name in ready:
                remaining.remove(name)
                for other_name in remaining:
                    if name in graph[other_name]:
                        graph[other_name].remove(name)
                        in_degree[other_name] -= 1
        
        return phases
    
    async def _create_kubernetes_namespace(self) -> bool:
        """Create Kubernetes namespace for microservices"""
        try:
            namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
  labels:
    name: {self.namespace}
    component: microservices
    managed-by: xorb-deployment
    istio-injection: enabled
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: {self.namespace}-quota
  namespace: {self.namespace}
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    limits.cpu: "100"
    limits.memory: 200Gi
    persistentvolumeclaims: "50"
    services: "50"
    count/deployments.apps: "100"
    count/horizontalpodautoscalers.autoscaling: "50"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: {self.namespace}-limits
  namespace: {self.namespace}
spec:
  limits:
  - default:
      cpu: "1"
      memory: "1Gi"
    defaultRequest:
      cpu: "100m"
      memory: "128Mi"
    type: Container
"""
            
            return await self._kubectl_apply(namespace_yaml)
            
        except Exception as e:
            logger.error(f"❌ Failed to create microservices namespace: {e}")
            return False
    
    async def _deploy_microservice(self, service_deployment: MicroserviceDeployment) -> bool:
        """Deploy individual microservice"""
        try:
            config = service_deployment.config
            logger.info(f"🔧 Deploying microservice: {config.name}")
            
            service_deployment.status = ServiceStatus.DEPLOYING
            service_deployment.start_time = datetime.now()
            
            if self.platform == "kubernetes":
                success = await self._deploy_kubernetes_microservice(service_deployment)
            else:
                success = await self._deploy_docker_microservice(service_deployment)
            
            service_deployment.end_time = datetime.now()
            
            if success:
                service_deployment.status = ServiceStatus.RUNNING
                logger.info(f"✅ Microservice {config.name} deployed successfully")
                
                # Wait for service to become healthy
                await self._wait_for_service_health(service_deployment)
                
                return True
            else:
                service_deployment.status = ServiceStatus.FAILED
                logger.error(f"❌ Microservice {config.name} deployment failed")
                return False
                
        except Exception as e:
            service_deployment.status = ServiceStatus.FAILED
            service_deployment.error_message = str(e)
            service_deployment.end_time = datetime.now()
            logger.error(f"❌ Microservice {service_deployment.config.name} deployment failed: {e}")
            return False
    
    async def _deploy_kubernetes_microservice(self, service_deployment: MicroserviceDeployment) -> bool:
        """Deploy microservice to Kubernetes"""
        try:
            config = service_deployment.config
            
            # Generate all Kubernetes manifests
            manifests = await self._generate_kubernetes_manifests(service_deployment)
            
            # Apply manifests in order
            manifest_order = [
                "serviceaccount",
                "configmap",
                "secret",
                "service",
                "deployment",
                "hpa",
                "servicemonitor",
                "networkpolicy",
                "poddisruptionbudget"
            ]
            
            for manifest_type in manifest_order:
                if manifest_type in manifests:
                    logger.info(f"📄 Applying {manifest_type} for {config.name}")
                    success = await self._kubectl_apply(manifests[manifest_type])
                    if not success:
                        logger.error(f"❌ Failed to apply {manifest_type} for {config.name}")
                        return False
            
            # Store manifests
            service_deployment.deployment_manifest = manifests.get("deployment", "")
            service_deployment.service_manifest = manifests.get("service", "")
            
            # Set endpoints
            service_deployment.endpoints = {
                "http": f"{config.name}.{self.namespace}.svc.cluster.local:{config.port}",
                "metrics": f"{config.name}.{self.namespace}.svc.cluster.local:{config.prometheus_port}"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Kubernetes microservice deployment failed: {e}")
            return False
    
    async def _generate_kubernetes_manifests(self, service_deployment: MicroserviceDeployment) -> Dict[str, str]:
        """Generate comprehensive Kubernetes manifests for microservice"""
        config = service_deployment.config
        manifests = {}
        
        # ServiceAccount
        manifests["serviceaccount"] = f"""
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {config.name}-sa
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: microservice
  annotations:
    {"helm.sh/resource-policy": "keep"}
automountServiceAccountToken: true
"""
        
        # ConfigMap for application configuration
        manifests["configmap"] = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {config.name}-config
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: microservice
data:
  application.yaml: |
    server:
      port: {config.port}
    logging:
      level: {config.log_level}
    management:
      endpoints:
        web:
          exposure:
            include: health,info,metrics,prometheus
      endpoint:
        health:
          show-details: always
    spring:
      application:
        name: {config.name}
"""
        
        # Secret for sensitive data
        manifests["secret"] = f"""
apiVersion: v1
kind: Secret
metadata:
  name: {config.name}-secrets
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: microservice
type: Opaque
data:
  database-password: {self._base64_encode("xorb_secure_password_123!")}
  redis-password: {self._base64_encode("redis_secure_password_123!")}
  jwt-secret: {self._base64_encode("xorb_jwt_secret_key_123456789")}
"""
        
        # Service
        manifests["service"] = f"""
apiVersion: v1
kind: Service
metadata:
  name: {config.name}
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: microservice
    service: {config.name}
  annotations:
    {"prometheus.io/scrape": "true" if config.enable_prometheus_scraping else "false"}
    {"prometheus.io/port": str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
    {"prometheus.io/path": config.prometheus_path if config.enable_prometheus_scraping else ""}
    {"service.alpha.kubernetes.io/tolerate-unready-endpoints": "true"}
spec:
  selector:
    app: {config.name}
  ports:
  - name: http
    port: {config.port}
    targetPort: {config.port}
    protocol: TCP
  {"- name: metrics" if config.enable_prometheus_scraping else ""}
  {"  port: " + str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
  {"  targetPort: " + str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
  {"  protocol: TCP" if config.enable_prometheus_scraping else ""}
  type: ClusterIP
  sessionAffinity: None
"""
        
        # Deployment
        strategy_config = self._get_deployment_strategy_config(config.strategy)
        
        manifests["deployment"] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.name}
  namespace: {self.namespace}
  labels:
    app: {config.name}
    version: {config.version}
    component: microservice
    managed-by: xorb-deployment
  annotations:
    deployment.kubernetes.io/revision: "1"
    xorb.io/deployment-id: "{self.deployment_id}"
    xorb.io/deployment-strategy: "{config.strategy.value}"
spec:
  replicas: {config.replicas}
  strategy:
{strategy_config}
  selector:
    matchLabels:
      app: {config.name}
  template:
    metadata:
      labels:
        app: {config.name}
        version: {config.version}
        component: microservice
        {"sidecar.istio.io/inject": "true" if config.service_mesh.enabled else "false"}
      annotations:
        {"prometheus.io/scrape": "true" if config.enable_prometheus_scraping else "false"}
        {"prometheus.io/port": str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
        {"prometheus.io/path": config.prometheus_path if config.enable_prometheus_scraping else ""}
        {"sidecar.istio.io/inject": "true" if config.service_mesh.enabled else "false"}
        {"co.elastic.logs/enabled": "true" if config.enable_logging else "false"}
        {"co.elastic.logs/json.keys_under_root": "true" if config.enable_logging else "false"}
    spec:
      serviceAccountName: {config.service_account or f"{config.name}-sa"}
      securityContext:
        runAsNonRoot: {str(config.security.run_as_non_root).lower()}
        runAsUser: {config.security.run_as_user}
        runAsGroup: {config.security.run_as_group}
        fsGroup: {config.security.fs_group}
        seccompProfile:
          type: {config.security.seccomp_profile}
      {"nodeSelector:" if config.node_selector else ""}
{self._format_yaml_dict(config.node_selector, 8) if config.node_selector else ""}
      {"tolerations:" if config.tolerations else ""}
{self._format_yaml_list(config.tolerations, 6) if config.tolerations else ""}
      {"affinity:" if config.affinity else ""}
{self._format_yaml_dict(config.affinity, 8) if config.affinity else ""}
      containers:
      - name: {config.name}
        image: {config.image}:{config.version}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: {config.port}
          name: http
          protocol: TCP
        {"- containerPort: " + str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
        {"  name: metrics" if config.enable_prometheus_scraping else ""}
        {"  protocol: TCP" if config.enable_prometheus_scraping else ""}
        env:
"""
        
        # Add environment variables
        for env_key, env_value in config.environment_variables.items():
            manifests["deployment"] += f"""
        - name: {env_key}
          value: "{env_value}"
"""
        
        # Add common environment variables
        manifests["deployment"] += f"""
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: KUBERNETES_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: KUBERNETES_POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: KUBERNETES_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: SERVICE_NAME
          value: "{config.name}"
        - name: SERVICE_VERSION
          value: "{config.version}"
        - name: DEPLOYMENT_ID
          value: "{self.deployment_id}"
        {"- name: JAEGER_AGENT_HOST" if config.enable_jaeger_tracing else ""}
        {"  valueFrom:" if config.enable_jaeger_tracing else ""}
        {"    fieldRef:" if config.enable_jaeger_tracing else ""}
        {"      fieldPath: status.hostIP" if config.enable_jaeger_tracing else ""}
        {"- name: JAEGER_SERVICE_NAME" if config.enable_jaeger_tracing else ""}
        {"  value: " + config.name if config.enable_jaeger_tracing else ""}
        envFrom:
        - configMapRef:
            name: {config.name}-config
        - secretRef:
            name: {config.name}-secrets
        resources:
          requests:
            memory: "{config.resource_requirements.memory_request}"
            cpu: "{config.resource_requirements.cpu_request}"
            {"ephemeral-storage: " + config.resource_requirements.storage_request if config.resource_requirements.storage_request else ""}
          limits:
            memory: "{config.resource_requirements.memory_limit}"
            cpu: "{config.resource_requirements.cpu_limit}"
            {"ephemeral-storage: " + config.resource_requirements.ephemeral_storage_limit if config.resource_requirements.ephemeral_storage_limit else ""}
        {"volumeMounts:" if config.volume_mounts else ""}
{self._format_yaml_list(config.volume_mounts, 8) if config.volume_mounts else ""}
        livenessProbe:
{self._format_health_check(config.liveness_check or config.health_check, 10)}
        readinessProbe:
{self._format_health_check(config.readiness_check or config.health_check, 10)}
        startupProbe:
{self._format_health_check(config.startup_check or config.health_check, 10, startup=True)}
        securityContext:
          allowPrivilegeEscalation: {str(config.security.allow_privilege_escalation).lower()}
          readOnlyRootFilesystem: {str(config.security.read_only_root_filesystem).lower()}
          runAsNonRoot: {str(config.security.run_as_non_root).lower()}
          runAsUser: {config.security.run_as_user}
          runAsGroup: {config.security.run_as_group}
          capabilities:
            {"drop: " + str(config.security.drop_capabilities) if config.security.drop_capabilities else ""}
            {"add: " + str(config.security.add_capabilities) if config.security.add_capabilities else ""}
          seccompProfile:
            type: {config.security.seccomp_profile}
      {"volumes:" if config.volumes else ""}
{self._format_yaml_list(config.volumes, 6) if config.volumes else ""}
      terminationGracePeriodSeconds: 30
      dnsPolicy: ClusterFirst
      restartPolicy: Always
"""
        
        # HorizontalPodAutoscaler
        if config.autoscaling.enabled:
            manifests["hpa"] = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {config.name}-hpa
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: microservice
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {config.name}
  minReplicas: {config.autoscaling.min_replicas}
  maxReplicas: {config.autoscaling.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {config.autoscaling.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {config.autoscaling.target_memory_utilization}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: {config.autoscaling.scale_down_stabilization_window}
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: {config.autoscaling.scale_up_stabilization_window}
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
"""
        
        # ServiceMonitor for Prometheus
        if config.enable_prometheus_scraping:
            manifests["servicemonitor"] = f"""
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {config.name}-monitor
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: monitoring
spec:
  selector:
    matchLabels:
      app: {config.name}
  endpoints:
  - port: metrics
    interval: 30s
    path: {config.prometheus_path}
    scrapeTimeout: 10s
"""
        
        # NetworkPolicy
        manifests["networkpolicy"] = f"""
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {config.name}-netpol
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: security
spec:
  podSelector:
    matchLabels:
      app: {config.name}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: {self.namespace}
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - protocol: TCP
      port: {config.port}
    {"- protocol: TCP" if config.enable_prometheus_scraping else ""}
    {"  port: " + str(config.prometheus_port) if config.enable_prometheus_scraping else ""}
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: {self.namespace}
  - to:
    - namespaceSelector:
        matchLabels:
          name: xorb-infra
  - to: {{}}
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
"""
        
        # PodDisruptionBudget
        manifests["poddisruptionbudget"] = f"""
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {config.name}-pdb
  namespace: {self.namespace}
  labels:
    app: {config.name}
    component: availability
spec:
  selector:
    matchLabels:
      app: {config.name}
  {"maxUnavailable: 1" if config.replicas > 1 else "minAvailable: 0"}
"""
        
        return manifests
    
    def _get_deployment_strategy_config(self, strategy: DeploymentStrategy) -> str:
        """Get Kubernetes deployment strategy configuration"""
        if strategy == DeploymentStrategy.ROLLING_UPDATE:
            return """    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%"""
        elif strategy == DeploymentStrategy.RECREATE:
            return """    type: Recreate"""
        else:
            # Default to rolling update
            return """    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0"""
    
    def _format_health_check(self, health_check: HealthCheck, indent: int, startup: bool = False) -> str:
        """Format health check configuration"""
        spaces = " " * indent
        
        if health_check.type == HealthCheckType.HTTP:
            config = f"""{spaces}httpGet:
{spaces}  path: {health_check.path}
{spaces}  port: {health_check.port}
{spaces}  scheme: HTTP"""
        elif health_check.type == HealthCheckType.TCP:
            config = f"""{spaces}tcpSocket:
{spaces}  port: {health_check.port}"""
        elif health_check.type == HealthCheckType.EXEC:
            config = f"""{spaces}exec:
{spaces}  command: {health_check.command}"""
        else:
            config = f"""{spaces}httpGet:
{spaces}  path: /health
{spaces}  port: {health_check.port}"""
        
        # Add timing configuration
        initial_delay = health_check.initial_delay_seconds
        if startup:
            initial_delay = max(initial_delay, 10)
            
        config += f"""
{spaces}initialDelaySeconds: {initial_delay}
{spaces}periodSeconds: {health_check.period_seconds}
{spaces}timeoutSeconds: {health_check.timeout_seconds}
{spaces}failureThreshold: {health_check.failure_threshold}
{spaces}successThreshold: {health_check.success_threshold}"""
        
        return config
    
    def _format_yaml_dict(self, data: Dict[str, Any], indent: int) -> str:
        """Format dictionary as YAML with proper indentation"""
        if not data:
            return ""
        
        spaces = " " * indent
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{spaces}{key}:")
                lines.append(self._format_yaml_dict(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {value}")
        
        return "\n".join(lines)
    
    def _format_yaml_list(self, data: List[Any], indent: int) -> str:
        """Format list as YAML with proper indentation"""
        if not data:
            return ""
        
        spaces = " " * indent
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{spaces}- ")
                for key, value in item.items():
                    lines.append(f"{spaces}  {key}: {value}")
            else:
                lines.append(f"{spaces}- {item}")
        
        return "\n".join(lines)
    
    async def _deploy_docker_microservice(self, service_deployment: MicroserviceDeployment) -> bool:
        """Deploy microservice using Docker Compose"""
        try:
            # This would integrate with existing Docker Compose deployment
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker microservice deployment failed: {e}")
            return False
    
    async def _wait_for_service_health(self, service_deployment: MicroserviceDeployment, timeout: int = 300) -> bool:
        """Wait for service to become healthy"""
        try:
            config = service_deployment.config
            logger.info(f"⏳ Waiting for {config.name} to become healthy...")
            
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                health_status = await self._check_service_health(service_deployment)
                
                if health_status:
                    service_deployment.health_status = True
                    logger.info(f"✅ {config.name} is healthy")
                    return True
                
                logger.info(f"⏳ {config.name} not ready yet, waiting...")
                await asyncio.sleep(10)
            
            logger.error(f"❌ Timeout waiting for {config.name} to become healthy")
            return False
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {service_deployment.config.name}: {e}")
            return False
    
    async def _check_service_health(self, service_deployment: MicroserviceDeployment) -> bool:
        """Check service health status"""
        try:
            config = service_deployment.config
            
            if self.platform == "kubernetes":
                # Check deployment status
                result = subprocess.run([
                    "kubectl", "rollout", "status", 
                    f"deployment/{config.name}",
                    f"--namespace={self.namespace}",
                    "--timeout=60s"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Check pod readiness
                    result = subprocess.run([
                        "kubectl", "get", "pods",
                        "-l", f"app={config.name}",
                        "-n", self.namespace,
                        "--field-selector=status.phase=Running",
                        "-o", "jsonpath='{.items[*].status.conditions[?(@.type==\"Ready\")].status}'"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and "True" in result.stdout:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Service health check failed: {e}")
            return False
    
    async def _setup_service_discovery(self):
        """Setup service discovery and networking"""
        logger.info("🌐 Setting up service discovery")
        
        # This would setup additional networking components like:
        # - Service discovery
        # - Load balancing
        # - Circuit breakers
        # - Rate limiting
    
    async def _setup_microservices_monitoring(self):
        """Setup monitoring for all microservices"""
        logger.info("📊 Setting up microservices monitoring")
        
        # Create ServiceMonitor resources for Prometheus scraping
        for service_name, service_deployment in self.services.items():
            if (service_deployment.status == ServiceStatus.RUNNING and 
                service_deployment.config.enable_prometheus_scraping):
                await self._setup_service_monitoring(service_deployment)
    
    async def _setup_service_monitoring(self, service_deployment: MicroserviceDeployment):
        """Setup monitoring for individual service"""
        try:
            config = service_deployment.config
            
            if self.platform == "kubernetes":
                # ServiceMonitor should already be created in deployment manifests
                logger.info(f"📊 Monitoring configured for {config.name}")
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed for {service_deployment.config.name}: {e}")
    
    async def _kubectl_apply(self, yaml_content: str) -> bool:
        """Apply Kubernetes YAML configuration"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                f.write(yaml_content)
                temp_file = f.name
            
            cmd = ["kubectl", "apply", "-f", temp_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"❌ kubectl apply failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ kubectl apply failed: {e}")
            return False
    
    def _base64_encode(self, data: str) -> str:
        """Base64 encode string"""
        import base64
        return base64.b64encode(data.encode()).decode()
    
    async def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale microservice"""
        try:
            if service_name not in self.services:
                logger.error(f"❌ Service {service_name} not found")
                return False
            
            service_deployment = self.services[service_name]
            service_deployment.status = ServiceStatus.SCALING
            
            if self.platform == "kubernetes":
                result = subprocess.run([
                    "kubectl", "scale", "deployment", service_name,
                    f"--replicas={replicas}",
                    f"--namespace={self.namespace}"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    service_deployment.config.replicas = replicas
                    service_deployment.status = ServiceStatus.RUNNING
                    logger.info(f"✅ Scaled {service_name} to {replicas} replicas")
                    return True
                else:
                    service_deployment.status = ServiceStatus.FAILED
                    logger.error(f"❌ Failed to scale {service_name}: {result.stderr}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Service scaling failed: {e}")
            return False
    
    def get_microservices_status(self) -> Dict[str, Any]:
        """Get comprehensive microservices status"""
        status = {
            "deployment_id": self.deployment_id,
            "namespace": self.namespace,
            "platform": self.platform,
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "summary": {
                "total": len(self.services),
                "running": 0,
                "failed": 0,
                "deploying": 0,
                "pending": 0
            }
        }
        
        for name, service_deployment in self.services.items():
            config = service_deployment.config
            status["services"][name] = {
                "status": service_deployment.status.value,
                "health_status": service_deployment.health_status,
                "replicas": config.replicas,
                "current_replicas": service_deployment.current_replicas,
                "ready_replicas": service_deployment.ready_replicas,
                "endpoints": service_deployment.endpoints,
                "start_time": service_deployment.start_time.isoformat() if service_deployment.start_time else None,
                "end_time": service_deployment.end_time.isoformat() if service_deployment.end_time else None,
                "error_message": service_deployment.error_message,
                "image": f"{config.image}:{config.version}",
                "port": config.port,
                "autoscaling_enabled": config.autoscaling.enabled,
                "service_mesh_enabled": config.service_mesh.enabled
            }
            
            # Update summary counts
            if service_deployment.status == ServiceStatus.RUNNING:
                status["summary"]["running"] += 1
            elif service_deployment.status == ServiceStatus.FAILED:
                status["summary"]["failed"] += 1
            elif service_deployment.status == ServiceStatus.DEPLOYING:
                status["summary"]["deploying"] += 1
            elif service_deployment.status == ServiceStatus.PENDING:
                status["summary"]["pending"] += 1
        
        return status

async def main():
    """Main function for testing microservices deployment"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize microservices deployment
    microservices = XORBMicroservicesDeployment(
        namespace="xorb-services",
        platform="kubernetes"
    )
    
    # Deploy microservices
    success = await microservices.deploy_microservices()
    
    if success:
        print("🎉 Microservices deployment completed successfully!")
        
        # Get status
        status = microservices.get_microservices_status()
        print(f"📊 Microservices Status:")
        print(f"  Total Services: {status['summary']['total']}")
        print(f"  Running: {status['summary']['running']}")
        print(f"  Failed: {status['summary']['failed']}")
        print(f"  Deploying: {status['summary']['deploying']}")
        
        # Print service endpoints
        print(f"\n🔗 Service Endpoints:")
        for name, service in status['services'].items():
            if service['endpoints']:
                print(f"  {name}:")
                for endpoint_type, endpoint_url in service['endpoints'].items():
                    print(f"    {endpoint_type}: {endpoint_url}")
    else:
        print("❌ Microservices deployment failed!")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)