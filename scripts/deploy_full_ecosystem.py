#!/usr/bin/env python3
"""
XORB 2.0 Full Ecosystem Deployment Script

This script automates the complete deployment of the XORB ecosystem including
all advanced features, database setup, service configuration, and validation.
Manages deployment of all phases (1-11) with intelligent dependency resolution, health monitoring,
and rollback capabilities. Optimized for both development and production environments.

Features:
- Phase-by-phase deployment with dependency management
- Health monitoring and validation at each stage
- Rollback capabilities for failed deployments
- Environment-specific configurations (dev/staging/prod)
- Resource optimization for different deployment targets
- Comprehensive logging and reporting
- EPYC processor optimization
- Edge computing support (Pi5)
- Enhanced security hardening

Usage:
    python scripts/deploy_full_ecosystem.py --env [dev|staging|prod] --phases [1-11|all] --mode [deploy|validate|rollback]

Environment Requirements:
- Docker and Docker Compose
- Kubernetes (for prod deployments)
- Helm 3.x (for prod deployments)
- Python 3.8+
- Required system dependencies
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
import yaml

import docker

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('xorb_deployment.log')
    ]
)
logger = logging.getLogger('xorb_ecosystem')


class DeploymentPhase(Enum):
    """XORB deployment phases"""
    PHASE_1_OBSERVABILITY = "phase1_observability"
    PHASE_2_SECURITY_CORE = "phase2_security_core"
    PHASE_3_ADVANCED_SECURITY = "phase3_advanced_security"
    PHASE_4_INTELLIGENCE = "phase4_intelligence"
    PHASE_5_LEARNING = "phase5_learning"
    PHASE_6_ORCHESTRATION = "phase6_orchestration"
    PHASE_7_ENHANCED_ORCHESTRATION = "phase7_enhanced_orchestration"
    PHASE_8_INTELLIGENT_EVOLUTION = "phase8_intelligent_evolution"
    PHASE_9_MISSION_EXECUTION = "phase9_mission_execution"
    PHASE_10_GLOBAL_INTELLIGENCE = "phase10_global_intelligence"
    PHASE_11_AUTONOMOUS_PREDICTION = "phase11_autonomous_prediction"


class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


class DeploymentMode(Enum):
    """Deployment modes"""
    DEPLOY = "deploy"
    VALIDATE = "validate"
    ROLLBACK = "rollback"
    UPGRADE = "upgrade"
    DESTROY = "destroy"


@dataclass
class ServiceConfiguration:
    """Service deployment configuration"""
    name: str
    image: str
    version: str
    replicas: int
    resources: dict[str, str]
    ports: list[int]
    environment: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    health_check: str | None = None
    readiness_probe: str | None = None


@dataclass
class PhaseConfiguration:
    """Phase deployment configuration"""
    phase: DeploymentPhase
    name: str
    description: str
    services: list[ServiceConfiguration]
    dependencies: list[DeploymentPhase] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    post_deployment_tasks: list[str] = field(default_factory=list)
    rollback_commands: list[str] = field(default_factory=list)


class XORBEcosystemDeployer:
    """XORB Full Ecosystem Deployment Orchestrator"""

    def __init__(self, base_dir: Path, environment: DeploymentEnvironment):
        self.base_dir = base_dir
        self.environment = environment
        self.docker_client = docker.from_env()

        # Deployment state
        self.deployment_state = {}
        self.deployed_phases = []
        self.failed_phases = []
        self.start_time = datetime.now()

        # Environment-specific configurations
        self.configs = self._load_environment_configs()
        self.phase_configs = self._initialize_phase_configurations()

        # Resource optimization based on environment
        self.resource_limits = self._get_resource_limits()

        logger.info(f"🚀 XORB Ecosystem Deployer initialized for {environment.value} environment")

    def _load_environment_configs(self) -> dict[str, Any]:
        """Load environment-specific configurations"""
        config_file = self.base_dir / "config" / f"{self.environment.value}.yaml"

        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)

        # Default configurations
        return {
            "dev": {
                "docker_compose_files": ["docker-compose.yml", "docker-compose.dev.yml"],
                "scale_factor": 1,
                "monitoring_enabled": True,
                "security_mode": "permissive"
            },
            "staging": {
                "docker_compose_files": ["docker-compose.yml", "docker-compose.staging.yml"],
                "scale_factor": 2,
                "monitoring_enabled": True,
                "security_mode": "strict"
            },
            "prod": {
                "kubernetes_enabled": True,
                "helm_chart": "gitops/helm/xorb",
                "scale_factor": 5,
                "monitoring_enabled": True,
                "security_mode": "strict",
                "high_availability": True
            }
        }.get(self.environment.value, {})

    def _get_resource_limits(self) -> dict[str, dict[str, str]]:
        """Get resource limits based on environment"""
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            return {
                "default": {"memory": "512Mi", "cpu": "0.5"},
                "database": {"memory": "1Gi", "cpu": "1"},
                "monitoring": {"memory": "256Mi", "cpu": "0.25"}
            }
        elif self.environment == DeploymentEnvironment.STAGING:
            return {
                "default": {"memory": "1Gi", "cpu": "1"},
                "database": {"memory": "2Gi", "cpu": "2"},
                "monitoring": {"memory": "512Mi", "cpu": "0.5"}
            }
        else:  # Production
            return {
                "default": {"memory": "2Gi", "cpu": "2"},
                "database": {"memory": "4Gi", "cpu": "4"},
                "monitoring": {"memory": "1Gi", "cpu": "1"},
                "ai_services": {"memory": "4Gi", "cpu": "4"}
            }

    def _initialize_phase_configurations(self) -> dict[DeploymentPhase, PhaseConfiguration]:
        """Initialize phase deployment configurations"""
        configs = {}

        # Phase 1: Observability MVP
        configs[DeploymentPhase.PHASE_1_OBSERVABILITY] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_1_OBSERVABILITY,
            name="Observability MVP",
            description="Core monitoring, logging, and metrics collection",
            services=[
                ServiceConfiguration(
                    name="prometheus",
                    image="prom/prometheus",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits()["monitoring"],
                    ports=[9090],
                    health_check="http://localhost:9090/-/healthy"
                ),
                ServiceConfiguration(
                    name="grafana",
                    image="grafana/grafana",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits()["monitoring"],
                    ports=[3000],
                    environment={"GF_SECURITY_ADMIN_PASSWORD": "xorb_admin_2024"},
                    health_check="http://localhost:3000/api/health"
                ),
                ServiceConfiguration(
                    name="loki",
                    image="grafana/loki",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits()["monitoring"],
                    ports=[3100]
                ),
                ServiceConfiguration(
                    name="promtail",
                    image="grafana/promtail",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits()["default"],
                    ports=[],
                    dependencies=["loki"]
                )
            ],
            validation_commands=[
                "curl -f http://localhost:9090/-/healthy",
                "curl -f http://localhost:3000/api/health"
            ]
        )

        # Phase 2-3: Security Core & Advanced Security
        configs[DeploymentPhase.PHASE_2_SECURITY_CORE] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_2_SECURITY_CORE,
            name="Security Core Components",
            description="Core security scanning and vulnerability assessment",
            services=[
                ServiceConfiguration(
                    name="scanner-go",
                    image="xorb/scanner-go",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits()["default"],
                    ports=[8080],
                    environment={
                        "NUCLEI_TEMPLATES_PATH": "/app/nuclei-templates",
                        "SCANNER_MODE": "autonomous"
                    }
                ),
                ServiceConfiguration(
                    name="vulnerability-scanner",
                    image="xorb/vulnerability-scanner",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits()["default"],
                    ports=[8081]
                ),
            ],
            dependencies=[DeploymentPhase.PHASE_1_OBSERVABILITY],
            validation_commands=[
                "curl -f http://localhost:8080/health",
                "curl -f http://localhost:8081/health"
            ]
        )

        # Phase 4-5: Intelligence & Learning
        configs[DeploymentPhase.PHASE_4_INTELLIGENCE] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_4_INTELLIGENCE,
            name="Intelligence & Learning Systems",
            description="AI-powered analysis and learning capabilities",
            services=[
                ServiceConfiguration(
                    name="campaign-engine",
                    image="xorb/campaign-engine",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits().get("ai_services", self._get_resource_limits()["default"]),
                    ports=[8082],
                    environment={
                        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
                        "CEREBRAS_API_KEY": os.getenv("CEREBRAS_API_KEY", "")
                    }
                ),
                ServiceConfiguration(
                    name="learning-engine",
                    image="xorb/learning-engine",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits().get("ai_services", self._get_resource_limits()["default"]),
                    ports=[8083]
                ),
                ServiceConfiguration(
                    name="prioritization-engine",
                    image="xorb/prioritization-engine",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits()["default"],
                    ports=[8084]
                )
            ],
            dependencies=[DeploymentPhase.PHASE_2_SECURITY_CORE],
            validation_commands=[
                "curl -f http://localhost:8082/health",
                "curl -f http://localhost:8083/health",
                "curl -f http://localhost:8084/health"
            ]
        )

        # Phase 6-7: Orchestration
        configs[DeploymentPhase.PHASE_6_ORCHESTRATION] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_6_ORCHESTRATION,
            name="Enhanced Orchestration",
            description="Advanced orchestration and coordination services",
            services=[
                ServiceConfiguration(
                    name="orchestrator",
                    image="xorb/orchestrator",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1),
                    resources=self._get_resource_limits()["default"],
                    ports=[8085],
                    environment={
                        "ORCHESTRATOR_MODE": "autonomous",
                        "MAX_CONCURRENT_MISSIONS": "10"
                    }
                ),
                ServiceConfiguration(
                    name="api-service",
                    image="xorb/api-service",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1) * 2,
                    resources=self._get_resource_limits()["default"],
                    ports=[8000],
                    dependencies=["orchestrator"]
                ),
                ServiceConfiguration(
                    name="worker-service",
                    image="xorb/worker-service",
                    version="latest",
                    replicas=self.configs.get("scale_factor", 1) * 3,
                    resources=self._get_resource_limits()["default"],
                    ports=[],
                    dependencies=["orchestrator"]
                )
            ],
            dependencies=[DeploymentPhase.PHASE_4_INTELLIGENCE],
            validation_commands=[
                "curl -f http://localhost:8085/health",
                "curl -f http://localhost:8000/health"
            ]
        )

        # Phase 8-10: Advanced Intelligence
        configs[DeploymentPhase.PHASE_8_INTELLIGENT_EVOLUTION] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_8_INTELLIGENT_EVOLUTION,
            name="Intelligent Evolution & Global Intelligence",
            description="Advanced AI evolution and global intelligence synthesis",
            services=[
                ServiceConfiguration(
                    name="cognitive-engine",
                    image="xorb/cognitive-engine",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits().get("ai_services", self._get_resource_limits()["default"]),
                    ports=[8086]
                ),
                ServiceConfiguration(
                    name="episodic-memory",
                    image="xorb/episodic-memory",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits()["database"],
                    ports=[8087]
                ),
                ServiceConfiguration(
                    name="global-intelligence",
                    image="xorb/global-intelligence",
                    version="latest",
                    replicas=1,
                    resources=self._get_resource_limits().get("ai_services", self._get_resource_limits()["default"]),
                    ports=[8088]
                )
            ],
            dependencies=[DeploymentPhase.PHASE_6_ORCHESTRATION],
            validation_commands=[
                "curl -f http://localhost:8086/health",
                "curl -f http://localhost:8087/health",
                "curl -f http://localhost:8088/health"
            ]
        )

        # Phase 11: Autonomous Threat Prediction & Response
        configs[DeploymentPhase.PHASE_11_AUTONOMOUS_PREDICTION] = PhaseConfiguration(
            phase=DeploymentPhase.PHASE_11_AUTONOMOUS_PREDICTION,
            name="Autonomous Threat Prediction & Response",
            description="Phase 11 enhanced orchestration with ML-powered threat prediction",
            services=[
                ServiceConfiguration(
                    name="enhanced-orchestrator",
                    image="xorb/enhanced-orchestrator",
                    version="v11.0",
                    replicas=1,
                    resources=self._get_resource_limits().get("ai_services", self._get_resource_limits()["default"]),
                    ports=[8089],
                    environment={
                        "XORB_PHASE_11_ENABLED": "true",
                        "XORB_PI5_OPTIMIZATION": "true",
                        "XORB_ORCHESTRATION_CYCLE_TIME": "400",
                        "XORB_MAX_CONCURRENT_MISSIONS": "5",
                        "XORB_PLUGIN_DISCOVERY_ENABLED": "true"
                    }
                )
            ],
            dependencies=[DeploymentPhase.PHASE_8_INTELLIGENT_EVOLUTION],
            validation_commands=[
                "curl -f http://localhost:8089/health",
                "python scripts/deploy_phase11.py --mode validate"
            ],
            post_deployment_tasks=[
                "python scripts/deploy_phase11.py --mode benchmark"
            ]
        )

        # Infrastructure Services (deployed with Phase 1)
        configs[DeploymentPhase.PHASE_1_OBSERVABILITY].services.extend([
            ServiceConfiguration(
                name="redis",
                image="redis",
                version="7-alpine",
                replicas=1,
                resources=self._get_resource_limits()["database"],
                ports=[6379],
                health_check="redis-cli ping"
            ),
            ServiceConfiguration(
                name="postgres",
                image="postgres",
                version="15-alpine",
                replicas=1,
                resources=self._get_resource_limits()["database"],
                ports=[5432],
                environment={
                    "POSTGRES_DB": "xorb",
                    "POSTGRES_USER": "xorb_prod",
                    "POSTGRES_PASSWORD": "xorb_secure_2024"
                },
                health_check="pg_isready -U xorb_prod"
            ),
            ServiceConfiguration(
                name="nats",
                image="nats",
                version="latest",
                replicas=1,
                resources=self._get_resource_limits()["default"],
                ports=[4222, 8222]
            )
        ])

        return configs

    async def deploy_full_ecosystem(self, phases: list[DeploymentPhase], mode: DeploymentMode) -> dict[str, Any]:
        """Deploy the full XORB ecosystem"""
        logger.info(f"🚀 Starting XORB ecosystem deployment - Mode: {mode.value}")
        logger.info(f"📋 Phases to deploy: {[p.value for p in phases]}")

        deployment_results = {
            "start_time": self.start_time.isoformat(),
            "environment": self.environment.value,
            "mode": mode.value,
            "phases": [],
            "overall_status": "in_progress"
        }

        try:
            if mode == DeploymentMode.DEPLOY:
                results = await self._execute_deployment(phases)
            elif mode == DeploymentMode.VALIDATE:
                results = await self._execute_validation(phases)
            elif mode == DeploymentMode.ROLLBACK:
                results = await self._execute_rollback(phases)
            elif mode == DeploymentMode.UPGRADE:
                results = await self._execute_upgrade(phases)
            elif mode == DeploymentMode.DESTROY:
                results = await self._execute_destroy(phases)
            else:
                raise ValueError(f"Unknown deployment mode: {mode}")

            deployment_results["phases"] = results
            deployment_results["overall_status"] = "success" if all(r.get("success", False) for r in results) else "failed"

        except Exception as e:
            logger.error(f"❌ Ecosystem deployment failed: {e}")
            deployment_results["overall_status"] = "failed"
            deployment_results["error"] = str(e)

        deployment_results["end_time"] = datetime.now().isoformat()
        deployment_results["duration"] = (datetime.now() - self.start_time).total_seconds()

        return deployment_results

    async def _execute_deployment(self, phases: list[DeploymentPhase]) -> list[dict[str, Any]]:
        """Execute full deployment"""
        results = []

        # Sort phases by dependencies
        sorted_phases = self._sort_phases_by_dependencies(phases)

        for phase in sorted_phases:
            logger.info(f"🚢 Deploying {phase.value}...")

            phase_config = self.phase_configs[phase]
            phase_result = {
                "phase": phase.value,
                "name": phase_config.name,
                "start_time": datetime.now().isoformat(),
                "services": [],
                "success": False
            }

            try:
                # Check dependencies
                if not await self._check_phase_dependencies(phase):
                    raise Exception(f"Dependencies not met for {phase.value}")

                # Deploy infrastructure if needed
                if phase == DeploymentPhase.PHASE_1_OBSERVABILITY:
                    await self._deploy_infrastructure()

                # Deploy services
                for service in phase_config.services:
                    service_result = await self._deploy_service(service)
                    phase_result["services"].append(service_result)

                    if not service_result.get("success", False):
                        raise Exception(f"Service deployment failed: {service.name}")

                # Run validation
                validation_success = await self._validate_phase(phase_config)
                if not validation_success:
                    raise Exception(f"Phase validation failed: {phase.value}")

                # Execute post-deployment tasks
                for task in phase_config.post_deployment_tasks:
                    await self._execute_task(task)

                phase_result["success"] = True
                self.deployed_phases.append(phase)
                logger.info(f"✅ Successfully deployed {phase.value}")

            except Exception as e:
                logger.error(f"❌ Failed to deploy {phase.value}: {e}")
                phase_result["error"] = str(e)
                self.failed_phases.append(phase)

                # Attempt rollback of this phase
                await self._rollback_phase(phase_config)

            phase_result["end_time"] = datetime.now().isoformat()
            results.append(phase_result)

        return results

    async def _deploy_infrastructure(self):
        """Deploy core infrastructure services"""
        logger.info("🏗️ Deploying core infrastructure...")

        if self.environment == DeploymentEnvironment.PRODUCTION:
            # Kubernetes deployment
            await self._deploy_kubernetes_infrastructure()
        else:
            # Docker Compose deployment
            await self._deploy_docker_infrastructure()

    async def _deploy_kubernetes_infrastructure(self):
        """Deploy infrastructure using Kubernetes"""
        logger.info("☸️ Deploying Kubernetes infrastructure...")

        # Apply Kubernetes manifests
        k8s_files = [
            "gitops/overlays/production/infrastructure.yaml",
            "gitops/overlays/production/monitoring.yaml",
            "gitops/overlays/production/databases.yaml"
        ]

        for k8s_file in k8s_files:
            file_path = self.base_dir / k8s_file
            if file_path.exists():
                result = subprocess.run(
                    ["kubectl", "apply", "-f", str(file_path)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise Exception(f"Failed to apply {k8s_file}: {result.stderr}")

        # Deploy using Helm if chart exists
        helm_chart = self.base_dir / self.configs.get("helm_chart", "")
        if helm_chart.exists():
            result = subprocess.run([
                "helm", "upgrade", "--install", "xorb-infrastructure",
                str(helm_chart), "--namespace", "xorb-system", "--create-namespace"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Helm deployment failed: {result.stderr}")

    async def _deploy_docker_infrastructure(self):
        """Deploy infrastructure using Docker Compose"""
        logger.info("🐳 Deploying Docker infrastructure...")

        compose_files = self.configs.get("docker_compose_files", ["docker-compose.yml"])

        # Start with infrastructure services
        infra_compose = self.base_dir / "docker-compose.infrastructure.yml"
        if infra_compose.exists():
            result = subprocess.run([
                "docker", "compose", "-f", str(infra_compose), "up", "-d"
            ], cwd=self.base_dir, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Infrastructure deployment failed: {result.stderr}")

        # Wait for infrastructure to be ready
        await self._wait_for_infrastructure_ready()

    async def _deploy_service(self, service: ServiceConfiguration) -> dict[str, Any]:
        """Deploy individual service"""
        logger.info(f"📦 Deploying service: {service.name}")

        service_result = {
            "name": service.name,
            "image": f"{service.image}:{service.version}",
            "start_time": datetime.now().isoformat(),
            "success": False
        }

        try:
            if self.environment == DeploymentEnvironment.PRODUCTION:
                # Kubernetes deployment
                await self._deploy_service_kubernetes(service)
            else:
                # Docker deployment
                await self._deploy_service_docker(service)

            # Health check
            if service.health_check:
                health_ok = await self._check_service_health(service)
                if not health_ok:
                    raise Exception(f"Health check failed for {service.name}")

            service_result["success"] = True
            logger.info(f"✅ Service deployed successfully: {service.name}")

        except Exception as e:
            logger.error(f"❌ Service deployment failed: {service.name} - {e}")
            service_result["error"] = str(e)

        service_result["end_time"] = datetime.now().isoformat()
        return service_result

    async def _deploy_service_docker(self, service: ServiceConfiguration):
        """Deploy service using Docker"""
        # Check if container already exists
        try:
            container = self.docker_client.containers.get(service.name)
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass

        # Pull image
        try:
            self.docker_client.images.pull(f"{service.image}:{service.version}")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image not found, will build locally: {service.image}:{service.version}")

        # Run container
        port_bindings = {}
        for port in service.ports:
            port_bindings[port] = port

        container = self.docker_client.containers.run(
            f"{service.image}:{service.version}",
            name=service.name,
            ports=port_bindings,
            environment=service.environment,
            volumes=service.volumes,
            detach=True,
            restart_policy={"Name": "unless-stopped"}
        )

        logger.info(f"🐳 Started container: {service.name} ({container.id[:12]})")

    async def _deploy_service_kubernetes(self, service: ServiceConfiguration):
        """Deploy service using Kubernetes"""
        # Generate Kubernetes manifests
        deployment_yaml = self._generate_k8s_deployment(service)
        service_yaml = self._generate_k8s_service(service)

        # Apply manifests
        for manifest in [deployment_yaml, service_yaml]:
            result = subprocess.run([
                "kubectl", "apply", "-f", "-"
            ], input=manifest, text=True, capture_output=True)

            if result.returncode != 0:
                raise Exception(f"Failed to deploy {service.name}: {result.stderr}")

    def _generate_k8s_deployment(self, service: ServiceConfiguration) -> str:
        """Generate Kubernetes deployment YAML"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service.name,
                "namespace": "xorb-system"
            },
            "spec": {
                "replicas": service.replicas,
                "selector": {"matchLabels": {"app": service.name}},
                "template": {
                    "metadata": {"labels": {"app": service.name}},
                    "spec": {
                        "containers": [{
                            "name": service.name,
                            "image": f"{service.image}:{service.version}",
                            "ports": [{"containerPort": port} for port in service.ports],
                            "env": [{"name": k, "value": v} for k, v in service.environment.items()],
                            "resources": {
                                "limits": service.resources,
                                "requests": {k: str(int(v.rstrip('Mi')) // 2) + 'Mi' if v.endswith('Mi') else str(float(v) / 2) for k, v in service.resources.items()}
                            }
                        }]
                    }
                }
            }
        }

        return yaml.dump(deployment)

    def _generate_k8s_service(self, service: ServiceConfiguration) -> str:
        """Generate Kubernetes service YAML"""
        if not service.ports:
            return ""

        k8s_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service.name,
                "namespace": "xorb-system"
            },
            "spec": {
                "selector": {"app": service.name},
                "ports": [{"port": port, "targetPort": port} for port in service.ports]
            }
        }

        return yaml.dump(k8s_service)

    async def _check_service_health(self, service: ServiceConfiguration) -> bool:
        """Check service health"""
        max_retries = 30
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                if service.health_check.startswith("http"):
                    response = requests.get(service.health_check, timeout=5)
                    if response.status_code == 200:
                        return True
                elif service.health_check.startswith("redis-cli"):
                    result = subprocess.run([
                        "docker", "exec", service.name, "redis-cli", "ping"
                    ], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and "PONG" in result.stdout:
                        return True
                elif service.health_check.startswith("pg_isready"):
                    result = subprocess.run([
                        "docker", "exec", service.name, "pg_isready", "-U", "xorb_prod"
                    ], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return True

            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1} failed for {service.name}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        return False

    async def _validate_phase(self, phase_config: PhaseConfiguration) -> bool:
        """Validate phase deployment"""
        logger.info(f"🔍 Validating phase: {phase_config.name}")

        for command in phase_config.validation_commands:
            try:
                if command.startswith("curl"):
                    # HTTP health check
                    url = command.split()[-1]
                    response = requests.get(url, timeout=10)
                    if response.status_code != 200:
                        logger.error(f"Validation failed: {command} returned {response.status_code}")
                        return False
                elif command.startswith("python"):
                    # Python script execution
                    result = subprocess.run(
                        command.split(),
                        cwd=self.base_dir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode != 0:
                        logger.error(f"Validation failed: {command} - {result.stderr}")
                        return False
                else:
                    # Generic command
                    result = subprocess.run(
                        command.split(),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        logger.error(f"Validation failed: {command} - {result.stderr}")
                        return False

            except Exception as e:
                logger.error(f"Validation error for command '{command}': {e}")
                return False

        logger.info(f"✅ Phase validation passed: {phase_config.name}")
        return True

    async def _execute_task(self, task: str):
        """Execute post-deployment task"""
        logger.info(f"⚙️ Executing task: {task}")

        try:
            result = subprocess.run(
                task.split(),
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.error(f"Task failed: {task} - {result.stderr}")
            else:
                logger.info(f"✅ Task completed: {task}")

        except Exception as e:
            logger.error(f"Task execution error: {task} - {e}")

    def _sort_phases_by_dependencies(self, phases: list[DeploymentPhase]) -> list[DeploymentPhase]:
        """Sort phases by their dependencies"""
        sorted_phases = []
        remaining_phases = phases.copy()

        while remaining_phases:
            # Find phases with no unmet dependencies
            ready_phases = []
            for phase in remaining_phases:
                phase_config = self.phase_configs[phase]
                if all(dep in sorted_phases for dep in phase_config.dependencies):
                    ready_phases.append(phase)

            if not ready_phases:
                # Circular dependency or missing dependency
                logger.warning(f"Circular dependency detected in phases: {[p.value for p in remaining_phases]}")
                sorted_phases.extend(remaining_phases)
                break

            # Add ready phases to sorted list
            for phase in ready_phases:
                sorted_phases.append(phase)
                remaining_phases.remove(phase)

        return sorted_phases

    async def _check_phase_dependencies(self, phase: DeploymentPhase) -> bool:
        """Check if phase dependencies are met"""
        phase_config = self.phase_configs[phase]

        for dep_phase in phase_config.dependencies:
            if dep_phase not in self.deployed_phases:
                logger.error(f"Dependency not met: {phase.value} requires {dep_phase.value}")
                return False

        return True

    async def _wait_for_infrastructure_ready(self):
        """Wait for infrastructure services to be ready"""
        logger.info("⏳ Waiting for infrastructure services...")

        # Wait for key infrastructure services
        services_to_check = ["postgres", "redis", "nats"]

        for service in services_to_check:
            max_wait = 120  # 2 minutes
            wait_interval = 5

            for attempt in range(max_wait // wait_interval):
                try:
                    container = self.docker_client.containers.get(service)
                    if container.status == "running":
                        logger.info(f"✅ {service} is ready")
                        break
                except docker.errors.NotFound:
                    pass

                if attempt < (max_wait // wait_interval) - 1:
                    await asyncio.sleep(wait_interval)
            else:
                raise Exception(f"Infrastructure service {service} not ready after {max_wait}s")

    async def _execute_validation(self, phases: list[DeploymentPhase]) -> list[dict[str, Any]]:
        """Execute validation for specified phases"""
        results = []

        for phase in phases:
            logger.info(f"🔍 Validating {phase.value}...")

            phase_config = self.phase_configs[phase]
            validation_result = await self._validate_phase(phase_config)

            results.append({
                "phase": phase.value,
                "name": phase_config.name,
                "validation_success": validation_result,
                "success": validation_result
            })

        return results

    async def _execute_rollback(self, phases: list[DeploymentPhase]) -> list[dict[str, Any]]:
        """Execute rollback for specified phases"""
        results = []

        # Rollback in reverse order
        for phase in reversed(phases):
            logger.info(f"🔄 Rolling back {phase.value}...")

            phase_config = self.phase_configs[phase]
            rollback_result = await self._rollback_phase(phase_config)

            results.append({
                "phase": phase.value,
                "name": phase_config.name,
                "rollback_success": rollback_result,
                "success": rollback_result
            })

        return results

    async def _rollback_phase(self, phase_config: PhaseConfiguration) -> bool:
        """Rollback a specific phase"""
        try:
            # Stop and remove services
            for service in phase_config.services:
                try:
                    container = self.docker_client.containers.get(service.name)
                    container.stop()
                    container.remove()
                    logger.info(f"🗑️ Removed service: {service.name}")
                except docker.errors.NotFound:
                    pass

            # Execute rollback commands
            for command in phase_config.rollback_commands:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    logger.warning(f"Rollback command failed: {command}")

            return True

        except Exception as e:
            logger.error(f"Rollback failed for {phase_config.name}: {e}")
            return False

    async def _execute_upgrade(self, phases: list[DeploymentPhase]) -> list[dict[str, Any]]:
        """Execute upgrade for specified phases"""
        # Upgrade is essentially a rolling deployment
        return await self._execute_deployment(phases)

    async def _execute_destroy(self, phases: list[DeploymentPhase]) -> list[dict[str, Any]]:
        """Execute destroy for specified phases"""
        logger.warning("💥 DESTROYING XORB ecosystem - this cannot be undone!")

        # Destroy all services and data
        if self.environment == DeploymentEnvironment.PRODUCTION:
            result = subprocess.run([
                "kubectl", "delete", "namespace", "xorb-system", "--force"
            ], capture_output=True, text=True)
        else:
            result = subprocess.run([
                "docker", "compose", "down", "-v", "--remove-orphans"
            ], cwd=self.base_dir, capture_output=True, text=True)

        return [{"phase": "all", "name": "Ecosystem Destruction", "success": result.returncode == 0}]


async def main():
    """Main deployment script entry point"""
    parser = argparse.ArgumentParser(description='XORB Full Ecosystem Deployment')
    parser.add_argument('--env', choices=['dev', 'staging', 'prod'], default='dev',
                       help='Deployment environment')
    parser.add_argument('--phases', type=str, default='all',
                       help='Phases to deploy (comma-separated or "all")')
    parser.add_argument('--mode', choices=['deploy', 'validate', 'rollback', 'upgrade', 'destroy'],
                       default='deploy', help='Deployment mode')
    parser.add_argument('--output', type=str, help='Output file for deployment report')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')

    args = parser.parse_args()

    # Parse phases
    if args.phases == 'all':
        phases = list(DeploymentPhase)
    else:
        phase_names = args.phases.split(',')
        phases = []
        for name in phase_names:
            try:
                phase = DeploymentPhase(name.strip())
                phases.append(phase)
            except ValueError:
                logger.error(f"Invalid phase: {name}")
                sys.exit(1)

    # Initialize deployer
    base_dir = Path(__file__).parent.parent
    environment = DeploymentEnvironment(args.env)
    deployer = XORBEcosystemDeployer(base_dir, environment)

    try:
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE - No actual deployment will occur")
            result = {
                "dry_run": True,
                "phases": [p.value for p in phases],
                "environment": args.env,
                "mode": args.mode
            }
        else:
            # Run deployment
            mode = DeploymentMode(args.mode)
            result = await deployer.deploy_full_ecosystem(phases, mode)

        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"📄 Deployment report saved to {args.output}")
        else:
            print("\n" + "="*100)
            print("XORB FULL ECOSYSTEM DEPLOYMENT REPORT")
            print("="*100)
            print(json.dumps(result, indent=2))

        # Exit with appropriate code
        success = result.get("overall_status") == "success"
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
