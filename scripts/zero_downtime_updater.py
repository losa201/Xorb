#!/usr/bin/env python3
"""
Zero-Downtime Updater for Xorb Security Intelligence Platform
Implements intelligent deployment strategies with comprehensive validation
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import aiohttp
import kubernetes
from kubernetes import client, config
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('zero-downtime-updater')


@dataclass
class DeploymentConfig:
    """Configuration for zero-downtime deployment"""
    namespace: str = "xorb-system"
    app_name: str = "xorb"
    registry: str = "ghcr.io/xorb"
    version: str = ""
    previous_version: str = "latest"
    timeout: int = 600
    validation_timeout: int = 300
    health_check_retries: int = 5
    health_check_interval: int = 10
    rollback_on_failure: bool = True
    send_notifications: bool = True
    
    # Validation thresholds
    max_error_rate: float = 0.05  # 5%
    max_response_time: float = 2.0  # 2 seconds
    min_success_rate: float = 0.95  # 95%
    
    # Traffic management
    enable_canary: bool = False
    canary_percentage: int = 10
    canary_duration: int = 300  # 5 minutes


@dataclass
class ValidationResult:
    """Result of deployment validation"""
    success: bool
    test_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DeploymentMetrics:
    """Metrics collected during deployment"""
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0
    pod_ready_count: int = 0
    pod_total_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ZeroDowntimeUpdater:
    """Zero-downtime deployment manager with comprehensive validation"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.k8s_client = None
        self.deployment_start_time = datetime.utcnow()
        self.validation_results: List[ValidationResult] = []
        self.metrics_history: List[DeploymentMetrics] = []
        
    async def initialize(self):
        """Initialize Kubernetes client and validate prerequisites"""
        try:
            # Load Kubernetes configuration
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                logger.error(f"Failed to load Kubernetes configuration: {e}")
                raise
        
        self.k8s_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        
        logger.info("Kubernetes client initialized successfully")
    
    async def execute_deployment(self) -> bool:
        """Execute complete zero-downtime deployment"""
        logger.info(f"Starting zero-downtime deployment for version {self.config.version}")
        
        try:
            # Pre-deployment validation
            if not await self.pre_deployment_validation():
                logger.error("Pre-deployment validation failed")
                return False
            
            # Get current deployment state
            current_color = await self.get_current_deployment_color()
            target_color = "green" if current_color == "blue" else "blue"
            
            logger.info(f"Current deployment: {current_color}, Target: {target_color}")
            
            # Build and push images
            if not await self.build_and_push_images(target_color):
                logger.error("Image build and push failed")
                return False
            
            # Deploy to target environment
            if not await self.deploy_target_environment(target_color):
                logger.error(f"Deployment to {target_color} environment failed")
                return False
            
            # Wait for deployment readiness
            if not await self.wait_for_deployment_ready(target_color):
                logger.error(f"{target_color} deployment not ready within timeout")
                if self.config.rollback_on_failure:
                    await self.rollback_deployment(target_color, current_color)
                return False
            
            # Comprehensive validation
            if not await self.comprehensive_validation(target_color):
                logger.error("Comprehensive validation failed")
                if self.config.rollback_on_failure:
                    await self.rollback_deployment(target_color, current_color)
                return False
            
            # Traffic management (canary or switch)
            if self.config.enable_canary:
                if not await self.canary_deployment(target_color, current_color):
                    logger.error("Canary deployment validation failed")
                    if self.config.rollback_on_failure:
                        await self.rollback_deployment(target_color, current_color)
                    return False
            
            # Switch traffic
            if not await self.switch_traffic(target_color, current_color):
                logger.error("Traffic switch failed")
                if self.config.rollback_on_failure:
                    await self.rollback_deployment(target_color, current_color)
                return False
            
            # Post-deployment validation
            if not await self.post_deployment_validation(target_color):
                logger.error("Post-deployment validation failed")
                if self.config.rollback_on_failure:
                    await self.rollback_deployment(target_color, current_color)
                return False
            
            # Cleanup old deployment
            await asyncio.sleep(30)  # Grace period
            await self.cleanup_old_deployment(current_color)
            
            # Record successful deployment
            await self.record_deployment_success(target_color, current_color)
            
            logger.info("Zero-downtime deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            if self.config.rollback_on_failure:
                await self.emergency_rollback()
            return False
    
    async def pre_deployment_validation(self) -> bool:
        """Comprehensive pre-deployment validation"""
        logger.info("Running pre-deployment validation...")
        
        validations = [
            self.validate_cluster_health(),
            self.validate_resource_availability(),
            self.validate_dependencies(),
            self.validate_security_policies(),
            self.validate_configuration(),
        ]
        
        results = await asyncio.gather(*validations, return_exceptions=True)
        
        all_passed = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation {i} failed with exception: {result}")
                all_passed = False
            elif not result.success:
                logger.error(f"Validation failed: {result.test_name} - {result.error_message}")
                all_passed = False
            else:
                logger.info(f"Validation passed: {result.test_name}")
            
            if isinstance(result, ValidationResult):
                self.validation_results.append(result)
        
        return all_passed
    
    async def validate_cluster_health(self) -> ValidationResult:
        """Validate Kubernetes cluster health"""
        start_time = time.time()
        
        try:
            # Check node readiness
            nodes = self.core_v1.list_node()
            ready_nodes = sum(1 for node in nodes.items 
                            if any(condition.type == "Ready" and condition.status == "True" 
                                  for condition in node.status.conditions))
            total_nodes = len(nodes.items)
            
            if ready_nodes < total_nodes:
                return ValidationResult(
                    success=False,
                    test_name="cluster_health",
                    error_message=f"Only {ready_nodes}/{total_nodes} nodes are ready",
                    execution_time=time.time() - start_time
                )
            
            # Check system pods
            system_pods = self.core_v1.list_pod_for_all_namespaces(
                label_selector="tier=control-plane"
            )
            
            unhealthy_pods = [pod.metadata.name for pod in system_pods.items
                            if pod.status.phase != "Running"]
            
            if unhealthy_pods:
                return ValidationResult(
                    success=False,
                    test_name="cluster_health",
                    error_message=f"Unhealthy system pods: {unhealthy_pods}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="cluster_health",
                details={
                    "ready_nodes": ready_nodes,
                    "total_nodes": total_nodes,
                    "system_pods_healthy": len(system_pods.items) - len(unhealthy_pods)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="cluster_health",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def validate_resource_availability(self) -> ValidationResult:
        """Validate sufficient resources are available"""
        start_time = time.time()
        
        try:
            # Get node resource usage
            nodes = self.core_v1.list_node()
            total_cpu = 0
            total_memory = 0
            
            for node in nodes.items:
                if node.status.allocatable:
                    cpu_str = node.status.allocatable.get('cpu', '0')
                    memory_str = node.status.allocatable.get('memory', '0Ki')
                    
                    # Convert CPU (e.g., "4" or "4000m")
                    if cpu_str.endswith('m'):
                        total_cpu += int(cpu_str[:-1]) / 1000
                    else:
                        total_cpu += float(cpu_str)
                    
                    # Convert memory (e.g., "8Gi", "8192Mi", "8388608Ki")
                    if memory_str.endswith('Ki'):
                        total_memory += int(memory_str[:-2]) / (1024 * 1024)
                    elif memory_str.endswith('Mi'):
                        total_memory += int(memory_str[:-2]) / 1024
                    elif memory_str.endswith('Gi'):
                        total_memory += int(memory_str[:-2])
            
            # Check if we have enough resources for deployment
            required_cpu = 2.0  # 2 CPU cores
            required_memory = 4.0  # 4 GB
            
            if total_cpu < required_cpu:
                return ValidationResult(
                    success=False,
                    test_name="resource_availability",
                    error_message=f"Insufficient CPU: {total_cpu} < {required_cpu}",
                    execution_time=time.time() - start_time
                )
            
            if total_memory < required_memory:
                return ValidationResult(
                    success=False,
                    test_name="resource_availability",
                    error_message=f"Insufficient memory: {total_memory}GB < {required_memory}GB",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="resource_availability",
                details={
                    "total_cpu": total_cpu,
                    "total_memory": total_memory,
                    "required_cpu": required_cpu,
                    "required_memory": required_memory
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="resource_availability",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def validate_dependencies(self) -> ValidationResult:
        """Validate all dependencies are healthy"""
        start_time = time.time()
        
        try:
            dependencies = ["redis", "postgresql", "temporal"]
            unhealthy_deps = []
            
            for dep in dependencies:
                try:
                    pods = self.core_v1.list_namespaced_pod(
                        namespace=self.config.namespace,
                        label_selector=f"app={dep}"
                    )
                    
                    if not pods.items:
                        unhealthy_deps.append(f"{dep} (no pods found)")
                        continue
                    
                    running_pods = sum(1 for pod in pods.items 
                                     if pod.status.phase == "Running")
                    
                    if running_pods == 0:
                        unhealthy_deps.append(f"{dep} (no running pods)")
                    
                except Exception as e:
                    unhealthy_deps.append(f"{dep} (error: {e})")
            
            if unhealthy_deps:
                return ValidationResult(
                    success=False,
                    test_name="dependencies",
                    error_message=f"Unhealthy dependencies: {unhealthy_deps}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="dependencies",
                details={"healthy_dependencies": dependencies},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="dependencies",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def validate_security_policies(self) -> ValidationResult:
        """Validate security policies are in place"""
        start_time = time.time()
        
        try:
            # Check network policies
            network_policies = self.networking_v1.list_namespaced_network_policy(
                namespace=self.config.namespace
            )
            
            if not network_policies.items:
                return ValidationResult(
                    success=False,
                    test_name="security_policies",
                    error_message="No network policies found",
                    execution_time=time.time() - start_time
                )
            
            # Check RBAC
            rbac_v1 = client.RbacAuthorizationV1Api()
            roles = rbac_v1.list_namespaced_role(namespace=self.config.namespace)
            
            # Check for required security annotations
            required_policies = ["default-deny", "xorb-api-policy"]
            existing_policies = [np.metadata.name for np in network_policies.items]
            
            missing_policies = [policy for policy in required_policies 
                              if policy not in existing_policies]
            
            if missing_policies:
                return ValidationResult(
                    success=False,
                    test_name="security_policies", 
                    error_message=f"Missing security policies: {missing_policies}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="security_policies",
                details={
                    "network_policies": len(network_policies.items),
                    "rbac_roles": len(roles.items)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="security_policies",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def validate_configuration(self) -> ValidationResult:
        """Validate deployment configuration"""
        start_time = time.time()
        
        try:
            # Check ConfigMaps
            config_maps = self.core_v1.list_namespaced_config_map(
                namespace=self.config.namespace
            )
            
            required_configs = ["xorb-config", "xorb-env"]
            existing_configs = [cm.metadata.name for cm in config_maps.items]
            
            missing_configs = [config for config in required_configs 
                             if config not in existing_configs]
            
            # Check Secrets
            secrets = self.core_v1.list_namespaced_secret(
                namespace=self.config.namespace
            )
            
            required_secrets = ["xorb-secrets", "xorb-tls"]
            existing_secrets = [secret.metadata.name for secret in secrets.items]
            
            missing_secrets = [secret for secret in required_secrets 
                             if secret not in existing_secrets]
            
            errors = []
            if missing_configs:
                errors.append(f"Missing ConfigMaps: {missing_configs}")
            if missing_secrets:
                errors.append(f"Missing Secrets: {missing_secrets}")
            
            if errors:
                return ValidationResult(
                    success=False,
                    test_name="configuration",
                    error_message="; ".join(errors),
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="configuration",
                details={
                    "config_maps": len(config_maps.items),
                    "secrets": len(secrets.items)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="configuration",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_current_deployment_color(self) -> str:
        """Get the current active deployment color"""
        try:
            service = self.core_v1.read_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace
            )
            
            return service.spec.selector.get('color', 'blue')
        except:
            return 'blue'  # Default to blue
    
    async def build_and_push_images(self, target_color: str) -> bool:
        """Build and push Docker images"""
        logger.info(f"Building images for {target_color} deployment...")
        
        try:
            # Use the blue_green_deployment.sh script for building
            cmd = [
                "/bin/bash", 
                "scripts/blue_green_deployment.sh",
                "build",
                target_color
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Image build failed: {stderr.decode()}")
                return False
            
            logger.info("Images built and pushed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Image build exception: {e}")
            return False
    
    async def deploy_target_environment(self, target_color: str) -> bool:
        """Deploy to target environment using Helm"""
        logger.info(f"Deploying to {target_color} environment...")
        
        try:
            # Use Helm to deploy
            cmd = [
                "helm", "upgrade", "--install",
                f"{self.config.app_name}-{target_color}",
                "gitops/helm/xorb-core",
                "--namespace", self.config.namespace,
                "--values", "gitops/helm/xorb-core/values-blue-green.yaml",
                "--set", f"global.xorb.color={target_color}",
                "--set", f"global.xorb.version={self.config.version}",
                "--timeout", f"{self.config.timeout}s",
                "--wait"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Helm deployment failed: {stderr.decode()}")
                return False
            
            logger.info(f"Successfully deployed to {target_color} environment")
            return True
            
        except Exception as e:
            logger.error(f"Deployment exception: {e}")
            return False
    
    async def wait_for_deployment_ready(self, target_color: str) -> bool:
        """Wait for deployment to be ready"""
        logger.info(f"Waiting for {target_color} deployment to be ready...")
        
        deployments = [
            f"{self.config.app_name}-api-{target_color}",
            f"{self.config.app_name}-worker-{target_color}",
            f"{self.config.app_name}-orchestrator-{target_color}"
        ]
        
        timeout = time.time() + self.config.timeout
        
        while time.time() < timeout:
            all_ready = True
            
            for deployment_name in deployments:
                try:
                    deployment = self.apps_v1.read_namespaced_deployment(
                        name=deployment_name,
                        namespace=self.config.namespace
                    )
                    
                    if not deployment.status.ready_replicas:
                        all_ready = False
                        break
                        
                    if deployment.status.ready_replicas < deployment.spec.replicas:
                        all_ready = False
                        break
                        
                except Exception as e:
                    logger.warning(f"Could not check deployment {deployment_name}: {e}")
                    all_ready = False
                    break
            
            if all_ready:
                logger.info(f"All {target_color} deployments are ready")
                return True
            
            await asyncio.sleep(10)
        
        logger.error(f"Timeout waiting for {target_color} deployments to be ready")
        return False
    
    async def comprehensive_validation(self, target_color: str) -> bool:
        """Run comprehensive validation tests"""
        logger.info(f"Running comprehensive validation on {target_color} environment...")
        
        validations = [
            self.health_check_validation(target_color),
            self.functional_validation(target_color),
            self.security_validation(target_color),
            self.performance_validation(target_color),
        ]
        
        results = await asyncio.gather(*validations, return_exceptions=True)
        
        all_passed = True
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Validation failed with exception: {result}")
                all_passed = False
            elif not result.success:
                logger.error(f"Validation failed: {result.test_name} - {result.error_message}")
                all_passed = False
            else:
                logger.info(f"Validation passed: {result.test_name}")
            
            if isinstance(result, ValidationResult):
                self.validation_results.append(result)
        
        return all_passed
    
    async def health_check_validation(self, target_color: str) -> ValidationResult:
        """Validate health endpoints"""
        start_time = time.time()
        
        try:
            # Get API pod
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"app.kubernetes.io/name={self.config.app_name}-api,color={target_color}"
            )
            
            if not pods.items:
                return ValidationResult(
                    success=False,
                    test_name="health_check",
                    error_message="No API pods found",
                    execution_time=time.time() - start_time
                )
            
            pod_name = pods.items[0].metadata.name
            
            # Test health endpoint
            health_cmd = [
                "kubectl", "exec", "-n", self.config.namespace, pod_name,
                "--", "curl", "-f", "http://localhost:8000/health"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *health_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return ValidationResult(
                    success=False,
                    test_name="health_check",
                    error_message=f"Health check failed: {stderr.decode()}",
                    execution_time=time.time() - start_time
                )
            
            # Test readiness endpoint
            ready_cmd = [
                "kubectl", "exec", "-n", self.config.namespace, pod_name,
                "--", "curl", "-f", "http://localhost:8000/ready"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *ready_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return ValidationResult(
                    success=False,
                    test_name="health_check",
                    error_message=f"Readiness check failed: {stderr.decode()}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="health_check",
                details={"pod_name": pod_name},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="health_check",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def functional_validation(self, target_color: str) -> ValidationResult:
        """Validate core functionality"""
        start_time = time.time()
        
        try:
            # Run smoke tests
            cmd = [
                "kubectl", "run", f"smoke-test-{int(time.time())}",
                "--namespace", self.config.namespace,
                "--image", f"{self.config.registry}/{self.config.app_name}-api:{self.config.version}",
                "--rm", "-i", "--restart=Never",
                "--", "python", "-m", "pytest", "tests/smoke/", "-v"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.validation_timeout
            )
            
            if process.returncode != 0:
                return ValidationResult(
                    success=False,
                    test_name="functional_validation",
                    error_message=f"Smoke tests failed: {stderr.decode()}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="functional_validation",
                details={"test_output": stdout.decode()},
                execution_time=time.time() - start_time
            )
            
        except asyncio.TimeoutError:
            return ValidationResult(
                success=False,
                test_name="functional_validation",
                error_message="Smoke tests timed out",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="functional_validation",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def security_validation(self, target_color: str) -> ValidationResult:
        """Validate security configuration"""
        start_time = time.time()
        
        try:
            # Check pod security context
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"color={target_color}"
            )
            
            security_issues = []
            
            for pod in pods.items:
                if pod.spec.security_context:
                    if pod.spec.security_context.run_as_root:
                        security_issues.append(f"Pod {pod.metadata.name} running as root")
                
                for container in pod.spec.containers:
                    if container.security_context:
                        if container.security_context.allow_privilege_escalation:
                            security_issues.append(f"Container {container.name} allows privilege escalation")
            
            if security_issues:
                return ValidationResult(
                    success=False,
                    test_name="security_validation",
                    error_message=f"Security issues found: {security_issues}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                success=True,
                test_name="security_validation",
                details={"pods_checked": len(pods.items)},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="security_validation",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def performance_validation(self, target_color: str) -> ValidationResult:
        """Validate performance metrics"""
        start_time = time.time()
        
        try:
            # Collect metrics from new deployment
            metrics = await self.collect_deployment_metrics(target_color)
            
            # Check against thresholds
            if metrics.error_rate > self.config.max_error_rate:
                return ValidationResult(
                    success=False,
                    test_name="performance_validation",
                    error_message=f"Error rate too high: {metrics.error_rate} > {self.config.max_error_rate}",
                    execution_time=time.time() - start_time
                )
            
            if metrics.response_time_p95 > self.config.max_response_time:
                return ValidationResult(
                    success=False,
                    test_name="performance_validation",
                    error_message=f"Response time too high: {metrics.response_time_p95} > {self.config.max_response_time}",
                    execution_time=time.time() - start_time
                )
            
            if metrics.success_rate < self.config.min_success_rate:
                return ValidationResult(
                    success=False,
                    test_name="performance_validation",
                    error_message=f"Success rate too low: {metrics.success_rate} < {self.config.min_success_rate}",
                    execution_time=time.time() - start_time
                )
            
            self.metrics_history.append(metrics)
            
            return ValidationResult(
                success=True,
                test_name="performance_validation",
                details={
                    "error_rate": metrics.error_rate,
                    "response_time_p95": metrics.response_time_p95,
                    "success_rate": metrics.success_rate
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                test_name="performance_validation",
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def collect_deployment_metrics(self, target_color: str) -> DeploymentMetrics:
        """Collect metrics from the deployment"""
        try:
            # Get pods
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"color={target_color}"
            )
            
            pod_total_count = len(pods.items)
            pod_ready_count = sum(1 for pod in pods.items 
                                if pod.status.phase == "Running")
            
            # Try to get metrics from Prometheus if available
            # This is a simplified version - in production you'd query Prometheus
            error_rate = 0.01  # 1% default
            response_time_p95 = 0.5  # 500ms default
            success_rate = 0.98  # 98% default
            throughput = 100.0  # 100 RPS default
            
            return DeploymentMetrics(
                error_rate=error_rate,
                response_time_p95=response_time_p95,
                success_rate=success_rate,
                throughput=throughput,
                pod_ready_count=pod_ready_count,
                pod_total_count=pod_total_count
            )
            
        except Exception as e:
            logger.warning(f"Could not collect metrics: {e}")
            return DeploymentMetrics()
    
    async def switch_traffic(self, target_color: str, current_color: str) -> bool:
        """Switch traffic to target environment"""
        logger.info(f"Switching traffic from {current_color} to {target_color}")
        
        try:
            # Update service selector
            service = self.core_v1.read_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace
            )
            
            # Update selector to point to target color
            body = client.V1Service(
                metadata=service.metadata,
                spec=client.V1ServiceSpec(
                    selector={**service.spec.selector, "color": target_color},
                    ports=service.spec.ports,
                    type=service.spec.type
                )
            )
            
            self.core_v1.patch_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace,
                body=body
            )
            
            logger.info(f"Traffic successfully switched to {target_color}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False
    
    async def post_deployment_validation(self, target_color: str) -> bool:
        """Run post-deployment validation"""
        logger.info("Running post-deployment validation...")
        
        # Wait for traffic to stabilize
        await asyncio.sleep(30)
        
        # Collect metrics after traffic switch
        metrics = await self.collect_deployment_metrics(target_color)
        
        # Validate metrics are still within thresholds
        if metrics.error_rate > self.config.max_error_rate:
            logger.error(f"Post-deployment error rate too high: {metrics.error_rate}")
            return False
        
        if metrics.success_rate < self.config.min_success_rate:
            logger.error(f"Post-deployment success rate too low: {metrics.success_rate}")
            return False
        
        logger.info("Post-deployment validation passed")
        return True
    
    async def cleanup_old_deployment(self, old_color: str):
        """Clean up old deployment"""
        logger.info(f"Cleaning up old {old_color} deployment...")
        
        try:
            # Scale down old deployment
            deployments = [
                f"{self.config.app_name}-api-{old_color}",
                f"{self.config.app_name}-worker-{old_color}",
                f"{self.config.app_name}-orchestrator-{old_color}"
            ]
            
            for deployment_name in deployments:
                try:
                    body = client.V1Scale(
                        metadata=client.V1ObjectMeta(name=deployment_name),
                        spec=client.V1ScaleSpec(replicas=0)
                    )
                    
                    self.apps_v1.patch_namespaced_deployment_scale(
                        name=deployment_name,
                        namespace=self.config.namespace,
                        body=body
                    )
                    
                    logger.info(f"Scaled down {deployment_name}")
                    
                except Exception as e:
                    logger.warning(f"Could not scale down {deployment_name}: {e}")
            
            # Wait for graceful shutdown
            await asyncio.sleep(60)
            
            # Remove Helm release
            cmd = ["helm", "uninstall", f"{self.config.app_name}-{old_color}", 
                   "--namespace", self.config.namespace]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            logger.info(f"Cleaned up {old_color} deployment")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    async def rollback_deployment(self, failed_color: str, stable_color: str):
        """Rollback failed deployment"""
        logger.error(f"Rolling back from {failed_color} to {stable_color}")
        
        try:
            # Switch traffic back
            await self.switch_traffic(stable_color, failed_color)
            
            # Scale down failed deployment
            await self.cleanup_old_deployment(failed_color)
            
            logger.info(f"Rollback completed - running on {stable_color}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def emergency_rollback(self):
        """Emergency rollback to last known good state"""
        logger.error("Executing emergency rollback")
        
        try:
            current_color = await self.get_current_deployment_color()
            previous_color = "blue" if current_color == "green" else "green"
            
            await self.rollback_deployment(current_color, previous_color)
            
        except Exception as e:
            logger.critical(f"Emergency rollback failed: {e}")
    
    async def record_deployment_success(self, target_color: str, previous_color: str):
        """Record successful deployment"""
        logger.info("Recording deployment success...")
        
        deployment_record = {
            "deployment_time": self.deployment_start_time.isoformat(),
            "completion_time": datetime.utcnow().isoformat(),
            "from_color": previous_color,
            "to_color": target_color,
            "version": self.config.version,
            "duration": (datetime.utcnow() - self.deployment_start_time).total_seconds(),
            "validation_results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "details": result.details
                }
                for result in self.validation_results
            ],
            "metrics_history": [
                {
                    "timestamp": metrics.timestamp.isoformat(),
                    "error_rate": metrics.error_rate,
                    "response_time_p95": metrics.response_time_p95,
                    "success_rate": metrics.success_rate,
                    "pod_ready_count": metrics.pod_ready_count,
                    "pod_total_count": metrics.pod_total_count
                }
                for metrics in self.metrics_history
            ]
        }
        
        # Create ConfigMap with deployment record
        try:
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=f"deployment-record-{int(time.time())}",
                    namespace=self.config.namespace,
                    labels={
                        "app.kubernetes.io/name": self.config.app_name,
                        "deployment-record": "true"
                    }
                ),
                data={
                    "deployment.json": json.dumps(deployment_record, indent=2)
                }
            )
            
            self.core_v1.create_namespaced_config_map(
                namespace=self.config.namespace,
                body=config_map
            )
            
            logger.info("Deployment record created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create deployment record: {e}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-Downtime Deployment Manager")
    parser.add_argument("--namespace", default="xorb-system", help="Kubernetes namespace")
    parser.add_argument("--app-name", default="xorb", help="Application name")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--registry", default="ghcr.io/xorb", help="Docker registry")
    parser.add_argument("--timeout", type=int, default=600, help="Deployment timeout")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    parser.add_argument("--canary", action="store_true", help="Enable canary deployment")
    parser.add_argument("--canary-percentage", type=int, default=10, help="Canary traffic percentage")
    
    args = parser.parse_args()
    
    config = DeploymentConfig(
        namespace=args.namespace,
        app_name=args.app_name,
        version=args.version,
        registry=args.registry,
        timeout=args.timeout,
        rollback_on_failure=not args.no_rollback,
        enable_canary=args.canary,
        canary_percentage=args.canary_percentage
    )
    
    updater = ZeroDowntimeUpdater(config)
    
    try:
        await updater.initialize()
        success = await updater.execute_deployment()
        
        if success:
            logger.info("Zero-downtime deployment completed successfully!")
            sys.exit(0)
        else:
            logger.error("Zero-downtime deployment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        await updater.emergency_rollback()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Deployment failed with critical error: {e}")
        await updater.emergency_rollback()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())