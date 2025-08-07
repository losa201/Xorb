#!/usr/bin/env python3
"""
XORB Deployment Orchestrator
Enterprise-grade deployment management system with real-time monitoring,
automated rollbacks, and comprehensive operational intelligence.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
import psutil
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xorb-orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    PREPARATION = "preparation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETION = "completion"
    ROLLBACK = "rollback"

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class DeploymentConfig:
    environment: str
    namespace: str
    replicas: Dict[str, int]
    resources: Dict[str, Dict[str, str]]
    features: List[str]
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    security_scanning: bool = True
    
@dataclass
class ServiceHealth:
    name: str
    status: ServiceStatus
    replicas: Tuple[int, int]  # (ready, desired)
    cpu_usage: float
    memory_usage: float
    last_check: datetime
    errors: List[str]

@dataclass
class DeploymentState:
    deployment_id: str
    phase: DeploymentPhase
    start_time: datetime
    services: Dict[str, ServiceHealth]
    metrics: Dict[str, float]
    logs: List[str]
    rollback_point: Optional[str] = None

class XORBDeploymentOrchestrator:
    def __init__(self, config_path: str = "/root/Xorb/config/deployment-config.yaml"):
        self.config_path = config_path
        self.deployment_state: Optional[DeploymentState] = None
        self.config: Optional[DeploymentConfig] = None
        self.scripts_dir = Path("/root/Xorb/scripts")
        self.monitoring_session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the orchestrator with configuration and monitoring."""
        logger.info("Initializing XORB Deployment Orchestrator")
        
        # Load configuration
        await self._load_config()
        
        # Initialize monitoring
        self.monitoring_session = aiohttp.ClientSession()
        
        # Validate environment
        await self._validate_environment()
        
        logger.info("Orchestrator initialization complete")

    async def _load_config(self):
        """Load deployment configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self.config = DeploymentConfig(**config_data)
            else:
                # Default configuration
                self.config = DeploymentConfig(
                    environment="production",
                    namespace="xorb-system",
                    replicas={
                        "orchestrator": 3,
                        "redis": 3,
                        "postgres": 1,
                        "monitoring": 2
                    },
                    resources={
                        "orchestrator": {"cpu": "2000m", "memory": "4Gi"},
                        "redis": {"cpu": "1000m", "memory": "2Gi"},
                        "postgres": {"cpu": "1000m", "memory": "2Gi"}
                    },
                    features=[
                        "predictive_load_balancing",
                        "fault_tolerance",
                        "behavioral_intelligence",
                        "distributed_consensus",
                        "security_compliance"
                    ]
                )
                await self._save_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def _save_config(self):
        """Save current configuration."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(asdict(self.config), f, default_flow_style=False)

    async def _validate_environment(self):
        """Validate deployment environment prerequisites."""
        logger.info("Validating deployment environment")
        
        # Check Kubernetes connectivity
        result = await self._run_command("kubectl cluster-info")
        if result.returncode != 0:
            raise RuntimeError("Kubernetes cluster not accessible")
        
        # Check required tools
        tools = ["helm", "docker", "jq"]
        for tool in tools:
            result = await self._run_command(f"which {tool}")
            if result.returncode != 0:
                raise RuntimeError(f"Required tool not found: {tool}")
        
        # Check namespace
        result = await self._run_command(f"kubectl get namespace {self.config.namespace}")
        if result.returncode != 0:
            logger.info(f"Creating namespace {self.config.namespace}")
            await self._run_command(f"kubectl create namespace {self.config.namespace}")

    async def deploy(self, deployment_id: str = None) -> str:
        """Execute complete deployment with monitoring and rollback capability."""
        if not deployment_id:
            deployment_id = f"xorb-deploy-{int(time.time())}"
        
        logger.info(f"Starting deployment {deployment_id}")
        
        # Initialize deployment state
        self.deployment_state = DeploymentState(
            deployment_id=deployment_id,
            phase=DeploymentPhase.PREPARATION,
            start_time=datetime.utcnow(),
            services={},
            metrics={},
            logs=[]
        )
        
        try:
            # Phase 1: Preparation
            await self._execute_phase(DeploymentPhase.PREPARATION, self._prepare_deployment)
            
            # Phase 2: Validation
            await self._execute_phase(DeploymentPhase.VALIDATION, self._validate_deployment)
            
            # Phase 3: Deployment
            await self._execute_phase(DeploymentPhase.DEPLOYMENT, self._execute_deployment)
            
            # Phase 4: Monitoring
            await self._execute_phase(DeploymentPhase.MONITORING, self._monitor_deployment)
            
            # Phase 5: Completion
            await self._execute_phase(DeploymentPhase.COMPLETION, self._complete_deployment)
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            await self._execute_rollback()
            raise

    async def _execute_phase(self, phase: DeploymentPhase, phase_func):
        """Execute a deployment phase with monitoring."""
        logger.info(f"Executing phase: {phase.value}")
        self.deployment_state.phase = phase
        self.deployment_state.logs.append(f"Phase {phase.value} started at {datetime.utcnow()}")
        
        start_time = time.time()
        await phase_func()
        duration = time.time() - start_time
        
        self.deployment_state.metrics[f"{phase.value}_duration"] = duration
        self.deployment_state.logs.append(f"Phase {phase.value} completed in {duration:.2f}s")

    async def _prepare_deployment(self):
        """Prepare deployment environment and create rollback point."""
        logger.info("Preparing deployment environment")
        
        # Create rollback point
        rollback_id = f"rollback-{int(time.time())}"
        result = await self._run_command(
            f"{self.scripts_dir}/disaster-recovery.sh create_backup {rollback_id}"
        )
        if result.returncode == 0:
            self.deployment_state.rollback_point = rollback_id
            logger.info(f"Rollback point created: {rollback_id}")
        
        # Pre-deployment health check
        await self._check_system_health()
        
        # Validate resources
        await self._validate_resources()

    async def _validate_deployment(self):
        """Validate deployment configuration and prerequisites."""
        logger.info("Validating deployment configuration")
        
        # Validate Kubernetes resources
        result = await self._run_command("kubectl auth can-i '*' '*' --all-namespaces")
        if result.returncode != 0:
            raise RuntimeError("Insufficient Kubernetes permissions")
        
        # Validate configuration
        if not self.config.features:
            raise ValueError("No features specified for deployment")
        
        # Check resource availability
        node_info = await self._get_node_resources()
        required_cpu = sum(
            int(res["cpu"].replace("m", "")) for res in self.config.resources.values()
        )
        if node_info["available_cpu"] < required_cpu:
            raise RuntimeError("Insufficient CPU resources")

    async def _execute_deployment(self):
        """Execute the main deployment process."""
        logger.info("Executing deployment")
        
        # Run production deployment script
        env = os.environ.copy()
        env.update({
            "NAMESPACE": self.config.namespace,
            "ENVIRONMENT": self.config.environment,
            "FEATURES": ",".join(self.config.features)
        })
        
        result = await self._run_command(
            f"{self.scripts_dir}/deploy-production.sh",
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Deployment script failed: {result.stderr}")
        
        # Wait for initial deployment stabilization
        await asyncio.sleep(30)
        
        # Verify core services
        await self._verify_core_services()

    async def _monitor_deployment(self):
        """Monitor deployment health and stability."""
        logger.info("Monitoring deployment stability")
        
        monitoring_duration = 300  # 5 minutes
        check_interval = 30  # 30 seconds
        
        for i in range(0, monitoring_duration, check_interval):
            await self._update_service_health()
            
            # Check for any unhealthy services
            unhealthy_services = [
                name for name, health in self.deployment_state.services.items()
                if health.status == ServiceStatus.UNHEALTHY
            ]
            
            if unhealthy_services:
                logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                # Allow some time for self-healing
                if i > 120:  # After 2 minutes, consider rollback
                    raise RuntimeError(f"Services remain unhealthy: {unhealthy_services}")
            
            await asyncio.sleep(check_interval)

    async def _complete_deployment(self):
        """Complete deployment with post-deployment setup."""
        logger.info("Completing deployment")
        
        # Run post-deployment setup
        result = await self._run_command(f"{self.scripts_dir}/post-deployment-setup.sh")
        if result.returncode != 0:
            logger.warning("Post-deployment setup had issues")
        
        # Final health check
        await self._update_service_health()
        
        # Generate deployment report
        await self._generate_deployment_report()

    async def _execute_rollback(self):
        """Execute rollback to previous stable state."""
        if not self.deployment_state.rollback_point:
            logger.error("No rollback point available")
            return
        
        logger.info(f"Executing rollback to {self.deployment_state.rollback_point}")
        self.deployment_state.phase = DeploymentPhase.ROLLBACK
        
        result = await self._run_command(
            f"{self.scripts_dir}/disaster-recovery.sh restore_backup {self.deployment_state.rollback_point}"
        )
        
        if result.returncode == 0:
            logger.info("Rollback completed successfully")
        else:
            logger.error("Rollback failed - manual intervention required")

    async def _update_service_health(self):
        """Update health status for all services."""
        services = ["orchestrator", "redis", "postgres", "monitoring"]
        
        for service in services:
            try:
                health = await self._check_service_health(service)
                self.deployment_state.services[service] = health
            except Exception as e:
                logger.error(f"Failed to check health for {service}: {e}")
                self.deployment_state.services[service] = ServiceHealth(
                    name=service,
                    status=ServiceStatus.UNKNOWN,
                    replicas=(0, 0),
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    last_check=datetime.utcnow(),
                    errors=[str(e)]
                )

    async def _check_service_health(self, service: str) -> ServiceHealth:
        """Check health of a specific service."""
        # Get pod status
        result = await self._run_command(
            f"kubectl get pods -n {self.config.namespace} -l app={service} -o json"
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get pod status for {service}")
        
        pod_data = json.loads(result.stdout)
        pods = pod_data.get("items", [])
        
        ready_pods = sum(1 for pod in pods if self._is_pod_ready(pod))
        total_pods = len(pods)
        
        # Determine status
        if ready_pods == 0:
            status = ServiceStatus.UNHEALTHY
        elif ready_pods < total_pods:
            status = ServiceStatus.DEGRADED
        else:
            status = ServiceStatus.HEALTHY
        
        # Get resource usage (simplified)
        cpu_usage = await self._get_service_cpu_usage(service)
        memory_usage = await self._get_service_memory_usage(service)
        
        return ServiceHealth(
            name=service,
            status=status,
            replicas=(ready_pods, total_pods),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            last_check=datetime.utcnow(),
            errors=[]
        )

    def _is_pod_ready(self, pod: Dict) -> bool:
        """Check if a pod is ready."""
        conditions = pod.get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition.get("type") == "Ready":
                return condition.get("status") == "True"
        return False

    async def _get_service_cpu_usage(self, service: str) -> float:
        """Get CPU usage for a service (simplified implementation)."""
        try:
            result = await self._run_command(
                f"kubectl top pods -n {self.config.namespace} -l app={service} --no-headers"
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse CPU usage from kubectl top output
                lines = result.stdout.strip().split('\n')
                total_cpu = 0
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        cpu_str = parts[1].replace('m', '')
                        total_cpu += int(cpu_str) if cpu_str.isdigit() else 0
                return total_cpu
        except Exception:
            pass
        return 0.0

    async def _get_service_memory_usage(self, service: str) -> float:
        """Get memory usage for a service (simplified implementation)."""
        try:
            result = await self._run_command(
                f"kubectl top pods -n {self.config.namespace} -l app={service} --no-headers"
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse memory usage from kubectl top output
                lines = result.stdout.strip().split('\n')
                total_memory = 0
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        mem_str = parts[2].replace('Mi', '').replace('Gi', '')
                        if 'Gi' in parts[2]:
                            total_memory += float(mem_str) * 1024
                        else:
                            total_memory += float(mem_str) if mem_str.replace('.', '').isdigit() else 0
                return total_memory
        except Exception:
            pass
        return 0.0

    async def _verify_core_services(self):
        """Verify that core services are running."""
        core_services = ["orchestrator", "redis", "postgres"]
        
        for service in core_services:
            result = await self._run_command(
                f"kubectl rollout status deployment/{service} -n {self.config.namespace} --timeout=300s"
            )
            if result.returncode != 0:
                raise RuntimeError(f"Core service {service} failed to deploy")

    async def _check_system_health(self):
        """Check overall system health."""
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            raise RuntimeError(f"Insufficient disk space: {disk_usage.percent}% used")
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise RuntimeError(f"Insufficient memory: {memory.percent}% used")

    async def _validate_resources(self):
        """Validate available resources."""
        node_info = await self._get_node_resources()
        
        # Calculate required resources
        total_cpu = sum(
            int(res["cpu"].replace("m", "")) for res in self.config.resources.values()
        )
        total_memory = sum(
            self._parse_memory(res["memory"]) for res in self.config.resources.values()
        )
        
        if node_info["available_cpu"] < total_cpu * 1.2:  # 20% buffer
            raise RuntimeError(f"Insufficient CPU: need {total_cpu}m, have {node_info['available_cpu']}m")
        
        if node_info["available_memory"] < total_memory * 1.2:  # 20% buffer
            raise RuntimeError(f"Insufficient memory: need {total_memory}Mi, have {node_info['available_memory']}Mi")

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to MB."""
        if memory_str.endswith('Gi'):
            return int(float(memory_str[:-2]) * 1024)
        elif memory_str.endswith('Mi'):
            return int(memory_str[:-2])
        else:
            return int(memory_str)

    async def _get_node_resources(self) -> Dict[str, int]:
        """Get available node resources."""
        result = await self._run_command("kubectl top nodes --no-headers")
        if result.returncode != 0:
            return {"available_cpu": 8000, "available_memory": 16384}  # Default values
        
        # Simplified parsing - in production, use proper resource queries
        return {"available_cpu": 8000, "available_memory": 16384}

    async def _generate_deployment_report(self):
        """Generate comprehensive deployment report."""
        report = {
            "deployment_id": self.deployment_state.deployment_id,
            "start_time": self.deployment_state.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration": (datetime.utcnow() - self.deployment_state.start_time).total_seconds(),
            "phase": self.deployment_state.phase.value,
            "services": {
                name: {
                    "status": health.status.value,
                    "replicas": health.replicas,
                    "cpu_usage": health.cpu_usage,
                    "memory_usage": health.memory_usage,
                    "errors": health.errors
                }
                for name, health in self.deployment_state.services.items()
            },
            "metrics": self.deployment_state.metrics,
            "rollback_point": self.deployment_state.rollback_point,
            "configuration": asdict(self.config)
        }
        
        # Save report
        report_path = f"/var/log/xorb-deployment-{self.deployment_state.deployment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Deployment report saved to {report_path}")

    async def _run_command(self, command: str, env: Dict[str, str] = None) -> subprocess.CompletedProcess:
        """Run shell command asynchronously."""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else ""
        )

    async def get_deployment_status(self, deployment_id: str = None) -> Dict:
        """Get current deployment status."""
        if not self.deployment_state:
            return {"status": "no_active_deployment"}
        
        if deployment_id and self.deployment_state.deployment_id != deployment_id:
            return {"status": "deployment_not_found"}
        
        return {
            "deployment_id": self.deployment_state.deployment_id,
            "phase": self.deployment_state.phase.value,
            "start_time": self.deployment_state.start_time.isoformat(),
            "duration": (datetime.utcnow() - self.deployment_state.start_time).total_seconds(),
            "services": {
                name: {
                    "status": health.status.value,
                    "replicas": health.replicas,
                    "last_check": health.last_check.isoformat()
                }
                for name, health in self.deployment_state.services.items()
            },
            "metrics": self.deployment_state.metrics
        }

    async def cleanup(self):
        """Cleanup resources."""
        if self.monitoring_session:
            await self.monitoring_session.close()

async def main():
    """Main orchestrator function.""" 
    import argparse
    
    parser = argparse.ArgumentParser(description="XORB Deployment Orchestrator")
    parser.add_argument("action", choices=["deploy", "status", "rollback"], help="Action to perform")
    parser.add_argument("--deployment-id", help="Deployment ID for status/rollback operations")
    parser.add_argument("--config", default="/root/Xorb/config/deployment-config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    orchestrator = XORBDeploymentOrchestrator(args.config)
    
    try:
        await orchestrator.initialize()
        
        if args.action == "deploy":
            deployment_id = await orchestrator.deploy(args.deployment_id)
            print(f"Deployment started: {deployment_id}")
            
        elif args.action == "status":
            status = await orchestrator.get_deployment_status(args.deployment_id)
            print(json.dumps(status, indent=2, default=str))
            
        elif args.action == "rollback":
            if orchestrator.deployment_state and orchestrator.deployment_state.rollback_point:
                await orchestrator._execute_rollback()
                print("Rollback initiated")
            else:
                print("No rollback point available")
                
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        raise
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())