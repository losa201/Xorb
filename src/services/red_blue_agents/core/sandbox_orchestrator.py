"""
Sandbox Orchestrator for Red/Blue Agent Framework

Manages isolated execution environments using Docker sidecar containers and Kata containers.
Provides TTL management, resource quotas, and mission-based labeling.
"""

import asyncio
import logging
import json
import time
import ssl
import socket
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid

import docker
import docker.errors
from docker.models.containers import Container
from docker.models.networks import Network
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class SandboxType(Enum):
    """Types of sandbox environments"""
    DOCKER_SIDECAR = "docker_sidecar"
    KATA_CONTAINER = "kata_container"
    DOCKER_IN_DOCKER = "docker_in_docker"


class SandboxStatus(Enum):
    """Sandbox container statuses"""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    EXPIRED = "expired"


@dataclass
class ResourceConstraints:
    """Resource constraints for sandbox containers"""
    cpu_cores: float = 2.0
    memory_mb: int = 1024
    disk_mb: int = 2048
    network_bandwidth_mb: int = 100
    max_processes: int = 1000
    max_open_files: int = 1024


@dataclass
class NetworkPolicy:
    """Network policies for sandbox isolation"""
    isolation_mode: str = "bridge"  # bridge, host, none, custom
    allowed_outbound: List[str] = None
    blocked_outbound: List[str] = None
    allowed_inbound: List[str] = None
    blocked_inbound: List[str] = None
    dns_servers: List[str] = None
    
    def __post_init__(self):
        if self.allowed_outbound is None:
            self.allowed_outbound = []
        if self.blocked_outbound is None:
            self.blocked_outbound = []
        if self.allowed_inbound is None:
            self.allowed_inbound = []
        if self.blocked_inbound is None:
            self.blocked_inbound = []
        if self.dns_servers is None:
            self.dns_servers = ["8.8.8.8", "8.8.4.4"]


@dataclass
class SandboxConfig:
    """Configuration for a sandbox environment"""
    sandbox_id: str
    sandbox_type: SandboxType
    mission_id: str
    agent_type: str  # red_recon, red_exploit, blue_detect, etc.
    environment: str  # production, staging, development, cyber_range
    image: str
    command: List[str] = None
    working_dir: str = "/app"
    environment_vars: Dict[str, str] = None
    volumes: Dict[str, str] = None  # host_path -> container_path
    ports: Dict[int, int] = None  # host_port -> container_port
    resource_constraints: ResourceConstraints = None
    network_policy: NetworkPolicy = None
    ttl_seconds: int = 3600  # 1 hour default TTL
    idle_timeout_seconds: int = 300  # 5 minutes idle timeout
    privileged: bool = False
    capabilities: List[str] = None
    security_opts: List[str] = None
    
    def __post_init__(self):
        if self.command is None:
            self.command = ["/bin/bash"]
        if self.environment_vars is None:
            self.environment_vars = {}
        if self.volumes is None:
            self.volumes = {}
        if self.ports is None:
            self.ports = {}
        if self.resource_constraints is None:
            self.resource_constraints = ResourceConstraints()
        if self.network_policy is None:
            self.network_policy = NetworkPolicy()
        if self.capabilities is None:
            self.capabilities = []
        if self.security_opts is None:
            self.security_opts = ["no-new-privileges:true"]


@dataclass
class SandboxInstance:
    """Running sandbox instance"""
    config: SandboxConfig
    container_id: str
    container_name: str
    status: SandboxStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    ip_address: Optional[str] = None
    ports: Dict[int, int] = None
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ports is None:
            self.ports = {}
        if self.logs is None:
            self.logs = []
        if self.metrics is None:
            self.metrics = {}


class DockerSandboxManager:
    """Manages Docker-based sandbox containers"""
    
    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client
        self.tls_context = self._setup_tls_context()
        
    def _setup_tls_context(self) -> Optional[ssl.SSLContext]:
        """Setup TLS context for Docker-in-Docker communication"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        except Exception as e:
            logger.warning(f"Failed to setup TLS context: {e}")
            return None
            
    async def create_sandbox(self, config: SandboxConfig) -> SandboxInstance:
        """Create a new sandbox container"""
        try:
            # Generate unique container name
            container_name = f"xorb-{config.agent_type}-{config.sandbox_id}"
            
            # Prepare Docker run configuration
            docker_config = await self._prepare_docker_config(config, container_name)
            
            # Create container
            container = self.docker_client.containers.create(**docker_config)
            
            # Create sandbox instance
            instance = SandboxInstance(
                config=config,
                container_id=container.id,
                container_name=container_name,
                status=SandboxStatus.PENDING,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=config.ttl_seconds)
            )
            
            logger.info(f"Created sandbox container {container_name} ({container.id[:12]})")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
            
    async def _prepare_docker_config(self, config: SandboxConfig, container_name: str) -> Dict[str, Any]:
        """Prepare Docker container configuration"""
        docker_config = {
            "image": config.image,
            "name": container_name,
            "command": config.command,
            "working_dir": config.working_dir,
            "environment": config.environment_vars,
            "detach": True,
            "remove": False,  # We'll remove manually after TTL
            "labels": {
                "xorb.mission_id": config.mission_id,
                "xorb.agent_type": config.agent_type,
                "xorb.environment": config.environment,
                "xorb.sandbox_type": config.sandbox_type.value,
                "xorb.created_at": datetime.utcnow().isoformat(),
                "xorb.ttl_seconds": str(config.ttl_seconds)
            }
        }
        
        # Resource constraints
        if config.resource_constraints:
            docker_config["mem_limit"] = f"{config.resource_constraints.memory_mb}m"
            docker_config["cpu_count"] = config.resource_constraints.cpu_cores
            docker_config["pids_limit"] = config.resource_constraints.max_processes
            docker_config["ulimits"] = [
                docker.types.Ulimit(name="nofile", soft=config.resource_constraints.max_open_files, 
                                  hard=config.resource_constraints.max_open_files)
            ]
            
        # Volume mounts
        if config.volumes:
            docker_config["volumes"] = config.volumes
            
        # Port mappings
        if config.ports:
            docker_config["ports"] = config.ports
            
        # Security options
        if config.privileged:
            docker_config["privileged"] = True
        if config.capabilities:
            docker_config["cap_add"] = config.capabilities
        if config.security_opts:
            docker_config["security_opt"] = config.security_opts
            
        # Network configuration
        if config.network_policy.isolation_mode != "bridge":
            docker_config["network_mode"] = config.network_policy.isolation_mode
            
        return docker_config
        
    async def start_sandbox(self, instance: SandboxInstance) -> bool:
        """Start a sandbox container"""
        try:
            container = self.docker_client.containers.get(instance.container_id)
            container.start()
            
            instance.status = SandboxStatus.RUNNING
            instance.started_at = datetime.utcnow()
            instance.last_activity = datetime.utcnow()
            
            # Get container IP address
            container.reload()
            if container.attrs.get("NetworkSettings", {}).get("IPAddress"):
                instance.ip_address = container.attrs["NetworkSettings"]["IPAddress"]
                
            # Setup network policies
            await self._apply_network_policies(container, instance.config.network_policy)
            
            logger.info(f"Started sandbox container {instance.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sandbox {instance.container_name}: {e}")
            instance.status = SandboxStatus.ERROR
            return False
            
    async def _apply_network_policies(self, container: Container, policy: NetworkPolicy):
        """Apply network policies using iptables or container networks"""
        try:
            # This would implement network policy enforcement
            # For now, just log the policies that would be applied
            logger.info(f"Applying network policies to container {container.name}")
            
            if policy.blocked_outbound:
                logger.info(f"Would block outbound to: {policy.blocked_outbound}")
            if policy.allowed_outbound:
                logger.info(f"Would allow outbound to: {policy.allowed_outbound}")
                
        except Exception as e:
            logger.warning(f"Failed to apply network policies: {e}")
            
    async def stop_sandbox(self, instance: SandboxInstance, force: bool = False) -> bool:
        """Stop a sandbox container"""
        try:
            container = self.docker_client.containers.get(instance.container_id)
            
            if force:
                container.kill()
            else:
                container.stop(timeout=30)
                
            instance.status = SandboxStatus.STOPPED
            
            logger.info(f"Stopped sandbox container {instance.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox {instance.container_name}: {e}")
            return False
            
    async def remove_sandbox(self, instance: SandboxInstance) -> bool:
        """Remove a sandbox container"""
        try:
            container = self.docker_client.containers.get(instance.container_id)
            container.remove(force=True)
            
            logger.info(f"Removed sandbox container {instance.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove sandbox {instance.container_name}: {e}")
            return False
            
    async def get_sandbox_logs(self, instance: SandboxInstance, tail: int = 100) -> List[str]:
        """Get logs from a sandbox container"""
        try:
            container = self.docker_client.containers.get(instance.container_id)
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8').split('\n')
            return [log.strip() for log in logs if log.strip()]
            
        except Exception as e:
            logger.error(f"Failed to get logs for sandbox {instance.container_name}: {e}")
            return []
            
    async def get_sandbox_metrics(self, instance: SandboxInstance) -> Dict[str, Any]:
        """Get resource usage metrics for a sandbox container"""
        try:
            container = self.docker_client.containers.get(instance.container_id)
            stats = container.stats(stream=False)
            
            # Calculate metrics
            metrics = {
                "cpu_usage_percent": 0.0,
                "memory_usage_mb": 0,
                "memory_limit_mb": 0,
                "network_rx_bytes": 0,
                "network_tx_bytes": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Parse CPU usage
            if "cpu_stats" in stats and "precpu_stats" in stats:
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                           stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                              stats["precpu_stats"]["system_cpu_usage"]
                if system_delta > 0:
                    metrics["cpu_usage_percent"] = (cpu_delta / system_delta) * 100.0
                    
            # Parse memory usage
            if "memory_stats" in stats:
                metrics["memory_usage_mb"] = stats["memory_stats"].get("usage", 0) // (1024 * 1024)
                metrics["memory_limit_mb"] = stats["memory_stats"].get("limit", 0) // (1024 * 1024)
                
            # Parse network usage
            if "networks" in stats:
                for interface, net_stats in stats["networks"].items():
                    metrics["network_rx_bytes"] += net_stats.get("rx_bytes", 0)
                    metrics["network_tx_bytes"] += net_stats.get("tx_bytes", 0)
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics for sandbox {instance.container_name}: {e}")
            return {}


class KataSandboxManager:
    """Manages Kata container-based sandbox environments"""
    
    def __init__(self, runtime_path: str = "/usr/bin/kata-runtime"):
        self.runtime_path = runtime_path
        self.kata_available = self._check_kata_availability()
        
    def _check_kata_availability(self) -> bool:
        """Check if Kata containers runtime is available"""
        try:
            import subprocess
            result = subprocess.run([self.runtime_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
            
    async def create_sandbox(self, config: SandboxConfig) -> SandboxInstance:
        """Create a Kata container sandbox"""
        if not self.kata_available:
            raise RuntimeError("Kata containers runtime not available")
            
        # For now, delegate to Docker with Kata runtime
        # In a full implementation, this would use Kata APIs directly
        docker_manager = DockerSandboxManager(docker.from_env())
        
        # Modify config to use Kata runtime
        kata_config = config
        kata_config.security_opts.append("kata-containers")
        
        return await docker_manager.create_sandbox(kata_config)


class SandboxOrchestrator:
    """
    Main orchestrator for managing sandbox environments across the red/blue agent framework.
    
    Features:
    - Multi-type sandbox support (Docker sidecar, Kata containers, DinD)
    - TTL and quota management
    - Mission-based container labeling
    - Resource monitoring and cleanup
    - Network isolation and policies
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 docker_client: Optional[docker.DockerClient] = None):
        self.redis_client = redis_client
        self.docker_client = docker_client or docker.from_env()
        self.docker_manager = DockerSandboxManager(self.docker_client)
        self.kata_manager = KataSandboxManager()
        
        # Active sandbox instances
        self.active_sandboxes: Dict[str, SandboxInstance] = {}
        
        # Mission-based quotas
        self.mission_quotas: Dict[str, Dict[str, int]] = {}
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the sandbox orchestrator"""
        logger.info("Initializing Sandbox Orchestrator...")
        
        # Load existing sandboxes from Docker
        await self._discover_existing_sandboxes()
        
        # Start cleanup background task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Sandbox Orchestrator initialized with {len(self.active_sandboxes)} existing sandboxes")
        
    async def _discover_existing_sandboxes(self):
        """Discover existing XORB sandbox containers"""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"label": "xorb.mission_id"}
            )
            
            for container in containers:
                try:
                    # Reconstruct sandbox instance from container
                    labels = container.labels
                    
                    config = SandboxConfig(
                        sandbox_id=labels.get("xorb.sandbox_id", container.id[:12]),
                        sandbox_type=SandboxType(labels.get("xorb.sandbox_type", "docker_sidecar")),
                        mission_id=labels["xorb.mission_id"],
                        agent_type=labels.get("xorb.agent_type", "unknown"),
                        environment=labels.get("xorb.environment", "development"),
                        image=container.image.tags[0] if container.image.tags else "unknown",
                        ttl_seconds=int(labels.get("xorb.ttl_seconds", "3600"))
                    )
                    
                    instance = SandboxInstance(
                        config=config,
                        container_id=container.id,
                        container_name=container.name,
                        status=SandboxStatus(container.status.lower()) if container.status.lower() in 
                               [s.value for s in SandboxStatus] else SandboxStatus.UNKNOWN,
                        created_at=datetime.fromisoformat(labels.get("xorb.created_at", 
                                                        datetime.utcnow().isoformat()))
                    )
                    
                    self.active_sandboxes[instance.config.sandbox_id] = instance
                    
                except Exception as e:
                    logger.warning(f"Failed to reconstruct sandbox from container {container.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to discover existing sandboxes: {e}")
            
    async def create_sandbox(self, config: SandboxConfig) -> str:
        """Create a new sandbox environment"""
        # Check mission quotas
        await self._check_mission_quotas(config.mission_id, config.agent_type)
        
        # Choose appropriate manager
        if config.sandbox_type == SandboxType.KATA_CONTAINER:
            manager = self.kata_manager
        else:
            manager = self.docker_manager
            
        # Create sandbox
        instance = await manager.create_sandbox(config)
        
        # Store in active sandboxes
        self.active_sandboxes[config.sandbox_id] = instance
        
        # Update quotas
        await self._update_mission_quotas(config.mission_id, config.agent_type, 1)
        
        # Cache in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"sandbox:{config.sandbox_id}",
                    config.ttl_seconds,
                    json.dumps(asdict(instance), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache sandbox in Redis: {e}")
                
        logger.info(f"Created sandbox {config.sandbox_id} for mission {config.mission_id}")
        return config.sandbox_id
        
    async def start_sandbox(self, sandbox_id: str) -> bool:
        """Start a sandbox environment"""
        instance = self.active_sandboxes.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox {sandbox_id} not found")
            return False
            
        # Choose appropriate manager
        if instance.config.sandbox_type == SandboxType.KATA_CONTAINER:
            manager = self.kata_manager
        else:
            manager = self.docker_manager
            
        success = await manager.start_sandbox(instance)
        
        if success:
            # Update Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        f"sandbox:{sandbox_id}",
                        instance.config.ttl_seconds,
                        json.dumps(asdict(instance), default=str)
                    )
                except Exception as e:
                    logger.warning(f"Failed to update sandbox cache: {e}")
                    
        return success
        
    async def stop_sandbox(self, sandbox_id: str, force: bool = False) -> bool:
        """Stop a sandbox environment"""
        instance = self.active_sandboxes.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox {sandbox_id} not found")
            return False
            
        # Choose appropriate manager
        if instance.config.sandbox_type == SandboxType.KATA_CONTAINER:
            manager = self.kata_manager
        else:
            manager = self.docker_manager
            
        return await manager.stop_sandbox(instance, force)
        
    async def remove_sandbox(self, sandbox_id: str) -> bool:
        """Remove a sandbox environment"""
        instance = self.active_sandboxes.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox {sandbox_id} not found")
            return False
            
        # Choose appropriate manager
        if instance.config.sandbox_type == SandboxType.KATA_CONTAINER:
            manager = self.kata_manager
        else:
            manager = self.docker_manager
            
        success = await manager.remove_sandbox(instance)
        
        if success:
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            # Update quotas
            await self._update_mission_quotas(
                instance.config.mission_id, 
                instance.config.agent_type, 
                -1
            )
            
            # Remove from Redis
            if self.redis_client:
                try:
                    await self.redis_client.delete(f"sandbox:{sandbox_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove sandbox from cache: {e}")
                    
        return success
        
    async def get_sandbox_status(self, sandbox_id: str) -> Optional[SandboxInstance]:
        """Get status of a sandbox environment"""
        return self.active_sandboxes.get(sandbox_id)
        
    async def list_sandboxes(self, mission_id: Optional[str] = None, 
                           agent_type: Optional[str] = None) -> List[SandboxInstance]:
        """List sandbox environments with optional filtering"""
        sandboxes = list(self.active_sandboxes.values())
        
        if mission_id:
            sandboxes = [s for s in sandboxes if s.config.mission_id == mission_id]
        if agent_type:
            sandboxes = [s for s in sandboxes if s.config.agent_type == agent_type]
            
        return sandboxes
        
    async def _check_mission_quotas(self, mission_id: str, agent_type: str):
        """Check if creating a new sandbox would exceed mission quotas"""
        current_count = len([s for s in self.active_sandboxes.values() 
                           if s.config.mission_id == mission_id and s.config.agent_type == agent_type])
        
        # Default quota of 10 per agent type per mission
        quota = self.mission_quotas.get(mission_id, {}).get(agent_type, 10)
        
        if current_count >= quota:
            raise RuntimeError(f"Mission {mission_id} has reached quota for {agent_type} agents ({quota})")
            
    async def _update_mission_quotas(self, mission_id: str, agent_type: str, delta: int):
        """Update mission quota tracking"""
        if mission_id not in self.mission_quotas:
            self.mission_quotas[mission_id] = {}
        if agent_type not in self.mission_quotas[mission_id]:
            self.mission_quotas[mission_id][agent_type] = 0
            
        self.mission_quotas[mission_id][agent_type] += delta
        self.mission_quotas[mission_id][agent_type] = max(0, self.mission_quotas[mission_id][agent_type])
        
    async def _cleanup_loop(self):
        """Background task to cleanup expired sandboxes"""
        while True:
            try:
                await self._cleanup_expired_sandboxes()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_expired_sandboxes(self):
        """Remove expired sandbox containers"""
        now = datetime.utcnow()
        expired_sandboxes = []
        
        for sandbox_id, instance in self.active_sandboxes.items():
            if instance.expires_at and now > instance.expires_at:
                expired_sandboxes.append(sandbox_id)
            elif (instance.last_activity and 
                  now > instance.last_activity + timedelta(seconds=instance.config.idle_timeout_seconds)):
                instance.status = SandboxStatus.EXPIRED
                expired_sandboxes.append(sandbox_id)
                
        for sandbox_id in expired_sandboxes:
            try:
                logger.info(f"Cleaning up expired sandbox {sandbox_id}")
                await self.remove_sandbox(sandbox_id)
            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
                
    async def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        stats = {
            "total_sandboxes": len(self.active_sandboxes),
            "sandboxes_by_status": {},
            "sandboxes_by_type": {},
            "sandboxes_by_mission": {},
            "resource_usage": {
                "total_cpu_cores": 0.0,
                "total_memory_mb": 0,
                "total_containers": len(self.active_sandboxes)
            }
        }
        
        for instance in self.active_sandboxes.values():
            # Count by status
            status = instance.status.value
            stats["sandboxes_by_status"][status] = stats["sandboxes_by_status"].get(status, 0) + 1
            
            # Count by type
            stype = instance.config.sandbox_type.value
            stats["sandboxes_by_type"][stype] = stats["sandboxes_by_type"].get(stype, 0) + 1
            
            # Count by mission
            mission = instance.config.mission_id
            stats["sandboxes_by_mission"][mission] = stats["sandboxes_by_mission"].get(mission, 0) + 1
            
            # Sum resources
            if instance.config.resource_constraints:
                stats["resource_usage"]["total_cpu_cores"] += instance.config.resource_constraints.cpu_cores
                stats["resource_usage"]["total_memory_mb"] += instance.config.resource_constraints.memory_mb
                
        return stats
        
    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources"""
        logger.info("Shutting down Sandbox Orchestrator...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Stop all active sandboxes
        for sandbox_id in list(self.active_sandboxes.keys()):
            try:
                await self.stop_sandbox(sandbox_id, force=True)
                await self.remove_sandbox(sandbox_id)
            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {sandbox_id} during shutdown: {e}")
                
        logger.info("Sandbox Orchestrator shutdown complete")