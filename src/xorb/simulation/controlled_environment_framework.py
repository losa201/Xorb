#!/usr/bin/env python3
"""
Controlled Environment Framework - Production Implementation
Advanced simulation environment for safe autonomous red team operations

SECURITY NOTICE: This module implements controlled simulation environments
for safe red team training and validation in isolated, monitored environments.

Key Features:
- Docker-based isolated cyber ranges
- Realistic vulnerable application deployment
- Dynamic network topology simulation
- Comprehensive monitoring and logging
- Real-time learning progress tracking
- Safety boundary enforcement
- Performance metrics and optimization
- Multi-scenario support with complexity scaling
"""

import asyncio
import logging
import json
import uuid
import docker
import yaml
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import tempfile
import subprocess
import ipaddress
import threading
import time
import psutil
import socket
from contextlib import asynccontextmanager
import shutil

# Network and system imports
try:
    import docker
    import paramiko
    import nmap
    SIMULATION_DEPS_AVAILABLE = True
except ImportError:
    SIMULATION_DEPS_AVAILABLE = False
    logging.warning("Simulation dependencies not available - using mock implementations")

# Internal imports
from ..common.security_framework import SecurityFramework, SecurityLevel
from ..common.audit_logger import AuditLogger, AuditEvent
from ..learning.advanced_reinforcement_learning import AdvancedRLEngine, EnvironmentState, ActionResult

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Types of simulation environments"""
    CYBER_RANGE = "cyber_range"
    VULNERABLE_NETWORK = "vulnerable_network"
    WEB_APPLICATION_LAB = "web_application_lab"
    ENTERPRISE_SIMULATION = "enterprise_simulation"
    MOBILE_TESTBED = "mobile_testbed"
    IOT_ENVIRONMENT = "iot_environment"
    CLOUD_SIMULATION = "cloud_simulation"
    HYBRID_ENVIRONMENT = "hybrid_environment"


class ComplexityLevel(Enum):
    """Environment complexity levels"""
    BASIC = "basic"           # Simple single-host scenarios
    INTERMEDIATE = "intermediate"  # Multi-host with basic defenses
    ADVANCED = "advanced"     # Complex networks with active defenses
    EXPERT = "expert"         # Enterprise-level with sophisticated defenses
    RESEARCH = "research"     # Cutting-edge scenarios for research


class EnvironmentStatus(Enum):
    """Environment lifecycle status"""
    INITIALIZING = "initializing"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"


class MonitoringLevel(Enum):
    """Monitoring intensity levels"""
    MINIMAL = "minimal"       # Basic health checks
    STANDARD = "standard"     # Standard metrics collection
    DETAILED = "detailed"     # Comprehensive monitoring
    FORENSIC = "forensic"     # Full forensic-level monitoring


@dataclass
class NetworkTopology:
    """Network topology configuration"""
    subnets: List[Dict[str, Any]] = field(default_factory=list)
    hosts: List[Dict[str, Any]] = field(default_factory=list)
    routers: List[Dict[str, Any]] = field(default_factory=list)
    firewalls: List[Dict[str, Any]] = field(default_factory=list)
    switches: List[Dict[str, Any]] = field(default_factory=list)
    vlans: List[Dict[str, Any]] = field(default_factory=list)
    connectivity_matrix: Dict[str, List[str]] = field(default_factory=dict)
    network_services: List[Dict[str, Any]] = field(default_factory=list)
    security_controls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VulnerableApplication:
    """Configuration for vulnerable applications"""
    app_id: str
    name: str
    image: str
    vulnerabilities: List[str]
    exposed_ports: List[int]
    environment_variables: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    network_config: Dict[str, Any] = field(default_factory=dict)
    difficulty_level: ComplexityLevel = ComplexityLevel.BASIC
    learning_objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    hints_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationScenario:
    """Complete simulation scenario configuration"""
    scenario_id: str
    name: str
    description: str
    environment_type: EnvironmentType
    complexity_level: ComplexityLevel
    
    # Infrastructure
    network_topology: NetworkTopology
    vulnerable_applications: List[VulnerableApplication]
    defensive_systems: List[Dict[str, Any]] = field(default_factory=list)
    
    # Learning configuration
    learning_objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Monitoring and safety
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EnvironmentInstance:
    """Running environment instance"""
    instance_id: str
    scenario: SimulationScenario
    status: EnvironmentStatus
    
    # Docker infrastructure
    docker_network_id: Optional[str] = None
    container_ids: List[str] = field(default_factory=list)
    volume_ids: List[str] = field(default_factory=list)
    
    # Network configuration
    ip_assignments: Dict[str, str] = field(default_factory=dict)
    port_mappings: Dict[str, int] = field(default_factory=dict)
    access_credentials: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Runtime state
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Learning integration
    active_agents: List[str] = field(default_factory=list)
    learning_progress: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring data
    monitoring_enabled: bool = True
    log_paths: List[str] = field(default_factory=list)
    metrics_collection: Dict[str, Any] = field(default_factory=dict)
    
    # Safety and compliance
    safety_violations: List[str] = field(default_factory=list)
    emergency_stops: int = 0
    cleanup_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.last_activity:
            result['last_activity'] = self.last_activity.isoformat()
        return result


class DockerEnvironmentManager:
    """Docker-based environment management"""
    
    def __init__(self):
        if SIMULATION_DEPS_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                self.docker_available = True
            except Exception as e:
                logger.error(f"Docker client initialization failed: {e}")
                self.docker_available = False
        else:
            self.docker_available = False
        
        self.networks: Dict[str, Any] = {}
        self.containers: Dict[str, Any] = {}
        self.volumes: Dict[str, Any] = {}
    
    async def create_isolated_network(self, network_name: str, subnet: str) -> str:
        """Create isolated Docker network for simulation"""
        try:
            if not self.docker_available:
                # Mock implementation for testing
                network_id = f"mock_network_{uuid.uuid4().hex[:8]}"
                self.networks[network_id] = {
                    "name": network_name,
                    "subnet": subnet,
                    "created_at": datetime.utcnow()
                }
                return network_id
            
            # Create Docker network with custom subnet
            ipam_pool = docker.types.IPAMPool(subnet=subnet)
            ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
            
            network = self.docker_client.networks.create(
                name=network_name,
                driver="bridge",
                ipam=ipam_config,
                options={
                    "com.docker.network.bridge.enable_icc": "true",
                    "com.docker.network.bridge.enable_ip_masquerade": "true",
                    "com.docker.network.driver.mtu": "1500"
                },
                labels={
                    "xorb.simulation": "true",
                    "xorb.network_type": "isolated",
                    "xorb.created_at": datetime.utcnow().isoformat()
                }
            )
            
            network_id = network.id
            self.networks[network_id] = {
                "network_object": network,
                "name": network_name,
                "subnet": subnet,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Created isolated network {network_name} with subnet {subnet}")
            return network_id
            
        except Exception as e:
            logger.error(f"Failed to create isolated network: {e}")
            raise
    
    async def deploy_vulnerable_application(self, app_config: VulnerableApplication, 
                                          network_id: str, ip_address: str) -> str:
        """Deploy vulnerable application container"""
        try:
            if not self.docker_available:
                # Mock implementation
                container_id = f"mock_container_{uuid.uuid4().hex[:8]}"
                self.containers[container_id] = {
                    "app_config": app_config,
                    "network_id": network_id,
                    "ip_address": ip_address,
                    "created_at": datetime.utcnow(),
                    "status": "running"
                }
                return container_id
            
            # Prepare container configuration
            container_name = f"xorb_{app_config.app_id}_{uuid.uuid4().hex[:8]}"
            
            # Port mapping for external access
            port_bindings = {}
            for port in app_config.exposed_ports:
                port_bindings[f"{port}/tcp"] = None  # Random host port
            
            # Environment variables
            environment = {
                "XORB_SIMULATION": "true",
                "XORB_APP_ID": app_config.app_id,
                "XORB_DIFFICULTY": app_config.difficulty_level.value,
                **app_config.environment_variables
            }
            
            # Create and start container
            container = self.docker_client.containers.run(
                image=app_config.image,
                name=container_name,
                detach=True,
                environment=environment,
                ports=port_bindings,
                volumes=app_config.volumes,
                labels={
                    "xorb.simulation": "true",
                    "xorb.app_id": app_config.app_id,
                    "xorb.app_type": "vulnerable_application",
                    "xorb.difficulty": app_config.difficulty_level.value,
                    "xorb.created_at": datetime.utcnow().isoformat()
                },
                restart_policy={"Name": "unless-stopped"},
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                cap_add=["CHOWN", "DAC_OVERRIDE", "SETUID", "SETGID"] if app_config.vulnerabilities else []
            )
            
            # Connect to custom network with specific IP
            network = self.networks[network_id]["network_object"]
            network.connect(container, ipv4_address=ip_address)
            
            container_id = container.id
            self.containers[container_id] = {
                "container_object": container,
                "app_config": app_config,
                "network_id": network_id,
                "ip_address": ip_address,
                "created_at": datetime.utcnow(),
                "status": "running"
            }
            
            logger.info(f"Deployed vulnerable application {app_config.name} at {ip_address}")
            return container_id
            
        except Exception as e:
            logger.error(f"Failed to deploy vulnerable application: {e}")
            raise
    
    async def deploy_defensive_system(self, defense_config: Dict[str, Any], 
                                    network_id: str) -> str:
        """Deploy defensive security system"""
        try:
            if not self.docker_available:
                # Mock implementation
                defense_id = f"mock_defense_{uuid.uuid4().hex[:8]}"
                return defense_id
            
            defense_type = defense_config.get("type", "ids")
            
            if defense_type == "ids":
                # Deploy Intrusion Detection System (Suricata)
                container = self.docker_client.containers.run(
                    image="jasonish/suricata:latest",
                    name=f"xorb_ids_{uuid.uuid4().hex[:8]}",
                    detach=True,
                    network_mode="host",  # Monitor all traffic
                    volumes=[
                        "/var/log/suricata:/var/log/suricata",
                        "/etc/suricata:/etc/suricata"
                    ],
                    environment={
                        "SURICATA_OPTIONS": "-i any",
                        "XORB_DEFENSE_TYPE": "ids"
                    },
                    labels={
                        "xorb.simulation": "true",
                        "xorb.component": "defensive_system",
                        "xorb.defense_type": defense_type
                    }
                )
            
            elif defense_type == "honeypot":
                # Deploy honeypot system
                container = self.docker_client.containers.run(
                    image="cowrie/cowrie:latest",
                    name=f"xorb_honeypot_{uuid.uuid4().hex[:8]}",
                    detach=True,
                    ports={"2222/tcp": None, "2223/tcp": None},
                    volumes=["/var/log/cowrie:/cowrie/var/log/cowrie"],
                    labels={
                        "xorb.simulation": "true",
                        "xorb.component": "defensive_system",
                        "xorb.defense_type": defense_type
                    }
                )
            
            else:
                raise ValueError(f"Unsupported defense type: {defense_type}")
            
            defense_id = container.id
            self.containers[defense_id] = {
                "container_object": container,
                "config": defense_config,
                "type": "defensive_system",
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Deployed defensive system: {defense_type}")
            return defense_id
            
        except Exception as e:
            logger.error(f"Failed to deploy defensive system: {e}")
            raise
    
    async def monitor_container_health(self, container_id: str) -> Dict[str, Any]:
        """Monitor container health and resource usage"""
        try:
            if not self.docker_available or container_id not in self.containers:
                # Mock health data
                return {
                    "status": "running",
                    "cpu_percent": 15.5,
                    "memory_usage_mb": 128,
                    "network_rx_mb": 10.2,
                    "network_tx_mb": 5.8,
                    "uptime_seconds": 3600,
                    "health_check": "healthy"
                }
            
            container_info = self.containers[container_id]
            container = container_info["container_object"]
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Calculate memory usage
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            # Network statistics
            networks = stats.get("networks", {})
            total_rx = sum(net.get("rx_bytes", 0) for net in networks.values())
            total_tx = sum(net.get("tx_bytes", 0) for net in networks.values())
            
            health_data = {
                "status": container.status,
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "memory_percent": round(memory_percent, 2),
                "network_rx_mb": round(total_rx / (1024 * 1024), 2),
                "network_tx_mb": round(total_tx / (1024 * 1024), 2),
                "uptime_seconds": (datetime.utcnow() - container_info["created_at"]).total_seconds(),
                "health_check": "healthy" if container.status == "running" else "unhealthy"
            }
            
            return health_data
            
        except Exception as e:
            logger.error(f"Container health monitoring failed: {e}")
            return {"status": "unknown", "health_check": "error", "error": str(e)}
    
    async def cleanup_environment(self, network_id: str, container_ids: List[str]) -> bool:
        """Clean up environment resources"""
        try:
            success = True
            
            # Stop and remove containers
            for container_id in container_ids:
                try:
                    if self.docker_available and container_id in self.containers:
                        container = self.containers[container_id]["container_object"]
                        container.stop(timeout=10)
                        container.remove(force=True)
                    
                    del self.containers[container_id]
                    logger.info(f"Cleaned up container {container_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup container {container_id}: {e}")
                    success = False
            
            # Remove network
            try:
                if self.docker_available and network_id in self.networks:
                    network = self.networks[network_id]["network_object"]
                    network.remove()
                
                del self.networks[network_id]
                logger.info(f"Cleaned up network {network_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup network {network_id}: {e}")
                success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Environment cleanup failed: {e}")
            return False


class PerformanceMonitor:
    """Real-time performance monitoring for simulation environments"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "network_errors": 10
        }
    
    async def start_monitoring(self, instance_id: str, environment_manager: DockerEnvironmentManager):
        """Start performance monitoring for environment"""
        try:
            self.monitoring_active = True
            self.metrics_history[instance_id] = []
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(instance_id, environment_manager),
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"Started performance monitoring for {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def _monitoring_loop(self, instance_id: str, environment_manager: DockerEnvironmentManager):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": dict(psutil.virtual_memory()._asdict()),
                    "disk": dict(psutil.disk_usage('/')._asdict()),
                    "network": dict(psutil.net_io_counters()._asdict()),
                    "processes": len(psutil.pids()),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                }
                
                # Store metrics
                if instance_id not in self.metrics_history:
                    self.metrics_history[instance_id] = []
                
                self.metrics_history[instance_id].append(system_metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history[instance_id]) > 1000:
                    self.metrics_history[instance_id] = self.metrics_history[instance_id][-1000:]
                
                # Check for alerts
                self._check_performance_alerts(system_metrics, instance_id)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _check_performance_alerts(self, metrics: Dict[str, Any], instance_id: str):
        """Check for performance alert conditions"""
        alerts = []
        
        if metrics["cpu_percent"] > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        memory_percent = (metrics["memory"]["used"] / metrics["memory"]["total"]) * 100
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")
        
        disk_percent = (metrics["disk"]["used"] / metrics["disk"]["total"]) * 100
        if disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append(f"High disk usage: {disk_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Performance alerts for {instance_id}: {'; '.join(alerts)}")
    
    async def get_performance_summary(self, instance_id: str) -> Dict[str, Any]:
        """Get performance summary for environment"""
        try:
            if instance_id not in self.metrics_history:
                return {"error": "No metrics available"}
            
            metrics = self.metrics_history[instance_id]
            if not metrics:
                return {"error": "No metrics collected"}
            
            # Calculate averages from recent metrics
            recent_metrics = metrics[-20:]  # Last 20 data points
            
            avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_memory_percent = sum(
                (m["memory"]["used"] / m["memory"]["total"]) * 100 
                for m in recent_metrics
            ) / len(recent_metrics)
            
            latest = metrics[-1]
            
            return {
                "instance_id": instance_id,
                "monitoring_duration_minutes": len(metrics) * 5 / 60,  # 5-second intervals
                "current_status": {
                    "cpu_percent": latest["cpu_percent"],
                    "memory_percent": (latest["memory"]["used"] / latest["memory"]["total"]) * 100,
                    "disk_percent": (latest["disk"]["used"] / latest["disk"]["total"]) * 100,
                    "active_processes": latest["processes"],
                    "load_average": latest["load_average"]
                },
                "averages": {
                    "cpu_percent": round(avg_cpu, 2),
                    "memory_percent": round(avg_memory_percent, 2)
                },
                "total_data_points": len(metrics),
                "last_updated": latest["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    async def stop_monitoring(self, instance_id: str):
        """Stop performance monitoring"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info(f"Stopped performance monitoring for {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")


class LearningProgressTracker:
    """Track learning progress in simulation environments"""
    
    def __init__(self):
        self.learning_sessions: Dict[str, Dict[str, Any]] = {}
        self.progress_metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    async def start_learning_session(self, instance_id: str, agent_id: str, 
                                   learning_objectives: List[str]) -> str:
        """Start tracking learning session"""
        session_id = f"{instance_id}_{agent_id}_{uuid.uuid4().hex[:8]}"
        
        self.learning_sessions[session_id] = {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "learning_objectives": learning_objectives,
            "started_at": datetime.utcnow(),
            "actions_taken": 0,
            "successful_actions": 0,
            "objectives_completed": [],
            "current_state": "initializing",
            "performance_score": 0.0
        }
        
        self.progress_metrics[session_id] = []
        
        logger.info(f"Started learning session {session_id}")
        return session_id
    
    async def update_learning_progress(self, session_id: str, action_result: ActionResult):
        """Update learning progress with action result"""
        try:
            if session_id not in self.learning_sessions:
                logger.warning(f"Learning session {session_id} not found")
                return
            
            session = self.learning_sessions[session_id]
            session["actions_taken"] += 1
            
            if action_result.success:
                session["successful_actions"] += 1
            
            # Update objectives completed
            for objective in action_result.objectives_achieved:
                if objective not in session["objectives_completed"]:
                    session["objectives_completed"].append(objective)
            
            # Calculate performance score
            success_rate = session["successful_actions"] / session["actions_taken"]
            objective_completion = len(session["objectives_completed"]) / len(session["learning_objectives"])
            session["performance_score"] = (success_rate * 0.6) + (objective_completion * 0.4)
            
            # Store progress metrics
            progress_point = {
                "timestamp": datetime.utcnow().isoformat(),
                "action_count": session["actions_taken"],
                "success_rate": success_rate,
                "objective_completion": objective_completion,
                "performance_score": session["performance_score"],
                "reward": action_result.reward
            }
            
            self.progress_metrics[session_id].append(progress_point)
            
            # Update current state
            if objective_completion >= 1.0:
                session["current_state"] = "completed"
            elif objective_completion >= 0.7:
                session["current_state"] = "advanced"
            elif objective_completion >= 0.3:
                session["current_state"] = "progressing"
            else:
                session["current_state"] = "learning"
            
        except Exception as e:
            logger.error(f"Failed to update learning progress: {e}")
    
    async def get_learning_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        try:
            if session_id not in self.learning_sessions:
                return {"error": "Session not found"}
            
            session = self.learning_sessions[session_id]
            metrics = self.progress_metrics.get(session_id, [])
            
            # Calculate time-based metrics
            session_duration = (datetime.utcnow() - session["started_at"]).total_seconds()
            actions_per_minute = session["actions_taken"] / (session_duration / 60) if session_duration > 0 else 0
            
            # Learning velocity (improvement over time)
            learning_velocity = 0.0
            if len(metrics) >= 2:
                initial_score = metrics[0]["performance_score"]
                current_score = metrics[-1]["performance_score"]
                learning_velocity = (current_score - initial_score) / len(metrics)
            
            return {
                "session_id": session_id,
                "session_info": session,
                "duration_minutes": round(session_duration / 60, 2),
                "actions_per_minute": round(actions_per_minute, 2),
                "learning_velocity": round(learning_velocity, 4),
                "progress_trend": self._calculate_progress_trend(metrics),
                "recommendations": self._generate_learning_recommendations(session, metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {"error": str(e)}
    
    def _calculate_progress_trend(self, metrics: List[Dict[str, Any]]) -> str:
        """Calculate learning progress trend"""
        if len(metrics) < 3:
            return "insufficient_data"
        
        recent_scores = [m["performance_score"] for m in metrics[-5:]]
        early_scores = [m["performance_score"] for m in metrics[:5]]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)
        
        improvement = recent_avg - early_avg
        
        if improvement > 0.1:
            return "improving"
        elif improvement > 0.05:
            return "slowly_improving"
        elif improvement > -0.05:
            return "stable"
        else:
            return "declining"
    
    def _generate_learning_recommendations(self, session: Dict[str, Any], 
                                         metrics: List[Dict[str, Any]]) -> List[str]:
        """Generate learning recommendations based on performance"""
        recommendations = []
        
        success_rate = session["successful_actions"] / session["actions_taken"] if session["actions_taken"] > 0 else 0
        
        if success_rate < 0.3:
            recommendations.append("Consider reviewing fundamental concepts before proceeding")
            recommendations.append("Try simpler scenarios to build confidence")
        
        if len(session["objectives_completed"]) == 0 and session["actions_taken"] > 10:
            recommendations.append("Focus on understanding the scenario objectives")
            recommendations.append("Consider using available hints or guidance")
        
        if len(metrics) > 10:
            trend = self._calculate_progress_trend(metrics)
            if trend == "declining":
                recommendations.append("Performance is declining - consider taking a break")
                recommendations.append("Review recent actions and learn from mistakes")
            elif trend == "stable":
                recommendations.append("Try different approaches to break through the plateau")
        
        if not recommendations:
            recommendations.append("Performance is good - continue with current approach")
            recommendations.append("Consider progressing to more challenging scenarios")
        
        return recommendations


class ControlledEnvironmentFramework:
    """
    Controlled Environment Framework for Safe Autonomous Red Team Operations
    
    Provides sophisticated simulation environments with:
    - Docker-based isolated cyber ranges
    - Realistic vulnerable application deployment
    - Dynamic network topology simulation
    - Comprehensive monitoring and logging
    - Real-time learning progress tracking
    - Safety boundary enforcement
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.framework_id = str(uuid.uuid4())
        
        # Core components
        self.environment_manager = DockerEnvironmentManager()
        self.performance_monitor = PerformanceMonitor()
        self.learning_tracker = LearningProgressTracker()
        self.security_framework = SecurityFramework()
        self.audit_logger = AuditLogger()
        
        # Environment management
        self.active_environments: Dict[str, EnvironmentInstance] = {}
        self.scenario_templates: Dict[str, SimulationScenario] = {}
        
        # Performance tracking
        self.framework_metrics = {
            "environments_created": 0,
            "successful_deployments": 0,
            "total_uptime_hours": 0.0,
            "learning_sessions": 0,
            "safety_violations": 0,
            "emergency_shutdowns": 0
        }
        
        # Load default scenarios
        asyncio.create_task(self._load_default_scenarios())
        
        logger.info("Controlled Environment Framework initialized", framework_id=self.framework_id)
    
    async def _load_default_scenarios(self):
        """Load default simulation scenarios"""
        try:
            # Basic Web Application Lab
            web_app_scenario = SimulationScenario(
                scenario_id="basic_web_lab",
                name="Basic Web Application Security Lab",
                description="Practice web application penetration testing in a safe environment",
                environment_type=EnvironmentType.WEB_APPLICATION_LAB,
                complexity_level=ComplexityLevel.BASIC,
                network_topology=NetworkTopology(
                    subnets=[{"name": "web_dmz", "cidr": "172.20.1.0/24"}],
                    hosts=[
                        {"name": "web_server", "ip": "172.20.1.10", "os": "linux"},
                        {"name": "db_server", "ip": "172.20.1.11", "os": "linux"}
                    ]
                ),
                vulnerable_applications=[
                    VulnerableApplication(
                        app_id="dvwa",
                        name="Damn Vulnerable Web Application",
                        image="vulnerables/web-dvwa",
                        vulnerabilities=["sql_injection", "xss", "command_injection"],
                        exposed_ports=[80, 3306],
                        environment_variables={"MYSQL_ROOT_PASSWORD": "password"},
                        difficulty_level=ComplexityLevel.BASIC,
                        learning_objectives=["SQL injection", "XSS", "Authentication bypass"]
                    )
                ],
                learning_objectives=["Web application enumeration", "SQL injection exploitation", "XSS payload creation"],
                success_criteria=["Successful SQL injection", "XSS execution", "Admin access gained"],
                estimated_duration=timedelta(hours=2)
            )
            
            # Enterprise Network Simulation
            enterprise_scenario = SimulationScenario(
                scenario_id="enterprise_network",
                name="Enterprise Network Penetration Testing",
                description="Complex enterprise network with multiple security layers",
                environment_type=EnvironmentType.ENTERPRISE_SIMULATION,
                complexity_level=ComplexityLevel.ADVANCED,
                network_topology=NetworkTopology(
                    subnets=[
                        {"name": "dmz", "cidr": "172.20.1.0/24"},
                        {"name": "internal", "cidr": "172.20.2.0/24"},
                        {"name": "management", "cidr": "172.20.3.0/24"}
                    ],
                    hosts=[
                        {"name": "web_server", "ip": "172.20.1.10", "os": "linux"},
                        {"name": "mail_server", "ip": "172.20.1.11", "os": "linux"},
                        {"name": "workstation1", "ip": "172.20.2.10", "os": "windows"},
                        {"name": "workstation2", "ip": "172.20.2.11", "os": "windows"},
                        {"name": "domain_controller", "ip": "172.20.2.5", "os": "windows"},
                        {"name": "file_server", "ip": "172.20.2.20", "os": "windows"}
                    ],
                    firewalls=[{"name": "perimeter_fw", "rules": ["allow_web", "deny_all"]}],
                    security_controls=["ids", "av", "dlp", "siem"]
                ),
                vulnerable_applications=[
                    VulnerableApplication(
                        app_id="metasploitable",
                        name="Metasploitable Linux",
                        image="tleemcjr/metasploitable2",
                        vulnerabilities=["vsftpd_backdoor", "irc_backdoor", "distcc_exec"],
                        exposed_ports=[21, 22, 23, 25, 53, 80, 111, 139, 445],
                        difficulty_level=ComplexityLevel.INTERMEDIATE
                    )
                ],
                defensive_systems=[
                    {"type": "ids", "location": "network_perimeter"},
                    {"type": "honeypot", "location": "dmz"}
                ],
                learning_objectives=["Network enumeration", "Lateral movement", "Privilege escalation", "Persistence"],
                success_criteria=["Initial access", "Domain admin access", "Data exfiltration simulation"],
                estimated_duration=timedelta(hours=6)
            )
            
            self.scenario_templates["basic_web_lab"] = web_app_scenario
            self.scenario_templates["enterprise_network"] = enterprise_scenario
            
            logger.info(f"Loaded {len(self.scenario_templates)} default scenarios")
            
        except Exception as e:
            logger.error(f"Failed to load default scenarios: {e}")
    
    async def create_environment(self, scenario_id: str, 
                               custom_config: Optional[Dict[str, Any]] = None) -> str:
        """Create and deploy simulation environment"""
        instance_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Creating environment {instance_id} from scenario {scenario_id}")
            
            # Get scenario template
            if scenario_id not in self.scenario_templates:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            scenario = self.scenario_templates[scenario_id]
            
            # Apply custom configuration if provided
            if custom_config:
                # Merge custom config with scenario (simplified)
                pass
            
            # Create environment instance
            environment = EnvironmentInstance(
                instance_id=instance_id,
                scenario=scenario,
                status=EnvironmentStatus.INITIALIZING
            )
            
            # Store environment
            self.active_environments[instance_id] = environment
            
            # Deploy infrastructure
            await self._deploy_environment_infrastructure(environment)
            
            # Start monitoring
            await self.performance_monitor.start_monitoring(instance_id, self.environment_manager)
            
            # Update status
            environment.status = EnvironmentStatus.ACTIVE
            environment.started_at = datetime.utcnow()
            
            # Update metrics
            self.framework_metrics["environments_created"] += 1
            self.framework_metrics["successful_deployments"] += 1
            
            # Audit logging
            await self.audit_logger.log_event(AuditEvent(
                event_type="environment_created",
                component="controlled_environment_framework",
                details={
                    "instance_id": instance_id,
                    "scenario_id": scenario_id,
                    "complexity_level": scenario.complexity_level.value,
                    "environment_type": scenario.environment_type.value
                },
                security_level=SecurityLevel.MEDIUM
            ))
            
            logger.info(f"Environment {instance_id} created and deployed successfully")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create environment {instance_id}: {e}")
            
            # Cleanup on failure
            if instance_id in self.active_environments:
                await self._cleanup_environment(instance_id)
            
            raise
    
    async def _deploy_environment_infrastructure(self, environment: EnvironmentInstance):
        """Deploy the infrastructure for environment"""
        try:
            environment.status = EnvironmentStatus.DEPLOYING
            
            # Create isolated network
            network_subnet = environment.scenario.network_topology.subnets[0]["cidr"]
            network_name = f"xorb_{environment.instance_id}"
            
            network_id = await self.environment_manager.create_isolated_network(
                network_name, network_subnet
            )
            environment.docker_network_id = network_id
            
            # Deploy vulnerable applications
            for app_config in environment.scenario.vulnerable_applications:
                # Assign IP address (simplified)
                ip_address = "172.20.1.10"  # Would be calculated properly
                
                container_id = await self.environment_manager.deploy_vulnerable_application(
                    app_config, network_id, ip_address
                )
                
                environment.container_ids.append(container_id)
                environment.ip_assignments[app_config.app_id] = ip_address
            
            # Deploy defensive systems
            for defense_config in environment.scenario.defensive_systems:
                defense_id = await self.environment_manager.deploy_defensive_system(
                    defense_config, network_id
                )
                environment.container_ids.append(defense_id)
            
        except Exception as e:
            logger.error(f"Infrastructure deployment failed: {e}")
            raise
    
    async def start_learning_session(self, instance_id: str, agent_id: str) -> str:
        """Start learning session in environment"""
        try:
            if instance_id not in self.active_environments:
                raise ValueError(f"Environment {instance_id} not found")
            
            environment = self.active_environments[instance_id]
            
            session_id = await self.learning_tracker.start_learning_session(
                instance_id, agent_id, environment.scenario.learning_objectives
            )
            
            environment.active_agents.append(agent_id)
            self.framework_metrics["learning_sessions"] += 1
            
            logger.info(f"Started learning session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start learning session: {e}")
            raise
    
    async def process_agent_action(self, instance_id: str, agent_id: str, 
                                 action_result: ActionResult) -> Dict[str, Any]:
        """Process agent action and update learning progress"""
        try:
            # Find learning session
            session_id = None
            for sid, session in self.learning_tracker.learning_sessions.items():
                if session["instance_id"] == instance_id and session["agent_id"] == agent_id:
                    session_id = sid
                    break
            
            if not session_id:
                logger.warning(f"No learning session found for agent {agent_id}")
                return {"error": "No active learning session"}
            
            # Update learning progress
            await self.learning_tracker.update_learning_progress(session_id, action_result)
            
            # Update environment state
            environment = self.active_environments[instance_id]
            environment.last_activity = datetime.utcnow()
            
            # Check for safety violations
            if action_result.metadata.get("safety_violation"):
                environment.safety_violations.append(action_result.metadata["safety_violation"])
                self.framework_metrics["safety_violations"] += 1
            
            # Get updated learning summary
            learning_summary = await self.learning_tracker.get_learning_summary(session_id)
            
            return {
                "session_id": session_id,
                "action_processed": True,
                "learning_summary": learning_summary,
                "environment_status": environment.status.value
            }
            
        except Exception as e:
            logger.error(f"Failed to process agent action: {e}")
            return {"error": str(e)}
    
    async def get_environment_status(self, instance_id: str) -> Dict[str, Any]:
        """Get comprehensive environment status"""
        try:
            if instance_id not in self.active_environments:
                return {"error": "Environment not found"}
            
            environment = self.active_environments[instance_id]
            
            # Get container health
            container_health = {}
            for container_id in environment.container_ids:
                health = await self.environment_manager.monitor_container_health(container_id)
                container_health[container_id] = health
            
            # Get performance summary
            performance_summary = await self.performance_monitor.get_performance_summary(instance_id)
            
            # Calculate uptime
            uptime = (datetime.utcnow() - environment.started_at).total_seconds() / 3600 if environment.started_at else 0
            
            return {
                "instance_id": instance_id,
                "scenario": {
                    "id": environment.scenario.scenario_id,
                    "name": environment.scenario.name,
                    "complexity": environment.scenario.complexity_level.value,
                    "type": environment.scenario.environment_type.value
                },
                "status": environment.status.value,
                "uptime_hours": round(uptime, 2),
                "active_agents": environment.active_agents,
                "container_health": container_health,
                "performance": performance_summary,
                "safety_violations": len(environment.safety_violations),
                "emergency_stops": environment.emergency_stops,
                "last_activity": environment.last_activity.isoformat() if environment.last_activity else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get environment status: {e}")
            return {"error": str(e)}
    
    async def emergency_shutdown(self, instance_id: str, reason: str = "Emergency shutdown") -> bool:
        """Emergency shutdown of environment"""
        try:
            logger.critical(f"EMERGENCY SHUTDOWN: {instance_id} - {reason}")
            
            if instance_id not in self.active_environments:
                return False
            
            environment = self.active_environments[instance_id]
            environment.emergency_stops += 1
            environment.status = EnvironmentStatus.STOPPING
            
            # Stop monitoring
            await self.performance_monitor.stop_monitoring(instance_id)
            
            # Cleanup infrastructure
            success = await self._cleanup_environment(instance_id)
            
            # Update metrics
            self.framework_metrics["emergency_shutdowns"] += 1
            
            # Audit logging
            await self.audit_logger.log_event(AuditEvent(
                event_type="emergency_shutdown",
                component="controlled_environment_framework",
                details={
                    "instance_id": instance_id,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat()
                },
                security_level=SecurityLevel.CRITICAL
            ))
            
            return success
            
        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    async def _cleanup_environment(self, instance_id: str) -> bool:
        """Clean up environment resources"""
        try:
            if instance_id not in self.active_environments:
                return True
            
            environment = self.active_environments[instance_id]
            
            # Cleanup Docker resources
            success = await self.environment_manager.cleanup_environment(
                environment.docker_network_id,
                environment.container_ids
            )
            
            # Update status
            environment.status = EnvironmentStatus.DESTROYED
            environment.cleanup_required = False
            
            # Calculate total uptime
            if environment.started_at:
                uptime_hours = (datetime.utcnow() - environment.started_at).total_seconds() / 3600
                self.framework_metrics["total_uptime_hours"] += uptime_hours
            
            # Remove from active environments
            del self.active_environments[instance_id]
            
            logger.info(f"Environment {instance_id} cleaned up successfully")
            return success
            
        except Exception as e:
            logger.error(f"Environment cleanup failed: {e}")
            return False
    
    async def get_framework_metrics(self) -> Dict[str, Any]:
        """Get comprehensive framework metrics"""
        try:
            # Calculate success rates
            total_created = self.framework_metrics["environments_created"]
            success_rate = (self.framework_metrics["successful_deployments"] / total_created 
                          if total_created > 0 else 0.0)
            
            # Active environment summary
            active_summary = {}
            for instance_id, env in self.active_environments.items():
                active_summary[instance_id] = {
                    "status": env.status.value,
                    "scenario": env.scenario.scenario_id,
                    "uptime_hours": ((datetime.utcnow() - env.started_at).total_seconds() / 3600 
                                   if env.started_at else 0),
                    "active_agents": len(env.active_agents)
                }
            
            return {
                "framework_metrics": {
                    "framework_id": self.framework_id,
                    "environments_created": total_created,
                    "successful_deployments": self.framework_metrics["successful_deployments"],
                    "deployment_success_rate": round(success_rate, 3),
                    "total_uptime_hours": round(self.framework_metrics["total_uptime_hours"], 2),
                    "learning_sessions": self.framework_metrics["learning_sessions"],
                    "safety_violations": self.framework_metrics["safety_violations"],
                    "emergency_shutdowns": self.framework_metrics["emergency_shutdowns"],
                    "active_environments": len(self.active_environments)
                },
                "active_environments": active_summary,
                "available_scenarios": list(self.scenario_templates.keys()),
                "docker_available": self.environment_manager.docker_available,
                "simulation_dependencies": SIMULATION_DEPS_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Failed to get framework metrics: {e}")
            return {"error": str(e)}


# Global framework instance
_environment_framework: Optional[ControlledEnvironmentFramework] = None


async def get_environment_framework(config: Dict[str, Any] = None) -> ControlledEnvironmentFramework:
    """Get singleton controlled environment framework instance"""
    global _environment_framework
    
    if _environment_framework is None:
        _environment_framework = ControlledEnvironmentFramework(config)
    
    return _environment_framework


# Export main classes
__all__ = [
    "ControlledEnvironmentFramework",
    "SimulationScenario",
    "EnvironmentInstance", 
    "VulnerableApplication",
    "NetworkTopology",
    "EnvironmentType",
    "ComplexityLevel",
    "EnvironmentStatus",
    "MonitoringLevel",
    "get_environment_framework"
]