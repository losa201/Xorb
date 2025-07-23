#!/usr/bin/env python3

import asyncio
import logging
import psutil
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logging.warning("Docker client not available. Container management disabled.")


class DeploymentMode(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ResourcePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemResources:
    cpu_count: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_usage_percent: float
    network_io_mbps: float = 0.0
    load_average_1min: float = 0.0
    process_count: int = 0


@dataclass
class ComponentConfig:
    name: str
    enabled: bool = True
    cpu_limit_percent: float = 25.0
    memory_limit_mb: int = 512
    priority: ResourcePriority = ResourcePriority.MEDIUM
    auto_scale: bool = False
    min_instances: int = 1
    max_instances: int = 3
    health_check_interval: int = 30
    restart_policy: str = "on-failure"


class SystemMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.current_resources = SystemResources(
            cpu_count=psutil.cpu_count(),
            cpu_usage_percent=0.0,
            memory_total_gb=0.0,
            memory_available_gb=0.0,
            memory_usage_percent=0.0,
            disk_total_gb=0.0,
            disk_free_gb=0.0,
            disk_usage_percent=0.0
        )
        
        self.history: List[Tuple[datetime, SystemResources]] = []
        self.monitoring = False
        self.monitor_task = None
        
        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("System monitoring started")

    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Update current resource usage
                await self._update_resources()
                
                # Store history
                self.history.append((datetime.utcnow(), self.current_resources))
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.history = [(ts, res) for ts, res in self.history if ts > cutoff_time]
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.update_interval)

    async def _update_resources(self):
        """Update current resource measurements"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)  # GB
            memory_available = memory.available / (1024**3)  # GB
            memory_usage = memory.percent
            
            # Disk usage (root filesystem)
            disk = psutil.disk_usage('/')
            disk_total = disk.total / (1024**3)  # GB
            disk_free = disk.free / (1024**3)  # GB
            disk_usage = (disk.used / disk.total) * 100
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0]
            except AttributeError:
                load_avg = 0.0  # Windows doesn't have load average
            
            # Process count
            process_count = len(psutil.pids())
            
            self.current_resources = SystemResources(
                cpu_count=psutil.cpu_count(),
                cpu_usage_percent=cpu_usage,
                memory_total_gb=memory_total,
                memory_available_gb=memory_available,
                memory_usage_percent=memory_usage,
                disk_total_gb=disk_total,
                disk_free_gb=disk_free,
                disk_usage_percent=disk_usage,
                load_average_1min=load_avg,
                process_count=process_count
            )
            
        except Exception as e:
            self.logger.error(f"Resource update failed: {e}")

    def get_resource_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage trends over specified period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_data = [(ts, res) for ts, res in self.history if ts > cutoff_time]
        
        if len(recent_data) < 2:
            return {"error": "Insufficient data for trends"}
        
        # Calculate averages and trends
        cpu_values = [res.cpu_usage_percent for _, res in recent_data]
        memory_values = [res.memory_usage_percent for _, res in recent_data]
        disk_values = [res.disk_usage_percent for _, res in recent_data]
        
        return {
            "period_hours": hours,
            "data_points": len(recent_data),
            "cpu": {
                "current": cpu_values[-1],
                "average": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "trend": "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
            },
            "memory": {
                "current": memory_values[-1],
                "average": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "trend": "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            },
            "disk": {
                "current": disk_values[-1],
                "average": sum(disk_values) / len(disk_values),
                "min": min(disk_values),
                "max": max(disk_values),
                "trend": "increasing" if disk_values[-1] > disk_values[0] else "decreasing"
            }
        }


class ProcessManager:
    """Manages XORB component processes with resource limits"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.processes: Dict[str, psutil.Process] = {}
        self.component_configs: Dict[str, ComponentConfig] = {}
        self.process_stats: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger = logging.getLogger(__name__)

    def register_component(self, config: ComponentConfig):
        """Register a component for management"""
        self.component_configs[config.name] = config
        self.process_stats[config.name] = []
        self.logger.info(f"Registered component: {config.name}")

    async def start_component(self, component_name: str) -> bool:
        """Start a component process"""
        config = self.component_configs.get(component_name)
        if not config or not config.enabled:
            return False
        
        try:
            # Component startup logic would go here
            # For demo, we'll simulate process management
            self.logger.info(f"Starting component: {component_name}")
            
            # In production, this would start the actual process
            # For now, track the Python process itself as a demo
            current_process = psutil.Process()
            self.processes[component_name] = current_process
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start component {component_name}: {e}")
            return False

    async def stop_component(self, component_name: str) -> bool:
        """Stop a component process"""
        if component_name not in self.processes:
            return False
        
        try:
            process = self.processes[component_name]
            
            # Graceful shutdown attempt
            if process.is_running():
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    # Force kill if necessary
                    process.kill()
                    process.wait(timeout=5)
            
            del self.processes[component_name]
            self.logger.info(f"Stopped component: {component_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop component {component_name}: {e}")
            return False

    async def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """Check health of a component"""
        if component_name not in self.processes:
            return {"status": "not_running", "healthy": False}
        
        try:
            process = self.processes[component_name]
            config = self.component_configs[component_name]
            
            if not process.is_running():
                return {"status": "stopped", "healthy": False}
            
            # Get process stats
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Check resource limits
            cpu_ok = cpu_percent <= config.cpu_limit_percent
            memory_ok = memory_mb <= config.memory_limit_mb
            
            status = {
                "status": "running",
                "healthy": cpu_ok and memory_ok,
                "pid": process.pid,
                "cpu_percent": cpu_percent,
                "cpu_limit": config.cpu_limit_percent,
                "cpu_ok": cpu_ok,
                "memory_mb": memory_mb,
                "memory_limit_mb": config.memory_limit_mb,
                "memory_ok": memory_ok,
                "create_time": datetime.fromtimestamp(process.create_time()),
                "num_threads": process.num_threads()
            }
            
            # Store stats for trending
            self.process_stats[component_name].append({
                "timestamp": datetime.utcnow(),
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb
            })
            
            # Keep only recent stats
            if len(self.process_stats[component_name]) > 100:
                self.process_stats[component_name] = self.process_stats[component_name][-100:]
            
            return status
            
        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}

    async def restart_unhealthy_components(self):
        """Restart components that are not healthy"""
        for component_name in self.component_configs:
            health = await self.check_component_health(component_name)
            
            if not health["healthy"] and health["status"] == "running":
                self.logger.warning(f"Restarting unhealthy component: {component_name}")
                await self.stop_component(component_name)
                await asyncio.sleep(2)
                await self.start_component(component_name)


class ResourceOptimizer:
    """Optimizes resource allocation based on system capacity and workload"""
    
    def __init__(self, monitor: SystemMonitor, process_manager: ProcessManager):
        self.monitor = monitor
        self.process_manager = process_manager
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)

    async def optimize_deployment(self, deployment_mode: DeploymentMode) -> Dict[str, Any]:
        """Optimize deployment configuration based on system resources"""
        resources = self.monitor.current_resources
        
        # Get optimization recommendations based on available resources
        recommendations = await self._calculate_optimizations(resources, deployment_mode)
        
        # Apply optimizations
        applied_changes = await self._apply_optimizations(recommendations)
        
        # Record optimization
        optimization_record = {
            "timestamp": datetime.utcnow(),
            "deployment_mode": deployment_mode.value,
            "resources_at_optimization": resources.__dict__,
            "recommendations": recommendations,
            "applied_changes": applied_changes
        }
        
        self.optimization_history.append(optimization_record)
        
        return optimization_record

    async def _calculate_optimizations(self, 
                                     resources: SystemResources, 
                                     mode: DeploymentMode) -> Dict[str, Any]:
        """Calculate optimization recommendations"""
        
        recommendations = {
            "memory_optimizations": [],
            "cpu_optimizations": [],
            "component_adjustments": [],
            "scaling_recommendations": []
        }
        
        # Memory optimizations
        if resources.memory_usage_percent > 80:
            recommendations["memory_optimizations"].extend([
                "Reduce Redis cache size",
                "Implement memory-mapped file storage for knowledge atoms",
                "Enable aggressive garbage collection",
                "Reduce concurrent agent limit"
            ])
        
        # CPU optimizations
        if resources.cpu_usage_percent > 75:
            recommendations["cpu_optimizations"].extend([
                "Reduce ML model complexity",
                "Implement request queuing",
                "Optimize database queries",
                "Use CPU-friendly algorithms"
            ])
        
        # Component adjustments based on mode
        mode_adjustments = {
            DeploymentMode.DEVELOPMENT: {
                "enable_debug_logging": True,
                "reduce_security_checks": True,
                "increase_cache_ttl": True
            },
            DeploymentMode.PRODUCTION: {
                "enable_debug_logging": False,
                "enable_full_security": True,
                "optimize_for_stability": True,
                "enable_monitoring": True
            }
        }
        
        recommendations["component_adjustments"] = mode_adjustments.get(mode, {})
        
        # Scaling recommendations
        if resources.cpu_usage_percent > 70 or resources.memory_usage_percent > 70:
            recommendations["scaling_recommendations"].append(
                "Consider vertical scaling (more CPU/RAM)"
            )
        
        if mode == DeploymentMode.PRODUCTION and resources.load_average_1min > resources.cpu_count:
            recommendations["scaling_recommendations"].append(
                "Consider horizontal scaling (multiple instances)"
            )
        
        return recommendations

    async def _apply_optimizations(self, recommendations: Dict[str, Any]) -> List[str]:
        """Apply optimization recommendations"""
        applied = []
        
        # Apply memory optimizations
        for optimization in recommendations["memory_optimizations"]:
            if "Reduce concurrent agent limit" in optimization:
                # Simulate reducing agent limit
                applied.append("Reduced max concurrent agents to 3")
        
        # Apply component adjustments
        adjustments = recommendations["component_adjustments"]
        for component_name, config in self.process_manager.component_configs.items():
            if adjustments.get("optimize_for_stability"):
                config.restart_policy = "always"
                config.health_check_interval = 15  # More frequent health checks
                applied.append(f"Optimized {component_name} for stability")
        
        return applied

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations without applying them"""
        resources = self.monitor.current_resources
        
        recommendations = {
            "immediate_actions": [],
            "performance_improvements": [],
            "resource_warnings": []
        }
        
        # Immediate actions needed
        if resources.memory_usage_percent > 90:
            recommendations["immediate_actions"].append("Critical: Memory usage above 90%")
        
        if resources.disk_usage_percent > 90:
            recommendations["immediate_actions"].append("Critical: Disk usage above 90%")
        
        # Performance improvements
        if resources.cpu_usage_percent > 60:
            recommendations["performance_improvements"].append(
                "Consider optimizing CPU-intensive operations"
            )
        
        # Resource warnings
        if resources.memory_available_gb < 1.0:
            recommendations["resource_warnings"].append(
                "Warning: Less than 1GB memory available"
            )
        
        return recommendations


class XORBDeploymentOptimizer:
    """Main deployment optimizer orchestrating all optimization components"""
    
    def __init__(self, deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION):
        self.deployment_mode = deployment_mode
        
        # Initialize components
        self.system_monitor = SystemMonitor()
        self.process_manager = ProcessManager(self.system_monitor)
        self.resource_optimizer = ResourceOptimizer(self.system_monitor, self.process_manager)
        
        # Configuration
        self.auto_optimization_enabled = True
        self.optimization_interval_minutes = 15
        self.optimization_task = None
        
        self.logger = logging.getLogger(__name__)
        
        # Register default XORB components
        self._register_default_components()

    def _register_default_components(self):
        """Register standard XORB components"""
        components = [
            ComponentConfig(
                name="orchestrator",
                cpu_limit_percent=30.0,
                memory_limit_mb=1024,
                priority=ResourcePriority.CRITICAL,
                auto_scale=False,
                health_check_interval=30
            ),
            ComponentConfig(
                name="knowledge_fabric",
                cpu_limit_percent=20.0,
                memory_limit_mb=2048,
                priority=ResourcePriority.HIGH,
                auto_scale=True,
                max_instances=2
            ),
            ComponentConfig(
                name="agent_manager",
                cpu_limit_percent=40.0,
                memory_limit_mb=512,
                priority=ResourcePriority.HIGH,
                auto_scale=True,
                max_instances=3
            ),
            ComponentConfig(
                name="llm_client",
                cpu_limit_percent=15.0,
                memory_limit_mb=256,
                priority=ResourcePriority.MEDIUM,
                auto_scale=False
            ),
            ComponentConfig(
                name="report_generator",
                cpu_limit_percent=10.0,
                memory_limit_mb=256,
                priority=ResourcePriority.LOW,
                auto_scale=False
            ),
            ComponentConfig(
                name="monitoring_dashboard",
                cpu_limit_percent=5.0,
                memory_limit_mb=128,
                priority=ResourcePriority.LOW,
                auto_scale=False
            )
        ]
        
        for component in components:
            self.process_manager.register_component(component)

    async def start_optimization(self):
        """Start the deployment optimizer"""
        self.logger.info(f"Starting deployment optimizer in {self.deployment_mode.value} mode")
        
        # Start system monitoring
        await self.system_monitor.start_monitoring()
        
        # Start all enabled components
        for component_name in self.process_manager.component_configs:
            config = self.process_manager.component_configs[component_name]
            if config.enabled:
                await self.process_manager.start_component(component_name)
        
        # Start auto-optimization if enabled
        if self.auto_optimization_enabled:
            self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Deployment optimizer started")

    async def stop_optimization(self):
        """Stop the deployment optimizer"""
        self.logger.info("Stopping deployment optimizer")
        
        # Stop auto-optimization
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        # Stop all components
        for component_name in list(self.process_manager.processes.keys()):
            await self.process_manager.stop_component(component_name)
        
        # Stop system monitoring
        await self.system_monitor.stop_monitoring()
        
        self.logger.info("Deployment optimizer stopped")

    async def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            try:
                # Perform optimization
                result = await self.resource_optimizer.optimize_deployment(self.deployment_mode)
                
                if result["applied_changes"]:
                    self.logger.info(f"Applied optimizations: {result['applied_changes']}")
                
                # Check component health and restart if needed
                await self.process_manager.restart_unhealthy_components()
                
                # Wait for next optimization cycle
                await asyncio.sleep(self.optimization_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        
        # Get system resources
        resources = self.system_monitor.current_resources
        trends = self.system_monitor.get_resource_trends(hours=2)
        
        # Get component health
        component_health = {}
        for component_name in self.process_manager.component_configs:
            component_health[component_name] = await self.process_manager.check_component_health(component_name)
        
        # Get optimization recommendations
        recommendations = self.resource_optimizer.get_optimization_recommendations()
        
        return {
            "deployment_mode": self.deployment_mode.value,
            "system_resources": {
                "cpu_usage": resources.cpu_usage_percent,
                "memory_usage": resources.memory_usage_percent,
                "memory_available_gb": resources.memory_available_gb,
                "disk_usage": resources.disk_usage_percent,
                "load_average": resources.load_average_1min,
                "process_count": resources.process_count
            },
            "resource_trends": trends,
            "component_health": component_health,
            "optimization_recommendations": recommendations,
            "optimization_history_count": len(self.resource_optimizer.optimization_history),
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "monitoring_active": self.system_monitor.monitoring
        }

    async def manual_optimize(self) -> Dict[str, Any]:
        """Trigger manual optimization"""
        self.logger.info("Manual optimization triggered")
        return await self.resource_optimizer.optimize_deployment(self.deployment_mode)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_optimizer():
        """Demonstrate deployment optimizer"""
        print("=== XORB Deployment Optimizer Demo ===")
        
        optimizer = XORBDeploymentOptimizer(DeploymentMode.PRODUCTION)
        
        try:
            # Start optimization
            await optimizer.start_optimization()
            
            # Let it run for a bit
            await asyncio.sleep(10)
            
            # Show status
            status = await optimizer.get_deployment_status()
            print(f"Deployment Status:\n{json.dumps(status, indent=2, default=str)}")
            
            # Trigger manual optimization
            optimization_result = await optimizer.manual_optimize()
            print(f"Manual Optimization:\n{json.dumps(optimization_result, indent=2, default=str)}")
            
            # Wait a bit more
            print("Running optimizer for 30 seconds...")
            await asyncio.sleep(30)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await optimizer.stop_optimization()
    
    if "--demo" in sys.argv:
        asyncio.run(demo_optimizer())
    else:
        print("XORB Deployment Optimizer")
        print("Usage:")
        print("  python optimizer.py --demo    # Run demo")
        print("")
        print("Features:")
        print("  - Real-time system monitoring")
        print("  - Automatic resource optimization") 
        print("  - Component health monitoring")
        print("  - Intelligent process management")
        print("  - Deployment mode configurations")