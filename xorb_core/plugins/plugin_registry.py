#!/usr/bin/env python3
"""
XORB Plugin Registry - Composable Architecture System

Provides a plugin registry pattern for Phase 11 composability, allowing new mission 
strategies, agent types, and orchestration components to be plugged in without core 
modification. Optimized for Raspberry Pi 5 deployment.
"""

import asyncio
import importlib
import inspect
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog


class PluginType(Enum):
    """Types of plugins supported by XORB"""
    AGENT = "agent"
    MISSION_STRATEGY = "mission_strategy"
    PATTERN_DETECTOR = "pattern_detector"
    KPI_TRACKER = "kpi_tracker"
    CONFLICT_RESOLVER = "conflict_resolver"
    SIGNAL_PROCESSOR = "signal_processor"
    ORCHESTRATION_ENGINE = "orchestration_engine"
    ADAPTATION_STRATEGY = "adaptation_strategy"


class PluginStatus(Enum):
    """Plugin status states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata and registration information"""
    plugin_id: str
    name: str
    version: str
    plugin_type: PluginType
    
    # Plugin details
    description: str
    author: str
    homepage: Optional[str] = None
    
    # Dependencies and requirements
    dependencies: List[str] = field(default_factory=list)
    xorb_version_min: str = "11.0"
    python_version_min: str = "3.8"
    
    # Plugin capabilities
    capabilities: Set[str] = field(default_factory=set)
    config_schema: Optional[Dict[str, Any]] = None
    
    # Registration metadata
    registered_at: datetime = field(default_factory=datetime.now)
    status: PluginStatus = PluginStatus.UNLOADED
    load_priority: int = 50  # 0-100, higher = loaded earlier
    
    # Runtime information
    plugin_class: Optional[Type] = None
    plugin_instance: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Performance tracking
    load_time_ms: float = 0.0
    last_activity: Optional[datetime] = None


class PluginInterface(ABC):
    """Base interface for all XORB plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the plugin gracefully"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health status"""
        pass
    
    def get_capabilities(self) -> Set[str]:
        """Get plugin capabilities"""
        return self.metadata.capabilities
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure plugin at runtime"""
        return True


class AgentPlugin(PluginInterface):
    """Base class for agent plugins"""
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent task"""
        pass
    
    @abstractmethod
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status"""
        pass


class MissionStrategyPlugin(PluginInterface):
    """Base class for mission strategy plugins"""
    
    @abstractmethod
    async def plan_mission(self, objectives: List[Dict[str, Any]], 
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a mission using this strategy"""
        pass
    
    @abstractmethod
    async def adapt_mission(self, mission_plan: Dict[str, Any], 
                           adaptation_trigger: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt mission based on trigger"""
        pass


class SignalProcessorPlugin(PluginInterface):
    """Base class for signal processor plugins"""
    
    @abstractmethod
    async def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process a threat signal"""
        pass
    
    @abstractmethod
    async def detect_patterns(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in signals"""
        pass


class PluginRegistry:
    """
    Plugin Registry for XORB Composable Architecture
    
    Manages loading, registration, and lifecycle of plugins.
    Optimized for Raspberry Pi 5 with minimal overhead.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("xorb.plugin_registry")
        
        # Plugin storage
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugins_by_type: Dict[PluginType, List[PluginMetadata]] = {}
        self.active_plugins: Dict[str, Any] = {}
        
        # Plugin discovery
        self.plugin_directories: List[Path] = []
        self.auto_discovery_enabled = True
        
        # Performance optimization for Pi 5
        self.max_concurrent_loads = 3
        self.load_timeout_seconds = 30
        self.health_check_interval = 300  # 5 minutes
        
        # Plugin hooks and events
        self.event_hooks: Dict[str, List[Callable]] = {}
        self.plugin_dependencies: Dict[str, Set[str]] = {}
        
        # Initialize plugin directories
        self._initialize_plugin_directories()
    
    def _initialize_plugin_directories(self):
        """Initialize default plugin directories"""
        base_dir = Path(__file__).parent.parent.parent
        
        # Standard plugin directories
        plugin_dirs = [
            base_dir / "plugins",
            base_dir / "xorb_core" / "plugins",
            Path.home() / ".xorb" / "plugins",
            Path("/opt/xorb/plugins"),
        ]
        
        for plugin_dir in plugin_dirs:
            if plugin_dir.exists() or plugin_dir == base_dir / "plugins":
                self.plugin_directories.append(plugin_dir)
        
        self.logger.info(f"Initialized {len(self.plugin_directories)} plugin directories")
    
    async def discover_plugins(self) -> Dict[str, List[str]]:
        """Discover plugins in configured directories"""
        discovered = {"found": [], "errors": []}
        
        try:
            for plugin_dir in self.plugin_directories:
                if not plugin_dir.exists():
                    continue
                
                # Look for Python files and packages
                for plugin_path in plugin_dir.rglob("*.py"):
                    if plugin_path.name.startswith("__"):
                        continue
                    
                    try:
                        await self._discover_plugin_file(plugin_path)
                        discovered["found"].append(str(plugin_path))
                    except Exception as e:
                        error_msg = f"Failed to discover {plugin_path}: {str(e)}"
                        discovered["errors"].append(error_msg)
                        self.logger.warning(error_msg)
            
            self.logger.info(f"Plugin discovery completed",
                           found=len(discovered["found"]),
                           errors=len(discovered["errors"]))
            
            return discovered
            
        except Exception as e:
            self.logger.error("Plugin discovery failed", error=str(e))
            return {"found": [], "errors": [str(e)]}
    
    async def _discover_plugin_file(self, plugin_path: Path):
        """Discover plugins in a single file"""
        try:
            # Convert path to module name
            relative_path = plugin_path.relative_to(plugin_path.parts[0])
            module_name = str(relative_path.with_suffix("")).replace("/", ".")
            
            # Import module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if not spec or not spec.loader:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and
                    hasattr(obj, 'metadata')):
                    
                    # Create plugin metadata
                    try:
                        plugin_instance = obj()
                        metadata = plugin_instance.metadata
                        metadata.plugin_class = obj
                        
                        await self.register_plugin(metadata)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to instantiate plugin {name}", error=str(e))
            
        except Exception as e:
            self.logger.warning(f"Failed to discover plugin file {plugin_path}", error=str(e))
    
    async def register_plugin(self, metadata: PluginMetadata) -> bool:
        """Register a plugin with the registry"""
        try:
            # Validate plugin
            validation_result = await self._validate_plugin(metadata)
            if not validation_result["valid"]:
                self.logger.error(f"Plugin validation failed: {metadata.plugin_id}",
                                error=validation_result["error"])
                return False
            
            # Check for conflicts
            if metadata.plugin_id in self.plugins:
                existing = self.plugins[metadata.plugin_id]
                if existing.version >= metadata.version:
                    self.logger.warning(f"Plugin {metadata.plugin_id} already registered with same or newer version")
                    return False
            
            # Register plugin
            self.plugins[metadata.plugin_id] = metadata
            
            # Add to type index
            if metadata.plugin_type not in self.plugins_by_type:
                self.plugins_by_type[metadata.plugin_type] = []
            self.plugins_by_type[metadata.plugin_type].append(metadata)
            
            # Track dependencies
            if metadata.dependencies:
                self.plugin_dependencies[metadata.plugin_id] = set(metadata.dependencies)
            
            self.logger.info(f"Registered plugin: {metadata.plugin_id} v{metadata.version}",
                           plugin_type=metadata.plugin_type.value,
                           capabilities=len(metadata.capabilities))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin registration failed: {metadata.plugin_id}", error=str(e))
            return False
    
    async def load_plugin(self, plugin_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load and initialize a specific plugin"""
        try:
            if plugin_id not in self.plugins:
                self.logger.error(f"Plugin not registered: {plugin_id}")
                return False
            
            metadata = self.plugins[plugin_id]
            
            if metadata.status == PluginStatus.ACTIVE:
                self.logger.warning(f"Plugin already active: {plugin_id}")
                return True
            
            # Update status
            metadata.status = PluginStatus.LOADING
            load_start = asyncio.get_event_loop().time()
            
            # Check dependencies
            deps_loaded = await self._ensure_dependencies_loaded(plugin_id)
            if not deps_loaded:
                metadata.status = PluginStatus.ERROR
                metadata.error_message = "Dependencies not available"
                return False
            
            # Create plugin instance
            if not metadata.plugin_class:
                metadata.status = PluginStatus.ERROR
                metadata.error_message = "Plugin class not available"
                return False
            
            plugin_instance = metadata.plugin_class()
            
            # Initialize plugin
            config = config or {}
            init_success = await asyncio.wait_for(
                plugin_instance.initialize(config),
                timeout=self.load_timeout_seconds
            )
            
            if not init_success:
                metadata.status = PluginStatus.ERROR
                metadata.error_message = "Initialization failed"
                return False
            
            # Store instance and update metadata
            metadata.plugin_instance = plugin_instance
            metadata.status = PluginStatus.ACTIVE
            metadata.load_time_ms = (asyncio.get_event_loop().time() - load_start) * 1000
            metadata.last_activity = datetime.now()
            
            self.active_plugins[plugin_id] = plugin_instance
            
            # Trigger plugin loaded event
            await self._trigger_event("plugin_loaded", {"plugin_id": plugin_id, "metadata": metadata})
            
            self.logger.info(f"Loaded plugin: {plugin_id}",
                           load_time_ms=metadata.load_time_ms,
                           capabilities=len(metadata.capabilities))
            
            return True
            
        except asyncio.TimeoutError:
            metadata.status = PluginStatus.ERROR
            metadata.error_message = f"Load timeout after {self.load_timeout_seconds}s"
            self.logger.error(f"Plugin load timeout: {plugin_id}")
            return False
        except Exception as e:
            metadata.status = PluginStatus.ERROR
            metadata.error_message = str(e)
            self.logger.error(f"Plugin load failed: {plugin_id}", error=str(e))
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin gracefully"""
        try:
            if plugin_id not in self.plugins:
                return False
            
            metadata = self.plugins[plugin_id]
            
            if metadata.status != PluginStatus.ACTIVE:
                return True
            
            # Shutdown plugin
            if metadata.plugin_instance:
                await metadata.plugin_instance.shutdown()
            
            # Remove from active plugins
            if plugin_id in self.active_plugins:
                del self.active_plugins[plugin_id]
            
            # Update metadata
            metadata.status = PluginStatus.LOADED
            metadata.plugin_instance = None
            
            # Trigger plugin unloaded event
            await self._trigger_event("plugin_unloaded", {"plugin_id": plugin_id})
            
            self.logger.info(f"Unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin unload failed: {plugin_id}", error=str(e))
            return False
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """Get all plugins of a specific type"""
        return self.plugins_by_type.get(plugin_type, [])
    
    async def get_active_plugins_by_capability(self, capability: str) -> List[Any]:
        """Get active plugins with a specific capability"""
        matching_plugins = []
        
        for plugin_id, plugin_instance in self.active_plugins.items():
            metadata = self.plugins[plugin_id]
            if capability in metadata.capabilities:
                matching_plugins.append(plugin_instance)
        
        return matching_plugins
    
    async def execute_plugin_method(self, plugin_id: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on a specific plugin"""
        try:
            if plugin_id not in self.active_plugins:
                raise ValueError(f"Plugin not active: {plugin_id}")
            
            plugin_instance = self.active_plugins[plugin_id]
            
            if not hasattr(plugin_instance, method_name):
                raise AttributeError(f"Plugin {plugin_id} does not have method {method_name}")
            
            method = getattr(plugin_instance, method_name)
            
            # Update last activity
            self.plugins[plugin_id].last_activity = datetime.now()
            
            # Execute method
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Plugin method execution failed: {plugin_id}.{method_name}", error=str(e))
            raise
    
    async def health_check_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all active plugins"""
        health_results = {}
        
        for plugin_id, plugin_instance in self.active_plugins.items():
            try:
                health_result = await plugin_instance.health_check()
                health_results[plugin_id] = {
                    "status": "healthy",
                    "details": health_result
                }
            except Exception as e:
                health_results[plugin_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                self.logger.warning(f"Plugin health check failed: {plugin_id}", error=str(e))
        
        return health_results
    
    def register_event_hook(self, event_name: str, callback: Callable):
        """Register a callback for plugin events"""
        if event_name not in self.event_hooks:
            self.event_hooks[event_name] = []
        self.event_hooks[event_name].append(callback)
    
    async def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin registry status"""
        total_plugins = len(self.plugins)
        active_plugins = len(self.active_plugins)
        
        status_counts = {}
        for metadata in self.plugins.values():
            status = metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        type_counts = {}
        for plugin_type, plugins in self.plugins_by_type.items():
            type_counts[plugin_type.value] = len(plugins)
        
        return {
            "total_plugins": total_plugins,
            "active_plugins": active_plugins,
            "plugin_directories": [str(d) for d in self.plugin_directories],
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "auto_discovery_enabled": self.auto_discovery_enabled,
            "performance": {
                "max_concurrent_loads": self.max_concurrent_loads,
                "load_timeout_seconds": self.load_timeout_seconds,
                "health_check_interval": self.health_check_interval
            }
        }
    
    # Private helper methods
    
    async def _validate_plugin(self, metadata: PluginMetadata) -> Dict[str, Any]:
        """Validate plugin metadata and requirements"""
        try:
            # Basic validation
            if not metadata.plugin_id or not metadata.name:
                return {"valid": False, "error": "Missing required fields"}
            
            if not metadata.plugin_class:
                return {"valid": False, "error": "Plugin class not specified"}
            
            # Check if plugin class implements required interface
            if not issubclass(metadata.plugin_class, PluginInterface):
                return {"valid": False, "error": "Plugin class must implement PluginInterface"}
            
            # Validate version requirements
            # (In a real implementation, you'd check Python and XORB versions)
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    async def _ensure_dependencies_loaded(self, plugin_id: str) -> bool:
        """Ensure all plugin dependencies are loaded"""
        dependencies = self.plugin_dependencies.get(plugin_id, set())
        
        for dep_id in dependencies:
            if dep_id not in self.active_plugins:
                # Try to load dependency
                load_success = await self.load_plugin(dep_id)
                if not load_success:
                    self.logger.error(f"Failed to load dependency {dep_id} for plugin {plugin_id}")
                    return False
        
        return True
    
    async def _trigger_event(self, event_name: str, event_data: Dict[str, Any]):
        """Trigger plugin event hooks"""
        hooks = self.event_hooks.get(event_name, [])
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(event_data)
                else:
                    hook(event_data)
            except Exception as e:
                self.logger.warning(f"Event hook failed for {event_name}", error=str(e))


# Global plugin registry instance
plugin_registry = PluginRegistry()


# Example plugin implementations for demonstration

class ExampleAgentPlugin(AgentPlugin):
    """Example agent plugin implementation"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="example_agent",
            name="Example Agent Plugin",
            version="1.0.0",
            plugin_type=PluginType.AGENT,
            description="Example agent plugin for demonstration",
            author="XORB Team",
            capabilities={"task_execution", "status_reporting"}
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        return True
    
    async def shutdown(self) -> bool:
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "uptime": 3600}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "completed", "result": "task executed"}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        return {"status": "ready", "queue_size": 0}


class ExampleMissionStrategyPlugin(MissionStrategyPlugin):
    """Example mission strategy plugin"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="example_strategy",
            name="Example Mission Strategy",
            version="1.0.0",
            plugin_type=PluginType.MISSION_STRATEGY,
            description="Example mission strategy plugin",
            author="XORB Team",
            capabilities={"mission_planning", "adaptive_execution"}
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    async def shutdown(self) -> bool:
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy"}
    
    async def plan_mission(self, objectives: List[Dict[str, Any]], 
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "plan_id": str(uuid.uuid4()),
            "strategy": "example_strategy",
            "objectives": objectives,
            "estimated_duration": 1800
        }
    
    async def adapt_mission(self, mission_plan: Dict[str, Any], 
                           adaptation_trigger: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "adaptation_applied": True,
            "changes": ["increased_priority"],
            "confidence": 0.8
        }