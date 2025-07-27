"""
Advanced Agent Discovery and Registration System

This module provides sophisticated agent discovery, registration, and lifecycle management
for the XORB ecosystem with support for dynamic loading, health monitoring, and
capability-based agent selection.
"""

import asyncio
import importlib
import inspect
import json
import logging
import pkgutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union
from uuid import uuid4

import structlog
from prometheus_client import Counter, Gauge, Histogram

# Metrics
AGENT_DISCOVERY_COUNTER = Counter('xorb_agent_discovery_total', 'Total agent discoveries', ['status'])
AGENT_REGISTRATION_COUNTER = Counter('xorb_agent_registration_total', 'Total agent registrations', ['status'])
ACTIVE_AGENTS_GAUGE = Gauge('xorb_active_agents', 'Number of active agents', ['capability'])
AGENT_HEALTH_CHECK_DURATION = Histogram('xorb_agent_health_check_duration_seconds', 'Agent health check duration')

logger = structlog.get_logger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status enumeration."""
    DISCOVERED = "discovered"
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class AgentCapability(Enum):
    """Standard agent capabilities."""
    SCANNING = "scanning"
    DISCOVERY = "discovery"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    STEALTH = "stealth"
    MULTI_ENGINE = "multi_engine"
    WEB_CRAWLING = "web_crawling"
    API_TESTING = "api_testing"
    NETWORK_SCANNING = "network_scanning"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    SOCIAL_ENGINEERING = "social_engineering"
    WIRELESS_TESTING = "wireless_testing"
    MOBILE_TESTING = "mobile_testing"
    IOT_TESTING = "iot_testing"


@dataclass
class AgentMetadata:
    """Comprehensive agent metadata structure."""
    name: str
    version: str
    capabilities: Set[AgentCapability]
    description: str
    author: str
    license: str
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    supported_platforms: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    deprecated: bool = False
    experimental: bool = False
    priority: int = 100  # Lower number = higher priority


@dataclass
class AgentRegistration:
    """Agent registration record."""
    agent_id: str
    metadata: AgentMetadata
    agent_class: Type
    module_path: str
    discovery_time: float
    registration_time: Optional[float] = None
    status: AgentStatus = AgentStatus.DISCOVERED
    health_status: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[float] = None
    execution_count: int = 0
    error_count: int = 0
    average_execution_time: float = 0.0


class IAgentDiscoveryPlugin(ABC):
    """Interface for agent discovery plugins."""
    
    @abstractmethod
    async def discover_agents(self) -> List[AgentRegistration]:
        """Discover agents from a specific source."""
        pass
    
    @abstractmethod
    def get_plugin_name(self) -> str:
        """Get the plugin name."""
        pass


class FileSystemDiscoveryPlugin(IAgentDiscoveryPlugin):
    """Discovers agents from the filesystem."""
    
    def __init__(self, search_paths: List[Path]):
        self.search_paths = search_paths
    
    async def discover_agents(self) -> List[AgentRegistration]:
        """Discover agents from filesystem paths."""
        discovered_agents = []
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
                
            for py_file in search_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                    
                try:
                    agent_registration = await self._load_agent_from_file(py_file)
                    if agent_registration:
                        discovered_agents.append(agent_registration)
                        AGENT_DISCOVERY_COUNTER.labels(status="success").inc()
                except Exception as e:
                    logger.warning("Failed to load agent", file=str(py_file), error=str(e))
                    AGENT_DISCOVERY_COUNTER.labels(status="error").inc()
        
        return discovered_agents
    
    async def _load_agent_from_file(self, file_path: Path) -> Optional[AgentRegistration]:
        """Load agent from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for agent classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_agent_class(obj):
                    metadata = self._extract_metadata(obj)
                    return AgentRegistration(
                        agent_id=str(uuid4()),
                        metadata=metadata,
                        agent_class=obj,
                        module_path=str(file_path),
                        discovery_time=time.time()
                    )
        except Exception as e:
            logger.error("Error loading agent from file", file=str(file_path), error=str(e))
            
        return None
    
    def _is_agent_class(self, cls: Type) -> bool:
        """Check if a class is an agent class."""
        return (
            hasattr(cls, 'execute') and
            hasattr(cls, 'capabilities') and
            not inspect.isabstract(cls)
        )
    
    def _extract_metadata(self, agent_class: Type) -> AgentMetadata:
        """Extract metadata from agent class."""
        # Default metadata
        metadata = AgentMetadata(
            name=agent_class.__name__,
            version=getattr(agent_class, '__version__', '1.0.0'),
            capabilities=set(),
            description=agent_class.__doc__ or "No description available",
            author=getattr(agent_class, '__author__', 'Unknown'),
            license=getattr(agent_class, '__license__', 'Unknown')
        )
        
        # Extract capabilities
        if hasattr(agent_class, 'capabilities'):
            caps = agent_class.capabilities
            if isinstance(caps, (list, tuple, set)):
                for cap in caps:
                    if isinstance(cap, str):
                        try:
                            metadata.capabilities.add(AgentCapability(cap))
                        except ValueError:
                            logger.warning("Unknown capability", capability=cap)
                    elif isinstance(cap, AgentCapability):
                        metadata.capabilities.add(cap)
        
        # Extract additional metadata
        if hasattr(agent_class, '__metadata__'):
            meta_dict = agent_class.__metadata__
            for key, value in meta_dict.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        return metadata
    
    def get_plugin_name(self) -> str:
        return "filesystem_discovery"


class EntryPointDiscoveryPlugin(IAgentDiscoveryPlugin):
    """Discovers agents from setuptools entry points."""
    
    def __init__(self, entry_point_group: str = "xorb.agents"):
        self.entry_point_group = entry_point_group
    
    async def discover_agents(self) -> List[AgentRegistration]:
        """Discover agents from entry points."""
        discovered_agents = []
        
        try:
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points(self.entry_point_group):
                try:
                    agent_class = entry_point.load()
                    metadata = self._extract_metadata(agent_class, entry_point)
                    
                    registration = AgentRegistration(
                        agent_id=str(uuid4()),
                        metadata=metadata,
                        agent_class=agent_class,
                        module_path=f"entry_point:{entry_point.module_name}",
                        discovery_time=time.time()
                    )
                    
                    discovered_agents.append(registration)
                    AGENT_DISCOVERY_COUNTER.labels(status="success").inc()
                    
                except Exception as e:
                    logger.warning("Failed to load entry point agent", entry_point=str(entry_point), error=str(e))
                    AGENT_DISCOVERY_COUNTER.labels(status="error").inc()
                    
        except ImportError:
            logger.warning("pkg_resources not available for entry point discovery")
        
        return discovered_agents
    
    def _extract_metadata(self, agent_class: Type, entry_point) -> AgentMetadata:
        """Extract metadata from entry point agent."""
        metadata = AgentMetadata(
            name=entry_point.name,
            version=getattr(agent_class, '__version__', '1.0.0'),
            capabilities=set(),
            description=agent_class.__doc__ or "No description available",
            author=getattr(agent_class, '__author__', 'Unknown'),
            license=getattr(agent_class, '__license__', 'Unknown')
        )
        
        # Extract capabilities from agent class
        if hasattr(agent_class, 'capabilities'):
            caps = agent_class.capabilities
            if isinstance(caps, (list, tuple, set)):
                for cap in caps:
                    if isinstance(cap, str):
                        try:
                            metadata.capabilities.add(AgentCapability(cap))
                        except ValueError:
                            pass
                    elif isinstance(cap, AgentCapability):
                        metadata.capabilities.add(cap)
        
        return metadata
    
    def get_plugin_name(self) -> str:
        return "entry_point_discovery"


class AdvancedAgentRegistry:
    """Advanced agent registry with discovery, registration, and lifecycle management."""
    
    def __init__(self):
        self.agents: Dict[str, AgentRegistration] = {}
        self.discovery_plugins: List[IAgentDiscoveryPlugin] = []
        self.capability_index: Dict[AgentCapability, Set[str]] = {}
        self.running = False
        self._discovery_interval = 300  # 5 minutes
        self._health_check_interval = 60  # 1 minute
    
    def add_discovery_plugin(self, plugin: IAgentDiscoveryPlugin):
        """Add a discovery plugin."""
        self.discovery_plugins.append(plugin)
        logger.info("Added discovery plugin", plugin=plugin.get_plugin_name())
    
    async def start_discovery(self):
        """Start the discovery service."""
        self.running = True
        
        # Add default plugins
        self._setup_default_plugins()
        
        # Start discovery and health check tasks
        discovery_task = asyncio.create_task(self._discovery_loop())
        health_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Agent discovery service started")
        
        try:
            await asyncio.gather(discovery_task, health_task)
        except asyncio.CancelledError:
            logger.info("Agent discovery service stopped")
    
    async def stop_discovery(self):
        """Stop the discovery service."""
        self.running = False
    
    def _setup_default_plugins(self):
        """Setup default discovery plugins."""
        # Filesystem discovery
        fs_plugin = FileSystemDiscoveryPlugin([
            Path("xorb_core/agents"),
            Path("plugins/agents"),
            Path("/usr/local/lib/xorb/agents")
        ])
        self.add_discovery_plugin(fs_plugin)
        
        # Entry point discovery
        ep_plugin = EntryPointDiscoveryPlugin()
        self.add_discovery_plugin(ep_plugin)
    
    async def _discovery_loop(self):
        """Main discovery loop."""
        while self.running:
            try:
                await self.discover_agents()
                await asyncio.sleep(self._discovery_interval)
            except Exception as e:
                logger.error("Error in discovery loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _health_check_loop(self):
        """Health check loop for registered agents."""
        while self.running:
            try:
                await self.health_check_all_agents()
                await asyncio.sleep(self._health_check_interval)
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(30)
    
    async def discover_agents(self) -> List[AgentRegistration]:
        """Discover agents using all plugins."""
        all_discovered = []
        
        for plugin in self.discovery_plugins:
            try:
                discovered = await plugin.discover_agents()
                all_discovered.extend(discovered)
                logger.info("Discovery completed", plugin=plugin.get_plugin_name(), count=len(discovered))
            except Exception as e:
                logger.error("Discovery plugin failed", plugin=plugin.get_plugin_name(), error=str(e))
        
        # Register newly discovered agents
        for agent_reg in all_discovered:
            await self.register_agent(agent_reg)
        
        return all_discovered
    
    async def register_agent(self, agent_registration: AgentRegistration) -> bool:
        """Register an agent."""
        try:
            # Check if agent already exists
            existing_agent = self._find_existing_agent(agent_registration)
            if existing_agent:
                # Update existing agent if newer version
                if self._is_newer_version(agent_registration, existing_agent):
                    await self._update_agent(existing_agent.agent_id, agent_registration)
                return True
            
            # Register new agent
            agent_registration.registration_time = time.time()
            agent_registration.status = AgentStatus.REGISTERED
            
            # Perform initial health check
            health_ok = await self._perform_health_check(agent_registration)
            if health_ok:
                agent_registration.status = AgentStatus.ACTIVE
            
            self.agents[agent_registration.agent_id] = agent_registration
            self._update_capability_index(agent_registration)
            
            # Update metrics
            AGENT_REGISTRATION_COUNTER.labels(status="success").inc()
            for capability in agent_registration.metadata.capabilities:
                ACTIVE_AGENTS_GAUGE.labels(capability=capability.value).inc()
            
            logger.info("Agent registered", 
                       name=agent_registration.metadata.name,
                       agent_id=agent_registration.agent_id,
                       capabilities=list(agent_registration.metadata.capabilities))
            
            return True
            
        except Exception as e:
            logger.error("Failed to register agent", 
                        name=agent_registration.metadata.name, 
                        error=str(e))
            AGENT_REGISTRATION_COUNTER.labels(status="error").inc()
            return False
    
    def _find_existing_agent(self, new_agent: AgentRegistration) -> Optional[AgentRegistration]:
        """Find existing agent with same name."""
        for agent in self.agents.values():
            if agent.metadata.name == new_agent.metadata.name:
                return agent
        return None
    
    def _is_newer_version(self, new_agent: AgentRegistration, existing_agent: AgentRegistration) -> bool:
        """Check if new agent has a newer version."""
        try:
            from packaging import version
            return version.parse(new_agent.metadata.version) > version.parse(existing_agent.metadata.version)
        except:
            return new_agent.discovery_time > existing_agent.discovery_time
    
    async def _update_agent(self, agent_id: str, new_registration: AgentRegistration):
        """Update existing agent with new version."""
        old_agent = self.agents[agent_id]
        new_registration.agent_id = agent_id
        new_registration.registration_time = time.time()
        
        # Preserve execution stats
        new_registration.execution_count = old_agent.execution_count
        new_registration.error_count = old_agent.error_count
        new_registration.average_execution_time = old_agent.average_execution_time
        
        self.agents[agent_id] = new_registration
        self._update_capability_index(new_registration)
        
        logger.info("Agent updated", 
                   name=new_registration.metadata.name,
                   old_version=old_agent.metadata.version,
                   new_version=new_registration.metadata.version)
    
    def _update_capability_index(self, agent_registration: AgentRegistration):
        """Update the capability index."""
        for capability in agent_registration.metadata.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent_registration.agent_id)
    
    async def health_check_all_agents(self):
        """Perform health checks on all registered agents."""
        for agent_id, agent_reg in list(self.agents.items()):
            if agent_reg.status in (AgentStatus.ACTIVE, AgentStatus.REGISTERED):
                await self._perform_health_check(agent_reg)
    
    @AGENT_HEALTH_CHECK_DURATION.time()
    async def _perform_health_check(self, agent_registration: AgentRegistration) -> bool:
        """Perform health check on a single agent."""
        try:
            # Create agent instance for health check
            agent_instance = agent_registration.agent_class()
            
            if hasattr(agent_instance, 'health_check'):
                health_result = await agent_instance.health_check()
                agent_registration.health_status = health_result
                agent_registration.last_health_check = time.time()
                
                is_healthy = health_result.get('status') == 'healthy'
                if is_healthy:
                    agent_registration.status = AgentStatus.ACTIVE
                else:
                    agent_registration.status = AgentStatus.INACTIVE
                
                return is_healthy
            else:
                # Default health check - just try to instantiate
                agent_registration.health_status = {'status': 'healthy', 'message': 'Basic health check passed'}
                agent_registration.last_health_check = time.time()
                agent_registration.status = AgentStatus.ACTIVE
                return True
                
        except Exception as e:
            agent_registration.health_status = {'status': 'error', 'message': str(e)}
            agent_registration.last_health_check = time.time()
            agent_registration.status = AgentStatus.ERROR
            logger.warning("Agent health check failed", 
                          name=agent_registration.metadata.name,
                          error=str(e))
            return False
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentRegistration]:
        """Get all active agents with a specific capability."""
        agent_ids = self.capability_index.get(capability, set())
        return [
            self.agents[agent_id] 
            for agent_id in agent_ids 
            if self.agents[agent_id].status == AgentStatus.ACTIVE
        ]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentRegistration]:
        """Get all registered agents."""
        return list(self.agents.values())
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        stats = {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
            'inactive_agents': len([a for a in self.agents.values() if a.status == AgentStatus.INACTIVE]),
            'error_agents': len([a for a in self.agents.values() if a.status == AgentStatus.ERROR]),
            'capabilities': {},
            'discovery_plugins': len(self.discovery_plugins),
            'last_discovery': max([a.discovery_time for a in self.agents.values()], default=0)
        }
        
        # Capability statistics
        for capability, agent_ids in self.capability_index.items():
            active_count = len([
                aid for aid in agent_ids 
                if self.agents[aid].status == AgentStatus.ACTIVE
            ])
            stats['capabilities'][capability.value] = {
                'total': len(agent_ids),
                'active': active_count
            }
        
        return stats


# Global registry instance
agent_registry = AdvancedAgentRegistry()


async def initialize_agent_discovery():
    """Initialize the agent discovery system."""
    await agent_registry.start_discovery()


async def shutdown_agent_discovery():
    """Shutdown the agent discovery system."""
    await agent_registry.stop_discovery()


def get_agent_registry() -> AdvancedAgentRegistry:
    """Get the global agent registry."""
    return agent_registry