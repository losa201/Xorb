#!/usr/bin/env python3
"""
Enhanced Orchestrator for Xorb 2.0 with Dynamic Agent Discovery and Concurrent Execution

This module provides an advanced orchestrator that:
- Discovers agents dynamically using entry points and plugin directories
- Executes agents concurrently with asyncio and Temporal orchestration
- Emits CloudEvents for each agent event
- Records structured logs and Prometheus metrics
- Implements graceful retry mechanisms with tenacity
- Supports EPYC-optimized concurrent execution patterns
"""

import asyncio
import logging
import uuid
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.util
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from cloudevents.http import CloudEvent
from cloudevents.conversion import to_json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog
import nats
from nats.js import JetStreamContext

# Import existing components
from .scheduler import CampaignScheduler
from .audit_logger import AuditLogger
from .roe_compliance import RoEValidator
from .event_system import EventBus
from ..agents.base_agent import BaseAgent, AgentCapability, AgentStatus


class ExecutionStatus(str, Enum):
    """Execution status for agents and campaigns"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AgentDiscoveryMethod(str, Enum):
    """Methods for discovering agents"""
    ENTRY_POINTS = "entry_points"
    PLUGIN_DIRECTORY = "plugin_directory"
    REGISTRY_API = "registry_api"
    KUBERNETES_CRD = "kubernetes_crd"


@dataclass
class AgentMetadata:
    """Metadata for discovered agents"""
    name: str
    version: str
    capabilities: List[AgentCapability]
    resource_requirements: Dict[str, Any]
    discovery_method: AgentDiscoveryMethod
    plugin_path: Optional[str] = None
    entry_point: Optional[str] = None
    last_seen: datetime = field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for agent execution"""
    campaign_id: str
    agent_id: str
    agent_name: str
    target: Dict[str, Any]
    config: Dict[str, Any]
    timeout: int = 300
    max_retries: int = 3
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0


class AgentRegistry:
    """Dynamic agent discovery and registry"""
    
    def __init__(self, plugin_directories: List[str] = None):
        self.logger = structlog.get_logger(__name__)
        self.agents: Dict[str, AgentMetadata] = {}
        self.plugin_directories = plugin_directories or ["./agents", "./plugins"]
        self.discovery_interval = 300  # 5 minutes
        self._discovery_task: Optional[asyncio.Task] = None
        
    async def start_discovery(self):
        """Start periodic agent discovery"""
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        
    async def stop_discovery(self):
        """Stop agent discovery"""
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
                
    async def _discovery_loop(self):
        """Main discovery loop"""
        while True:
            try:
                await self.discover_agents()
                await asyncio.sleep(self.discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Agent discovery failed", error=str(e))
                await asyncio.sleep(60)  # Retry in 1 minute on error
                
    async def discover_agents(self) -> Set[str]:
        """Discover all available agents"""
        discovered = set()
        
        # Discover via entry points
        discovered.update(await self._discover_entry_points())
        
        # Discover via plugin directories
        for directory in self.plugin_directories:
            discovered.update(await self._discover_plugin_directory(directory))
            
        # Update last seen timestamp for discovered agents
        now = datetime.utcnow()
        for agent_name in discovered:
            if agent_name in self.agents:
                self.agents[agent_name].last_seen = now
                
        self.logger.info("Agent discovery completed", 
                        discovered_count=len(discovered),
                        total_agents=len(self.agents))
        return discovered
        
    async def _discover_entry_points(self) -> Set[str]:
        """Discover agents via Python entry points"""
        discovered = set()
        
        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points('xorb.agents'):
                try:
                    agent_class = entry_point.load()
                    if self._is_valid_agent_class(agent_class):
                        metadata = await self._create_agent_metadata(
                            agent_class, 
                            AgentDiscoveryMethod.ENTRY_POINTS,
                            entry_point=entry_point.name
                        )
                        self.agents[metadata.name] = metadata
                        discovered.add(metadata.name)
                        
                except Exception as e:
                    self.logger.warning("Failed to load entry point agent",
                                      entry_point=entry_point.name, error=str(e))
                                      
        except ImportError:
            self.logger.debug("pkg_resources not available, skipping entry point discovery")
            
        return discovered
        
    async def _discover_plugin_directory(self, directory: str) -> Set[str]:
        """Discover agents in plugin directory"""
        discovered = set()
        plugin_path = Path(directory)
        
        if not plugin_path.exists():
            return discovered
            
        for python_file in plugin_path.rglob("*.py"):
            if python_file.name.startswith("_"):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(
                    f"xorb.agents.{python_file.stem}", python_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find agent classes in module
                    for name, obj in inspect.getmembers(module):
                        if self._is_valid_agent_class(obj):
                            metadata = await self._create_agent_metadata(
                                obj,
                                AgentDiscoveryMethod.PLUGIN_DIRECTORY,
                                plugin_path=str(python_file)
                            )
                            self.agents[metadata.name] = metadata
                            discovered.add(metadata.name)
                            
            except Exception as e:
                self.logger.warning("Failed to load plugin",
                                  file=str(python_file), error=str(e))
                                  
        return discovered
        
    def _is_valid_agent_class(self, obj) -> bool:
        """Check if object is a valid agent class"""
        return (
            inspect.isclass(obj) and
            issubclass(obj, BaseAgent) and
            obj != BaseAgent and
            hasattr(obj, 'name') and
            hasattr(obj, 'capabilities')
        )
        
    async def _create_agent_metadata(self, agent_class, discovery_method: AgentDiscoveryMethod,
                                   plugin_path: str = None, entry_point: str = None) -> AgentMetadata:
        """Create metadata for discovered agent"""
        return AgentMetadata(
            name=agent_class.name,
            version=getattr(agent_class, 'version', '1.0.0'),
            capabilities=getattr(agent_class, 'capabilities', []),
            resource_requirements=getattr(agent_class, 'resource_requirements', {}),
            discovery_method=discovery_method,
            plugin_path=plugin_path,
            entry_point=entry_point
        )
        
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentMetadata]:
        """Get agents that have a specific capability"""
        return [
            metadata for metadata in self.agents.values()
            if capability in metadata.capabilities
        ]
        
    def get_agent_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """Get metadata for a specific agent"""
        return self.agents.get(agent_name)


class MetricsCollector:
    """Prometheus metrics collector for orchestrator"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Counters
        self.agent_executions = Counter(
            'xorb_agent_executions_total',
            'Total number of agent executions',
            ['agent_name', 'status', 'environment'],
            registry=self.registry
        )
        
        self.campaign_operations = Counter(
            'xorb_campaign_operations_total',
            'Total number of campaign operations',
            ['operation', 'status', 'environment'],
            registry=self.registry
        )
        
        self.cloudevents_sent = Counter(
            'xorb_cloudevents_sent_total',
            'Total number of CloudEvents sent',
            ['event_type', 'environment'],
            registry=self.registry
        )
        
        # Histograms
        self.agent_execution_duration = Histogram(
            'xorb_agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_name', 'environment'],
            registry=self.registry
        )
        
        self.campaign_duration = Histogram(
            'xorb_campaign_duration_seconds',
            'Campaign duration',
            ['environment'],
            registry=self.registry
        )
        
        # Gauges
        self.active_campaigns = Gauge(
            'xorb_active_campaigns',
            'Number of active campaigns',
            ['environment'],
            registry=self.registry
        )
        
        self.active_agents = Gauge(
            'xorb_active_agents',
            'Number of active agents',
            ['environment'],
            registry=self.registry
        )
        
        self.discovered_agents = Gauge(
            'xorb_discovered_agents_total',
            'Total number of discovered agents',
            ['discovery_method', 'environment'],
            registry=self.registry
        )
        
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')


class EnhancedOrchestrator:
    """
    Enhanced orchestrator with dynamic agent discovery and concurrent execution
    
    Features:
    - Dynamic agent discovery via entry points and plugin directories
    - Concurrent agent execution with asyncio and thread pools
    - CloudEvents integration for event-driven architecture
    - Prometheus metrics and structured logging
    - Graceful error handling and retry mechanisms
    - EPYC-optimized concurrent execution patterns
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 nats_url: str = "nats://localhost:4222",
                 plugin_directories: List[str] = None,
                 max_concurrent_agents: int = 32,  # EPYC optimized
                 max_concurrent_campaigns: int = 10):
        
        self.logger = structlog.get_logger(__name__)
        self.redis_client = redis.from_url(redis_url)
        self.nats_url = nats_url
        self.nats_client: Optional[nats.NATS] = None
        self.jetstream: Optional[JetStreamContext] = None
        
        # Core components
        self.agent_registry = AgentRegistry(plugin_directories)
        self.scheduler = CampaignScheduler()
        self.audit_logger = AuditLogger()
        self.roe_validator = RoEValidator()
        self.event_system = EventBus()
        self.metrics = MetricsCollector()
        
        # Execution configuration
        self.max_concurrent_agents = max_concurrent_agents
        self.max_concurrent_campaigns = max_concurrent_campaigns
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_agents)
        
        # State management
        self.campaigns: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_agents)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
    async def start(self):
        """Start the enhanced orchestrator"""
        self.logger.info("Starting Enhanced Xorb Orchestrator")
        
        try:
            # Connect to external services
            await self.redis_client.ping()
            self.logger.info("Connected to Redis")
            
            # Connect to NATS
            self.nats_client = await nats.connect(self.nats_url)
            self.jetstream = self.nats_client.jetstream()
            self.logger.info("Connected to NATS JetStream")
            
            # Start agent discovery
            await self.agent_registry.start_discovery()
            self.logger.info("Started agent discovery")
            
            # Start scheduler
            await self.scheduler.start()
            self.logger.info("Started campaign scheduler")
            
            # Initialize metrics
            await self._update_metrics()
            
            await self.audit_logger.log_event("orchestrator_start", {
                "timestamp": datetime.utcnow().isoformat(),
                "max_concurrent_agents": self.max_concurrent_agents,
                "max_concurrent_campaigns": self.max_concurrent_campaigns
            })
            
        except Exception as e:
            self.logger.error("Failed to start orchestrator", error=str(e))
            raise
            
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down Enhanced Xorb Orchestrator")
        
        # Cancel all active executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)
            
        # Stop components
        await self.agent_registry.stop_discovery()
        await self.scheduler.stop()
        
        # Close connections
        if self.nats_client:
            await self.nats_client.close()
        await self.redis_client.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        await self.audit_logger.log_event("orchestrator_shutdown", {
            "timestamp": datetime.utcnow().isoformat()
        })
        
    async def create_campaign(self, name: str, targets: List[Dict], 
                            agent_requirements: List[AgentCapability] = None,
                            config: Dict[str, Any] = None) -> str:
        """Create a new campaign with dynamic agent selection"""
        
        campaign_id = str(uuid.uuid4())
        
        # Validate targets against RoE
        validated_targets = []
        for target_data in targets:
            if await self.roe_validator.validate_target_dict(target_data):
                validated_targets.append(target_data)
            else:
                self.logger.warning("Target failed RoE validation", target=target_data)
                
        if not validated_targets:
            raise ValueError("No valid targets after RoE validation")
            
        # Select appropriate agents based on requirements
        selected_agents = await self._select_agents_for_campaign(
            agent_requirements or [], validated_targets, config or {}
        )
        
        campaign = {
            "id": campaign_id,
            "name": name,
            "targets": validated_targets,
            "selected_agents": selected_agents,
            "config": config or {},
            "status": ExecutionStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "executions": []
        }
        
        self.campaigns[campaign_id] = campaign
        await self._persist_campaign(campaign)
        
        # Emit CloudEvent
        await self._emit_cloudevent("xorb.campaign.created", {
            "campaign_id": campaign_id,
            "name": name,
            "target_count": len(validated_targets),
            "agent_count": len(selected_agents)
        })
        
        self.metrics.campaign_operations.labels(
            operation="create", status="success", 
            environment=self._get_environment()
        ).inc()
        
        self.logger.info("Created campaign", 
                        campaign_id=campaign_id, 
                        name=name,
                        targets=len(validated_targets),
                        agents=len(selected_agents))
        
        return campaign_id
        
    async def start_campaign(self, campaign_id: str) -> bool:
        """Start campaign execution with concurrent agent processing"""
        
        if campaign_id not in self.campaigns:
            self.logger.error("Campaign not found", campaign_id=campaign_id)
            return False
            
        campaign = self.campaigns[campaign_id]
        
        if campaign["status"] != ExecutionStatus.PENDING:
            self.logger.warning("Campaign not in pending state", 
                              campaign_id=campaign_id, 
                              status=campaign["status"])
            return False
            
        # Check concurrent campaign limit
        active_campaigns = sum(1 for c in self.campaigns.values() 
                             if c["status"] == ExecutionStatus.RUNNING)
        
        if active_campaigns >= self.max_concurrent_campaigns:
            self.logger.info("Maximum concurrent campaigns reached, queuing",
                           campaign_id=campaign_id)
            await self.scheduler.queue_campaign(campaign_id)
            return True
            
        campaign["status"] = ExecutionStatus.RUNNING
        campaign["started_at"] = datetime.utcnow().isoformat()
        
        # Create execution contexts for all agent-target combinations
        execution_contexts = []
        for target in campaign["targets"]:
            for agent_name in campaign["selected_agents"]:
                context = ExecutionContext(
                    campaign_id=campaign_id,
                    agent_id=str(uuid.uuid4()),
                    agent_name=agent_name,
                    target=target,
                    config=campaign["config"],
                    timeout=campaign["config"].get("timeout", 300),
                    max_retries=campaign["config"].get("max_retries", 3)
                )
                execution_contexts.append(context)
                
        campaign["executions"] = [ctx.agent_id for ctx in execution_contexts]
        
        # Start concurrent execution
        asyncio.create_task(self._execute_campaign_concurrent(campaign_id, execution_contexts))
        
        await self._persist_campaign(campaign)
        await self._emit_cloudevent("xorb.campaign.started", {
            "campaign_id": campaign_id,
            "execution_count": len(execution_contexts)
        })
        
        self.metrics.campaign_operations.labels(
            operation="start", status="success",
            environment=self._get_environment()
        ).inc()
        
        self.metrics.active_campaigns.labels(
            environment=self._get_environment()
        ).inc()
        
        self.logger.info("Started campaign", campaign_id=campaign_id,
                        executions=len(execution_contexts))
        
        return True
        
    async def _execute_campaign_concurrent(self, campaign_id: str, 
                                         execution_contexts: List[ExecutionContext]):
        """Execute campaign with concurrent agent processing"""
        
        start_time = time.time()
        
        try:
            # Execute all contexts concurrently with semaphore control
            tasks = []
            for context in execution_contexts:
                self.active_executions[context.agent_id] = context
                task = asyncio.create_task(self._execute_agent_with_retry(context))
                tasks.append(task)
                
            # Wait for all executions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful = 0
            failed = 0
            for i, result in enumerate(results):
                context = execution_contexts[i]
                if isinstance(result, Exception):
                    context.status = ExecutionStatus.FAILED
                    context.error = str(result)
                    failed += 1
                else:
                    context.result = result
                    context.status = ExecutionStatus.COMPLETED
                    successful += 1
                    
                # Remove from active executions
                self.active_executions.pop(context.agent_id, None)
                
            # Update campaign status
            campaign = self.campaigns[campaign_id]
            campaign["status"] = ExecutionStatus.COMPLETED
            campaign["completed_at"] = datetime.utcnow().isoformat()
            campaign["results"] = {
                "successful": successful,
                "failed": failed,
                "total": len(execution_contexts)
            }
            
            await self._persist_campaign(campaign)
            
            # Emit completion event
            await self._emit_cloudevent("xorb.campaign.completed", {
                "campaign_id": campaign_id,
                "successful": successful,
                "failed": failed,
                "duration_seconds": time.time() - start_time
            })
            
            self.metrics.campaign_duration.labels(
                environment=self._get_environment()
            ).observe(time.time() - start_time)
            
            self.metrics.active_campaigns.labels(
                environment=self._get_environment()
            ).dec()
            
            self.logger.info("Campaign completed", 
                           campaign_id=campaign_id,
                           successful=successful,
                           failed=failed,
                           duration=time.time() - start_time)
            
        except Exception as e:
            # Mark campaign as failed
            campaign = self.campaigns[campaign_id]
            campaign["status"] = ExecutionStatus.FAILED
            campaign["completed_at"] = datetime.utcnow().isoformat()
            campaign["error"] = str(e)
            
            await self._persist_campaign(campaign)
            
            self.logger.error("Campaign execution failed", 
                            campaign_id=campaign_id, error=str(e))
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _execute_agent_with_retry(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute agent with retry logic and semaphore control"""
        
        async with self.execution_semaphore:
            start_time = time.time()
            context.started_at = datetime.utcnow()
            context.status = ExecutionStatus.RUNNING
            
            try:
                # Get agent metadata
                agent_metadata = self.agent_registry.get_agent_metadata(context.agent_name)
                if not agent_metadata:
                    raise ValueError(f"Agent {context.agent_name} not found in registry")
                    
                # Load and instantiate agent
                agent_instance = await self._load_agent_instance(agent_metadata)
                
                # Execute agent
                result = await asyncio.wait_for(
                    agent_instance.execute(context.target, context.config),
                    timeout=context.timeout
                )
                
                context.completed_at = datetime.utcnow()
                context.status = ExecutionStatus.COMPLETED
                context.result = result
                
                # Emit success event
                await self._emit_cloudevent("xorb.agent.completed", {
                    "campaign_id": context.campaign_id,
                    "agent_id": context.agent_id,
                    "agent_name": context.agent_name,
                    "execution_time": time.time() - start_time
                })
                
                self.metrics.agent_executions.labels(
                    agent_name=context.agent_name,
                    status="success",
                    environment=self._get_environment()
                ).inc()
                
                self.metrics.agent_execution_duration.labels(
                    agent_name=context.agent_name,
                    environment=self._get_environment()
                ).observe(time.time() - start_time)
                
                self.logger.info("Agent execution completed",
                               campaign_id=context.campaign_id,
                               agent_name=context.agent_name,
                               duration=time.time() - start_time)
                
                return result
                
            except Exception as e:
                context.retry_count += 1
                context.error = str(e)
                
                if context.retry_count >= context.max_retries:
                    context.status = ExecutionStatus.FAILED
                    context.completed_at = datetime.utcnow()
                    
                    self.metrics.agent_executions.labels(
                        agent_name=context.agent_name,
                        status="failed",
                        environment=self._get_environment()
                    ).inc()
                    
                    self.logger.error("Agent execution failed permanently",
                                    campaign_id=context.campaign_id,
                                    agent_name=context.agent_name,
                                    error=str(e),
                                    retry_count=context.retry_count)
                else:
                    context.status = ExecutionStatus.RETRYING
                    
                    self.logger.warning("Agent execution failed, retrying",
                                      campaign_id=context.campaign_id,
                                      agent_name=context.agent_name,
                                      error=str(e),
                                      retry_count=context.retry_count)
                
                raise
                
    async def _load_agent_instance(self, metadata: AgentMetadata) -> BaseAgent:
        """Load and instantiate an agent from metadata"""
        
        if metadata.discovery_method == AgentDiscoveryMethod.ENTRY_POINTS:
            import pkg_resources
            entry_point = pkg_resources.get_entry_info(
                'xorb', 'xorb.agents', metadata.entry_point
            )
            agent_class = entry_point.load()
            
        elif metadata.discovery_method == AgentDiscoveryMethod.PLUGIN_DIRECTORY:
            spec = importlib.util.spec_from_file_location(
                f"xorb.agents.{metadata.name}", metadata.plugin_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the agent class in the module
            agent_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAgent) and 
                    getattr(obj, 'name', None) == metadata.name):
                    agent_class = obj
                    break
                    
            if not agent_class:
                raise ValueError(f"Agent class not found in {metadata.plugin_path}")
                
        else:
            raise ValueError(f"Unsupported discovery method: {metadata.discovery_method}")
            
        return agent_class()
        
    async def _select_agents_for_campaign(self, requirements: List[AgentCapability],
                                        targets: List[Dict], config: Dict) -> List[str]:
        """Select appropriate agents for campaign based on requirements"""
        
        selected_agents = []
        
        if not requirements:
            # If no specific requirements, select default agents based on target types
            requirements = [AgentCapability.RECONNAISSANCE, AgentCapability.VULNERABILITY_SCANNING]
            
        for capability in requirements:
            capable_agents = self.agent_registry.get_agents_by_capability(capability)
            if capable_agents:
                # Select the first available agent with this capability
                # TODO: Implement more sophisticated selection logic
                selected_agents.append(capable_agents[0].name)
                
        return list(set(selected_agents))  # Remove duplicates
        
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        
        if execution_id not in self.active_executions:
            return False
            
        context = self.active_executions[execution_id]
        context.status = ExecutionStatus.CANCELLED
        context.completed_at = datetime.utcnow()
        
        await self._emit_cloudevent("xorb.agent.cancelled", {
            "campaign_id": context.campaign_id,
            "agent_id": context.agent_id,
            "agent_name": context.agent_name
        })
        
        self.logger.info("Execution cancelled", 
                        execution_id=execution_id,
                        agent_name=context.agent_name)
        
        return True
        
    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict]:
        """Get detailed campaign status"""
        
        if campaign_id not in self.campaigns:
            return None
            
        campaign = self.campaigns[campaign_id]
        
        # Get execution statuses
        execution_statuses = {}
        for exec_id in campaign.get("executions", []):
            if exec_id in self.active_executions:
                context = self.active_executions[exec_id]
                execution_statuses[exec_id] = {
                    "agent_name": context.agent_name,
                    "status": context.status,
                    "retry_count": context.retry_count,
                    "created_at": context.created_at.isoformat(),
                    "started_at": context.started_at.isoformat() if context.started_at else None,
                    "completed_at": context.completed_at.isoformat() if context.completed_at else None
                }
                
        status = {
            **campaign,
            "execution_details": execution_statuses,
            "active_executions": len([e for e in execution_statuses.values() 
                                    if e["status"] == ExecutionStatus.RUNNING])
        }
        
        return status
        
    async def list_discovered_agents(self) -> Dict[str, AgentMetadata]:
        """List all discovered agents"""
        return self.agent_registry.agents.copy()
        
    async def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        await self._update_metrics()
        return self.metrics.get_metrics()
        
    async def _emit_cloudevent(self, event_type: str, data: Dict[str, Any]):
        """Emit CloudEvent via NATS"""
        
        if not self.jetstream:
            return
            
        try:
            event = CloudEvent({
                "type": event_type,
                "source": "xorb.orchestrator",
                "specversion": "1.0",
                "id": str(uuid.uuid4()),
                "time": datetime.utcnow().isoformat() + "Z",
                "datacontenttype": "application/json"
            }, data)
            
            event_json = to_json(event)
            
            await self.jetstream.publish(
                f"xorb.events.{event_type.replace('.', '-')}",
                event_json
            )
            
            self.metrics.cloudevents_sent.labels(
                event_type=event_type,
                environment=self._get_environment()
            ).inc()
            
        except Exception as e:
            self.logger.error("Failed to emit CloudEvent", 
                            event_type=event_type, error=str(e))
            
    async def _persist_campaign(self, campaign: Dict[str, Any]):
        """Persist campaign to Redis"""
        
        try:
            await self.redis_client.hset(
                f"campaign:{campaign['id']}", 
                mapping={"data": json.dumps(campaign, default=str)}
            )
        except Exception as e:
            self.logger.error("Failed to persist campaign", 
                            campaign_id=campaign["id"], error=str(e))
            
    async def _update_metrics(self):
        """Update metrics with current state"""
        
        env = self._get_environment()
        
        # Update discovered agents metrics
        discovery_counts = {}
        for agent in self.agent_registry.agents.values():
            method = agent.discovery_method.value
            discovery_counts[method] = discovery_counts.get(method, 0) + 1
            
        for method, count in discovery_counts.items():
            self.metrics.discovered_agents.labels(
                discovery_method=method,
                environment=env
            ).set(count)
            
        # Update active agents
        self.metrics.active_agents.labels(environment=env).set(
            len(self.active_executions)
        )
        
    def _get_environment(self) -> str:
        """Get current environment name"""
        import os
        return os.environ.get("XORB_ENVIRONMENT", "development")


# Example usage and integration
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Enhanced Xorb Orchestrator")
        parser.add_argument("--redis-url", default="redis://localhost:6379")
        parser.add_argument("--nats-url", default="nats://localhost:4222")
        parser.add_argument("--plugin-dirs", nargs="+", default=["./agents"])
        parser.add_argument("--max-agents", type=int, default=32)
        parser.add_argument("--max-campaigns", type=int, default=10)
        
        args = parser.parse_args()
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        orchestrator = EnhancedOrchestrator(
            redis_url=args.redis_url,
            nats_url=args.nats_url,
            plugin_directories=args.plugin_dirs,
            max_concurrent_agents=args.max_agents,
            max_concurrent_campaigns=args.max_campaigns
        )
        
        try:
            await orchestrator.start()
            
            # Example: Create and start a test campaign
            campaign_id = await orchestrator.create_campaign(
                name="Test Campaign",
                targets=[{"hostname": "example.com", "ports": [80, 443]}],
                agent_requirements=[AgentCapability.RECONNAISSANCE]
            )
            
            await orchestrator.start_campaign(campaign_id)
            
            print(f"Started campaign: {campaign_id}")
            print("Orchestrator running... Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(10)
                status = await orchestrator.get_campaign_status(campaign_id)
                if status:
                    print(f"Campaign status: {status['status']}")
                    if status["status"] in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                        break
                        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await orchestrator.shutdown()
            
    asyncio.run(main())