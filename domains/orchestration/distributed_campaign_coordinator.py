"""
Distributed Campaign Coordination System

This module provides comprehensive distributed campaign coordination capabilities
including multi-node orchestration, workload distribution, fault tolerance,
consensus algorithms, and real-time coordination for the XORB ecosystem.
"""

import asyncio
import json
import time
import uuid
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from collections import defaultdict, deque
import socket
import aiohttp

import structlog
from prometheus_client import Counter, Gauge, Histogram

# Metrics
CAMPAIGN_COORDINATION_EVENTS = Counter('xorb_campaign_coordination_events_total', 'Campaign coordination events', ['event_type', 'node_id'])
ACTIVE_CAMPAIGNS_DISTRIBUTED = Gauge('xorb_active_campaigns_distributed', 'Active distributed campaigns', ['node_id'])
COORDINATION_LATENCY = Histogram('xorb_coordination_latency_seconds', 'Coordination message latency')
NODE_HEALTH_STATUS = Gauge('xorb_node_health_status', 'Node health status', ['node_id', 'status'])

logger = structlog.get_logger(__name__)


class NodeRole(Enum):
    """Node roles in distributed coordination."""
    COORDINATOR = "coordinator"        # Primary coordination node
    EXECUTOR = "executor"             # Campaign execution node
    OBSERVER = "observer"             # Monitoring/backup node
    SPECIALIST = "specialist"         # Specialized capability node


class CampaignState(Enum):
    """Distributed campaign states."""
    PENDING = "pending"
    COORDINATING = "coordinating"
    DISTRIBUTING = "distributing"
    EXECUTING = "executing"
    SYNCHRONIZING = "synchronizing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CoordinationMessageType(Enum):
    """Types of coordination messages."""
    HEARTBEAT = "heartbeat"
    CAMPAIGN_PROPOSAL = "campaign_proposal"
    CAMPAIGN_ACCEPTANCE = "campaign_acceptance"
    CAMPAIGN_REJECTION = "campaign_rejection"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    SYNCHRONIZATION = "synchronization"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    FAULT_DETECTION = "fault_detection"
    RECOVERY_INITIATION = "recovery_initiation"


class ConsensusAlgorithm(Enum):
    """Consensus algorithms for distributed decisions."""
    RAFT = "raft"
    PBFT = "pbft"          # Practical Byzantine Fault Tolerance
    SIMPLE_MAJORITY = "simple_majority"


@dataclass
class NodeInfo:
    """Information about a coordination node."""
    node_id: str
    host: str
    port: int
    role: NodeRole
    capabilities: Set[str] = field(default_factory=set)
    
    # Status
    online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    health_score: float = 1.0  # 0.0-1.0
    
    # Performance
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_campaigns: int = 0
    max_concurrent_campaigns: int = 10
    
    # Network
    network_latency: float = 0.0
    bandwidth_mbps: float = 1000.0
    
    # Metadata
    version: str = "2.0.0"
    region: str = "default"
    environment: str = "development"
    
    def get_endpoint(self) -> str:
        """Get node endpoint URL."""
        return f"http://{self.host}:{self.port}"
    
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (self.online and 
                self.health_score > 0.5 and
                time.time() - self.last_heartbeat < 60)
    
    def get_load_factor(self) -> float:
        """Calculate current load factor."""
        if self.max_concurrent_campaigns == 0:
            return 1.0
        return self.active_campaigns / self.max_concurrent_campaigns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['capabilities'] = list(self.capabilities)
        return data


@dataclass
class CoordinationMessage:
    """Message for distributed coordination."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    message_type: CoordinationMessageType = CoordinationMessageType.HEARTBEAT
    
    # Routing
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    correlation_id: Optional[str] = None
    
    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, higher is more important
    
    # Reliability
    requires_ack: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationMessage':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DistributedTask:
    """A task in a distributed campaign."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str = ""
    
    # Task definition
    task_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: Set[str] = field(default_factory=set)  # Required capabilities
    
    # Scheduling
    priority: int = 1
    estimated_duration: float = 300.0  # seconds
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    
    # Assignment
    assigned_node: Optional[str] = None
    assigned_at: Optional[float] = None
    
    # Execution
    status: str = "pending"  # pending, assigned, executing, completed, failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 2
    
    def get_duration(self) -> Optional[float]:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def is_ready_to_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute based on dependencies."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['requirements'] = list(self.requirements)
        return data


@dataclass
class DistributedCampaign:
    """A distributed security campaign."""
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    
    # Coordination
    coordinator_node: str = ""
    participating_nodes: Set[str] = field(default_factory=set)
    
    # Campaign definition
    campaign_type: str = "security_assessment"
    targets: List[Dict[str, Any]] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    
    # Tasks
    tasks: List[DistributedTask] = field(default_factory=list)
    task_graph: Dict[str, List[str]] = field(default_factory=dict)  # Dependencies
    
    # State management
    state: CampaignState = CampaignState.PENDING
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    
    # Configuration
    consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.SIMPLE_MAJORITY
    fault_tolerance_level: int = 1  # Number of node failures to tolerate
    synchronization_interval: float = 30.0  # seconds
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_state_change(self, new_state: CampaignState, reason: str = "", node_id: str = ""):
        """Add state change to history."""
        self.state = new_state
        self.state_history.append({
            "timestamp": time.time(),
            "state": new_state.value,
            "reason": reason,
            "node_id": node_id
        })
    
    def get_completion_percentage(self) -> float:
        """Get campaign completion percentage."""
        if not self.tasks:
            return 0.0
        
        completed_tasks = len([t for t in self.tasks if t.status == "completed"])
        return (completed_tasks / len(self.tasks)) * 100.0
    
    def get_failed_tasks(self) -> List[DistributedTask]:
        """Get failed tasks that might need retry."""
        return [t for t in self.tasks if t.status == "failed" and t.retry_count < t.max_retries]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['participating_nodes'] = list(self.participating_nodes)
        return data


class IConsensusEngine(ABC):
    """Interface for consensus algorithms."""
    
    @abstractmethod
    async def propose_decision(self, proposal_id: str, proposal_data: Dict[str, Any], 
                              participating_nodes: List[str]) -> bool:
        """Propose a decision for consensus."""
        pass
    
    @abstractmethod
    async def vote_on_proposal(self, proposal_id: str, vote: bool, node_id: str) -> bool:
        """Vote on a proposal."""
        pass
    
    @abstractmethod
    def get_consensus_result(self, proposal_id: str) -> Optional[bool]:
        """Get consensus result for a proposal."""
        pass


class SimpleMajorityConsensus(IConsensusEngine):
    """Simple majority consensus implementation."""
    
    def __init__(self):
        self.proposals = {}  # proposal_id -> proposal data
        self.votes = {}      # proposal_id -> {node_id: vote}
        self.results = {}    # proposal_id -> consensus result
    
    async def propose_decision(self, proposal_id: str, proposal_data: Dict[str, Any], 
                              participating_nodes: List[str]) -> bool:
        """Propose a decision for consensus."""
        self.proposals[proposal_id] = {
            "data": proposal_data,
            "nodes": participating_nodes,
            "created_at": time.time(),
            "required_votes": len(participating_nodes)
        }
        self.votes[proposal_id] = {}
        
        logger.info("Consensus proposal created", 
                   proposal_id=proposal_id, 
                   participating_nodes=len(participating_nodes))
        return True
    
    async def vote_on_proposal(self, proposal_id: str, vote: bool, node_id: str) -> bool:
        """Vote on a proposal."""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        if node_id not in proposal["nodes"]:
            return False
        
        self.votes[proposal_id][node_id] = vote
        
        # Check if we have consensus
        await self._check_consensus(proposal_id)
        
        logger.debug("Vote recorded", 
                    proposal_id=proposal_id, 
                    node_id=node_id, 
                    vote=vote)
        return True
    
    async def _check_consensus(self, proposal_id: str):
        """Check if consensus has been reached."""
        if proposal_id not in self.proposals:
            return
        
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        if len(votes) == proposal["required_votes"]:
            # All votes received
            yes_votes = sum(1 for vote in votes.values() if vote)
            total_votes = len(votes)
            
            # Simple majority
            consensus_reached = yes_votes > total_votes / 2
            self.results[proposal_id] = consensus_reached
            
            logger.info("Consensus reached", 
                       proposal_id=proposal_id, 
                       result=consensus_reached,
                       yes_votes=yes_votes,
                       total_votes=total_votes)
    
    def get_consensus_result(self, proposal_id: str) -> Optional[bool]:
        """Get consensus result for a proposal."""
        return self.results.get(proposal_id)


class ITaskScheduler(ABC):
    """Interface for distributed task scheduling."""
    
    @abstractmethod
    async def schedule_tasks(self, campaign: DistributedCampaign, 
                           available_nodes: List[NodeInfo]) -> Dict[str, str]:
        """Schedule tasks to nodes. Returns task_id -> node_id mapping."""
        pass


class CapabilityBasedScheduler(ITaskScheduler):
    """Schedule tasks based on node capabilities and load."""
    
    async def schedule_tasks(self, campaign: DistributedCampaign, 
                           available_nodes: List[NodeInfo]) -> Dict[str, str]:
        """Schedule tasks to nodes based on capabilities and load."""
        assignments = {}
        
        # Filter healthy nodes
        healthy_nodes = [node for node in available_nodes if node.is_healthy()]
        
        if not healthy_nodes:
            logger.error("No healthy nodes available for task scheduling")
            return assignments
        
        # Get ready tasks (dependencies satisfied)
        completed_task_ids = {t.task_id for t in campaign.tasks if t.status == "completed"}
        ready_tasks = [t for t in campaign.tasks 
                      if t.status == "pending" and t.is_ready_to_execute(completed_task_ids)]
        
        # Sort tasks by priority
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in ready_tasks:
            # Find capable nodes
            capable_nodes = [
                node for node in healthy_nodes
                if task.requirements.issubset(node.capabilities) or not task.requirements
            ]
            
            if not capable_nodes:
                logger.warning("No capable nodes found for task", 
                             task_id=task.task_id, 
                             requirements=list(task.requirements))
                continue
            
            # Select node with lowest load
            best_node = min(capable_nodes, key=lambda n: n.get_load_factor())
            
            # Check if node can handle another campaign
            if best_node.get_load_factor() >= 1.0:
                logger.warning("All capable nodes at capacity", task_id=task.task_id)
                continue
            
            assignments[task.task_id] = best_node.node_id
            best_node.active_campaigns += 1  # Update for scheduling
            
            logger.debug("Task scheduled", 
                        task_id=task.task_id, 
                        node_id=best_node.node_id,
                        load_factor=best_node.get_load_factor())
        
        return assignments


class DistributedCampaignCoordinator:
    """Main distributed campaign coordination system."""
    
    def __init__(self, node_id: str = None, host: str = "localhost", port: int = 8080):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
        
        # Node information
        self.node_info = NodeInfo(
            node_id=self.node_id,
            host=host,
            port=port,
            role=NodeRole.COORDINATOR,
            capabilities={"scanning", "analysis", "reporting"}
        )
        
        # Cluster state
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.known_nodes[self.node_id] = self.node_info
        
        # Campaign management
        self.active_campaigns: Dict[str, DistributedCampaign] = {}
        self.completed_campaigns: List[DistributedCampaign] = []
        
        # Coordination components
        self.consensus_engine = SimpleMajorityConsensus()
        self.task_scheduler = CapabilityBasedScheduler()
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.pending_acks = {}  # message_id -> (timestamp, callback)
        
        # State
        self.running = False
        self.is_coordinator = True
        
        # Statistics
        self.stats = {
            "campaigns_coordinated": 0,
            "tasks_distributed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "consensus_decisions": 0,
            "node_failures_detected": 0
        }
    
    async def start_coordinator(self):
        """Start the distributed campaign coordinator."""
        self.running = True
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        message_processing_task = asyncio.create_task(self._message_processing_loop())
        health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        campaign_management_task = asyncio.create_task(self._campaign_management_loop())
        
        logger.info("Distributed campaign coordinator started", 
                   node_id=self.node_id,
                   endpoint=self.node_info.get_endpoint())
        
        try:
            await asyncio.gather(
                heartbeat_task,
                message_processing_task, 
                health_monitoring_task,
                campaign_management_task
            )
        except asyncio.CancelledError:
            logger.info("Distributed campaign coordinator stopped")
    
    async def stop_coordinator(self):
        """Stop the coordinator."""
        self.running = False
        
        # Complete active campaigns gracefully
        for campaign in self.active_campaigns.values():
            campaign.add_state_change(CampaignState.CANCELLED, "Coordinator shutdown")
    
    async def join_cluster(self, coordinator_endpoint: str):
        """Join an existing cluster."""
        try:
            # Send join request to coordinator
            join_message = CoordinationMessage(
                message_type=CoordinationMessageType.HEARTBEAT,
                sender_id=self.node_id,
                payload={
                    "action": "join_cluster",
                    "node_info": self.node_info.to_dict()
                }
            )
            
            await self._send_message_to_endpoint(coordinator_endpoint, join_message)
            self.is_coordinator = False
            
            logger.info("Joined cluster", coordinator=coordinator_endpoint)
            
        except Exception as e:
            logger.error("Failed to join cluster", error=str(e))
    
    async def create_distributed_campaign(self, title: str, description: str,
                                        campaign_type: str = "security_assessment",
                                        targets: List[Dict[str, Any]] = None,
                                        objectives: List[str] = None,
                                        tasks: List[Dict[str, Any]] = None) -> str:
        """Create a new distributed campaign."""
        
        campaign = DistributedCampaign(
            title=title,
            description=description,
            campaign_type=campaign_type,
            coordinator_node=self.node_id,
            targets=targets or [],
            objectives=objectives or []
        )
        
        # Create tasks
        if tasks:
            for task_data in tasks:
                task = DistributedTask(
                    campaign_id=campaign.campaign_id,
                    task_type=task_data.get("type", "generic"),
                    parameters=task_data.get("parameters", {}),
                    requirements=set(task_data.get("requirements", [])),
                    priority=task_data.get("priority", 1),
                    estimated_duration=task_data.get("estimated_duration", 300.0),
                    dependencies=task_data.get("dependencies", [])
                )
                campaign.tasks.append(task)
        
        # Propose campaign to cluster
        if len(self.known_nodes) > 1:
            proposal_id = f"campaign_{campaign.campaign_id}"
            participating_nodes = list(self.known_nodes.keys())
            
            await self.consensus_engine.propose_decision(
                proposal_id,
                {
                    "action": "create_campaign",
                    "campaign": campaign.to_dict()
                },
                participating_nodes
            )
            
            # Send proposal to all nodes
            proposal_message = CoordinationMessage(
                message_type=CoordinationMessageType.CAMPAIGN_PROPOSAL,
                sender_id=self.node_id,
                payload={
                    "proposal_id": proposal_id,
                    "campaign": campaign.to_dict()
                }
            )
            
            await self._broadcast_message(proposal_message)
            
            # Wait for consensus (simplified for demo)
            await asyncio.sleep(5)
            
            consensus_result = self.consensus_engine.get_consensus_result(proposal_id)
            if consensus_result:
                campaign.participating_nodes = set(participating_nodes)
                campaign.add_state_change(CampaignState.COORDINATING, "Consensus reached")
            else:
                campaign.add_state_change(CampaignState.FAILED, "Consensus not reached")
                return campaign.campaign_id
        
        # Add to active campaigns
        self.active_campaigns[campaign.campaign_id] = campaign
        self.stats["campaigns_coordinated"] += 1
        
        # Start campaign coordination
        await self._coordinate_campaign(campaign)
        
        logger.info("Distributed campaign created", 
                   campaign_id=campaign.campaign_id,
                   participating_nodes=len(campaign.participating_nodes))
        
        return campaign.campaign_id
    
    async def _coordinate_campaign(self, campaign: DistributedCampaign):
        """Coordinate the execution of a distributed campaign."""
        try:
            campaign.add_state_change(CampaignState.DISTRIBUTING, "Starting task distribution")
            
            # Get available nodes
            available_nodes = [node for node in self.known_nodes.values() if node.is_healthy()]
            
            if not available_nodes:
                campaign.add_state_change(CampaignState.FAILED, "No healthy nodes available")
                return
            
            # Schedule tasks
            task_assignments = await self.task_scheduler.schedule_tasks(campaign, available_nodes)
            
            if not task_assignments:
                campaign.add_state_change(CampaignState.FAILED, "No tasks could be scheduled")
                return
            
            # Send task assignments
            campaign.add_state_change(CampaignState.EXECUTING, "Tasks assigned and executing")
            campaign.started_at = time.time()
            
            for task_id, node_id in task_assignments.items():
                task = next(t for t in campaign.tasks if t.task_id == task_id)
                task.assigned_node = node_id
                task.assigned_at = time.time()
                task.status = "assigned"
                
                # Send task assignment message
                assignment_message = CoordinationMessage(
                    message_type=CoordinationMessageType.TASK_ASSIGNMENT,
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    payload={
                        "campaign_id": campaign.campaign_id,
                        "task": task.to_dict()
                    }
                )
                
                await self._send_message(assignment_message)
                self.stats["tasks_distributed"] += 1
            
            logger.info("Campaign tasks distributed", 
                       campaign_id=campaign.campaign_id,
                       tasks_assigned=len(task_assignments))
            
        except Exception as e:
            logger.error("Campaign coordination failed", 
                        campaign_id=campaign.campaign_id, 
                        error=str(e))
            campaign.add_state_change(CampaignState.FAILED, f"Coordination error: {str(e)}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain cluster health."""
        while self.running:
            try:
                # Update node health
                self.node_info.last_heartbeat = time.time()
                self.node_info.cpu_usage = random.uniform(0.1, 0.8)  # Simulated
                self.node_info.memory_usage = random.uniform(0.2, 0.7)  # Simulated
                
                # Send heartbeat to all known nodes
                heartbeat_message = CoordinationMessage(
                    message_type=CoordinationMessageType.HEARTBEAT,
                    sender_id=self.node_id,
                    payload={
                        "node_info": self.node_info.to_dict(),
                        "timestamp": time.time()
                    }
                )
                
                await self._broadcast_message(heartbeat_message)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error("Heartbeat loop failed", error=str(e))
                await asyncio.sleep(30)
    
    async def _message_processing_loop(self):
        """Process incoming coordination messages."""
        while self.running:
            try:
                # In a real implementation, this would receive messages from network
                # For demo, we'll process from queue
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._handle_coordination_message(message)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Message processing loop failed", error=str(e))
                await asyncio.sleep(5)
    
    async def _health_monitoring_loop(self):
        """Monitor cluster health and detect failures."""
        while self.running:
            try:
                current_time = time.time()
                failed_nodes = []
                
                for node_id, node_info in self.known_nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    # Check if node has missed heartbeats
                    if current_time - node_info.last_heartbeat > 90:  # 90 seconds timeout
                        if node_info.online:
                            node_info.online = False
                            failed_nodes.append(node_id)
                            self.stats["node_failures_detected"] += 1
                            
                            logger.warning("Node failure detected", node_id=node_id)
                
                # Handle node failures
                for node_id in failed_nodes:
                    await self._handle_node_failure(node_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Health monitoring loop failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _campaign_management_loop(self):
        """Manage ongoing campaigns."""
        while self.running:
            try:
                completed_campaigns = []
                
                for campaign_id, campaign in self.active_campaigns.items():
                    # Check campaign progress
                    completion_percentage = campaign.get_completion_percentage()
                    
                    # Check if campaign is completed
                    if completion_percentage >= 100.0:
                        campaign.add_state_change(CampaignState.COMPLETED, "All tasks completed")
                        campaign.completed_at = time.time()
                        completed_campaigns.append(campaign_id)
                    
                    # Check for deadline violations
                    elif campaign.deadline and time.time() > campaign.deadline:
                        campaign.add_state_change(CampaignState.FAILED, "Deadline exceeded")
                        completed_campaigns.append(campaign_id)
                    
                    # Handle failed tasks
                    failed_tasks = campaign.get_failed_tasks()
                    for task in failed_tasks:
                        if task.retry_count < task.max_retries:
                            # Retry task
                            task.retry_count += 1
                            task.status = "pending"
                            task.assigned_node = None
                            task.assigned_at = None
                            
                            logger.info("Retrying failed task", 
                                       task_id=task.task_id, 
                                       retry_count=task.retry_count)
                
                # Move completed campaigns
                for campaign_id in completed_campaigns:
                    campaign = self.active_campaigns.pop(campaign_id)
                    self.completed_campaigns.append(campaign)
                    
                    logger.info("Campaign completed", 
                               campaign_id=campaign_id,
                               completion_percentage=campaign.get_completion_percentage())
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Campaign management loop failed", error=str(e))
                await asyncio.sleep(30)
    
    async def _handle_coordination_message(self, message: CoordinationMessage):
        """Handle incoming coordination message."""
        self.stats["messages_received"] += 1
        
        try:
            if message.message_type == CoordinationMessageType.HEARTBEAT:
                await self._handle_heartbeat_message(message)
            
            elif message.message_type == CoordinationMessageType.CAMPAIGN_PROPOSAL:
                await self._handle_campaign_proposal(message)
            
            elif message.message_type == CoordinationMessageType.TASK_ASSIGNMENT:
                await self._handle_task_assignment(message)
            
            elif message.message_type == CoordinationMessageType.TASK_COMPLETION:
                await self._handle_task_completion(message)
            
            elif message.message_type == CoordinationMessageType.CONSENSUS_REQUEST:
                await self._handle_consensus_request(message)
            
            # Send acknowledgment if required
            if message.requires_ack:
                await self._send_acknowledgment(message)
        
        except Exception as e:
            logger.error("Failed to handle coordination message", 
                        message_type=message.message_type.value,
                        error=str(e))
    
    async def _handle_heartbeat_message(self, message: CoordinationMessage):
        """Handle heartbeat message."""
        sender_id = message.sender_id
        node_data = message.payload.get("node_info", {})
        
        if sender_id in self.known_nodes:
            # Update existing node
            node = self.known_nodes[sender_id]
            node.last_heartbeat = message.timestamp
            node.online = True
            node.cpu_usage = node_data.get("cpu_usage", 0.0)
            node.memory_usage = node_data.get("memory_usage", 0.0)
            node.health_score = node_data.get("health_score", 1.0)
        else:
            # New node joining
            if node_data:
                new_node = NodeInfo(**node_data)
                self.known_nodes[sender_id] = new_node
                
                logger.info("New node joined cluster", node_id=sender_id)
    
    async def _handle_campaign_proposal(self, message: CoordinationMessage):
        """Handle campaign proposal message."""
        proposal_id = message.payload.get("proposal_id")
        campaign_data = message.payload.get("campaign")
        
        if proposal_id and campaign_data:
            # Simple acceptance logic (in production, this would be more sophisticated)
            accept_proposal = self.node_info.get_load_factor() < 0.8
            
            # Vote on proposal
            await self.consensus_engine.vote_on_proposal(
                proposal_id, accept_proposal, self.node_id
            )
            
            # Send response
            response_message = CoordinationMessage(
                message_type=CoordinationMessageType.CAMPAIGN_ACCEPTANCE if accept_proposal 
                            else CoordinationMessageType.CAMPAIGN_REJECTION,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                payload={
                    "proposal_id": proposal_id,
                    "vote": accept_proposal
                }
            )
            
            await self._send_message(response_message)
    
    async def _handle_task_assignment(self, message: CoordinationMessage):
        """Handle task assignment message."""
        campaign_id = message.payload.get("campaign_id")
        task_data = message.payload.get("task")
        
        if task_data:
            # Simulate task execution
            task = DistributedTask(**task_data)
            task.status = "executing"
            task.started_at = time.time()
            
            # Simulate work
            await asyncio.sleep(random.uniform(1, 5))  # 1-5 seconds
            
            # Complete task
            task.status = "completed"
            task.completed_at = time.time()
            task.result = {"status": "success", "output": "Task completed successfully"}
            
            # Send completion message
            completion_message = CoordinationMessage(
                message_type=CoordinationMessageType.TASK_COMPLETION,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                payload={
                    "campaign_id": campaign_id,
                    "task": task.to_dict()
                }
            )
            
            await self._send_message(completion_message)
            
            logger.info("Task completed", 
                       task_id=task.task_id, 
                       campaign_id=campaign_id,
                       duration=task.get_duration())
    
    async def _handle_task_completion(self, message: CoordinationMessage):
        """Handle task completion message."""
        campaign_id = message.payload.get("campaign_id")
        task_data = message.payload.get("task")
        
        if campaign_id in self.active_campaigns and task_data:
            campaign = self.active_campaigns[campaign_id]
            task_id = task_data.get("task_id")
            
            # Update task status
            for task in campaign.tasks:
                if task.task_id == task_id:
                    task.status = task_data.get("status", "completed")
                    task.completed_at = task_data.get("completed_at")
                    task.result = task_data.get("result")
                    break
            
            logger.debug("Task completion recorded", 
                        task_id=task_id, 
                        campaign_id=campaign_id)
    
    async def _handle_consensus_request(self, message: CoordinationMessage):
        """Handle consensus request message."""
        # Implementation would depend on specific consensus algorithm
        pass
    
    async def _handle_node_failure(self, failed_node_id: str):
        """Handle node failure."""
        logger.warning("Handling node failure", node_id=failed_node_id)
        
        # Reassign tasks from failed node
        for campaign in self.active_campaigns.values():
            failed_tasks = [
                task for task in campaign.tasks 
                if task.assigned_node == failed_node_id and task.status in ["assigned", "executing"]
            ]
            
            for task in failed_tasks:
                task.status = "pending"
                task.assigned_node = None
                task.assigned_at = None
                
                logger.info("Task reassigned due to node failure", 
                           task_id=task.task_id, 
                           failed_node=failed_node_id)
    
    async def _broadcast_message(self, message: CoordinationMessage):
        """Broadcast message to all known nodes."""
        for node_id, node_info in self.known_nodes.items():
            if node_id != self.node_id and node_info.is_healthy():
                message.recipient_id = node_id
                await self._send_message(message)
    
    async def _send_message(self, message: CoordinationMessage):
        """Send message to specific node."""
        # In production, this would send over network
        # For demo, we'll add to local queue to simulate
        self.stats["messages_sent"] += 1
        
        CAMPAIGN_COORDINATION_EVENTS.labels(
            event_type=message.message_type.value,
            node_id=self.node_id
        ).inc()
        
        logger.debug("Message sent", 
                    message_type=message.message_type.value,
                    recipient=message.recipient_id)
    
    async def _send_message_to_endpoint(self, endpoint: str, message: CoordinationMessage):
        """Send message to specific endpoint."""
        # Placeholder for actual network communication
        pass
    
    async def _send_acknowledgment(self, original_message: CoordinationMessage):
        """Send acknowledgment for a message."""
        ack_message = CoordinationMessage(
            message_type=CoordinationMessageType.HEARTBEAT,  # Using heartbeat as ack
            sender_id=self.node_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.message_id,
            payload={"ack": True}
        )
        
        await self._send_message(ack_message)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        healthy_nodes = [node for node in self.known_nodes.values() if node.is_healthy()]
        
        return {
            "node_id": self.node_id,
            "is_coordinator": self.is_coordinator,
            "cluster_size": len(self.known_nodes),
            "healthy_nodes": len(healthy_nodes),
            "active_campaigns": len(self.active_campaigns),
            "completed_campaigns": len(self.completed_campaigns),
            "nodes": {node_id: node.to_dict() for node_id, node in self.known_nodes.items()},
            "statistics": self.stats
        }
    
    def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific campaign."""
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            # Check completed campaigns
            campaign = next(
                (c for c in self.completed_campaigns if c.campaign_id == campaign_id), 
                None
            )
        
        if campaign:
            return {
                "campaign_id": campaign.campaign_id,
                "title": campaign.title,
                "state": campaign.state.value,
                "completion_percentage": campaign.get_completion_percentage(),
                "participating_nodes": list(campaign.participating_nodes),
                "total_tasks": len(campaign.tasks),
                "completed_tasks": len([t for t in campaign.tasks if t.status == "completed"]),
                "failed_tasks": len([t for t in campaign.tasks if t.status == "failed"]),
                "created_at": campaign.created_at,
                "started_at": campaign.started_at,
                "completed_at": campaign.completed_at
            }
        
        return None


# Global distributed campaign coordinator
distributed_coordinator = DistributedCampaignCoordinator()


async def initialize_distributed_coordination():
    """Initialize the distributed campaign coordination system."""
    await distributed_coordinator.start_coordinator()


async def shutdown_distributed_coordination():
    """Shutdown the distributed coordination system."""
    await distributed_coordinator.stop_coordinator()


def get_distributed_coordinator() -> DistributedCampaignCoordinator:
    """Get the global distributed coordinator."""
    return distributed_coordinator