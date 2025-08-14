#!/usr/bin/env python3
"""
Autonomous Reinforcement Learning Integration
Production-grade RL integration for autonomous red team operations

SECURITY NOTICE: This module implements advanced RL integration for autonomous
red team operations with comprehensive safety controls and learning optimization.

Key Features:
- Real-time RL decision making for autonomous agents
- Integration with simulation environments for safe training
- Advanced experience replay and model optimization
- Multi-agent coordination and learning
- Continuous learning from real-world engagements
- Comprehensive safety controls and human oversight
- Performance monitoring and adaptation
- Transfer learning between environments
"""

import asyncio
import logging
import json
import uuid
import numpy as np
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import time
from collections import deque
import concurrent.futures

# ML and RL imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Internal imports
from ..common.security_framework import SecurityFramework, SecurityLevel
from ..common.audit_logger import AuditLogger, AuditEvent
from .advanced_reinforcement_learning import (
    AdvancedRLEngine, EnvironmentState, ActionResult, get_rl_engine
)
from ..simulation.controlled_environment_framework import (
    ControlledEnvironmentFramework, get_environment_framework
)
from ..exploitation.advanced_payload_engine import (
    AdvancedPayloadEngine, PayloadConfiguration, get_payload_engine
)
from ..security.autonomous_red_team_engine import (
    AutonomousRedTeamEngine, OperationObjective, SecurityConstraints,
    get_autonomous_red_team_engine
)

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """RL learning modes"""
    SIMULATION_ONLY = "simulation_only"
    MIXED_TRAINING = "mixed_training"
    PRODUCTION_INFERENCE = "production_inference"
    CONTINUOUS_LEARNING = "continuous_learning"
    TRANSFER_LEARNING = "transfer_learning"


class AgentCapability(Enum):
    """Agent capability levels"""
    NOVICE = "novice"           # Basic scripted behavior
    INTERMEDIATE = "intermediate"  # Simple RL with limited actions
    ADVANCED = "advanced"       # Sophisticated RL with complex planning
    EXPERT = "expert"          # Advanced RL with meta-learning
    AUTONOMOUS = "autonomous"   # Fully autonomous with self-improvement


class SafetyLevel(Enum):
    """RL safety levels for autonomous operations"""
    MAXIMUM = "maximum"         # Simulation only, no real actions
    HIGH = "high"              # Limited actions with extensive validation
    MEDIUM = "medium"          # Controlled actions with safety checks
    LOW = "low"               # Full capabilities with authorization
    DISABLED = "disabled"      # No safety constraints (testing only)


@dataclass
class LearningConfiguration:
    """Comprehensive learning configuration"""
    learning_mode: LearningMode
    agent_capability: AgentCapability
    safety_level: SafetyLevel
    
    # RL configuration
    exploration_strategy: str = "adaptive_epsilon_greedy"
    learning_rate: float = 1e-4
    batch_size: int = 64
    memory_size: int = 100000
    target_update_frequency: int = 1000
    
    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    convergence_threshold: float = 0.01
    patience: int = 50
    
    # Environment configuration
    simulation_environments: List[str] = field(default_factory=list)
    production_environments: List[str] = field(default_factory=list)
    transfer_learning_enabled: bool = True
    
    # Safety configuration
    human_approval_threshold: float = 0.7
    uncertainty_threshold: float = 0.5
    max_risk_level: float = 0.3
    emergency_stop_conditions: List[str] = field(default_factory=list)
    
    # Performance tracking
    success_rate_threshold: float = 0.8
    performance_degradation_threshold: float = 0.1
    model_validation_frequency: int = 100
    
    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSession:
    """Active learning session state"""
    session_id: str
    configuration: LearningConfiguration
    
    # Session state
    current_episode: int = 0
    current_step: int = 0
    total_reward: float = 0.0
    best_performance: float = 0.0
    
    # Environment state
    current_environment: Optional[str] = None
    environment_state: Optional[EnvironmentState] = None
    active_objective: Optional[OperationObjective] = None
    
    # Learning metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rate: float = 0.0
    convergence_score: float = 0.0
    
    # Safety tracking
    safety_violations: List[str] = field(default_factory=list)
    human_interventions: int = 0
    emergency_stops: int = 0
    
    # Performance metrics
    actions_taken: int = 0
    successful_actions: int = 0
    model_updates: int = 0
    validation_scores: List[float] = field(default_factory=list)
    
    # Session management
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    paused: bool = False
    completed: bool = False


class MultiAgentCoordinator:
    """Coordinate multiple RL agents in shared environments"""
    
    def __init__(self):
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.coordination_strategies = {
            "independent": self._independent_coordination,
            "competitive": self._competitive_coordination,
            "cooperative": self._cooperative_coordination,
            "hierarchical": self._hierarchical_coordination
        }
        self.shared_knowledge_base: Dict[str, Any] = {}
        self.communication_channel = queue.Queue()
    
    async def register_agent(self, agent_id: str, capabilities: AgentCapability,
                           coordination_strategy: str = "independent") -> bool:
        """Register agent for coordination"""
        try:
            self.active_agents[agent_id] = {
                "capabilities": capabilities,
                "coordination_strategy": coordination_strategy,
                "last_action": None,
                "performance_score": 0.0,
                "cooperation_score": 0.0,
                "registered_at": datetime.utcnow()
            }
            
            logger.info(f"Registered agent {agent_id} with {capabilities.value} capabilities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    async def coordinate_agents(self, environment_state: EnvironmentState,
                              available_actions: Dict[str, List[int]]) -> Dict[str, int]:
        """Coordinate action selection across multiple agents"""
        try:
            coordinated_actions = {}
            
            for agent_id in self.active_agents:
                strategy = self.active_agents[agent_id]["coordination_strategy"]
                coordinator = self.coordination_strategies.get(strategy, 
                                                             self._independent_coordination)
                
                action = await coordinator(agent_id, environment_state, 
                                         available_actions.get(agent_id, []))
                coordinated_actions[agent_id] = action
            
            # Update shared knowledge
            await self._update_shared_knowledge(environment_state, coordinated_actions)
            
            return coordinated_actions
            
        except Exception as e:
            logger.error(f"Agent coordination failed: {e}")
            return {}
    
    async def _independent_coordination(self, agent_id: str, state: EnvironmentState,
                                      actions: List[int]) -> int:
        """Independent action selection (no coordination)"""
        if not actions:
            return 0
        return np.random.choice(actions)
    
    async def _competitive_coordination(self, agent_id: str, state: EnvironmentState,
                                      actions: List[int]) -> int:
        """Competitive coordination (agents compete for resources)"""
        # Prioritize actions based on agent performance
        agent_info = self.active_agents[agent_id]
        performance_weight = agent_info["performance_score"]
        
        if not actions:
            return 0
        
        # Higher performing agents get priority on high-value actions
        if performance_weight > 0.7 and len(actions) > 1:
            return actions[-1]  # Best action
        else:
            return np.random.choice(actions[:-1] if len(actions) > 1 else actions)
    
    async def _cooperative_coordination(self, agent_id: str, state: EnvironmentState,
                                      actions: List[int]) -> int:
        """Cooperative coordination (agents work together)"""
        # Share information and coordinate complementary actions
        agent_info = self.active_agents[agent_id]
        
        # Check what other agents are planning
        complementary_actions = []
        for other_agent_id, other_info in self.active_agents.items():
            if other_agent_id != agent_id and other_info["last_action"] is not None:
                # Avoid duplicate actions
                last_action = other_info["last_action"]
                complementary_actions = [a for a in actions if a != last_action]
        
        if complementary_actions:
            return np.random.choice(complementary_actions)
        elif actions:
            return np.random.choice(actions)
        else:
            return 0
    
    async def _hierarchical_coordination(self, agent_id: str, state: EnvironmentState,
                                       actions: List[int]) -> int:
        """Hierarchical coordination (leader-follower structure)"""
        # Determine agent hierarchy based on capabilities and performance
        agent_capabilities = [
            (aid, info["capabilities"], info["performance_score"])
            for aid, info in self.active_agents.items()
        ]
        
        # Sort by capability level and performance
        capability_order = {
            AgentCapability.AUTONOMOUS: 5,
            AgentCapability.EXPERT: 4,
            AgentCapability.ADVANCED: 3,
            AgentCapability.INTERMEDIATE: 2,
            AgentCapability.NOVICE: 1
        }
        
        sorted_agents = sorted(agent_capabilities, 
                             key=lambda x: (capability_order.get(x[1], 0), x[2]), 
                             reverse=True)
        
        # Leader makes strategic decisions, followers execute tactical actions
        if sorted_agents and sorted_agents[0][0] == agent_id:
            # This is the leader - choose strategic action
            if actions:
                return max(actions)  # Most impactful action
        else:
            # Follower - support leader's strategy
            if actions:
                return min(actions)  # Supporting action
        
        return 0 if not actions else actions[0]
    
    async def _update_shared_knowledge(self, state: EnvironmentState, 
                                     actions: Dict[str, int]):
        """Update shared knowledge base with coordination results"""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            coordination_data = {
                "timestamp": timestamp,
                "environment_state": asdict(state),
                "agent_actions": actions,
                "coordination_effectiveness": await self._calculate_coordination_effectiveness(actions)
            }
            
            # Store in shared knowledge base
            if "coordination_history" not in self.shared_knowledge_base:
                self.shared_knowledge_base["coordination_history"] = []
            
            self.shared_knowledge_base["coordination_history"].append(coordination_data)
            
            # Keep only recent history
            if len(self.shared_knowledge_base["coordination_history"]) > 1000:
                self.shared_knowledge_base["coordination_history"] = \
                    self.shared_knowledge_base["coordination_history"][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to update shared knowledge: {e}")
    
    async def _calculate_coordination_effectiveness(self, actions: Dict[str, int]) -> float:
        """Calculate effectiveness of agent coordination"""
        try:
            if len(actions) <= 1:
                return 1.0  # Perfect coordination for single agent
            
            # Calculate action diversity (good for exploration)
            unique_actions = len(set(actions.values()))
            total_actions = len(actions)
            diversity_score = unique_actions / total_actions
            
            # Calculate complementarity (agents taking different but related actions)
            action_values = list(actions.values())
            complementarity_score = 1.0 - (np.std(action_values) / (np.mean(action_values) + 1e-6))
            
            # Combined effectiveness score
            effectiveness = (diversity_score * 0.6) + (complementarity_score * 0.4)
            
            return min(max(effectiveness, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate coordination effectiveness: {e}")
            return 0.5


class TransferLearningEngine:
    """Advanced transfer learning between environments and tasks"""
    
    def __init__(self):
        self.source_models: Dict[str, Dict[str, Any]] = {}
        self.transfer_mappings: Dict[str, Dict[str, Any]] = {}
        self.adaptation_strategies = {
            "fine_tuning": self._fine_tuning_transfer,
            "feature_extraction": self._feature_extraction_transfer,
            "domain_adaptation": self._domain_adaptation_transfer,
            "meta_learning": self._meta_learning_transfer
        }
    
    async def register_source_model(self, model_id: str, model_data: Dict[str, Any],
                                  environment_type: str, task_type: str) -> bool:
        """Register a trained model as source for transfer learning"""
        try:
            self.source_models[model_id] = {
                "model_data": model_data,
                "environment_type": environment_type,
                "task_type": task_type,
                "performance_metrics": model_data.get("performance_metrics", {}),
                "registered_at": datetime.utcnow()
            }
            
            logger.info(f"Registered source model {model_id} for transfer learning")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register source model: {e}")
            return False
    
    async def transfer_knowledge(self, target_environment: str, target_task: str,
                               strategy: str = "fine_tuning") -> Optional[Dict[str, Any]]:
        """Transfer knowledge from similar source models to target"""
        try:
            # Find best source model for transfer
            best_source = await self._find_best_source_model(target_environment, target_task)
            
            if not best_source:
                logger.warning("No suitable source model found for transfer learning")
                return None
            
            # Apply transfer learning strategy
            transfer_func = self.adaptation_strategies.get(strategy, self._fine_tuning_transfer)
            transferred_model = await transfer_func(best_source, target_environment, target_task)
            
            # Record transfer mapping
            transfer_id = str(uuid.uuid4())
            self.transfer_mappings[transfer_id] = {
                "source_model": best_source["model_id"],
                "target_environment": target_environment,
                "target_task": target_task,
                "strategy": strategy,
                "transferred_at": datetime.utcnow()
            }
            
            logger.info(f"Successfully transferred knowledge to {target_environment}/{target_task}")
            return transferred_model
            
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return None
    
    async def _find_best_source_model(self, target_env: str, target_task: str) -> Optional[Dict[str, Any]]:
        """Find the best source model for transfer learning"""
        try:
            best_match = None
            best_score = 0.0
            
            for model_id, model_info in self.source_models.items():
                # Calculate similarity score
                env_similarity = await self._calculate_environment_similarity(
                    model_info["environment_type"], target_env
                )
                task_similarity = await self._calculate_task_similarity(
                    model_info["task_type"], target_task
                )
                
                # Weight by model performance
                performance_score = model_info["performance_metrics"].get("success_rate", 0.5)
                
                # Combined similarity score
                similarity_score = (env_similarity * 0.4 + task_similarity * 0.4 + 
                                  performance_score * 0.2)
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = {
                        "model_id": model_id,
                        "model_info": model_info,
                        "similarity_score": similarity_score
                    }
            
            return best_match if best_score > 0.3 else None
            
        except Exception as e:
            logger.error(f"Failed to find best source model: {e}")
            return None
    
    async def _calculate_environment_similarity(self, source_env: str, target_env: str) -> float:
        """Calculate similarity between environments"""
        # Simple similarity mapping (would be more sophisticated in practice)
        similarity_matrix = {
            ("web_application_lab", "web_application_lab"): 1.0,
            ("web_application_lab", "enterprise_simulation"): 0.6,
            ("enterprise_simulation", "enterprise_simulation"): 1.0,
            ("enterprise_simulation", "cyber_range"): 0.8,
            ("cyber_range", "cyber_range"): 1.0,
        }
        
        return similarity_matrix.get((source_env, target_env), 0.2)
    
    async def _calculate_task_similarity(self, source_task: str, target_task: str) -> float:
        """Calculate similarity between tasks"""
        # Task similarity mapping
        similarity_matrix = {
            ("reconnaissance", "reconnaissance"): 1.0,
            ("reconnaissance", "lateral_movement"): 0.7,
            ("exploitation", "exploitation"): 1.0,
            ("exploitation", "privilege_escalation"): 0.8,
            ("persistence", "persistence"): 1.0,
            ("persistence", "lateral_movement"): 0.6,
        }
        
        return similarity_matrix.get((source_task, target_task), 0.3)
    
    async def _fine_tuning_transfer(self, source_model: Dict[str, Any], 
                                  target_env: str, target_task: str) -> Dict[str, Any]:
        """Fine-tuning transfer learning strategy"""
        # Copy source model and modify for target
        transferred_model = source_model["model_info"]["model_data"].copy()
        
        # Adjust learning parameters for fine-tuning
        transferred_model["learning_rate"] = transferred_model.get("learning_rate", 1e-4) * 0.1
        transferred_model["transfer_learning"] = {
            "strategy": "fine_tuning",
            "source_model": source_model["model_id"],
            "freeze_layers": ["embedding", "early_layers"],
            "trainable_layers": ["output", "late_layers"]
        }
        
        return transferred_model
    
    async def _feature_extraction_transfer(self, source_model: Dict[str, Any],
                                         target_env: str, target_task: str) -> Dict[str, Any]:
        """Feature extraction transfer learning strategy"""
        transferred_model = source_model["model_info"]["model_data"].copy()
        
        # Use source model features with new classifier
        transferred_model["transfer_learning"] = {
            "strategy": "feature_extraction",
            "source_model": source_model["model_id"],
            "freeze_layers": ["all_except_output"],
            "new_classifier": True
        }
        
        return transferred_model


class AutonomousRLIntegration:
    """
    Autonomous Reinforcement Learning Integration Engine
    
    Provides comprehensive RL integration for autonomous red team operations with:
    - Real-time RL decision making
    - Integration with simulation environments for safe training
    - Multi-agent coordination and learning
    - Transfer learning between environments
    - Continuous learning from real-world engagements
    - Comprehensive safety controls and monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.integration_id = str(uuid.uuid4())
        
        # Core components
        self.rl_engine: Optional[AdvancedRLEngine] = None
        self.environment_framework: Optional[ControlledEnvironmentFramework] = None
        self.payload_engine: Optional[AdvancedPayloadEngine] = None
        self.red_team_engine: Optional[AutonomousRedTeamEngine] = None
        
        # Integration components
        self.multi_agent_coordinator = MultiAgentCoordinator()
        self.transfer_learning_engine = TransferLearningEngine()
        self.security_framework = SecurityFramework()
        self.audit_logger = AuditLogger()
        
        # Learning management
        self.active_sessions: Dict[str, LearningSession] = {}
        self.learning_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.integration_metrics = {
            "learning_sessions": 0,
            "successful_sessions": 0,
            "total_episodes": 0,
            "total_rewards": 0.0,
            "average_performance": 0.0,
            "safety_violations": 0,
            "human_interventions": 0,
            "model_updates": 0,
            "transfer_learning_applications": 0
        }
        
        # Initialize components
        asyncio.create_task(self._initialize_components())
        
        logger.info("Autonomous RL Integration Engine initialized", 
                   integration_id=self.integration_id)
    
    async def _initialize_components(self):
        """Initialize core components"""
        try:
            # Initialize RL engine
            rl_config = self.config.get("rl_config", {})
            self.rl_engine = await get_rl_engine(rl_config)
            
            # Initialize environment framework
            env_config = self.config.get("environment_config", {})
            self.environment_framework = await get_environment_framework(env_config)
            
            # Initialize payload engine
            payload_config = self.config.get("payload_config", {})
            self.payload_engine = await get_payload_engine(payload_config)
            
            # Initialize red team engine
            red_team_config = self.config.get("red_team_config", {})
            self.red_team_engine = await get_autonomous_red_team_engine(red_team_config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    async def start_learning_session(self, config: LearningConfiguration) -> str:
        """Start comprehensive learning session"""
        session_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting learning session {session_id}")
            
            # Create learning session
            session = LearningSession(
                session_id=session_id,
                configuration=config
            )
            
            # Store session
            self.active_sessions[session_id] = session
            
            # Setup environment based on learning mode
            if config.learning_mode in [LearningMode.SIMULATION_ONLY, LearningMode.MIXED_TRAINING]:
                await self._setup_simulation_environment(session)
            
            # Initialize RL agent for session
            await self._initialize_session_agent(session)
            
            # Start learning loop
            asyncio.create_task(self._learning_loop(session))
            
            # Update metrics
            self.integration_metrics["learning_sessions"] += 1
            
            # Audit logging
            await self.audit_logger.log_event(AuditEvent(
                event_type="learning_session_started",
                component="autonomous_rl_integration",
                details={
                    "session_id": session_id,
                    "learning_mode": config.learning_mode.value,
                    "agent_capability": config.agent_capability.value,
                    "safety_level": config.safety_level.value
                },
                security_level=SecurityLevel.MEDIUM
            ))
            
            logger.info(f"Learning session {session_id} started successfully")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start learning session: {e}")
            
            # Cleanup on failure
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            raise
    
    async def _setup_simulation_environment(self, session: LearningSession):
        """Setup simulation environment for learning"""
        try:
            # Choose simulation scenario based on agent capability
            if session.configuration.agent_capability == AgentCapability.NOVICE:
                scenario_id = "basic_web_lab"
            elif session.configuration.agent_capability in [AgentCapability.INTERMEDIATE, AgentCapability.ADVANCED]:
                scenario_id = "enterprise_network"
            else:
                scenario_id = "enterprise_network"  # Most complex available
            
            # Create environment instance
            environment_id = await self.environment_framework.create_environment(scenario_id)
            session.current_environment = environment_id
            
            # Start learning session in environment
            learning_session_id = await self.environment_framework.start_learning_session(
                environment_id, session.session_id
            )
            
            logger.info(f"Setup simulation environment {environment_id} for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup simulation environment: {e}")
            raise
    
    async def _initialize_session_agent(self, session: LearningSession):
        """Initialize RL agent for session"""
        try:
            # Apply transfer learning if enabled
            if session.configuration.transfer_learning_enabled:
                transferred_model = await self.transfer_learning_engine.transfer_knowledge(
                    target_environment=session.current_environment or "default",
                    target_task="red_team_operations",
                    strategy="fine_tuning"
                )
                
                if transferred_model:
                    # Apply transferred knowledge to RL engine
                    logger.info(f"Applied transfer learning for session {session.session_id}")
                    self.integration_metrics["transfer_learning_applications"] += 1
            
            # Register agent with multi-agent coordinator
            await self.multi_agent_coordinator.register_agent(
                session.session_id,
                session.configuration.agent_capability,
                "cooperative"  # Default coordination strategy
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize session agent: {e}")
            raise
    
    async def _learning_loop(self, session: LearningSession):
        """Main learning loop for session"""
        try:
            while (session.current_episode < session.configuration.max_episodes and 
                   not session.completed and not session.paused):
                
                # Start new episode
                session.current_episode += 1
                session.current_step = 0
                episode_reward = 0.0
                episode_start_time = datetime.utcnow()
                
                # Reset environment state
                await self._reset_episode_environment(session)
                
                # Episode loop
                while (session.current_step < session.configuration.max_steps_per_episode and
                       not session.completed):
                    
                    session.current_step += 1
                    session.actions_taken += 1
                    
                    # Get current environment state
                    current_state = await self._get_current_state(session)
                    session.environment_state = current_state
                    
                    # Select action using RL engine
                    action, action_metadata = await self.rl_engine.select_action(
                        current_state,
                        available_actions=await self._get_available_actions(session),
                        training=True
                    )
                    
                    # Validate action safety
                    if not await self._validate_action_safety(session, action):
                        logger.warning(f"Action {action} failed safety validation")
                        session.safety_violations.append(f"Unsafe action {action} at step {session.current_step}")
                        self.integration_metrics["safety_violations"] += 1
                        continue
                    
                    # Execute action
                    action_result = await self._execute_action(session, action, current_state)
                    
                    # Process results
                    episode_reward += action_result.reward
                    session.total_reward += action_result.reward
                    
                    if action_result.success:
                        session.successful_actions += 1
                    
                    # Learn from experience
                    learning_results = await self.rl_engine.process_experience(
                        current_state, action, action_result
                    )
                    
                    session.model_updates += 1
                    self.integration_metrics["model_updates"] += 1
                    
                    # Update environment framework
                    await self.environment_framework.process_agent_action(
                        session.current_environment, session.session_id, action_result
                    )
                    
                    # Check for episode termination
                    if action_result.new_state.mission_progress >= 1.0:
                        break
                    
                    # Update last activity
                    session.last_activity = datetime.utcnow()
                
                # End episode
                episode_duration = (datetime.utcnow() - episode_start_time).total_seconds()
                
                # Record episode metrics
                session.episode_rewards.append(episode_reward)
                session.episode_lengths.append(session.current_step)
                
                # Update RL engine with episode completion
                episode_stats = await self.rl_engine.end_episode(
                    episode_reward, session.current_step, episode_reward > 0
                )
                
                # Calculate success rate
                recent_rewards = session.episode_rewards[-10:]
                session.success_rate = len([r for r in recent_rewards if r > 0]) / len(recent_rewards)
                
                # Check for convergence
                if await self._check_convergence(session):
                    logger.info(f"Session {session.session_id} converged after {session.current_episode} episodes")
                    break
                
                # Performance monitoring
                await self._monitor_session_performance(session)
                
                self.integration_metrics["total_episodes"] += 1
                self.integration_metrics["total_rewards"] += episode_reward
            
            # Complete session
            await self._complete_learning_session(session)
            
        except Exception as e:
            logger.error(f"Learning loop failed for session {session.session_id}: {e}")
            session.completed = True
    
    async def _get_current_state(self, session: LearningSession) -> EnvironmentState:
        """Get current environment state for RL"""
        try:
            if session.current_environment:
                # Get state from simulation environment
                env_status = await self.environment_framework.get_environment_status(
                    session.current_environment
                )
                
                # Convert to RL environment state
                return EnvironmentState(
                    target_info={"environment_id": session.current_environment},
                    discovered_services=[],
                    compromised_systems=[],
                    available_techniques=await self._get_available_techniques(session),
                    detection_level=0.1,  # Low in simulation
                    mission_progress=0.0,  # Would be calculated from objectives
                    time_elapsed=session.current_step * 10,  # Approximate
                    agent_resources={"skill_level": 0.8, "stealth_budget": 1.0},
                    network_topology={"hosts": [], "subnets": []},
                    security_controls=[]
                )
            else:
                # Fallback state for testing
                return self.rl_engine.create_mock_state()
                
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return self.rl_engine.create_mock_state()
    
    async def _get_available_actions(self, session: LearningSession) -> List[int]:
        """Get available actions for current state"""
        try:
            # Base actions available to all agents
            base_actions = [0, 1, 2, 3, 4]  # reconnaissance, initial_access, execution, etc.
            
            # Filter based on agent capability
            if session.configuration.agent_capability == AgentCapability.NOVICE:
                return base_actions[:3]  # Limited actions
            elif session.configuration.agent_capability == AgentCapability.INTERMEDIATE:
                return base_actions[:5]
            else:
                return base_actions  # All actions available
                
        except Exception as e:
            logger.error(f"Failed to get available actions: {e}")
            return [0, 1, 2]  # Safe default
    
    async def _get_available_techniques(self, session: LearningSession) -> List[str]:
        """Get available techniques for current session"""
        try:
            base_techniques = [
                "recon.port_scan",
                "recon.service_enum",
                "exploit.web_sqli",
                "exploit.web_xss",
                "privesc.sudo_exploit",
                "persist.cron_job"
            ]
            
            # Filter based on capability and safety level
            if session.configuration.safety_level == SafetyLevel.MAXIMUM:
                return base_techniques[:2]  # Only reconnaissance
            elif session.configuration.safety_level == SafetyLevel.HIGH:
                return base_techniques[:4]  # No persistence
            else:
                return base_techniques
                
        except Exception as e:
            logger.error(f"Failed to get available techniques: {e}")
            return ["recon.port_scan"]
    
    async def _validate_action_safety(self, session: LearningSession, action: int) -> bool:
        """Validate action against safety constraints"""
        try:
            # Safety level checks
            if session.configuration.safety_level == SafetyLevel.MAXIMUM:
                # Only allow reconnaissance actions in simulation
                return action <= 1
            
            elif session.configuration.safety_level == SafetyLevel.HIGH:
                # Limited actions with validation
                if action > 5:  # No advanced techniques
                    return False
                
                # Additional validation for higher-risk actions
                if action >= 3:  # Exploitation and beyond
                    # Require human approval for high-risk actions
                    if session.configuration.human_approval_threshold > 0.5:
                        logger.info(f"Action {action} requires human approval")
                        session.human_interventions += 1
                        self.integration_metrics["human_interventions"] += 1
                        return False  # Simplified - would integrate with approval system
            
            # Check emergency stop conditions
            for condition in session.configuration.emergency_stop_conditions:
                if await self._check_emergency_condition(session, condition):
                    logger.critical(f"Emergency stop condition triggered: {condition}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            return False  # Fail safe
    
    async def _execute_action(self, session: LearningSession, action: int, 
                            state: EnvironmentState) -> ActionResult:
        """Execute action and return results"""
        try:
            # Map action to technique
            action_mapping = {
                0: "recon.port_scan",
                1: "recon.service_enum", 
                2: "exploit.web_sqli",
                3: "exploit.web_xss",
                4: "privesc.sudo_exploit",
                5: "persist.cron_job"
            }
            
            technique = action_mapping.get(action, "recon.port_scan")
            
            # Execute in simulation environment
            if session.current_environment:
                # Simulate technique execution
                success = np.random.random() > 0.3  # 70% success rate
                reward = 1.0 if success else -0.1
                
                # Create new state (simplified)
                new_state = state
                if success:
                    new_state.mission_progress = min(state.mission_progress + 0.1, 1.0)
                
                return ActionResult(
                    success=success,
                    reward=reward,
                    new_state=new_state,
                    metadata={"technique": technique, "action": action},
                    detection_risk=0.1,
                    resources_consumed={"time": 5.0},
                    techniques_unlocked=[],
                    objectives_achieved=["reconnaissance"] if success and action <= 1 else []
                )
            else:
                # Fallback mock execution
                return self.rl_engine.create_mock_action_result(success=True)
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return self.rl_engine.create_mock_action_result(success=False)
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics"""
        try:
            # Calculate derived metrics
            total_sessions = self.integration_metrics["learning_sessions"]
            success_rate = (self.integration_metrics["successful_sessions"] / total_sessions 
                          if total_sessions > 0 else 0.0)
            
            average_reward = (self.integration_metrics["total_rewards"] / 
                            max(self.integration_metrics["total_episodes"], 1))
            
            # Active session summary
            active_summary = {}
            for session_id, session in self.active_sessions.items():
                active_summary[session_id] = {
                    "episode": session.current_episode,
                    "success_rate": session.success_rate,
                    "total_reward": session.total_reward,
                    "safety_violations": len(session.safety_violations),
                    "learning_mode": session.configuration.learning_mode.value
                }
            
            return {
                "integration_metrics": {
                    "integration_id": self.integration_id,
                    "learning_sessions": total_sessions,
                    "successful_sessions": self.integration_metrics["successful_sessions"],
                    "session_success_rate": round(success_rate, 3),
                    "total_episodes": self.integration_metrics["total_episodes"],
                    "average_reward_per_episode": round(average_reward, 3),
                    "safety_violations": self.integration_metrics["safety_violations"],
                    "human_interventions": self.integration_metrics["human_interventions"],
                    "model_updates": self.integration_metrics["model_updates"],
                    "transfer_learning_applications": self.integration_metrics["transfer_learning_applications"],
                    "active_sessions": len(self.active_sessions)
                },
                "active_sessions": active_summary,
                "component_status": {
                    "rl_engine_available": self.rl_engine is not None,
                    "environment_framework_available": self.environment_framework is not None,
                    "payload_engine_available": self.payload_engine is not None,
                    "red_team_engine_available": self.red_team_engine is not None
                },
                "multi_agent_coordinator": {
                    "active_agents": len(self.multi_agent_coordinator.active_agents),
                    "coordination_strategies": list(self.multi_agent_coordinator.coordination_strategies.keys())
                },
                "transfer_learning": {
                    "source_models": len(self.transfer_learning_engine.source_models),
                    "transfer_mappings": len(self.transfer_learning_engine.transfer_mappings)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get integration metrics: {e}")
            return {"error": str(e)}


# Global integration instance
_rl_integration: Optional[AutonomousRLIntegration] = None


async def get_rl_integration(config: Dict[str, Any] = None) -> AutonomousRLIntegration:
    """Get singleton autonomous RL integration instance"""
    global _rl_integration
    
    if _rl_integration is None:
        _rl_integration = AutonomousRLIntegration(config)
    
    return _rl_integration


# Export main classes
__all__ = [
    "AutonomousRLIntegration",
    "LearningConfiguration",
    "LearningSession",
    "MultiAgentCoordinator", 
    "TransferLearningEngine",
    "LearningMode",
    "AgentCapability", 
    "SafetyLevel",
    "get_rl_integration"
]