#!/usr/bin/env python3
"""
Deep Q-Network Agent Selection for Xorb 2.0
EPYC-optimized reinforcement learning for intelligent agent scheduling

This module implements a DQN-based agent selection system that learns
from campaign outcomes to optimize agent allocation and improve success rates.
Designed specifically for AMD EPYC processors with high core counts.
"""

import asyncio
import logging
import numpy as np
import pickle
import random
from collections import deque, namedtuple
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

from .orchestrator import Campaign, CampaignStatus
from ..agents.base_agent import AgentCapability


@dataclass
class AgentSelectionState:
    """State representation for DQN agent selection"""
    target_complexity: float  # 0.0-1.0
    target_security_score: float  # 0.0-1.0
    available_agents: int
    campaign_priority: float  # 0.0-1.0
    time_constraints: float  # 0.0-1.0 (urgency)
    resource_utilization: float  # Current EPYC core usage
    historical_success_rate: float  # For this target type
    knowledge_confidence: float  # From knowledge fabric
    current_queue_length: int
    epyc_numa_locality: float  # NUMA memory locality score


@dataclass
class AgentSelectionAction:
    """Action representation for agent selection"""
    agent_type: str
    resource_allocation: float  # Fraction of available resources
    parallelism_level: int  # Number of parallel instances
    priority_boost: float  # Priority adjustment


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class EPYCOptimizedDQN(nn.Module):
    """
    Deep Q-Network optimized for EPYC processors
    Uses layer sizes that align with EPYC cache hierarchy
    """
    
    def __init__(self, state_size: int, action_size: int, epyc_cores: int = 64):
        super(EPYCOptimizedDQN, self).__init__()
        
        # Network architecture optimized for EPYC cache hierarchy
        # L1: 32KB per core, L2: 512KB per core, L3: 256MB shared
        hidden1_size = min(epyc_cores * 4, 256)  # Fits in L2 cache per core
        hidden2_size = min(epyc_cores * 2, 128)  # Efficient for CCX structure
        hidden3_size = min(epyc_cores, 64)       # Aligns with core count
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden1_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2_size),
            nn.Dropout(0.1),
            
            nn.Linear(hidden2_size, hidden3_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden3_size),
            
            nn.Linear(hidden3_size, action_size)
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer with EPYC-optimized memory layout"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgentSelector:
    """
    Deep Q-Network based agent selection system
    Optimized for AMD EPYC processors with high core counts
    """
    
    def __init__(self, 
                 state_size: int = 10,
                 action_size: int = 20,
                 epyc_cores: int = 64,
                 learning_rate: float = 0.001,
                 model_path: str = "./models/dqn_agent_selector"):
        
        self.state_size = state_size
        self.action_size = action_size
        self.epyc_cores = epyc_cores
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # DQN hyperparameters
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.target_update_frequency = 100
        self.training_frequency = 4
        
        # EPYC-specific optimizations
        self.numa_nodes = 2  # Typical EPYC configuration
        self.ccx_per_numa = 4  # CCX (Core Complex) per NUMA node
        self.cores_per_ccx = epyc_cores // (self.numa_nodes * self.ccx_per_numa)
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=50000)
        self.training_step = 0
        
        # Initialize models if PyTorch is available
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._initialize_networks()
        else:
            self.logger.warning("PyTorch not available. Using fallback selection strategy.")
            self.device = None
            self.q_network = None
            self.target_network = None
    
    def _initialize_networks(self):
        """Initialize main and target Q-networks"""
        self.q_network = EPYCOptimizedDQN(
            self.state_size, 
            self.action_size, 
            self.epyc_cores
        ).to(self.device)
        
        self.target_network = EPYCOptimizedDQN(
            self.state_size, 
            self.action_size, 
            self.epyc_cores
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with EPYC-specific learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Try to load existing model
        self._load_model()
    
    async def select_agents(self, 
                          campaign: Campaign, 
                          available_agents: Dict[str, Any],
                          system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select optimal agents for a campaign using DQN
        
        Args:
            campaign: Campaign requiring agent selection
            available_agents: Dictionary of available agents and their capabilities
            system_state: Current system resource utilization and metrics
            
        Returns:
            List of selected agents with allocation parameters
        """
        if not TORCH_AVAILABLE or self.q_network is None:
            return await self._fallback_agent_selection(campaign, available_agents)
        
        # Extract state features
        state_vector = await self._extract_state_features(campaign, available_agents, system_state)
        
        # Get Q-values for all possible actions
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state_vector).unsqueeze(0).to(self.device))
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Exploit: best Q-value action
            action_idx = q_values.argmax().item()
        
        # Convert action index to agent selection
        selected_agents = await self._action_to_agents(action_idx, available_agents, system_state)
        
        # Store state-action pair for later reward calculation
        await self._store_selection_context(campaign.id, state_vector, action_idx)
        
        self.logger.info(f"DQN selected {len(selected_agents)} agents for campaign {campaign.id}")
        return selected_agents
    
    async def update_from_campaign_result(self, 
                                        campaign_id: str, 
                                        campaign_result: Dict[str, Any]):
        """
        Update DQN based on campaign results
        
        Args:
            campaign_id: ID of completed campaign
            campaign_result: Results including success metrics
        """
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        
        # Calculate reward based on campaign performance
        reward = await self._calculate_reward(campaign_result)
        
        # Retrieve stored context
        context = await self._get_selection_context(campaign_id)
        if not context:
            return
        
        state_vector, action_idx = context['state'], context['action']
        
        # For terminal state, next_state is None
        next_state = None
        done = True
        
        # Store experience in replay buffer
        experience = Experience(
            state=state_vector,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.memory.push(experience)
        
        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size and self.training_step % self.training_frequency == 0:
            await self._train_dqn()
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_frequency == 0:
            self._update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.logger.debug(f"Updated DQN from campaign {campaign_id} with reward {reward}")
    
    async def _extract_state_features(self, 
                                    campaign: Campaign, 
                                    available_agents: Dict[str, Any],
                                    system_state: Dict[str, Any]) -> np.ndarray:
        """Extract state features for DQN input"""
        
        # Calculate target complexity score
        target_complexity = 0.0
        if campaign.targets:
            complexity_factors = []
            for target in campaign.targets:
                # Factor in number of ports, services, subdomains
                ports = len(target.get('ports', []))
                subdomains = len(target.get('subdomains', []))
                services = len(target.get('services', []))
                complexity_factors.append((ports + subdomains + services) / 30.0)  # Normalize
            target_complexity = min(1.0, np.mean(complexity_factors))
        
        # Get security score from metadata or estimate
        target_security_score = 0.5  # Default
        if campaign.metadata:
            target_security_score = campaign.metadata.get('security_score', 0.5)
        
        # EPYC-specific system metrics
        epyc_utilization = system_state.get('cpu_utilization', 0.5)
        numa_locality = system_state.get('numa_locality_score', 0.8)
        memory_pressure = system_state.get('memory_pressure', 0.3)
        
        # Campaign priority mapping
        priority_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        campaign_priority = priority_map.get(campaign.priority.lower(), 0.5)
        
        # Time constraints
        created_hours_ago = (datetime.utcnow() - campaign.created_at).total_seconds() / 3600
        time_urgency = min(1.0, created_hours_ago / 24.0)  # Normalize to 24 hours
        
        # Historical success rate for similar campaigns
        historical_success = await self._get_historical_success_rate(campaign)
        
        # Knowledge confidence from campaign metadata
        knowledge_confidence = campaign.metadata.get('knowledge_confidence', 0.5) if campaign.metadata else 0.5
        
        # System queue length
        queue_length = system_state.get('campaign_queue_length', 0)
        normalized_queue = min(1.0, queue_length / 50.0)  # Normalize to max 50 campaigns
        
        # Construct state vector
        state_vector = np.array([
            target_complexity,
            target_security_score,
            len(available_agents) / 20.0,  # Normalize to max 20 agent types
            campaign_priority,
            time_urgency,
            epyc_utilization,
            historical_success,
            knowledge_confidence,
            normalized_queue,
            numa_locality
        ], dtype=np.float32)
        
        return state_vector
    
    async def _action_to_agents(self, 
                              action_idx: int, 
                              available_agents: Dict[str, Any],
                              system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert DQN action index to agent selection"""
        
        # Define action space mapping
        # Actions represent different agent selection strategies
        action_strategies = [
            # Conservative strategies (low resource, high precision)
            {'strategy': 'conservative', 'agents': ['recon', 'nuclei'], 'parallelism': 1, 'resources': 0.2},
            {'strategy': 'conservative', 'agents': ['web_crawler'], 'parallelism': 2, 'resources': 0.3},
            
            # Balanced strategies (medium resource, balanced approach)
            {'strategy': 'balanced', 'agents': ['recon', 'web_crawler', 'nuclei'], 'parallelism': 2, 'resources': 0.5},
            {'strategy': 'balanced', 'agents': ['vulnerability_scanner', 'web_crawler'], 'parallelism': 3, 'resources': 0.6},
            
            # Aggressive strategies (high resource, comprehensive coverage)
            {'strategy': 'aggressive', 'agents': ['recon', 'web_crawler', 'nuclei', 'vulnerability_scanner'], 'parallelism': 4, 'resources': 0.8},
            {'strategy': 'aggressive', 'agents': ['all_available'], 'parallelism': 6, 'resources': 1.0},
            
            # EPYC-optimized strategies (leverage high core count)
            {'strategy': 'epyc_optimized', 'agents': ['parallel_recon'], 'parallelism': 8, 'resources': 0.4},
            {'strategy': 'epyc_optimized', 'agents': ['distributed_scan'], 'parallelism': 16, 'resources': 0.7},
            {'strategy': 'epyc_optimized', 'agents': ['massively_parallel'], 'parallelism': 32, 'resources': 1.0},
            
            # Specialized strategies
            {'strategy': 'stealth', 'agents': ['stealth_recon'], 'parallelism': 1, 'resources': 0.1},
            {'strategy': 'deep_analysis', 'agents': ['advanced_analyzer'], 'parallelism': 2, 'resources': 0.6},
            {'strategy': 'rapid_scan', 'agents': ['fast_scanner'], 'parallelism': 8, 'resources': 0.5},
            
            # Hybrid strategies
            {'strategy': 'hybrid_intel', 'agents': ['recon', 'osint'], 'parallelism': 3, 'resources': 0.4},
            {'strategy': 'hybrid_exploit', 'agents': ['vulnerability_scanner', 'exploit_analyzer'], 'parallelism': 4, 'resources': 0.7},
            
            # Resource-aware strategies
            {'strategy': 'low_resource', 'agents': ['lightweight_scanner'], 'parallelism': 1, 'resources': 0.1},
            {'strategy': 'medium_resource', 'agents': ['standard_suite'], 'parallelism': 3, 'resources': 0.5},
            {'strategy': 'high_resource', 'agents': ['comprehensive_suite'], 'parallelism': 8, 'resources': 0.9},
            
            # NUMA-aware strategies for EPYC
            {'strategy': 'numa_node_0', 'agents': ['local_agents'], 'parallelism': 16, 'resources': 0.5},
            {'strategy': 'numa_node_1', 'agents': ['local_agents'], 'parallelism': 16, 'resources': 0.5},
            {'strategy': 'cross_numa', 'agents': ['distributed_agents'], 'parallelism': 32, 'resources': 0.8},
        ]
        
        # Select strategy based on action index
        strategy = action_strategies[action_idx % len(action_strategies)]
        
        # Build agent selection based on strategy
        selected_agents = []
        
        if strategy['agents'] == ['all_available']:
            # Use all available agents
            for agent_name, agent_info in available_agents.items():
                selected_agents.append({
                    'agent_name': agent_name,
                    'agent_type': agent_info.get('type', 'unknown'),
                    'resource_allocation': strategy['resources'] / len(available_agents),
                    'parallelism': min(strategy['parallelism'], agent_info.get('max_instances', 4)),
                    'numa_preference': self._get_numa_preference(system_state),
                    'strategy': strategy['strategy']
                })
        else:
            # Use specific agents from strategy
            for agent_type in strategy['agents']:
                # Find matching available agents
                matching_agents = [
                    (name, info) for name, info in available_agents.items()
                    if agent_type in name.lower() or agent_type in info.get('capabilities', [])
                ]
                
                for agent_name, agent_info in matching_agents[:2]:  # Limit to 2 per type
                    selected_agents.append({
                        'agent_name': agent_name,
                        'agent_type': agent_info.get('type', agent_type),
                        'resource_allocation': strategy['resources'] / len(strategy['agents']),
                        'parallelism': min(strategy['parallelism'] // len(strategy['agents']), 
                                         agent_info.get('max_instances', 4)),
                        'numa_preference': self._get_numa_preference(system_state),
                        'strategy': strategy['strategy']
                    })
        
        # Ensure at least one agent is selected
        if not selected_agents and available_agents:
            # Fallback: select first available agent
            agent_name, agent_info = next(iter(available_agents.items()))
            selected_agents.append({
                'agent_name': agent_name,
                'agent_type': agent_info.get('type', 'fallback'),
                'resource_allocation': 0.3,
                'parallelism': 1,
                'numa_preference': 0,
                'strategy': 'fallback'
            })
        
        return selected_agents
    
    def _get_numa_preference(self, system_state: Dict[str, Any]) -> int:
        """Determine optimal NUMA node based on current utilization"""
        numa_0_util = system_state.get('numa_0_utilization', 0.5)
        numa_1_util = system_state.get('numa_1_utilization', 0.5)
        
        # Prefer less utilized NUMA node
        return 0 if numa_0_util <= numa_1_util else 1
    
    async def _calculate_reward(self, campaign_result: Dict[str, Any]) -> float:
        """
        Calculate reward based on campaign performance
        Higher rewards for better performance with resource efficiency
        """
        base_reward = 0.0
        
        # Success metrics
        findings_count = campaign_result.get('findings_count', 0)
        high_severity_findings = campaign_result.get('high_severity_findings', 0)
        success_rate = campaign_result.get('success_rate', 0.0)
        
        # Resource efficiency metrics
        resource_utilization = campaign_result.get('resource_utilization', 1.0)
        execution_time = campaign_result.get('execution_time_hours', 1.0)
        cost_effectiveness = campaign_result.get('cost_effectiveness', 0.5)
        
        # EPYC-specific metrics
        epyc_efficiency = campaign_result.get('epyc_core_efficiency', 0.5)
        numa_locality = campaign_result.get('numa_locality_score', 0.5)
        
        # Calculate reward components
        finding_reward = (findings_count * 10 + high_severity_findings * 25) / 100.0
        success_reward = success_rate * 50
        efficiency_reward = (2.0 - resource_utilization) * 20  # Reward efficiency
        speed_reward = max(0, (4.0 - execution_time) * 10)  # Reward faster completion
        cost_reward = cost_effectiveness * 30
        epyc_reward = epyc_efficiency * 20
        numa_reward = numa_locality * 15
        
        # Combine rewards
        base_reward = (
            finding_reward + 
            success_reward + 
            efficiency_reward + 
            speed_reward + 
            cost_reward + 
            epyc_reward + 
            numa_reward
        )
        
        # Apply penalties for poor performance
        if success_rate < 0.1:
            base_reward -= 50  # Heavy penalty for very poor performance
        elif success_rate < 0.3:
            base_reward -= 25  # Moderate penalty for poor performance
        
        if resource_utilization > 1.5:
            base_reward -= 30  # Penalty for resource waste
        
        # Normalize reward to [-100, 100] range
        return np.clip(base_reward, -100, 100)
    
    async def _train_dqn(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state if e.next_state is not None else np.zeros(self.state_size) for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.logger.debug(f"DQN training loss: {loss.item():.4f}")
    
    def _update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    async def _store_selection_context(self, campaign_id: str, state: np.ndarray, action: int):
        """Store context for later reward calculation"""
        # In production, this would use Redis or database
        # For now, use in-memory storage with expiration
        if not hasattr(self, '_selection_contexts'):
            self._selection_contexts = {}
        
        self._selection_contexts[campaign_id] = {
            'state': state,
            'action': action,
            'timestamp': datetime.utcnow()
        }
        
        # Clean up old contexts (older than 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._selection_contexts = {
            k: v for k, v in self._selection_contexts.items()
            if v['timestamp'] > cutoff
        }
    
    async def _get_selection_context(self, campaign_id: str) -> Optional[Dict]:
        """Retrieve stored selection context"""
        if not hasattr(self, '_selection_contexts'):
            return None
        return self._selection_contexts.get(campaign_id)
    
    async def _get_historical_success_rate(self, campaign: Campaign) -> float:
        """Get historical success rate for similar campaigns"""
        # Simplified implementation - in production would query knowledge fabric
        if campaign.metadata and 'target_type' in campaign.metadata:
            target_type = campaign.metadata['target_type']
            # Mock historical data based on target type
            historical_rates = {
                'web_application': 0.65,
                'api': 0.70,
                'network': 0.45,
                'mobile': 0.55,
                'iot': 0.40,
                'cloud': 0.60
            }
            return historical_rates.get(target_type, 0.50)
        return 0.50  # Default
    
    async def _fallback_agent_selection(self, 
                                      campaign: Campaign,
                                      available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback agent selection when DQN is not available"""
        selected_agents = []
        
        # Simple heuristic-based selection
        agent_priority = ['recon', 'web_crawler', 'nuclei', 'vulnerability_scanner']
        
        for agent_type in agent_priority:
            matching_agents = [
                (name, info) for name, info in available_agents.items()
                if agent_type in name.lower()
            ]
            
            if matching_agents:
                agent_name, agent_info = matching_agents[0]
                selected_agents.append({
                    'agent_name': agent_name,
                    'agent_type': agent_info.get('type', agent_type),
                    'resource_allocation': 0.5,
                    'parallelism': 2,
                    'numa_preference': 0,
                    'strategy': 'fallback_heuristic'
                })
                
                if len(selected_agents) >= 3:  # Limit to 3 agents
                    break
        
        self.logger.info(f"Fallback selection: {len(selected_agents)} agents for campaign {campaign.id}")
        return selected_agents
    
    def _save_model(self):
        """Save DQN model and metadata"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
            
        try:
            model_data = {
                'q_network_state': self.q_network.state_dict(),
                'target_network_state': self.target_network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min,
                    'batch_size': self.batch_size
                }
            }
            
            torch.save(model_data, self.model_path / 'dqn_model.pth')
            
            # Save replay buffer separately
            with open(self.model_path / 'replay_buffer.pkl', 'wb') as f:
                pickle.dump(list(self.memory.buffer), f)
            
            self.logger.info(f"DQN model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save DQN model: {e}")
    
    def _load_model(self):
        """Load DQN model and metadata"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            model_file = self.model_path / 'dqn_model.pth'
            if not model_file.exists():
                self.logger.info("No saved DQN model found, starting fresh")
                return
            
            model_data = torch.load(model_file, map_location=self.device)
            
            self.q_network.load_state_dict(model_data['q_network_state'])
            self.target_network.load_state_dict(model_data['target_network_state'])
            self.optimizer.load_state_dict(model_data['optimizer_state'])
            
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.training_step = model_data.get('training_step', 0)
            
            # Load replay buffer
            buffer_file = self.model_path / 'replay_buffer.pkl'
            if buffer_file.exists():
                with open(buffer_file, 'rb') as f:
                    experiences = pickle.load(f)
                    for exp in experiences:
                        self.memory.push(exp)
            
            self.logger.info(f"DQN model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load DQN model: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get DQN performance metrics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'replay_buffer_size': len(self.memory),
            'model_available': self.q_network is not None,
            'device': str(self.device) if self.device else 'none',
            'epyc_cores': self.epyc_cores,
            'numa_nodes': self.numa_nodes,
            'ccx_per_numa': self.ccx_per_numa,
            'torch_available': TORCH_AVAILABLE
        }
    
    async def shutdown(self):
        """Shutdown DQN selector and save model"""
        self._save_model()
        self.logger.info("DQN Agent Selector shutdown complete")


# Integration example with existing orchestrator
class DQNEnhancedOrchestrator:
    """Enhanced orchestrator with DQN-based agent selection"""
    
    def __init__(self, base_orchestrator, epyc_cores: int = 64):
        self.base_orchestrator = base_orchestrator
        self.dqn_selector = DQNAgentSelector(epyc_cores=epyc_cores)
        self.logger = logging.getLogger(__name__)
    
    async def create_intelligent_campaign(self, *args, **kwargs) -> str:
        """Create campaign with DQN-enhanced agent selection"""
        # Use base orchestrator to create campaign
        campaign_id = await self.base_orchestrator.create_campaign(*args, **kwargs)
        
        # Get campaign details
        campaign = self.base_orchestrator.campaigns.get(campaign_id)
        if not campaign:
            return campaign_id
        
        # Get available agents and system state
        available_agents = await self.base_orchestrator.agent_registry.list_discovered_agents()
        system_state = await self._get_system_state()
        
        # Use DQN to select optimal agents
        selected_agents = await self.dqn_selector.select_agents(
            campaign, available_agents, system_state
        )
        
        # Update campaign with DQN selections
        campaign.metadata = campaign.metadata or {}
        campaign.metadata['dqn_selected_agents'] = selected_agents
        campaign.metadata['selection_method'] = 'dqn'
        
        self.logger.info(f"DQN selected {len(selected_agents)} agents for campaign {campaign_id}")
        return campaign_id
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for DQN input"""
        # This would integrate with actual system monitoring
        return {
            'cpu_utilization': 0.45,  # Mock data
            'memory_pressure': 0.30,
            'numa_0_utilization': 0.40,
            'numa_1_utilization': 0.50,
            'numa_locality_score': 0.85,
            'campaign_queue_length': 5,
            'active_campaigns': 3
        }
    
    async def update_dqn_from_results(self, campaign_id: str, results: Dict[str, Any]):
        """Update DQN based on campaign results"""
        await self.dqn_selector.update_from_campaign_result(campaign_id, results)


if __name__ == "__main__":
    async def main():
        # Example usage
        dqn_selector = DQNAgentSelector(epyc_cores=64)
        
        # Mock campaign and agents for testing
        from ..orchestration.orchestrator import Campaign
        
        mock_campaign = Campaign(
            id="test-123",
            name="Test Campaign",
            targets=[{"hostname": "example.com", "ports": [80, 443]}],
            priority="high",
            created_at=datetime.utcnow()
        )
        
        mock_agents = {
            "recon_agent": {"type": "reconnaissance", "capabilities": ["port_scan", "dns_enum"]},
            "web_crawler": {"type": "web_analysis", "capabilities": ["crawling", "js_analysis"]},
            "nuclei_scanner": {"type": "vulnerability_scan", "capabilities": ["cve_detection"]}
        }
        
        mock_system_state = {
            'cpu_utilization': 0.3,
            'memory_pressure': 0.2,
            'numa_locality_score': 0.9
        }
        
        # Test agent selection
        selected = await dqn_selector.select_agents(mock_campaign, mock_agents, mock_system_state)
        print(f"Selected agents: {selected}")
        
        # Test reward calculation
        mock_results = {
            'findings_count': 5,
            'high_severity_findings': 2,
            'success_rate': 0.8,
            'resource_utilization': 0.7,
            'execution_time_hours': 2.0,
            'epyc_core_efficiency': 0.85
        }
        
        await dqn_selector.update_from_campaign_result("test-123", mock_results)
        
        # Show performance metrics
        metrics = await dqn_selector.get_performance_metrics()
        print(f"DQN Metrics: {metrics}")
    
    asyncio.run(main())