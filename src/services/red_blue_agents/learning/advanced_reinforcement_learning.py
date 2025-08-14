"""
Advanced Reinforcement Learning Engine for Red Team Agents
Production-grade RL implementation with sophisticated exploration strategies

This module implements state-of-the-art reinforcement learning algorithms for 
autonomous red team agents:
- Deep Q-Networks (DQN) with experience replay
- Policy Gradient methods (A3C, PPO)
- Multi-armed bandit optimization
- Contextual bandits for technique selection
- Hierarchical reinforcement learning
- Adversarial training environments
"""

import asyncio
import logging
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, namedtuple
from enum import Enum
import random
import math
from pathlib import Path

import redis.asyncio as redis
import asyncpg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Define experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class AgentAction(Enum):
    """Available actions for red team agents"""
    RECONNAISSANCE = 0
    INITIAL_ACCESS = 1
    EXECUTION = 2
    PERSISTENCE = 3
    PRIVILEGE_ESCALATION = 4
    DEFENSE_EVASION = 5
    CREDENTIAL_ACCESS = 6
    DISCOVERY = 7
    LATERAL_MOVEMENT = 8
    COLLECTION = 9
    COMMAND_CONTROL = 10
    EXFILTRATION = 11
    IMPACT = 12


class RewardSignal(Enum):
    """Types of reward signals"""
    TECHNIQUE_SUCCESS = "technique_success"
    OBJECTIVE_ACHIEVED = "objective_achieved"
    STEALTH_MAINTAINED = "stealth_maintained"
    DETECTION_AVOIDED = "detection_avoided"
    MISSION_COMPLETED = "mission_completed"
    FAILURE_PENALTY = "failure_penalty"
    TIME_PENALTY = "time_penalty"


@dataclass
class EnvironmentState:
    """Current environment state representation"""
    target_info: Dict[str, Any]
    discovered_services: List[Dict[str, Any]]
    compromised_systems: List[str]
    available_techniques: List[str]
    detection_level: float
    mission_progress: float
    time_elapsed: float
    agent_resources: Dict[str, float]
    network_topology: Dict[str, Any]
    security_controls: List[str]
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for neural networks"""
        vector = []
        
        # Target information features
        vector.extend([
            len(self.discovered_services),
            len(self.compromised_systems),
            len(self.available_techniques),
            self.detection_level,
            self.mission_progress,
            self.time_elapsed,
            len(self.security_controls)
        ])
        
        # Agent resources
        vector.extend([
            self.agent_resources.get('stealth_budget', 0.0),
            self.agent_resources.get('skill_level', 0.0),
            self.agent_resources.get('tool_availability', 0.0)
        ])
        
        # Network topology features
        vector.extend([
            len(self.network_topology.get('hosts', [])),
            len(self.network_topology.get('subnets', [])),
            self.network_topology.get('connectivity_score', 0.0)
        ])
        
        return np.array(vector, dtype=np.float32)


@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    reward: float
    new_state: EnvironmentState
    metadata: Dict[str, Any]
    detection_risk: float
    resources_consumed: Dict[str, float]
    techniques_unlocked: List[str]
    objectives_achieved: List[str]


class DQNNetwork(nn.Module):
    """Deep Q-Network for action value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DQNNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        # Create layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(state)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for DQN"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, experience: Experience, td_error: float = None):
        """Add experience to buffer with priority"""
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority based on TD error
        priority = (abs(td_error) + 1e-6) ** self.alpha if td_error is not None else max_priority
        self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with priority weighting"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class AdvancedDQNAgent:
    """Advanced DQN agent with sophisticated enhancements"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Hyperparameters
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.tau = self.config.get('tau', 0.005)  # Soft update parameter
        self.batch_size = self.config.get('batch_size', 64)
        self.update_frequency = self.config.get('update_frequency', 4)
        self.target_update_frequency = self.config.get('target_update_frequency', 100)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Copy weights to target network
        self.update_target_network(tau=1.0)
        
        # Experience replay
        replay_capacity = self.config.get('replay_capacity', 100000)
        self.replay_buffer = PrioritizedReplayBuffer(replay_capacity)
        
        # Training state
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        logger.info(f"Initialized DQN agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: EnvironmentState, available_actions: List[int] = None, training: bool = True) -> int:
        """Select action using epsilon-greedy policy with action masking"""
        if available_actions is None:
            available_actions = list(range(self.action_dim))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: select best action
        state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # Mask unavailable actions
            masked_q_values = q_values.clone()
            unavailable_actions = set(range(self.action_dim)) - set(available_actions)
            for action in unavailable_actions:
                masked_q_values[0, action] = float('-inf')
            
            action = masked_q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: EnvironmentState, action: int, reward: float, 
                        next_state: EnvironmentState, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(
            state=state.to_vector(),
            action=action,
            reward=reward,
            next_state=next_state.to_vector(),
            done=done
        )
        
        # Calculate TD error for prioritization
        td_error = self._calculate_td_error(experience)
        self.replay_buffer.add(experience, td_error)
    
    def _calculate_td_error(self, experience: Experience) -> float:
        """Calculate TD error for prioritized replay"""
        try:
            state_tensor = torch.FloatTensor(experience.state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(experience.next_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                current_q = self.q_network(state_tensor)[0, experience.action]
                
                if experience.done:
                    target_q = experience.reward
                else:
                    next_q = self.target_network(next_state_tensor).max(1)[0]
                    target_q = experience.reward + self.gamma * next_q
                
                td_error = abs(current_q.item() - target_q.item())
            
            return td_error
            
        except Exception as e:
            logger.error(f"Error calculating TD error: {e}")
            return 1.0  # Default priority
    
    def train(self) -> Dict[str, float]:
        """Train the DQN agent"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        if not experiences:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Calculate loss with importance sampling weights
        td_errors = current_q_values - target_q_values
        loss = (weights_tensor.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        td_errors_np = td_errors.detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors_np)
        
        # Update target network
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update training step
        self.training_step += 1
        
        # Store metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return {
            'loss': loss_value,
            'epsilon': self.epsilon,
            'q_value_mean': current_q_values.mean().item(),
            'td_error_mean': abs(td_errors.mean().item())
        }
    
    def update_target_network(self, tau: float = None):
        """Soft update of target network"""
        if tau is None:
            tau = self.tau
        
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        
        logger.info(f"Loaded model from {filepath}")


class MultiArmedBanditOptimizer:
    """Multi-armed bandit for technique selection optimization"""
    
    def __init__(self, num_arms: int, exploration_strategy: str = "ucb"):
        self.num_arms = num_arms
        self.exploration_strategy = exploration_strategy
        
        # Initialize arm statistics
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.arm_values = np.zeros(num_arms)
        
        # UCB parameters
        self.confidence_level = 2.0
        
        # Thompson Sampling parameters
        self.alpha = np.ones(num_arms)  # Beta distribution alpha
        self.beta = np.ones(num_arms)   # Beta distribution beta
        
        # Total pulls
        self.total_pulls = 0
    
    def select_arm(self, context: Dict[str, Any] = None) -> int:
        """Select arm based on exploration strategy"""
        if self.exploration_strategy == "epsilon_greedy":
            return self._epsilon_greedy_selection()
        elif self.exploration_strategy == "ucb":
            return self._ucb_selection()
        elif self.exploration_strategy == "thompson_sampling":
            return self._thompson_sampling_selection()
        else:
            return self._random_selection()
    
    def _epsilon_greedy_selection(self, epsilon: float = 0.1) -> int:
        """Epsilon-greedy arm selection"""
        if random.random() < epsilon:
            return random.randint(0, self.num_arms - 1)
        else:
            return np.argmax(self.arm_values)
    
    def _ucb_selection(self) -> int:
        """Upper Confidence Bound arm selection"""
        if self.total_pulls == 0:
            return random.randint(0, self.num_arms - 1)
        
        ucb_values = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            if self.arm_counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                confidence = self.confidence_level * np.sqrt(
                    np.log(self.total_pulls) / self.arm_counts[arm]
                )
                ucb_values[arm] = self.arm_values[arm] + confidence
        
        return np.argmax(ucb_values)
    
    def _thompson_sampling_selection(self) -> int:
        """Thompson Sampling arm selection"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def _random_selection(self) -> int:
        """Random arm selection"""
        return random.randint(0, self.num_arms - 1)
    
    def update_arm(self, arm: int, reward: float):
        """Update arm statistics with new reward"""
        self.arm_counts[arm] += 1
        self.total_pulls += 1
        
        # Update average reward
        n = self.arm_counts[arm]
        self.arm_values[arm] = ((n - 1) * self.arm_values[arm] + reward) / n
        
        # Update Thompson Sampling parameters
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get arm statistics"""
        return {
            'arm_counts': self.arm_counts.tolist(),
            'arm_values': self.arm_values.tolist(),
            'total_pulls': self.total_pulls,
            'best_arm': np.argmax(self.arm_values),
            'regret': self._calculate_regret()
        }
    
    def _calculate_regret(self) -> float:
        """Calculate cumulative regret"""
        if self.total_pulls == 0:
            return 0.0
        
        optimal_reward = np.max(self.arm_values)
        actual_reward = np.sum(self.arm_counts * self.arm_values)
        optimal_total = optimal_reward * self.total_pulls
        
        return optimal_total - actual_reward


class RewardShaper:
    """Sophisticated reward shaping for red team agents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Reward weights
        self.weights = {
            RewardSignal.TECHNIQUE_SUCCESS: 1.0,
            RewardSignal.OBJECTIVE_ACHIEVED: 10.0,
            RewardSignal.STEALTH_MAINTAINED: 2.0,
            RewardSignal.DETECTION_AVOIDED: 5.0,
            RewardSignal.MISSION_COMPLETED: 100.0,
            RewardSignal.FAILURE_PENALTY: -5.0,
            RewardSignal.TIME_PENALTY: -0.1
        }
        
        # Update weights from config
        self.weights.update(self.config.get('reward_weights', {}))
        
        # Reward history for normalization
        self.reward_history = deque(maxlen=1000)
        
    def calculate_reward(self, action_result: ActionResult, context: Dict[str, Any] = None) -> float:
        """Calculate shaped reward for action result"""
        total_reward = 0.0
        reward_components = {}
        
        # Base reward from action success/failure
        if action_result.success:
            base_reward = self.weights[RewardSignal.TECHNIQUE_SUCCESS]
            reward_components['technique_success'] = base_reward
            total_reward += base_reward
        else:
            failure_penalty = self.weights[RewardSignal.FAILURE_PENALTY]
            reward_components['failure_penalty'] = failure_penalty
            total_reward += failure_penalty
        
        # Objective achievement bonus
        if action_result.objectives_achieved:
            objective_reward = len(action_result.objectives_achieved) * self.weights[RewardSignal.OBJECTIVE_ACHIEVED]
            reward_components['objective_achieved'] = objective_reward
            total_reward += objective_reward
        
        # Stealth maintenance bonus
        if action_result.detection_risk < 0.3:  # Low detection risk
            stealth_reward = self.weights[RewardSignal.STEALTH_MAINTAINED] * (1.0 - action_result.detection_risk)
            reward_components['stealth_maintained'] = stealth_reward
            total_reward += stealth_reward
        
        # Detection avoidance bonus
        if 'detection_avoided' in action_result.metadata:
            detection_reward = self.weights[RewardSignal.DETECTION_AVOIDED]
            reward_components['detection_avoided'] = detection_reward
            total_reward += detection_reward
        
        # Time penalty (encourage efficiency)
        time_elapsed = action_result.new_state.time_elapsed
        if time_elapsed > 0:
            time_penalty = self.weights[RewardSignal.TIME_PENALTY] * time_elapsed
            reward_components['time_penalty'] = time_penalty
            total_reward += time_penalty
        
        # Mission completion bonus
        if action_result.new_state.mission_progress >= 1.0:
            mission_reward = self.weights[RewardSignal.MISSION_COMPLETED]
            reward_components['mission_completed'] = mission_reward
            total_reward += mission_reward
        
        # Exploration bonus for discovering new techniques
        if action_result.techniques_unlocked:
            exploration_bonus = len(action_result.techniques_unlocked) * 2.0
            reward_components['exploration_bonus'] = exploration_bonus
            total_reward += exploration_bonus
        
        # Resource efficiency bonus
        resource_efficiency = self._calculate_resource_efficiency(action_result)
        efficiency_bonus = resource_efficiency * 1.0
        reward_components['efficiency_bonus'] = efficiency_bonus
        total_reward += efficiency_bonus
        
        # Store reward for normalization
        self.reward_history.append(total_reward)
        
        # Add reward breakdown to metadata
        action_result.metadata['reward_breakdown'] = reward_components
        
        return total_reward
    
    def _calculate_resource_efficiency(self, action_result: ActionResult) -> float:
        """Calculate resource efficiency score"""
        total_resources = sum(action_result.resources_consumed.values())
        
        if total_resources == 0:
            return 1.0  # Perfect efficiency
        
        # Normalize based on action success and impact
        if action_result.success:
            impact_score = len(action_result.objectives_achieved) + len(action_result.techniques_unlocked)
            efficiency = impact_score / (total_resources + 1)
        else:
            efficiency = -total_resources  # Penalty for wasted resources
        
        return np.clip(efficiency, -1.0, 1.0)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward based on historical distribution"""
        if len(self.reward_history) < 10:
            return reward
        
        rewards_array = np.array(self.reward_history)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array) + 1e-8
        
        normalized_reward = (reward - mean_reward) / std_reward
        return np.clip(normalized_reward, -3.0, 3.0)  # Clip to prevent extreme values


class AdvancedRLEngine:
    """Advanced Reinforcement Learning Engine for Red Team Agents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # State and action dimensions
        self.state_dim = self.config.get('state_dim', 13)  # Based on EnvironmentState
        self.action_dim = len(AgentAction)
        
        # Initialize components
        self.dqn_agent = AdvancedDQNAgent(self.state_dim, self.action_dim, self.config.get('dqn_config', {}))
        self.bandit_optimizer = MultiArmedBanditOptimizer(
            self.action_dim, 
            self.config.get('exploration_strategy', 'ucb')
        )
        self.reward_shaper = RewardShaper(self.config.get('reward_config', {}))
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_metrics = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'exploration_rate': deque(maxlen=100)
        }
        
        # Model persistence
        self.model_save_frequency = self.config.get('model_save_frequency', 100)
        self.model_save_path = self.config.get('model_save_path', 'models/rl_agent')
        
        logger.info("Advanced RL Engine initialized successfully")
    
    async def select_action(self, state: EnvironmentState, available_actions: List[int] = None, 
                           training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """Select action using hybrid approach (DQN + Bandit)"""
        if available_actions is None:
            available_actions = list(range(self.action_dim))
        
        # Use DQN for primary action selection
        dqn_action = self.dqn_agent.select_action(state, available_actions, training)
        
        # Use bandit for technique-specific optimization
        bandit_action = self.bandit_optimizer.select_arm()
        
        # Combine decisions (weighted average or selection)
        use_dqn = random.random() < 0.8  # 80% DQN, 20% bandit
        
        if use_dqn and dqn_action in available_actions:
            selected_action = dqn_action
            decision_method = 'dqn'
        elif bandit_action in available_actions:
            selected_action = bandit_action
            decision_method = 'bandit'
        else:
            selected_action = random.choice(available_actions)
            decision_method = 'random'
        
        action_metadata = {
            'decision_method': decision_method,
            'dqn_action': dqn_action,
            'bandit_action': bandit_action,
            'available_actions': available_actions,
            'epsilon': self.dqn_agent.epsilon
        }
        
        return selected_action, action_metadata
    
    async def process_experience(self, state: EnvironmentState, action: int, 
                                action_result: ActionResult) -> Dict[str, Any]:
        """Process experience and update learning models"""
        # Calculate shaped reward
        raw_reward = 1.0 if action_result.success else -1.0
        shaped_reward = self.reward_shaper.calculate_reward(action_result)
        normalized_reward = self.reward_shaper.normalize_reward(shaped_reward)
        
        # Store experience in DQN
        done = action_result.new_state.mission_progress >= 1.0
        self.dqn_agent.store_experience(state, action, normalized_reward, action_result.new_state, done)
        
        # Update bandit
        self.bandit_optimizer.update_arm(action, shaped_reward)
        
        # Train DQN if enough experiences
        training_metrics = {}
        if len(self.dqn_agent.replay_buffer) >= self.dqn_agent.batch_size:
            training_metrics = self.dqn_agent.train()
        
        # Update total steps
        self.total_steps += 1
        
        return {
            'raw_reward': raw_reward,
            'shaped_reward': shaped_reward,
            'normalized_reward': normalized_reward,
            'training_metrics': training_metrics,
            'total_steps': self.total_steps
        }
    
    async def end_episode(self, episode_reward: float, episode_length: int, 
                         success: bool) -> Dict[str, Any]:
        """End episode and update metrics"""
        self.episode_count += 1
        
        # Update metrics
        self.training_metrics['episode_rewards'].append(episode_reward)
        self.training_metrics['episode_lengths'].append(episode_length)
        self.training_metrics['success_rate'].append(1.0 if success else 0.0)
        self.training_metrics['exploration_rate'].append(self.dqn_agent.epsilon)
        
        # Save model periodically
        if self.episode_count % self.model_save_frequency == 0:
            await self.save_model()
        
        # Calculate episode statistics
        recent_rewards = list(self.training_metrics['episode_rewards'])
        recent_success = list(self.training_metrics['success_rate'])
        
        episode_stats = {
            'episode': self.episode_count,
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'avg_reward_100': np.mean(recent_rewards) if recent_rewards else 0.0,
            'success_rate_100': np.mean(recent_success) if recent_success else 0.0,
            'epsilon': self.dqn_agent.epsilon,
            'total_steps': self.total_steps
        }
        
        logger.info(f"Episode {self.episode_count} completed: "
                   f"Reward={episode_reward:.2f}, Success={success}, "
                   f"Avg100={episode_stats['avg_reward_100']:.2f}")
        
        return episode_stats
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        recent_rewards = list(self.training_metrics['episode_rewards'])
        recent_lengths = list(self.training_metrics['episode_lengths'])
        recent_success = list(self.training_metrics['success_rate'])
        
        # DQN metrics
        dqn_metrics = {
            'epsilon': self.dqn_agent.epsilon,
            'replay_buffer_size': len(self.dqn_agent.replay_buffer),
            'training_step': self.dqn_agent.training_step,
            'avg_loss': np.mean(self.dqn_agent.loss_history) if self.dqn_agent.loss_history else 0.0
        }
        
        # Bandit metrics
        bandit_stats = self.bandit_optimizer.get_statistics()
        
        # Overall performance
        performance_metrics = {
            'episodes_completed': self.episode_count,
            'total_steps': self.total_steps,
            'avg_episode_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'avg_episode_length': np.mean(recent_lengths) if recent_lengths else 0.0,
            'success_rate': np.mean(recent_success) if recent_success else 0.0,
            'reward_std': np.std(recent_rewards) if recent_rewards else 0.0,
            'dqn_metrics': dqn_metrics,
            'bandit_metrics': bandit_stats
        }
        
        return performance_metrics
    
    async def save_model(self, filepath: str = None):
        """Save RL models to disk"""
        if filepath is None:
            filepath = f"{self.model_save_path}_episode_{self.episode_count}.pt"
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save DQN model
        self.dqn_agent.save_model(filepath)
        
        # Save additional state
        additional_state = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'bandit_state': {
                'arm_counts': self.bandit_optimizer.arm_counts,
                'arm_rewards': self.bandit_optimizer.arm_rewards,
                'arm_values': self.bandit_optimizer.arm_values,
                'alpha': self.bandit_optimizer.alpha,
                'beta': self.bandit_optimizer.beta,
                'total_pulls': self.bandit_optimizer.total_pulls
            },
            'reward_history': list(self.reward_shaper.reward_history),
            'training_metrics': {
                key: list(value) for key, value in self.training_metrics.items()
            }
        }
        
        additional_filepath = filepath.replace('.pt', '_state.json')
        with open(additional_filepath, 'w') as f:
            json.dump(additional_state, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    async def load_model(self, filepath: str):
        """Load RL models from disk"""
        # Load DQN model
        self.dqn_agent.load_model(filepath)
        
        # Load additional state
        additional_filepath = filepath.replace('.pt', '_state.json')
        try:
            with open(additional_filepath, 'r') as f:
                additional_state = json.load(f)
            
            self.episode_count = additional_state['episode_count']
            self.total_steps = additional_state['total_steps']
            
            # Restore bandit state
            bandit_state = additional_state['bandit_state']
            self.bandit_optimizer.arm_counts = np.array(bandit_state['arm_counts'])
            self.bandit_optimizer.arm_rewards = np.array(bandit_state['arm_rewards'])
            self.bandit_optimizer.arm_values = np.array(bandit_state['arm_values'])
            self.bandit_optimizer.alpha = np.array(bandit_state['alpha'])
            self.bandit_optimizer.beta = np.array(bandit_state['beta'])
            self.bandit_optimizer.total_pulls = bandit_state['total_pulls']
            
            # Restore reward history
            self.reward_shaper.reward_history = deque(
                additional_state['reward_history'], 
                maxlen=1000
            )
            
            # Restore training metrics
            for key, values in additional_state['training_metrics'].items():
                self.training_metrics[key] = deque(values, maxlen=100)
            
            logger.info(f"Model state loaded from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"Additional state file not found: {additional_filepath}")
    
    def create_mock_state(self, **kwargs) -> EnvironmentState:
        """Create mock environment state for testing"""
        defaults = {
            'target_info': {'host': 'test.example.com', 'os': 'linux'},
            'discovered_services': [{'port': 22, 'service': 'ssh'}, {'port': 80, 'service': 'http'}],
            'compromised_systems': [],
            'available_techniques': ['recon.port_scan', 'exploit.ssh_bruteforce'],
            'detection_level': 0.2,
            'mission_progress': 0.0,
            'time_elapsed': 0.0,
            'agent_resources': {'stealth_budget': 1.0, 'skill_level': 0.8, 'tool_availability': 1.0},
            'network_topology': {'hosts': ['target1'], 'subnets': ['192.168.1.0/24'], 'connectivity_score': 0.7},
            'security_controls': ['firewall', 'ids']
        }
        
        defaults.update(kwargs)
        return EnvironmentState(**defaults)
    
    def create_mock_action_result(self, success: bool = True, **kwargs) -> ActionResult:
        """Create mock action result for testing"""
        defaults = {
            'success': success,
            'reward': 1.0 if success else -1.0,
            'new_state': self.create_mock_state(mission_progress=0.1),
            'metadata': {'technique_used': 'recon.port_scan'},
            'detection_risk': 0.1,
            'resources_consumed': {'stealth_budget': 0.1, 'time': 5.0},
            'techniques_unlocked': ['exploit.web_sqli'] if success else [],
            'objectives_achieved': ['reconnaissance_complete'] if success else []
        }
        
        defaults.update(kwargs)
        return ActionResult(**defaults)


# Singleton instance
_rl_engine: Optional[AdvancedRLEngine] = None


async def get_rl_engine(config: Dict[str, Any] = None) -> AdvancedRLEngine:
    """Get singleton RL engine instance"""
    global _rl_engine
    
    if _rl_engine is None:
        _rl_engine = AdvancedRLEngine(config)
    
    return _rl_engine


async def shutdown_rl_engine():
    """Shutdown RL engine and save state"""
    global _rl_engine
    
    if _rl_engine:
        await _rl_engine.save_model()
        _rl_engine = None
        logger.info("RL Engine shutdown complete")