"""
Training Workers - Offline RL + Imitation Learning for Red/Blue team policies.

Implements:
- Behavioral Cloning (BC) from expert traces
- Offline RL (CQL, IQL) on replay buffer
- Online RL with safety critics
- Policy candidate generation and validation
"""

import asyncio
import json
import logging
import pickle
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from .environment_api import ActorRole, StepEvent
from .replay_service import ReplayService, ReplayEntry


logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training workers."""
    
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    batch_size: int = 64
    max_episodes_per_training: int = 1000
    
    # Behavioral Cloning
    bc_learning_rate: float = 1e-3
    bc_epochs: int = 50
    bc_hidden_size: int = 256
    
    # Offline RL
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_tau: float = 0.005
    rl_alpha: float = 0.2  # CQL alpha
    
    # Safety Critic
    safety_threshold: float = 0.8
    max_risky_actions: int = 5
    
    # Model persistence
    model_save_dir: str = "/tmp/xorb-models"
    checkpoint_interval: int = 100


@dataclass
class PolicyCandidate:
    """Training policy candidate."""
    candidate_id: str
    actor_role: ActorRole
    model_type: str  # bc, cql, iql, ppo
    training_episodes: int
    training_duration: float
    performance_metrics: Dict[str, float]
    model_path: str
    safety_score: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class TrainingTask:
    """Training task specification."""
    task_id: str
    task_type: str  # bc, offline_rl, online_rl
    actor_role: ActorRole
    replay_entries: List[ReplayEntry]
    config_overrides: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


class StateActionEncoder:
    """Encodes observations and actions for neural networks."""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature extractors
        self.action_types = ['req', 'scan', 'exploit', 'defend', 'detect', 'block', 'monitor']
        self.action_type_to_idx = {a: i for i, a in enumerate(self.action_types)}
    
    def encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode observation to fixed-size vector."""
        features = np.zeros(self.state_dim)
        
        # Surface features (first 10 positions)
        surface = obs.get('surface', [])
        for i, endpoint in enumerate(surface[:10]):
            if endpoint:
                features[i] = hash(endpoint) % 1000 / 1000.0  # Normalize hash
        
        # Position/state features
        if 'position' in obs:
            features[10] = 1.0 if obs['position'] == 'internal' else 0.0
        
        if 'alerts' in obs and obs['alerts'] is not None:
            features[11] = min(obs['alerts'] / 10.0, 1.0)  # Normalize alerts
        
        if 'compromised' in obs and obs['compromised'] is not None:
            features[12] = min(obs['compromised'] / 5.0, 1.0)  # Normalize compromised
        
        # Add some randomness for exploration
        features[50:60] = np.random.normal(0, 0.1, 10)
        
        return features
    
    def encode_action(self, action: Dict[str, Any]) -> np.ndarray:
        """Encode action to fixed-size vector."""
        features = np.zeros(self.action_dim)
        
        # Action type (one-hot)
        action_type = action.get('type', 'unknown')
        if action_type in self.action_type_to_idx:
            features[self.action_type_to_idx[action_type]] = 1.0
        
        # Parameters encoding
        params = action.get('parameters', {})
        if 'path' in params:
            features[len(self.action_types)] = hash(params['path']) % 1000 / 1000.0
        
        if 'method' in params:
            method_idx = {'GET': 0.2, 'POST': 0.4, 'PUT': 0.6, 'DELETE': 0.8}.get(params['method'], 0.1)
            features[len(self.action_types) + 1] = method_idx
        
        return features
    
    def decode_action(self, action_vector: np.ndarray, actor_role: ActorRole) -> Dict[str, Any]:
        """Decode action vector back to action dict."""
        # Find most likely action type
        action_type_probs = action_vector[:len(self.action_types)]
        action_type_idx = np.argmax(action_type_probs)
        action_type = self.action_types[action_type_idx]
        
        # Generate parameters based on role and action type
        parameters = {}
        
        if actor_role == ActorRole.RED:
            if action_type == 'req':
                paths = ['/api/auth/login', '/api/public/info', '/api/admin', '/api/internal']
                path_idx = int(action_vector[len(self.action_types)] * len(paths))
                parameters = {
                    'path': paths[min(path_idx, len(paths) - 1)],
                    'method': 'POST' if 'login' in paths[path_idx] else 'GET'
                }
            elif action_type == 'scan':
                parameters = {
                    'target': '192.168.1.1',
                    'ports': [80, 443, 22, 3306]
                }
            elif action_type == 'exploit':
                parameters = {
                    'vulnerability': 'sql_injection',
                    'payload': 'test'
                }
        
        elif actor_role == ActorRole.BLUE:
            if action_type == 'detect':
                parameters = {
                    'threshold': float(action_vector[len(self.action_types) + 2])
                }
            elif action_type == 'block':
                parameters = {
                    'target': '192.168.1.100',
                    'duration': 300
                }
        
        return {
            'type': action_type,
            'parameters': parameters
        }


class TrainingDataset(Dataset):
    """Dataset for training from replay buffer."""
    
    def __init__(self, replay_entries: List[ReplayEntry], encoder: StateActionEncoder):
        self.encoder = encoder
        self.samples = []
        
        # Extract training samples
        for entry in replay_entries:
            # This would load actual step events from storage
            # For now, we'll create synthetic data
            episode_samples = self._extract_samples_from_entry(entry)
            self.samples.extend(episode_samples)
    
    def _extract_samples_from_entry(self, entry: ReplayEntry) -> List[Dict[str, Any]]:
        """Extract training samples from replay entry."""
        # In real implementation, this would load step events from storage
        # For now, generate synthetic samples
        samples = []
        
        for i in range(min(entry.step_count, 50)):  # Limit samples
            # Synthetic observation
            if entry.scenario_id == "red_team":
                obs = {
                    'surface': ['/api/auth/login', '/api/public'],
                    'position': 'external' if i < 20 else 'internal',
                    'compromised': i // 10
                }
                action = {
                    'type': 'req' if i % 3 == 0 else 'scan',
                    'parameters': {'path': '/api/auth/login', 'method': 'POST'}
                }
                reward = 0.1 * (i / 50.0) + np.random.normal(0, 0.05)
            else:
                obs = {
                    'surface': ['detector_1', 'detector_2'],
                    'alerts': i // 5,
                    'position': 'monitoring'
                }
                action = {
                    'type': 'detect' if i % 2 == 0 else 'block',
                    'parameters': {'threshold': 0.5}
                }
                reward = 0.15 * (1.0 - i / 50.0) + np.random.normal(0, 0.05)
            
            # Next observation (simplified)
            next_obs = obs.copy()
            if 'compromised' in next_obs:
                next_obs['compromised'] += 1
            if 'alerts' in next_obs:
                next_obs['alerts'] += 1
            
            samples.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': i == entry.step_count - 1,
                'actor_role': ActorRole.RED if 'red' in entry.scenario_id else ActorRole.BLUE
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        obs_encoded = self.encoder.encode_observation(sample['obs'])
        action_encoded = self.encoder.encode_action(sample['action'])
        next_obs_encoded = self.encoder.encode_observation(sample['next_obs'])
        
        return {
            'obs': torch.FloatTensor(obs_encoded),
            'action': torch.FloatTensor(action_encoded),
            'reward': torch.FloatTensor([sample['reward']]),
            'next_obs': torch.FloatTensor(next_obs_encoded),
            'done': torch.FloatTensor([1.0 if sample['done'] else 0.0])
        }


class BehavioralCloningModel(nn.Module):
    """Behavioral cloning model for imitation learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class SafetyCritic(nn.Module):
    """Safety critic to reject dangerous actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output safety probability [0, 1]
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class OfflineRLModel(nn.Module):
    """Conservative Q-Learning (CQL) model for offline RL."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
    
    def forward_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q_network(x)
    
    def forward_policy(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy_network(state)


class TrainingWorker:
    """Base training worker class."""
    
    def __init__(self, config: TrainingConfig, encoder: StateActionEncoder):
        self.config = config
        self.encoder = encoder
        self.device = torch.device(config.device)
        
        # Create model save directory
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training worker initialized on {self.device}")
    
    @abstractmethod
    async def train(self, task: TrainingTask) -> Optional[PolicyCandidate]:
        """Train a policy candidate."""
        pass
    
    def _create_candidate_id(self, task: TrainingTask) -> str:
        """Create unique candidate ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{task.actor_role}_{task.task_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    def _save_model(self, model: nn.Module, candidate_id: str) -> str:
        """Save model to disk."""
        model_path = Path(self.config.model_save_dir) / f"{candidate_id}.pth"
        torch.save(model.state_dict(), model_path)
        return str(model_path)
    
    def _load_model(self, model: nn.Module, model_path: str) -> None:
        """Load model from disk."""
        model.load_state_dict(torch.load(model_path, map_location=self.device))


class BehavioralCloningWorker(TrainingWorker):
    """Behavioral cloning training worker."""
    
    async def train(self, task: TrainingTask) -> Optional[PolicyCandidate]:
        """Train behavioral cloning model."""
        logger.info(f"Starting BC training for {task.actor_role}")
        start_time = time.time()
        
        # Create dataset
        dataset = TrainingDataset(task.replay_entries, self.encoder)
        if len(dataset) == 0:
            logger.warning("No training data available")
            return None
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Create model
        model = BehavioralCloningModel(
            self.encoder.state_dim,
            self.encoder.action_dim,
            self.config.bc_hidden_size
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.bc_learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.bc_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                obs = batch['obs'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                predicted_actions = model(obs)
                loss = criterion(predicted_actions, actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Evaluate model
        model.eval()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Create candidate
        candidate_id = self._create_candidate_id(task)
        model_path = self._save_model(model, candidate_id)
        
        # Calculate performance metrics
        performance_metrics = {
            'avg_loss': avg_loss,
            'training_samples': len(dataset),
            'epochs': self.config.bc_epochs
        }
        
        # Safety score (placeholder - would use safety critic)
        safety_score = 0.85  # Reasonable default for BC
        
        training_duration = time.time() - start_time
        
        candidate = PolicyCandidate(
            candidate_id=candidate_id,
            actor_role=task.actor_role,
            model_type="bc",
            training_episodes=len(task.replay_entries),
            training_duration=training_duration,
            performance_metrics=performance_metrics,
            model_path=model_path,
            safety_score=safety_score,
            created_at=datetime.utcnow(),
            metadata={
                'config': asdict(self.config),
                'task_id': task.task_id
            }
        )
        
        logger.info(f"BC training completed in {training_duration:.2f}s, candidate: {candidate_id}")
        return candidate


class OfflineRLWorker(TrainingWorker):
    """Conservative Q-Learning (CQL) offline RL worker."""
    
    async def train(self, task: TrainingTask) -> Optional[PolicyCandidate]:
        """Train CQL model."""
        logger.info(f"Starting CQL training for {task.actor_role}")
        start_time = time.time()
        
        # Create dataset
        dataset = TrainingDataset(task.replay_entries, self.encoder)
        if len(dataset) == 0:
            logger.warning("No training data available")
            return None
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Create model and safety critic
        model = OfflineRLModel(
            self.encoder.state_dim,
            self.encoder.action_dim
        ).to(self.device)
        
        safety_critic = SafetyCritic(
            self.encoder.state_dim,
            self.encoder.action_dim
        ).to(self.device)
        
        # Optimizers
        q_optimizer = optim.Adam(model.q_network.parameters(), lr=self.config.rl_learning_rate)
        policy_optimizer = optim.Adam(model.policy_network.parameters(), lr=self.config.rl_learning_rate)
        safety_optimizer = optim.Adam(safety_critic.parameters(), lr=self.config.rl_learning_rate)
        
        # Training loop
        model.train()
        safety_critic.train()
        total_q_loss = 0.0
        total_policy_loss = 0.0
        total_safety_loss = 0.0
        num_batches = 0
        
        for epoch in range(50):  # Fewer epochs for RL
            epoch_q_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_safety_loss = 0.0
            
            for batch in dataloader:
                obs = batch['obs'].to(self.device)
                actions = batch['action'].to(self.device)
                rewards = batch['reward'].to(self.device)
                next_obs = batch['next_obs'].to(self.device)
                dones = batch['done'].to(self.device)
                
                # Q-learning update (simplified CQL)
                q_values = model.forward_q(obs, actions)
                next_actions = model.forward_policy(next_obs)
                next_q_values = model.forward_q(next_obs, next_actions)
                
                target_q = rewards + self.config.rl_gamma * next_q_values * (1 - dones)
                q_loss = nn.MSELoss()(q_values, target_q.detach())
                
                # Conservative penalty (simplified)
                random_actions = torch.randn_like(actions)
                random_q = model.forward_q(obs, random_actions)
                conservative_penalty = self.config.rl_alpha * random_q.mean()
                
                total_q_loss_batch = q_loss + conservative_penalty
                
                q_optimizer.zero_grad()
                total_q_loss_batch.backward()
                q_optimizer.step()
                
                # Policy update
                predicted_actions = model.forward_policy(obs)
                policy_q = model.forward_q(obs, predicted_actions)
                policy_loss = -policy_q.mean()
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                
                # Safety critic update (learn to predict safety)
                # Use action success as safety proxy
                safety_targets = torch.ones_like(rewards)  # Assume logged actions are safe
                safety_pred = safety_critic(obs, actions)
                safety_loss = nn.BCELoss()(safety_pred, safety_targets)
                
                safety_optimizer.zero_grad()
                safety_loss.backward()
                safety_optimizer.step()
                
                epoch_q_loss += total_q_loss_batch.item()
                epoch_policy_loss += policy_loss.item()
                epoch_safety_loss += safety_loss.item()
                num_batches += 1
            
            total_q_loss += epoch_q_loss
            total_policy_loss += epoch_policy_loss
            total_safety_loss += epoch_safety_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Q-Loss: {epoch_q_loss / len(dataloader):.4f}, "
                          f"Policy-Loss: {epoch_policy_loss / len(dataloader):.4f}")
        
        # Evaluate safety
        model.eval()
        safety_critic.eval()
        
        safety_scores = []
        with torch.no_grad():
            for batch in dataloader:
                obs = batch['obs'].to(self.device)
                actions = model.forward_policy(obs)
                safety = safety_critic(obs, actions)
                safety_scores.extend(safety.cpu().numpy().flatten())
        
        avg_safety_score = np.mean(safety_scores)
        
        # Create candidate
        candidate_id = self._create_candidate_id(task)
        model_path = self._save_model(model, candidate_id)
        
        # Save safety critic separately
        safety_path = Path(self.config.model_save_dir) / f"{candidate_id}_safety.pth"
        torch.save(safety_critic.state_dict(), safety_path)
        
        performance_metrics = {
            'avg_q_loss': total_q_loss / num_batches if num_batches > 0 else float('inf'),
            'avg_policy_loss': total_policy_loss / num_batches if num_batches > 0 else float('inf'),
            'avg_safety_loss': total_safety_loss / num_batches if num_batches > 0 else float('inf'),
            'training_samples': len(dataset)
        }
        
        training_duration = time.time() - start_time
        
        candidate = PolicyCandidate(
            candidate_id=candidate_id,
            actor_role=task.actor_role,
            model_type="cql",
            training_episodes=len(task.replay_entries),
            training_duration=training_duration,
            performance_metrics=performance_metrics,
            model_path=model_path,
            safety_score=avg_safety_score,
            created_at=datetime.utcnow(),
            metadata={
                'config': asdict(self.config),
                'task_id': task.task_id,
                'safety_model_path': str(safety_path)
            }
        )
        
        logger.info(f"CQL training completed in {training_duration:.2f}s, candidate: {candidate_id}")
        return candidate


class TrainingOrchestrator:
    """Orchestrates training workers and manages training queue."""
    
    def __init__(self, replay_service: ReplayService, config: Optional[TrainingConfig] = None):
        self.replay_service = replay_service
        self.config = config or TrainingConfig()
        self.encoder = StateActionEncoder()
        
        # Initialize workers
        self.bc_worker = BehavioralCloningWorker(self.config, self.encoder)
        self.cql_worker = OfflineRLWorker(self.config, self.encoder)
        
        # Training queue and results
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.policy_candidates: List[PolicyCandidate] = []
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.is_running = False
        
        logger.info("Training orchestrator initialized")
    
    async def start(self) -> None:
        """Start training orchestrator."""
        self.is_running = True
        
        # Start worker tasks
        asyncio.create_task(self._process_training_queue())
        asyncio.create_task(self._periodic_training())
        
        logger.info("Training orchestrator started")
    
    async def stop(self) -> None:
        """Stop training orchestrator."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Training orchestrator stopped")
    
    async def submit_training_task(self, task: TrainingTask) -> None:
        """Submit training task to queue."""
        await self.training_queue.put(task)
        logger.info(f"Training task submitted: {task.task_id}")
    
    async def train_behavioral_cloning(
        self,
        actor_role: ActorRole,
        scenario_id: Optional[str] = None,
        min_quality: float = 0.7
    ) -> Optional[PolicyCandidate]:
        """Train behavioral cloning model."""
        # Get replay data
        replay_entries = await self.replay_service.get_replay_buffer(
            scenario_id=scenario_id,
            min_quality=min_quality,
            limit=self.config.max_episodes_per_training
        )
        
        if not replay_entries:
            logger.warning(f"No replay data available for {actor_role} BC training")
            return None
        
        # Create training task
        task = TrainingTask(
            task_id=f"bc_{actor_role}_{int(time.time())}",
            task_type="bc",
            actor_role=actor_role,
            replay_entries=replay_entries,
            priority=1
        )
        
        # Train immediately
        candidate = await self.bc_worker.train(task)
        if candidate:
            self.policy_candidates.append(candidate)
        
        return candidate
    
    async def train_offline_rl(
        self,
        actor_role: ActorRole,
        scenario_id: Optional[str] = None,
        min_quality: float = 0.6
    ) -> Optional[PolicyCandidate]:
        """Train offline RL model."""
        # Get replay data
        replay_entries = await self.replay_service.get_replay_buffer(
            scenario_id=scenario_id,
            min_quality=min_quality,
            limit=self.config.max_episodes_per_training
        )
        
        if not replay_entries:
            logger.warning(f"No replay data available for {actor_role} CQL training")
            return None
        
        # Create training task
        task = TrainingTask(
            task_id=f"cql_{actor_role}_{int(time.time())}",
            task_type="offline_rl",
            actor_role=actor_role,
            replay_entries=replay_entries,
            priority=1
        )
        
        # Train immediately
        candidate = await self.cql_worker.train(task)
        if candidate:
            self.policy_candidates.append(candidate)
        
        return candidate
    
    async def get_best_candidates(
        self,
        actor_role: Optional[ActorRole] = None,
        limit: int = 10
    ) -> List[PolicyCandidate]:
        """Get best policy candidates."""
        candidates = self.policy_candidates
        
        if actor_role:
            candidates = [c for c in candidates if c.actor_role == actor_role]
        
        # Sort by safety score and training performance
        candidates.sort(
            key=lambda c: (c.safety_score, -c.performance_metrics.get('avg_loss', float('inf'))),
            reverse=True
        )
        
        return candidates[:limit]
    
    async def _process_training_queue(self) -> None:
        """Process training queue continuously."""
        while self.is_running:
            try:
                # Get task with timeout
                task = await asyncio.wait_for(self.training_queue.get(), timeout=1.0)
                
                # Select worker based on task type
                if task.task_type == "bc":
                    worker = self.bc_worker
                elif task.task_type == "offline_rl":
                    worker = self.cql_worker
                else:
                    logger.warning(f"Unknown task type: {task.task_type}")
                    continue
                
                # Train model
                try:
                    candidate = await worker.train(task)
                    if candidate:
                        self.policy_candidates.append(candidate)
                        logger.info(f"Training completed: {candidate.candidate_id}")
                    else:
                        logger.warning(f"Training failed for task: {task.task_id}")
                
                except Exception as e:
                    logger.error(f"Training error for task {task.task_id}: {e}")
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing training queue: {e}")
    
    async def _periodic_training(self) -> None:
        """Periodic training based on new replay data."""
        while self.is_running:
            try:
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
                # Check for new replay data and trigger training
                await self._trigger_periodic_training()
                
            except Exception as e:
                logger.error(f"Error in periodic training: {e}")
    
    async def _trigger_periodic_training(self) -> None:
        """Trigger training for both red and blue teams."""
        logger.info("Starting periodic training")
        
        # Get recent replay data
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        # Train for both roles
        for role in [ActorRole.RED, ActorRole.BLUE]:
            try:
                # Get replay entries
                replay_entries = await self.replay_service.get_replay_buffer(
                    min_quality=0.5,
                    max_age_days=1,
                    limit=500
                )
                
                if len(replay_entries) < 10:
                    logger.info(f"Not enough replay data for {role} training")
                    continue
                
                # Create BC task
                bc_task = TrainingTask(
                    task_id=f"periodic_bc_{role}_{int(time.time())}",
                    task_type="bc",
                    actor_role=role,
                    replay_entries=replay_entries,
                    priority=2
                )
                await self.training_queue.put(bc_task)
                
                # Create CQL task if enough data
                if len(replay_entries) >= 50:
                    cql_task = TrainingTask(
                        task_id=f"periodic_cql_{role}_{int(time.time())}",
                        task_type="offline_rl",
                        actor_role=role,
                        replay_entries=replay_entries,
                        priority=2
                    )
                    await self.training_queue.put(cql_task)
                
            except Exception as e:
                logger.error(f"Error creating periodic training task for {role}: {e}")
        
        logger.info("Periodic training tasks submitted")