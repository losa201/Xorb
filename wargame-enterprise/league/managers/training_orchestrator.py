#!/usr/bin/env python3
"""
Training Orchestrator for Cyber Range League
BC→IQL training pipeline with policy promotion and league management
"""

import os
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
import numpy as np
import hashlib
import pickle

# Machine Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. ML training features disabled.")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Reinforcement Learning
try:
    import gymnasium as gym
    from stable_baselines3 import SAC, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable-Baselines3 not available. RL training features disabled.")

class TrainingPhase(Enum):
    """Training pipeline phases"""
    DATA_COLLECTION = "data_collection"
    BEHAVIORAL_CLONING = "behavioral_cloning"
    IQL_TRAINING = "iql_training"
    POLICY_EVALUATION = "policy_evaluation"
    LEAGUE_PROMOTION = "league_promotion"

class PolicyStatus(Enum):
    """Policy status in the league"""
    TRAINING = "training"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CHAMPION = "champion"
    RETIRED = "retired"
    FAILED = "failed"

class AgentRole(Enum):
    """Agent roles in the cyber range"""
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    PURPLE_TEAM = "purple_team"

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data collection
    min_episodes_for_training: int = 100
    replay_buffer_size: int = 100000
    data_collection_workers: int = 4
    
    # Behavioral Cloning
    bc_epochs: int = 50
    bc_batch_size: int = 256
    bc_learning_rate: float = 3e-4
    bc_validation_split: float = 0.2
    
    # IQL Training
    iql_total_timesteps: int = 500000
    iql_learning_rate: float = 3e-4
    iql_gamma: float = 0.99
    iql_tau: float = 0.7  # IQL temperature parameter
    iql_beta: float = 3.0  # IQL regularization
    
    # Policy evaluation
    evaluation_episodes: int = 50
    evaluation_opponents: int = 3
    promotion_threshold: float = 0.55  # Must win >55% vs current champion
    
    # League management
    max_active_policies: int = 10
    champion_retention_days: int = 30
    training_frequency_hours: int = 6
    
    # Resource limits
    max_concurrent_trainings: int = 2
    training_timeout_hours: int = 12
    gpu_memory_limit: str = "8GB"

@dataclass
class EpisodeData:
    """Data from a single episode for training"""
    episode_id: str
    agent_role: AgentRole
    agent_id: str
    timestamp: str
    
    # Episode outcome
    total_reward: float
    success: bool
    duration_seconds: float
    
    # State-action trajectories
    states: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[Dict[str, Any]] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # Context
    environment_id: str = ""
    opponent_policies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyModel:
    """Trained policy model with metadata"""
    policy_id: str
    name: str
    agent_role: AgentRole
    status: PolicyStatus
    
    # Model details
    model_type: str  # "bc", "iql", "hybrid"
    model_path: str
    model_size_mb: float
    
    # Training metadata
    training_episodes: int
    training_duration_hours: float
    bc_accuracy: float = 0.0
    iql_reward: float = 0.0
    
    # Performance metrics
    win_rate: float = 0.0
    avg_reward: float = 0.0
    evaluation_episodes: int = 0
    last_evaluation: Optional[str] = None
    
    # League standings
    elo_rating: float = 1500.0
    league_wins: int = 0
    league_losses: int = 0
    promotion_date: Optional[str] = None
    
    # Versioning
    parent_policy_id: Optional[str] = None
    version: str = "1.0.0"
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class TrainingJob:
    """Training job configuration and status"""
    job_id: str
    name: str
    agent_role: AgentRole
    training_phase: TrainingPhase
    
    # Configuration
    config: TrainingConfig
    data_source: str  # Path to training data
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    current_epoch: int = 0
    
    # Resources
    assigned_gpu: Optional[str] = None
    process_id: Optional[int] = None
    memory_usage_mb: float = 0.0
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    estimated_completion: Optional[str] = None
    
    # Results
    result_policy: Optional[PolicyModel] = None
    training_logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

class BehavioralCloningTrainer:
    """Behavioral Cloning trainer for imitation learning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu")
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def prepare_data(self, episodes: List[EpisodeData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare episode data for behavioral cloning"""
        if not episodes:
            raise ValueError("No episode data provided")
        
        states = []
        actions = []
        
        for episode in episodes:
            for i, (state, action) in enumerate(zip(episode.states, episode.actions)):
                # Convert state to feature vector
                state_features = self._extract_state_features(state)
                # Convert action to target vector
                action_vector = self._extract_action_features(action)
                
                states.append(state_features)
                actions.append(action_vector)
        
        X = np.array(states)
        y = np.array(actions)
        
        # Normalize features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        return X, y
    
    def _extract_state_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from state"""
        features = []
        
        # Network state features
        features.extend([
            state.get("network_connectivity", 0.0),
            state.get("compromised_hosts", 0.0) / 10.0,  # Normalize
            state.get("detected_activities", 0.0) / 5.0,
            state.get("security_level", 0.5),
        ])
        
        # Agent state features
        features.extend([
            state.get("current_access_level", 0.0) / 3.0,  # 0=none, 1=user, 2=admin, 3=root
            state.get("available_actions", 0.0) / 20.0,
            state.get("session_duration", 0.0) / 3600.0,  # Hours
            state.get("stealth_score", 0.5),
        ])
        
        # Environment features
        features.extend([
            state.get("time_of_day", 12.0) / 24.0,
            state.get("system_load", 0.5),
            state.get("monitoring_level", 0.5),
            state.get("vulnerability_count", 0.0) / 10.0,
        ])
        
        # Opponent features
        features.extend([
            state.get("blue_team_activity", 0.0) / 10.0,
            state.get("defense_strength", 0.5),
            state.get("incident_response_time", 300.0) / 3600.0,  # Normalize to hours
        ])
        
        # Pad or truncate to fixed size
        target_size = 64
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_action_features(self, action: Dict[str, Any]) -> np.ndarray:
        """Extract action features for prediction target"""
        features = []
        
        # Action type (one-hot encoded)
        action_types = [
            "reconnaissance", "initial_access", "persistence", "privilege_escalation",
            "defense_evasion", "credential_access", "discovery", "lateral_movement",
            "collection", "exfiltration", "impact", "defense", "response"
        ]
        
        action_type = action.get("action_type", "").lower()
        action_one_hot = [1.0 if action_type == at else 0.0 for at in action_types]
        features.extend(action_one_hot)
        
        # Action parameters
        features.extend([
            action.get("intensity", 0.5),  # Action intensity/aggressiveness
            action.get("stealth", 0.5),    # Stealth level
            action.get("risk", 0.5),       # Risk level
            action.get("duration", 1.0) / 10.0,  # Expected duration
        ])
        
        # Target selection
        features.extend([
            action.get("target_priority", 0.5),
            action.get("target_vulnerability", 0.5),
            action.get("success_probability", 0.5),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def train(self, episodes: List[EpisodeData], job: TrainingJob) -> PolicyModel:
        """Train behavioral cloning model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for BC training")
        
        job.status = "running"
        job.start_time = datetime.utcnow().isoformat()
        
        try:
            # Prepare data
            X, y = self.prepare_data(episodes)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.config.bc_validation_split)
            
            # Create model
            input_size = X.shape[1]
            output_size = y.shape[1]
            
            model = self._create_bc_model(input_size, output_size).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.config.bc_learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(self.config.bc_epochs):
                job.current_epoch = epoch
                job.progress = epoch / self.config.bc_epochs
                
                # Training phase
                model.train()
                train_loss = 0.0
                
                for i in range(0, len(X_train), self.config.bc_batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i+self.config.bc_batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train[i:i+self.config.bc_batch_size]).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for i in range(0, len(X_val), self.config.bc_batch_size):
                        batch_X = torch.FloatTensor(X_val[i:i+self.config.bc_batch_size]).to(self.device)
                        batch_y = torch.FloatTensor(y_val[i:i+self.config.bc_batch_size]).to(self.device)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / (len(X_train) // self.config.bc_batch_size + 1)
                avg_val_loss = val_loss / (len(X_val) // self.config.bc_batch_size + 1)
                
                job.training_logs.append(
                    f"Epoch {epoch+1}/{self.config.bc_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f"/tmp/bc_model_{job.job_id}_best.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        job.training_logs.append(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Calculate final accuracy
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(X_val).to(self.device))
                val_predictions = val_outputs.cpu().numpy()
                
                # For multi-output regression, calculate R² score
                from sklearn.metrics import r2_score
                accuracy = r2_score(y_val, val_predictions)
            
            # Save final model
            model_path = f"/tmp/bc_model_{job.job_id}_final.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scaler,
                'input_size': input_size,
                'output_size': output_size,
                'accuracy': accuracy
            }, model_path)
            
            # Create policy model
            policy = PolicyModel(
                policy_id=f"bc_{job.job_id}_{int(time.time())}",
                name=f"BC Policy {job.name}",
                agent_role=job.agent_role,
                status=PolicyStatus.CANDIDATE,
                model_type="bc",
                model_path=model_path,
                model_size_mb=os.path.getsize(model_path) / (1024 * 1024),
                training_episodes=len(episodes),
                training_duration_hours=(datetime.utcnow() - datetime.fromisoformat(job.start_time.replace('Z', '+00:00'))).total_seconds() / 3600,
                bc_accuracy=accuracy
            )
            
            job.status = "completed"
            job.end_time = datetime.utcnow().isoformat()
            job.result_policy = policy
            
            return policy
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.utcnow().isoformat()
            raise
    
    def _create_bc_model(self, input_size: int, output_size: int) -> nn.Module:
        """Create behavioral cloning neural network"""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size),
            nn.Tanh()  # Output in [-1, 1] range
        )

class IQLTrainer:
    """Implicit Q-Learning trainer for offline RL"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu")
    
    def train(self, episodes: List[EpisodeData], bc_policy: Optional[PolicyModel], job: TrainingJob) -> PolicyModel:
        """Train IQL model from offline data"""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for IQL training")
        
        job.status = "running"
        if not job.start_time:
            job.start_time = datetime.utcnow().isoformat()
        
        try:
            # Prepare offline dataset
            replay_buffer = self._prepare_offline_dataset(episodes)
            
            # Create environment wrapper
            env = self._create_cyber_range_env()
            
            # Initialize IQL model
            model = self._create_iql_model(env, bc_policy)
            
            # Train model
            model.learn(
                total_timesteps=self.config.iql_total_timesteps,
                callback=self._create_training_callback(job)
            )
            
            # Evaluate final performance
            mean_reward = self._evaluate_policy(model, env)
            
            # Save model
            model_path = f"/tmp/iql_model_{job.job_id}_final.zip"
            model.save(model_path)
            
            # Create policy model
            policy = PolicyModel(
                policy_id=f"iql_{job.job_id}_{int(time.time())}",
                name=f"IQL Policy {job.name}",
                agent_role=job.agent_role,
                status=PolicyStatus.CANDIDATE,
                model_type="iql",
                model_path=model_path,
                model_size_mb=os.path.getsize(model_path) / (1024 * 1024),
                training_episodes=len(episodes),
                training_duration_hours=(datetime.utcnow() - datetime.fromisoformat(job.start_time.replace('Z', '+00:00'))).total_seconds() / 3600,
                iql_reward=mean_reward,
                parent_policy_id=bc_policy.policy_id if bc_policy else None
            )
            
            job.status = "completed"
            job.end_time = datetime.utcnow().isoformat()
            job.result_policy = policy
            
            return policy
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.utcnow().isoformat()
            raise
    
    def _prepare_offline_dataset(self, episodes: List[EpisodeData]) -> Dict[str, np.ndarray]:
        """Convert episodes to offline RL dataset"""
        # Implementation would create replay buffer from episodes
        # This is a simplified version
        return {"observations": np.array([]), "actions": np.array([]), "rewards": np.array([])}
    
    def _create_cyber_range_env(self):
        """Create cyber range environment for RL"""
        # This would create a gym environment wrapper for the cyber range
        # For now, return a dummy environment
        class DummyCyberRangeEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(20,))
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
            
            def reset(self, **kwargs):
                return np.zeros(64), {}
            
            def step(self, action):
                return np.zeros(64), 0.0, False, False, {}
        
        return DummyCyberRangeEnv()
    
    def _create_iql_model(self, env, bc_policy: Optional[PolicyModel]):
        """Create IQL model with optional BC initialization"""
        # For demo purposes, using SAC as placeholder for IQL
        return SAC("MlpPolicy", env, learning_rate=self.config.iql_learning_rate)
    
    def _create_training_callback(self, job: TrainingJob):
        """Create callback for training progress"""
        class TrainingCallback:
            def __init__(self, job):
                self.job = job
                self.start_time = time.time()
            
            def __call__(self, locals, globals):
                elapsed = time.time() - self.start_time
                progress = locals.get('self').num_timesteps / locals.get('self').total_timesteps
                self.job.progress = progress
                return True
        
        return TrainingCallback(job)
    
    def _evaluate_policy(self, model, env, n_eval_episodes: int = 10) -> float:
        """Evaluate policy performance"""
        total_reward = 0.0
        
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / n_eval_episodes

class LeagueManager:
    """Manages policy league and promotion system"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.policies: Dict[str, PolicyModel] = {}
        self.champions: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}
        self.active_policies: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}
        
        # ELO rating system
        self.base_elo = 1500.0
        self.k_factor = 32.0
        
        # Statistics
        self.matches_played = 0
        self.promotions = 0
        self.retirements = 0
        
        # Load existing policies
        self._load_policies()
    
    def _load_policies(self):
        """Load existing policies from storage"""
        # Implementation would load from persistent storage
        pass
    
    def register_policy(self, policy: PolicyModel):
        """Register a new policy for evaluation"""
        self.policies[policy.policy_id] = policy
        policy.status = PolicyStatus.CANDIDATE
        
        logging.info(f"Registered new policy: {policy.name} ({policy.policy_id})")
    
    def evaluate_candidate(self, candidate_id: str) -> bool:
        """Evaluate a candidate policy for promotion"""
        if candidate_id not in self.policies:
            return False
        
        candidate = self.policies[candidate_id]
        if candidate.status != PolicyStatus.CANDIDATE:
            return False
        
        # Get current champion for this role
        champions = self.champions[candidate.agent_role]
        if not champions:
            # No current champion, automatic promotion
            self._promote_policy(candidate_id)
            return True
        
        current_champion_id = champions[0]
        current_champion = self.policies[current_champion_id]
        
        # Evaluate against current champion and historical champions
        win_rate = self._evaluate_against_champions(candidate, candidate.agent_role)
        
        if win_rate >= self.config.promotion_threshold:
            self._promote_policy(candidate_id)
            return True
        else:
            candidate.status = PolicyStatus.FAILED
            candidate.win_rate = win_rate
            logging.info(f"Policy {candidate.name} failed promotion with {win_rate:.2%} win rate")
            return False
    
    def _evaluate_against_champions(self, candidate: PolicyModel, role: AgentRole) -> float:
        """Evaluate candidate against current and historical champions"""
        champions = self.champions[role]
        if not champions:
            return 1.0  # Automatic win if no champions
        
        total_matches = 0
        total_wins = 0
        
        # Evaluate against current champion
        current_champion_id = champions[0]
        if current_champion_id in self.policies:
            wins, matches = self._simulate_matches(candidate, self.policies[current_champion_id])
            total_wins += wins
            total_matches += matches
        
        # Evaluate against historical champions (last 2)
        for champion_id in champions[1:3]:  # Last 2 historical champions
            if champion_id in self.policies:
                wins, matches = self._simulate_matches(candidate, self.policies[champion_id])
                total_wins += wins
                total_matches += matches
        
        return total_wins / total_matches if total_matches > 0 else 0.0
    
    def _simulate_matches(self, policy1: PolicyModel, policy2: PolicyModel) -> Tuple[int, int]:
        """Simulate matches between two policies"""
        # This would run actual cyber range episodes
        # For demo, using simplified simulation
        
        wins = 0
        total_matches = self.config.evaluation_episodes
        
        for _ in range(total_matches):
            # Simulate match outcome based on policy metrics
            p1_strength = (policy1.bc_accuracy + policy1.iql_reward + policy1.elo_rating / 2000.0) / 3.0
            p2_strength = (policy2.bc_accuracy + policy2.iql_reward + policy2.elo_rating / 2000.0) / 3.0
            
            # Add some randomness
            p1_strength += np.random.normal(0, 0.1)
            p2_strength += np.random.normal(0, 0.1)
            
            if p1_strength > p2_strength:
                wins += 1
        
        # Update ELO ratings
        self._update_elo_ratings(policy1, policy2, wins, total_matches)
        self.matches_played += total_matches
        
        return wins, total_matches
    
    def _update_elo_ratings(self, policy1: PolicyModel, policy2: PolicyModel, p1_wins: int, total_matches: int):
        """Update ELO ratings based on match results"""
        p1_score = p1_wins / total_matches
        p2_score = 1.0 - p1_score
        
        # Calculate expected scores
        p1_expected = 1 / (1 + 10 ** ((policy2.elo_rating - policy1.elo_rating) / 400))
        p2_expected = 1 / (1 + 10 ** ((policy1.elo_rating - policy2.elo_rating) / 400))
        
        # Update ratings
        policy1.elo_rating += self.k_factor * (p1_score - p1_expected)
        policy2.elo_rating += self.k_factor * (p2_score - p2_expected)
        
        # Update match records
        policy1.league_wins += p1_wins
        policy1.league_losses += (total_matches - p1_wins)
        policy2.league_wins += (total_matches - p1_wins)
        policy2.league_losses += p1_wins
    
    def _promote_policy(self, policy_id: str):
        """Promote a policy to champion status"""
        policy = self.policies[policy_id]
        role = policy.agent_role
        
        # Update status
        policy.status = PolicyStatus.CHAMPION
        policy.promotion_date = datetime.utcnow().isoformat()
        
        # Update champions list
        champions = self.champions[role]
        if policy_id not in champions:
            champions.insert(0, policy_id)  # New champion at front
        
        # Demote old champions
        for i, champion_id in enumerate(champions[1:], 1):
            if champion_id in self.policies:
                old_champion = self.policies[champion_id]
                if i == 1:  # Previous champion becomes active
                    old_champion.status = PolicyStatus.ACTIVE
                    if champion_id not in self.active_policies[role]:
                        self.active_policies[role].append(champion_id)
                else:  # Older champions retire
                    old_champion.status = PolicyStatus.RETIRED
                    if champion_id in self.active_policies[role]:
                        self.active_policies[role].remove(champion_id)
        
        # Limit number of champions retained
        max_champions = 5
        if len(champions) > max_champions:
            retired_ids = champions[max_champions:]
            for retired_id in retired_ids:
                if retired_id in self.policies:
                    self.policies[retired_id].status = PolicyStatus.RETIRED
            self.champions[role] = champions[:max_champions]
        
        self.promotions += 1
        logging.info(f"Promoted policy {policy.name} to champion for {role.value}")
    
    def get_active_policies(self, role: AgentRole) -> List[PolicyModel]:
        """Get currently active policies for a role"""
        active_ids = self.champions[role][:self.config.max_active_policies]
        return [self.policies[pid] for pid in active_ids if pid in self.policies]
    
    def get_league_standings(self, role: AgentRole) -> List[PolicyModel]:
        """Get league standings for a role"""
        role_policies = [p for p in self.policies.values() if p.agent_role == role]
        return sorted(role_policies, key=lambda x: x.elo_rating, reverse=True)
    
    def retire_old_policies(self):
        """Retire policies that are too old or underperforming"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.champion_retention_days)
        
        for policy in self.policies.values():
            if policy.status == PolicyStatus.ACTIVE:
                if policy.promotion_date:
                    promotion_dt = datetime.fromisoformat(policy.promotion_date.replace('Z', '+00:00'))
                    if promotion_dt < cutoff_date:
                        policy.status = PolicyStatus.RETIRED
                        self.retirements += 1
                        logging.info(f"Retired old policy: {policy.name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get league statistics"""
        role_stats = {}
        for role in AgentRole:
            role_policies = [p for p in self.policies.values() if p.agent_role == role]
            role_stats[role.value] = {
                "total_policies": len(role_policies),
                "active_policies": len([p for p in role_policies if p.status == PolicyStatus.ACTIVE]),
                "champion_policies": len([p for p in role_policies if p.status == PolicyStatus.CHAMPION]),
                "candidate_policies": len([p for p in role_policies if p.status == PolicyStatus.CANDIDATE]),
                "avg_elo": np.mean([p.elo_rating for p in role_policies]) if role_policies else 1500.0
            }
        
        return {
            "total_policies": len(self.policies),
            "matches_played": self.matches_played,
            "promotions": self.promotions,
            "retirements": self.retirements,
            "role_statistics": role_stats,
            "last_updated": datetime.utcnow().isoformat()
        }

class TrainingOrchestrator:
    """Main orchestrator for training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.bc_trainer = BehavioralCloningTrainer(config)
        self.iql_trainer = IQLTrainer(config)
        self.league_manager = LeagueManager(config)
        
        # Job management
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[str] = []
        self.active_jobs: Set[str] = set()
        
        # Data management
        self.episode_buffer: List[EpisodeData] = []
        
        # Threading
        self._stop_event = threading.Event()
        self._orchestrator_thread = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TrainingOrchestrator')
        
        # Start orchestrator
        self.start()
    
    def start(self):
        """Start the training orchestrator"""
        if self._orchestrator_thread is None or not self._orchestrator_thread.is_alive():
            self._stop_event.clear()
            self._orchestrator_thread = threading.Thread(target=self._orchestrator_worker)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()
            self.logger.info("Training orchestrator started")
    
    def stop(self):
        """Stop the training orchestrator"""
        self._stop_event.set()
        if self._orchestrator_thread:
            self._orchestrator_thread.join(timeout=30)
        self.logger.info("Training orchestrator stopped")
    
    def add_episode_data(self, episode: EpisodeData):
        """Add episode data to the training buffer"""
        self.episode_buffer.append(episode)
        
        # Trigger training if we have enough data
        if len(self.episode_buffer) >= self.config.min_episodes_for_training:
            self._maybe_trigger_training(episode.agent_role)
    
    def _maybe_trigger_training(self, agent_role: AgentRole):
        """Check if we should trigger a new training job"""
        # Check if we already have enough active jobs
        if len(self.active_jobs) >= self.config.max_concurrent_trainings:
            return
        
        # Check if we recently trained for this role
        recent_jobs = [
            job for job in self.training_jobs.values()
            if job.agent_role == agent_role and job.status in ["completed", "running"]
        ]
        
        if recent_jobs:
            last_training = max(recent_jobs, key=lambda x: x.start_time or "")
            if last_training.start_time:
                last_time = datetime.fromisoformat(last_training.start_time.replace('Z', '+00:00'))
                if datetime.utcnow() - last_time < timedelta(hours=self.config.training_frequency_hours):
                    return
        
        # Create new training job
        self._create_training_job(agent_role)
    
    def _create_training_job(self, agent_role: AgentRole):
        """Create a new training job"""
        job_id = f"train_{agent_role.value}_{int(time.time())}"
        
        # Filter episodes for this role
        role_episodes = [ep for ep in self.episode_buffer if ep.agent_role == agent_role]
        
        if len(role_episodes) < self.config.min_episodes_for_training:
            return
        
        job = TrainingJob(
            job_id=job_id,
            name=f"Training {agent_role.value} policy",
            agent_role=agent_role,
            training_phase=TrainingPhase.BEHAVIORAL_CLONING,
            config=self.config,
            data_source=f"episode_buffer_{len(role_episodes)}_episodes"
        )
        
        self.training_jobs[job_id] = job
        self.job_queue.append(job_id)
        self.logger.info(f"Created training job: {job_id}")
    
    def _orchestrator_worker(self):
        """Main orchestrator worker loop"""
        while not self._stop_event.is_set():
            try:
                # Process job queue
                self._process_job_queue()
                
                # Check active jobs
                self._check_active_jobs()
                
                # Retire old policies
                self.league_manager.retire_old_policies()
                
                # Sleep before next iteration
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in orchestrator worker: {e}")
                time.sleep(30)
    
    def _process_job_queue(self):
        """Process pending training jobs"""
        while (self.job_queue and 
               len(self.active_jobs) < self.config.max_concurrent_trainings):
            
            job_id = self.job_queue.pop(0)
            if job_id not in self.training_jobs:
                continue
            
            job = self.training_jobs[job_id]
            
            # Start training job in separate thread
            training_thread = threading.Thread(target=self._execute_training_job, args=(job,))
            training_thread.daemon = True
            training_thread.start()
            
            self.active_jobs.add(job_id)
            self.logger.info(f"Started training job: {job_id}")
    
    def _execute_training_job(self, job: TrainingJob):
        """Execute a training job"""
        try:
            # Get episodes for this role
            role_episodes = [ep for ep in self.episode_buffer if ep.agent_role == job.agent_role]
            
            # Phase 1: Behavioral Cloning
            job.training_phase = TrainingPhase.BEHAVIORAL_CLONING
            bc_policy = self.bc_trainer.train(role_episodes, job)
            
            if bc_policy and job.status == "completed":
                # Phase 2: IQL Training
                job.training_phase = TrainingPhase.IQL_TRAINING
                iql_policy = self.iql_trainer.train(role_episodes, bc_policy, job)
                
                if iql_policy and job.status == "completed":
                    # Phase 3: League Evaluation
                    job.training_phase = TrainingPhase.LEAGUE_PROMOTION
                    self.league_manager.register_policy(iql_policy)
                    
                    if self.league_manager.evaluate_candidate(iql_policy.policy_id):
                        self.logger.info(f"Policy promoted: {iql_policy.name}")
                    else:
                        self.logger.info(f"Policy failed promotion: {iql_policy.name}")
        
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            self.logger.error(f"Training job {job.job_id} failed: {e}")
        
        finally:
            self.active_jobs.discard(job.job_id)
    
    def _check_active_jobs(self):
        """Check status of active training jobs"""
        completed_jobs = []
        
        for job_id in list(self.active_jobs):
            if job_id not in self.training_jobs:
                self.active_jobs.discard(job_id)
                continue
            
            job = self.training_jobs[job_id]
            
            # Check for timeout
            if job.start_time:
                start_time = datetime.fromisoformat(job.start_time.replace('Z', '+00:00'))
                if datetime.utcnow() - start_time > timedelta(hours=self.config.training_timeout_hours):
                    job.status = "failed"
                    job.error_message = "Training timeout"
                    self.active_jobs.discard(job_id)
                    self.logger.warning(f"Training job {job_id} timed out")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "total_jobs": len(self.training_jobs),
            "episode_buffer_size": len(self.episode_buffer),
            "league_statistics": self.league_manager.get_statistics(),
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                    "progress": job.progress,
                    "phase": job.training_phase.value
                }
                for job in sorted(self.training_jobs.values(), 
                                key=lambda x: x.start_time or "", reverse=True)[:10]
            ]
        }

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Training Orchestrator...")
    
    # Create configuration
    config = TrainingConfig(
        min_episodes_for_training=5,  # Lower for testing
        bc_epochs=3,  # Reduced for testing
        iql_total_timesteps=1000,  # Reduced for testing
        evaluation_episodes=3
    )
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    try:
        # Create some mock episode data
        for i in range(10):
            episode = EpisodeData(
                episode_id=f"ep_{i:03d}",
                agent_role=AgentRole.RED_TEAM,
                agent_id=f"red_agent_{i % 3}",
                timestamp=datetime.utcnow().isoformat(),
                total_reward=np.random.uniform(0, 100),
                success=np.random.random() > 0.5,
                duration_seconds=np.random.uniform(300, 1800),
                states=[{"network_connectivity": np.random.random()} for _ in range(10)],
                actions=[{"action_type": "reconnaissance", "intensity": np.random.random()} for _ in range(10)],
                rewards=[np.random.uniform(-1, 10) for _ in range(10)]
            )
            orchestrator.add_episode_data(episode)
        
        # Let it run for a short time
        time.sleep(5)
        
        # Get status
        status = orchestrator.get_status()
        print(f"Orchestrator status: {json.dumps(status, indent=2)}")
        
        # Get league standings
        standings = orchestrator.league_manager.get_league_standings(AgentRole.RED_TEAM)
        print(f"Red team policies: {len(standings)}")
        
    finally:
        orchestrator.stop()
    
    print("Training orchestrator test completed!")