#!/usr/bin/env python3
"""
Transformer-Based Policy Selector for Advanced Agent Orchestration

This module implements a transformer architecture for context-aware agent selection,
replacing the DQN approach with attention mechanisms that can better handle
sequential dependencies and multi-modal campaign features.
"""

import asyncio
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

@dataclass
class CampaignContext:
    """Rich context representation for transformer input"""
    # Campaign features
    target_complexity_vector: torch.Tensor  # [n_targets, feature_dim]
    agent_capability_matrix: torch.Tensor   # [n_agents, capability_dim]
    historical_embeddings: torch.Tensor     # [sequence_len, hidden_dim]
    temporal_features: torch.Tensor         # [time_features]
    resource_state: torch.Tensor           # [epyc_cores, numa_nodes, thermal, memory]
    
    # Attention masks
    target_mask: torch.Tensor              # Valid targets mask
    agent_mask: torch.Tensor               # Available agents mask
    sequence_mask: torch.Tensor            # Valid history mask


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention between campaign targets and available agents"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, targets: torch.Tensor, agents: torch.Tensor, 
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention: targets query agents
        attended, attention_weights = self.attention(
            query=targets, key=agents, value=agents, 
            key_padding_mask=target_mask
        )
        return self.norm(attended + targets), attention_weights


class EPYCResourceEncoder(nn.Module):
    """EPYC-specific resource state encoder"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        
        # EPYC topology-aware encoding
        self.numa_encoder = nn.Linear(2, 32)  # 2 NUMA nodes
        self.ccx_encoder = nn.Linear(8, 64)   # 8 CCX complexes
        self.thermal_encoder = nn.Linear(4, 32)  # Thermal sensors
        self.memory_encoder = nn.Linear(4, 32)   # Memory channels
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 64 + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, resource_state: torch.Tensor) -> torch.Tensor:
        # Decompose resource state
        numa_state = resource_state[:, :2]      # NUMA utilization
        ccx_state = resource_state[:, 2:10]     # CCX utilization  
        thermal_state = resource_state[:, 10:14]  # Thermal metrics
        memory_state = resource_state[:, 14:18]   # Memory metrics
        
        # Encode each component
        numa_encoded = self.numa_encoder(numa_state)
        ccx_encoded = self.ccx_encoder(ccx_state)
        thermal_encoded = self.thermal_encoder(thermal_state)
        memory_encoded = self.memory_encoder(memory_state)
        
        # Fuse representations
        combined = torch.cat([numa_encoded, ccx_encoded, thermal_encoded, memory_encoded], dim=-1)
        return self.fusion(combined)


class TransformerPolicySelector(nn.Module):
    """
    Transformer-based policy selector for intelligent agent orchestration
    """
    
    def __init__(self, 
                 agent_vocab_size: int = 50,
                 target_feature_dim: int = 64,
                 agent_capability_dim: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_sequence_length: int = 100):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Input embeddings
        self.agent_embedding = nn.Embedding(agent_vocab_size, hidden_dim)
        self.target_projection = nn.Linear(target_feature_dim, hidden_dim)
        self.capability_projection = nn.Linear(agent_capability_dim, hidden_dim)
        
        # Positional encoding for sequences
        self.positional_encoding = self._create_positional_encoding(max_sequence_length, hidden_dim)
        
        # EPYC resource encoder
        self.resource_encoder = EPYCResourceEncoder(input_dim=18, hidden_dim=hidden_dim)
        
        # Multi-modal transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention between targets and agents
        self.cross_attention = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        
        # Context fusion layers
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # targets + agents + resources
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Policy head - outputs probability distribution over agent selections
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, agent_vocab_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head for policy gradient methods
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output uncertainty between 0-1
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, context: CampaignContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context.target_complexity_vector.size(0)
        
        # Encode inputs
        target_embeddings = self.target_projection(context.target_complexity_vector)
        agent_embeddings = self.capability_projection(context.agent_capability_matrix)
        resource_embeddings = self.resource_encoder(context.resource_state)
        
        # Add positional encoding to historical embeddings
        seq_len = context.historical_embeddings.size(1)
        if seq_len <= self.max_sequence_length:
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(context.historical_embeddings.device)
            historical_with_pos = context.historical_embeddings + pos_encoding
        else:
            historical_with_pos = context.historical_embeddings
        
        # Apply transformer to historical sequence
        historical_encoded = self.transformer(
            historical_with_pos,
            src_key_padding_mask=context.sequence_mask
        )
        
        # Use last hidden state as campaign history representation
        campaign_history = historical_encoded[:, -1, :]  # [batch_size, hidden_dim]
        
        # Cross-attention between targets and agents
        target_agent_attended, attention_weights = self.cross_attention(
            target_embeddings, agent_embeddings, context.target_mask
        )
        
        # Pool target representations (mean pooling with mask)
        if context.target_mask is not None:
            mask_expanded = context.target_mask.unsqueeze(-1).expand_as(target_agent_attended)
            masked_targets = target_agent_attended.masked_fill(mask_expanded, 0)
            target_pooled = masked_targets.sum(dim=1) / (~context.target_mask).sum(dim=1, keepdim=True).float()
        else:
            target_pooled = target_agent_attended.mean(dim=1)
        
        # Pool agent representations
        if context.agent_mask is not None:
            mask_expanded = context.agent_mask.unsqueeze(-1).expand_as(agent_embeddings)
            masked_agents = agent_embeddings.masked_fill(mask_expanded, 0)
            agent_pooled = masked_agents.sum(dim=1) / (~context.agent_mask).sum(dim=1, keepdim=True).float()
        else:
            agent_pooled = agent_embeddings.mean(dim=1)
        
        # Combine all context representations
        combined_context = torch.cat([
            target_pooled,           # Target-agent attention result
            agent_pooled,            # Agent capabilities
            resource_embeddings      # EPYC resource state
        ], dim=-1)
        
        # Fuse contexts
        fused_context = self.context_fusion(combined_context)
        
        # Add campaign history
        policy_input = fused_context + campaign_history
        
        # Generate outputs
        action_probs = self.policy_head(policy_input)
        state_value = self.value_head(policy_input)
        uncertainty = self.uncertainty_head(policy_input)
        
        return action_probs, state_value, uncertainty


@dataclass
class TransformerTrainingConfig:
    """Configuration for transformer training"""
    learning_rate: float = 3e-4
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 4000
    max_epochs: int = 100
    patience: int = 10
    
    # Policy gradient specific
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    
    # Uncertainty-aware training
    uncertainty_weight: float = 0.1
    uncertainty_threshold: float = 0.3


class TransformerOrchestrator:
    """
    Advanced orchestrator using transformer-based policy selection
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any] = None,
                 training_config: TransformerTrainingConfig = None,
                 device: str = 'auto'):
        
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model configuration
        default_config = {
            'agent_vocab_size': 50,
            'target_feature_dim': 64,
            'agent_capability_dim': 32,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
            'max_sequence_length': 100
        }
        self.model_config = {**default_config, **(model_config or {})}
        
        # Training configuration
        self.training_config = training_config or TransformerTrainingConfig()
        
        # Initialize model
        self.model = TransformerPolicySelector(**self.model_config).to(self.device)
        
        # Optimizer with warm-up schedule
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        # Training state
        self.training_step = 0
        self.epoch = 0
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        
        # Experience buffer for policy gradient training
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Metrics tracking
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'uncertainty_loss': [],
            'total_loss': [],
            'validation_accuracy': [],
            'average_reward': []
        }
        
        self.logger.info(f"Transformer orchestrator initialized on {self.device}")
    
    async def select_agents(self, 
                          campaign_context: Dict[str, Any],
                          available_agents: List[Dict[str, Any]],
                          system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select optimal agents using transformer policy
        """
        
        # Convert context to transformer input format
        context = await self._prepare_transformer_context(
            campaign_context, available_agents, system_state
        )
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            action_probs, state_value, uncertainty = self.model(context)
        
        # Convert probabilities to agent selections
        selected_agents = await self._probs_to_agent_selection(
            action_probs, available_agents, uncertainty
        )
        
        # Store experience for training (if in training mode)
        if self.model.training:
            experience = {
                'context': context,
                'action_probs': action_probs,
                'state_value': state_value,
                'uncertainty': uncertainty,
                'selected_agents': selected_agents,
                'timestamp': datetime.utcnow()
            }
            self.experience_buffer.append(experience)
            
            # Limit buffer size
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
        
        return selected_agents
    
    async def _prepare_transformer_context(self,
                                         campaign_context: Dict[str, Any],
                                         available_agents: List[Dict[str, Any]],
                                         system_state: Dict[str, Any]) -> CampaignContext:
        """Convert raw context to transformer-ready format"""
        
        # Extract target features
        targets = campaign_context.get('targets', [])
        target_features = []
        
        for target in targets:
            # Feature engineering for targets
            features = [
                len(target.get('ports', [])) / 100.0,      # Normalized port count
                len(target.get('services', [])) / 50.0,    # Normalized service count
                target.get('complexity_score', 0.5),       # Complexity score
                target.get('security_score', 0.5),         # Security score
                target.get('criticality', 0.5)             # Criticality
            ]
            # Pad to target_feature_dim
            features.extend([0.0] * (self.model_config['target_feature_dim'] - len(features)))
            target_features.append(features)
        
        # Pad to ensure consistent batch size
        if len(target_features) == 0:
            target_features = [[0.0] * self.model_config['target_feature_dim']]
        
        # Extract agent capabilities
        agent_capabilities = []
        for agent in available_agents:
            capabilities = [
                agent.get('resource_requirement', 0.5),
                agent.get('success_rate', 0.5),
                agent.get('execution_time', 0.5),
                agent.get('reliability', 0.5)
            ]
            # Pad to agent_capability_dim
            capabilities.extend([0.0] * (self.model_config['agent_capability_dim'] - len(capabilities)))
            agent_capabilities.append(capabilities)
        
        if len(agent_capabilities) == 0:
            agent_capabilities = [[0.0] * self.model_config['agent_capability_dim']]
        
        # Extract historical context (mock for now)
        # In production, this would come from campaign history
        historical_features = []
        for i in range(10):  # Last 10 campaigns
            hist_vector = [0.5] * self.model_config['hidden_dim']  # Mock historical embedding
            historical_features.append(hist_vector)
        
        # Extract EPYC resource state
        resource_features = [
            system_state.get('numa_0_utilization', 0.5),
            system_state.get('numa_1_utilization', 0.5),
            # CCX utilization (8 CCX complexes)
            *[system_state.get(f'ccx_{i}_utilization', 0.5) for i in range(8)],
            # Thermal metrics
            system_state.get('thermal_avg', 50.0) / 100.0,
            system_state.get('thermal_max', 60.0) / 100.0,
            system_state.get('thermal_throttling', 0.0),
            system_state.get('power_efficiency', 0.8),
            # Memory metrics  
            system_state.get('memory_utilization', 0.5),
            system_state.get('memory_bandwidth', 0.6),
            system_state.get('cache_hit_ratio', 0.8),
            system_state.get('memory_pressure', 0.3)
        ]
        
        # Convert to tensors
        target_tensor = torch.tensor(target_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        agent_tensor = torch.tensor(agent_capabilities, dtype=torch.float32, device=self.device).unsqueeze(0)
        historical_tensor = torch.tensor(historical_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        resource_tensor = torch.tensor(resource_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Create attention masks
        target_mask = torch.zeros(1, len(target_features), dtype=torch.bool, device=self.device)
        agent_mask = torch.zeros(1, len(agent_capabilities), dtype=torch.bool, device=self.device)
        sequence_mask = torch.zeros(1, len(historical_features), dtype=torch.bool, device=self.device)
        
        return CampaignContext(
            target_complexity_vector=target_tensor,
            agent_capability_matrix=agent_tensor,
            historical_embeddings=historical_tensor,
            temporal_features=torch.tensor([0.5], device=self.device).unsqueeze(0),  # Mock
            resource_state=resource_tensor,
            target_mask=target_mask,
            agent_mask=agent_mask,
            sequence_mask=sequence_mask
        )
    
    async def _probs_to_agent_selection(self,
                                      action_probs: torch.Tensor,
                                      available_agents: List[Dict[str, Any]],
                                      uncertainty: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert action probabilities to concrete agent selections"""
        
        # Get top-k agents based on probabilities
        probs_np = action_probs.cpu().numpy()[0]  # Remove batch dimension
        uncertainty_val = uncertainty.cpu().item()
        
        # Adjust selection strategy based on uncertainty
        if uncertainty_val > self.training_config.uncertainty_threshold:
            # High uncertainty - use exploration
            top_k = min(5, len(available_agents))
            selected_indices = np.random.choice(
                len(probs_np), size=top_k, replace=False, p=probs_np / probs_np.sum()
            )
        else:
            # Low uncertainty - use exploitation
            top_k = min(3, len(available_agents))
            selected_indices = np.argsort(probs_np)[-top_k:]
        
        # Convert indices to agent selections
        selected_agents = []
        for idx in selected_indices:
            if idx < len(available_agents):
                agent = available_agents[idx].copy()
                agent['selection_probability'] = float(probs_np[idx])
                agent['selection_uncertainty'] = uncertainty_val
                agent['selection_method'] = 'transformer_policy'
                selected_agents.append(agent)
        
        return selected_agents
    
    async def train_on_experience(self, rewards: List[float], dones: List[bool]) -> Dict[str, float]:
        """Train the transformer using policy gradient methods"""
        
        if len(self.experience_buffer) < self.training_config.batch_size:
            return {}
        
        self.model.train()
        
        # Sample batch from experience buffer
        batch_size = min(self.training_config.batch_size, len(self.experience_buffer))
        batch_experiences = self.experience_buffer[-batch_size:]
        
        # Calculate returns and advantages using GAE
        returns, advantages = self._calculate_gae(rewards, dones)
        
        # Prepare batch tensors
        policy_losses = []
        value_losses = []
        entropy_losses = []
        uncertainty_losses = []
        
        for i, (experience, return_val, advantage) in enumerate(zip(batch_experiences, returns, advantages)):
            # Forward pass
            action_probs, state_value, uncertainty = self.model(experience['context'])
            
            # Policy loss (PPO-style)
            log_probs = torch.log(action_probs + 1e-8)
            old_log_probs = torch.log(experience['action_probs'] + 1e-8)
            
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)  # PPO clip
            
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_value.squeeze(), torch.tensor(return_val, device=self.device))
            
            # Entropy loss (for exploration)
            entropy = -(action_probs * log_probs).sum()
            entropy_loss = -self.training_config.entropy_coefficient * entropy
            
            # Uncertainty loss (encourage calibrated uncertainty)
            prediction_error = abs(return_val - state_value.item())
            uncertainty_target = min(1.0, prediction_error / 100.0)  # Normalize error
            uncertainty_loss = F.mse_loss(uncertainty.squeeze(), 
                                        torch.tensor(uncertainty_target, device=self.device))
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)
            uncertainty_losses.append(uncertainty_loss)
        
        # Combine losses
        total_policy_loss = torch.stack(policy_losses).mean()
        total_value_loss = torch.stack(value_losses).mean()
        total_entropy_loss = torch.stack(entropy_losses).mean()
        total_uncertainty_loss = torch.stack(uncertainty_losses).mean()
        
        total_loss = (total_policy_loss + 
                     self.training_config.value_coefficient * total_value_loss +
                     total_entropy_loss +
                     self.training_config.uncertainty_weight * total_uncertainty_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        metrics = {
            'policy_loss': total_policy_loss.item(),
            'value_loss': total_value_loss.item(), 
            'entropy_loss': total_entropy_loss.item(),
            'uncertainty_loss': total_uncertainty_loss.item(),
            'total_loss': total_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
        
        self.training_step += 1
        
        return metrics
    
    def _calculate_gae(self, rewards: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Calculate Generalized Advantage Estimation"""
        
        returns = []
        advantages = []
        
        # Calculate returns (Monte Carlo)
        running_return = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + self.training_config.gamma * running_return
            returns.insert(0, running_return)
        
        # Calculate advantages using GAE
        # For simplicity, using simplified advantage calculation
        # In production, would use value function predictions
        mean_return = np.mean(returns) if returns else 0
        advantages = [r - mean_return for r in returns]
        
        return returns, advantages
    
    async def save_model(self, path: str):
        """Save transformer model and training state"""
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_step': self.training_step,
            'epoch': self.epoch,
            'training_metrics': self.training_metrics
        }
        
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
    
    async def load_model(self, path: str):
        """Load transformer model and training state"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.training_metrics = checkpoint.get('training_metrics', {})
        
        self.logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    async def main():
        # Example usage
        orchestrator = TransformerOrchestrator()
        
        # Mock campaign context
        campaign_context = {
            'targets': [
                {'ports': [80, 443], 'services': ['http', 'https'], 
                 'complexity_score': 0.7, 'security_score': 0.6}
            ]
        }
        
        # Mock available agents
        available_agents = [
            {'name': 'recon_agent', 'resource_requirement': 0.3, 'success_rate': 0.8},
            {'name': 'web_crawler', 'resource_requirement': 0.4, 'success_rate': 0.7},
            {'name': 'vulnerability_scanner', 'resource_requirement': 0.6, 'success_rate': 0.9}
        ]
        
        # Mock system state
        system_state = {
            'numa_0_utilization': 0.4,
            'numa_1_utilization': 0.3,
            'thermal_avg': 65.0,
            'memory_utilization': 0.5
        }
        
        # Test agent selection
        selected_agents = await orchestrator.select_agents(
            campaign_context, available_agents, system_state
        )
        
        print(f"Selected {len(selected_agents)} agents:")
        for agent in selected_agents:
            print(f"  {agent['name']}: prob={agent.get('selection_probability', 0):.3f}")
        
        # Test training
        rewards = [10.0, 15.0, 8.0]  # Mock rewards
        dones = [False, False, True]  # Mock episode endings
        
        if len(orchestrator.experience_buffer) > 0:
            metrics = await orchestrator.train_on_experience(rewards, dones)
            print(f"Training metrics: {metrics}")
    
    asyncio.run(main())