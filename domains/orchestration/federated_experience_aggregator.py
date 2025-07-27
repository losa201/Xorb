#!/usr/bin/env python3
"""
Federated Experience Aggregation for Multi-Node RL

This module implements federated learning capabilities for aggregating
reinforcement learning experiences across multiple Xorb instances,
enabling shared learning while maintaining data locality and privacy.
"""

import asyncio
import logging
import hashlib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os


@dataclass
class ExperienceEntry:
    """Single experience entry for federated learning"""
    experience_id: str
    node_id: str
    campaign_id: str
    state_vector: np.ndarray
    action_taken: int
    reward_received: float
    next_state_vector: np.ndarray
    done: bool
    timestamp: datetime
    
    # Privacy-preserving metadata
    encrypted_context: Optional[bytes] = None
    privacy_level: str = "normal"  # "normal", "sensitive", "confidential"
    
    # Aggregation metadata
    confidence_score: float = 1.0
    node_reputation: float = 1.0
    experience_quality: float = 1.0


@dataclass
class NodeProfile:
    """Profile of a participating federated node"""
    node_id: str
    node_type: str  # "primary", "secondary", "observer"
    reputation_score: float = 1.0
    total_experiences: int = 0
    successful_campaigns: int = 0
    avg_reward: float = 0.0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    # Resource capabilities
    epyc_cores: int = 64
    numa_nodes: int = 2
    max_concurrent_campaigns: int = 10
    
    # Network configuration
    endpoint_url: str = ""
    api_key_hash: str = ""
    
    # Trust metrics
    data_quality_score: float = 1.0
    communication_reliability: float = 1.0
    security_compliance: float = 1.0


@dataclass
class AggregationRound:
    """Information about a federated aggregation round"""
    round_id: str
    start_time: datetime
    participating_nodes: List[str]
    experiences_collected: int
    convergence_metrics: Dict[str, float]
    model_updates: Dict[str, torch.Tensor]
    round_duration_seconds: float = 0.0
    
    # Consensus metadata
    consensus_reached: bool = False
    consensus_threshold: float = 0.8
    byzantine_nodes_detected: List[str] = field(default_factory=list)


class PrivacyPreservingEncoder:
    """Privacy-preserving encoding for sensitive experience data"""
    
    def __init__(self, password: str = None):
        if password is None:
            password = os.environ.get('XORB_FEDERATION_KEY', 'default_key_change_in_production')
        
        # Derive encryption key from password
        password_bytes = password.encode()
        salt = b'xorb_federation_salt'  # In production, use random salt per node
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher = Fernet(key)
    
    def encrypt_experience(self, experience: ExperienceEntry) -> bytes:
        """Encrypt sensitive parts of experience"""
        sensitive_data = {
            'state_vector': experience.state_vector.tolist(),
            'next_state_vector': experience.next_state_vector.tolist(),
            'campaign_id': experience.campaign_id,
            'node_id': experience.node_id
        }
        
        serialized = json.dumps(sensitive_data).encode()
        return self.cipher.encrypt(serialized)
    
    def decrypt_experience(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt experience data"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted.decode())
    
    def hash_identifier(self, identifier: str) -> str:
        """Create privacy-preserving hash of identifier"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


class ByzantineDetector:
    """Detector for byzantine/malicious nodes in federation"""
    
    def __init__(self, detection_threshold: float = 0.3):
        self.detection_threshold = detection_threshold
        self.node_behavior_history: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def analyze_node_contributions(self, 
                                 round_contributions: Dict[str, List[ExperienceEntry]]) -> List[str]:
        """Detect potentially byzantine nodes based on their contributions"""
        
        suspicious_nodes = []
        
        # Calculate reward statistics across all nodes
        all_rewards = []
        node_reward_stats = {}
        
        for node_id, experiences in round_contributions.items():
            rewards = [exp.reward_received for exp in experiences]
            if rewards:
                node_reward_stats[node_id] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'count': len(rewards)
                }
                all_rewards.extend(rewards)
        
        if not all_rewards:
            return suspicious_nodes
        
        global_mean = np.mean(all_rewards)
        global_std = np.std(all_rewards)
        
        # Detect nodes with abnormal reward patterns
        for node_id, stats in node_reward_stats.items():
            # Check for significantly different mean rewards
            if abs(stats['mean'] - global_mean) > 2 * global_std:
                suspicious_score = abs(stats['mean'] - global_mean) / (global_std + 1e-6)
                
                # Additional checks
                if self._check_temporal_anomalies(node_id, round_contributions[node_id]):
                    suspicious_score += 0.2
                
                if self._check_state_space_anomalies(round_contributions[node_id]):
                    suspicious_score += 0.15
                
                if suspicious_score > self.detection_threshold:
                    suspicious_nodes.append(node_id)
                    self.logger.warning(f"Detected suspicious node: {node_id} (score: {suspicious_score:.3f})")
        
        return suspicious_nodes
    
    def _check_temporal_anomalies(self, node_id: str, experiences: List[ExperienceEntry]) -> bool:
        """Check for temporal anomalies in experience patterns"""
        
        if len(experiences) < 5:
            return False
        
        # Check for identical timestamps (indicating batch generation)
        timestamps = [exp.timestamp for exp in experiences]
        unique_timestamps = set(timestamps)
        
        if len(unique_timestamps) < len(timestamps) * 0.5:  # More than 50% duplicate timestamps
            return True
        
        # Check for unrealistic temporal patterns
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs and np.std(time_diffs) < 0.1:  # Suspiciously regular timing
            return True
        
        return False
    
    def _check_state_space_anomalies(self, experiences: List[ExperienceEntry]) -> bool:
        """Check for anomalies in state space representation"""
        
        if len(experiences) < 10:
            return False
        
        # Check for identical or suspiciously similar state vectors
        state_vectors = [exp.state_vector for exp in experiences]
        
        # Calculate pairwise similarities
        identical_count = 0
        total_pairs = 0
        
        for i in range(len(state_vectors)):
            for j in range(i + 1, len(state_vectors)):
                total_pairs += 1
                similarity = np.corrcoef(state_vectors[i], state_vectors[j])[0, 1]
                if not np.isnan(similarity) and similarity > 0.95:
                    identical_count += 1
        
        if total_pairs > 0 and identical_count / total_pairs > 0.3:  # More than 30% highly similar
            return True
        
        return False


class FederatedExperienceAggregator:
    """
    Main federated learning coordinator for Xorb experience aggregation
    """
    
    def __init__(self,
                 node_id: str,
                 federation_config: Dict[str, Any] = None,
                 privacy_level: str = "normal"):
        
        self.node_id = node_id
        self.privacy_level = privacy_level
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        default_config = {
            'aggregation_interval_minutes': 30,
            'min_participants': 3,
            'max_participants': 10,
            'convergence_threshold': 0.95,
            'max_round_duration_minutes': 15,
            'experience_batch_size': 100,
            'reputation_decay_factor': 0.99,
            'quality_threshold': 0.7
        }
        self.config = {**default_config, **(federation_config or {})}
        
        # Core components
        self.privacy_encoder = PrivacyPreservingEncoder()
        self.byzantine_detector = ByzantineDetector()
        
        # Federation state
        self.participating_nodes: Dict[str, NodeProfile] = {}
        self.local_experience_buffer: List[ExperienceEntry] = []
        self.aggregated_experiences: List[ExperienceEntry] = []
        self.aggregation_history: List[AggregationRound] = []
        
        # Current round state
        self.current_round: Optional[AggregationRound] = None
        self.round_contributions: Dict[str, List[ExperienceEntry]] = {}
        
        # Network communication
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Performance metrics
        self.federation_metrics = {
            'total_rounds_completed': 0,
            'total_experiences_aggregated': 0,
            'average_participants_per_round': 0.0,
            'byzantine_detections': 0,
            'convergence_failures': 0,
            'avg_round_duration_minutes': 0.0,
            'data_quality_score': 1.0,
            'network_efficiency': 1.0
        }
        
        # Start background tasks
        self._aggregation_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info(f"Federated aggregator initialized for node {node_id}")
    
    async def start_federation(self):
        """Start federated learning coordination"""
        
        if self._running:
            return
        
        self._running = True
        
        # Initialize HTTP session
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Start background aggregation task
        self._aggregation_task = asyncio.create_task(self._federation_loop())
        
        self.logger.info("Federated learning started")
    
    async def stop_federation(self):
        """Stop federated learning coordination"""
        
        if not self._running:
            return
        
        self._running = False
        
        # Cancel aggregation task
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        self.logger.info("Federated learning stopped")
    
    async def add_experience(self, 
                           campaign_id: str,
                           state_vector: np.ndarray,
                           action_taken: int,
                           reward_received: float,
                           next_state_vector: np.ndarray,
                           done: bool,
                           confidence_score: float = 1.0):
        """Add local experience to federation buffer"""
        
        experience_id = hashlib.md5(
            f"{self.node_id}_{campaign_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        experience = ExperienceEntry(
            experience_id=experience_id,
            node_id=self.node_id,
            campaign_id=campaign_id,
            state_vector=state_vector,
            action_taken=action_taken,
            reward_received=reward_received,
            next_state_vector=next_state_vector,
            done=done,
            timestamp=datetime.utcnow(),
            confidence_score=confidence_score,
            experience_quality=self._assess_experience_quality(state_vector, reward_received)
        )
        
        # Encrypt sensitive data if required
        if self.privacy_level in ["sensitive", "confidential"]:
            experience.encrypted_context = self.privacy_encoder.encrypt_experience(experience)
        
        self.local_experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.local_experience_buffer) > 1000:
            self.local_experience_buffer = self.local_experience_buffer[-1000:]
        
        self.logger.debug(f"Added experience {experience_id} to local buffer")
    
    def _assess_experience_quality(self, state_vector: np.ndarray, reward: float) -> float:
        """Assess the quality of an experience entry"""
        
        quality_score = 1.0
        
        # Check for reasonable state vector values
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            quality_score *= 0.1  # Very low quality for invalid states
        
        # Check for reasonable reward values
        if abs(reward) > 1000:  # Suspiciously high rewards
            quality_score *= 0.5
        
        # Check for state vector diversity
        if np.std(state_vector) < 0.01:  # Very low variance
            quality_score *= 0.7
        
        return max(0.1, min(1.0, quality_score))
    
    async def register_node(self, node_profile: NodeProfile):
        """Register a new node in the federation"""
        
        # Validate node profile
        if not self._validate_node_profile(node_profile):
            self.logger.warning(f"Invalid node profile for {node_profile.node_id}")
            return False
        
        self.participating_nodes[node_profile.node_id] = node_profile
        
        self.logger.info(f"Registered node {node_profile.node_id} in federation")
        return True
    
    def _validate_node_profile(self, profile: NodeProfile) -> bool:
        """Validate node profile for security"""
        
        # Basic validation
        if not profile.node_id or not profile.endpoint_url:
            return False
        
        # Check reputation bounds
        if not (0.0 <= profile.reputation_score <= 1.0):
            return False
        
        # Additional security checks would go here
        # (certificate validation, API key verification, etc.)
        
        return True
    
    async def _federation_loop(self):
        """Main federation coordination loop"""
        
        while self._running:
            try:
                # Wait for aggregation interval
                await asyncio.sleep(self.config['aggregation_interval_minutes'] * 60)
                
                # Check if we have enough participants and experiences
                if (len(self.participating_nodes) >= self.config['min_participants'] and
                    len(self.local_experience_buffer) >= 10):
                    
                    await self._run_aggregation_round()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in federation loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _run_aggregation_round(self):
        """Execute a single aggregation round"""
        
        round_id = hashlib.md5(f"{self.node_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting aggregation round {round_id}")
        
        # Initialize round
        self.current_round = AggregationRound(
            round_id=round_id,
            start_time=start_time,
            participating_nodes=[],
            experiences_collected=0,
            convergence_metrics={},
            model_updates={}
        )
        
        try:
            # Phase 1: Collect experiences from participants
            await self._collect_experiences_phase()
            
            # Phase 2: Byzantine detection
            await self._byzantine_detection_phase()
            
            # Phase 3: Experience aggregation
            await self._experience_aggregation_phase()
            
            # Phase 4: Model update computation
            await self._model_update_phase()
            
            # Phase 5: Consensus and distribution
            await self._consensus_distribution_phase()
            
            # Mark round as completed
            self.current_round.round_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
            self.aggregation_history.append(self.current_round)
            
            # Update federation metrics
            self._update_federation_metrics()
            
            self.logger.info(f"Completed aggregation round {round_id} with {len(self.current_round.participating_nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Error in aggregation round {round_id}: {e}")
            self.federation_metrics['convergence_failures'] += 1
        
        finally:
            self.current_round = None
            self.round_contributions.clear()
    
    async def _collect_experiences_phase(self):
        """Phase 1: Collect experiences from all participating nodes"""
        
        collection_tasks = []
        
        # Add local experiences
        local_experiences = self._select_experiences_for_sharing()
        self.round_contributions[self.node_id] = local_experiences
        self.current_round.participating_nodes.append(self.node_id)
        
        # Collect from remote nodes
        for node_id, node_profile in self.participating_nodes.items():
            if node_id != self.node_id:
                task = asyncio.create_task(
                    self._request_experiences_from_node(node_profile)
                )
                collection_tasks.append((node_id, task))
        
        # Wait for all collections with timeout
        for node_id, task in collection_tasks:
            try:
                experiences = await asyncio.wait_for(task, timeout=120)  # 2 minute timeout
                if experiences:
                    self.round_contributions[node_id] = experiences
                    self.current_round.participating_nodes.append(node_id)
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout collecting experiences from node {node_id}")
            except Exception as e:
                self.logger.error(f"Error collecting from node {node_id}: {e}")
        
        total_experiences = sum(len(exps) for exps in self.round_contributions.values())
        self.current_round.experiences_collected = total_experiences
        
        self.logger.debug(f"Collected {total_experiences} experiences from {len(self.round_contributions)} nodes")
    
    def _select_experiences_for_sharing(self) -> List[ExperienceEntry]:
        """Select high-quality experiences for sharing"""
        
        # Filter by quality threshold
        quality_experiences = [
            exp for exp in self.local_experience_buffer
            if exp.experience_quality >= self.config['quality_threshold']
        ]
        
        # Sort by quality and recency
        quality_experiences.sort(
            key=lambda x: (x.experience_quality, x.timestamp),
            reverse=True
        )
        
        # Limit batch size
        return quality_experiences[:self.config['experience_batch_size']]
    
    async def _request_experiences_from_node(self, node_profile: NodeProfile) -> List[ExperienceEntry]:
        """Request experiences from a remote node"""
        
        if not self.http_session:
            return []
        
        try:
            url = f"{node_profile.endpoint_url}/api/federation/experiences"
            headers = {
                'Authorization': f'Bearer {node_profile.api_key_hash}',
                'X-Node-ID': self.node_id,
                'X-Round-ID': self.current_round.round_id
            }
            
            async with self.http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Deserialize experiences
                    experiences = []
                    for exp_data in data.get('experiences', []):
                        experience = self._deserialize_experience(exp_data)
                        if experience:
                            experiences.append(experience)
                    
                    return experiences
                else:
                    self.logger.warning(f"Failed to get experiences from {node_profile.node_id}: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Network error requesting from {node_profile.node_id}: {e}")
        
        return []
    
    def _deserialize_experience(self, exp_data: Dict[str, Any]) -> Optional[ExperienceEntry]:
        """Deserialize experience data from network"""
        
        try:
            # Handle encrypted experiences
            if 'encrypted_context' in exp_data and exp_data['encrypted_context']:
                # Decrypt if we have the capability
                if self.privacy_level in ["sensitive", "confidential"]:
                    decrypted = self.privacy_encoder.decrypt_experience(
                        base64.b64decode(exp_data['encrypted_context'])
                    )
                    state_vector = np.array(decrypted['state_vector'])
                    next_state_vector = np.array(decrypted['next_state_vector'])
                else:
                    # Skip encrypted experiences if we can't decrypt
                    return None
            else:
                state_vector = np.array(exp_data['state_vector'])
                next_state_vector = np.array(exp_data['next_state_vector'])
            
            return ExperienceEntry(
                experience_id=exp_data['experience_id'],
                node_id=exp_data['node_id'],
                campaign_id=exp_data.get('campaign_id', 'unknown'),
                state_vector=state_vector,
                action_taken=exp_data['action_taken'],
                reward_received=exp_data['reward_received'],
                next_state_vector=next_state_vector,
                done=exp_data['done'],
                timestamp=datetime.fromisoformat(exp_data['timestamp']),
                confidence_score=exp_data.get('confidence_score', 1.0),
                experience_quality=exp_data.get('experience_quality', 1.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error deserializing experience: {e}")
            return None
    
    async def _byzantine_detection_phase(self):
        """Phase 2: Detect and filter out byzantine nodes"""
        
        suspicious_nodes = self.byzantine_detector.analyze_node_contributions(self.round_contributions)
        
        if suspicious_nodes:
            self.logger.warning(f"Detected {len(suspicious_nodes)} suspicious nodes: {suspicious_nodes}")
            
            # Remove suspicious nodes from round
            for node_id in suspicious_nodes:
                if node_id in self.round_contributions:
                    del self.round_contributions[node_id]
                if node_id in self.current_round.participating_nodes:
                    self.current_round.participating_nodes.remove(node_id)
                
                # Update node reputation
                if node_id in self.participating_nodes:
                    self.participating_nodes[node_id].reputation_score *= 0.7  # Penalty
            
            self.current_round.byzantine_nodes_detected = suspicious_nodes
            self.federation_metrics['byzantine_detections'] += len(suspicious_nodes)
    
    async def _experience_aggregation_phase(self):
        """Phase 3: Aggregate experiences with quality weighting"""
        
        all_experiences = []
        
        for node_id, experiences in self.round_contributions.items():
            node_reputation = self.participating_nodes.get(node_id, NodeProfile(node_id="unknown")).reputation_score
            
            # Weight experiences by node reputation and quality
            for exp in experiences:
                exp.node_reputation = node_reputation
                # Apply reputation weighting to experience quality
                exp.experience_quality *= node_reputation
                all_experiences.append(exp)
        
        # Sort by quality and select top experiences
        all_experiences.sort(key=lambda x: x.experience_quality, reverse=True)
        
        # Add to aggregated buffer
        max_aggregated = 500  # Limit aggregated buffer size
        self.aggregated_experiences.extend(all_experiences[:max_aggregated])
        
        # Maintain buffer size
        if len(self.aggregated_experiences) > 2000:
            self.aggregated_experiences = self.aggregated_experiences[-2000:]
        
        self.logger.debug(f"Aggregated {len(all_experiences)} experiences")
    
    async def _model_update_phase(self):
        """Phase 4: Compute model updates from aggregated experiences"""
        
        if not self.aggregated_experiences:
            return
        
        # Extract state-action-reward sequences
        states = np.array([exp.state_vector for exp in self.aggregated_experiences[-100:]])
        actions = np.array([exp.action_taken for exp in self.aggregated_experiences[-100:]])
        rewards = np.array([exp.reward_received for exp in self.aggregated_experiences[-100:]])
        
        # Compute simple statistics for model updates
        # In production, this would involve actual model training
        
        self.current_round.model_updates = {
            'mean_state': torch.tensor(np.mean(states, axis=0), dtype=torch.float32),
            'std_state': torch.tensor(np.std(states, axis=0), dtype=torch.float32),
            'action_distribution': torch.tensor(np.bincount(actions, minlength=10), dtype=torch.float32),
            'reward_statistics': torch.tensor([np.mean(rewards), np.std(rewards)], dtype=torch.float32)
        }
        
        # Calculate convergence metrics
        self.current_round.convergence_metrics = {
            'reward_stability': float(np.std(rewards[-20:]) < 0.1 if len(rewards) >= 20 else False),
            'state_coverage': float(np.mean(np.std(states, axis=0))),
            'action_entropy': float(-np.sum(np.bincount(actions) / len(actions) * 
                                           np.log(np.bincount(actions) / len(actions) + 1e-8)))
        }
    
    async def _consensus_distribution_phase(self):
        """Phase 5: Reach consensus and distribute updates"""
        
        # Simple consensus: majority of nodes must agree on convergence
        convergence_votes = []
        
        for node_id in self.current_round.participating_nodes:
            # In production, would query nodes for convergence opinion
            # For now, use local convergence metrics
            convergence_score = np.mean(list(self.current_round.convergence_metrics.values()))
            convergence_votes.append(convergence_score > self.config['convergence_threshold'])
        
        # Check consensus
        if sum(convergence_votes) / len(convergence_votes) >= self.current_round.consensus_threshold:
            self.current_round.consensus_reached = True
            
            # Distribute updates to participating nodes
            await self._distribute_model_updates()
        else:
            self.logger.info(f"Consensus not reached in round {self.current_round.round_id}")
    
    async def _distribute_model_updates(self):
        """Distribute model updates to participating nodes"""
        
        update_tasks = []
        
        for node_id in self.current_round.participating_nodes:
            if node_id != self.node_id and node_id in self.participating_nodes:
                task = asyncio.create_task(
                    self._send_model_update(self.participating_nodes[node_id])
                )
                update_tasks.append(task)
        
        # Wait for all updates with timeout
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        successful_updates = sum(1 for r in results if not isinstance(r, Exception))
        self.logger.debug(f"Successfully distributed updates to {successful_updates} nodes")
    
    async def _send_model_update(self, node_profile: NodeProfile) -> bool:
        """Send model update to a specific node"""
        
        if not self.http_session or not self.current_round:
            return False
        
        try:
            url = f"{node_profile.endpoint_url}/api/federation/model_update"
            headers = {
                'Authorization': f'Bearer {node_profile.api_key_hash}',
                'X-Node-ID': self.node_id,
                'X-Round-ID': self.current_round.round_id
            }
            
            # Serialize model updates
            update_data = {
                'round_id': self.current_round.round_id,
                'convergence_metrics': self.current_round.convergence_metrics,
                'model_updates': {
                    key: tensor.tolist() for key, tensor in self.current_round.model_updates.items()
                }
            }
            
            async with self.http_session.post(url, headers=headers, json=update_data) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Error sending update to {node_profile.node_id}: {e}")
            return False
    
    def _update_federation_metrics(self):
        """Update federation performance metrics"""
        
        if not self.current_round:
            return
        
        self.federation_metrics['total_rounds_completed'] += 1
        self.federation_metrics['total_experiences_aggregated'] += self.current_round.experiences_collected
        
        # Update running averages
        rounds = self.federation_metrics['total_rounds_completed']
        
        self.federation_metrics['average_participants_per_round'] = (
            (self.federation_metrics['average_participants_per_round'] * (rounds - 1) + 
             len(self.current_round.participating_nodes)) / rounds
        )
        
        self.federation_metrics['avg_round_duration_minutes'] = (
            (self.federation_metrics['avg_round_duration_minutes'] * (rounds - 1) + 
             self.current_round.round_duration_seconds / 60.0) / rounds
        )
        
        # Update data quality score
        if self.aggregated_experiences:
            recent_quality = np.mean([exp.experience_quality for exp in self.aggregated_experiences[-100:]])
            self.federation_metrics['data_quality_score'] = recent_quality
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status and metrics"""
        
        return {
            'node_id': self.node_id,
            'federation_active': self._running,
            'participating_nodes': len(self.participating_nodes),
            'local_experiences': len(self.local_experience_buffer),
            'aggregated_experiences': len(self.aggregated_experiences),
            'current_round': {
                'round_id': self.current_round.round_id if self.current_round else None,
                'participants': len(self.current_round.participating_nodes) if self.current_round else 0,
                'experiences_collected': self.current_round.experiences_collected if self.current_round else 0
            },
            'metrics': self.federation_metrics.copy(),
            'node_reputations': {
                node_id: profile.reputation_score 
                for node_id, profile in self.participating_nodes.items()
            }
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        aggregator = FederatedExperienceAggregator(
            node_id="xorb_node_001",
            privacy_level="normal"
        )
        
        # Add some mock experiences
        for i in range(20):
            await aggregator.add_experience(
                campaign_id=f"campaign_{i//5}",
                state_vector=np.random.randn(10),
                action_taken=i % 5,
                reward_received=np.random.normal(10, 2),
                next_state_vector=np.random.randn(10),
                done=(i % 5 == 4)
            )
        
        print(f"Added {len(aggregator.local_experience_buffer)} experiences")
        
        # Get status
        status = await aggregator.get_federation_status()
        print(f"Federation status: {json.dumps(status, indent=2, default=str)}")
    
    asyncio.run(main())