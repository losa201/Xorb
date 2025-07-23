#!/usr/bin/env python3
"""
Temporal Memory Systems for Xorb 2.0

This module implements sophisticated temporal memory capabilities that enable agents
to maintain memory across campaigns, detect temporal patterns, adapt to environmental
drift, and preserve critical knowledge over extended periods.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import pickle
import hashlib
from pathlib import Path


class MemoryType(Enum):
    """Types of temporal memories"""
    EPISODIC = "episodic"           # Specific events and experiences
    SEMANTIC = "semantic"           # General knowledge and patterns
    PROCEDURAL = "procedural"       # Skills and learned procedures
    WORKING = "working"             # Short-term active memory
    PROSPECTIVE = "prospective"     # Future intentions and plans


class MemoryImportance(Enum):
    """Importance levels for memory consolidation"""
    CRITICAL = "critical"           # Must be preserved
    HIGH = "high"                   # Important for performance
    MEDIUM = "medium"               # Useful but not essential
    LOW = "low"                     # Can be forgotten if needed
    TRANSIENT = "transient"         # Temporary, can be quickly forgotten


@dataclass
class TemporalMemoryItem:
    """Individual temporal memory item"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[np.ndarray]
    
    # Temporal attributes
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    # Importance and relevance
    importance: MemoryImportance = MemoryImportance.MEDIUM
    relevance_score: float = 1.0
    decay_rate: float = 0.01
    
    # Contextual information
    campaign_context: Dict[str, Any] = field(default_factory=dict)
    environmental_state: Dict[str, Any] = field(default_factory=dict)
    associated_agents: List[str] = field(default_factory=list)
    
    # Consolidation metadata
    consolidation_strength: float = 1.0
    interference_count: int = 0
    rehearsal_history: List[datetime] = field(default_factory=list)
    
    # Temporal pattern information
    temporal_tags: List[str] = field(default_factory=list)
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    drift_indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryConsolidationEvent:
    """Event representing memory consolidation"""
    event_id: str
    memories_consolidated: List[str]  # Memory IDs
    consolidation_type: str  # 'interference', 'rehearsal', 'sleep_like'
    timestamp: datetime
    
    # Consolidation results
    memories_strengthened: List[str]
    memories_weakened: List[str]
    memories_forgotten: List[str]
    new_connections_formed: List[Tuple[str, str, float]]
    
    # Performance impact
    memory_efficiency_change: float
    retrieval_accuracy_change: float
    capacity_utilization: float


class TemporalEmbeddingNetwork(nn.Module):
    """Neural network for generating temporal embeddings"""
    
    def __init__(self,
                 content_dim: int = 256,
                 temporal_dim: int = 64,
                 context_dim: int = 128,
                 embedding_dim: int = 512):
        super().__init__()
        
        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, len(MemoryImportance)),
            nn.Softmax(dim=-1)
        )
        
        # Decay rate predictor
        self.decay_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                content_features: torch.Tensor,
                temporal_features: torch.Tensor,
                context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Encode different feature types
        content_encoded = self.content_encoder(content_features)
        temporal_encoded = self.temporal_encoder(temporal_features)
        context_encoded = self.context_encoder(context_features)
        
        # Concatenate and fuse
        combined = torch.cat([content_encoded, temporal_encoded, context_encoded], dim=-1)
        embedding = self.fusion_layer(combined)
        
        # Predict attributes
        importance_probs = self.importance_predictor(embedding)
        decay_rate = self.decay_predictor(embedding)
        
        return {
            'embedding': embedding,
            'importance_probabilities': importance_probs,
            'predicted_decay_rate': decay_rate
        }


class MemoryConsolidationEngine:
    """Engine for memory consolidation and forgetting"""
    
    def __init__(self, 
                 consolidation_threshold: float = 0.8,
                 interference_threshold: float = 0.9,
                 forgetting_threshold: float = 0.1):
        
        self.consolidation_threshold = consolidation_threshold
        self.interference_threshold = interference_threshold
        self.forgetting_threshold = forgetting_threshold
        
        self.logger = logging.getLogger(__name__)
        self.consolidation_history: List[MemoryConsolidationEvent] = []
    
    async def consolidate_memories(self, 
                                 memories: List[TemporalMemoryItem],
                                 consolidation_type: str = 'rehearsal') -> MemoryConsolidationEvent:
        """Perform memory consolidation"""
        
        event_id = f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Calculate memory similarities for interference detection
        similarity_matrix = self._calculate_similarity_matrix(memories)
        
        # Identify memories for different treatments
        memories_to_strengthen = []
        memories_to_weaken = []
        memories_to_forget = []
        new_connections = []
        
        for i, memory in enumerate(memories):
            # Update consolidation strength based on access pattern
            time_since_access = (datetime.utcnow() - memory.last_accessed).total_seconds()
            recency_factor = np.exp(-time_since_access / 86400)  # Decay over days
            
            # Calculate interference from similar memories
            interference_score = self._calculate_interference(i, similarity_matrix, memories)
            
            # Decide on consolidation action
            if memory.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
                # Critical memories are always strengthened
                if memory.consolidation_strength < self.consolidation_threshold:
                    memory.consolidation_strength = min(1.0, memory.consolidation_strength + 0.2)
                    memories_to_strengthen.append(memory.memory_id)
                
            elif interference_score > self.interference_threshold:
                # High interference - weaken or forget
                if memory.consolidation_strength > self.forgetting_threshold:
                    memory.consolidation_strength *= 0.8
                    memory.interference_count += 1
                    memories_to_weaken.append(memory.memory_id)
                else:
                    memories_to_forget.append(memory.memory_id)
                
            elif recency_factor > 0.5 and memory.access_count > 2:
                # Recently accessed and frequently used - strengthen
                memory.consolidation_strength = min(1.0, memory.consolidation_strength + 0.1)
                memories_to_strengthen.append(memory.memory_id)
            
            # Find new connections based on similarity
            for j, other_memory in enumerate(memories):
                if i < j and similarity_matrix[i][j] > 0.7:
                    connection_strength = similarity_matrix[i][j] * min(memory.consolidation_strength,
                                                                       other_memory.consolidation_strength)
                    new_connections.append((memory.memory_id, other_memory.memory_id, connection_strength))
        
        # Create consolidation event
        consolidation_event = MemoryConsolidationEvent(
            event_id=event_id,
            memories_consolidated=[m.memory_id for m in memories],
            consolidation_type=consolidation_type,
            timestamp=datetime.utcnow(),
            memories_strengthened=memories_to_strengthen,
            memories_weakened=memories_to_weaken,
            memories_forgotten=memories_to_forget,
            new_connections_formed=new_connections,
            memory_efficiency_change=self._calculate_efficiency_change(memories),
            retrieval_accuracy_change=0.0,  # Would be calculated from performance metrics
            capacity_utilization=len(memories) / 10000.0  # Assuming max capacity of 10k
        )
        
        self.consolidation_history.append(consolidation_event)
        
        self.logger.info(f"Memory consolidation completed: {len(memories_to_strengthen)} strengthened, "
                        f"{len(memories_to_weaken)} weakened, {len(memories_to_forget)} forgotten")
        
        return consolidation_event
    
    def _calculate_similarity_matrix(self, memories: List[TemporalMemoryItem]) -> np.ndarray:
        """Calculate similarity matrix between memories"""
        
        n = len(memories)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_memory_similarity(memories[i], memories[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _calculate_memory_similarity(self, 
                                   memory1: TemporalMemoryItem,
                                   memory2: TemporalMemoryItem) -> float:
        """Calculate similarity between two memories"""
        
        # Embedding similarity
        if memory1.embedding is not None and memory2.embedding is not None:
            embedding_sim = np.dot(memory1.embedding, memory2.embedding) / (
                np.linalg.norm(memory1.embedding) * np.linalg.norm(memory2.embedding) + 1e-8
            )
        else:
            embedding_sim = 0.0
        
        # Context similarity
        context_sim = self._calculate_context_similarity(
            memory1.campaign_context, memory2.campaign_context
        )
        
        # Temporal similarity
        time_diff = abs((memory1.created_at - memory2.created_at).total_seconds())
        temporal_sim = np.exp(-time_diff / (86400 * 7))  # Decay over weeks
        
        # Memory type similarity
        type_sim = 1.0 if memory1.memory_type == memory2.memory_type else 0.5
        
        # Weighted combination
        total_similarity = (
            0.4 * embedding_sim +
            0.3 * context_sim +
            0.2 * temporal_sim +
            0.1 * type_sim
        )
        
        return max(0.0, min(1.0, total_similarity))
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, similarity))
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_interference(self, 
                              memory_idx: int,
                              similarity_matrix: np.ndarray,
                              memories: List[TemporalMemoryItem]) -> float:
        """Calculate interference score for a memory"""
        
        interference = 0.0
        
        for j, other_memory in enumerate(memories):
            if j != memory_idx:
                similarity = similarity_matrix[memory_idx][j]
                
                # Interference is higher when similar memories have different outcomes
                if similarity > 0.7:
                    # Simplified interference calculation
                    strength_diff = abs(memories[memory_idx].consolidation_strength - 
                                      other_memory.consolidation_strength)
                    interference += similarity * strength_diff
        
        return min(1.0, interference)
    
    def _calculate_efficiency_change(self, memories: List[TemporalMemoryItem]) -> float:
        """Calculate change in memory efficiency"""
        
        # Simplified efficiency calculation based on consolidation strength distribution
        strengths = [m.consolidation_strength for m in memories]
        
        # Good distribution has high variance (some very strong, some weak)
        strength_variance = np.var(strengths)
        mean_strength = np.mean(strengths)
        
        # Efficiency improves with good strength distribution and reasonable mean
        efficiency = strength_variance * min(mean_strength, 1.0 - mean_strength)
        
        return efficiency


class TemporalPatternDetector:
    """Detector for temporal patterns in memory and behavior"""
    
    def __init__(self, pattern_window_hours: int = 168):  # 1 week default
        self.pattern_window_hours = pattern_window_hours
        self.detected_patterns: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def detect_temporal_patterns(self, 
                                     memories: List[TemporalMemoryItem]) -> Dict[str, Any]:
        """Detect temporal patterns in memories"""
        
        patterns = {
            'cyclical_patterns': await self._detect_cyclical_patterns(memories),
            'trend_patterns': await self._detect_trend_patterns(memories),
            'anomaly_patterns': await self._detect_anomaly_patterns(memories),
            'seasonal_patterns': await self._detect_seasonal_patterns(memories),
            'drift_patterns': await self._detect_drift_patterns(memories)
        }
        
        return patterns
    
    async def _detect_cyclical_patterns(self, memories: List[TemporalMemoryItem]) -> List[Dict[str, Any]]:
        """Detect cyclical patterns in memory creation and access"""
        
        cyclical_patterns = []
        
        # Group memories by hour of day
        hourly_counts = defaultdict(int)
        for memory in memories:
            hour = memory.created_at.hour
            hourly_counts[hour] += 1
        
        # Detect daily cycles
        hours = list(range(24))
        counts = [hourly_counts[h] for h in hours]
        
        if len(set(counts)) > 1:  # Has variation
            # Simple peak detection
            peaks = []
            for i in range(1, len(counts) - 1):
                if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
                    peaks.append((i, counts[i]))
            
            if peaks:
                cyclical_patterns.append({
                    'type': 'daily_cycle',
                    'period_hours': 24,
                    'peaks': peaks,
                    'strength': np.std(counts) / (np.mean(counts) + 1e-6),
                    'description': f'Daily activity peaks at hours: {[p[0] for p in peaks]}'
                })
        
        # Group memories by day of week
        weekly_counts = defaultdict(int)
        for memory in memories:
            day = memory.created_at.weekday()
            weekly_counts[day] += 1
        
        days = list(range(7))
        daily_counts = [weekly_counts[d] for d in days]
        
        if len(set(daily_counts)) > 1:
            weekly_strength = np.std(daily_counts) / (np.mean(daily_counts) + 1e-6)
            if weekly_strength > 0.3:  # Significant weekly pattern
                cyclical_patterns.append({
                    'type': 'weekly_cycle',
                    'period_hours': 168,
                    'pattern': daily_counts,
                    'strength': weekly_strength,
                    'description': f'Weekly activity pattern with peaks on days: {np.argsort(daily_counts)[-2:].tolist()}'
                })
        
        return cyclical_patterns
    
    async def _detect_trend_patterns(self, memories: List[TemporalMemoryItem]) -> List[Dict[str, Any]]:
        """Detect trending patterns in memory characteristics"""
        
        trend_patterns = []
        
        if len(memories) < 10:
            return trend_patterns
        
        # Sort memories by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        # Analyze trends in memory importance
        importance_scores = []
        for memory in sorted_memories:
            score = {
                MemoryImportance.CRITICAL: 5,
                MemoryImportance.HIGH: 4,
                MemoryImportance.MEDIUM: 3,
                MemoryImportance.LOW: 2,
                MemoryImportance.TRANSIENT: 1
            }.get(memory.importance, 3)
            importance_scores.append(score)
        
        # Calculate trend using linear regression
        x = np.arange(len(importance_scores))
        if len(set(importance_scores)) > 1:
            correlation = np.corrcoef(x, importance_scores)[0, 1]
            
            if abs(correlation) > 0.3:
                trend_patterns.append({
                    'type': 'importance_trend',
                    'correlation': correlation,
                    'direction': 'increasing' if correlation > 0 else 'decreasing',
                    'strength': abs(correlation),
                    'description': f'Memory importance is {"increasing" if correlation > 0 else "decreasing"} over time'
                })
        
        # Analyze trends in memory types
        type_distribution_over_time = []
        window_size = max(10, len(sorted_memories) // 5)
        
        for i in range(0, len(sorted_memories) - window_size + 1, window_size // 2):
            window_memories = sorted_memories[i:i + window_size]
            type_counts = defaultdict(int)
            
            for memory in window_memories:
                type_counts[memory.memory_type.value] += 1
            
            type_distribution_over_time.append(dict(type_counts))
        
        if len(type_distribution_over_time) >= 3:
            # Analyze shifts in memory type distribution
            all_types = set()
            for dist in type_distribution_over_time:
                all_types.update(dist.keys())
            
            for mem_type in all_types:
                proportions = [dist.get(mem_type, 0) / sum(dist.values()) 
                              for dist in type_distribution_over_time]
                
                if len(set(proportions)) > 1:
                    x_time = np.arange(len(proportions))
                    correlation = np.corrcoef(x_time, proportions)[0, 1]
                    
                    if abs(correlation) > 0.4:
                        trend_patterns.append({
                            'type': 'memory_type_trend',
                            'memory_type': mem_type,
                            'correlation': correlation,
                            'direction': 'increasing' if correlation > 0 else 'decreasing',
                            'strength': abs(correlation),
                            'description': f'{mem_type} memory usage is {"increasing" if correlation > 0 else "decreasing"}'
                        })
        
        return trend_patterns
    
    async def _detect_anomaly_patterns(self, memories: List[TemporalMemoryItem]) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in memory behavior"""
        
        anomaly_patterns = []
        
        if len(memories) < 20:
            return anomaly_patterns
        
        # Detect anomalous spikes in memory creation
        time_sorted = sorted(memories, key=lambda m: m.created_at)
        
        # Calculate inter-arrival times
        inter_arrivals = []
        for i in range(1, len(time_sorted)):
            diff = (time_sorted[i].created_at - time_sorted[i-1].created_at).total_seconds()
            inter_arrivals.append(diff)
        
        if inter_arrivals:
            mean_interval = np.mean(inter_arrivals)
            std_interval = np.std(inter_arrivals)
            
            # Find anomalously short intervals (bursts)
            burst_threshold = max(1.0, mean_interval - 2 * std_interval)
            bursts = [interval for interval in inter_arrivals if interval < burst_threshold]
            
            if len(bursts) > len(inter_arrivals) * 0.1:  # More than 10% are bursts
                anomaly_patterns.append({
                    'type': 'memory_creation_bursts',
                    'burst_count': len(bursts),
                    'burst_percentage': len(bursts) / len(inter_arrivals),
                    'average_burst_interval': np.mean(bursts),
                    'description': f'Detected {len(bursts)} memory creation bursts'
                })
        
        # Detect anomalous memory importance spikes
        importance_scores = []
        for memory in memories:
            score = {
                MemoryImportance.CRITICAL: 5,
                MemoryImportance.HIGH: 4,
                MemoryImportance.MEDIUM: 3,
                MemoryImportance.LOW: 2,
                MemoryImportance.TRANSIENT: 1
            }.get(memory.importance, 3)
            importance_scores.append(score)
        
        if importance_scores:
            mean_importance = np.mean(importance_scores)
            std_importance = np.std(importance_scores)
            
            anomalous_threshold = mean_importance + 2 * std_importance
            anomalous_memories = [m for m, score in zip(memories, importance_scores) 
                                 if score > anomalous_threshold]
            
            if len(anomalous_memories) > 0:
                anomaly_patterns.append({
                    'type': 'importance_anomalies',
                    'anomalous_count': len(anomalous_memories),
                    'percentage': len(anomalous_memories) / len(memories),
                    'threshold': anomalous_threshold,
                    'description': f'Detected {len(anomalous_memories)} memories with anomalously high importance'
                })
        
        return anomaly_patterns
    
    async def _detect_seasonal_patterns(self, memories: List[TemporalMemoryItem]) -> List[Dict[str, Any]]:
        """Detect seasonal patterns in memory characteristics"""
        
        seasonal_patterns = []
        
        # Group memories by month
        monthly_stats = defaultdict(lambda: {'count': 0, 'importance_sum': 0, 'types': defaultdict(int)})
        
        for memory in memories:
            month = memory.created_at.month
            monthly_stats[month]['count'] += 1
            
            importance_score = {
                MemoryImportance.CRITICAL: 5,
                MemoryImportance.HIGH: 4,
                MemoryImportance.MEDIUM: 3,
                MemoryImportance.LOW: 2,
                MemoryImportance.TRANSIENT: 1
            }.get(memory.importance, 3)
            monthly_stats[month]['importance_sum'] += importance_score
            monthly_stats[month]['types'][memory.memory_type.value] += 1
        
        # Analyze monthly patterns
        months = list(range(1, 13))
        monthly_counts = [monthly_stats[m]['count'] for m in months]
        monthly_avg_importance = [
            monthly_stats[m]['importance_sum'] / max(monthly_stats[m]['count'], 1)
            for m in months
        ]
        
        # Detect seasonal count patterns
        if len(set(monthly_counts)) > 1:
            count_variance = np.var(monthly_counts)
            count_mean = np.mean(monthly_counts)
            
            if count_variance > count_mean * 0.5:  # Significant seasonal variation
                peak_months = [i for i, count in enumerate(monthly_counts, 1) 
                              if count > count_mean + np.std(monthly_counts)]
                
                seasonal_patterns.append({
                    'type': 'seasonal_activity',
                    'pattern': 'monthly_variation',
                    'peak_months': peak_months,
                    'variation_coefficient': np.std(monthly_counts) / count_mean,
                    'description': f'Seasonal activity peaks in months: {peak_months}'
                })
        
        # Detect seasonal importance patterns
        if len(set(monthly_avg_importance)) > 1:
            importance_variance = np.var(monthly_avg_importance)
            
            if importance_variance > 0.5:
                high_importance_months = [i for i, importance in enumerate(monthly_avg_importance, 1)
                                        if importance > np.mean(monthly_avg_importance) + np.std(monthly_avg_importance)]
                
                seasonal_patterns.append({
                    'type': 'seasonal_importance',
                    'pattern': 'monthly_importance_variation',
                    'high_importance_months': high_importance_months,
                    'variation_coefficient': np.std(monthly_avg_importance) / np.mean(monthly_avg_importance),
                    'description': f'Higher importance memories in months: {high_importance_months}'
                })
        
        return seasonal_patterns
    
    async def _detect_drift_patterns(self, memories: List[TemporalMemoryItem]) -> List[Dict[str, Any]]:
        """Detect concept drift patterns in memory content"""
        
        drift_patterns = []
        
        if len(memories) < 50:
            return drift_patterns
        
        # Sort memories by creation time
        time_sorted = sorted(memories, key=lambda m: m.created_at)
        
        # Split into early and late periods
        split_point = len(time_sorted) // 2
        early_memories = time_sorted[:split_point]
        late_memories = time_sorted[split_point:]
        
        # Compare memory type distributions
        early_types = defaultdict(int)
        late_types = defaultdict(int)
        
        for memory in early_memories:
            early_types[memory.memory_type.value] += 1
        
        for memory in late_memories:
            late_types[memory.memory_type.value] += 1
        
        # Calculate distribution shift
        all_types = set(early_types.keys()) | set(late_types.keys())
        
        early_total = sum(early_types.values())
        late_total = sum(late_types.values())
        
        distribution_changes = {}
        for mem_type in all_types:
            early_prop = early_types[mem_type] / early_total
            late_prop = late_types[mem_type] / late_total
            change = late_prop - early_prop
            
            if abs(change) > 0.1:  # Significant change
                distribution_changes[mem_type] = change
        
        if distribution_changes:
            drift_patterns.append({
                'type': 'memory_type_drift',
                'changes': distribution_changes,
                'drift_magnitude': sum(abs(change) for change in distribution_changes.values()),
                'description': f'Memory type distribution has shifted significantly'
            })
        
        # Compare importance distributions
        early_importance = [m.importance.value for m in early_memories]
        late_importance = [m.importance.value for m in late_memories]
        
        importance_counts_early = defaultdict(int)
        importance_counts_late = defaultdict(int)
        
        for imp in early_importance:
            importance_counts_early[imp] += 1
        for imp in late_importance:
            importance_counts_late[imp] += 1
        
        # Calculate importance shift
        all_importance_levels = set(importance_counts_early.keys()) | set(importance_counts_late.keys())
        importance_shifts = {}
        
        for imp_level in all_importance_levels:
            early_prop = importance_counts_early[imp_level] / len(early_importance)
            late_prop = importance_counts_late[imp_level] / len(late_importance)
            shift = late_prop - early_prop
            
            if abs(shift) > 0.1:
                importance_shifts[imp_level] = shift
        
        if importance_shifts:
            drift_patterns.append({
                'type': 'importance_drift',
                'shifts': importance_shifts,
                'drift_magnitude': sum(abs(shift) for shift in importance_shifts.values()),
                'description': f'Memory importance patterns have drifted over time'
            })
        
        return drift_patterns


class TemporalMemorySystem:
    """Main temporal memory system orchestrating all memory functions"""
    
    def __init__(self,
                 max_memory_items: int = 50000,
                 consolidation_interval_hours: int = 24,
                 device: str = 'auto'):
        
        self.max_memory_items = max_memory_items
        self.consolidation_interval_hours = consolidation_interval_hours
        
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Core components
        self.embedding_network = TemporalEmbeddingNetwork().to(self.device)
        self.consolidation_engine = MemoryConsolidationEngine()
        self.pattern_detector = TemporalPatternDetector()
        
        # Memory storage
        self.memory_store: Dict[str, TemporalMemoryItem] = {}
        self.memory_index: Dict[MemoryType, List[str]] = {
            mem_type: [] for mem_type in MemoryType
        }
        
        # Temporal indices
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> memory_ids
        self.agent_index: Dict[str, List[str]] = defaultdict(list)    # agent_id -> memory_ids
        
        # Background tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.memory_metrics = {
            'total_memories': 0,
            'memories_by_type': {mem_type.value: 0 for mem_type in MemoryType},
            'consolidation_events': 0,
            'average_memory_age_days': 0.0,
            'memory_efficiency': 0.8,
            'retrieval_accuracy': 0.9,
            'pattern_detection_accuracy': 0.85
        }
        
        self.logger.info(f"Temporal memory system initialized on {self.device}")
    
    async def start_memory_system(self):
        """Start the temporal memory system"""
        
        if self._running:
            return
        
        self._running = True
        
        # Start background consolidation task
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        
        self.logger.info("Temporal memory system started")
    
    async def stop_memory_system(self):
        """Stop the temporal memory system"""
        
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Temporal memory system stopped")
    
    async def store_memory(self,
                         agent_id: str,
                         memory_type: MemoryType,
                         content: Dict[str, Any],
                         campaign_context: Dict[str, Any] = None,
                         importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """Store a new memory in the temporal memory system"""
        
        # Generate memory ID
        memory_id = f"{memory_type.value}_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Generate embedding
        embedding = await self._generate_memory_embedding(content, campaign_context or {})
        
        # Create memory item
        memory_item = TemporalMemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            importance=importance,
            campaign_context=campaign_context or {},
            environmental_state=await self._capture_environmental_state(),
            associated_agents=[agent_id]
        )
        
        # Store memory
        self.memory_store[memory_id] = memory_item
        
        # Update indices
        self.memory_index[memory_type].append(memory_id)
        
        date_key = memory_item.created_at.strftime('%Y-%m-%d')
        self.temporal_index[date_key].append(memory_id)
        
        self.agent_index[agent_id].append(memory_id)
        
        # Update metrics
        self.memory_metrics['total_memories'] += 1
        self.memory_metrics['memories_by_type'][memory_type.value] += 1
        
        # Check capacity and trigger cleanup if needed
        if len(self.memory_store) > self.max_memory_items:
            await self._cleanup_old_memories()
        
        self.logger.debug(f"Stored memory {memory_id} for agent {agent_id}")
        return memory_id
    
    async def retrieve_memories(self,
                              query: Dict[str, Any],
                              memory_types: List[MemoryType] = None,
                              time_range: Tuple[datetime, datetime] = None,
                              limit: int = 10) -> List[TemporalMemoryItem]:
        """Retrieve memories based on query criteria"""
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        candidate_memories = []
        
        # Collect candidates from relevant indices
        for memory_type in memory_types:
            candidate_memories.extend([
                self.memory_store[mid] for mid in self.memory_index[memory_type]
                if mid in self.memory_store
            ])
        
        # Apply time range filter
        if time_range:
            start_time, end_time = time_range
            candidate_memories = [
                memory for memory in candidate_memories
                if start_time <= memory.created_at <= end_time
            ]
        
        # Calculate relevance scores
        memory_scores = []
        query_embedding = await self._generate_query_embedding(query)
        
        for memory in candidate_memories:
            relevance_score = await self._calculate_memory_relevance(memory, query, query_embedding)
            memory_scores.append((memory, relevance_score))
        
        # Sort by relevance and return top results
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update access information for retrieved memories
        retrieved_memories = []
        for memory, score in memory_scores[:limit]:
            memory.last_accessed = datetime.utcnow()
            memory.access_count += 1
            memory.relevance_score = score
            retrieved_memories.append(memory)
        
        self.logger.debug(f"Retrieved {len(retrieved_memories)} memories for query")
        return retrieved_memories
    
    async def analyze_temporal_patterns(self, 
                                      agent_id: str = None,
                                      time_window_days: int = 30) -> Dict[str, Any]:
        """Analyze temporal patterns in memories"""
        
        # Get memories for analysis
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        
        if agent_id:
            relevant_memory_ids = self.agent_index.get(agent_id, [])
        else:
            relevant_memory_ids = list(self.memory_store.keys())
        
        relevant_memories = [
            self.memory_store[mid] for mid in relevant_memory_ids
            if mid in self.memory_store and self.memory_store[mid].created_at >= cutoff_date
        ]
        
        if len(relevant_memories) < 10:
            return {'status': 'insufficient_data', 'memory_count': len(relevant_memories)}
        
        # Detect patterns
        patterns = await self.pattern_detector.detect_temporal_patterns(relevant_memories)
        
        # Add memory statistics
        memory_stats = {
            'total_memories_analyzed': len(relevant_memories),
            'memory_type_distribution': defaultdict(int),
            'importance_distribution': defaultdict(int),
            'average_memory_age_hours': 0.0,
            'memory_creation_rate_per_day': 0.0
        }
        
        current_time = datetime.utcnow()
        total_age_hours = 0
        
        for memory in relevant_memories:
            memory_stats['memory_type_distribution'][memory.memory_type.value] += 1
            memory_stats['importance_distribution'][memory.importance.value] += 1
            
            age_hours = (current_time - memory.created_at).total_seconds() / 3600
            total_age_hours += age_hours
        
        memory_stats['average_memory_age_hours'] = total_age_hours / len(relevant_memories)
        memory_stats['memory_creation_rate_per_day'] = len(relevant_memories) / time_window_days
        
        # Convert defaultdicts to regular dicts for JSON serialization
        memory_stats['memory_type_distribution'] = dict(memory_stats['memory_type_distribution'])
        memory_stats['importance_distribution'] = dict(memory_stats['importance_distribution'])
        
        return {
            'analysis_window_days': time_window_days,
            'agent_id': agent_id,
            'memory_statistics': memory_stats,
            'temporal_patterns': patterns,
            'analysis_timestamp': datetime.utcnow()
        }
    
    async def get_memory_recommendations(self, 
                                       agent_id: str,
                                       current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations for memory management"""
        
        recommendations = []
        
        # Get agent's memories
        agent_memory_ids = self.agent_index.get(agent_id, [])
        agent_memories = [
            self.memory_store[mid] for mid in agent_memory_ids
            if mid in self.memory_store
        ]
        
        if len(agent_memories) < 5:
            return [{
                'type': 'memory_collection',
                'priority': 'high',
                'message': 'Agent needs to accumulate more memories to enable effective recommendations'
            }]
        
        # Analyze memory usage patterns
        recent_memories = [
            m for m in agent_memories
            if (datetime.utcnow() - m.last_accessed).total_seconds() < 86400 * 7  # Last week
        ]
        
        old_unused_memories = [
            m for m in agent_memories
            if (datetime.utcnow() - m.last_accessed).total_seconds() > 86400 * 30  # Older than month
            and m.importance in [MemoryImportance.LOW, MemoryImportance.TRANSIENT]
        ]
        
        # Memory cleanup recommendations
        if len(old_unused_memories) > len(agent_memories) * 0.3:
            recommendations.append({
                'type': 'memory_cleanup',
                'priority': 'medium',
                'message': f'Consider cleaning up {len(old_unused_memories)} old unused memories',
                'affected_memories': len(old_unused_memories),
                'potential_space_saving': 'high'
            })
        
        # Memory consolidation recommendations
        similar_memories = await self._find_similar_memories(agent_memories)
        if len(similar_memories) > 5:
            recommendations.append({
                'type': 'memory_consolidation',
                'priority': 'medium',
                'message': f'Found {len(similar_memories)} similar memories that could be consolidated',
                'consolidation_candidates': len(similar_memories),
                'efficiency_gain': 'medium'
            })
        
        # Pattern-based recommendations
        patterns = await self.analyze_temporal_patterns(agent_id, 14)  # 2 weeks
        
        if 'temporal_patterns' in patterns:
            cyclical_patterns = patterns['temporal_patterns'].get('cyclical_patterns', [])
            
            for pattern in cyclical_patterns:
                if pattern.get('strength', 0) > 0.5:
                    recommendations.append({
                        'type': 'pattern_exploitation',
                        'priority': 'high',
                        'message': f'Detected {pattern["type"]} - optimize memory access for this pattern',
                        'pattern_type': pattern['type'],
                        'pattern_strength': pattern['strength']
                    })
        
        # Context-specific recommendations
        relevant_memories = await self.retrieve_memories(
            current_context, limit=20
        )
        
        if len(relevant_memories) < 3:
            recommendations.append({
                'type': 'context_learning',
                'priority': 'high',
                'message': 'Current context has limited relevant memories - focus on learning',
                'relevant_memory_count': len(relevant_memories)
            })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda r: priority_order.get(r.get('priority', 'low'), 1), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def _generate_memory_embedding(self, 
                                       content: Dict[str, Any], 
                                       context: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for memory content"""
        
        # Simple feature extraction - in production, use more sophisticated encoding
        content_features = []
        
        # Extract numerical features from content
        for key, value in content.items():
            if isinstance(value, (int, float)):
                content_features.append(float(value))
            elif isinstance(value, str):
                content_features.append(float(hash(value) % 1000000) / 1000000.0)
            elif isinstance(value, bool):
                content_features.append(1.0 if value else 0.0)
            else:
                content_features.append(0.5)  # Default for complex types
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(content_features) < target_size:
            content_features.extend([0.0] * (target_size - len(content_features)))
        else:
            content_features = content_features[:target_size]
        
        # Generate temporal features
        now = datetime.utcnow()
        temporal_features = [
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.month / 12.0,
            (now.timestamp() % 86400) / 86400.0  # Time within day
        ]
        temporal_features.extend([0.0] * (64 - len(temporal_features)))
        
        # Generate context features
        context_features = []
        for key, value in context.items():
            if isinstance(value, (int, float)):
                context_features.append(float(value))
            elif isinstance(value, str):
                context_features.append(float(hash(value) % 1000000) / 1000000.0)
            else:
                context_features.append(0.5)
        
        context_target_size = 128
        if len(context_features) < context_target_size:
            context_features.extend([0.0] * (context_target_size - len(context_features)))
        else:
            context_features = context_features[:context_target_size]
        
        # Generate embedding using neural network
        content_tensor = torch.tensor(content_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        context_tensor = torch.tensor(context_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            embedding_output = self.embedding_network(content_tensor, temporal_tensor, context_tensor)
            embedding = embedding_output['embedding'].cpu().numpy()[0]
        
        return embedding
    
    async def _generate_query_embedding(self, query: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for a query"""
        
        # Use same process as memory embedding but treat query as content
        return await self._generate_memory_embedding(query, {})
    
    async def _calculate_memory_relevance(self,
                                        memory: TemporalMemoryItem,
                                        query: Dict[str, Any],
                                        query_embedding: np.ndarray) -> float:
        """Calculate relevance score between memory and query"""
        
        relevance_score = 0.0
        
        # Embedding similarity
        if memory.embedding is not None:
            embedding_sim = np.dot(memory.embedding, query_embedding) / (
                np.linalg.norm(memory.embedding) * np.linalg.norm(query_embedding) + 1e-8
            )
            relevance_score += 0.4 * max(0.0, embedding_sim)
        
        # Content keyword matching
        memory_text = str(memory.content).lower()
        query_text = str(query).lower()
        
        # Simple keyword overlap
        memory_words = set(memory_text.split())
        query_words = set(query_text.split())
        
        if query_words:
            keyword_overlap = len(memory_words & query_words) / len(query_words)
            relevance_score += 0.3 * keyword_overlap
        
        # Recency bonus
        age_days = (datetime.utcnow() - memory.created_at).total_seconds() / 86400
        recency_score = np.exp(-age_days / 30.0)  # Decay over month
        relevance_score += 0.2 * recency_score
        
        # Importance bonus
        importance_scores = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TRANSIENT: 0.2
        }
        importance_bonus = importance_scores.get(memory.importance, 0.6)
        relevance_score += 0.1 * importance_bonus
        
        return min(1.0, relevance_score)
    
    async def _capture_environmental_state(self) -> Dict[str, Any]:
        """Capture current environmental state"""
        
        # Simplified environmental state capture
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_load': 0.5,  # Would be actual system metrics
            'active_campaigns': 0,  # Would be actual campaign count
            'resource_utilization': {
                'cpu': 0.4,
                'memory': 0.6,
                'network': 0.3
            }
        }
    
    async def _consolidation_loop(self):
        """Background consolidation loop"""
        
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval_hours * 3600)
                
                if len(self.memory_store) > 100:  # Only consolidate if we have enough memories
                    memories_to_consolidate = list(self.memory_store.values())
                    
                    # Perform consolidation
                    consolidation_event = await self.consolidation_engine.consolidate_memories(
                        memories_to_consolidate, 'scheduled'
                    )
                    
                    # Remove forgotten memories
                    for memory_id in consolidation_event.memories_forgotten:
                        if memory_id in self.memory_store:
                            memory = self.memory_store[memory_id]
                            
                            # Remove from indices
                            self.memory_index[memory.memory_type].remove(memory_id)
                            
                            for agent_id in memory.associated_agents:
                                if memory_id in self.agent_index[agent_id]:
                                    self.agent_index[agent_id].remove(memory_id)
                            
                            date_key = memory.created_at.strftime('%Y-%m-%d')
                            if memory_id in self.temporal_index[date_key]:
                                self.temporal_index[date_key].remove(memory_id)
                            
                            # Remove from store
                            del self.memory_store[memory_id]
                    
                    # Update metrics
                    self.memory_metrics['consolidation_events'] += 1
                    self.memory_metrics['total_memories'] = len(self.memory_store)
                    self.memory_metrics['memory_efficiency'] = consolidation_event.memory_efficiency_change
                    
                    self.logger.info(f"Memory consolidation completed: "
                                   f"{len(consolidation_event.memories_forgotten)} memories forgotten")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _cleanup_old_memories(self):
        """Clean up old, low-importance memories to free space"""
        
        # Sort memories by priority for removal
        removal_candidates = []
        
        for memory in self.memory_store.values():
            age_days = (datetime.utcnow() - memory.created_at).total_seconds() / 86400
            access_recency_days = (datetime.utcnow() - memory.last_accessed).total_seconds() / 86400
            
            # Calculate removal priority (higher = more likely to remove)
            removal_priority = 0.0
            
            # Age penalty
            removal_priority += age_days / 365.0  # Older memories more likely to remove
            
            # Access recency penalty
            removal_priority += access_recency_days / 30.0  # Long unused memories
            
            # Importance penalty (inverted)
            importance_values = {
                MemoryImportance.CRITICAL: -10.0,  # Never remove
                MemoryImportance.HIGH: -2.0,
                MemoryImportance.MEDIUM: 0.0,
                MemoryImportance.LOW: 1.0,
                MemoryImportance.TRANSIENT: 2.0
            }
            removal_priority += importance_values.get(memory.importance, 0.0)
            
            # Access frequency bonus (inverted)
            if memory.access_count > 0:
                removal_priority -= memory.access_count / 10.0
            
            removal_candidates.append((memory.memory_id, removal_priority))
        
        # Sort by removal priority and remove top candidates
        removal_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Remove memories until we're under capacity
        memories_to_remove = max(0, len(self.memory_store) - int(self.max_memory_items * 0.9))
        
        for memory_id, _ in removal_candidates[:memories_to_remove]:
            if memory_id in self.memory_store:
                memory = self.memory_store[memory_id]
                
                # Skip critical memories
                if memory.importance == MemoryImportance.CRITICAL:
                    continue
                
                # Remove from indices
                self.memory_index[memory.memory_type].remove(memory_id)
                
                for agent_id in memory.associated_agents:
                    if memory_id in self.agent_index[agent_id]:
                        self.agent_index[agent_id].remove(memory_id)
                
                date_key = memory.created_at.strftime('%Y-%m-%d')
                if memory_id in self.temporal_index[date_key]:
                    self.temporal_index[date_key].remove(memory_id)
                
                # Remove from store
                del self.memory_store[memory_id]
        
        self.logger.info(f"Cleaned up {min(memories_to_remove, len(removal_candidates))} old memories")
    
    async def _find_similar_memories(self, memories: List[TemporalMemoryItem]) -> List[Tuple[str, str, float]]:
        """Find pairs of similar memories"""
        
        similar_pairs = []
        
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                memory1, memory2 = memories[i], memories[j]
                
                # Calculate similarity
                if memory1.embedding is not None and memory2.embedding is not None:
                    similarity = np.dot(memory1.embedding, memory2.embedding) / (
                        np.linalg.norm(memory1.embedding) * np.linalg.norm(memory2.embedding) + 1e-8
                    )
                    
                    if similarity > 0.8:  # High similarity threshold
                        similar_pairs.append((memory1.memory_id, memory2.memory_id, similarity))
        
        return similar_pairs
    
    async def get_memory_system_status(self) -> Dict[str, Any]:
        """Get current status of the memory system"""
        
        # Calculate memory age statistics
        if self.memory_store:
            current_time = datetime.utcnow()
            ages_days = [
                (current_time - memory.created_at).total_seconds() / 86400
                for memory in self.memory_store.values()
            ]
            self.memory_metrics['average_memory_age_days'] = np.mean(ages_days)
        
        return {
            'system_active': self._running,
            'total_memories': len(self.memory_store),
            'memory_capacity_used': len(self.memory_store) / self.max_memory_items,
            'metrics': self.memory_metrics.copy(),
            'index_sizes': {
                'memory_types': {mem_type.value: len(memory_ids) 
                               for mem_type, memory_ids in self.memory_index.items()},
                'agents': len(self.agent_index),
                'temporal_days': len(self.temporal_index)
            },
            'consolidation_history': len(self.consolidation_engine.consolidation_history),
            'recent_activity': {
                'memories_created_last_hour': len([
                    m for m in self.memory_store.values()
                    if (datetime.utcnow() - m.created_at).total_seconds() < 3600
                ]),
                'memories_accessed_last_hour': len([
                    m for m in self.memory_store.values()
                    if (datetime.utcnow() - m.last_accessed).total_seconds() < 3600
                ])
            }
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        memory_system = TemporalMemorySystem()
        await memory_system.start_memory_system()
        
        # Store some example memories
        for i in range(50):
            agent_id = f"agent_{i % 3}"
            memory_type = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL][i % 3]
            
            content = {
                'action': f'action_{i}',
                'target': f'target_{i % 5}',
                'outcome': np.random.choice(['success', 'failure', 'partial']),
                'confidence': np.random.random(),
                'execution_time': np.random.random() * 300
            }
            
            context = {
                'campaign_id': f'campaign_{i // 10}',
                'phase': np.random.choice(['recon', 'exploit', 'persist'])
            }
            
            memory_id = await memory_system.store_memory(
                agent_id, memory_type, content, context
            )
            
            print(f"Stored memory {memory_id}")
        
        # Retrieve memories
        query = {'action': 'action_5', 'target': 'target_1'}
        retrieved = await memory_system.retrieve_memories(query, limit=5)
        print(f"\nRetrieved {len(retrieved)} memories for query")
        
        # Analyze patterns
        patterns = await memory_system.analyze_temporal_patterns("agent_0", 7)
        print(f"\nTemporal patterns: {json.dumps(patterns, indent=2, default=str)}")
        
        # Get recommendations
        recommendations = await memory_system.get_memory_recommendations("agent_0", query)
        print(f"\nRecommendations: {json.dumps(recommendations, indent=2)}")
        
        # Get system status
        status = await memory_system.get_memory_system_status()
        print(f"\nSystem status: {json.dumps(status, indent=2)}")
        
        await memory_system.stop_memory_system()
    
    asyncio.run(main())