#!/usr/bin/env python3

import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .atom import KnowledgeAtom, AtomType


class ShardTier(str, Enum):
    HOT = "hot"      # Redis - frequently accessed, high-performance
    WARM = "warm"    # SQLite - moderate access, persistent
    COLD = "cold"    # S3/GCS - archived, long-term storage


@dataclass
class ShardConfig:
    tier: ShardTier
    storage_backend: str
    connection_string: str
    max_size: Optional[int] = None
    ttl_seconds: Optional[int] = None
    compression: bool = False


class AtomShardManager:
    def __init__(self):
        self.shards: Dict[ShardTier, List[ShardConfig]] = {
            ShardTier.HOT: [],
            ShardTier.WARM: [],
            ShardTier.COLD: []
        }
        
        self.shard_ring: Dict[int, str] = {}
        self.replication_factor = 3
        self.logger = logging.getLogger(__name__)
        
        self._initialize_default_shards()

    def _initialize_default_shards(self):
        self.shards[ShardTier.HOT] = [
            ShardConfig(
                tier=ShardTier.HOT,
                storage_backend="redis",
                connection_string="redis://localhost:6379/0",
                ttl_seconds=3600
            ),
            ShardConfig(
                tier=ShardTier.HOT,
                storage_backend="redis",
                connection_string="redis://localhost:6379/1",
                ttl_seconds=3600
            )
        ]
        
        self.shards[ShardTier.WARM] = [
            ShardConfig(
                tier=ShardTier.WARM,
                storage_backend="sqlite",
                connection_string="sqlite+aiosqlite:///./xorb_shard_0.db",
                max_size=10000
            ),
            ShardConfig(
                tier=ShardTier.WARM,
                storage_backend="sqlite", 
                connection_string="sqlite+aiosqlite:///./xorb_shard_1.db",
                max_size=10000
            )
        ]
        
        self.shards[ShardTier.COLD] = [
            ShardConfig(
                tier=ShardTier.COLD,
                storage_backend="s3",
                connection_string="s3://xorb-knowledge-archive",
                compression=True
            )
        ]
        
        self._build_shard_ring()

    def _build_shard_ring(self):
        self.shard_ring.clear()
        
        for tier, configs in self.shards.items():
            for i, config in enumerate(configs):
                shard_key = f"{tier.value}:{i}"
                
                for replica in range(self.replication_factor):
                    hash_input = f"{shard_key}:{replica}"
                    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                    self.shard_ring[hash_value] = shard_key

    def get_shard_for_atom(self, atom_id: str, tier: ShardTier) -> str:
        hash_value = int(hashlib.md5(atom_id.encode()).hexdigest(), 16)
        
        tier_shards = [k for k, v in self.shard_ring.items() if v.startswith(tier.value)]
        if not tier_shards:
            return f"{tier.value}:0"
        
        tier_shards.sort()
        
        for shard_hash in tier_shards:
            if hash_value <= shard_hash:
                return self.shard_ring[shard_hash]
        
        return self.shard_ring[tier_shards[0]]

    def get_shard_config(self, shard_key: str) -> Optional[ShardConfig]:
        tier_name, shard_index = shard_key.split(":", 1)
        tier = ShardTier(tier_name)
        index = int(shard_index)
        
        if tier in self.shards and index < len(self.shards[tier]):
            return self.shards[tier][index]
        
        return None

    def determine_storage_tier(self, atom: KnowledgeAtom) -> ShardTier:
        if atom.usage_count > 10 and atom.confidence > 0.7:
            return ShardTier.HOT
        
        if atom.predictive_score > 0.5 or atom.confidence > 0.5:
            return ShardTier.WARM
        
        return ShardTier.COLD

    def should_promote_atom(self, atom: KnowledgeAtom, current_tier: ShardTier) -> Optional[ShardTier]:
        if current_tier == ShardTier.COLD:
            if atom.usage_count > 5 or atom.confidence > 0.6:
                return ShardTier.WARM
        
        if current_tier == ShardTier.WARM:
            if atom.usage_count > 15 and atom.confidence > 0.7:
                return ShardTier.HOT
        
        return None

    def should_demote_atom(self, atom: KnowledgeAtom, current_tier: ShardTier) -> Optional[ShardTier]:
        if current_tier == ShardTier.HOT:
            if atom.usage_count < 5 or atom.confidence < 0.4:
                return ShardTier.WARM
        
        if current_tier == ShardTier.WARM:
            if atom.usage_count == 0 and atom.confidence < 0.3:
                return ShardTier.COLD
        
        return None

    def get_replication_shards(self, atom_id: str, tier: ShardTier) -> List[str]:
        primary_shard = self.get_shard_for_atom(atom_id, tier)
        replicas = []
        
        hash_value = int(hashlib.md5(atom_id.encode()).hexdigest(), 16)
        tier_shards = [k for k, v in self.shard_ring.items() if v.startswith(tier.value)]
        tier_shards.sort()
        
        primary_index = None
        for i, shard_hash in enumerate(tier_shards):
            if self.shard_ring[shard_hash] == primary_shard:
                primary_index = i
                break
        
        if primary_index is not None:
            for i in range(1, min(self.replication_factor, len(tier_shards))):
                replica_index = (primary_index + i) % len(tier_shards)
                replica_shard = self.shard_ring[tier_shards[replica_index]]
                replicas.append(replica_shard)
        
        return replicas

    def add_shard(self, tier: ShardTier, config: ShardConfig):
        self.shards[tier].append(config)
        self._build_shard_ring()
        self.logger.info(f"Added new shard to {tier.value} tier")

    def remove_shard(self, tier: ShardTier, shard_index: int) -> bool:
        if tier in self.shards and shard_index < len(self.shards[tier]):
            removed = self.shards[tier].pop(shard_index)
            self._build_shard_ring()
            self.logger.info(f"Removed shard from {tier.value} tier: {removed.connection_string}")
            return True
        return False

    def get_shard_distribution(self) -> Dict[str, Any]:
        distribution = {}
        
        for tier in ShardTier:
            distribution[tier.value] = {
                "shard_count": len(self.shards[tier]),
                "replication_factor": self.replication_factor,
                "shards": []
            }
            
            for i, config in enumerate(self.shards[tier]):
                distribution[tier.value]["shards"].append({
                    "index": i,
                    "backend": config.storage_backend,
                    "connection": config.connection_string,
                    "max_size": config.max_size,
                    "ttl_seconds": config.ttl_seconds,
                    "compression": config.compression
                })
        
        return distribution

    def calculate_shard_load(self, atom_counts: Dict[str, int]) -> Dict[str, float]:
        shard_loads = {}
        total_atoms = sum(atom_counts.values())
        
        if total_atoms == 0:
            return {}
        
        for shard_key, count in atom_counts.items():
            load_percentage = (count / total_atoms) * 100
            shard_loads[shard_key] = load_percentage
        
        return shard_loads

    def recommend_rebalancing(self, shard_loads: Dict[str, float]) -> List[Dict[str, Any]]:
        recommendations = []
        
        avg_load = sum(shard_loads.values()) / len(shard_loads) if shard_loads else 0
        threshold = 20.0  # 20% deviation from average
        
        for shard_key, load in shard_loads.items():
            deviation = abs(load - avg_load)
            
            if deviation > threshold:
                recommendation = {
                    "shard": shard_key,
                    "current_load": load,
                    "average_load": avg_load,
                    "deviation": deviation,
                    "action": "rebalance" if load > avg_load + threshold else "consolidate"
                }
                recommendations.append(recommendation)
        
        return recommendations

    def get_optimal_shard_count(self, estimated_atoms: int, target_atoms_per_shard: int = 5000) -> Dict[ShardTier, int]:
        optimal_counts = {}
        
        hot_atoms = int(estimated_atoms * 0.2)  # 20% hot
        warm_atoms = int(estimated_atoms * 0.6)  # 60% warm  
        cold_atoms = int(estimated_atoms * 0.2)  # 20% cold
        
        optimal_counts[ShardTier.HOT] = max(1, (hot_atoms // target_atoms_per_shard) + 1)
        optimal_counts[ShardTier.WARM] = max(1, (warm_atoms // target_atoms_per_shard) + 1)
        optimal_counts[ShardTier.COLD] = max(1, (cold_atoms // (target_atoms_per_shard * 10)) + 1)  # Cold storage can handle more
        
        return optimal_counts