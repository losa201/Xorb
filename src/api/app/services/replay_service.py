"""
Replay Service - Normalizes traces, deduplicates, labels outcomes, and manages versioned replay store.

Handles:
- Trace normalization and deduplication
- Episode artifact management (PCAP, logs, metrics)
- Versioned replay store with S3-compatible storage
- Outcome labeling and metadata enrichment
- Replay buffer management for training
"""

import asyncio
import gzip
import hashlib
import json
import os
import tempfile
import time
import uuid
import zstandard as zstd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict

import aiofiles
import pandas as pd
import numpy as np
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError

from .environment_api import EpisodeManifest, StepEvent, ActorRole


class ReplayConfig(BaseModel):
    """Configuration for replay service."""
    storage_backend: str = "s3"  # s3, local, minio
    bucket_name: str = "xorb-replays"
    local_storage_path: str = "/tmp/xorb-replays"
    compression: str = "zstd"  # zstd, gzip, none
    dedup_window_hours: int = 24
    max_replay_age_days: int = 90
    batch_size: int = 100
    
    # S3 Configuration
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_region: str = "us-east-1"


@dataclass
class ReplayArtifact:
    """Replay artifact metadata."""
    artifact_type: str  # pcap, logs, metrics, manifest
    file_path: str
    file_size: int
    checksum: str
    compression: str
    created_at: datetime


@dataclass
class ReplayEntry:
    """Replay buffer entry."""
    episode_id: str
    scenario_id: str
    step_count: int
    duration_seconds: float
    outcome: str  # red_win, blue_win, draw
    red_score: float
    blue_score: float
    quality_score: float  # Overall episode quality
    artifacts: List[ReplayArtifact]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'artifacts': [asdict(a) for a in self.artifacts],
            'created_at': self.created_at.isoformat()
        }


class ReplayDeduplicator:
    """Deduplicates replay episodes based on content similarity."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.episode_hashes: Dict[str, str] = {}
        self.episode_signatures: Dict[str, Set[str]] = {}
    
    def calculate_episode_signature(self, step_events: List[StepEvent]) -> Set[str]:
        """Calculate episode signature for deduplication."""
        signatures = set()
        
        # Create signatures based on action sequences
        for i, event in enumerate(step_events):
            # Action signature
            action_sig = f"{event.actor}:{event.act.get('type')}:{event.act.get('parameters', {}).get('path', 'none')}"
            signatures.add(action_sig)
            
            # Sequence signature (overlapping windows)
            if i > 0:
                prev_event = step_events[i-1]
                seq_sig = f"{prev_event.actor}->{event.actor}:{prev_event.act.get('type')}->{event.act.get('type')}"
                signatures.add(seq_sig)
        
        return signatures
    
    def calculate_similarity(self, sig1: Set[str], sig2: Set[str]) -> float:
        """Calculate Jaccard similarity between episode signatures."""
        if not sig1 or not sig2:
            return 0.0
        
        intersection = len(sig1.intersection(sig2))
        union = len(sig1.union(sig2))
        
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, episode_id: str, step_events: List[StepEvent]) -> Tuple[bool, Optional[str]]:
        """Check if episode is a duplicate of existing episodes."""
        current_signature = self.calculate_episode_signature(step_events)
        
        for existing_episode_id, existing_signature in self.episode_signatures.items():
            if existing_episode_id == episode_id:
                continue
            
            similarity = self.calculate_similarity(current_signature, existing_signature)
            if similarity >= self.similarity_threshold:
                return True, existing_episode_id
        
        # Store signature for future comparisons
        self.episode_signatures[episode_id] = current_signature
        return False, None


class StorageBackend:
    """Abstract storage backend interface."""
    
    async def upload_file(self, file_path: str, key: str) -> str:
        raise NotImplementedError
    
    async def download_file(self, key: str, file_path: str) -> bool:
        raise NotImplementedError
    
    async def list_objects(self, prefix: str) -> List[str]:
        raise NotImplementedError
    
    async def delete_object(self, key: str) -> bool:
        raise NotImplementedError


class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend."""
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.s3_access_key_id,
            aws_secret_access_key=config.s3_secret_access_key,
            region_name=config.s3_region
        )
        self.bucket_name = config.bucket_name
    
    async def upload_file(self, file_path: str, key: str) -> str:
        """Upload file to S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.upload_file,
                file_path, self.bucket_name, key
            )
            return f"s3://{self.bucket_name}/{key}"
        except ClientError as e:
            raise Exception(f"Failed to upload {key}: {e}")
    
    async def download_file(self, key: str, file_path: str) -> bool:
        """Download file from S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.download_file,
                self.bucket_name, key, file_path
            )
            return True
        except ClientError:
            return False
    
    async def list_objects(self, prefix: str) -> List[str]:
        """List objects with prefix."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.s3_client.list_objects_v2,
                {'Bucket': self.bucket_name, 'Prefix': prefix}
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError:
            return []
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.delete_object,
                {'Bucket': self.bucket_name, 'Key': key}
            )
            return True
        except ClientError:
            return False


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, config: ReplayConfig):
        self.base_path = Path(config.local_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(self, file_path: str, key: str) -> str:
        """Copy file to local storage."""
        dest_path = self.base_path / key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'rb') as src:
            async with aiofiles.open(dest_path, 'wb') as dst:
                content = await src.read()
                await dst.write(content)
        
        return str(dest_path)
    
    async def download_file(self, key: str, file_path: str) -> bool:
        """Copy file from local storage."""
        src_path = self.base_path / key
        if not src_path.exists():
            return False
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(src_path, 'rb') as src:
            async with aiofiles.open(file_path, 'wb') as dst:
                content = await src.read()
                await dst.write(content)
        
        return True
    
    async def list_objects(self, prefix: str) -> List[str]:
        """List objects with prefix."""
        prefix_path = self.base_path / prefix
        if not prefix_path.exists():
            return []
        
        objects = []
        for path in prefix_path.rglob('*'):
            if path.is_file():
                relative_path = path.relative_to(self.base_path)
                objects.append(str(relative_path))
        
        return objects
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from local storage."""
        file_path = self.base_path / key
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class ReplayService:
    """Main replay service for managing training data."""
    
    def __init__(self, config: Optional[ReplayConfig] = None):
        self.config = config or ReplayConfig()
        
        # Initialize storage backend
        if self.config.storage_backend == "s3":
            self.storage = S3StorageBackend(self.config)
        else:
            self.storage = LocalStorageBackend(self.config)
        
        # Initialize deduplicator
        self.deduplicator = ReplayDeduplicator()
        
        # Replay buffer cache
        self.replay_buffer: List[ReplayEntry] = []
        self.buffer_last_updated = None
        
        # Compression
        if self.config.compression == "zstd":
            self.compressor = zstd.ZstdCompressor(level=3)
            self.decompressor = zstd.ZstdDecompressor()
        
    async def ingest_episode(
        self,
        manifest: EpisodeManifest,
        step_events: List[StepEvent],
        pcap_data: Optional[bytes] = None,
        raw_logs: Optional[List[str]] = None
    ) -> Optional[ReplayEntry]:
        """
        Ingest a completed episode into the replay system.
        
        Args:
            manifest: Episode manifest
            step_events: List of step events (JSONL format)
            pcap_data: Optional PCAP data
            raw_logs: Optional raw log lines
        
        Returns:
            ReplayEntry if successfully ingested, None if duplicate
        """
        # Check for duplicates
        is_dup, dup_episode = self.deduplicator.is_duplicate(manifest.episode_id, step_events)
        if is_dup:
            print(f"Episode {manifest.episode_id} is duplicate of {dup_episode}, skipping")
            return None
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(step_events, manifest)
        
        # Skip low-quality episodes
        if quality_score < 0.3:
            print(f"Episode {manifest.episode_id} has low quality ({quality_score:.2f}), skipping")
            return None
        
        # Create storage paths
        date_str = manifest.created_at.strftime("%Y-%m-%d")
        episode_prefix = f"scenario={manifest.scenario_id}/date={date_str}/{manifest.episode_id}"
        
        artifacts = []
        
        # Store step events as compressed JSONL
        events_data = '\n'.join([event.to_jsonl() for event in step_events])
        events_file = await self._compress_and_store(
            events_data.encode(), f"{episode_prefix}/events.jsonl.zst"
        )
        artifacts.append(ReplayArtifact(
            artifact_type="events",
            file_path=events_file,
            file_size=len(events_data.encode()),
            checksum=hashlib.sha256(events_data.encode()).hexdigest(),
            compression=self.config.compression,
            created_at=datetime.utcnow()
        ))
        
        # Store rewards (calculated during episode)
        rewards_data = []
        for event in step_events:
            if event.reward is not None:
                rewards_data.append({
                    "t": event.t,
                    "actor": event.actor,
                    "reward": event.reward
                })
        
        if rewards_data:
            rewards_jsonl = '\n'.join([json.dumps(r) for r in rewards_data])
            rewards_file = await self._compress_and_store(
                rewards_jsonl.encode(), f"{episode_prefix}/rewards.jsonl.zst"
            )
            artifacts.append(ReplayArtifact(
                artifact_type="rewards",
                file_path=rewards_file,
                file_size=len(rewards_jsonl.encode()),
                checksum=hashlib.sha256(rewards_jsonl.encode()).hexdigest(),
                compression=self.config.compression,
                created_at=datetime.utcnow()
            ))
        
        # Store metrics as Parquet
        metrics_df = self._create_metrics_dataframe(step_events, manifest)
        if not metrics_df.empty:
            metrics_file = await self._store_parquet(
                metrics_df, f"{episode_prefix}/metrics.parquet"
            )
            artifacts.append(ReplayArtifact(
                artifact_type="metrics",
                file_path=metrics_file,
                file_size=0,  # Will be calculated by storage
                checksum="",
                compression="none",
                created_at=datetime.utcnow()
            ))
        
        # Store PCAP if provided
        if pcap_data:
            pcap_file = await self._compress_and_store(
                pcap_data, f"{episode_prefix}/trace.pcap.zst"
            )
            artifacts.append(ReplayArtifact(
                artifact_type="pcap",
                file_path=pcap_file,
                file_size=len(pcap_data),
                checksum=hashlib.sha256(pcap_data).hexdigest(),
                compression=self.config.compression,
                created_at=datetime.utcnow()
            ))
        
        # Store raw logs if provided
        if raw_logs:
            logs_data = '\n'.join(raw_logs)
            logs_file = await self._compress_and_store(
                logs_data.encode(), f"{episode_prefix}/logs.jsonl.zst"
            )
            artifacts.append(ReplayArtifact(
                artifact_type="logs",
                file_path=logs_file,
                file_size=len(logs_data.encode()),
                checksum=hashlib.sha256(logs_data.encode()).hexdigest(),
                compression=self.config.compression,
                created_at=datetime.utcnow()
            ))
        
        # Store manifest
        manifest_data = manifest.json()
        manifest_file = await self._store_text(
            manifest_data, f"{episode_prefix}/manifest.json"
        )
        artifacts.append(ReplayArtifact(
            artifact_type="manifest",
            file_path=manifest_file,
            file_size=len(manifest_data.encode()),
            checksum=hashlib.sha256(manifest_data.encode()).hexdigest(),
            compression="none",
            created_at=datetime.utcnow()
        ))
        
        # Create replay entry
        replay_entry = ReplayEntry(
            episode_id=manifest.episode_id,
            scenario_id=manifest.scenario_id,
            step_count=manifest.steps,
            duration_seconds=manifest.duration_seconds,
            outcome=manifest.result.get("winner", "draw"),
            red_score=manifest.result.get("red_score", 0.0),
            blue_score=manifest.result.get("blue_score", 0.0),
            quality_score=quality_score,
            artifacts=artifacts,
            metadata=manifest.metadata or {},
            created_at=manifest.created_at
        )
        
        # Add to buffer
        self.replay_buffer.append(replay_entry)
        
        print(f"Ingested episode {manifest.episode_id} with quality {quality_score:.2f}")
        return replay_entry
    
    async def get_replay_buffer(
        self,
        scenario_id: Optional[str] = None,
        min_quality: float = 0.5,
        max_age_days: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[ReplayEntry]:
        """
        Get replay buffer entries for training.
        
        Args:
            scenario_id: Filter by scenario
            min_quality: Minimum quality score
            max_age_days: Maximum age in days
            limit: Maximum number of entries
        
        Returns:
            List of replay entries
        """
        # Update buffer if needed
        await self._update_replay_buffer()
        
        # Apply filters
        filtered_entries = []
        cutoff_date = None
        if max_age_days:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        for entry in self.replay_buffer:
            # Scenario filter
            if scenario_id and entry.scenario_id != scenario_id:
                continue
            
            # Quality filter
            if entry.quality_score < min_quality:
                continue
            
            # Age filter
            if cutoff_date and entry.created_at < cutoff_date:
                continue
            
            filtered_entries.append(entry)
        
        # Sort by quality (descending) and age (recent first)
        filtered_entries.sort(
            key=lambda x: (x.quality_score, x.created_at.timestamp()),
            reverse=True
        )
        
        # Apply limit
        if limit:
            filtered_entries = filtered_entries[:limit]
        
        return filtered_entries
    
    async def get_episode_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete episode data including all artifacts.
        
        Args:
            episode_id: Episode to retrieve
        
        Returns:
            Dictionary with all episode data
        """
        # Find replay entry
        entry = None
        for replay_entry in self.replay_buffer:
            if replay_entry.episode_id == episode_id:
                entry = replay_entry
                break
        
        if not entry:
            return None
        
        episode_data = {
            "manifest": entry.to_dict(),
            "artifacts": {}
        }
        
        # Download and decompress artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            for artifact in entry.artifacts:
                try:
                    temp_file = os.path.join(temp_dir, f"{artifact.artifact_type}_temp")
                    
                    # Extract key from file path
                    if artifact.file_path.startswith("s3://"):
                        key = artifact.file_path.split("/", 3)[-1]
                    else:
                        key = os.path.relpath(artifact.file_path, self.config.local_storage_path)
                    
                    # Download
                    success = await self.storage.download_file(key, temp_file)
                    if not success:
                        continue
                    
                    # Decompress and read
                    if artifact.compression == "zstd":
                        with open(temp_file, 'rb') as f:
                            compressed_data = f.read()
                        data = self.decompressor.decompress(compressed_data)
                    else:
                        with open(temp_file, 'rb') as f:
                            data = f.read()
                    
                    # Parse based on type
                    if artifact.artifact_type in ["events", "rewards", "logs"]:
                        episode_data["artifacts"][artifact.artifact_type] = data.decode().strip().split('\n')
                    elif artifact.artifact_type == "manifest":
                        episode_data["artifacts"][artifact.artifact_type] = json.loads(data.decode())
                    else:
                        episode_data["artifacts"][artifact.artifact_type] = data
                
                except Exception as e:
                    print(f"Error loading artifact {artifact.artifact_type}: {e}")
        
        return episode_data
    
    async def cleanup_old_replays(self, max_age_days: int = 90) -> int:
        """
        Clean up old replay data.
        
        Args:
            max_age_days: Maximum age to keep
        
        Returns:
            Number of episodes cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Update buffer
        await self._update_replay_buffer()
        
        # Find old entries
        old_entries = [
            entry for entry in self.replay_buffer
            if entry.created_at < cutoff_date
        ]
        
        # Delete artifacts for old entries
        for entry in old_entries:
            try:
                for artifact in entry.artifacts:
                    if artifact.file_path.startswith("s3://"):
                        key = artifact.file_path.split("/", 3)[-1]
                    else:
                        key = os.path.relpath(artifact.file_path, self.config.local_storage_path)
                    
                    await self.storage.delete_object(key)
                
                cleaned_count += 1
            except Exception as e:
                print(f"Error cleaning up episode {entry.episode_id}: {e}")
        
        # Remove from buffer
        self.replay_buffer = [
            entry for entry in self.replay_buffer
            if entry.created_at >= cutoff_date
        ]
        
        print(f"Cleaned up {cleaned_count} old episodes")
        return cleaned_count
    
    # Private methods
    
    async def _compress_and_store(self, data: bytes, key: str) -> str:
        """Compress data and store it."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            if self.config.compression == "zstd":
                compressed_data = self.compressor.compress(data)
            elif self.config.compression == "gzip":
                compressed_data = gzip.compress(data)
            else:
                compressed_data = data
            
            temp_file.write(compressed_data)
            temp_file.flush()
            
            # Upload to storage
            file_path = await self.storage.upload_file(temp_file.name, key)
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return file_path
    
    async def _store_text(self, text: str, key: str) -> str:
        """Store text data."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(text)
            temp_file.flush()
            
            file_path = await self.storage.upload_file(temp_file.name, key)
            os.unlink(temp_file.name)
            
            return file_path
    
    async def _store_parquet(self, df: pd.DataFrame, key: str) -> str:
        """Store DataFrame as Parquet."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            df.to_parquet(temp_file.name, compression='snappy')
            
            file_path = await self.storage.upload_file(temp_file.name, key)
            os.unlink(temp_file.name)
            
            return file_path
    
    def _calculate_quality_score(self, step_events: List[StepEvent], manifest: EpisodeManifest) -> float:
        """Calculate episode quality score."""
        if not step_events:
            return 0.0
        
        # Factors for quality assessment
        score = 0.0
        
        # 1. Episode length (prefer moderate length)
        optimal_steps = 100
        length_score = 1.0 - abs(len(step_events) - optimal_steps) / optimal_steps
        length_score = max(0.0, min(1.0, length_score))
        score += 0.2 * length_score
        
        # 2. Action diversity
        action_types = set()
        actors = set()
        for event in step_events:
            action_types.add(event.act.get('type', 'unknown'))
            actors.add(event.actor)
        
        diversity_score = (len(action_types) / 5.0 + len(actors) / 2.0) / 2.0
        diversity_score = min(1.0, diversity_score)
        score += 0.3 * diversity_score
        
        # 3. Outcome balance (prefer close games)
        red_score = manifest.result.get("red_score", 0.0)
        blue_score = manifest.result.get("blue_score", 0.0)
        if red_score + blue_score > 0:
            balance = 1.0 - abs(red_score - blue_score) / (red_score + blue_score)
        else:
            balance = 0.5
        score += 0.3 * balance
        
        # 4. Reward distribution (prefer episodes with varied rewards)
        rewards = [event.reward for event in step_events if event.reward is not None]
        if rewards:
            reward_std = np.std(rewards)
            reward_score = min(1.0, reward_std / 0.5)  # Normalize by expected std
        else:
            reward_score = 0.0
        score += 0.2 * reward_score
        
        return score
    
    def _create_metrics_dataframe(self, step_events: List[StepEvent], manifest: EpisodeManifest) -> pd.DataFrame:
        """Create metrics DataFrame for analysis."""
        if not step_events:
            return pd.DataFrame()
        
        metrics_data = []
        
        for event in step_events:
            metrics_data.append({
                'episode_id': manifest.episode_id,
                'step': event.t,
                'actor': event.actor,
                'action_type': event.act.get('type', 'unknown'),
                'reward': event.reward,
                'latency_ms': event.r.get('latency_ms', 0),
                'success': event.r.get('success', False),
                'timestamp': time.time()  # Would be actual timestamp in real implementation
            })
        
        return pd.DataFrame(metrics_data)
    
    async def _update_replay_buffer(self) -> None:
        """Update replay buffer from storage."""
        # Check if update is needed (cache for 5 minutes)
        if (self.buffer_last_updated and 
            time.time() - self.buffer_last_updated < 300):
            return
        
        # This would scan storage for new episodes
        # For now, we'll just mark as updated
        self.buffer_last_updated = time.time()