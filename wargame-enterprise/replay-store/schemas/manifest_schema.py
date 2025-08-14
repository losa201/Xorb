#!/usr/bin/env python3
"""
Replay Store Manifest Schema
Episode metadata and manifest management for cyber range replay system
"""

import json
import hashlib
import gzip
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from pathlib import Path
import uuid

class EpisodeStatus(Enum):
    """Episode execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class EpisodeType(Enum):
    """Type of cyber range episode"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    SHADOW_LEAGUE = "shadow_league"
    MAIN_LEAGUE = "main_league"
    COMPLIANCE = "compliance"
    CAMPAIGN = "campaign"
    STRESS_TEST = "stress_test"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    NIST = "nist"
    FISMA = "fisma"

@dataclass
class FileManifestEntry:
    """Manifest entry for each file in the episode"""
    file_path: str
    file_type: str  # events, metrics, pcap, logs, artifacts
    file_size_bytes: int
    compression: str  # none, gzip, zstd
    checksum_sha256: str
    created_timestamp: str
    schema_version: str
    record_count: Optional[int] = None
    
    @classmethod
    def from_file(cls, file_path: str, file_type: str, schema_version: str = "1.0.0") -> "FileManifestEntry":
        """Create manifest entry from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate file size
        file_size = path.stat().st_size
        
        # Calculate checksum
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        checksum = sha256_hash.hexdigest()
        
        # Determine compression
        compression = "none"
        if file_path.endswith(".gz"):
            compression = "gzip"
        elif file_path.endswith(".zst"):
            compression = "zstd"
        
        return cls(
            file_path=str(path),
            file_type=file_type,
            file_size_bytes=file_size,
            compression=compression,
            checksum_sha256=checksum,
            created_timestamp=datetime.utcnow().isoformat(),
            schema_version=schema_version
        )

@dataclass
class AgentConfig:
    """Configuration for an agent in the episode"""
    agent_id: str
    agent_type: str  # red, blue, purple
    model_name: str
    model_version: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    policy_version: Optional[str] = None
    training_checkpoint: Optional[str] = None

@dataclass
class EnvironmentConfig:
    """Environment configuration for the episode"""
    environment_id: str
    organization_profile: str
    network_topology: Dict[str, Any] = field(default_factory=dict)
    initial_vulnerabilities: List[str] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    difficulty_level: str = "medium"  # easy, medium, hard, expert
    chaos_engineering: bool = False

@dataclass
class SafetyConfig:
    """Safety and RoE configuration"""
    rules_of_engagement: Dict[str, Any] = field(default_factory=dict)
    safety_critic_enabled: bool = True
    scope_enforcement: Dict[str, Any] = field(default_factory=dict)
    kill_switch_triggers: List[str] = field(default_factory=list)
    max_episode_duration_minutes: int = 60
    max_resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class EpisodeMetadata:
    """Rich metadata for episode analysis"""
    experiment_name: Optional[str] = None
    experiment_version: Optional[str] = None
    researcher_id: Optional[str] = None
    research_objective: Optional[str] = None
    
    # League information
    league_season: Optional[str] = None
    league_tier: Optional[str] = None
    promotion_candidate: bool = False
    
    # Curriculum information
    curriculum_tier: str = "T1"  # T1, T2, T3
    curriculum_scenario: Optional[str] = None
    prerequisite_episodes: List[str] = field(default_factory=list)
    
    # Quality-diversity
    behavioral_descriptors: Dict[str, float] = field(default_factory=dict)
    novelty_score: Optional[float] = None
    quality_score: Optional[float] = None

@dataclass
class EpisodeResults:
    """Episode execution results"""
    # Overall results
    success: bool = False
    termination_reason: str = ""
    
    # Performance metrics
    total_duration_seconds: float = 0.0
    rounds_completed: int = 0
    
    # Team performance
    red_team_score: float = 0.0
    blue_team_score: float = 0.0
    red_team_success_rate: float = 0.0
    blue_team_detection_rate: float = 0.0
    
    # Security metrics
    assets_compromised: int = 0
    vulnerabilities_exploited: List[str] = field(default_factory=list)
    persistence_achieved: bool = False
    data_exfiltrated_mb: float = 0.0
    
    # Defense metrics
    threats_detected: int = 0
    threats_blocked: int = 0
    false_positives: int = 0
    countermeasures_deployed: int = 0
    
    # Learning metrics
    policy_updates: int = 0
    reward_achieved: float = 0.0
    exploration_bonus: float = 0.0
    
    # Compliance
    compliance_violations: int = 0
    audit_completeness: float = 1.0
    regulatory_notifications: int = 0

@dataclass
class EpisodeManifest:
    """Complete episode manifest with all metadata and file references"""
    # Episode identification
    episode_id: str = field(default_factory=lambda: f"ep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}")
    session_id: str = ""
    parent_episode_id: Optional[str] = None
    
    # Temporal information
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_timestamp: Optional[str] = None
    completed_timestamp: Optional[str] = None
    
    # Episode configuration
    episode_type: EpisodeType = EpisodeType.TRAINING
    episode_status: EpisodeStatus = EpisodeStatus.PENDING
    
    # Configuration objects
    agents: List[AgentConfig] = field(default_factory=list)
    environment: Optional[EnvironmentConfig] = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)
    
    # Execution results
    results: EpisodeResults = field(default_factory=EpisodeResults)
    
    # File manifest
    files: List[FileManifestEntry] = field(default_factory=list)
    
    # Versioning and compatibility
    manifest_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    
    # Data retention
    retention_policy: str = "standard"  # minimal, standard, extended, permanent
    purge_after_days: Optional[int] = None
    
    def add_file(self, file_path: str, file_type: str):
        """Add a file to the manifest"""
        manifest_entry = FileManifestEntry.from_file(file_path, file_type, self.schema_version)
        self.files.append(manifest_entry)
    
    def get_file_by_type(self, file_type: str) -> List[FileManifestEntry]:
        """Get all files of a specific type"""
        return [f for f in self.files if f.file_type == file_type]
    
    def get_total_size_bytes(self) -> int:
        """Calculate total size of all files"""
        return sum(f.file_size_bytes for f in self.files)
    
    def get_file_types(self) -> Set[str]:
        """Get all file types in the manifest"""
        return {f.file_type for f in self.files}
    
    def verify_integrity(self) -> List[str]:
        """Verify file integrity against checksums"""
        errors = []
        
        for file_entry in self.files:
            try:
                path = Path(file_entry.file_path)
                if not path.exists():
                    errors.append(f"Missing file: {file_entry.file_path}")
                    continue
                
                # Recalculate checksum
                sha256_hash = hashlib.sha256()
                with open(file_entry.file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                current_checksum = sha256_hash.hexdigest()
                
                if current_checksum != file_entry.checksum_sha256:
                    errors.append(f"Checksum mismatch for {file_entry.file_path}")
                
            except Exception as e:
                errors.append(f"Error verifying {file_entry.file_path}: {str(e)}")
        
        return errors
    
    def update_status(self, status: EpisodeStatus, timestamp: Optional[str] = None):
        """Update episode status with timestamp"""
        self.episode_status = status
        current_time = timestamp or datetime.utcnow().isoformat()
        
        if status == EpisodeStatus.RUNNING and not self.started_timestamp:
            self.started_timestamp = current_time
        elif status in [EpisodeStatus.COMPLETED, EpisodeStatus.FAILED, EpisodeStatus.TIMEOUT, EpisodeStatus.CANCELLED]:
            self.completed_timestamp = current_time
    
    def calculate_duration(self) -> Optional[timedelta]:
        """Calculate episode duration if available"""
        if not self.started_timestamp or not self.completed_timestamp:
            return None
        
        start_dt = datetime.fromisoformat(self.started_timestamp.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(self.completed_timestamp.replace('Z', '+00:00'))
        return end_dt - start_dt
    
    def to_json(self, pretty: bool = True) -> str:
        """Serialize manifest to JSON"""
        manifest_dict = asdict(self)
        
        # Convert enums to values
        manifest_dict['episode_type'] = self.episode_type.value
        manifest_dict['episode_status'] = self.episode_status.value
        
        if self.environment:
            manifest_dict['environment']['compliance_frameworks'] = [
                cf.value for cf in self.environment.compliance_frameworks
            ]
        
        indent = 2 if pretty else None
        return json.dumps(manifest_dict, indent=indent, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "EpisodeManifest":
        """Deserialize manifest from JSON"""
        data = json.loads(json_str)
        
        # Convert enum values back to enums
        if 'episode_type' in data:
            data['episode_type'] = EpisodeType(data['episode_type'])
        if 'episode_status' in data:
            data['episode_status'] = EpisodeStatus(data['episode_status'])
        
        # Handle compliance frameworks
        if 'environment' in data and 'compliance_frameworks' in data['environment']:
            data['environment']['compliance_frameworks'] = [
                ComplianceFramework(cf) for cf in data['environment']['compliance_frameworks']
            ]
        
        # Create objects from dicts
        manifest = cls()
        manifest.__dict__.update(data)
        
        # Reconstruct nested objects
        if 'agents' in data:
            manifest.agents = [AgentConfig(**agent) for agent in data['agents']]
        
        if 'environment' in data:
            manifest.environment = EnvironmentConfig(**data['environment'])
        
        if 'safety' in data:
            manifest.safety = SafetyConfig(**data['safety'])
        
        if 'metadata' in data:
            manifest.metadata = EpisodeMetadata(**data['metadata'])
        
        if 'results' in data:
            manifest.results = EpisodeResults(**data['results'])
        
        if 'files' in data:
            manifest.files = [FileManifestEntry(**file_entry) for file_entry in data['files']]
        
        return manifest
    
    def save_to_file(self, file_path: str, compress: bool = True):
        """Save manifest to file with optional compression"""
        json_data = self.to_json()
        
        if compress:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "EpisodeManifest":
        """Load manifest from file with automatic compression detection"""
        try:
            # Try compressed first
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                json_data = f.read()
        except (gzip.BadGzipFile, OSError):
            # Fall back to uncompressed
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
        
        return cls.from_json(json_data)

class ManifestManager:
    """Manager for episode manifests with indexing and search"""
    
    def __init__(self, storage_root: str):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_root / "manifest_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load manifest index from file"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save manifest index to file"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def store_manifest(self, manifest: EpisodeManifest) -> str:
        """Store manifest and update index"""
        manifest_file = self.storage_root / f"{manifest.episode_id}_manifest.json.gz"
        manifest.save_to_file(str(manifest_file), compress=True)
        
        # Update index
        self.index[manifest.episode_id] = {
            "file_path": str(manifest_file),
            "episode_type": manifest.episode_type.value,
            "episode_status": manifest.episode_status.value,
            "created_timestamp": manifest.created_timestamp,
            "total_size_bytes": manifest.get_total_size_bytes(),
            "file_types": list(manifest.get_file_types())
        }
        
        self._save_index()
        return str(manifest_file)
    
    def get_manifest(self, episode_id: str) -> Optional[EpisodeManifest]:
        """Retrieve manifest by episode ID"""
        if episode_id not in self.index:
            return None
        
        file_path = self.index[episode_id]["file_path"]
        return EpisodeManifest.load_from_file(file_path)
    
    def list_episodes(self, episode_type: Optional[EpisodeType] = None,
                     status: Optional[EpisodeStatus] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """List episodes with optional filtering"""
        episodes = []
        
        for episode_id, episode_info in self.index.items():
            # Apply filters
            if episode_type and episode_info["episode_type"] != episode_type.value:
                continue
            if status and episode_info["episode_status"] != status.value:
                continue
            
            episodes.append({
                "episode_id": episode_id,
                **episode_info
            })
        
        # Sort by creation time (newest first)
        episodes.sort(key=lambda x: x["created_timestamp"], reverse=True)
        
        return episodes[:limit]
    
    def cleanup_old_episodes(self, days_old: int = 30):
        """Clean up episodes older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        cutoff_iso = cutoff_date.isoformat()
        
        to_remove = []
        for episode_id, episode_info in self.index.items():
            if episode_info["created_timestamp"] < cutoff_iso:
                # Remove files
                try:
                    Path(episode_info["file_path"]).unlink()
                except FileNotFoundError:
                    pass
                to_remove.append(episode_id)
        
        # Update index
        for episode_id in to_remove:
            del self.index[episode_id]
        
        self._save_index()
        return len(to_remove)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing manifest schema and management...")
    
    # Create sample manifest
    manifest = EpisodeManifest(
        episode_type=EpisodeType.TRAINING,
        session_id="session_001"
    )
    
    # Add agents
    red_agent = AgentConfig(
        agent_id="red_001",
        agent_type="red",
        model_name="qwen3-235b-a22b",
        model_version="1.0.0",
        hyperparameters={"temperature": 0.7, "max_tokens": 1024}
    )
    
    blue_agent = AgentConfig(
        agent_id="blue_001",
        agent_type="blue",
        model_name="claude-3-sonnet",
        model_version="1.0.0",
        policy_version="v2.1"
    )
    
    manifest.agents = [red_agent, blue_agent]
    
    # Add environment
    manifest.environment = EnvironmentConfig(
        environment_id="meridian_dynamics_v1",
        organization_profile="tech_consulting_50_employees",
        compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.GDPR],
        difficulty_level="medium"
    )
    
    # Add metadata
    manifest.metadata = EpisodeMetadata(
        experiment_name="curriculum_tier_2_evaluation",
        curriculum_tier="T2",
        behavioral_descriptors={"stealth": 0.8, "persistence": 0.6}
    )
    
    # Test serialization
    json_str = manifest.to_json()
    print(f"Manifest JSON size: {len(json_str)} characters")
    
    # Test deserialization
    restored_manifest = EpisodeManifest.from_json(json_str)
    print(f"Restored episode ID: {restored_manifest.episode_id}")
    
    # Test file operations
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "test_manifest.json.gz"
        manifest.save_to_file(str(manifest_path))
        
        loaded_manifest = EpisodeManifest.load_from_file(str(manifest_path))
        print(f"Loaded manifest episode type: {loaded_manifest.episode_type.value}")
        
        # Test manifest manager
        manager = ManifestManager(temp_dir)
        stored_path = manager.store_manifest(manifest)
        print(f"Stored manifest at: {stored_path}")
        
        episodes = manager.list_episodes()
        print(f"Episodes in index: {len(episodes)}")
    
    print("Manifest schema test completed successfully!")