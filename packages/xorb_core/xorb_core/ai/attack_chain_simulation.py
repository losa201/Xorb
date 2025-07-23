#!/usr/bin/env python3
"""
Attack Chain Simulation Engine for Xorb 2.0

This module implements sophisticated attack chain simulation capabilities using
generative models (GANs and VAEs) to create synthetic attack campaigns for training,
testing, and red team exercises while maintaining realistic attack patterns.
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
import random
import pickle
from pathlib import Path


class AttackPhase(Enum):
    """Phases of an attack chain"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class AttackComplexity(Enum):
    """Complexity levels for attack campaigns"""
    BASIC = "basic"           # Simple, linear attack chains
    INTERMEDIATE = "intermediate"  # Moderate branching and techniques
    ADVANCED = "advanced"     # Complex multi-path campaigns
    APT = "apt"              # Advanced persistent threat level


@dataclass
class AttackTechnique:
    """Represents a single attack technique"""
    technique_id: str
    name: str
    description: str
    phase: AttackPhase
    
    # MITRE ATT&CK mapping
    mitre_technique_id: str
    mitre_tactic: str
    
    # Execution parameters
    success_probability: float = 0.7
    detection_probability: float = 0.3
    execution_time_range: Tuple[int, int] = (60, 300)  # seconds
    
    # Requirements and effects
    prerequisites: List[str] = field(default_factory=list)
    capabilities_gained: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    
    # Resource requirements
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    stealth_level: float = 0.5  # 0=very noisy, 1=very stealthy
    
    # Simulation parameters
    simulation_complexity: float = 0.5
    realistic_variations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AttackChainNode:
    """Node in an attack chain representing a technique execution"""
    node_id: str
    technique: AttackTechnique
    timestamp: datetime
    
    # Execution results
    success: bool = False
    detected: bool = False
    execution_time: float = 0.0
    
    # Context
    target_system: str = ""
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Chain relationships
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    
    # Simulation metadata
    generated_by_model: str = ""
    generation_confidence: float = 0.0
    realism_score: float = 0.0


@dataclass
class SimulatedAttackCampaign:
    """Complete simulated attack campaign"""
    campaign_id: str
    campaign_name: str
    complexity_level: AttackComplexity
    
    # Campaign metadata
    attacker_profile: Dict[str, Any]
    target_environment: Dict[str, Any]
    campaign_objectives: List[str]
    
    # Attack chain
    attack_nodes: List[AttackChainNode] = field(default_factory=list)
    attack_sequence: List[str] = field(default_factory=list)  # Node IDs in order
    
    # Timeline
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    
    # Success metrics
    overall_success: bool = False
    phases_completed: List[AttackPhase] = field(default_factory=list)
    objectives_achieved: List[str] = field(default_factory=list)
    
    # Detection and response
    detected_nodes: List[str] = field(default_factory=list)
    detection_timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Simulation quality
    realism_score: float = 0.0
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    generation_method: str = ""


class AttackChainVAE(nn.Module):
    """Variational Autoencoder for generating attack chain sequences"""
    
    def __init__(self,
                 technique_vocab_size: int = 200,
                 max_sequence_length: int = 20,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 latent_dim: int = 64):
        super().__init__()
        
        self.technique_vocab_size = technique_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Embedding layers
        self.technique_embedding = nn.Embedding(technique_vocab_size, embedding_dim)
        self.phase_embedding = nn.Embedding(len(AttackPhase), embedding_dim // 2)
        self.complexity_embedding = nn.Embedding(len(AttackComplexity), embedding_dim // 2)
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            embedding_dim + embedding_dim // 2 + embedding_dim // 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Latent space
        self.mu_layer = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            latent_dim + embedding_dim,
            hidden_dim,
            batch_first=True
        )
        
        # Output layers
        self.technique_output = nn.Linear(hidden_dim, technique_vocab_size)
        self.phase_output = nn.Linear(hidden_dim, len(AttackPhase))
        self.success_output = nn.Linear(hidden_dim, 1)
        self.timing_output = nn.Linear(hidden_dim, 1)
    
    def encode(self, technique_seq, phase_seq, complexity_seq):
        """Encode attack sequence to latent space"""
        
        # Embed inputs
        tech_emb = self.technique_embedding(technique_seq)
        phase_emb = self.phase_embedding(phase_seq)
        complexity_emb = self.complexity_embedding(complexity_seq)
        
        # Combine embeddings
        combined_emb = torch.cat([tech_emb, phase_emb, complexity_emb], dim=-1)
        
        # Encode sequence
        _, (hidden, _) = self.encoder_lstm(combined_emb)
        
        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        # Generate latent parameters
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, max_length=None):
        """Decode latent vector to attack sequence"""
        
        if max_length is None:
            max_length = self.max_sequence_length
        
        batch_size = z.size(0)
        device = z.device
        
        # Initialize decoder
        hidden = None
        
        # Start token (technique 0)
        current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        outputs = []
        
        for _ in range(max_length):
            # Embed current technique
            tech_emb = self.technique_embedding(current_input)
            
            # Combine with latent vector
            decoder_input = torch.cat([z.unsqueeze(1), tech_emb], dim=-1)
            
            # Decode step
            output, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # Generate predictions
            technique_logits = self.technique_output(output)
            phase_logits = self.phase_output(output)
            success_prob = torch.sigmoid(self.success_output(output))
            timing = torch.relu(self.timing_output(output))
            
            outputs.append({
                'technique_logits': technique_logits,
                'phase_logits': phase_logits,
                'success_prob': success_prob,
                'timing': timing
            })
            
            # Next input (sampling or argmax)
            current_input = torch.argmax(technique_logits, dim=-1)
        
        return outputs
    
    def forward(self, technique_seq, phase_seq, complexity_seq):
        """Forward pass through VAE"""
        
        # Encode
        mu, logvar = self.encode(technique_seq, phase_seq, complexity_seq)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        outputs = self.decode(z)
        
        return outputs, mu, logvar


class AttackChainGAN(nn.Module):
    """Generative Adversarial Network for attack chain generation"""
    
    def __init__(self,
                 technique_vocab_size: int = 200,
                 max_sequence_length: int = 20,
                 noise_dim: int = 100,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.technique_vocab_size = technique_vocab_size
        self.max_sequence_length = max_sequence_length
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim


class Generator(nn.Module):
    """GAN Generator for attack sequences"""
    
    def __init__(self,
                 noise_dim: int = 100,
                 technique_vocab_size: int = 200,
                 max_sequence_length: int = 20,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.technique_vocab_size = technique_vocab_size
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.initial_projection = nn.Linear(noise_dim, hidden_dim)
        
        # LSTM generator
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.technique_output = nn.Linear(hidden_dim, technique_vocab_size)
        self.phase_output = nn.Linear(hidden_dim, len(AttackPhase))
        self.timing_output = nn.Linear(hidden_dim, 1)
    
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Project noise
        initial_hidden = self.initial_projection(noise).unsqueeze(1)
        
        # Generate sequence
        outputs = []
        hidden = None
        
        for _ in range(self.max_sequence_length):
            output, hidden = self.lstm(initial_hidden, hidden)
            
            technique_logits = self.technique_output(output)
            phase_logits = self.phase_output(output)
            timing = torch.relu(self.timing_output(output))
            
            outputs.append({
                'technique_logits': technique_logits,
                'phase_logits': phase_logits,
                'timing': timing
            })
            
            # Use output as next input (with some transformation)
            initial_hidden = output
        
        return outputs


class Discriminator(nn.Module):
    """GAN Discriminator for attack sequences"""
    
    def __init__(self,
                 technique_vocab_size: int = 200,
                 max_sequence_length: int = 20,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.technique_embedding = nn.Embedding(technique_vocab_size, embedding_dim)
        self.phase_embedding = nn.Embedding(len(AttackPhase), embedding_dim // 2)
        
        # Sequence encoder
        self.lstm = nn.LSTM(
            embedding_dim + embedding_dim // 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, technique_seq, phase_seq):
        # Embed sequences
        tech_emb = self.technique_embedding(technique_seq)
        phase_emb = self.phase_embedding(phase_seq)
        
        # Combine embeddings
        combined_emb = torch.cat([tech_emb, phase_emb], dim=-1)
        
        # Encode sequence
        _, (hidden, _) = self.lstm(combined_emb)
        
        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        # Classify
        output = self.classifier(hidden)
        
        return output


class AttackTechniqueLibrary:
    """Library of attack techniques for simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.techniques: Dict[str, AttackTechnique] = {}
        self.phase_techniques: Dict[AttackPhase, List[str]] = defaultdict(list)
        self._initialize_technique_library()
    
    def _initialize_technique_library(self):
        """Initialize the attack technique library"""
        
        # Reconnaissance techniques
        self._add_technique(AttackTechnique(
            technique_id="T1595.001",
            name="Active Scanning: Scanning IP Blocks",
            description="Scan IP blocks to identify live systems",
            phase=AttackPhase.RECONNAISSANCE,
            mitre_technique_id="T1595.001",
            mitre_tactic="Reconnaissance",
            success_probability=0.9,
            detection_probability=0.2,
            execution_time_range=(300, 1800),
            capabilities_gained=["network_map", "live_hosts"],
            stealth_level=0.6,
            simulation_complexity=0.3
        ))
        
        # Initial Access techniques
        self._add_technique(AttackTechnique(
            technique_id="T1566.001",
            name="Phishing: Spearphishing Attachment",
            description="Send malicious attachment via email",
            phase=AttackPhase.INITIAL_ACCESS,
            mitre_technique_id="T1566.001",
            mitre_tactic="Initial Access",
            success_probability=0.3,
            detection_probability=0.6,
            execution_time_range=(1800, 7200),
            prerequisites=["email_addresses"],
            capabilities_gained=["initial_foothold"],
            artifacts_created=["malicious_file", "email_logs"],
            stealth_level=0.4,
            simulation_complexity=0.6
        ))
        
        # Execution techniques
        self._add_technique(AttackTechnique(
            technique_id="T1059.001",
            name="Command and Scripting: PowerShell",
            description="Execute commands via PowerShell",
            phase=AttackPhase.EXECUTION,
            mitre_technique_id="T1059.001",
            mitre_tactic="Execution",
            success_probability=0.8,
            detection_probability=0.4,
            execution_time_range=(60, 300),
            prerequisites=["initial_foothold"],
            capabilities_gained=["command_execution"],
            artifacts_created=["powershell_logs", "command_history"],
            stealth_level=0.5,
            simulation_complexity=0.4
        ))
        
        # Persistence techniques
        self._add_technique(AttackTechnique(
            technique_id="T1547.001",
            name="Boot or Logon Autostart: Registry Run Keys",
            description="Establish persistence via registry run keys",
            phase=AttackPhase.PERSISTENCE,
            mitre_technique_id="T1547.001",
            mitre_tactic="Persistence",
            success_probability=0.7,
            detection_probability=0.5,
            execution_time_range=(120, 600),
            prerequisites=["command_execution"],
            capabilities_gained=["persistence"],
            artifacts_created=["registry_modification"],
            stealth_level=0.6,
            simulation_complexity=0.5
        ))
        
        # Privilege Escalation techniques
        self._add_technique(AttackTechnique(
            technique_id="T1055",
            name="Process Injection",
            description="Inject code into running processes",
            phase=AttackPhase.PRIVILEGE_ESCALATION,
            mitre_technique_id="T1055",
            mitre_tactic="Privilege Escalation",
            success_probability=0.6,
            detection_probability=0.7,
            execution_time_range=(180, 900),
            prerequisites=["command_execution"],
            capabilities_gained=["elevated_privileges"],
            artifacts_created=["injected_process"],
            stealth_level=0.3,
            simulation_complexity=0.8
        ))
        
        # Defense Evasion techniques
        self._add_technique(AttackTechnique(
            technique_id="T1562.001",
            name="Impair Defenses: Disable or Modify Tools",
            description="Disable security tools",
            phase=AttackPhase.DEFENSE_EVASION,
            mitre_technique_id="T1562.001",
            mitre_tactic="Defense Evasion",
            success_probability=0.5,
            detection_probability=0.8,
            execution_time_range=(300, 1200),
            prerequisites=["elevated_privileges"],
            capabilities_gained=["defense_evasion"],
            artifacts_created=["security_tool_modification"],
            stealth_level=0.2,
            simulation_complexity=0.7
        ))
        
        # Credential Access techniques
        self._add_technique(AttackTechnique(
            technique_id="T1003.001",
            name="OS Credential Dumping: LSASS Memory",
            description="Dump credentials from LSASS memory",
            phase=AttackPhase.CREDENTIAL_ACCESS,
            mitre_technique_id="T1003.001",
            mitre_tactic="Credential Access",
            success_probability=0.7,
            detection_probability=0.9,
            execution_time_range=(60, 300),
            prerequisites=["elevated_privileges"],
            capabilities_gained=["credentials"],
            artifacts_created=["credential_dump"],
            stealth_level=0.1,
            simulation_complexity=0.6
        ))
        
        # Discovery techniques
        self._add_technique(AttackTechnique(
            technique_id="T1057",
            name="Process Discovery",
            description="Discover running processes",
            phase=AttackPhase.DISCOVERY,
            mitre_technique_id="T1057",
            mitre_tactic="Discovery",
            success_probability=0.9,
            detection_probability=0.3,
            execution_time_range=(30, 120),
            prerequisites=["command_execution"],
            capabilities_gained=["process_list"],
            artifacts_created=["process_enumeration"],
            stealth_level=0.7,
            simulation_complexity=0.2
        ))
        
        # Lateral Movement techniques
        self._add_technique(AttackTechnique(
            technique_id="T1021.001",
            name="Remote Services: Remote Desktop Protocol",
            description="Move laterally using RDP",
            phase=AttackPhase.LATERAL_MOVEMENT,
            mitre_technique_id="T1021.001",
            mitre_tactic="Lateral Movement",
            success_probability=0.6,
            detection_probability=0.5,
            execution_time_range=(300, 1800),
            prerequisites=["credentials"],
            capabilities_gained=["lateral_access"],
            artifacts_created=["rdp_logs", "authentication_logs"],
            stealth_level=0.4,
            simulation_complexity=0.6
        ))
        
        # Collection techniques
        self._add_technique(AttackTechnique(
            technique_id="T1005",
            name="Data from Local System",
            description="Collect data from local system",
            phase=AttackPhase.COLLECTION,
            mitre_technique_id="T1005",
            mitre_tactic="Collection",
            success_probability=0.8,
            detection_probability=0.4,
            execution_time_range=(600, 3600),
            prerequisites=["command_execution"],
            capabilities_gained=["collected_data"],
            artifacts_created=["data_collection_logs"],
            stealth_level=0.5,
            simulation_complexity=0.4
        ))
        
        # Command and Control techniques
        self._add_technique(AttackTechnique(
            technique_id="T1071.001",
            name="Application Layer Protocol: Web Protocols",
            description="Establish C2 using web protocols",
            phase=AttackPhase.COMMAND_AND_CONTROL,
            mitre_technique_id="T1071.001",
            mitre_tactic="Command and Control",
            success_probability=0.7,
            detection_probability=0.3,
            execution_time_range=(180, 900),
            prerequisites=["initial_foothold"],
            capabilities_gained=["c2_channel"],
            artifacts_created=["network_traffic", "c2_communications"],
            stealth_level=0.6,
            simulation_complexity=0.5
        ))
        
        # Exfiltration techniques
        self._add_technique(AttackTechnique(
            technique_id="T1041",
            name="Exfiltration Over C2 Channel",
            description="Exfiltrate data over C2 channel",
            phase=AttackPhase.EXFILTRATION,
            mitre_technique_id="T1041",
            mitre_tactic="Exfiltration",
            success_probability=0.6,
            detection_probability=0.6,
            execution_time_range=(1800, 7200),
            prerequisites=["c2_channel", "collected_data"],
            capabilities_gained=["data_exfiltration"],
            artifacts_created=["exfiltration_traffic"],
            stealth_level=0.3,
            simulation_complexity=0.7
        ))
        
        # Impact techniques
        self._add_technique(AttackTechnique(
            technique_id="T1486",
            name="Data Encrypted for Impact",
            description="Encrypt data for impact/ransom",
            phase=AttackPhase.IMPACT,
            mitre_technique_id="T1486",
            mitre_tactic="Impact",
            success_probability=0.8,
            detection_probability=0.9,
            execution_time_range=(3600, 14400),
            prerequisites=["elevated_privileges"],
            capabilities_gained=["data_encryption"],
            artifacts_created=["encrypted_files", "ransom_note"],
            stealth_level=0.1,
            simulation_complexity=0.8
        ))
    
    def _add_technique(self, technique: AttackTechnique):
        """Add a technique to the library"""
        self.techniques[technique.technique_id] = technique
        self.phase_techniques[technique.phase].append(technique.technique_id)
    
    def get_technique(self, technique_id: str) -> Optional[AttackTechnique]:
        """Get a technique by ID"""
        return self.techniques.get(technique_id)
    
    def get_techniques_by_phase(self, phase: AttackPhase) -> List[AttackTechnique]:
        """Get all techniques for a specific phase"""
        technique_ids = self.phase_techniques.get(phase, [])
        return [self.techniques[tid] for tid in technique_ids]
    
    def get_available_techniques(self, capabilities: List[str]) -> List[AttackTechnique]:
        """Get techniques available given current capabilities"""
        available = []
        
        for technique in self.techniques.values():
            # Check if all prerequisites are met
            if all(prereq in capabilities for prereq in technique.prerequisites):
                available.append(technique)
        
        return available


class AttackChainSimulationEngine:
    """Main engine for simulating attack chains"""
    
    def __init__(self,
                 model_type: str = "vae",  # "vae" or "gan"
                 device: str = "auto"):
        
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.technique_library = AttackTechniqueLibrary()
        self.model_type = model_type
        
        # Initialize generative models
        if model_type == "vae":
            self.vae_model = AttackChainVAE().to(self.device)
        else:
            self.generator = Generator().to(self.device)
            self.discriminator = Discriminator().to(self.device)
        
        # Simulation state
        self.generated_campaigns: List[SimulatedAttackCampaign] = []
        self.technique_vocabulary = self._build_technique_vocabulary()
        
        # Performance metrics
        self.simulation_metrics = {
            'total_campaigns_generated': 0,
            'avg_campaign_length': 0.0,
            'avg_realism_score': 0.0,
            'technique_usage_frequency': defaultdict(int),
            'phase_transition_patterns': defaultdict(int),
            'generation_time_avg': 0.0
        }
        
        self.logger.info(f"Attack chain simulation engine initialized with {model_type} model on {self.device}")
    
    def _build_technique_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping from technique IDs to indices"""
        
        vocab = {}
        for i, technique_id in enumerate(self.technique_library.techniques.keys()):
            vocab[technique_id] = i
        
        return vocab
    
    async def generate_attack_campaign(self,
                                     campaign_config: Dict[str, Any]) -> SimulatedAttackCampaign:
        """Generate a complete attack campaign"""
        
        generation_start = datetime.utcnow()
        
        # Extract configuration
        complexity = AttackComplexity(campaign_config.get('complexity', 'intermediate'))
        attacker_profile = campaign_config.get('attacker_profile', {})
        target_environment = campaign_config.get('target_environment', {})
        objectives = campaign_config.get('objectives', ['data_access'])
        
        # Generate campaign ID
        campaign_id = f"sim_campaign_{generation_start.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Generate attack chain based on model type
        if self.model_type == "vae":
            attack_nodes = await self._generate_chain_with_vae(complexity, objectives)
        else:
            attack_nodes = await self._generate_chain_with_gan(complexity, objectives)
        
        # Build attack sequence
        attack_sequence = [node.node_id for node in attack_nodes]
        
        # Calculate campaign metrics
        phases_completed = list(set(node.technique.phase for node in attack_nodes))
        overall_success = await self._calculate_campaign_success(attack_nodes, objectives)
        objectives_achieved = await self._determine_achieved_objectives(attack_nodes, objectives)
        
        # Simulate detection and response
        detected_nodes, detection_timeline, response_actions = await self._simulate_detection_and_response(attack_nodes)
        
        # Calculate quality scores
        realism_score = await self._calculate_realism_score(attack_nodes)
        diversity_score = await self._calculate_diversity_score(attack_nodes)
        novelty_score = await self._calculate_novelty_score(attack_nodes)
        
        # Create campaign
        campaign = SimulatedAttackCampaign(
            campaign_id=campaign_id,
            campaign_name=f"Simulated {complexity.value.title()} Campaign",
            complexity_level=complexity,
            attacker_profile=attacker_profile,
            target_environment=target_environment,
            campaign_objectives=objectives,
            attack_nodes=attack_nodes,
            attack_sequence=attack_sequence,
            start_time=generation_start,
            end_time=attack_nodes[-1].timestamp if attack_nodes else generation_start,
            total_duration=(attack_nodes[-1].timestamp - generation_start).total_seconds() if attack_nodes else 0,
            overall_success=overall_success,
            phases_completed=phases_completed,
            objectives_achieved=objectives_achieved,
            detected_nodes=detected_nodes,
            detection_timeline=detection_timeline,
            response_actions=response_actions,
            realism_score=realism_score,
            diversity_score=diversity_score,
            novelty_score=novelty_score,
            generation_method=self.model_type
        )
        
        # Store campaign
        self.generated_campaigns.append(campaign)
        
        # Update metrics
        await self._update_simulation_metrics(campaign)
        
        generation_time = (datetime.utcnow() - generation_start).total_seconds()
        self.logger.info(f"Generated attack campaign {campaign_id}: "
                        f"{len(attack_nodes)} techniques, "
                        f"realism={realism_score:.2f}, "
                        f"time={generation_time:.2f}s")
        
        return campaign
    
    async def _generate_chain_with_vae(self,
                                     complexity: AttackComplexity,
                                     objectives: List[str]) -> List[AttackChainNode]:
        """Generate attack chain using VAE model"""
        
        # For this example, we'll use a rule-based approach
        # In production, this would use the trained VAE model
        
        nodes = []
        current_capabilities = []
        current_time = datetime.utcnow()
        
        # Phase sequence based on complexity
        if complexity == AttackComplexity.BASIC:
            target_phases = [
                AttackPhase.RECONNAISSANCE,
                AttackPhase.INITIAL_ACCESS,
                AttackPhase.EXECUTION,
                AttackPhase.COLLECTION
            ]
        elif complexity == AttackComplexity.INTERMEDIATE:
            target_phases = [
                AttackPhase.RECONNAISSANCE,
                AttackPhase.INITIAL_ACCESS,
                AttackPhase.EXECUTION,
                AttackPhase.PERSISTENCE,
                AttackPhase.DISCOVERY,
                AttackPhase.COLLECTION,
                AttackPhase.EXFILTRATION
            ]
        elif complexity == AttackComplexity.ADVANCED:
            target_phases = [
                AttackPhase.RECONNAISSANCE,
                AttackPhase.INITIAL_ACCESS,
                AttackPhase.EXECUTION,
                AttackPhase.PERSISTENCE,
                AttackPhase.PRIVILEGE_ESCALATION,
                AttackPhase.DEFENSE_EVASION,
                AttackPhase.DISCOVERY,
                AttackPhase.LATERAL_MOVEMENT,
                AttackPhase.COLLECTION,
                AttackPhase.COMMAND_AND_CONTROL,
                AttackPhase.EXFILTRATION
            ]
        else:  # APT
            target_phases = list(AttackPhase)
        
        # Generate techniques for each phase
        for phase in target_phases:
            available_techniques = self.technique_library.get_techniques_by_phase(phase)
            
            # Filter by prerequisites
            viable_techniques = [
                tech for tech in available_techniques
                if all(prereq in current_capabilities for prereq in tech.prerequisites)
            ]
            
            if not viable_techniques:
                # Skip phase if no viable techniques
                continue
            
            # Select technique (with some randomness for variety)
            selected_technique = random.choice(viable_techniques)
            
            # Create attack node
            node_id = f"node_{len(nodes)+1:03d}"
            
            # Simulate execution
            execution_time = random.uniform(*selected_technique.execution_time_range)
            success = random.random() < selected_technique.success_probability
            detected = random.random() < selected_technique.detection_probability
            
            node = AttackChainNode(
                node_id=node_id,
                technique=selected_technique,
                timestamp=current_time,
                success=success,
                detected=detected,
                execution_time=execution_time,
                target_system=f"target_{random.randint(1, 5)}",
                execution_context={'phase': phase.value, 'objective_driven': True},
                parent_nodes=[nodes[-1].node_id] if nodes else [],
                generated_by_model="vae_simulation",
                generation_confidence=0.8,
                realism_score=0.7 + random.uniform(-0.1, 0.1)
            )
            
            nodes.append(node)
            
            # Update capabilities if successful
            if success:
                current_capabilities.extend(selected_technique.capabilities_gained)
            
            # Advance time
            current_time += timedelta(seconds=execution_time)
            
            # Add some random delay between techniques
            current_time += timedelta(seconds=random.uniform(60, 600))
        
        return nodes
    
    async def _generate_chain_with_gan(self,
                                     complexity: AttackComplexity,
                                     objectives: List[str]) -> List[AttackChainNode]:
        """Generate attack chain using GAN model"""
        
        # Similar to VAE but with different generation logic
        # This would use the trained GAN generator in production
        
        return await self._generate_chain_with_vae(complexity, objectives)
    
    async def _calculate_campaign_success(self,
                                        attack_nodes: List[AttackChainNode],
                                        objectives: List[str]) -> bool:
        """Calculate overall campaign success"""
        
        # Success depends on achieving objectives and technique success
        successful_nodes = [node for node in attack_nodes if node.success]
        success_rate = len(successful_nodes) / len(attack_nodes) if attack_nodes else 0
        
        # Check if critical phases were successful
        critical_phases = [AttackPhase.INITIAL_ACCESS, AttackPhase.EXECUTION]
        critical_success = any(
            node.technique.phase in critical_phases and node.success
            for node in attack_nodes
        )
        
        return success_rate > 0.5 and critical_success
    
    async def _determine_achieved_objectives(self,
                                           attack_nodes: List[AttackChainNode],
                                           objectives: List[str]) -> List[str]:
        """Determine which objectives were achieved"""
        
        achieved = []
        
        # Map objectives to required capabilities
        objective_requirements = {
            'data_access': ['initial_foothold', 'collected_data'],
            'data_exfiltration': ['collected_data', 'data_exfiltration'],
            'system_disruption': ['elevated_privileges', 'data_encryption'],
            'persistence': ['persistence'],
            'credential_theft': ['credentials']
        }
        
        # Gather all capabilities gained
        all_capabilities = []
        for node in attack_nodes:
            if node.success:
                all_capabilities.extend(node.technique.capabilities_gained)
        
        # Check each objective
        for objective in objectives:
            requirements = objective_requirements.get(objective, [])
            if all(req in all_capabilities for req in requirements):
                achieved.append(objective)
        
        return achieved
    
    async def _simulate_detection_and_response(self,
                                             attack_nodes: List[AttackChainNode]) -> Tuple[List[str], List[Tuple[datetime, str]], List[Dict[str, Any]]]:
        """Simulate detection and incident response"""
        
        detected_nodes = []
        detection_timeline = []
        response_actions = []
        
        for node in attack_nodes:
            if node.detected:
                detected_nodes.append(node.node_id)
                detection_timeline.append((node.timestamp, node.technique.name))
                
                # Simulate response actions
                if node.technique.phase in [AttackPhase.INITIAL_ACCESS, AttackPhase.PERSISTENCE]:
                    response_actions.append({
                        'timestamp': node.timestamp + timedelta(minutes=random.randint(5, 30)),
                        'action': 'isolate_host',
                        'target': node.target_system,
                        'success': random.random() > 0.3
                    })
                
                if node.technique.detection_probability > 0.7:
                    response_actions.append({
                        'timestamp': node.timestamp + timedelta(minutes=random.randint(1, 10)),
                        'action': 'alert_analyst',
                        'technique': node.technique.name,
                        'success': True
                    })
        
        return detected_nodes, detection_timeline, response_actions
    
    async def _calculate_realism_score(self, attack_nodes: List[AttackChainNode]) -> float:
        """Calculate realism score for the attack chain"""
        
        if not attack_nodes:
            return 0.0
        
        scores = []
        
        # Phase progression realism
        phases = [node.technique.phase for node in attack_nodes]
        phase_order_score = self._evaluate_phase_order(phases)
        scores.append(phase_order_score)
        
        # Prerequisite satisfaction
        prereq_score = self._evaluate_prerequisite_satisfaction(attack_nodes)
        scores.append(prereq_score)
        
        # Timing realism
        timing_score = self._evaluate_timing_realism(attack_nodes)
        scores.append(timing_score)
        
        # Success rate realism
        success_score = self._evaluate_success_realism(attack_nodes)
        scores.append(success_score)
        
        return np.mean(scores)
    
    def _evaluate_phase_order(self, phases: List[AttackPhase]) -> float:
        """Evaluate the realism of phase progression"""
        
        # Define typical phase orders
        typical_orders = [
            [AttackPhase.RECONNAISSANCE, AttackPhase.INITIAL_ACCESS],
            [AttackPhase.INITIAL_ACCESS, AttackPhase.EXECUTION],
            [AttackPhase.EXECUTION, AttackPhase.PERSISTENCE],
            [AttackPhase.PERSISTENCE, AttackPhase.PRIVILEGE_ESCALATION],
            [AttackPhase.PRIVILEGE_ESCALATION, AttackPhase.DISCOVERY],
            [AttackPhase.DISCOVERY, AttackPhase.LATERAL_MOVEMENT],
            [AttackPhase.LATERAL_MOVEMENT, AttackPhase.COLLECTION],
            [AttackPhase.COLLECTION, AttackPhase.EXFILTRATION]
        ]
        
        score = 0.0
        for i in range(len(phases) - 1):
            current_phase = phases[i]
            next_phase = phases[i + 1]
            
            # Check if this transition is typical
            if [current_phase, next_phase] in typical_orders:
                score += 1.0
            elif current_phase == next_phase:
                score += 0.8  # Same phase is ok
            else:
                score += 0.3  # Unusual but not impossible
        
        return score / max(1, len(phases) - 1)
    
    def _evaluate_prerequisite_satisfaction(self, attack_nodes: List[AttackChainNode]) -> float:
        """Evaluate prerequisite satisfaction"""
        
        capabilities = []
        violations = 0
        
        for node in attack_nodes:
            # Check if prerequisites are met
            for prereq in node.technique.prerequisites:
                if prereq not in capabilities:
                    violations += 1
            
            # Add capabilities if technique succeeded
            if node.success:
                capabilities.extend(node.technique.capabilities_gained)
        
        total_prereqs = sum(len(node.technique.prerequisites) for node in attack_nodes)
        
        if total_prereqs == 0:
            return 1.0
        
        return 1.0 - (violations / total_prereqs)
    
    def _evaluate_timing_realism(self, attack_nodes: List[AttackChainNode]) -> float:
        """Evaluate timing realism"""
        
        if len(attack_nodes) < 2:
            return 1.0
        
        time_gaps = []
        for i in range(1, len(attack_nodes)):
            gap = (attack_nodes[i].timestamp - attack_nodes[i-1].timestamp).total_seconds()
            time_gaps.append(gap)
        
        # Evaluate if gaps are realistic (not too fast, not too slow)
        realistic_gaps = 0
        for gap in time_gaps:
            if 60 <= gap <= 86400:  # Between 1 minute and 1 day
                realistic_gaps += 1
            elif gap < 60:
                realistic_gaps += 0.3  # Too fast
            else:
                realistic_gaps += 0.7  # Slow but possible
        
        return realistic_gaps / len(time_gaps)
    
    def _evaluate_success_realism(self, attack_nodes: List[AttackChainNode]) -> float:
        """Evaluate success rate realism"""
        
        if not attack_nodes:
            return 1.0
        
        success_rate = sum(1 for node in attack_nodes if node.success) / len(attack_nodes)
        
        # Realistic success rates are typically between 0.3 and 0.8
        if 0.3 <= success_rate <= 0.8:
            return 1.0
        elif success_rate < 0.3:
            return success_rate / 0.3  # Linear penalty for too low
        else:
            return (1.0 - success_rate) / 0.2  # Linear penalty for too high
    
    async def _calculate_diversity_score(self, attack_nodes: List[AttackChainNode]) -> float:
        """Calculate diversity score for the attack chain"""
        
        if not attack_nodes:
            return 0.0
        
        # Phase diversity
        unique_phases = len(set(node.technique.phase for node in attack_nodes))
        total_phases = len(AttackPhase)
        phase_diversity = unique_phases / total_phases
        
        # Technique diversity
        unique_techniques = len(set(node.technique.technique_id for node in attack_nodes))
        technique_diversity = unique_techniques / len(attack_nodes)
        
        # Timing diversity (variation in execution times)
        execution_times = [node.execution_time for node in attack_nodes]
        if len(execution_times) > 1:
            timing_diversity = np.std(execution_times) / np.mean(execution_times)
            timing_diversity = min(1.0, timing_diversity)  # Cap at 1.0
        else:
            timing_diversity = 0.0
        
        return np.mean([phase_diversity, technique_diversity, timing_diversity])
    
    async def _calculate_novelty_score(self, attack_nodes: List[AttackChainNode]) -> float:
        """Calculate novelty score compared to historical campaigns"""
        
        if not self.generated_campaigns:
            return 1.0  # First campaign is completely novel
        
        # Compare technique sequences
        current_sequence = [node.technique.technique_id for node in attack_nodes]
        
        similarities = []
        
        for campaign in self.generated_campaigns[-10:]:  # Compare with last 10 campaigns
            historical_sequence = [node.technique.technique_id for node in campaign.attack_nodes]
            
            # Calculate sequence similarity
            similarity = self._calculate_sequence_similarity(current_sequence, historical_sequence)
            similarities.append(similarity)
        
        if similarities:
            max_similarity = max(similarities)
            novelty = 1.0 - max_similarity
        else:
            novelty = 1.0
        
        return novelty
    
    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two technique sequences"""
        
        if not seq1 or not seq2:
            return 0.0
        
        # Use Jaccard similarity on technique sets
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also consider sequence order (simplified)
        order_similarity = 0.0
        min_len = min(len(seq1), len(seq2))
        
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
            order_similarity = matches / min_len
        
        # Combine Jaccard and order similarities
        return 0.7 * jaccard_similarity + 0.3 * order_similarity
    
    async def _update_simulation_metrics(self, campaign: SimulatedAttackCampaign):
        """Update simulation metrics with new campaign data"""
        
        self.simulation_metrics['total_campaigns_generated'] += 1
        
        # Update average campaign length
        total_campaigns = self.simulation_metrics['total_campaigns_generated']
        current_avg_length = self.simulation_metrics['avg_campaign_length']
        campaign_length = len(campaign.attack_nodes)
        new_avg_length = ((current_avg_length * (total_campaigns - 1)) + campaign_length) / total_campaigns
        self.simulation_metrics['avg_campaign_length'] = new_avg_length
        
        # Update average realism score
        current_avg_realism = self.simulation_metrics['avg_realism_score']
        new_avg_realism = ((current_avg_realism * (total_campaigns - 1)) + campaign.realism_score) / total_campaigns
        self.simulation_metrics['avg_realism_score'] = new_avg_realism
        
        # Update technique usage frequency
        for node in campaign.attack_nodes:
            self.simulation_metrics['technique_usage_frequency'][node.technique.technique_id] += 1
        
        # Update phase transition patterns
        for i in range(len(campaign.attack_nodes) - 1):
            current_phase = campaign.attack_nodes[i].technique.phase.value
            next_phase = campaign.attack_nodes[i + 1].technique.phase.value
            transition = f"{current_phase}->{next_phase}"
            self.simulation_metrics['phase_transition_patterns'][transition] += 1
    
    async def export_campaign_to_json(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Export a campaign to JSON format"""
        
        campaign = next((c for c in self.generated_campaigns if c.campaign_id == campaign_id), None)
        
        if not campaign:
            return None
        
        # Convert campaign to JSON-serializable format
        campaign_dict = {
            'campaign_id': campaign.campaign_id,
            'campaign_name': campaign.campaign_name,
            'complexity_level': campaign.complexity_level.value,
            'attacker_profile': campaign.attacker_profile,
            'target_environment': campaign.target_environment,
            'campaign_objectives': campaign.campaign_objectives,
            'start_time': campaign.start_time.isoformat(),
            'end_time': campaign.end_time.isoformat() if campaign.end_time else None,
            'total_duration': campaign.total_duration,
            'overall_success': campaign.overall_success,
            'phases_completed': [phase.value for phase in campaign.phases_completed],
            'objectives_achieved': campaign.objectives_achieved,
            'realism_score': campaign.realism_score,
            'diversity_score': campaign.diversity_score,
            'novelty_score': campaign.novelty_score,
            'generation_method': campaign.generation_method,
            'attack_nodes': [],
            'detection_timeline': [(dt.isoformat(), desc) for dt, desc in campaign.detection_timeline],
            'response_actions': campaign.response_actions
        }
        
        # Add attack nodes
        for node in campaign.attack_nodes:
            node_dict = {
                'node_id': node.node_id,
                'technique_id': node.technique.technique_id,
                'technique_name': node.technique.name,
                'phase': node.technique.phase.value,
                'timestamp': node.timestamp.isoformat(),
                'success': node.success,
                'detected': node.detected,
                'execution_time': node.execution_time,
                'target_system': node.target_system,
                'execution_context': node.execution_context,
                'parent_nodes': node.parent_nodes,
                'child_nodes': node.child_nodes,
                'realism_score': node.realism_score
            }
            campaign_dict['attack_nodes'].append(node_dict)
        
        return campaign_dict
    
    async def get_simulation_status(self) -> Dict[str, Any]:
        """Get current status of the simulation engine"""
        
        return {
            'engine_active': True,
            'model_type': self.model_type,
            'device': str(self.device),
            'total_campaigns_generated': len(self.generated_campaigns),
            'technique_library_size': len(self.technique_library.techniques),
            'metrics': dict(self.simulation_metrics),
            'recent_campaigns': [
                {
                    'campaign_id': c.campaign_id,
                    'complexity': c.complexity_level.value,
                    'realism_score': c.realism_score,
                    'chain_length': len(c.attack_nodes)
                }
                for c in self.generated_campaigns[-5:]  # Last 5 campaigns
            ]
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        simulation_engine = AttackChainSimulationEngine(model_type="vae")
        
        # Configure campaign generation
        campaign_config = {
            'complexity': 'intermediate',
            'attacker_profile': {
                'skill_level': 'intermediate',
                'motivation': 'financial',
                'resources': 'moderate'
            },
            'target_environment': {
                'size': 'medium',
                'security_posture': 'standard',
                'industry': 'financial'
            },
            'objectives': ['data_access', 'data_exfiltration']
        }
        
        # Generate attack campaign
        campaign = await simulation_engine.generate_attack_campaign(campaign_config)
        
        print(f"Generated Attack Campaign:")
        print(f"  Campaign ID: {campaign.campaign_id}")
        print(f"  Complexity: {campaign.complexity_level.value}")
        print(f"  Chain Length: {len(campaign.attack_nodes)}")
        print(f"  Overall Success: {campaign.overall_success}")
        print(f"  Realism Score: {campaign.realism_score:.2f}")
        print(f"  Diversity Score: {campaign.diversity_score:.2f}")
        print(f"  Novelty Score: {campaign.novelty_score:.2f}")
        print(f"  Phases Completed: {[p.value for p in campaign.phases_completed]}")
        print(f"  Objectives Achieved: {campaign.objectives_achieved}")
        print(f"  Detected Nodes: {len(campaign.detected_nodes)}")
        
        print(f"\nAttack Chain:")
        for i, node in enumerate(campaign.attack_nodes, 1):
            status = "✓" if node.success else "✗"
            detected = "🔍" if node.detected else "  "
            print(f"  {i:2d}. {status} {detected} {node.technique.name} ({node.technique.phase.value})")
        
        # Export to JSON
        campaign_json = await simulation_engine.export_campaign_to_json(campaign.campaign_id)
        print(f"\nExported campaign size: {len(json.dumps(campaign_json, indent=2))} characters")
        
        # Get engine status
        status = await simulation_engine.get_simulation_status()
        print(f"\nSimulation Engine Status:")
        print(f"  Total Campaigns: {status['total_campaigns_generated']}")
        print(f"  Avg Campaign Length: {status['metrics']['avg_campaign_length']:.1f}")
        print(f"  Avg Realism Score: {status['metrics']['avg_realism_score']:.2f}")
    
    asyncio.run(main())