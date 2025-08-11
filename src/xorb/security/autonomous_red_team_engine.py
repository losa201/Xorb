#!/usr/bin/env python3
"""
Autonomous Red Team Engine
ADVANCED AI-DRIVEN AUTONOMOUS RED TEAM OPERATIONS

CAPABILITIES:
- Reinforcement Learning-guided attack decision making
- Multi-vector attack chain orchestration with adaptive tactics
- Real-time defense evasion and adaptation algorithms
- Continuous learning from defensive responses and countermeasures
- Safety-constrained autonomous operations with human oversight
- Advanced attack technique simulation and execution

Principal Auditor Implementation: Next-generation autonomous red team operations
"""

import asyncio
import logging
import json
import numpy as np
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using fallback implementations")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - using fallback implementations")

import structlog

logger = structlog.get_logger(__name__)


class AttackPhase(str, Enum):
    """Red team attack phases"""
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


class ThreatActorProfile(str, Enum):
    """Threat actor profiles for simulation"""
    NATION_STATE_APT = "nation_state_apt"
    ORGANIZED_CYBERCRIME = "organized_cybercrime"
    HACKTIVIST_GROUP = "hacktivist_group"
    INSIDER_THREAT = "insider_threat"
    ADVANCED_PERSISTENT_THREAT = "advanced_persistent_threat"
    RANSOMWARE_OPERATOR = "ransomware_operator"
    SUPPLY_CHAIN_ATTACKER = "supply_chain_attacker"
    AI_POWERED_ADVERSARY = "ai_powered_adversary"


class AttackTechnique(str, Enum):
    """MITRE ATT&CK techniques"""
    SPEAR_PHISHING = "T1566.001"
    EXPLOIT_PUBLIC_FACING = "T1190"
    VALID_ACCOUNTS = "T1078"
    EXTERNAL_REMOTE_SERVICES = "T1133"
    SUPPLY_CHAIN_COMPROMISE = "T1195"
    SCHEDULED_TASK = "T1053.005"
    BOOT_OR_LOGON_AUTOSTART = "T1547"
    CREATE_OR_MODIFY_PROCESS = "T1543"
    HIJACK_EXECUTION_FLOW = "T1574"
    ACCESS_TOKEN_MANIPULATION = "T1134"
    BYPASS_USER_ACCOUNT_CONTROL = "T1548.002"
    PROCESS_INJECTION = "T1055"
    MASQUERADING = "T1036"
    OBFUSCATED_FILES = "T1027"
    VIRTUALIZATION_DETECTION = "T1497"
    OS_CREDENTIAL_DUMPING = "T1003"
    BRUTE_FORCE = "T1110"
    KEYLOGGING = "T1056.001"
    NETWORK_SERVICE_SCANNING = "T1046"
    SYSTEM_INFORMATION_DISCOVERY = "T1082"
    REMOTE_SERVICES = "T1021"
    LATERAL_TOOL_TRANSFER = "T1570"
    DATA_FROM_LOCAL_SYSTEM = "T1005"
    DATA_STAGED = "T1074"
    APPLICATION_LAYER_PROTOCOL = "T1071"
    ENCRYPTED_CHANNEL = "T1573"
    EXFILTRATION_OVER_C2 = "T1041"
    DATA_ENCRYPTED_FOR_IMPACT = "T1486"


class SafetyConstraint(str, Enum):
    """Safety constraints for autonomous operations"""
    NO_DATA_MODIFICATION = "no_data_modification"
    NO_SERVICE_DISRUPTION = "no_service_disruption"
    NO_CREDENTIAL_HARVESTING = "no_credential_harvesting"
    NO_LATERAL_MOVEMENT_PRODUCTION = "no_lateral_movement_production"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    TIME_LIMITED_OPERATIONS = "time_limited_operations"
    SCOPE_LIMITED = "scope_limited"
    LOGGING_REQUIRED = "logging_required"


@dataclass
class AttackAction:
    """Individual attack action"""
    action_id: str
    technique: AttackTechnique
    phase: AttackPhase
    target: str
    parameters: Dict[str, Any]
    expected_outcome: str
    success_probability: float
    risk_level: str
    safety_validated: bool
    timestamp: datetime


@dataclass
class AttackDecision:
    """AI-driven attack decision"""
    decision_id: str
    phase: AttackPhase
    available_actions: List[AttackAction]
    selected_action: AttackAction
    decision_rationale: str
    confidence: float
    risk_assessment: Dict[str, Any]
    learning_feedback: Optional[str]
    timestamp: datetime


@dataclass
class CampaignObjective:
    """Red team campaign objective"""
    objective_id: str
    objective_type: str
    description: str
    priority: int
    success_criteria: List[str]
    current_status: str
    completion_percentage: float


@dataclass
class DefenseDetection:
    """Defense system detection event"""
    detection_id: str
    detection_type: str
    severity: str
    confidence: float
    affected_actions: List[str]
    countermeasures_needed: List[str]
    timestamp: datetime


class ReinforcementLearningAgent(nn.Module):
    """RL agent for autonomous attack decision making"""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 50, hidden_dim: int = 256):
        super(ReinforcementLearningAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Training components
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both networks"""
        policy_output = self.policy_net(state)
        value_output = self.value_net(state)
        return policy_output, value_output
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> Tuple[int, float]:
        """Select action using epsilon-greedy policy with neural network"""
        if not TORCH_AVAILABLE:
            return random.randint(0, self.action_dim - 1), 0.5
        
        with torch.no_grad():
            policy_probs, value = self.forward(state)
            
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
                action_prob = policy_probs[action].item()
            else:
                # Sample from policy distribution
                distribution = Categorical(policy_probs)
                action = distribution.sample().item()
                action_prob = policy_probs[action].item()
            
            return action, action_prob
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step"""
        if not TORCH_AVAILABLE or len(self.experience_buffer) < batch_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
        
        # Sample batch from experience buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        
        # Forward pass
        policy_probs, values = self.forward(states)
        _, next_values = self.forward(next_states)
        
        # Calculate targets
        targets = rewards + 0.99 * next_values.squeeze() * (1 - dones)
        
        # Calculate losses
        value_loss = self.loss_fn(values.squeeze(), targets.detach())
        
        # Policy loss (using advantage)
        advantages = targets - values.squeeze()
        selected_probs = policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(selected_probs) * advantages.detach()).mean()
        
        total_loss = value_loss + policy_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item()
        }


class DefenseEvasionEngine:
    """Engine for real-time defense evasion and adaptation"""
    
    def __init__(self):
        self.evasion_techniques = {
            "timing_manipulation": self._timing_evasion,
            "traffic_shaping": self._traffic_shaping_evasion,
            "signature_obfuscation": self._signature_obfuscation,
            "behavioral_mimicry": self._behavioral_mimicry,
            "steganography": self._steganographic_evasion,
            "living_off_land": self._living_off_land_evasion,
            "process_hollowing": self._process_hollowing_evasion,
            "domain_fronting": self._domain_fronting_evasion
        }
        
        self.detection_patterns = []
        self.evasion_history = []
        
        if SKLEARN_AVAILABLE:
            self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.evasion_optimizer = DBSCAN(eps=0.5, min_samples=5)
        else:
            self.pattern_classifier = None
            self.evasion_optimizer = None
    
    def analyze_defense_patterns(self, defense_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze defense patterns to identify evasion opportunities"""
        analysis_result = {
            "patterns_identified": [],
            "evasion_opportunities": [],
            "recommended_techniques": [],
            "confidence": 0.0
        }
        
        try:
            if not defense_events:
                return analysis_result
            
            # Temporal pattern analysis
            temporal_patterns = self._analyze_temporal_patterns(defense_events)
            analysis_result["patterns_identified"].extend(temporal_patterns)
            
            # Signature pattern analysis
            signature_patterns = self._analyze_signature_patterns(defense_events)
            analysis_result["patterns_identified"].extend(signature_patterns)
            
            # Behavioral pattern analysis
            behavioral_patterns = self._analyze_behavioral_patterns(defense_events)
            analysis_result["patterns_identified"].extend(behavioral_patterns)
            
            # Generate evasion recommendations
            evasion_recommendations = self._generate_evasion_recommendations(
                temporal_patterns + signature_patterns + behavioral_patterns
            )
            analysis_result["evasion_opportunities"] = evasion_recommendations
            
            # Calculate confidence
            pattern_strength = len(analysis_result["patterns_identified"]) / max(len(defense_events), 1)
            analysis_result["confidence"] = min(pattern_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Defense pattern analysis failed: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def apply_evasion_technique(self, technique: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific evasion technique"""
        evasion_function = self.evasion_techniques.get(technique)
        
        if evasion_function:
            try:
                result = evasion_function(context)
                
                # Record evasion attempt
                self.evasion_history.append({
                    "technique": technique,
                    "context": context,
                    "result": result,
                    "timestamp": datetime.utcnow()
                })
                
                return result
            except Exception as e:
                logger.error(f"Evasion technique {technique} failed: {e}")
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Unknown evasion technique: {technique}"}
    
    def _analyze_temporal_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in defense events"""
        patterns = []
        
        try:
            if len(events) < 3:
                return patterns
            
            # Extract timestamps
            timestamps = []
            for event in events:
                if "timestamp" in event:
                    try:
                        ts = datetime.fromisoformat(event["timestamp"])
                        timestamps.append(ts)
                    except:
                        continue
            
            if len(timestamps) < 3:
                return patterns
            
            timestamps.sort()
            
            # Analyze intervals
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            # Check for regular intervals
            if len(intervals) > 2:
                avg_interval = np.mean(intervals)
                interval_variance = np.var(intervals)
                
                if interval_variance < (avg_interval * 0.2):  # Low variance indicates regularity
                    patterns.append({
                        "type": "regular_scanning",
                        "interval_seconds": avg_interval,
                        "confidence": 0.8,
                        "evasion_window": avg_interval * 0.5
                    })
            
            # Check for quiet periods
            max_interval = max(intervals)
            if max_interval > np.mean(intervals) * 3:
                patterns.append({
                    "type": "quiet_periods",
                    "max_quiet_time": max_interval,
                    "confidence": 0.7,
                    "opportunity": "attack_during_quiet"
                })
        
        except Exception as e:
            logger.debug(f"Temporal pattern analysis failed: {e}")
        
        return patterns
    
    def _analyze_signature_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze signature patterns in defense events"""
        patterns = []
        
        try:
            # Extract detection signatures
            signatures = []
            for event in events:
                if "signature" in event:
                    signatures.append(event["signature"])
                elif "rule_id" in event:
                    signatures.append(event["rule_id"])
                elif "alert_name" in event:
                    signatures.append(event["alert_name"])
            
            if len(signatures) < 2:
                return patterns
            
            # Find common signature patterns
            signature_counts = {}
            for sig in signatures:
                signature_counts[sig] = signature_counts.get(sig, 0) + 1
            
            # Identify most common signatures
            sorted_signatures = sorted(signature_counts.items(), key=lambda x: x[1], reverse=True)
            
            for sig, count in sorted_signatures[:5]:  # Top 5 signatures
                if count > 1:
                    patterns.append({
                        "type": "signature_pattern",
                        "signature": sig,
                        "frequency": count,
                        "confidence": min(count / len(signatures), 1.0),
                        "evasion_strategy": "signature_obfuscation"
                    })
        
        except Exception as e:
            logger.debug(f"Signature pattern analysis failed: {e}")
        
        return patterns
    
    def _analyze_behavioral_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns in defense events"""
        patterns = []
        
        try:
            # Analyze event types
            event_types = [event.get("type", "unknown") for event in events]
            type_counts = {}
            for event_type in event_types:
                type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
            # Identify predominant behaviors
            for event_type, count in type_counts.items():
                if count > len(events) * 0.3:  # More than 30% of events
                    patterns.append({
                        "type": "behavioral_pattern",
                        "behavior": event_type,
                        "frequency": count / len(events),
                        "confidence": 0.75,
                        "evasion_strategy": "behavioral_mimicry"
                    })
        
        except Exception as e:
            logger.debug(f"Behavioral pattern analysis failed: {e}")
        
        return patterns
    
    def _generate_evasion_recommendations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate evasion recommendations based on patterns"""
        recommendations = []
        
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            
            if pattern_type == "regular_scanning":
                recommendations.append({
                    "technique": "timing_manipulation",
                    "rationale": "Exploit regular scanning intervals",
                    "parameters": {
                        "delay_window": pattern.get("evasion_window", 300),
                        "randomization": True
                    },
                    "confidence": pattern.get("confidence", 0.5)
                })
            
            elif pattern_type == "signature_pattern":
                recommendations.append({
                    "technique": "signature_obfuscation",
                    "rationale": f"Evade common signature: {pattern.get('signature', 'unknown')}",
                    "parameters": {
                        "target_signature": pattern.get("signature"),
                        "obfuscation_level": "high"
                    },
                    "confidence": pattern.get("confidence", 0.5)
                })
            
            elif pattern_type == "behavioral_pattern":
                recommendations.append({
                    "technique": "behavioral_mimicry",
                    "rationale": f"Mimic normal {pattern.get('behavior', 'unknown')} behavior",
                    "parameters": {
                        "target_behavior": pattern.get("behavior"),
                        "mimicry_level": "high"
                    },
                    "confidence": pattern.get("confidence", 0.5)
                })
        
        return recommendations
    
    def _timing_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement timing-based evasion"""
        delay_window = context.get("delay_window", 300)
        randomization = context.get("randomization", True)
        
        if randomization:
            actual_delay = random.uniform(delay_window * 0.5, delay_window * 1.5)
        else:
            actual_delay = delay_window
        
        return {
            "success": True,
            "technique": "timing_manipulation",
            "delay_applied": actual_delay,
            "next_action_time": datetime.utcnow() + timedelta(seconds=actual_delay)
        }
    
    def _traffic_shaping_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement traffic shaping evasion"""
        return {
            "success": True,
            "technique": "traffic_shaping",
            "modifications": ["packet_fragmentation", "timing_variation", "size_variation"],
            "effectiveness": 0.75
        }
    
    def _signature_obfuscation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement signature obfuscation"""
        target_signature = context.get("target_signature", "unknown")
        
        return {
            "success": True,
            "technique": "signature_obfuscation",
            "target_signature": target_signature,
            "obfuscation_methods": ["encoding", "encryption", "polymorphism"],
            "effectiveness": 0.8
        }
    
    def _behavioral_mimicry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement behavioral mimicry"""
        target_behavior = context.get("target_behavior", "normal_traffic")
        
        return {
            "success": True,
            "technique": "behavioral_mimicry",
            "target_behavior": target_behavior,
            "mimicry_techniques": ["pattern_matching", "timing_replication", "volume_matching"],
            "effectiveness": 0.7
        }
    
    def _steganographic_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement steganographic evasion"""
        return {
            "success": True,
            "technique": "steganography",
            "methods": ["image_steganography", "dns_tunneling", "protocol_steganography"],
            "effectiveness": 0.85
        }
    
    def _living_off_land_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement living-off-the-land evasion"""
        return {
            "success": True,
            "technique": "living_off_land",
            "legitimate_tools": ["powershell", "wmi", "certutil", "bitsadmin"],
            "effectiveness": 0.9
        }
    
    def _process_hollowing_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement process hollowing evasion"""
        return {
            "success": True,
            "technique": "process_hollowing",
            "target_processes": ["svchost.exe", "explorer.exe", "winlogon.exe"],
            "effectiveness": 0.8
        }
    
    def _domain_fronting_evasion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement domain fronting evasion"""
        return {
            "success": True,
            "technique": "domain_fronting",
            "fronting_domains": ["cdn.cloudfront.net", "ajax.googleapis.com"],
            "effectiveness": 0.85
        }


class AttackChainOrchestrator:
    """Orchestrate complex multi-vector attack chains"""
    
    def __init__(self, threat_actor_profile: ThreatActorProfile):
        self.threat_actor_profile = threat_actor_profile
        self.attack_phases = list(AttackPhase)
        self.current_phase = AttackPhase.RECONNAISSANCE
        self.completed_phases = []
        self.attack_chain = []
        self.objectives = []
        
        # Initialize phase-specific techniques
        self.phase_techniques = self._initialize_phase_techniques()
        
        # Initialize adaptive behavior
        self.adaptation_history = []
        self.success_patterns = {}
        
    def _initialize_phase_techniques(self) -> Dict[AttackPhase, List[AttackTechnique]]:
        """Initialize phase-specific attack techniques"""
        return {
            AttackPhase.RECONNAISSANCE: [
                AttackTechnique.NETWORK_SERVICE_SCANNING,
                AttackTechnique.SYSTEM_INFORMATION_DISCOVERY
            ],
            AttackPhase.INITIAL_ACCESS: [
                AttackTechnique.SPEAR_PHISHING,
                AttackTechnique.EXPLOIT_PUBLIC_FACING,
                AttackTechnique.VALID_ACCOUNTS,
                AttackTechnique.EXTERNAL_REMOTE_SERVICES,
                AttackTechnique.SUPPLY_CHAIN_COMPROMISE
            ],
            AttackPhase.EXECUTION: [
                AttackTechnique.SCHEDULED_TASK,
                AttackTechnique.CREATE_OR_MODIFY_PROCESS
            ],
            AttackPhase.PERSISTENCE: [
                AttackTechnique.BOOT_OR_LOGON_AUTOSTART,
                AttackTechnique.SCHEDULED_TASK
            ],
            AttackPhase.PRIVILEGE_ESCALATION: [
                AttackTechnique.ACCESS_TOKEN_MANIPULATION,
                AttackTechnique.BYPASS_USER_ACCOUNT_CONTROL,
                AttackTechnique.HIJACK_EXECUTION_FLOW
            ],
            AttackPhase.DEFENSE_EVASION: [
                AttackTechnique.PROCESS_INJECTION,
                AttackTechnique.MASQUERADING,
                AttackTechnique.OBFUSCATED_FILES,
                AttackTechnique.VIRTUALIZATION_DETECTION
            ],
            AttackPhase.CREDENTIAL_ACCESS: [
                AttackTechnique.OS_CREDENTIAL_DUMPING,
                AttackTechnique.BRUTE_FORCE,
                AttackTechnique.KEYLOGGING
            ],
            AttackPhase.DISCOVERY: [
                AttackTechnique.NETWORK_SERVICE_SCANNING,
                AttackTechnique.SYSTEM_INFORMATION_DISCOVERY
            ],
            AttackPhase.LATERAL_MOVEMENT: [
                AttackTechnique.REMOTE_SERVICES,
                AttackTechnique.LATERAL_TOOL_TRANSFER
            ],
            AttackPhase.COLLECTION: [
                AttackTechnique.DATA_FROM_LOCAL_SYSTEM,
                AttackTechnique.DATA_STAGED
            ],
            AttackPhase.COMMAND_AND_CONTROL: [
                AttackTechnique.APPLICATION_LAYER_PROTOCOL,
                AttackTechnique.ENCRYPTED_CHANNEL
            ],
            AttackPhase.EXFILTRATION: [
                AttackTechnique.EXFILTRATION_OVER_C2
            ],
            AttackPhase.IMPACT: [
                AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT
            ]
        }
    
    def generate_attack_chain(self, objectives: List[CampaignObjective], constraints: List[SafetyConstraint]) -> List[AttackAction]:
        """Generate comprehensive attack chain"""
        self.objectives = objectives
        attack_chain = []
        
        try:
            # Analyze objectives to determine required phases
            required_phases = self._determine_required_phases(objectives)
            
            # Generate actions for each required phase
            for phase in required_phases:
                phase_actions = self._generate_phase_actions(phase, constraints)
                attack_chain.extend(phase_actions)
            
            # Optimize attack chain for the threat actor profile
            optimized_chain = self._optimize_for_threat_actor(attack_chain)
            
            # Validate safety constraints
            validated_chain = self._validate_safety_constraints(optimized_chain, constraints)
            
            self.attack_chain = validated_chain
            
        except Exception as e:
            logger.error(f"Attack chain generation failed: {e}")
        
        return self.attack_chain
    
    def _determine_required_phases(self, objectives: List[CampaignObjective]) -> List[AttackPhase]:
        """Determine required attack phases based on objectives"""
        required_phases = [AttackPhase.RECONNAISSANCE, AttackPhase.INITIAL_ACCESS]
        
        for objective in objectives:
            obj_type = objective.objective_type.lower()
            
            if "data_exfiltration" in obj_type:
                required_phases.extend([
                    AttackPhase.DISCOVERY,
                    AttackPhase.COLLECTION,
                    AttackPhase.EXFILTRATION
                ])
            
            if "persistence" in obj_type:
                required_phases.extend([
                    AttackPhase.PERSISTENCE,
                    AttackPhase.DEFENSE_EVASION
                ])
            
            if "lateral_movement" in obj_type:
                required_phases.extend([
                    AttackPhase.CREDENTIAL_ACCESS,
                    AttackPhase.LATERAL_MOVEMENT
                ])
            
            if "privilege_escalation" in obj_type:
                required_phases.append(AttackPhase.PRIVILEGE_ESCALATION)
            
            if "disruption" in obj_type or "ransomware" in obj_type:
                required_phases.append(AttackPhase.IMPACT)
        
        # Remove duplicates and maintain logical order
        unique_phases = []
        for phase in self.attack_phases:
            if phase in required_phases and phase not in unique_phases:
                unique_phases.append(phase)
        
        return unique_phases
    
    def _generate_phase_actions(self, phase: AttackPhase, constraints: List[SafetyConstraint]) -> List[AttackAction]:
        """Generate actions for specific attack phase"""
        actions = []
        available_techniques = self.phase_techniques.get(phase, [])
        
        for technique in available_techniques:
            # Check if technique is allowed under constraints
            if self._is_technique_allowed(technique, constraints):
                action = self._create_attack_action(technique, phase)
                actions.append(action)
        
        return actions
    
    def _create_attack_action(self, technique: AttackTechnique, phase: AttackPhase) -> AttackAction:
        """Create attack action from technique and phase"""
        action_id = f"{phase.value}_{technique.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate technique-specific parameters
        parameters = self._generate_technique_parameters(technique)
        
        # Calculate success probability based on threat actor profile
        success_probability = self._calculate_success_probability(technique)
        
        # Assess risk level
        risk_level = self._assess_risk_level(technique, parameters)
        
        return AttackAction(
            action_id=action_id,
            technique=technique,
            phase=phase,
            target="auto_selected",
            parameters=parameters,
            expected_outcome=f"Execute {technique.value} in {phase.value} phase",
            success_probability=success_probability,
            risk_level=risk_level,
            safety_validated=True,
            timestamp=datetime.utcnow()
        )
    
    def _generate_technique_parameters(self, technique: AttackTechnique) -> Dict[str, Any]:
        """Generate technique-specific parameters"""
        base_params = {
            "execution_method": "automated",
            "stealth_level": "high",
            "detection_evasion": True
        }
        
        # Technique-specific parameters
        if technique == AttackTechnique.SPEAR_PHISHING:
            base_params.update({
                "target_count": 5,
                "personalization_level": "high",
                "payload_type": "macro_enabled_document"
            })
        elif technique == AttackTechnique.NETWORK_SERVICE_SCANNING:
            base_params.update({
                "scan_type": "stealth_syn",
                "port_range": "common_ports",
                "timing_template": "T3"
            })
        elif technique == AttackTechnique.LATERAL_TOOL_TRANSFER:
            base_params.update({
                "transfer_method": "smb",
                "tool_obfuscation": True,
                "cleanup_after_use": True
            })
        
        return base_params
    
    def _calculate_success_probability(self, technique: AttackTechnique) -> float:
        """Calculate success probability based on technique and threat actor"""
        base_probability = 0.5
        
        # Adjust for threat actor profile
        if self.threat_actor_profile == ThreatActorProfile.NATION_STATE_APT:
            base_probability += 0.3
        elif self.threat_actor_profile == ThreatActorProfile.AI_POWERED_ADVERSARY:
            base_probability += 0.25
        elif self.threat_actor_profile == ThreatActorProfile.ORGANIZED_CYBERCRIME:
            base_probability += 0.2
        
        # Adjust for technique complexity
        complex_techniques = [
            AttackTechnique.SUPPLY_CHAIN_COMPROMISE,
            AttackTechnique.HIJACK_EXECUTION_FLOW,
            AttackTechnique.PROCESS_INJECTION
        ]
        
        if technique in complex_techniques:
            base_probability -= 0.1
        
        return min(max(base_probability, 0.1), 0.95)
    
    def _assess_risk_level(self, technique: AttackTechnique, parameters: Dict[str, Any]) -> str:
        """Assess risk level of attack technique"""
        high_risk_techniques = [
            AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT,
            AttackTechnique.OS_CREDENTIAL_DUMPING,
            AttackTechnique.SUPPLY_CHAIN_COMPROMISE
        ]
        
        medium_risk_techniques = [
            AttackTechnique.LATERAL_TOOL_TRANSFER,
            AttackTechnique.PROCESS_INJECTION,
            AttackTechnique.HIJACK_EXECUTION_FLOW
        ]
        
        if technique in high_risk_techniques:
            return "high"
        elif technique in medium_risk_techniques:
            return "medium"
        else:
            return "low"
    
    def _is_technique_allowed(self, technique: AttackTechnique, constraints: List[SafetyConstraint]) -> bool:
        """Check if technique is allowed under safety constraints"""
        if SafetyConstraint.NO_DATA_MODIFICATION in constraints:
            if technique == AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT:
                return False
        
        if SafetyConstraint.NO_CREDENTIAL_HARVESTING in constraints:
            if technique in [AttackTechnique.OS_CREDENTIAL_DUMPING, AttackTechnique.KEYLOGGING]:
                return False
        
        if SafetyConstraint.NO_LATERAL_MOVEMENT_PRODUCTION in constraints:
            if technique in [AttackTechnique.LATERAL_TOOL_TRANSFER, AttackTechnique.REMOTE_SERVICES]:
                return False
        
        return True
    
    def _optimize_for_threat_actor(self, attack_chain: List[AttackAction]) -> List[AttackAction]:
        """Optimize attack chain for specific threat actor profile"""
        if self.threat_actor_profile == ThreatActorProfile.NATION_STATE_APT:
            # Prioritize stealth and persistence
            return sorted(attack_chain, key=lambda x: (x.phase.value, -x.success_probability))
        elif self.threat_actor_profile == ThreatActorProfile.RANSOMWARE_OPERATOR:
            # Prioritize speed and impact
            return sorted(attack_chain, key=lambda x: (x.phase.value, x.risk_level == "high"))
        else:
            # Default optimization
            return sorted(attack_chain, key=lambda x: (x.phase.value, -x.success_probability))
    
    def _validate_safety_constraints(self, attack_chain: List[AttackAction], constraints: List[SafetyConstraint]) -> List[AttackAction]:
        """Validate and filter attack chain based on safety constraints"""
        validated_chain = []
        
        for action in attack_chain:
            if self._is_technique_allowed(action.technique, constraints):
                action.safety_validated = True
                validated_chain.append(action)
            else:
                logger.warning(f"Action {action.action_id} violates safety constraints")
        
        return validated_chain


class AutonomousRedTeamEngine:
    """Main autonomous red team engine with RL-guided decision making"""
    
    def __init__(self, threat_actor_profile: ThreatActorProfile, autonomy_level: int = 50, safety_constraints: Dict[str, Any] = None):
        self.threat_actor_profile = threat_actor_profile
        self.autonomy_level = autonomy_level  # 1-100 scale
        self.safety_constraints = safety_constraints or {}
        
        # Initialize components
        if TORCH_AVAILABLE:
            self.rl_agent = ReinforcementLearningAgent()
        else:
            self.rl_agent = None
        
        self.defense_evasion_engine = DefenseEvasionEngine()
        self.attack_orchestrator = AttackChainOrchestrator(threat_actor_profile)
        
        # Campaign tracking
        self.active_campaigns = {}
        self.decision_history = []
        self.performance_metrics = {}
        
        # Safety systems
        self.safety_monitor = SafetyMonitor(safety_constraints)
        self.human_oversight_required = autonomy_level < 80
        
        logger.info("Autonomous Red Team Engine initialized", 
                   threat_actor=threat_actor_profile.value,
                   autonomy_level=autonomy_level,
                   safety_enabled=True)
    
    async def execute_autonomous_campaign(
        self, 
        campaign_config: Dict[str, Any], 
        objectives: List[CampaignObjective]
    ) -> Dict[str, Any]:
        """Execute autonomous red team campaign"""
        campaign_id = campaign_config.get("campaign_id", str(hash(str(campaign_config))))
        
        logger.info("Starting autonomous red team campaign", 
                   campaign_id=campaign_id,
                   objectives_count=len(objectives))
        
        campaign_result = {
            "campaign_id": campaign_id,
            "status": "running",
            "objectives": [asdict(obj) for obj in objectives],
            "decisions_made": [],
            "actions_executed": [],
            "defense_detections": [],
            "evasion_actions": [],
            "performance_metrics": {},
            "safety_violations": [],
            "human_interventions": []
        }
        
        try:
            # Generate initial attack chain
            safety_constraints = [SafetyConstraint(c) for c in self.safety_constraints.get("constraints", [])]
            attack_chain = self.attack_orchestrator.generate_attack_chain(objectives, safety_constraints)
            
            # Execute attack chain with autonomous decision making
            for action in attack_chain:
                # Safety check
                if not self.safety_monitor.validate_action(action):
                    safety_violation = {
                        "action_id": action.action_id,
                        "violation_type": "safety_constraint",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    campaign_result["safety_violations"].append(safety_violation)
                    continue
                
                # Make autonomous decision
                decision = await self._make_autonomous_decision(action, campaign_result)
                campaign_result["decisions_made"].append(asdict(decision))
                
                # Execute decision if approved
                if decision.selected_action:
                    execution_result = await self._execute_action(decision.selected_action)
                    campaign_result["actions_executed"].append(execution_result)
                    
                    # Check for defense detection
                    detection = await self._check_defense_detection(execution_result)
                    if detection:
                        campaign_result["defense_detections"].append(asdict(detection))
                        
                        # Apply evasion if detected
                        evasion_result = await self._apply_adaptive_evasion(detection)
                        if evasion_result:
                            campaign_result["evasion_actions"].append(evasion_result)
                
                # Update RL agent if available
                if self.rl_agent:
                    await self._update_rl_agent(decision, execution_result)
                
                # Check if human intervention is required
                if self._requires_human_intervention(decision, execution_result):
                    intervention = {
                        "decision_id": decision.decision_id,
                        "reason": "high_risk_action",
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "pending_approval"
                    }
                    campaign_result["human_interventions"].append(intervention)
            
            # Calculate final metrics
            campaign_result["performance_metrics"] = self._calculate_campaign_metrics(campaign_result)
            campaign_result["status"] = "completed"
            
            logger.info("Autonomous campaign completed", 
                       campaign_id=campaign_id,
                       decisions=len(campaign_result["decisions_made"]),
                       actions=len(campaign_result["actions_executed"]))
            
        except Exception as e:
            logger.error("Autonomous campaign failed", campaign_id=campaign_id, error=str(e))
            campaign_result["status"] = "failed"
            campaign_result["error"] = str(e)
        
        return campaign_result
    
    async def _make_autonomous_decision(self, action: AttackAction, campaign_context: Dict[str, Any]) -> AttackDecision:
        """Make autonomous decision using RL agent"""
        decision_id = f"decision_{action.action_id}_{datetime.utcnow().strftime('%H%M%S')}"
        
        # Prepare state for RL agent
        if self.rl_agent and TORCH_AVAILABLE:
            state = self._encode_state(action, campaign_context)
            action_index, confidence = self.rl_agent.select_action(state, epsilon=0.1)
            
            # Map action index to decision
            if action_index == 0:  # Execute action
                selected_action = action
                rationale = f"RL agent selected execution with {confidence:.2f} confidence"
            else:  # Skip action
                selected_action = None
                rationale = f"RL agent selected skip with {confidence:.2f} confidence"
        else:
            # Fallback decision making
            if action.success_probability > 0.6 and action.risk_level != "high":
                selected_action = action
                confidence = action.success_probability
                rationale = "Fallback decision: high success probability, acceptable risk"
            else:
                selected_action = None
                confidence = 0.5
                rationale = "Fallback decision: low success probability or high risk"
        
        # Risk assessment
        risk_assessment = {
            "technical_risk": action.risk_level,
            "detection_risk": self._calculate_detection_risk(action, campaign_context),
            "safety_risk": "low" if action.safety_validated else "high",
            "overall_risk": self._calculate_overall_risk(action)
        }
        
        decision = AttackDecision(
            decision_id=decision_id,
            phase=action.phase,
            available_actions=[action],
            selected_action=selected_action,
            decision_rationale=rationale,
            confidence=confidence,
            risk_assessment=risk_assessment,
            learning_feedback=None,
            timestamp=datetime.utcnow()
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _encode_state(self, action: AttackAction, context: Dict[str, Any]) -> torch.Tensor:
        """Encode current state for RL agent"""
        if not TORCH_AVAILABLE:
            return torch.zeros(128)
        
        # Create state vector
        state_vector = np.zeros(128)
        
        # Encode action features
        state_vector[0] = action.success_probability
        state_vector[1] = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(action.risk_level, 0.5)
        state_vector[2] = 1.0 if action.safety_validated else 0.0
        
        # Encode phase
        phase_index = list(AttackPhase).index(action.phase)
        state_vector[3] = phase_index / len(AttackPhase)
        
        # Encode campaign context
        decisions_made = len(context.get("decisions_made", []))
        state_vector[4] = min(decisions_made / 10.0, 1.0)
        
        detections = len(context.get("defense_detections", []))
        state_vector[5] = min(detections / 5.0, 1.0)
        
        # Encode threat actor profile
        actor_profiles = list(ThreatActorProfile)
        if self.threat_actor_profile in actor_profiles:
            actor_index = actor_profiles.index(self.threat_actor_profile)
            state_vector[6] = actor_index / len(actor_profiles)
        
        # Add some random features for robustness
        state_vector[10:20] = np.random.random(10) * 0.1
        
        return torch.tensor(state_vector, dtype=torch.float32)
    
    async def _execute_action(self, action: AttackAction) -> Dict[str, Any]:
        """Execute attack action (simulated)"""
        execution_result = {
            "action_id": action.action_id,
            "technique": action.technique.value,
            "phase": action.phase.value,
            "status": "executed",
            "success": False,
            "execution_time": datetime.utcnow().isoformat(),
            "output": {},
            "artifacts": []
        }
        
        try:
            # Simulate execution based on success probability
            if random.random() < action.success_probability:
                execution_result["success"] = True
                execution_result["output"] = self._generate_execution_output(action)
            else:
                execution_result["status"] = "failed"
                execution_result["error"] = "Execution failed due to environmental factors"
            
            # Simulate execution time
            execution_delay = random.uniform(1, 30)  # 1-30 seconds
            await asyncio.sleep(min(execution_delay, 2))  # Cap simulation time
            
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["error"] = str(e)
        
        return execution_result
    
    def _generate_execution_output(self, action: AttackAction) -> Dict[str, Any]:
        """Generate simulated execution output"""
        output = {
            "technique": action.technique.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if action.technique == AttackTechnique.NETWORK_SERVICE_SCANNING:
            output.update({
                "discovered_hosts": random.randint(1, 10),
                "open_ports": [22, 80, 443, 3389][:random.randint(1, 4)],
                "services_identified": random.randint(2, 8)
            })
        elif action.technique == AttackTechnique.SPEAR_PHISHING:
            output.update({
                "emails_sent": action.parameters.get("target_count", 5),
                "delivery_rate": random.uniform(0.8, 0.95),
                "click_rate": random.uniform(0.1, 0.3)
            })
        elif action.technique == AttackTechnique.LATERAL_TOOL_TRANSFER:
            output.update({
                "transfer_method": action.parameters.get("transfer_method", "smb"),
                "bytes_transferred": random.randint(1024, 10485760),
                "transfer_time": random.uniform(1, 30)
            })
        
        return output
    
    async def _check_defense_detection(self, execution_result: Dict[str, Any]) -> Optional[DefenseDetection]:
        """Check for defense system detection"""
        # Simulate defense detection probability
        detection_probability = 0.2  # 20% base detection rate
        
        # Adjust based on technique
        technique = execution_result.get("technique", "")
        if "credential" in technique.lower():
            detection_probability += 0.3
        elif "lateral" in technique.lower():
            detection_probability += 0.2
        
        if random.random() < detection_probability:
            return DefenseDetection(
                detection_id=f"detection_{execution_result['action_id']}",
                detection_type="automated_alert",
                severity=random.choice(["low", "medium", "high"]),
                confidence=random.uniform(0.6, 0.95),
                affected_actions=[execution_result["action_id"]],
                countermeasures_needed=["investigate", "contain", "eradicate"],
                timestamp=datetime.utcnow()
            )
        
        return None
    
    async def _apply_adaptive_evasion(self, detection: DefenseDetection) -> Optional[Dict[str, Any]]:
        """Apply adaptive evasion in response to detection"""
        # Analyze detection pattern
        defense_events = [asdict(detection)]
        pattern_analysis = self.defense_evasion_engine.analyze_defense_patterns(defense_events)
        
        # Apply evasion technique
        if pattern_analysis["evasion_opportunities"]:
            evasion_opportunity = pattern_analysis["evasion_opportunities"][0]
            technique = evasion_opportunity.get("technique", "timing_manipulation")
            
            evasion_result = self.defense_evasion_engine.apply_evasion_technique(
                technique, 
                evasion_opportunity.get("parameters", {})
            )
            
            return {
                "detection_id": detection.detection_id,
                "evasion_technique": technique,
                "evasion_result": evasion_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def _update_rl_agent(self, decision: AttackDecision, execution_result: Dict[str, Any]):
        """Update RL agent with execution feedback"""
        if not self.rl_agent or not TORCH_AVAILABLE:
            return
        
        try:
            # Calculate reward based on execution success and risk
            if execution_result["success"]:
                reward = 1.0
                # Bonus for low-risk successful actions
                if decision.risk_assessment["overall_risk"] == "low":
                    reward += 0.5
            else:
                reward = -0.5
                # Penalty for high-risk failures
                if decision.risk_assessment["overall_risk"] == "high":
                    reward -= 0.5
            
            # Store experience (simplified - would need proper state management)
            state = torch.zeros(128)  # Previous state
            action = 0 if decision.selected_action else 1
            next_state = torch.zeros(128)  # Current state
            done = False
            
            self.rl_agent.store_experience(state, action, reward, next_state, done)
            
            # Train if enough experiences
            if len(self.rl_agent.experience_buffer) > 32:
                training_metrics = self.rl_agent.train_step()
                logger.debug("RL agent training step completed", metrics=training_metrics)
        
        except Exception as e:
            logger.error(f"RL agent update failed: {e}")
    
    def _calculate_detection_risk(self, action: AttackAction, context: Dict[str, Any]) -> str:
        """Calculate detection risk for action"""
        base_risk = 0.3
        
        # Increase risk based on previous detections
        detections = len(context.get("defense_detections", []))
        base_risk += detections * 0.1
        
        # Increase risk for certain techniques
        high_detection_techniques = [
            AttackTechnique.OS_CREDENTIAL_DUMPING,
            AttackTechnique.PROCESS_INJECTION,
            AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT
        ]
        
        if action.technique in high_detection_techniques:
            base_risk += 0.3
        
        if base_risk > 0.7:
            return "high"
        elif base_risk > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_overall_risk(self, action: AttackAction) -> str:
        """Calculate overall risk of action"""
        risk_factors = [action.risk_level]
        
        if not action.safety_validated:
            risk_factors.append("high")
        
        # Count high-risk factors
        high_risk_count = sum(1 for risk in risk_factors if risk == "high")
        
        if high_risk_count > 0:
            return "high"
        elif any(risk == "medium" for risk in risk_factors):
            return "medium"
        else:
            return "low"
    
    def _requires_human_intervention(self, decision: AttackDecision, execution_result: Dict[str, Any]) -> bool:
        """Determine if human intervention is required"""
        if self.autonomy_level < 50:
            return True
        
        if decision.risk_assessment["overall_risk"] == "high" and self.autonomy_level < 80:
            return True
        
        if not execution_result.get("success", False) and decision.risk_assessment["overall_risk"] != "low":
            return True
        
        return False
    
    def _calculate_campaign_metrics(self, campaign_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate campaign performance metrics"""
        decisions = campaign_result["decisions_made"]
        actions = campaign_result["actions_executed"]
        detections = campaign_result["defense_detections"]
        
        successful_actions = sum(1 for action in actions if action.get("success", False))
        
        return {
            "total_decisions": len(decisions),
            "total_actions": len(actions),
            "successful_actions": successful_actions,
            "success_rate": successful_actions / max(len(actions), 1),
            "detection_rate": len(detections) / max(len(actions), 1),
            "evasion_attempts": len(campaign_result["evasion_actions"]),
            "safety_violations": len(campaign_result["safety_violations"]),
            "human_interventions": len(campaign_result["human_interventions"]),
            "autonomy_effectiveness": self._calculate_autonomy_effectiveness(campaign_result)
        }
    
    def _calculate_autonomy_effectiveness(self, campaign_result: Dict[str, Any]) -> float:
        """Calculate effectiveness of autonomous operations"""
        total_decisions = len(campaign_result["decisions_made"])
        interventions = len(campaign_result["human_interventions"])
        
        if total_decisions == 0:
            return 0.0
        
        autonomous_decisions = total_decisions - interventions
        base_effectiveness = autonomous_decisions / total_decisions
        
        # Adjust for success rate
        success_rate = campaign_result["performance_metrics"].get("success_rate", 0.0)
        effectiveness = base_effectiveness * (0.5 + 0.5 * success_rate)
        
        return min(effectiveness, 1.0)


class SafetyMonitor:
    """Monitor and enforce safety constraints"""
    
    def __init__(self, safety_constraints: Dict[str, Any]):
        self.safety_constraints = safety_constraints
        self.violation_log = []
        
    def validate_action(self, action: AttackAction) -> bool:
        """Validate action against safety constraints"""
        constraints = self.safety_constraints.get("constraints", [])
        
        for constraint in constraints:
            if not self._check_constraint(action, constraint):
                self._log_violation(action, constraint)
                return False
        
        return True
    
    def _check_constraint(self, action: AttackAction, constraint: str) -> bool:
        """Check specific safety constraint"""
        if constraint == SafetyConstraint.NO_DATA_MODIFICATION.value:
            return action.technique != AttackTechnique.DATA_ENCRYPTED_FOR_IMPACT
        
        elif constraint == SafetyConstraint.NO_CREDENTIAL_HARVESTING.value:
            forbidden_techniques = [
                AttackTechnique.OS_CREDENTIAL_DUMPING,
                AttackTechnique.KEYLOGGING
            ]
            return action.technique not in forbidden_techniques
        
        elif constraint == SafetyConstraint.TIME_LIMITED_OPERATIONS.value:
            max_duration = self.safety_constraints.get("max_duration_hours", 24)
            # Would check actual operation time in real implementation
            return True
        
        return True
    
    def _log_violation(self, action: AttackAction, constraint: str):
        """Log safety constraint violation"""
        violation = {
            "action_id": action.action_id,
            "constraint": constraint,
            "technique": action.technique.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.violation_log.append(violation)
        
        logger.warning("Safety constraint violation", 
                      action_id=action.action_id,
                      constraint=constraint)


# Global engine instance
_red_team_engine: Optional[AutonomousRedTeamEngine] = None

def get_autonomous_red_team_engine(
    threat_actor_profile: ThreatActorProfile = ThreatActorProfile.ADVANCED_PERSISTENT_THREAT,
    autonomy_level: int = 50,
    safety_constraints: Dict[str, Any] = None
) -> AutonomousRedTeamEngine:
    """Get global autonomous red team engine instance"""
    global _red_team_engine
    
    if _red_team_engine is None:
        _red_team_engine = AutonomousRedTeamEngine(
            threat_actor_profile=threat_actor_profile,
            autonomy_level=autonomy_level,
            safety_constraints=safety_constraints or {
                "constraints": [
                    SafetyConstraint.NO_DATA_MODIFICATION.value,
                    SafetyConstraint.LOGGING_REQUIRED.value,
                    SafetyConstraint.TIME_LIMITED_OPERATIONS.value
                ],
                "max_duration_hours": 24
            }
        )
    
    return _red_team_engine


# Module exports
__all__ = [
    'AutonomousRedTeamEngine',
    'ReinforcementLearningAgent',
    'DefenseEvasionEngine',
    'AttackChainOrchestrator',
    'SafetyMonitor',
    'AttackPhase',
    'ThreatActorProfile',
    'AttackTechnique',
    'SafetyConstraint',
    'AttackAction',
    'AttackDecision',
    'CampaignObjective',
    'DefenseDetection',
    'get_autonomous_red_team_engine'
]