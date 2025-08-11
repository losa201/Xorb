"""
Environment API - Gym-style interface for Red/Blue team training scenarios.

Provides a standardized interface for scenario orchestration with support for:
- Multi-agent environments (Red/Blue teams)
- Scenario management and seeding
- Episode recording and replay
- Observation/action/reward tracking
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActorRole(str, Enum):
    """Actor roles in the environment."""
    RED = "red"
    BLUE = "blue"
    PURPLE = "purple"  # Orchestrator/observer


class ActionType(str, Enum):
    """Types of actions actors can take."""
    REQUEST = "req"
    SCAN = "scan"
    EXPLOIT = "exploit"
    DEFEND = "defend"
    DETECT = "detect"
    BLOCK = "block"
    MONITOR = "monitor"


@dataclass
class Observation:
    """Standardized observation structure."""
    timestamp: float
    actor: ActorRole
    surface: List[str]  # Available attack/defense surfaces
    state: Dict[str, Any]  # Actor-specific state
    global_state: Dict[str, Any]  # Shared environment state
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    """Standardized action structure."""
    type: ActionType
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Result of an environment step."""
    success: bool
    latency_ms: int
    status_code: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepEvent:
    """JSONL step event for training data."""
    t: int  # Step number
    actor: ActorRole
    obs: Dict[str, Any]  # Minimal observation
    act: Dict[str, Any]  # Action taken
    r: Dict[str, Any]   # Result
    reward: Optional[float] = None
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(asdict(self), separators=(',', ':'))


class RewardConfig(BaseModel):
    """Reward function configuration."""
    
    # Red team weights
    red_impact_weight: float = 0.4
    red_dwell_weight: float = 0.2
    red_novelty_weight: float = 0.2
    red_detection_penalty: float = 0.15
    red_cost_penalty: float = 0.05
    
    # Blue team weights
    blue_speed_weight: float = 0.3
    blue_containment_weight: float = 0.3
    blue_generalization_weight: float = 0.2
    blue_false_pos_penalty: float = 0.15
    blue_residual_penalty: float = 0.05


class EpisodeManifest(BaseModel):
    """Episode manifest structure."""
    episode_id: str
    scenario_id: str
    roles: List[str]
    seed: int
    steps: int
    duration_seconds: float
    result: Dict[str, Any]
    artifacts: Dict[str, str]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class EnvironmentAPI:
    """Gym-style environment for Red/Blue team training."""
    
    def __init__(self, reward_config: Optional[RewardConfig] = None):
        self.reward_config = reward_config or RewardConfig()
        self.current_episode: Optional[str] = None
        self.step_count: int = 0
        self.episode_start: Optional[float] = None
        self.step_events: List[StepEvent] = []
        self.episode_artifacts: Dict[str, Any] = {}
        self.scenario_state: Dict[str, Any] = {}
        
        # Environment state
        self.red_state: Dict[str, Any] = {}
        self.blue_state: Dict[str, Any] = {}
        self.global_state: Dict[str, Any] = {}
        
        logger.info("EnvironmentAPI initialized")
    
    async def reset(
        self, 
        scenario_id: str, 
        seed: Optional[int] = None,
        roles: Optional[List[str]] = None
    ) -> Tuple[Dict[ActorRole, Observation], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            scenario_id: Scenario to load
            seed: Random seed for reproducibility
            roles: Actor roles to initialize
        
        Returns:
            Tuple of (initial observations per role, metadata)
        """
        # Generate episode ID
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        episode_suffix = f"{seed or np.random.randint(1000, 9999)}"
        self.current_episode = f"{timestamp}_{scenario_id}_{episode_suffix}"
        
        # Set random seed
        if seed:
            np.random.seed(seed)
        
        # Reset state
        self.step_count = 0
        self.episode_start = time.time()
        self.step_events = []
        self.episode_artifacts = {}
        
        # Initialize scenario
        await self._load_scenario(scenario_id)
        
        # Initialize actor states
        self.red_state = self._initialize_red_state()
        self.blue_state = self._initialize_blue_state()
        self.global_state = self._initialize_global_state()
        
        # Create initial observations
        observations = {}
        if not roles:
            roles = ["red", "blue"]
        
        for role in roles:
            if role == "red":
                observations[ActorRole.RED] = self._create_red_observation()
            elif role == "blue":
                observations[ActorRole.BLUE] = self._create_blue_observation()
        
        metadata = {
            "episode_id": self.current_episode,
            "scenario_id": scenario_id,
            "seed": seed,
            "roles": roles,
            "timestamp": time.time()
        }
        
        logger.info(f"Environment reset for episode {self.current_episode}")
        return observations, metadata
    
    async def step(
        self, 
        actor: ActorRole, 
        action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actor: Role taking the action
            action: Action to execute
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.current_episode:
            raise ValueError("Environment not reset. Call reset() first.")
        
        self.step_count += 1
        step_start = time.time()
        
        # Execute action
        try:
            result = await self._execute_action(actor, action)
            
            # Update state based on action and result
            await self._update_state(actor, action, result)
            
            # Create new observation
            observation = self._create_observation(actor)
            
            # Calculate reward
            reward = self._calculate_reward(actor, action, result)
            
            # Check if episode is done
            done = self._is_episode_done()
            
            # Record step event
            step_event = StepEvent(
                t=self.step_count,
                actor=actor,
                obs=self._minimal_observation(observation),
                act=action.to_dict(),
                r=result.to_dict(),
                reward=reward
            )
            self.step_events.append(step_event)
            
            # Additional info
            info = {
                "step_duration_ms": int((time.time() - step_start) * 1000),
                "total_steps": self.step_count,
                "episode_duration": time.time() - (self.episode_start or time.time())
            }
            
            logger.debug(f"Step {self.step_count}: {actor} -> {action.type} (reward: {reward:.3f})")
            
            return observation, reward, done, info
            
        except Exception as e:
            logger.error(f"Error in step {self.step_count}: {e}")
            # Return error observation
            error_obs = self._create_error_observation(actor, str(e))
            return error_obs, -1.0, True, {"error": str(e)}
    
    async def render(self, mode: str = "summary") -> Dict[str, Any]:
        """
        Render current environment state.
        
        Args:
            mode: Rendering mode (summary, logs, metrics)
        
        Returns:
            Rendered data
        """
        if mode == "summary":
            return {
                "episode_id": self.current_episode,
                "step_count": self.step_count,
                "duration": time.time() - (self.episode_start or time.time()),
                "red_state": self.red_state,
                "blue_state": self.blue_state,
                "global_state": self.global_state
            }
        elif mode == "logs":
            return {
                "step_events": [event.to_jsonl() for event in self.step_events]
            }
        elif mode == "metrics":
            return self._calculate_episode_metrics()
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    
    async def snapshot(self) -> Dict[str, Any]:
        """Create environment snapshot for quick restore."""
        return {
            "episode_id": self.current_episode,
            "step_count": self.step_count,
            "episode_start": self.episode_start,
            "red_state": self.red_state.copy(),
            "blue_state": self.blue_state.copy(),
            "global_state": self.global_state.copy(),
            "scenario_state": self.scenario_state.copy(),
            "step_events": [asdict(event) for event in self.step_events]
        }
    
    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore environment from snapshot."""
        self.current_episode = snapshot["episode_id"]
        self.step_count = snapshot["step_count"]
        self.episode_start = snapshot["episode_start"]
        self.red_state = snapshot["red_state"]
        self.blue_state = snapshot["blue_state"]
        self.global_state = snapshot["global_state"]
        self.scenario_state = snapshot["scenario_state"]
        self.step_events = [
            StepEvent(**event_data) for event_data in snapshot["step_events"]
        ]
        logger.info(f"Environment restored to step {self.step_count}")
    
    async def get_episode_manifest(self) -> Optional[EpisodeManifest]:
        """Generate episode manifest for completed episode."""
        if not self.current_episode or not self.episode_start:
            return None
        
        duration = time.time() - self.episode_start
        
        # Calculate final scores
        red_score = self._calculate_final_score(ActorRole.RED)
        blue_score = self._calculate_final_score(ActorRole.BLUE)
        winner = "red" if red_score > blue_score else "blue"
        
        return EpisodeManifest(
            episode_id=self.current_episode,
            scenario_id=self.scenario_state.get("scenario_id", "unknown"),
            roles=["red", "blue"],
            seed=self.scenario_state.get("seed", 0),
            steps=self.step_count,
            duration_seconds=duration,
            result={
                "red_score": red_score,
                "blue_score": blue_score,
                "winner": winner
            },
            artifacts=self.episode_artifacts,
            created_at=datetime.utcnow(),
            metadata=self.scenario_state
        )
    
    # Private methods
    
    async def _load_scenario(self, scenario_id: str) -> None:
        """Load scenario configuration."""
        # This would load from scenario database/files
        self.scenario_state = {
            "scenario_id": scenario_id,
            "network_topology": self._generate_network_topology(),
            "vulnerabilities": self._generate_vulnerabilities(),
            "detection_rules": self._generate_detection_rules(),
            "time_limit": 1800  # 30 minutes
        }
        logger.info(f"Loaded scenario: {scenario_id}")
    
    def _initialize_red_state(self) -> Dict[str, Any]:
        """Initialize red team state."""
        return {
            "position": "external",
            "discovered_hosts": [],
            "compromised_hosts": [],
            "extracted_data": [],
            "tools_used": [],
            "detection_events": 0,
            "stealth_score": 1.0
        }
    
    def _initialize_blue_state(self) -> Dict[str, Any]:
        """Initialize blue team state."""
        return {
            "alerts_generated": 0,
            "alerts_investigated": 0,
            "true_positives": 0,
            "false_positives": 0,
            "blocked_attempts": 0,
            "containment_actions": 0,
            "rule_updates": 0
        }
    
    def _initialize_global_state(self) -> Dict[str, Any]:
        """Initialize global environment state."""
        return {
            "network_health": 1.0,
            "security_posture": 0.8,
            "business_impact": 0.0,
            "attack_vectors": [],
            "active_connections": [],
            "system_load": 0.1
        }
    
    def _create_red_observation(self) -> Observation:
        """Create observation for red team."""
        surface = []
        if self.red_state["position"] == "external":
            surface = ["/api/auth/login", "/api/public/info"]
        elif self.red_state["position"] == "internal":
            surface.extend(["/api/admin", "/api/internal"])
        
        return Observation(
            timestamp=time.time(),
            actor=ActorRole.RED,
            surface=surface,
            state=self.red_state.copy(),
            global_state=self._filtered_global_state("red")
        )
    
    def _create_blue_observation(self) -> Observation:
        """Create observation for blue team."""
        return Observation(
            timestamp=time.time(),
            actor=ActorRole.BLUE,
            surface=list(self.scenario_state.get("detection_rules", {}).keys()),
            state=self.blue_state.copy(),
            global_state=self._filtered_global_state("blue")
        )
    
    def _create_observation(self, actor: ActorRole) -> Observation:
        """Create observation for given actor."""
        if actor == ActorRole.RED:
            return self._create_red_observation()
        elif actor == ActorRole.BLUE:
            return self._create_blue_observation()
        else:
            raise ValueError(f"Unknown actor: {actor}")
    
    def _create_error_observation(self, actor: ActorRole, error: str) -> Observation:
        """Create error observation."""
        return Observation(
            timestamp=time.time(),
            actor=actor,
            surface=[],
            state={"error": error},
            global_state={}
        )
    
    async def _execute_action(self, actor: ActorRole, action: Action) -> StepResult:
        """Execute action and return result."""
        start_time = time.time()
        
        try:
            if actor == ActorRole.RED:
                result = await self._execute_red_action(action)
            elif actor == ActorRole.BLUE:
                result = await self._execute_blue_action(action)
            else:
                raise ValueError(f"Unknown actor: {actor}")
            
            latency = int((time.time() - start_time) * 1000)
            result.latency_ms = latency
            return result
            
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return StepResult(
                success=False,
                latency_ms=latency,
                error=str(e)
            )
    
    async def _execute_red_action(self, action: Action) -> StepResult:
        """Execute red team action."""
        # Simulate various red team actions
        await asyncio.sleep(0.1)  # Simulate action execution time
        
        if action.type == ActionType.REQUEST:
            # Simulate API request
            path = action.parameters.get("path", "/")
            method = action.parameters.get("method", "GET")
            
            # Simple simulation logic
            success = np.random.random() > 0.3
            status_code = 200 if success else np.random.choice([401, 403, 404, 500])
            
            return StepResult(
                success=success,
                latency_ms=0,  # Will be set by caller
                status_code=status_code,
                response_data={"method": method, "path": path}
            )
        
        elif action.type == ActionType.SCAN:
            # Simulate network scan
            target = action.parameters.get("target", "localhost")
            ports = action.parameters.get("ports", [80, 443])
            
            open_ports = [p for p in ports if np.random.random() > 0.5]
            
            return StepResult(
                success=True,
                latency_ms=0,
                response_data={"target": target, "open_ports": open_ports}
            )
        
        elif action.type == ActionType.EXPLOIT:
            # Simulate exploitation attempt
            vulnerability = action.parameters.get("vulnerability", "sql_injection")
            success_rate = 0.4 if vulnerability == "sql_injection" else 0.2
            
            success = np.random.random() < success_rate
            
            return StepResult(
                success=success,
                latency_ms=0,
                response_data={"vulnerability": vulnerability, "exploited": success}
            )
        
        else:
            return StepResult(success=False, latency_ms=0, error="Unknown action type")
    
    async def _execute_blue_action(self, action: Action) -> StepResult:
        """Execute blue team action."""
        await asyncio.sleep(0.05)  # Simulate action execution time
        
        if action.type == ActionType.DETECT:
            # Simulate detection action
            threshold = action.parameters.get("threshold", 0.5)
            detection_accuracy = 0.85  # 85% accuracy
            
            # Check if there's actually something to detect
            red_active = len(self.red_state.get("tools_used", [])) > 0
            detected = red_active and np.random.random() < detection_accuracy
            
            return StepResult(
                success=True,
                latency_ms=0,
                response_data={"detected": detected, "confidence": np.random.random()}
            )
        
        elif action.type == ActionType.BLOCK:
            # Simulate blocking action
            target = action.parameters.get("target", "unknown")
            success = np.random.random() > 0.1  # 90% success rate
            
            return StepResult(
                success=success,
                latency_ms=0,
                response_data={"target": target, "blocked": success}
            )
        
        else:
            return StepResult(success=False, latency_ms=0, error="Unknown action type")
    
    async def _update_state(self, actor: ActorRole, action: Action, result: StepResult) -> None:
        """Update environment state based on action and result."""
        if actor == ActorRole.RED and result.success:
            if action.type == ActionType.REQUEST:
                self.red_state["tools_used"].append("web_request")
                if result.status_code == 200:
                    self.red_state["stealth_score"] *= 0.95
            elif action.type == ActionType.SCAN:
                self.red_state["tools_used"].append("port_scanner")
                self.red_state["stealth_score"] *= 0.9
            elif action.type == ActionType.EXPLOIT:
                if result.response_data and result.response_data.get("exploited"):
                    self.red_state["compromised_hosts"].append("target_host")
                    self.red_state["stealth_score"] *= 0.8
        
        elif actor == ActorRole.BLUE and result.success:
            if action.type == ActionType.DETECT:
                self.blue_state["alerts_generated"] += 1
                if result.response_data and result.response_data.get("detected"):
                    self.blue_state["true_positives"] += 1
                else:
                    self.blue_state["false_positives"] += 1
            elif action.type == ActionType.BLOCK:
                if result.response_data and result.response_data.get("blocked"):
                    self.blue_state["blocked_attempts"] += 1
    
    def _calculate_reward(self, actor: ActorRole, action: Action, result: StepResult) -> float:
        """Calculate reward for the action."""
        if actor == ActorRole.RED:
            reward = 0.0
            
            # Impact reward
            if result.success and action.type == ActionType.EXPLOIT:
                reward += self.reward_config.red_impact_weight * 1.0
            
            # Stealth bonus
            stealth_score = self.red_state.get("stealth_score", 1.0)
            reward += self.reward_config.red_novelty_weight * stealth_score
            
            # Detection penalty
            if self.blue_state.get("true_positives", 0) > 0:
                detection_conf = min(1.0, self.blue_state["true_positives"] / 10.0)
                reward -= self.reward_config.red_detection_penalty * detection_conf
            
            # Cost penalty for failed actions
            if not result.success:
                reward -= self.reward_config.red_cost_penalty * 0.1
            
            return reward
        
        elif actor == ActorRole.BLUE:
            reward = 0.0
            
            # Detection speed reward
            if action.type == ActionType.DETECT and result.success:
                if result.response_data and result.response_data.get("detected"):
                    reward += self.reward_config.blue_speed_weight * 1.0
            
            # Containment reward
            if action.type == ActionType.BLOCK and result.success:
                reward += self.reward_config.blue_containment_weight * 0.5
            
            # False positive penalty
            fp_rate = self.blue_state.get("false_positives", 0) / max(1, self.blue_state.get("alerts_generated", 1))
            reward -= self.reward_config.blue_false_pos_penalty * fp_rate
            
            return reward
        
        return 0.0
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end."""
        # Episode ends if:
        # 1. Time limit reached
        # 2. Red team fully compromised the system
        # 3. Blue team completely contained the threat
        # 4. Too many steps taken
        
        if self.step_count >= 1000:  # Max steps
            return True
        
        if self.episode_start:
            duration = time.time() - self.episode_start
            time_limit = self.scenario_state.get("time_limit", 1800)
            if duration >= time_limit:
                return True
        
        # Red team wins if they compromise multiple hosts
        if len(self.red_state.get("compromised_hosts", [])) >= 3:
            return True
        
        # Blue team wins if they block enough attempts
        if self.blue_state.get("blocked_attempts", 0) >= 10:
            return True
        
        return False
    
    def _minimal_observation(self, obs: Observation) -> Dict[str, Any]:
        """Create minimal observation for JSONL logging."""
        return {
            "surface": obs.surface[:3],  # Limit surface for space
            "position": obs.state.get("position"),
            "alerts": obs.state.get("alerts_generated"),
            "compromised": len(obs.state.get("compromised_hosts", []))
        }
    
    def _filtered_global_state(self, actor_type: str) -> Dict[str, Any]:
        """Filter global state based on actor visibility."""
        filtered = {}
        
        if actor_type == "red":
            # Red team has limited visibility
            filtered = {
                "network_health": self.global_state.get("network_health"),
                "system_load": self.global_state.get("system_load")
            }
        elif actor_type == "blue":
            # Blue team has full visibility
            filtered = self.global_state.copy()
        
        return filtered
    
    def _calculate_final_score(self, actor: ActorRole) -> float:
        """Calculate final score for an actor."""
        if actor == ActorRole.RED:
            impact = len(self.red_state.get("compromised_hosts", []))
            stealth = self.red_state.get("stealth_score", 1.0)
            return impact * 0.6 + stealth * 0.4
        
        elif actor == ActorRole.BLUE:
            tp = self.blue_state.get("true_positives", 0)
            fp = self.blue_state.get("false_positives", 0)
            blocked = self.blue_state.get("blocked_attempts", 0)
            
            if tp + fp > 0:
                accuracy = tp / (tp + fp)
            else:
                accuracy = 0.5
            
            return accuracy * 0.5 + blocked * 0.1
        
        return 0.0
    
    def _calculate_episode_metrics(self) -> Dict[str, Any]:
        """Calculate episode performance metrics."""
        return {
            "total_steps": self.step_count,
            "duration": time.time() - (self.episode_start or time.time()),
            "red_score": self._calculate_final_score(ActorRole.RED),
            "blue_score": self._calculate_final_score(ActorRole.BLUE),
            "red_metrics": {
                "compromised_hosts": len(self.red_state.get("compromised_hosts", [])),
                "stealth_score": self.red_state.get("stealth_score", 1.0),
                "tools_used": len(set(self.red_state.get("tools_used", [])))
            },
            "blue_metrics": {
                "detection_rate": self.blue_state.get("true_positives", 0) / max(1, self.blue_state.get("alerts_generated", 1)),
                "false_positive_rate": self.blue_state.get("false_positives", 0) / max(1, self.blue_state.get("alerts_generated", 1)),
                "blocked_attempts": self.blue_state.get("blocked_attempts", 0)
            }
        }
    
    def _generate_network_topology(self) -> Dict[str, Any]:
        """Generate network topology for scenario."""
        return {
            "hosts": ["web-server", "db-server", "admin-workstation"],
            "networks": ["dmz", "internal", "admin"],
            "connections": [
                {"from": "external", "to": "web-server", "port": 80},
                {"from": "web-server", "to": "db-server", "port": 3306},
                {"from": "admin-workstation", "to": "db-server", "port": 3306}
            ]
        }
    
    def _generate_vulnerabilities(self) -> Dict[str, Any]:
        """Generate vulnerabilities for scenario."""
        return {
            "web-server": ["sql_injection", "xss"],
            "db-server": ["weak_password", "unpatched_service"],
            "admin-workstation": ["phishing_susceptible", "weak_2fa"]
        }
    
    def _generate_detection_rules(self) -> Dict[str, Any]:
        """Generate detection rules for scenario."""
        return {
            "sql_injection_detector": {"threshold": 0.8, "accuracy": 0.9},
            "anomaly_detector": {"threshold": 0.7, "accuracy": 0.75},
            "network_scanner_detector": {"threshold": 0.9, "accuracy": 0.95}
        }