"""
Autonomous Red Team Orchestrator
Advanced orchestrator for coordinating sophisticated multi-agent red team operations
with AI-driven decision making, attack chain planning, and autonomous adaptation.

This system implements production-grade autonomous red team capabilities with 
comprehensive safety controls and ethical boundaries.
"""

import asyncio
import logging
import json
import uuid
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import random
import numpy as np
from collections import defaultdict, deque

from ..agents.production_red_team_agent import ProductionRedTeamAgent, SafetyConstraints
from ..agents.base_agent import AgentConfiguration, AgentType
from ..learning.advanced_reinforcement_learning import AdvancedRLEngine, EnvironmentState
from ..learning.autonomous_explorer import AutonomousExplorer

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    """Mission execution phases following MITRE ATT&CK"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class ObjectiveType(Enum):
    """Mission objective types"""
    GAIN_INITIAL_ACCESS = "gain_initial_access"
    ESCALATE_PRIVILEGES = "escalate_privileges"
    ESTABLISH_PERSISTENCE = "establish_persistence"
    MOVE_LATERALLY = "move_laterally"
    COLLECT_INTELLIGENCE = "collect_intelligence"
    MAINTAIN_ACCESS = "maintain_access"
    DEMONSTRATE_IMPACT = "demonstrate_impact"
    EVADE_DETECTION = "evade_detection"


class DecisionType(Enum):
    """Types of autonomous decisions"""
    TECHNIQUE_SELECTION = "technique_selection"
    TARGET_PRIORITIZATION = "target_prioritization"
    RESOURCE_ALLOCATION = "resource_allocation"
    RISK_MITIGATION = "risk_mitigation"
    PHASE_TRANSITION = "phase_transition"
    MISSION_ADAPTATION = "mission_adaptation"


@dataclass
class MissionObjective:
    """Individual mission objective"""
    id: str
    type: ObjectiveType
    description: str
    priority: int  # 1-10, 10 being highest
    prerequisites: List[str]  # IDs of prerequisite objectives
    target_systems: List[str]
    success_criteria: Dict[str, Any]
    time_limit: Optional[timedelta] = None
    completed: bool = False
    completion_time: Optional[datetime] = None


@dataclass
class MissionPlan:
    """Comprehensive mission plan"""
    mission_id: str
    name: str
    description: str
    objectives: List[MissionObjective]
    target_environment: Dict[str, Any]
    constraints: SafetyConstraints
    timeline: Dict[str, Any]
    success_metrics: Dict[str, Any]
    risk_tolerance: str
    created_at: datetime
    estimated_duration: timedelta


@dataclass
class AgentAssignment:
    """Agent assignment for mission tasks"""
    agent_id: str
    agent_type: str
    assigned_objectives: List[str]
    current_task: Optional[str]
    status: str  # idle, active, completed, failed
    performance_score: float
    last_activity: datetime


@dataclass
class AutonomousDecision:
    """Record of autonomous decision made by orchestrator"""
    decision_id: str
    decision_type: DecisionType
    context: Dict[str, Any]
    options_considered: List[Dict[str, Any]]
    selected_option: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: datetime
    outcome: Optional[Dict[str, Any]] = None


@dataclass
class ThreatIntelligence:
    """Threat intelligence data for mission planning"""
    target_profile: Dict[str, Any]
    known_vulnerabilities: List[Dict[str, Any]]
    defensive_capabilities: List[str]
    network_topology: Dict[str, Any]
    security_controls: List[Dict[str, Any]]
    threat_landscape: Dict[str, Any]
    intelligence_confidence: float
    last_updated: datetime


class AttackGraphPlanner:
    """Attack graph planning for sophisticated attack chains"""
    
    def __init__(self):
        self.mitre_attack_graph = self._build_mitre_attack_graph()
        self.technique_dependencies = self._load_technique_dependencies()
        self.success_probabilities = defaultdict(lambda: 0.5)
        
    def _build_mitre_attack_graph(self) -> nx.DiGraph:
        """Build directed graph of MITRE ATT&CK techniques"""
        graph = nx.DiGraph()
        
        # Add MITRE ATT&CK technique nodes and edges
        mitre_techniques = {
            # Reconnaissance
            "T1046": {"name": "Network Service Scanning", "phase": "reconnaissance"},
            "T1040": {"name": "Network Sniffing", "phase": "reconnaissance"},
            "T1018": {"name": "Remote System Discovery", "phase": "discovery"},
            
            # Initial Access
            "T1190": {"name": "Exploit Public-Facing Application", "phase": "initial_access"},
            "T1078": {"name": "Valid Accounts", "phase": "initial_access"},
            "T1566": {"name": "Phishing", "phase": "initial_access"},
            
            # Execution
            "T1059": {"name": "Command and Scripting Interpreter", "phase": "execution"},
            "T1053": {"name": "Scheduled Task/Job", "phase": "execution"},
            
            # Persistence
            "T1543": {"name": "Create or Modify System Process", "phase": "persistence"},
            "T1547": {"name": "Boot or Logon Autostart Execution", "phase": "persistence"},
            
            # Privilege Escalation
            "T1068": {"name": "Exploitation for Privilege Escalation", "phase": "privilege_escalation"},
            "T1055": {"name": "Process Injection", "phase": "privilege_escalation"},
            
            # Defense Evasion
            "T1027": {"name": "Obfuscated Files or Information", "phase": "defense_evasion"},
            "T1070": {"name": "Indicator Removal on Host", "phase": "defense_evasion"},
            
            # Credential Access
            "T1003": {"name": "OS Credential Dumping", "phase": "credential_access"},
            "T1110": {"name": "Brute Force", "phase": "credential_access"},
            
            # Discovery
            "T1082": {"name": "System Information Discovery", "phase": "discovery"},
            "T1083": {"name": "File and Directory Discovery", "phase": "discovery"},
            
            # Lateral Movement
            "T1021": {"name": "Remote Services", "phase": "lateral_movement"},
            "T1075": {"name": "Pass the Hash", "phase": "lateral_movement"},
            
            # Collection
            "T1005": {"name": "Data from Local System", "phase": "collection"},
            "T1039": {"name": "Data from Network Shared Drive", "phase": "collection"},
            
            # Command and Control
            "T1071": {"name": "Application Layer Protocol", "phase": "command_control"},
            "T1090": {"name": "Proxy", "phase": "command_control"},
            
            # Exfiltration
            "T1041": {"name": "Exfiltration Over C2 Channel", "phase": "exfiltration"},
            "T1567": {"name": "Exfiltration Over Web Service", "phase": "exfiltration"}
        }
        
        # Add nodes
        for technique_id, info in mitre_techniques.items():
            graph.add_node(technique_id, **info)
        
        # Add edges based on logical dependencies
        technique_dependencies = [
            ("T1046", "T1190"),  # Port scan -> Exploit service
            ("T1190", "T1059"),  # Exploit -> Command execution
            ("T1059", "T1082"),  # Command execution -> System discovery
            ("T1082", "T1068"),  # System discovery -> Privilege escalation
            ("T1068", "T1003"),  # Privilege escalation -> Credential dumping
            ("T1003", "T1021"),  # Credentials -> Lateral movement
            ("T1021", "T1083"),  # Lateral movement -> File discovery
            ("T1083", "T1005"),  # File discovery -> Data collection
            ("T1059", "T1543"),  # Command execution -> Persistence
            ("T1068", "T1055"),  # Privilege escalation -> Process injection
            ("T1055", "T1027"),  # Process injection -> Obfuscation
            ("T1005", "T1071"),  # Data collection -> C2 communication
            ("T1071", "T1041"),  # C2 -> Exfiltration
        ]
        
        for source, target in technique_dependencies:
            graph.add_edge(source, target)
        
        return graph
    
    def _load_technique_dependencies(self) -> Dict[str, List[str]]:
        """Load technique dependencies and prerequisites"""
        return {
            "T1190": ["T1046"],  # Exploit requires reconnaissance
            "T1068": ["T1190", "T1059"],  # Privilege escalation requires initial access
            "T1003": ["T1068"],  # Credential dumping requires privileges
            "T1021": ["T1003", "T1078"],  # Lateral movement requires credentials
            "T1005": ["T1021", "T1083"],  # Data collection requires access and discovery
            "T1041": ["T1005", "T1071"],  # Exfiltration requires data and C2
        }
    
    def plan_attack_sequence(self, objectives: List[MissionObjective], 
                           threat_intel: ThreatIntelligence) -> List[str]:
        """Plan optimal attack sequence to achieve objectives"""
        
        # Map objectives to MITRE techniques
        objective_techniques = self._map_objectives_to_techniques(objectives)
        
        # Find optimal path through attack graph
        attack_paths = []
        for technique_set in objective_techniques:
            paths = self._find_attack_paths(technique_set, threat_intel)
            attack_paths.extend(paths)
        
        # Select best overall attack sequence
        optimal_sequence = self._select_optimal_sequence(attack_paths, threat_intel)
        
        return optimal_sequence
    
    def _map_objectives_to_techniques(self, objectives: List[MissionObjective]) -> List[List[str]]:
        """Map mission objectives to MITRE ATT&CK techniques"""
        objective_mapping = {
            ObjectiveType.GAIN_INITIAL_ACCESS: ["T1190", "T1078", "T1566"],
            ObjectiveType.ESCALATE_PRIVILEGES: ["T1068", "T1055"],
            ObjectiveType.ESTABLISH_PERSISTENCE: ["T1543", "T1547"],
            ObjectiveType.MOVE_LATERALLY: ["T1021", "T1075"],
            ObjectiveType.COLLECT_INTELLIGENCE: ["T1005", "T1039", "T1083"],
            ObjectiveType.EVADE_DETECTION: ["T1027", "T1070"],
        }
        
        technique_sets = []
        for objective in objectives:
            if objective.type in objective_mapping:
                technique_sets.append(objective_mapping[objective.type])
        
        return technique_sets
    
    def _find_attack_paths(self, technique_set: List[str], 
                          threat_intel: ThreatIntelligence) -> List[List[str]]:
        """Find possible attack paths through technique set"""
        paths = []
        
        for start_technique in technique_set:
            for end_technique in technique_set:
                if start_technique != end_technique:
                    try:
                        path = nx.shortest_path(self.mitre_attack_graph, start_technique, end_technique)
                        # Score path based on threat intelligence
                        path_score = self._score_attack_path(path, threat_intel)
                        paths.append((path, path_score))
                    except nx.NetworkXNoPath:
                        continue
        
        # Return paths sorted by score
        paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in paths[:5]]  # Top 5 paths
    
    def _score_attack_path(self, path: List[str], threat_intel: ThreatIntelligence) -> float:
        """Score attack path based on success probability and stealth"""
        base_score = 1.0
        
        # Factor in known vulnerabilities
        vulnerability_bonus = 0.0
        for vuln in threat_intel.known_vulnerabilities:
            if any(technique in vuln.get("applicable_techniques", []) for technique in path):
                vulnerability_bonus += 0.2
        
        # Factor in defensive capabilities
        defense_penalty = 0.0
        for defense in threat_intel.defensive_capabilities:
            if "edr" in defense.lower() or "antivirus" in defense.lower():
                defense_penalty += 0.1
        
        # Factor in technique success probabilities
        success_prob = np.mean([self.success_probabilities[t] for t in path])
        
        return base_score + vulnerability_bonus - defense_penalty + success_prob
    
    def _select_optimal_sequence(self, attack_paths: List[List[str]], 
                                threat_intel: ThreatIntelligence) -> List[str]:
        """Select optimal attack sequence from available paths"""
        if not attack_paths:
            return []
        
        # Combine and deduplicate techniques while maintaining logical order
        all_techniques = []
        seen_techniques = set()
        
        # Sort paths by estimated success probability
        scored_paths = [(path, self._score_attack_path(path, threat_intel)) for path in attack_paths]
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        # Build sequence from highest scored paths
        for path, score in scored_paths:
            for technique in path:
                if technique not in seen_techniques:
                    all_techniques.append(technique)
                    seen_techniques.add(technique)
        
        return all_techniques


class AutonomousOrchestrator:
    """Advanced autonomous orchestrator for red team operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.orchestrator_id = str(uuid.uuid4())
        
        # Core components
        self.attack_planner = AttackGraphPlanner()
        self.rl_engine = None  # Will be initialized
        self.autonomous_explorer = None  # Will be initialized
        
        # Mission management
        self.active_missions: Dict[str, MissionPlan] = {}
        self.agent_pool: Dict[str, ProductionRedTeamAgent] = {}
        self.agent_assignments: Dict[str, AgentAssignment] = {}
        
        # Decision making
        self.decision_history: List[AutonomousDecision] = []
        self.performance_metrics = defaultdict(list)
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        
        # Autonomous operation parameters
        self.autonomy_level = self.config.get("autonomy_level", 0.7)  # 0-1, higher = more autonomous
        self.decision_confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.risk_tolerance = self.config.get("risk_tolerance", "medium")
        
        # Safety and control
        self.safety_override = False
        self.human_oversight_required = self.config.get("human_oversight", True)
        self.max_concurrent_operations = self.config.get("max_concurrent_ops", 5)
        
        logger.info(f"Autonomous Orchestrator initialized with autonomy level {self.autonomy_level}")
    
    async def initialize(self, rl_engine: AdvancedRLEngine = None, 
                        autonomous_explorer: AutonomousExplorer = None):
        """Initialize orchestrator with AI components"""
        self.rl_engine = rl_engine
        self.autonomous_explorer = autonomous_explorer
        
        if self.rl_engine:
            logger.info("RL engine integrated for decision optimization")
        
        if self.autonomous_explorer:
            logger.info("Autonomous explorer integrated for adaptive learning")
    
    async def create_mission_plan(self, mission_config: Dict[str, Any]) -> MissionPlan:
        """Create comprehensive mission plan with AI-driven optimization"""
        
        mission_id = str(uuid.uuid4())
        
        # Parse mission configuration
        objectives = self._parse_mission_objectives(mission_config.get("objectives", []))
        target_environment = mission_config.get("target_environment", {})
        constraints = self._create_safety_constraints(mission_config.get("constraints", {}))
        
        # Gather threat intelligence
        threat_intel = await self._gather_threat_intelligence(target_environment)
        
        # Plan attack sequence using AI
        attack_sequence = self.attack_planner.plan_attack_sequence(objectives, threat_intel)
        
        # Estimate mission timeline
        timeline = self._estimate_mission_timeline(attack_sequence, objectives)
        
        # Create mission plan
        mission_plan = MissionPlan(
            mission_id=mission_id,
            name=mission_config.get("name", f"Mission {mission_id[:8]}"),
            description=mission_config.get("description", "Autonomous red team operation"),
            objectives=objectives,
            target_environment=target_environment,
            constraints=constraints,
            timeline=timeline,
            success_metrics=mission_config.get("success_metrics", {}),
            risk_tolerance=mission_config.get("risk_tolerance", "medium"),
            created_at=datetime.utcnow(),
            estimated_duration=timeline.get("total_duration", timedelta(hours=8))
        )
        
        # Store mission plan
        self.active_missions[mission_id] = mission_plan
        
        # Store threat intelligence
        self.threat_intelligence[mission_id] = threat_intel
        
        logger.info(f"Mission plan created: {mission_id} with {len(objectives)} objectives")
        
        return mission_plan
    
    async def execute_mission(self, mission_id: str) -> Dict[str, Any]:
        """Execute mission with autonomous coordination"""
        
        if mission_id not in self.active_missions:
            raise ValueError(f"Mission {mission_id} not found")
        
        mission_plan = self.active_missions[mission_id]
        threat_intel = self.threat_intelligence[mission_id]
        
        logger.info(f"Starting autonomous execution of mission {mission_id}")
        
        # Initialize mission execution state
        execution_state = {
            "mission_id": mission_id,
            "phase": MissionPhase.RECONNAISSANCE,
            "completed_objectives": [],
            "active_objectives": [],
            "failed_objectives": [],
            "start_time": datetime.utcnow(),
            "agent_assignments": {},
            "autonomous_decisions": []
        }
        
        # Allocate agents to mission
        agent_assignments = await self._allocate_agents_to_mission(mission_plan, threat_intel)
        execution_state["agent_assignments"] = agent_assignments
        
        # Execute mission phases autonomously
        for phase in MissionPhase:
            phase_result = await self._execute_mission_phase(
                mission_plan, phase, execution_state, threat_intel
            )
            
            execution_state["autonomous_decisions"].extend(phase_result.get("decisions", []))
            
            # Check if mission objectives are met
            if self._check_mission_completion(mission_plan, execution_state):
                break
            
            # Autonomous adaptation based on results
            if self.autonomy_level > 0.7:
                adaptation_decision = await self._autonomous_mission_adaptation(
                    mission_plan, execution_state, phase_result
                )
                if adaptation_decision:
                    execution_state["autonomous_decisions"].append(adaptation_decision)
        
        # Calculate mission results
        mission_results = self._calculate_mission_results(mission_plan, execution_state)
        
        # Learn from mission execution
        if self.rl_engine:
            await self._update_mission_learning(mission_plan, execution_state, mission_results)
        
        logger.info(f"Mission {mission_id} execution completed with {mission_results['success_rate']:.2%} success rate")
        
        return mission_results
    
    async def _execute_mission_phase(self, mission_plan: MissionPlan, phase: MissionPhase,
                                   execution_state: Dict[str, Any], 
                                   threat_intel: ThreatIntelligence) -> Dict[str, Any]:
        """Execute specific mission phase with autonomous decision making"""
        
        logger.info(f"Executing mission phase: {phase.value}")
        
        phase_start_time = datetime.utcnow()
        phase_decisions = []
        phase_results = {}
        
        # Get objectives for this phase
        phase_objectives = self._get_phase_objectives(mission_plan.objectives, phase)
        
        if not phase_objectives:
            return {"phase": phase.value, "decisions": [], "results": {}}
        
        # Autonomous agent coordination for phase
        for objective in phase_objectives:
            # Make autonomous decision on technique selection
            technique_decision = await self._autonomous_technique_selection(
                objective, threat_intel, execution_state
            )
            phase_decisions.append(technique_decision)
            
            # Execute technique with selected agent
            agent_id = technique_decision.selected_option.get("agent_id")
            if agent_id and agent_id in self.agent_pool:
                agent = self.agent_pool[agent_id]
                
                technique_params = technique_decision.selected_option.get("parameters", {})
                
                try:
                    execution_result = await agent.execute_task(
                        task_id=str(uuid.uuid4()),
                        technique_id=technique_decision.selected_option.get("technique_id"),
                        parameters=technique_params
                    )
                    
                    # Record outcome
                    technique_decision.outcome = asdict(execution_result)
                    
                    # Update objective status
                    if execution_result.status.value == "completed":
                        objective.completed = True
                        objective.completion_time = datetime.utcnow()
                        execution_state["completed_objectives"].append(objective.id)
                    
                    phase_results[objective.id] = asdict(execution_result)
                    
                except Exception as e:
                    logger.error(f"Technique execution failed: {e}")
                    technique_decision.outcome = {"error": str(e)}
                    execution_state["failed_objectives"].append(objective.id)
        
        # Phase completion analysis
        phase_duration = (datetime.utcnow() - phase_start_time).total_seconds()
        phase_success_rate = len([obj for obj in phase_objectives if obj.completed]) / len(phase_objectives)
        
        # Autonomous phase transition decision
        if phase_success_rate >= 0.7 or self.autonomy_level > 0.8:
            transition_decision = await self._autonomous_phase_transition(
                phase, phase_success_rate, execution_state
            )
            phase_decisions.append(transition_decision)
        
        return {
            "phase": phase.value,
            "decisions": phase_decisions,
            "results": phase_results,
            "duration": phase_duration,
            "success_rate": phase_success_rate
        }
    
    async def _autonomous_technique_selection(self, objective: MissionObjective,
                                            threat_intel: ThreatIntelligence,
                                            execution_state: Dict[str, Any]) -> AutonomousDecision:
        """Make autonomous decision on technique selection"""
        
        decision_id = str(uuid.uuid4())
        
        # Get available techniques for objective
        available_techniques = self._get_techniques_for_objective(objective)
        
        # Get available agents
        available_agents = self._get_available_agents(execution_state)
        
        # Generate options
        options = []
        for technique in available_techniques:
            for agent_id in available_agents:
                agent = self.agent_pool[agent_id]
                
                # Check if agent can execute technique
                if technique in await agent.get_capabilities():
                    # Calculate option score
                    score = await self._score_technique_option(
                        technique, agent_id, objective, threat_intel
                    )
                    
                    options.append({
                        "technique_id": technique,
                        "agent_id": agent_id,
                        "score": score,
                        "parameters": self._generate_technique_parameters(technique, objective)
                    })
        
        # Select best option
        if options:
            # Use RL engine for selection if available
            if self.rl_engine and self.autonomy_level > 0.6:
                selected_option = await self._rl_technique_selection(options, objective, threat_intel)
            else:
                # Simple scoring-based selection
                options.sort(key=lambda x: x["score"], reverse=True)
                selected_option = options[0]
        else:
            selected_option = {"error": "No viable options available"}
        
        # Calculate decision confidence
        confidence = self._calculate_decision_confidence(options, selected_option)
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(objective, options, selected_option)
        
        decision = AutonomousDecision(
            decision_id=decision_id,
            decision_type=DecisionType.TECHNIQUE_SELECTION,
            context={
                "objective": asdict(objective),
                "threat_intel_confidence": threat_intel.intelligence_confidence,
                "available_options": len(options)
            },
            options_considered=options,
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        
        self.decision_history.append(decision)
        
        logger.info(f"Autonomous technique selection: {selected_option.get('technique_id')} "
                   f"with confidence {confidence:.2f}")
        
        return decision
    
    async def _rl_technique_selection(self, options: List[Dict[str, Any]], 
                                    objective: MissionObjective,
                                    threat_intel: ThreatIntelligence) -> Dict[str, Any]:
        """Use RL engine for intelligent technique selection"""
        
        if not self.autonomous_explorer:
            # Fallback to highest scoring option
            options.sort(key=lambda x: x["score"], reverse=True)
            return options[0]
        
        # Create context for RL decision
        context = {
            "objective_type": objective.type.value,
            "target_environment": objective.target_systems,
            "threat_intelligence": {
                "vulnerability_count": len(threat_intel.known_vulnerabilities),
                "defense_strength": len(threat_intel.defensive_capabilities),
                "network_complexity": len(threat_intel.network_topology.get("subnets", []))
            },
            "available_techniques": [opt["technique_id"] for opt in options]
        }
        
        # Get suggestion from autonomous explorer
        suggestion = await self.autonomous_explorer.suggest_technique(context)
        
        # Find matching option
        suggested_technique = suggestion.get("technique_id")
        if suggested_technique:
            for option in options:
                if option["technique_id"] == suggested_technique:
                    return option
        
        # Fallback to highest scoring option
        options.sort(key=lambda x: x["score"], reverse=True)
        return options[0]
    
    async def _score_technique_option(self, technique: str, agent_id: str,
                                    objective: MissionObjective,
                                    threat_intel: ThreatIntelligence) -> float:
        """Score technique option based on multiple factors"""
        
        base_score = 0.5
        
        # Agent performance history
        agent_assignments = self.agent_assignments.get(agent_id)
        if agent_assignments:
            base_score += agent_assignments.performance_score * 0.3
        
        # Technique success probability from threat intel
        for vuln in threat_intel.known_vulnerabilities:
            if technique in vuln.get("applicable_techniques", []):
                base_score += 0.2
        
        # Objective priority factor
        priority_factor = objective.priority / 10.0
        base_score += priority_factor * 0.2
        
        # Environmental factors
        if threat_intel.defensive_capabilities:
            # Reduce score if strong defenses present
            defense_penalty = len(threat_intel.defensive_capabilities) * 0.05
            base_score -= defense_penalty
        
        # Stealth considerations
        if "stealth" in technique.lower() or "evasion" in technique.lower():
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    async def _autonomous_phase_transition(self, current_phase: MissionPhase,
                                         success_rate: float,
                                         execution_state: Dict[str, Any]) -> AutonomousDecision:
        """Make autonomous decision on phase transition"""
        
        decision_id = str(uuid.uuid4())
        
        # Determine next phase options
        phase_sequence = list(MissionPhase)
        current_index = phase_sequence.index(current_phase)
        
        options = []
        
        # Option 1: Continue to next phase
        if current_index < len(phase_sequence) - 1:
            next_phase = phase_sequence[current_index + 1]
            options.append({
                "action": "proceed_to_next_phase",
                "target_phase": next_phase.value,
                "score": success_rate
            })
        
        # Option 2: Repeat current phase if success rate is low
        if success_rate < 0.5:
            options.append({
                "action": "repeat_current_phase",
                "target_phase": current_phase.value,
                "score": 1.0 - success_rate
            })
        
        # Option 3: Skip to more advanced phase if highly successful
        if success_rate > 0.9 and current_index < len(phase_sequence) - 2:
            skip_phase = phase_sequence[current_index + 2]
            options.append({
                "action": "skip_to_advanced_phase",
                "target_phase": skip_phase.value,
                "score": success_rate * 0.8
            })
        
        # Select best option
        if options:
            options.sort(key=lambda x: x["score"], reverse=True)
            selected_option = options[0]
        else:
            selected_option = {"action": "mission_complete", "target_phase": None}
        
        confidence = success_rate if success_rate > 0.5 else 0.3
        
        reasoning = f"Phase {current_phase.value} achieved {success_rate:.1%} success rate. "
        reasoning += f"Selected action: {selected_option['action']}"
        
        decision = AutonomousDecision(
            decision_id=decision_id,
            decision_type=DecisionType.PHASE_TRANSITION,
            context={
                "current_phase": current_phase.value,
                "success_rate": success_rate,
                "completed_objectives": len(execution_state["completed_objectives"]),
                "failed_objectives": len(execution_state["failed_objectives"])
            },
            options_considered=options,
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        
        return decision
    
    async def _autonomous_mission_adaptation(self, mission_plan: MissionPlan,
                                           execution_state: Dict[str, Any],
                                           phase_result: Dict[str, Any]) -> Optional[AutonomousDecision]:
        """Make autonomous adaptations to mission based on results"""
        
        if self.autonomy_level < 0.7:
            return None  # Not autonomous enough for adaptations
        
        decision_id = str(uuid.uuid4())
        
        # Analyze mission progress
        total_objectives = len(mission_plan.objectives)
        completed_objectives = len(execution_state["completed_objectives"])
        failed_objectives = len(execution_state["failed_objectives"])
        overall_success_rate = completed_objectives / total_objectives
        
        # Determine adaptation options
        options = []
        
        # Option 1: Adjust risk tolerance if too many failures
        if failed_objectives > completed_objectives:
            options.append({
                "adaptation_type": "increase_risk_tolerance",
                "description": "Increase risk tolerance to use more aggressive techniques",
                "impact": "higher_success_probability",
                "score": 0.8
            })
        
        # Option 2: Add additional objectives if too successful
        if overall_success_rate > 0.9:
            options.append({
                "adaptation_type": "add_stretch_objectives",
                "description": "Add additional objectives to maximize engagement value",
                "impact": "increased_scope",
                "score": 0.7
            })
        
        # Option 3: Adjust techniques based on defensive responses
        if phase_result.get("success_rate", 0) < 0.3:
            options.append({
                "adaptation_type": "switch_to_evasive_techniques",
                "description": "Switch to more evasive techniques due to strong defenses",
                "impact": "reduced_detection_risk",
                "score": 0.9
            })
        
        if not options:
            return None
        
        # Select best adaptation
        options.sort(key=lambda x: x["score"], reverse=True)
        selected_option = options[0]
        
        confidence = min(0.8, self.autonomy_level)  # Cap confidence for adaptations
        
        reasoning = f"Mission adaptation triggered by: overall success rate {overall_success_rate:.1%}, "
        reasoning += f"phase success rate {phase_result.get('success_rate', 0):.1%}"
        
        decision = AutonomousDecision(
            decision_id=decision_id,
            decision_type=DecisionType.MISSION_ADAPTATION,
            context={
                "overall_success_rate": overall_success_rate,
                "phase_success_rate": phase_result.get("success_rate", 0),
                "autonomy_level": self.autonomy_level
            },
            options_considered=options,
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        
        # Apply adaptation
        await self._apply_mission_adaptation(mission_plan, selected_option)
        
        logger.info(f"Autonomous mission adaptation: {selected_option['adaptation_type']}")
        
        return decision
    
    # Helper methods
    
    def _parse_mission_objectives(self, objectives_config: List[Dict[str, Any]]) -> List[MissionObjective]:
        """Parse mission objectives from configuration"""
        objectives = []
        
        for i, obj_config in enumerate(objectives_config):
            objective = MissionObjective(
                id=obj_config.get("id", str(uuid.uuid4())),
                type=ObjectiveType(obj_config.get("type", "gain_initial_access")),
                description=obj_config.get("description", f"Objective {i+1}"),
                priority=obj_config.get("priority", 5),
                prerequisites=obj_config.get("prerequisites", []),
                target_systems=obj_config.get("target_systems", []),
                success_criteria=obj_config.get("success_criteria", {}),
                time_limit=timedelta(hours=obj_config.get("time_limit_hours", 4))
            )
            objectives.append(objective)
        
        return objectives
    
    def _create_safety_constraints(self, constraints_config: Dict[str, Any]) -> SafetyConstraints:
        """Create safety constraints from configuration"""
        return SafetyConstraints(
            environment=constraints_config.get("environment", "staging"),
            max_impact_level=constraints_config.get("max_impact_level", "medium"),
            allowed_techniques=constraints_config.get("allowed_techniques", []),
            forbidden_techniques=constraints_config.get("forbidden_techniques", []),
            target_whitelist=constraints_config.get("target_whitelist", []),
            require_authorization=constraints_config.get("require_authorization", True),
            auto_cleanup=constraints_config.get("auto_cleanup", True),
            monitoring_required=constraints_config.get("monitoring_required", True)
        )
    
    async def _gather_threat_intelligence(self, target_environment: Dict[str, Any]) -> ThreatIntelligence:
        """Gather threat intelligence for mission planning"""
        
        # Simulate threat intelligence gathering
        # In production, this would integrate with real threat intel sources
        
        mock_vulnerabilities = [
            {
                "cve_id": "CVE-2021-44228",
                "severity": "critical",
                "description": "Log4j RCE vulnerability",
                "applicable_techniques": ["T1190", "T1059"]
            },
            {
                "cve_id": "CVE-2021-34527",
                "severity": "high",
                "description": "Print Spooler privilege escalation",
                "applicable_techniques": ["T1068"]
            }
        ]
        
        mock_defenses = [
            "Windows Defender",
            "Network IDS",
            "Application firewall",
            "EDR solution"
        ]
        
        mock_topology = {
            "subnets": ["10.0.1.0/24", "10.0.2.0/24"],
            "hosts": target_environment.get("hosts", ["target1", "target2"]),
            "critical_systems": ["domain_controller", "database_server"]
        }
        
        return ThreatIntelligence(
            target_profile=target_environment,
            known_vulnerabilities=mock_vulnerabilities,
            defensive_capabilities=mock_defenses,
            network_topology=mock_topology,
            security_controls=[],
            threat_landscape={},
            intelligence_confidence=0.7,
            last_updated=datetime.utcnow()
        )
    
    def _estimate_mission_timeline(self, attack_sequence: List[str], 
                                 objectives: List[MissionObjective]) -> Dict[str, Any]:
        """Estimate mission timeline based on attack sequence and objectives"""
        
        # Estimate duration for each technique
        technique_durations = {
            "T1046": timedelta(minutes=30),  # Network scanning
            "T1190": timedelta(hours=2),     # Exploitation
            "T1059": timedelta(minutes=15),  # Command execution
            "T1068": timedelta(hours=1),     # Privilege escalation
            "T1003": timedelta(minutes=45),  # Credential dumping
            "T1021": timedelta(hours=1),     # Lateral movement
            "T1005": timedelta(minutes=30),  # Data collection
        }
        
        total_duration = timedelta()
        phase_durations = {}
        
        for technique in attack_sequence:
            duration = technique_durations.get(technique, timedelta(minutes=30))
            total_duration += duration
        
        # Add buffer time
        total_duration = total_duration * 1.5
        
        return {
            "total_duration": total_duration,
            "estimated_start": datetime.utcnow(),
            "estimated_completion": datetime.utcnow() + total_duration,
            "phase_durations": phase_durations
        }
    
    def _get_phase_objectives(self, objectives: List[MissionObjective], 
                            phase: MissionPhase) -> List[MissionObjective]:
        """Get objectives that belong to a specific mission phase"""
        
        phase_mapping = {
            MissionPhase.RECONNAISSANCE: [ObjectiveType.COLLECT_INTELLIGENCE],
            MissionPhase.INITIAL_ACCESS: [ObjectiveType.GAIN_INITIAL_ACCESS],
            MissionPhase.PRIVILEGE_ESCALATION: [ObjectiveType.ESCALATE_PRIVILEGES],
            MissionPhase.PERSISTENCE: [ObjectiveType.ESTABLISH_PERSISTENCE],
            MissionPhase.LATERAL_MOVEMENT: [ObjectiveType.MOVE_LATERALLY],
            MissionPhase.COLLECTION: [ObjectiveType.COLLECT_INTELLIGENCE],
            MissionPhase.DEFENSE_EVASION: [ObjectiveType.EVADE_DETECTION]
        }
        
        phase_objective_types = phase_mapping.get(phase, [])
        return [obj for obj in objectives if obj.type in phase_objective_types]
    
    def _get_techniques_for_objective(self, objective: MissionObjective) -> List[str]:
        """Get applicable techniques for an objective"""
        
        technique_mapping = {
            ObjectiveType.GAIN_INITIAL_ACCESS: [
                "recon.advanced_port_scan", "exploit.web_application", "exploit.network_service"
            ],
            ObjectiveType.ESCALATE_PRIVILEGES: [
                "exploit.privilege_escalation", "post.credential_harvesting"
            ],
            ObjectiveType.ESTABLISH_PERSISTENCE: [
                "post.establish_persistence"
            ],
            ObjectiveType.MOVE_LATERALLY: [
                "post.lateral_movement"
            ],
            ObjectiveType.COLLECT_INTELLIGENCE: [
                "post.data_collection", "recon.service_enumeration"
            ],
            ObjectiveType.EVADE_DETECTION: [
                "evasion.anti_forensics", "evasion.process_hiding"
            ]
        }
        
        return technique_mapping.get(objective.type, [])
    
    def _get_available_agents(self, execution_state: Dict[str, Any]) -> List[str]:
        """Get list of available agent IDs"""
        available = []
        
        for agent_id, assignment in execution_state.get("agent_assignments", {}).items():
            if assignment.get("status") in ["idle", "active"]:
                available.append(agent_id)
        
        return available
    
    def _generate_technique_parameters(self, technique: str, objective: MissionObjective) -> Dict[str, Any]:
        """Generate parameters for technique execution"""
        
        base_params = {
            "objective_id": objective.id,
            "priority": objective.priority,
            "target_systems": objective.target_systems
        }
        
        # Technique-specific parameters
        if "port_scan" in technique:
            base_params.update({
                "scan_type": "stealth",
                "ports": "1-1000"
            })
        elif "web_application" in technique:
            base_params.update({
                "technique": "sql_injection",
                "target_url": objective.target_systems[0] if objective.target_systems else "http://target"
            })
        elif "persistence" in technique:
            base_params.update({
                "persistence_type": "registry",
                "target_system": objective.target_systems[0] if objective.target_systems else "target"
            })
        
        return base_params
    
    def _calculate_decision_confidence(self, options: List[Dict[str, Any]], 
                                     selected_option: Dict[str, Any]) -> float:
        """Calculate confidence in decision based on option quality"""
        
        if not options or "error" in selected_option:
            return 0.1
        
        selected_score = selected_option.get("score", 0.5)
        
        # Factor in score spread
        scores = [opt.get("score", 0.5) for opt in options]
        score_std = np.std(scores) if len(scores) > 1 else 0.1
        
        # Higher confidence if selected option is clearly better
        confidence = selected_score
        
        if score_std > 0.1:
            confidence += 0.2  # Bonus for clear winner
        
        return min(1.0, confidence)
    
    def _generate_decision_reasoning(self, objective: MissionObjective,
                                   options: List[Dict[str, Any]],
                                   selected_option: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for decision"""
        
        if "error" in selected_option:
            return f"No viable options available for objective {objective.type.value}"
        
        reasoning = f"Selected {selected_option.get('technique_id', 'unknown')} "
        reasoning += f"for {objective.type.value} objective. "
        reasoning += f"Score: {selected_option.get('score', 0):.2f} "
        reasoning += f"among {len(options)} options. "
        
        if objective.priority > 7:
            reasoning += "High priority objective. "
        
        return reasoning
    
    async def _allocate_agents_to_mission(self, mission_plan: MissionPlan,
                                        threat_intel: ThreatIntelligence) -> Dict[str, AgentAssignment]:
        """Allocate agents to mission objectives"""
        
        assignments = {}
        
        # Create mock agent assignments
        # In production, this would allocate from actual agent pool
        for i, objective in enumerate(mission_plan.objectives):
            agent_id = f"agent_{i+1}"
            
            assignment = AgentAssignment(
                agent_id=agent_id,
                agent_type="production_red_team",
                assigned_objectives=[objective.id],
                current_task=None,
                status="idle",
                performance_score=0.8,
                last_activity=datetime.utcnow()
            )
            
            assignments[agent_id] = assignment
        
        return assignments
    
    def _check_mission_completion(self, mission_plan: MissionPlan,
                                execution_state: Dict[str, Any]) -> bool:
        """Check if mission objectives are completed"""
        
        completed_count = len(execution_state["completed_objectives"])
        total_count = len(mission_plan.objectives)
        
        # Mission complete if 80% of objectives achieved
        return completed_count / total_count >= 0.8
    
    def _calculate_mission_results(self, mission_plan: MissionPlan,
                                 execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive mission results"""
        
        total_objectives = len(mission_plan.objectives)
        completed_objectives = len(execution_state["completed_objectives"])
        failed_objectives = len(execution_state["failed_objectives"])
        
        success_rate = completed_objectives / total_objectives
        failure_rate = failed_objectives / total_objectives
        
        mission_duration = (datetime.utcnow() - execution_state["start_time"]).total_seconds()
        
        return {
            "mission_id": mission_plan.mission_id,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "completed_objectives": completed_objectives,
            "total_objectives": total_objectives,
            "duration_seconds": mission_duration,
            "autonomous_decisions_made": len(execution_state["autonomous_decisions"]),
            "agent_assignments": len(execution_state["agent_assignments"]),
            "overall_assessment": "success" if success_rate >= 0.7 else "partial" if success_rate >= 0.4 else "failure"
        }
    
    async def _update_mission_learning(self, mission_plan: MissionPlan,
                                     execution_state: Dict[str, Any],
                                     mission_results: Dict[str, Any]):
        """Update learning models based on mission execution"""
        
        if not self.rl_engine:
            return
        
        # Update RL engine with mission outcomes
        for decision in execution_state["autonomous_decisions"]:
            if decision.outcome:
                success = decision.outcome.get("status") == "completed"
                
                await self.rl_engine.record_technique_result(
                    technique_id=decision.selected_option.get("technique_id", "unknown"),
                    success=success,
                    execution_time=decision.outcome.get("execution_time", 60.0),
                    context={
                        "decision_type": decision.decision_type.value,
                        "confidence": decision.confidence,
                        "mission_success_rate": mission_results["success_rate"]
                    }
                )
        
        logger.info(f"Mission learning update completed for {mission_plan.mission_id}")
    
    async def _apply_mission_adaptation(self, mission_plan: MissionPlan,
                                      adaptation: Dict[str, Any]):
        """Apply autonomous mission adaptation"""
        
        adaptation_type = adaptation.get("adaptation_type")
        
        if adaptation_type == "increase_risk_tolerance":
            # Increase risk tolerance for mission
            mission_plan.risk_tolerance = "high"
            logger.info("Increased mission risk tolerance")
        
        elif adaptation_type == "add_stretch_objectives":
            # Add additional objectives
            new_objective = MissionObjective(
                id=str(uuid.uuid4()),
                type=ObjectiveType.DEMONSTRATE_IMPACT,
                description="Additional stretch objective",
                priority=3,
                prerequisites=[],
                target_systems=mission_plan.target_environment.get("hosts", []),
                success_criteria={}
            )
            mission_plan.objectives.append(new_objective)
            logger.info("Added stretch objective to mission")
        
        elif adaptation_type == "switch_to_evasive_techniques":
            # This would modify technique selection preferences
            # Implementation would depend on specific technique registry
            logger.info("Switched to evasive techniques")
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        
        active_mission_count = len(self.active_missions)
        total_decisions = len(self.decision_history)
        
        recent_decisions = [d for d in self.decision_history 
                          if d.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        avg_confidence = np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0.0
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "autonomy_level": self.autonomy_level,
            "active_missions": active_mission_count,
            "agent_pool_size": len(self.agent_pool),
            "total_decisions_made": total_decisions,
            "recent_decisions": len(recent_decisions),
            "average_decision_confidence": avg_confidence,
            "safety_override_active": self.safety_override,
            "human_oversight_required": self.human_oversight_required,
            "max_concurrent_operations": self.max_concurrent_operations
        }
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        
        # Cancel active missions
        for mission_id in list(self.active_missions.keys()):
            logger.info(f"Cancelling active mission: {mission_id}")
            # Implementation would gracefully stop mission execution
        
        # Shutdown agents
        for agent in self.agent_pool.values():
            await agent.shutdown()
        
        logger.info("Autonomous Orchestrator shutdown complete")