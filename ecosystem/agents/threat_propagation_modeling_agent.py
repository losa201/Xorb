#!/usr/bin/env python3
"""
🌊 ThreatPropagationModelingAgent - Phase 12.3 Implementation
Models how threats spread across networks and systems using epidemic modeling and graph neural networks.

Part of the XORB Ecosystem - Phase 12: Autonomous Defense & Planetary Scale Operations
"""

import asyncio
import logging
import time
import uuid
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import random
import math
from concurrent.futures import ThreadPoolExecutor
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# Metrics
propagation_models_created = Counter('xorb_propagation_models_created_total', 'Total propagation models created')
propagation_simulations_run = Counter('xorb_propagation_simulations_run_total', 'Total propagation simulations executed')
containment_strategies_generated = Counter('xorb_containment_strategies_generated_total', 'Total containment strategies generated')
simulation_duration_seconds = Histogram('xorb_simulation_duration_seconds', 'Simulation execution duration')
network_infection_rate = Gauge('xorb_network_infection_rate', 'Current network infection rate')
containment_effectiveness = Gauge('xorb_containment_effectiveness', 'Current containment effectiveness score')

logger = structlog.get_logger("threat_propagation_modeling_agent")

class PropagationModel(Enum):
    """Types of threat propagation models"""
    SIR = "sir"  # Susceptible-Infected-Recovered
    SEIR = "seir"  # Susceptible-Exposed-Infected-Recovered
    SIS = "sis"  # Susceptible-Infected-Susceptible
    SIRS = "sirs"  # Susceptible-Infected-Recovered-Susceptible
    NETWORK_DIFFUSION = "network_diffusion"
    PERCOLATION = "percolation"
    CASCADE = "cascade"
    MULTI_LAYER = "multi_layer"

class NodeState(Enum):
    """States a network node can be in"""
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTED = "infected"
    RECOVERED = "recovered"
    IMMUNIZED = "immunized"
    QUARANTINED = "quarantined"
    COMPROMISED = "compromised"

class ThreatType(Enum):
    """Types of threats that can propagate"""
    MALWARE = "malware"
    WORM = "worm"
    RANSOMWARE = "ransomware"
    BOTNET = "botnet"
    APT = "apt"
    DDOS = "ddos"
    DATA_BREACH = "data_breach"
    INSIDER_THREAT = "insider_threat"

@dataclass
class NetworkNode:
    """Represents a node in the network"""
    node_id: str
    node_type: str  # server, workstation, mobile, iot, etc.
    state: NodeState
    infection_time: Optional[datetime]
    recovery_time: Optional[datetime]
    vulnerability_score: float
    connectivity: int
    criticality: float
    security_controls: List[str]
    geographic_location: Optional[Tuple[float, float]]
    last_updated: datetime

@dataclass
class NetworkEdge:
    """Represents a connection between nodes"""
    source_id: str
    target_id: str
    connection_type: str  # network, trust, data_flow, etc.
    weight: float
    bandwidth: float
    latency: float
    security_level: str
    last_activity: datetime

@dataclass
class ThreatVector:
    """Represents a specific threat and its characteristics"""
    threat_id: str
    threat_type: ThreatType
    virulence: float  # How infectious it is
    persistence: float  # How long it stays active
    detection_difficulty: float
    payload_complexity: float
    lateral_movement_capability: float
    propagation_methods: List[str]
    target_preferences: Dict[str, float]
    temporal_behavior: Dict[str, Any]
    created_at: datetime

@dataclass
class PropagationScenario:
    """Represents a propagation simulation scenario"""
    scenario_id: str
    network_topology: Dict[str, Any]
    threat_vector: ThreatVector
    initial_infection_nodes: List[str]
    simulation_parameters: Dict[str, Any]
    propagation_model: PropagationModel
    time_horizon: int  # simulation steps
    containment_measures: List[Dict[str, Any]]
    created_at: datetime

@dataclass
class SimulationResult:
    """Results from a propagation simulation"""
    scenario_id: str
    final_infection_rate: float
    peak_infection_time: int
    total_infected_nodes: int
    propagation_speed: float
    containment_effectiveness: float
    critical_paths: List[List[str]]
    bottleneck_nodes: List[str]
    recovery_time: int
    simulation_metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ContainmentStrategy:
    """Strategy for containing threat propagation"""
    strategy_id: str
    strategy_type: str  # isolation, patching, monitoring, etc.
    target_nodes: List[str]
    target_edges: List[Tuple[str, str]]
    implementation_cost: float
    effectiveness_score: float
    implementation_time: int
    side_effects: Dict[str, Any]
    prerequisites: List[str]
    created_at: datetime

class ThreatPropagationModelingAgent:
    """
    🌊 Threat Propagation Modeling Agent
    
    Models threat spread using advanced network analysis:
    - Epidemic-style propagation models
    - Graph neural network analysis  
    - Network topology optimization
    - Containment strategy generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agent_id = f"threat-propagation-{uuid.uuid4().hex[:8]}"
        self.is_running = False
        
        # Model parameters
        self.default_infection_rate = self.config.get('default_infection_rate', 0.3)
        self.default_recovery_rate = self.config.get('default_recovery_rate', 0.1)
        self.simulation_steps = self.config.get('simulation_steps', 1000)
        self.monte_carlo_runs = self.config.get('monte_carlo_runs', 100)
        
        # Network analysis parameters
        self.max_network_size = self.config.get('max_network_size', 10000)
        self.topology_update_interval = self.config.get('topology_update_interval', 3600)
        self.simulation_batch_size = self.config.get('simulation_batch_size', 10)
        
        # Storage and communication
        self.redis_pool = None
        self.db_pool = None
        
        # Network state
        self.network_graph = nx.Graph()
        self.node_registry: Dict[str, NetworkNode] = {}
        self.edge_registry: Dict[Tuple[str, str], NetworkEdge] = {}
        
        # Simulation state
        self.active_scenarios: Dict[str, PropagationScenario] = {}
        self.simulation_results: Dict[str, List[SimulationResult]] = {}
        self.containment_strategies: Dict[str, List[ContainmentStrategy]] = {}
        
        # Model cache
        self.model_cache: Dict[str, Any] = {}
        self.topology_cache: Dict[str, Any] = {}
        
        logger.info("ThreatPropagationModelingAgent initialized", agent_id=self.agent_id)

    async def initialize(self):
        """Initialize the threat propagation modeling agent"""
        try:
            # Initialize Redis connection
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                max_connections=20
            )
            
            # Initialize PostgreSQL connection
            self.db_pool = await asyncpg.create_pool(
                self.config.get('postgres_url', 'postgresql://localhost:5432/xorb'),
                min_size=5,
                max_size=20
            )
            
            # Initialize database schema
            await self._initialize_database()
            
            # Load network topology
            await self._load_network_topology()
            
            # Initialize threat models
            await self._initialize_threat_models()
            
            logger.info("ThreatPropagationModelingAgent initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ThreatPropagationModelingAgent", error=str(e))
            raise

    async def start_modeling(self):
        """Start the threat propagation modeling process"""
        if self.is_running:
            logger.warning("Modeling already running")
            return
            
        self.is_running = True
        logger.info("Starting threat propagation modeling process")
        
        try:
            # Start modeling loops
            topology_task = asyncio.create_task(self._topology_monitoring_loop())
            simulation_task = asyncio.create_task(self._simulation_loop())
            containment_task = asyncio.create_task(self._containment_optimization_loop())
            analysis_task = asyncio.create_task(self._network_analysis_loop())
            
            await asyncio.gather(topology_task, simulation_task, containment_task, analysis_task)
            
        except Exception as e:
            logger.error("Modeling process failed", error=str(e))
            raise
        finally:
            self.is_running = False

    async def stop_modeling(self):
        """Stop the threat propagation modeling process"""
        logger.info("Stopping threat propagation modeling process")
        self.is_running = False

    async def _topology_monitoring_loop(self):
        """Monitor and update network topology"""
        while self.is_running:
            try:
                logger.debug("Updating network topology")
                await self._update_network_topology()
                await asyncio.sleep(self.topology_update_interval)
                
            except Exception as e:
                logger.error("Topology monitoring failed", error=str(e))
                await asyncio.sleep(60)

    async def _simulation_loop(self):
        """Main simulation execution loop"""
        while self.is_running:
            try:
                # Process pending scenarios
                await self._process_simulation_queue()
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error("Simulation loop failed", error=str(e))
                await asyncio.sleep(60)

    async def _containment_optimization_loop(self):
        """Generate and optimize containment strategies"""
        while self.is_running:
            try:
                await self._generate_containment_strategies()
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                logger.error("Containment optimization failed", error=str(e))
                await asyncio.sleep(60)

    async def _network_analysis_loop(self):
        """Perform continuous network analysis"""
        while self.is_running:
            try:
                await self._analyze_network_vulnerabilities()
                await self._identify_critical_paths()
                await self._calculate_network_metrics()
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except Exception as e:
                logger.error("Network analysis failed", error=str(e))
                await asyncio.sleep(60)

    async def _load_network_topology(self):
        """Load current network topology from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Load nodes
                node_rows = await conn.fetch("SELECT * FROM network_nodes WHERE active = true")
                for row in node_rows:
                    node = NetworkNode(
                        node_id=row['node_id'],
                        node_type=row['node_type'],
                        state=NodeState(row['state']),
                        infection_time=row['infection_time'],
                        recovery_time=row['recovery_time'],
                        vulnerability_score=row['vulnerability_score'],
                        connectivity=row['connectivity'],
                        criticality=row['criticality'],
                        security_controls=json.loads(row['security_controls']),
                        geographic_location=(row['latitude'], row['longitude']) if row['latitude'] else None,
                        last_updated=row['last_updated']
                    )
                    self.node_registry[node.node_id] = node
                    self.network_graph.add_node(node.node_id, **asdict(node))
                
                # Load edges
                edge_rows = await conn.fetch("SELECT * FROM network_edges WHERE active = true")
                for row in edge_rows:
                    edge = NetworkEdge(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        connection_type=row['connection_type'],
                        weight=row['weight'],
                        bandwidth=row['bandwidth'],
                        latency=row['latency'],
                        security_level=row['security_level'],
                        last_activity=row['last_activity']
                    )
                    edge_key = (edge.source_id, edge.target_id)
                    self.edge_registry[edge_key] = edge
                    self.network_graph.add_edge(
                        edge.source_id, edge.target_id, 
                        weight=edge.weight, **asdict(edge)
                    )
                
                logger.info("Network topology loaded", 
                          nodes=len(self.node_registry), 
                          edges=len(self.edge_registry))
                
        except Exception as e:
            logger.error("Failed to load network topology", error=str(e))
            # Create minimal test topology
            await self._create_test_topology()

    async def _create_test_topology(self):
        """Create a test network topology for simulation"""
        logger.info("Creating test network topology")
        
        # Create test nodes
        node_types = ['server', 'workstation', 'mobile', 'iot']
        for i in range(100):
            node_id = f"node_{i:03d}"
            node = NetworkNode(
                node_id=node_id,
                node_type=random.choice(node_types),
                state=NodeState.SUSCEPTIBLE,
                infection_time=None,
                recovery_time=None,
                vulnerability_score=random.uniform(0.1, 0.9),
                connectivity=random.randint(1, 10),
                criticality=random.uniform(0.1, 1.0),
                security_controls=random.sample(['firewall', 'antivirus', 'ids', 'patch_mgmt'], 
                                              random.randint(1, 3)),
                geographic_location=(random.uniform(-90, 90), random.uniform(-180, 180)),
                last_updated=datetime.utcnow()
            )
            self.node_registry[node_id] = node
            self.network_graph.add_node(node_id, **asdict(node))
        
        # Create test edges with small-world properties
        for node_id in list(self.node_registry.keys()):
            # Connect to nearby nodes
            for _ in range(random.randint(2, 8)):
                target_id = random.choice(list(self.node_registry.keys()))
                if target_id != node_id and not self.network_graph.has_edge(node_id, target_id):
                    edge = NetworkEdge(
                        source_id=node_id,
                        target_id=target_id,
                        connection_type=random.choice(['ethernet', 'wifi', 'vpn', 'internet']),
                        weight=random.uniform(0.1, 1.0),
                        bandwidth=random.uniform(1, 1000),  # Mbps
                        latency=random.uniform(1, 100),  # ms
                        security_level=random.choice(['low', 'medium', 'high']),
                        last_activity=datetime.utcnow()
                    )
                    edge_key = (node_id, target_id)
                    self.edge_registry[edge_key] = edge
                    self.network_graph.add_edge(node_id, target_id, weight=edge.weight, **asdict(edge))
        
        logger.info("Test topology created", 
                   nodes=len(self.node_registry), 
                   edges=len(self.edge_registry))

    async def _update_network_topology(self):
        """Update network topology based on current state"""
        try:
            # This would integrate with network monitoring systems
            # For now, simulate dynamic changes
            
            # Randomly update node states
            for node_id, node in list(self.node_registry.items())[:10]:
                if random.random() < 0.1:  # 10% chance of state change
                    if node.state == NodeState.SUSCEPTIBLE and random.random() < 0.05:
                        node.state = NodeState.EXPOSED
                        node.infection_time = datetime.utcnow()
                    elif node.state == NodeState.EXPOSED and random.random() < 0.3:
                        node.state = NodeState.INFECTED
                    elif node.state == NodeState.INFECTED and random.random() < 0.2:
                        node.state = NodeState.RECOVERED
                        node.recovery_time = datetime.utcnow()
                    
                    node.last_updated = datetime.utcnow()
            
            # Update network metrics
            infected_count = len([n for n in self.node_registry.values() 
                                if n.state == NodeState.INFECTED])
            infection_rate = infected_count / len(self.node_registry) if self.node_registry else 0
            network_infection_rate.set(infection_rate)
            
            logger.debug("Network topology updated", infection_rate=infection_rate)
            
        except Exception as e:
            logger.error("Failed to update network topology", error=str(e))

    async def _initialize_threat_models(self):
        """Initialize threat propagation models"""
        self.threat_models = {
            PropagationModel.SIR: self._sir_model,
            PropagationModel.SEIR: self._seir_model,
            PropagationModel.SIS: self._sis_model,
            PropagationModel.NETWORK_DIFFUSION: self._network_diffusion_model,
            PropagationModel.CASCADE: self._cascade_model
        }
        
        logger.info("Threat models initialized", models=list(self.threat_models.keys()))

    async def create_propagation_scenario(self, 
                                        threat_type: ThreatType,
                                        initial_nodes: List[str],
                                        propagation_model: PropagationModel = PropagationModel.SEIR,
                                        custom_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new propagation scenario"""
        
        scenario_id = str(uuid.uuid4())
        
        # Create threat vector
        threat_vector = ThreatVector(
            threat_id=str(uuid.uuid4()),
            threat_type=threat_type,
            virulence=custom_parameters.get('virulence', 0.3) if custom_parameters else 0.3,
            persistence=custom_parameters.get('persistence', 0.8) if custom_parameters else 0.8,
            detection_difficulty=custom_parameters.get('detection_difficulty', 0.6) if custom_parameters else 0.6,
            payload_complexity=custom_parameters.get('payload_complexity', 0.5) if custom_parameters else 0.5,
            lateral_movement_capability=custom_parameters.get('lateral_movement', 0.7) if custom_parameters else 0.7,
            propagation_methods=['network', 'email', 'usb', 'web'],
            target_preferences={'server': 0.8, 'workstation': 0.6, 'mobile': 0.4, 'iot': 0.3},
            temporal_behavior={'peak_hours': [9, 17], 'weekend_factor': 0.5},
            created_at=datetime.utcnow()
        )
        
        # Create scenario
        scenario = PropagationScenario(
            scenario_id=scenario_id,
            network_topology=await self._export_network_topology(),
            threat_vector=threat_vector,
            initial_infection_nodes=initial_nodes,
            simulation_parameters=custom_parameters or self._get_default_simulation_parameters(),
            propagation_model=propagation_model,
            time_horizon=self.simulation_steps,
            containment_measures=[],
            created_at=datetime.utcnow()
        )
        
        self.active_scenarios[scenario_id] = scenario
        
        # Persist scenario
        await self._persist_scenario(scenario)
        
        propagation_models_created.inc()
        logger.info("Propagation scenario created", 
                   scenario_id=scenario_id, 
                   threat_type=threat_type.value,
                   initial_nodes=len(initial_nodes))
        
        return scenario_id

    async def run_propagation_simulation(self, scenario_id: str) -> SimulationResult:
        """Run a propagation simulation for a scenario"""
        
        if scenario_id not in self.active_scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        scenario = self.active_scenarios[scenario_id]
        
        logger.info("Running propagation simulation", scenario_id=scenario_id)
        start_time = time.time()
        
        with simulation_duration_seconds.time():
            # Select appropriate model
            model_func = self.threat_models.get(scenario.propagation_model)
            if not model_func:
                raise ValueError(f"Model {scenario.propagation_model} not implemented")
            
            # Run simulation
            result = await model_func(scenario)
            
            # Store result
            if scenario_id not in self.simulation_results:
                self.simulation_results[scenario_id] = []
            self.simulation_results[scenario_id].append(result)
            
            # Persist result
            await self._persist_simulation_result(result)
        
        simulation_duration = time.time() - start_time
        propagation_simulations_run.inc()
        
        logger.info("Simulation completed", 
                   scenario_id=scenario_id,
                   duration=simulation_duration,
                   final_infection_rate=result.final_infection_rate)
        
        return result

    async def _sir_model(self, scenario: PropagationScenario) -> SimulationResult:
        """Susceptible-Infected-Recovered model"""
        
        # Initialize node states
        node_states = {}
        for node_id in self.node_registry:
            if node_id in scenario.initial_infection_nodes:
                node_states[node_id] = NodeState.INFECTED
            else:
                node_states[node_id] = NodeState.SUSCEPTIBLE
        
        # Simulation parameters
        beta = scenario.simulation_parameters.get('infection_rate', self.default_infection_rate)
        gamma = scenario.simulation_parameters.get('recovery_rate', self.default_recovery_rate)
        
        # Track simulation state
        state_history = []
        critical_paths = []
        infection_times = {}
        
        for step in range(scenario.time_horizon):
            step_infections = []
            step_recoveries = []
            
            # Infection process
            infected_nodes = [n for n, s in node_states.items() if s == NodeState.INFECTED]
            
            for infected_node in infected_nodes:
                # Get neighbors
                neighbors = list(self.network_graph.neighbors(infected_node))
                
                for neighbor in neighbors:
                    if node_states[neighbor] == NodeState.SUSCEPTIBLE:
                        # Calculate infection probability
                        edge_data = self.network_graph.get_edge_data(infected_node, neighbor)
                        transmission_prob = beta * edge_data.get('weight', 1.0)
                        
                        # Apply node vulnerability
                        node_vuln = self.node_registry[neighbor].vulnerability_score
                        transmission_prob *= node_vuln
                        
                        # Apply threat-specific factors
                        threat_effectiveness = self._calculate_threat_effectiveness(
                            scenario.threat_vector, 
                            self.node_registry[neighbor]
                        )
                        transmission_prob *= threat_effectiveness
                        
                        if random.random() < transmission_prob:
                            step_infections.append(neighbor)
                            infection_times[neighbor] = step
                            critical_paths.append([infected_node, neighbor])
            
            # Recovery process
            for infected_node in infected_nodes:
                if random.random() < gamma:
                    step_recoveries.append(infected_node)
            
            # Update states
            for node in step_infections:
                node_states[node] = NodeState.INFECTED
            
            for node in step_recoveries:
                node_states[node] = NodeState.RECOVERED
            
            # Record state
            susceptible = len([n for n, s in node_states.items() if s == NodeState.SUSCEPTIBLE])
            infected = len([n for n, s in node_states.items() if s == NodeState.INFECTED])
            recovered = len([n for n, s in node_states.items() if s == NodeState.RECOVERED])
            
            state_history.append({
                'step': step,
                'susceptible': susceptible,
                'infected': infected,
                'recovered': recovered,
                'infection_rate': infected / len(node_states)
            })
            
            # Early termination if no infected nodes
            if infected == 0:
                break
        
        # Calculate results
        max_infected_step = max(state_history, key=lambda x: x['infected'])
        final_state = state_history[-1]
        
        # Identify bottleneck nodes (high betweenness centrality)
        betweenness = nx.betweenness_centrality(self.network_graph)
        bottleneck_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
        
        result = SimulationResult(
            scenario_id=scenario.scenario_id,
            final_infection_rate=final_state['infection_rate'],
            peak_infection_time=max_infected_step['step'],
            total_infected_nodes=final_state['infected'] + final_state['recovered'],
            propagation_speed=len(infection_times) / max(1, max(infection_times.values()) if infection_times else 1),
            containment_effectiveness=0.0,  # No containment in base model
            critical_paths=critical_paths[:100],  # Limit for storage
            bottleneck_nodes=bottleneck_nodes,
            recovery_time=len(state_history),
            simulation_metadata={
                'model': 'SIR',
                'parameters': {'beta': beta, 'gamma': gamma},
                'state_history': state_history[-10:]  # Last 10 steps
            },
            timestamp=datetime.utcnow()
        )
        
        return result

    async def _seir_model(self, scenario: PropagationScenario) -> SimulationResult:
        """Susceptible-Exposed-Infected-Recovered model"""
        
        # Initialize node states
        node_states = {}
        exposure_times = {}
        
        for node_id in self.node_registry:
            if node_id in scenario.initial_infection_nodes:
                node_states[node_id] = NodeState.INFECTED
            else:
                node_states[node_id] = NodeState.SUSCEPTIBLE
        
        # Simulation parameters
        beta = scenario.simulation_parameters.get('infection_rate', self.default_infection_rate)
        sigma = scenario.simulation_parameters.get('incubation_rate', 0.2)  # Rate of becoming infectious
        gamma = scenario.simulation_parameters.get('recovery_rate', self.default_recovery_rate)
        
        # Track simulation state
        state_history = []
        critical_paths = []
        infection_times = {}
        
        for step in range(scenario.time_horizon):
            step_exposures = []
            step_infections = []
            step_recoveries = []
            
            # Exposure process
            infected_nodes = [n for n, s in node_states.items() if s == NodeState.INFECTED]
            
            for infected_node in infected_nodes:
                neighbors = list(self.network_graph.neighbors(infected_node))
                
                for neighbor in neighbors:
                    if node_states[neighbor] == NodeState.SUSCEPTIBLE:
                        # Calculate exposure probability
                        edge_data = self.network_graph.get_edge_data(infected_node, neighbor)
                        transmission_prob = beta * edge_data.get('weight', 1.0)
                        
                        # Apply node vulnerability and threat effectiveness
                        node_vuln = self.node_registry[neighbor].vulnerability_score
                        threat_effectiveness = self._calculate_threat_effectiveness(
                            scenario.threat_vector, 
                            self.node_registry[neighbor]
                        )
                        transmission_prob *= node_vuln * threat_effectiveness
                        
                        if random.random() < transmission_prob:
                            step_exposures.append(neighbor)
                            exposure_times[neighbor] = step
                            critical_paths.append([infected_node, neighbor])
            
            # Infection process (exposed -> infected)
            exposed_nodes = [n for n, s in node_states.items() if s == NodeState.EXPOSED]
            for exposed_node in exposed_nodes:
                if random.random() < sigma:
                    step_infections.append(exposed_node)
                    infection_times[exposed_node] = step
            
            # Recovery process
            for infected_node in infected_nodes:
                if random.random() < gamma:
                    step_recoveries.append(infected_node)
            
            # Update states
            for node in step_exposures:
                node_states[node] = NodeState.EXPOSED
            
            for node in step_infections:
                node_states[node] = NodeState.INFECTED
            
            for node in step_recoveries:
                node_states[node] = NodeState.RECOVERED
            
            # Record state
            susceptible = len([n for n, s in node_states.items() if s == NodeState.SUSCEPTIBLE])
            exposed = len([n for n, s in node_states.items() if s == NodeState.EXPOSED])
            infected = len([n for n, s in node_states.items() if s == NodeState.INFECTED])
            recovered = len([n for n, s in node_states.items() if s == NodeState.RECOVERED])
            
            total_affected = exposed + infected + recovered
            infection_rate = total_affected / len(node_states)
            
            state_history.append({
                'step': step,
                'susceptible': susceptible,
                'exposed': exposed,
                'infected': infected,
                'recovered': recovered,
                'infection_rate': infection_rate
            })
            
            # Early termination
            if infected == 0 and exposed == 0:
                break
        
        # Calculate results
        max_affected_step = max(state_history, key=lambda x: x['exposed'] + x['infected'])
        final_state = state_history[-1]
        
        # Identify bottleneck nodes
        betweenness = nx.betweenness_centrality(self.network_graph)
        bottleneck_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
        
        result = SimulationResult(
            scenario_id=scenario.scenario_id,
            final_infection_rate=final_state['infection_rate'],
            peak_infection_time=max_affected_step['step'],
            total_infected_nodes=final_state['infected'] + final_state['recovered'],
            propagation_speed=len(infection_times) / max(1, max(infection_times.values()) if infection_times else 1),
            containment_effectiveness=0.0,
            critical_paths=critical_paths[:100],
            bottleneck_nodes=bottleneck_nodes,
            recovery_time=len(state_history),
            simulation_metadata={
                'model': 'SEIR',
                'parameters': {'beta': beta, 'sigma': sigma, 'gamma': gamma},
                'state_history': state_history[-10:]
            },
            timestamp=datetime.utcnow()
        )
        
        return result

    async def _network_diffusion_model(self, scenario: PropagationScenario) -> SimulationResult:
        """Network diffusion model based on graph properties"""
        
        # Initialize
        infected_nodes = set(scenario.initial_infection_nodes)
        newly_infected = set(scenario.initial_infection_nodes)
        all_infected = set(scenario.initial_infection_nodes)
        
        # Parameters
        threshold = scenario.simulation_parameters.get('infection_threshold', 0.3)
        decay = scenario.simulation_parameters.get('decay_rate', 0.05)
        
        state_history = []
        critical_paths = []
        step = 0
        
        while newly_infected and step < scenario.time_horizon:
            next_infected = set()
            
            for infected_node in newly_infected:
                neighbors = list(self.network_graph.neighbors(infected_node))
                
                for neighbor in neighbors:
                    if neighbor not in all_infected:
                        # Calculate influence
                        edge_data = self.network_graph.get_edge_data(infected_node, neighbor)
                        influence = edge_data.get('weight', 1.0)
                        
                        # Factor in node properties
                        neighbor_node = self.node_registry[neighbor]
                        vulnerability = neighbor_node.vulnerability_score
                        
                        # Calculate total influence from all infected neighbors
                        infected_neighbors = [n for n in neighbors if n in all_infected]
                        total_influence = sum(
                            self.network_graph.get_edge_data(neighbor, inf_n).get('weight', 1.0)
                            for inf_n in infected_neighbors
                        )
                        
                        # Apply threshold model
                        if total_influence * vulnerability > threshold:
                            next_infected.add(neighbor)
                            critical_paths.append([infected_node, neighbor])
            
            # Apply decay to infection probability
            if step > 0:
                threshold *= (1 + decay)
            
            # Update state
            newly_infected = next_infected
            all_infected.update(newly_infected)
            
            infection_rate = len(all_infected) / len(self.node_registry)
            
            state_history.append({
                'step': step,
                'newly_infected': len(newly_infected),
                'total_infected': len(all_infected),
                'infection_rate': infection_rate
            })
            
            step += 1
        
        # Calculate results
        max_step = max(state_history, key=lambda x: x['newly_infected']) if state_history else state_history[0]
        final_state = state_history[-1] if state_history else {'infection_rate': 0, 'total_infected': 0}
        
        # Identify bottlenecks
        betweenness = nx.betweenness_centrality(self.network_graph)
        bottleneck_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
        
        result = SimulationResult(
            scenario_id=scenario.scenario_id,
            final_infection_rate=final_state['infection_rate'],
            peak_infection_time=max_step['step'],
            total_infected_nodes=final_state['total_infected'],
            propagation_speed=final_state['total_infected'] / max(1, step),
            containment_effectiveness=0.0,
            critical_paths=critical_paths[:100],
            bottleneck_nodes=bottleneck_nodes,
            recovery_time=step,
            simulation_metadata={
                'model': 'network_diffusion',
                'parameters': {'threshold': threshold, 'decay': decay},
                'state_history': state_history[-10:]
            },
            timestamp=datetime.utcnow()
        )
        
        return result

    async def _sis_model(self, scenario: PropagationScenario) -> SimulationResult:
        """Susceptible-Infected-Susceptible model for persistent threats"""
        # Similar to SIR but recovered nodes become susceptible again
        # Implementation would be similar to SIR with additional transition
        return await self._sir_model(scenario)  # Simplified for now

    async def _cascade_model(self, scenario: PropagationScenario) -> SimulationResult:
        """Cascade failure model"""
        # Implementation for cascade failures in networks
        return await self._network_diffusion_model(scenario)  # Simplified for now

    def _calculate_threat_effectiveness(self, threat: ThreatVector, target_node: NetworkNode) -> float:
        """Calculate how effective a threat is against a specific node"""
        
        base_effectiveness = threat.virulence
        
        # Adjust for node type preferences
        node_type_factor = threat.target_preferences.get(target_node.node_type, 0.5)
        
        # Adjust for security controls
        security_reduction = 0.0
        for control in target_node.security_controls:
            if control == 'firewall':
                security_reduction += 0.2
            elif control == 'antivirus':
                security_reduction += 0.15
            elif control == 'ids':
                security_reduction += 0.1
            elif control == 'patch_mgmt':
                security_reduction += 0.25
        
        # Apply temporal factors
        current_hour = datetime.utcnow().hour
        if current_hour in threat.temporal_behavior.get('peak_hours', []):
            base_effectiveness *= 1.2
        
        # Final calculation
        effectiveness = base_effectiveness * node_type_factor * (1.0 - min(0.8, security_reduction))
        
        return max(0.0, min(1.0, effectiveness))

    async def _get_default_simulation_parameters(self) -> Dict[str, Any]:
        """Get default simulation parameters"""
        return {
            'infection_rate': self.default_infection_rate,
            'recovery_rate': self.default_recovery_rate,
            'incubation_rate': 0.2,
            'infection_threshold': 0.3,
            'decay_rate': 0.05,
            'monte_carlo_runs': self.monte_carlo_runs,
            'containment_delay': 0,
            'containment_effectiveness': 0.8
        }

    async def _export_network_topology(self) -> Dict[str, Any]:
        """Export current network topology for scenario storage"""
        return {
            'nodes': {node_id: asdict(node) for node_id, node in self.node_registry.items()},
            'edges': {f"{e[0]}-{e[1]}": asdict(edge) for e, edge in self.edge_registry.items()},
            'graph_metrics': await self._calculate_basic_graph_metrics(),
            'exported_at': datetime.utcnow().isoformat()
        }

    async def _calculate_basic_graph_metrics(self) -> Dict[str, Any]:
        """Calculate basic graph metrics"""
        if not self.network_graph.nodes():
            return {}
        
        return {
            'node_count': self.network_graph.number_of_nodes(),
            'edge_count': self.network_graph.number_of_edges(),
            'density': nx.density(self.network_graph),
            'average_clustering': nx.average_clustering(self.network_graph),
            'average_shortest_path': nx.average_shortest_path_length(self.network_graph) 
                                   if nx.is_connected(self.network_graph) else -1,
            'diameter': nx.diameter(self.network_graph) if nx.is_connected(self.network_graph) else -1
        }

    async def _process_simulation_queue(self):
        """Process pending simulations from queue"""
        try:
            # Get pending scenarios from Redis queue
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            pending_scenarios = await redis.lrange('simulation_queue', 0, self.simulation_batch_size - 1)
            
            if pending_scenarios:
                await redis.ltrim('simulation_queue', len(pending_scenarios), -1)
                
                tasks = []
                for scenario_data in pending_scenarios:
                    scenario_info = json.loads(scenario_data)
                    scenario_id = scenario_info['scenario_id']
                    
                    if scenario_id in self.active_scenarios:
                        task = asyncio.create_task(self.run_propagation_simulation(scenario_id))
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.info("Processed simulation batch", count=len(tasks))
        
        except Exception as e:
            logger.error("Failed to process simulation queue", error=str(e))

    async def _generate_containment_strategies(self):
        """Generate optimized containment strategies"""
        try:
            # Analyze recent simulation results to identify optimal containment points
            for scenario_id, results in self.simulation_results.items():
                if not results:
                    continue
                
                latest_result = results[-1]
                strategies = await self._optimize_containment_for_result(latest_result)
                
                if scenario_id not in self.containment_strategies:
                    self.containment_strategies[scenario_id] = []
                
                self.containment_strategies[scenario_id].extend(strategies)
                
                # Persist strategies
                for strategy in strategies:
                    await self._persist_containment_strategy(strategy)
                
                containment_strategies_generated.inc(len(strategies))
            
            logger.debug("Containment strategies generated")
            
        except Exception as e:
            logger.error("Failed to generate containment strategies", error=str(e))

    async def _optimize_containment_for_result(self, result: SimulationResult) -> List[ContainmentStrategy]:
        """Generate optimal containment strategies for a simulation result"""
        strategies = []
        
        # Strategy 1: Isolate bottleneck nodes
        if result.bottleneck_nodes:
            strategy = ContainmentStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_type='node_isolation',
                target_nodes=result.bottleneck_nodes[:5],  # Top 5 bottlenecks
                target_edges=[],
                implementation_cost=len(result.bottleneck_nodes[:5]) * 10.0,
                effectiveness_score=0.8,
                implementation_time=30,  # 30 minutes
                side_effects={'connectivity_loss': 0.2},
                prerequisites=['admin_access', 'network_control'],
                created_at=datetime.utcnow()
            )
            strategies.append(strategy)
        
        # Strategy 2: Cut critical paths
        if result.critical_paths:
            critical_edges = [(path[0], path[1]) for path in result.critical_paths[:10]]
            strategy = ContainmentStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_type='edge_cutting',
                target_nodes=[],
                target_edges=critical_edges,
                implementation_cost=len(critical_edges) * 5.0,
                effectiveness_score=0.6,
                implementation_time=15,
                side_effects={'network_fragmentation': 0.3},
                prerequisites=['network_admin'],
                created_at=datetime.utcnow()
            )
            strategies.append(strategy)
        
        # Strategy 3: Targeted immunization
        # Find high-degree nodes that aren't in critical paths
        degrees = dict(self.network_graph.degree())
        high_degree_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
        
        immunization_targets = [node for node in high_degree_nodes 
                              if node not in result.bottleneck_nodes[:5]][:5]
        
        if immunization_targets:
            strategy = ContainmentStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_type='immunization',
                target_nodes=immunization_targets,
                target_edges=[],
                implementation_cost=len(immunization_targets) * 15.0,
                effectiveness_score=0.7,
                implementation_time=60,
                side_effects={'resource_usage': 0.4},
                prerequisites=['patch_management', 'endpoint_access'],
                created_at=datetime.utcnow()
            )
            strategies.append(strategy)
        
        return strategies

    async def _analyze_network_vulnerabilities(self):
        """Analyze network for structural vulnerabilities"""
        try:
            # Calculate centrality measures
            betweenness = nx.betweenness_centrality(self.network_graph)
            closeness = nx.closeness_centrality(self.network_graph)
            degree = nx.degree_centrality(self.network_graph)
            
            # Identify critical nodes
            critical_nodes = []
            for node_id in self.network_graph.nodes():
                criticality_score = (
                    betweenness.get(node_id, 0) * 0.4 +
                    closeness.get(node_id, 0) * 0.3 +
                    degree.get(node_id, 0) * 0.3
                )
                
                if criticality_score > 0.7:
                    critical_nodes.append((node_id, criticality_score))
            
            # Update node criticality scores
            for node_id, score in critical_nodes:
                if node_id in self.node_registry:
                    self.node_registry[node_id].criticality = score
            
            logger.debug("Network vulnerability analysis completed", critical_nodes=len(critical_nodes))
            
        except Exception as e:
            logger.error("Network vulnerability analysis failed", error=str(e))

    async def _identify_critical_paths(self):
        """Identify critical paths for threat propagation"""
        try:
            # Find shortest paths between high-value nodes
            high_value_nodes = [
                node_id for node_id, node in self.node_registry.items()
                if node.criticality > 0.8
            ]
            
            critical_paths = []
            for source in high_value_nodes[:5]:  # Limit computation
                for target in high_value_nodes[:5]:
                    if source != target:
                        try:
                            path = nx.shortest_path(self.network_graph, source, target)
                            if len(path) > 2:  # Exclude direct connections
                                critical_paths.append(path)
                        except nx.NetworkXNoPath:
                            continue
            
            # Store critical paths for use in containment strategies
            self.topology_cache['critical_paths'] = critical_paths[:20]  # Store top 20
            
            logger.debug("Critical paths identified", count=len(critical_paths))
            
        except Exception as e:
            logger.error("Critical path identification failed", error=str(e))

    async def _calculate_network_metrics(self):
        """Calculate and update network metrics"""
        try:
            metrics = await self._calculate_basic_graph_metrics()
            
            # Additional metrics
            if self.network_graph.nodes():
                # Assortativity (tendency for similar nodes to connect)
                try:
                    assortativity = nx.degree_assortativity_coefficient(self.network_graph)
                    metrics['assortativity'] = assortativity
                except:
                    metrics['assortativity'] = 0.0
                
                # Modularity (community structure)
                try:
                    communities = nx.community.greedy_modularity_communities(self.network_graph)
                    modularity = nx.community.modularity(self.network_graph, communities)
                    metrics['modularity'] = modularity
                    metrics['community_count'] = len(communities)
                except:
                    metrics['modularity'] = 0.0
                    metrics['community_count'] = 0
            
            # Store metrics
            self.topology_cache['network_metrics'] = metrics
            
            # Update Prometheus metrics
            if 'density' in metrics:
                containment_effectiveness.set(1.0 - metrics['density'])  # Lower density = easier containment
            
            logger.debug("Network metrics calculated", metrics=metrics)
            
        except Exception as e:
            logger.error("Network metrics calculation failed", error=str(e))

    async def _persist_scenario(self, scenario: PropagationScenario):
        """Persist scenario to database"""
        try:
            async with self.db_pool.acquire() as conn:
                scenario_data = json.dumps(asdict(scenario), default=str)
                await conn.execute("""
                    INSERT INTO propagation_scenarios 
                    (scenario_id, threat_type, scenario_data, created_at)
                    VALUES ($1, $2, $3, $4)
                """, scenario.scenario_id, scenario.threat_vector.threat_type.value, 
                    scenario_data, scenario.created_at)
        
        except Exception as e:
            logger.error("Failed to persist scenario", scenario_id=scenario.scenario_id, error=str(e))

    async def _persist_simulation_result(self, result: SimulationResult):
        """Persist simulation result to database"""
        try:
            async with self.db_pool.acquire() as conn:
                result_data = json.dumps(asdict(result), default=str)
                await conn.execute("""
                    INSERT INTO simulation_results 
                    (scenario_id, final_infection_rate, peak_infection_time, 
                     total_infected_nodes, result_data, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, result.scenario_id, result.final_infection_rate, 
                    result.peak_infection_time, result.total_infected_nodes,
                    result_data, result.timestamp)
        
        except Exception as e:
            logger.error("Failed to persist simulation result", 
                        scenario_id=result.scenario_id, error=str(e))

    async def _persist_containment_strategy(self, strategy: ContainmentStrategy):
        """Persist containment strategy to database"""
        try:
            async with self.db_pool.acquire() as conn:
                strategy_data = json.dumps(asdict(strategy), default=str)
                await conn.execute("""
                    INSERT INTO containment_strategies 
                    (strategy_id, strategy_type, effectiveness_score, 
                     implementation_cost, strategy_data, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, strategy.strategy_id, strategy.strategy_type, 
                    strategy.effectiveness_score, strategy.implementation_cost,
                    strategy_data, strategy.created_at)
        
        except Exception as e:
            logger.error("Failed to persist containment strategy", 
                        strategy_id=strategy.strategy_id, error=str(e))

    async def _initialize_database(self):
        """Initialize database schema"""
        async with self.db_pool.acquire() as conn:
            # Network topology tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS network_nodes (
                    node_id VARCHAR PRIMARY KEY,
                    node_type VARCHAR NOT NULL,
                    state VARCHAR NOT NULL,
                    infection_time TIMESTAMP,
                    recovery_time TIMESTAMP,
                    vulnerability_score REAL NOT NULL,
                    connectivity INTEGER NOT NULL,
                    criticality REAL NOT NULL,
                    security_controls JSONB NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    active BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS network_edges (
                    id SERIAL PRIMARY KEY,
                    source_id VARCHAR NOT NULL,
                    target_id VARCHAR NOT NULL,
                    connection_type VARCHAR NOT NULL,
                    weight REAL NOT NULL,
                    bandwidth REAL NOT NULL,
                    latency REAL NOT NULL,
                    security_level VARCHAR NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_id, target_id)
                )
            """)
            
            # Simulation tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS propagation_scenarios (
                    scenario_id VARCHAR PRIMARY KEY,
                    threat_type VARCHAR NOT NULL,
                    scenario_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    id SERIAL PRIMARY KEY,
                    scenario_id VARCHAR NOT NULL,
                    final_infection_rate REAL NOT NULL,
                    peak_infection_time INTEGER NOT NULL,
                    total_infected_nodes INTEGER NOT NULL,
                    result_data JSONB NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS containment_strategies (
                    strategy_id VARCHAR PRIMARY KEY,
                    strategy_type VARCHAR NOT NULL,
                    effectiveness_score REAL NOT NULL,
                    implementation_cost REAL NOT NULL,
                    strategy_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON network_nodes(node_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_state ON network_nodes(state)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON network_edges(source_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON network_edges(target_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_scenarios_threat ON propagation_scenarios(threat_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_results_scenario ON simulation_results(scenario_id)")

    async def get_modeling_status(self) -> Dict[str, Any]:
        """Get current modeling status"""
        network_metrics = self.topology_cache.get('network_metrics', {})
        
        return {
            "status": "running" if self.is_running else "stopped",
            "network_size": {
                "nodes": len(self.node_registry),
                "edges": len(self.edge_registry)
            },
            "network_metrics": network_metrics,
            "active_scenarios": len(self.active_scenarios),
            "simulation_results": sum(len(results) for results in self.simulation_results.values()),
            "containment_strategies": sum(len(strategies) for strategies in self.containment_strategies.values()),
            "critical_paths": len(self.topology_cache.get('critical_paths', [])),
            "current_infection_rate": network_infection_rate._value.get(),
            "last_topology_update": max((node.last_updated for node in self.node_registry.values()), 
                                      default=datetime.utcnow()).isoformat()
        }

    async def get_scenario_results(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific scenario"""
        if scenario_id not in self.simulation_results:
            return None
        
        results = self.simulation_results[scenario_id]
        strategies = self.containment_strategies.get(scenario_id, [])
        
        return {
            "scenario_id": scenario_id,
            "simulation_count": len(results),
            "latest_result": asdict(results[-1]) if results else None,
            "containment_strategies": [asdict(s) for s in strategies],
            "average_infection_rate": np.mean([r.final_infection_rate for r in results]),
            "average_containment_effectiveness": np.mean([r.containment_effectiveness for r in results])
        }

    async def shutdown(self):
        """Shutdown the threat propagation modeling agent"""
        logger.info("Shutting down ThreatPropagationModelingAgent")
        
        self.is_running = False
        
        # Close connections
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("ThreatPropagationModelingAgent shutdown complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XORB Threat Propagation Modeling Agent")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--simulation-steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--network-size", type=int, default=100, help="Test network size")
    
    args = parser.parse_args()
    
    config = {
        'simulation_steps': args.simulation_steps,
        'max_network_size': args.network_size,
        'redis_url': 'redis://localhost:6379',
        'postgres_url': 'postgresql://localhost:5432/xorb'
    }
    
    async def main():
        agent = ThreatPropagationModelingAgent(config)
        await agent.initialize()
        
        # Create test scenario
        scenario_id = await agent.create_propagation_scenario(
            threat_type=ThreatType.MALWARE,
            initial_nodes=['node_000', 'node_001'],
            propagation_model=PropagationModel.SEIR
        )
        
        # Run simulation
        result = await agent.run_propagation_simulation(scenario_id)
        print(f"Simulation completed: {result.final_infection_rate:.2%} infection rate")
        
        # Start continuous modeling
        await agent.start_modeling()
    
    asyncio.run(main())