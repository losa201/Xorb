import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import random
import json
from datetime import datetime
import logging
from abc import ABC, abstractmethod

class WorldState:
    """Represents the current state of the simulation world"""
    def __init__(self):
        # Network topology graph (nodes represent systems, edges represent connections)
        self.topology_graph = nx.DiGraph()
        # Resource availability map (CPU, memory, bandwidth)
        self.resource_map: Dict[str, Dict[str, float]] = {}
        # Agent states (position, status, resources)
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        # Threat zones (areas with high detection probability)
        self.threat_zones: List[Dict[str, Any]] = []
        # Communication network state
        self.comms_network: Dict[str, Any] = {}
        # Current simulation time
        self.timestamp = datetime.now().isoformat()

    def update_topology(self, updates: List[Dict[str, Any]]) -> None:
        """Update the network topology with new information"""
        for update in updates:
            if update["type"] == "node_add":
                self.topology_graph.add_node(update["node_id"], **update.get("attributes", {}))
            elif update["type"] == "edge_add":
                self.topology_graph.add_edge(update["source"], update["target"], **update.get("attributes", {}))
            elif update["type"] == "node_remove":
                if update["node_id"] in self.topology_graph:
                    self.topology_graph.remove_node(update["node_id"])
            elif update["type"] == "edge_remove":
                if self.topology_graph.has_edge(update["source"], update["target"]):
                    self.topology_graph.remove_edge(update["source"], update["target"])

    def get_visible_topology(self, agent_id: str) -> Dict[str, Any]:
        """Get the portion of the topology visible to a specific agent"""
        if agent_id not in self.agent_states:
            return {"nodes": [], "edges": []}
        
        # Get the subgraph visible to this agent based on their current knowledge
        visible_nodes = list(nx.single_source_shortest_path_length(
            self.topology_graph, 
            self.agent_states[agent_id]["position"],
            cutoff=2  # Agent can see 2 hops away
        ).keys())
        
        subgraph = self.topology_graph.subgraph(visible_nodes)
        return {
            "nodes": [{"id": n, **subgraph.nodes[n]} for n in subgraph.nodes()],
            "edges": [{"source": u, "target": v, **subgraph.edges[u, v]} for u, v in subgraph.edges()]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert world state to serializable dictionary"""
        return {
            "timestamp": self.timestamp,
            "topology": {
                "nodes": [{"id": n, **self.topology_graph.nodes[n]} for n in self.topology_graph.nodes()],
                "edges": [{"source": u, "target": v, **self.topology_graph.edges[u, v]} for u, v in self.topology_graph.edges()]
            },
            "resource_map": self.resource_map,
            "agent_states": self.agent_states,
            "threat_zones": self.threat_zones,
            "comms_network": self.comms_network
        }

class SimulationEngine:
    """Core simulation engine for the XORB framework"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_state = WorldState()
        self.agents: Dict[str, "BaseAgent"] = {}
        self.logger = logging.getLogger("SimulationEngine")
        self._initialize_simulation()

    def _initialize_simulation(self) -> None:
        """Initialize the simulation with default or loaded state"""
        # Initialize with default topology if configured
        if self.config.get("generate_default_topology", True):
            self._generate_default_topology()
            
        # Initialize resource map
        self._initialize_resources()
        
        # Initialize communication network
        self._initialize_comms()

    def _generate_default_topology(self) -> None:
        """Generate a default network topology for testing"""
        # Create a simple star topology
        self.world_state.topology_graph.add_node("central_hub", type="server", critical=True)
        
        for i in range(1, 6):  # Create 5 peripheral nodes
            node_id = f"node_{i}"
            self.world_state.topology_graph.add_node(
                node_id, 
                type="workstation",
                security_level=random.choice(["low", "medium", "high"])
            )
            self.world_state.topology_graph.add_edge(node_id, "central_hub", weight=1.0)
            self.world_state.topology_graph.add_edge("central_hub", node_id, weight=1.0)

    def _initialize_resources(self) -> None:
        """Initialize resource availability across the network"""
        for node in self.world_state.topology_graph.nodes():
            self.world_state.resource_map[node] = {
                "cpu": random.uniform(0.5, 0.8),  # 50-80% available
                "memory": random.uniform(0.4, 0.7),
                "bandwidth": random.uniform(0.6, 0.9)
            }

    def _initialize_comms(self) -> None:
        """Initialize communication network state"""
        self.world_state.comms_network = {
            "latency": {
                "mean": 50.0,  # ms
                "std_dev": 10.0
            },
            "bandwidth": {
                "available": 1000.0,  # Mbps
                "used": 300.0
            },
            "reliability": 0.98
        }

    def register_agent(self, agent: "BaseAgent") -> None:
        """Register a new agent with the simulation"""
        self.agents[agent.agent_id] = agent
        self.world_state.agent_states[agent.agent_id] = {
            "position": agent.initial_position,
            "resources": agent.get_resources(),
            "status": "active",
            "last_action": None,
            "stealth_level": agent.stealth_level
        }

    def step(self) -> None:
        """Run a single simulation step"""
        # Update world state
        self._update_world_state()
        
        # Agents take actions
        for agent_id, agent in self.agents.items():
            if self.world_state.agent_states[agent_id]["status"] == "active":
                try:
                    action = agent.perceive_and_act(self.world_state)
                    self._execute_action(agent_id, action)
                except Exception as e:
                    self.logger.error(f"Error in agent {agent_id}: {str(e)}")
                    self.world_state.agent_states[agent_id]["status"] = "failed"
        
        # Update simulation timestamp
        self.world_state.timestamp = datetime.now().isoformat()

    def _update_world_state(self) -> None:
        """Update the world state based on simulation rules"""
        # Update threat zones based on agent activities
        self._update_threat_zones()
        
        # Update resource availability
        self._update_resources()
        
        # Update communication network
        self._update_comms()

    def _update_threat_zones(self) -> None:
        """Update threat zones based on agent activities"""
        # Clear old threat zones
        self.world_state.threat_zones = []
        
        # Create new threat zones based on agent actions
        for agent_id, state in self.world_state.agent_states.items():
            if state["status"] == "active" and state.get("last_action"):
                # Higher stealth level means less threat generation
                threat_level = max(0.1, 1.0 - state["stealth_level"])
                
                # Create a threat zone around the agent's position
                self.world_state.threat_zones.append({
                    "center": state["position"],
                    "radius": threat_level * 2,  # Higher threat = larger radius
                    "intensity": threat_level,
                    "duration": 5  # Lasts for 5 simulation steps
                })

    def _update_resources(self) -> None:
        """Update resource availability across the network"""
        # Simple resource regeneration model
        for node, resources in self.world_state.resource_map.items():
            for resource_type in resources:
                # Resources slowly regenerate over time
                resources[resource_type] = min(1.0, resources[resource_type] + 0.01)

    def _update_comms(self) -> None:
        """Update communication network state"""
        # Simple model: bandwidth usage fluctuates based on agent activity
        active_agents = sum(1 for state in self.world_state.agent_states.values() 
                          if state["status"] == "active")
        
        # More active agents = more bandwidth usage
        base_usage = 300.0
        activity_factor = active_agents * 20.0
        
        self.world_state.comms_network["bandwidth"]["used"] = min(
            self.world_state.comms_network["bandwidth"]["available"],
            base_usage + activity_factor
        )

    def _execute_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Execute an agent's action and update world state"""
        action_type = action.get("type")
        
        if action_type == "move":
            self._handle_move_action(agent_id, action)
        elif action_type == "exploit":
            self._handle_exploit_action(agent_id, action)
        elif action_type == "exfiltrate":
            self._handle_exfiltrate_action(agent_id, action)
        elif action_type == "maintain_stealth":
            self._handle_stealth_action(agent_id, action)
        
        # Update agent's last action time
        self.world_state.agent_states[agent_id]["last_action"] = action
        
    def _handle_move_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Handle a move action from an agent"""
        target = action.get("target")
        if target in self.world_state.topology_graph.nodes():
            self.world_state.agent_states[agent_id]["position"] = target
            
    def _handle_exploit_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Handle an exploit action from an agent"""
        target = action.get("target")
        vulnerability = action.get("vulnerability")
        
        if target in self.world_state.topology_graph.nodes():
            # Simple exploitation model
            node = self.world_state.topology_graph.nodes()[target]
            if vulnerability in node.get("vulnerabilities", []):
                # Success! Agent gains access
                node["access_level"] = max(node.get("access_level", 0), 2)
                
    def _handle_exfiltrate_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Handle a data exfiltration action"""
        target = action.get("target")
        data_size = action.get("data_size", 1.0)
        
        if target in self.world_state.topology_graph.nodes():
            # Simple exfiltration model
            comms = self.world_state.comms_network
            available_bandwidth = comms["bandwidth"]["available"] - comms["bandwidth"]["used"]
            
            if available_bandwidth > data_size * 10:
                # Success! Data exfiltrated
                comms["bandwidth"]["used"] += data_size * 10
                
    def _handle_stealth_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Handle a stealth maintenance action"""
        # Simple stealth model - reduces threat generation temporarily
        self.world_state.agent_states[agent_id]["stealth_level"] = min(1.0, 
            self.world_state.agent_states[agent_id].get("stealth_level", 0.5) + 0.1)

    def run_simulation(self, steps: int) -> None:
        """Run the simulation for a specified number of steps"""
        for _ in range(steps):
            self.step()
            
    def save_state(self, path: str) -> None:
        """Save the current simulation state to a file"""
        with open(path, 'w') as f:
            json.dump(self.world_state.to_dict(), f, indent=2)

    def load_state(self, path: str) -> None:
        """Load a simulation state from a file"""
        with open(path, 'r') as f:
            state_data = json.load(f)
            # Reconstruct the topology graph
            self.world_state.topology_graph = nx.DiGraph()
            for node in state_data["topology"]["nodes"]:
                self.world_state.topology_graph.add_node(node["id"], **node)
            for edge in state_data["topology"]["edges"]:
                self.world_state.topology_graph.add_edge(edge["source"], edge["target"], **edge)
            
            # Restore other state components
            self.world_state.resource_map = state_data["resource_map"]
            self.world_state.agent_states = state_data["agent_states"]
            self.world_state.threat_zones = state_data["threat_zones"]
            self.world_state.comms_network = state_data["comms_network"]
            self.world_state.timestamp = state_data["timestamp"]


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    def __init__(self, agent_id: str, initial_position: str, stealth_level: float = 0.5):
        self.agent_id = agent_id
        self.initial_position = initial_position
        self.stealth_level = stealth_level
        self.logger = logging.getLogger(f"Agent-{agent_id}")

    @abstractmethod
    def perceive_and_act(self, world_state: WorldState) -> Dict[str, Any]:
        """Perceive the world and decide on an action"""
        pass

    def get_resources(self) -> Dict[str, float]:
        """Get the agent's current resource levels"""
        return {
            "cpu": random.uniform(0.7, 1.0),  # Agents have varying resource availability
            "memory": random.uniform(0.6, 0.9),
            "bandwidth": random.uniform(0.5, 0.8)
        }

    def is_action_valid(self, action: Dict[str, Any], world_state: WorldState) -> bool:
        """Check if an action is valid given the current world state"""
        action_type = action.get("type")
        
        if action_type == "move":
            # Check if target exists and is reachable
            target = action.get("target")
            return target in world_state.topology_graph.nodes()
        elif action_type == "exploit":
            # Check if target exists and has vulnerabilities
            target = action.get("target")
            return (target in world_state.topology_graph.nodes() and 
                   len(world_state.topology_graph.nodes()[target].get("vulnerabilities", [])) > 0)
        elif action_type == "exfiltrate":
            # Check if target exists and has data
            target = action.get("target")
            return (target in world_state.topology_graph.nodes() and 
                   world_state.topology_graph.nodes()[target].get("data_size", 0) > 0)
        
        return True  # Default to valid for unknown action types

    def calculate_action_cost(self, action: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the resource cost of an action"""
        action_type = action.get("type", "unknown")
        
        costs = {
            "cpu": 0.1,
            "memory": 0.05,
            "bandwidth": 0.05
        }
        
        if action_type == "exploit":
            costs["cpu"] += 0.2
            costs["bandwidth"] += 0.1
        elif action_type == "exfiltrate":
            data_size = action.get("data_size", 1.0)
            costs["cpu"] += 0.15
            costs["memory"] += 0.1
            costs["bandwidth"] += 0.2 * data_size
        
        return costs

    def has_sufficient_resources(self, action: Dict[str, Any], world_state: WorldState) -> bool:
        """Check if the agent has sufficient resources to perform an action"""
        resources = world_state.agent_states[self.agent_id]["resources"]
        action_cost = self.calculate_action_cost(action)
        
        return all(
            resources[res_type] >= cost 
            for res_type, cost in action_cost.items()
        )

    def consume_resources(self, action: Dict[str, Any], world_state: WorldState) -> None:
        """Consume resources after performing an action"""
        resources = world_state.agent_states[self.agent_id]["resources"]
        action_cost = self.calculate_action_cost(action)
        
        for res_type, cost in action_cost.items():
            resources[res_type] = max(0.0, resources[res_type] - cost)

    def check_detection(self, world_state: WorldState) -> bool:
        """Check if the agent is detected based on threat zones and stealth level"""
        position = world_state.agent_states[self.agent_id]["position"]
        
        # Check for threat zones
        for zone in world_state.threat_zones:
            if self._is_in_zone(position, zone):
                # Detection probability based on threat intensity and agent stealth
                detection_chance = zone["intensity"] * (1.0 - self.stealth_level)
                if random.random() < detection_chance:
                    return True
        
        return False

    def _is_in_zone(self, position: str, zone: Dict[str, Any]) -> bool:
        """Check if a position is within a threat zone"""
        # Simple check - in a real implementation we would use actual coordinates
        return position == zone["center"] or random.random() < 0.3  # Simplified for demo

    def handle_detection(self, world_state: WorldState) -> None:
        """Handle agent detection by defensive systems"""
        self.logger.info(f"Agent {self.agent_id} detected!")
        world_state.agent_states[self.agent_id]["status"] = "compromised"

    def get_visible_topology(self, world_state: WorldState) -> Dict[str, Any]:
        """Get the portion of the topology visible to this agent"""
        return world_state.get_visible_topology(self.agent_id)

    def get_nearby_agents(self, world_state: WorldState) -> List[str]:
        """Get list of agents in the same or adjacent nodes"""
        position = world_state.agent_states[self.agent_id]["position"]
        nearby_nodes = [position] + list(world_state.topology_graph.neighbors(position))
        
        return [
            aid for aid, state in world_state.agent_states.items()
            if state["position"] in nearby_nodes and aid != self.agent_id
        ]

    def get_resource_levels(self, world_state: WorldState) -> Dict[str, float]:
        """Get the agent's current resource levels"""
        return world_state.agent_states[self.agent_id]["resources"].copy()

    def get_threat_level(self, world_state: WorldState) -> float:
        """Get the current threat level for this agent"""
        position = world_state.agent_states[self.agent_id]["position"]
        
        threat_level = 0.0
        for zone in world_state.threat_zones:
            if self._is_in_zone(position, zone):
                threat_level = max(threat_level, zone["intensity"])
        
        return threat_level

    def get_mission_context(self, world_state: WorldState) -> Dict[str, Any]:
        """Get context information for mission planning"""
        return {
            "visible_topology": self.get_visible_topology(world_state),
            "nearby_agents": self.get_nearby_agents(world_state),
            "resource_levels": self.get_resource_levels(world_state),
            "threat_level": self.get_threat_level(world_state),
            "comms_status": world_state.comms_network
        }