import abc
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from xorbfw.core.simulation import SimulationObject, WorldState

class AgentState:
    """Represents the state of an agent in the simulation"""
    def __init__(self):
        self.id = ""
        self.type = ""
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.resources = {}
        self.status = "active"
        self.perception_range = 10.0
        self.mission = None
        self.memory = {}

class AgentAction:
    """Represents an action taken by an agent"""
    def __init__(self):
        self.type = ""
        self.target = None
        self.parameters = {}
        self.timestamp = 0.0

class Perception:
    """Represents an agent's perception of the world"""
    def __init__(self):
        self.visible_objects = []
        self.threat_level = 0.0
        self.resource_map = {}
        self.comms_data = {}
        self.mission_status = {}

class AgentPolicy(abc.ABC):
    """Abstract base class for agent policies"""
    @abc.abstractmethod
    def select_action(self, state: AgentState, perception: Perception) -> AgentAction:
        """Select an action based on the current state and perception"""
        pass

    @abc.abstractmethod
    def update(self, reward: float, done: bool):
        """Update the policy based on reward and episode status"""
        pass

class BaseAgent(SimulationObject):
    """Base class for all agents in the simulation"""
    def __init__(self, agent_id: str, agent_type: str):
        super().__init__(agent_id)
        self.state = AgentState()
        self.state.id = agent_id
        self.state.type = agent_type
        self.state.status = "initialized"
        self.policies = []
        self.current_policy = None
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
    def add_policy(self, policy: AgentPolicy):
        """Add a policy to the agent"""
        self.policies.append(policy)
        if not self.current_policy:
            self.current_policy = policy

    def set_policy(self, policy_index: int):
        """Set the current policy to use"""
        if 0 <= policy_index < len(self.policies):
            self.current_policy = self.policies[policy_index]
        else:
            self.logger.warning(f"Policy index {policy_index} out of range")

    def perceive(self, world_state: WorldState) -> Perception:
        """Perceive the world state"""
        perception = Perception()
        
        # Find visible objects
        perception.visible_objects = [
            obj for obj in world_state.objects.values()
            if self._is_visible(obj, world_state)
        ]
        
        # Get threat level
        perception.threat_level = self._calculate_threat(perception.visible_objects)
        
        # Get resource map
        perception.resource_map = self._map_resources(world_state)
        
        return perception

    def _is_visible(self, obj: SimulationObject, world_state: WorldState) -> bool:
        """Check if an object is within this agent's perception range"""
        if obj.id == self.id:
            return False
            
        # Calculate distance
        position = np.array(self.state.position)
        obj_position = np.array(obj.position if hasattr(obj, 'position') else [0.0, 0.0])
        distance = np.linalg.norm(position - obj_position)
        
        # Check if within range
        return distance <= self.state.perception_range

    def _calculate_threat(self, visible_objects: List[SimulationObject]) -> float:
        """Calculate threat level from visible objects"""
        threat = 0.0
        for obj in visible_objects:
            if isinstance(obj, BaseAgent) and obj.state.type != self.state.type:
                # Adversarial agent - add threat based on proximity and capability
                threat += 0.5  # Base threat
        return threat

    def _map_resources(self, world_state: WorldState) -> Dict[str, Any]:
        """Map available resources in the environment"""
        return world_state.get_resource_map()

    def plan(self, perception: Perception) -> Optional[AgentAction]:
        """Plan an action based on perception"""
        if not self.current_policy:
            self.logger.warning("No policy available for planning")
            return None
            
        return self.current_policy.select_action(self.state, perception)

    def execute(self, action: AgentAction, world_state: WorldState) -> Dict[str, Any]:
        """Execute an action in the world"""
        self.logger.debug(f"Executing action: {action.type}")
        
        # Default implementation - should be overridden by specific agent types
        result = {
            "success": True,
            "reward": 0.0,
            "done": False
        }
        
        # Update state based on action
        self._update_state_after_action(action)
        
        return result

    def _update_state_after_action(self, action: AgentAction):
        """Update agent state after executing an action"""
        # Default implementation - should be overridden
        pass

    def update(self, reward: float, done: bool):
        """Update agent's policy based on experience"""
        if self.current_policy:
            self.current_policy.update(reward, done)

    def get_state(self) -> Dict[str, Any]:
        """Get serialized state for telemetry"""
        return {
            "id": self.state.id,
            "type": self.state.type,
            "position": self.state.position.tolist(),
            "orientation": self.state.orientation,
            "resources": self.state.resources,
            "status": self.state.status,
            "mission": str(self.state.mission) if self.state.mission else None
        }

class RedTeamAgent(BaseAgent):
    """Agent representing red team (adversarial) forces"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "red")
        self.state.perception_range = 15.0


class BlueTeamAgent(BaseAgent):
    """Agent representing blue team (defensive) forces"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "blue")
        self.state.perception_range = 20.0


class WhiteTeamAgent(BaseAgent):
    """Agent representing white team (neutral/observer) forces"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "white")
        self.state.perception_range = 30.0

# Policy implementations
class RuleBasedPolicy(AgentPolicy):
    """Rule-based policy implementation"""
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules
        
    def select_action(self, state: AgentState, perception: Perception) -> AgentAction:
        """Select action based on rule matching"""
        action = AgentAction()
        action.timestamp = 0.0  # Will be set by the agent
        
        # Default action
        action.type = "wait"
        
        # Apply rules in priority order
        for rule in self.rules:
            if self._rule_condition_met(rule, state, perception):
                action.type = rule["action_type"]
                action.parameters = rule.get("parameters", {})
                action.target = rule.get("target", None)
                break
                
        return action

    def _rule_condition_met(self, rule: Dict[str, Any], 
                           state: AgentState, 
                           perception: Perception) -> bool:
        """Check if a rule's conditions are met"""
        # Default implementation - should be extended
        return True

    def update(self, reward: float, done: bool):
        """Rule-based policies don't learn from experience"""
        pass


class RLAgentPolicy(AgentPolicy):
    """Reinforcement learning policy implementation"""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        # In a real implementation, this would be a neural network or other model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the RL model"""
        # Placeholder for actual model implementation
        return None

    def select_action(self, state: AgentState, perception: Perception) -> AgentAction:
        """Select action using the RL model"""
        # Convert state and perception to observation
        observation = self._convert_to_observation(state, perception)
        
        # Get action from model (placeholder implementation)
        action = AgentAction()
        action.type = "rl_action"
        action.timestamp = 0.0
        
        return action

    def _convert_to_observation(self, state: AgentState, perception: Perception):
        """Convert state and perception to observation format"""
        # Placeholder for actual conversion logic
        return {}

    def update(self, reward: float, done: bool):
        """Update the RL model based on experience"""
        # Placeholder for actual update logic
        pass

# Mission system
class Mission:
    """Represents a mission for an agent to complete"""
    def __init__(self, mission_id: str, mission_type: str, objectives: List[Dict[str, Any]]):
        self.id = mission_id
        self.type = mission_type
        self.objectives = objectives
        self.progress = 0.0
        self.completed = False
        self.failed = False

    def update_progress(self, world_state: WorldState):
        """Update mission progress based on world state"""
        # Placeholder for actual mission logic
        pass

    def is_complete(self) -> bool:
        """Check if mission is complete"""
        return self.completed

    def is_failed(self) -> bool:
        """Check if mission has failed"""
        return self.failed


class MissionPlanner:
    """Plans and assigns missions to agents"""
    def __init__(self):
        self.mission_templates = {}
        
    def register_template(self, template_id: str, template: Dict[str, Any]):
        """Register a mission template"""
        self.mission_templates[template_id] = template

    def generate_mission(self, template_id: str, parameters: Dict[str, Any]) -> Mission:
        """Generate a mission from a template"""
        if template_id not in self.mission_templates:
            raise ValueError(f"Template {template_id} not found")
            
        template = self.mission_templates[template_id]
        # Apply parameters to template
        mission_id = f"{template_id}_{parameters.get('instance_id', '001')}"
        mission_type = template["type"]
        objectives = template["objectives"]
        
        # Create and return mission
        return Mission(mission_id, mission_type, objectives)

    def assign_mission(self, mission: Mission, agent: BaseAgent):
        """Assign a mission to an agent"""
        agent.state.mission = mission

    def update_missions(self, world_state: WorldState):
        """Update all missions based on world state"""
        for agent in world_state.agents.values():
            if agent.state.mission and not agent.state.mission.completed:
                agent.state.mission.update_progress(world_state)

# Resource system
class Resource:
    """Represents a resource in the simulation"""
    def __init__(self, resource_id: str, resource_type: str, amount: float):
        self.id = resource_id
        self.type = resource_type
        self.amount = amount
        self.max_capacity = amount
        self.location = [0.0, 0.0]
        self.owner = None

    def can_consume(self, amount: float) -> bool:
        """Check if the resource can be consumed"""
        return self.amount >= amount

    def consume(self, amount: float) -> float:
        """Consume some amount of the resource"""
        if self.can_consume(amount):
            self.amount -= amount
            return amount
        else:
            consumed = self.amount
            self.amount = 0.0
            return consumed

    def replenish(self, amount: float) -> float:
        """Replenish some amount of the resource"""
        available = self.max_capacity - self.amount
        add_amount = min(amount, available)
        self.amount += add_amount
        return add_amount

    def get_state(self) -> Dict[str, Any]:
        """Get serialized state for telemetry"""
        return {
            "id": self.id,
            "type": self.type,
            "amount": self.amount,
            "max_capacity": self.max_capacity,
            "location": self.location,
            "owner": self.owner
        }


class ResourceManager:
    """Manages resources in the simulation"""
    def __init__(self):
        self.resources = {}
        
    def add_resource(self, resource: Resource):
        """Add a new resource to the manager"""
        self.resources[resource.id] = resource

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID"""
        return self.resources.get(resource_id)

    def get_resources_in_range(self, location: List[float], radius: float) -> List[Resource]:
        """Get resources within a certain range of a location"""
        # Placeholder for actual spatial query
        return list(self.resources.values())

    def consume_resource(self, resource_id: str, amount: float) -> float:
        """Consume some amount of a resource"""
        resource = self.get_resource(resource_id)
        if resource:
            return resource.consume(amount)
        return 0.0

    def replenish_resource(self, resource_id: str, amount: float) -> float:
        """Replenish some amount of a resource"""
        resource = self.get_resource(resource_id)
        if resource:
            return resource.replenish(amount)
        return 0.0

    def get_all_resources(self) -> Dict[str, Resource]:
        """Get all resources"""
        return self.resources

    def get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Get resource usage statistics"""
        stats = {}
        for resource_id, resource in self.resources.items():
            stats[resource_id] = {
                "type": resource.type,
                "usage": resource.amount / resource.max_capacity if resource.max_capacity > 0 else 0,
                "amount": resource.amount,
                "capacity": resource.max_capacity
            }
        return stats

# Event system
class SimulationEvent:
    """Represents an event in the simulation"""
    def __init__(self, event_type: str, timestamp: float, data: Dict[str, Any]):
        self.type = event_type
        self.timestamp = timestamp
        self.data = data
        self.handled = False


class EventManager:
    """Manages simulation events"""
    def __init__(self):
        self.event_queue = []
        self.handlers = {}
        
    def register_handler(self, event_type: str, handler):
        """Register a handler for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def post_event(self, event: SimulationEvent):
        """Post an event to the queue"""
        self.event_queue.append(event)
        
    def process_events(self):
        """Process all events in the queue"""
        # Sort events by timestamp
        self.event_queue.sort(key=lambda e: e.timestamp)
        
        for event in self.event_queue:
            if not event.handled:
                self._dispatch_event(event)
                event.handled = True
        
        # Clear handled events
        self.event_queue = [e for e in self.event_queue if not e.handled]

    def _dispatch_event(self, event: SimulationEvent):
        """Dispatch an event to appropriate handlers"""
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error handling event {event.type}: {e}")

    def clear_events(self):
        """Clear all events from the queue"""
        self.event_queue = []

# Utility functions
def create_default_agent(agent_id: str, agent_type: str) -> BaseAgent:
    """Create an agent with default policies"""
    if agent_type == "red":
        agent = RedTeamAgent(agent_id)
        # Add default policies
        agent.add_policy(RuleBasedPolicy([]))
    elif agent_type == "blue":
        agent = BlueTeamAgent(agent_id)
        # Add default policies
        agent.add_policy(RuleBasedPolicy([]))
    elif agent_type == "white":
        agent = WhiteTeamAgent(agent_id)
        # Add default policies
        agent.add_policy(RuleBasedPolicy([]))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return agent


def create_rl_agent(agent_id: str, agent_type: str, observation_space, action_space) -> BaseAgent:
    """Create an agent with RL policy"""
    agent = create_default_agent(agent_id, agent_type)
    agent.add_policy(RLAgentPolicy(observation_space, action_space))
    agent.set_policy(1)  # Use RL policy
    return agent


def create_mission_planner() -> MissionPlanner:
    """Create a mission planner with default templates"""
    planner = MissionPlanner()
    # Add default mission templates
    planner.register_template("default_attack", {
        "type": "attack",
        "objectives": []
    })
    planner.register_template("default_defense", {
        "type": "defense",
        "objectives": []
    })
    return planner


def create_resource_manager() -> ResourceManager:
    """Create a resource manager with default resources"""
    return ResourceManager()


def create_event_manager() -> EventManager:
    """Create an event manager with default handlers"""
    return EventManager()

# Example usage
if __name__ == "__main__":
    # Create agents
    red_agent = create_default_agent("R-001", "red")
    blue_agent = create_default_agent("B-001", "blue")
    
    # Create mission planner and assign mission
    mission_planner = create_mission_planner()
    mission = mission_planner.generate_mission("default_attack", {"instance_id": "001"})
    mission_planner.assign_mission(mission, red_agent)
    
    # Create resource manager
    resource_manager = create_resource_manager()
    
    # Create event manager
    event_manager = create_event_manager()
    
    # Create world state and add agents
    world_state = WorldState()
    world_state.add_agent(red_agent)
    world_state.add_agent(blue_agent)
    
    # Simulation loop
    for step in range(100):
        # Update world state
        world_state.update()
        
        # Agents perceive and act
        for agent in world_state.agents.values():
            perception = agent.perceive(world_state)
            action = agent.plan(perception)
            if action:
                result = agent.execute(action, world_state)
                agent.update(result["reward"], result["done"])
        
        # Update missions
        mission_planner.update_missions(world_state)
        
        # Process events
        event_manager.process_events()
        
        # Log state
        if step % 10 == 0:
            print(f"\nStep {step}:")
            for agent in world_state.agents.values():
                print(f"{agent.state.id} state: {agent.get_state()}")
                
    print("\nSimulation complete")
