class EmergentBehaviorSystem:
    """System for managing emergent behaviors in multi-agent scenarios."""

    def __init__(self, swarm_size_threshold=5, coordination_radius=100):
        self.agents = {}
        self.swarm_size_threshold = swarm_size_threshold
        self.coordination_radius = coordination_radius
        self.behavior_patterns = self._initialize_behavior_patterns()

    def _initialize_behavior_patterns(self):
        """Initialize known behavior patterns."""
        return {
            "swarm_attack": {
                "conditions": ["high_threat", "group_formation"],
                "tactics": ["coordinated_attack", "flanking", "decoy_operations"],
                "priority": 3,
            },
            "defensive_formation": {
                "conditions": ["low_resources", "high_risk"],
                "tactics": ["perimeter_defense", "resource_consolidation"],
                "priority": 2,
            },
            "reconnaissance": {
                "conditions": ["unknown_environment", "low_threat"],
                "tactics": ["area_scanning", "target_identification"],
                "priority": 1,
            },
        }

    def register_agent(self, agent_id, position, capabilities):
        """Register an agent with the emergent behavior system.

        Args:
            agent_id: Unique identifier for the agent
            position: Current position of the agent
            capabilities: Dictionary of agent capabilities
        """
        self.agents[agent_id] = {
            "position": position,
            "capabilities": capabilities,
            "neighbors": [],
            "current_behavior": None,
        }

    def update_agent_position(self, agent_id, new_position):
        """Update an agent's position.

        Args:
            agent_id: Unique identifier for the agent
            new_position: New position of the agent
        """
        if agent_id in self.agents:
            self.agents[agent_id]["position"] = new_position

    def detect_neighbors(self, agent_id, radius=None):
        """Detect neighboring agents within a given radius.

        Args:
            agent_id: Unique identifier for the agent
            radius: Search radius (defaults to coordination radius)

        Returns:
            List of neighboring agent IDs
        """
        if radius is None:
            radius = self.coordination_radius

        neighbors = []
        if agent_id in self.agents:
            position = self.agents[agent_id]["position"]
            for other_id, other in self.agents.items():
                if other_id != agent_id:
                    distance = self._calculate_distance(position, other["position"])
                    if distance <= radius:
                        neighbors.append(other_id)
        return neighbors

    def _calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions.

        Args:
            pos1: First position (tuple)
            pos2: Second position (tuple)

        Returns:
            Euclidean distance
        """
        return sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)) ** 0.5

    def update(self):
        """Update emergent behaviors for all agents."""
        # Update neighbors for all agents
        for agent_id in self.agents:
            self.agents[agent_id]["neighbors"] = self.detect_neighbors(agent_id)

        # Evaluate and update behaviors
        for agent_id in self.agents:
            self._evaluate_behavior(agent_id)

    def _evaluate_behavior(self, agent_id):
        """Evaluate and potentially update an agent's behavior.

        Args:
            agent_id: Unique identifier for the agent
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return

        # Get current context
        context = self._get_context(agent_id)

        # Determine best behavior pattern
        best_pattern = self._select_best_pattern(context)

        # Update agent's behavior if needed
        if best_pattern and best_pattern != agent["current_behavior"]:
            self.agents[agent_id]["current_behavior"] = best_pattern
            self._initiate_behavior(agent_id, best_pattern)

    def _get_context(self, agent_id):
        """Get contextual information for behavior decision.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary of contextual information
        """
        context = {
            "agent_id": agent_id,
            "position": self.agents[agent_id]["position"],
            "capabilities": self.agents[agent_id]["capabilities"],
            "neighbors": self.agents[agent_id]["neighbors"],
            "neighbor_count": len(self.agents[agent_id]["neighbors"]),
            "group_size": 1 + len(self.agents[agent_id]["neighbors"]),
        }

        # Add environmental context
        context["environment"] = self._get_environmental_context(agent_id)

        return context

    def _get_environmental_context(self, agent_id):
        """Get environmental context for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary of environmental context
        """
        # This would be integrated with the simulation environment
        # For now, return placeholder values
        return {
            "threat_level": 0.5,  # 0-1 scale
            "resource_density": 0.3,  # 0-1 scale
            "visibility": 0.7,  # 0-1 scale
            "terrain_complexity": 0.4,  # 0-1 scale
        }

    def _select_best_pattern(self, context):
        """Select the best behavior pattern based on context.

        Args:
            context: Dictionary of contextual information

        Returns:
            Name of the selected behavior pattern
        """
        # For now, use a simple rule-based selection
        # In a real implementation, this would use ML models

        if context["group_size"] >= self.swarm_size_threshold:
            if context["environment"]["threat_level"] > 0.7:
                return "swarm_attack"
            elif context["environment"]["threat_level"] < 0.3:
                return "reconnaissance"
            else:
                return "defensive_formation"
        else:
            if context["environment"]["threat_level"] > 0.5:
                return "defensive_formation"
            else:
                return "reconnaissance"

    def _initiate_behavior(self, agent_id, behavior):
        """Initiate a new behavior for an agent.

        Args:
            agent_id: Unique identifier for the agent
            behavior: Name of the behavior to initiate
        """
        # This would trigger behavior-specific initialization
        # For now, just log the behavior change
        print(f"Agent {agent_id} initiating behavior: {behavior}")

    def get_current_behavior(self, agent_id):
        """Get the current behavior of an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Current behavior or None
        """
        if agent_id in self.agents:
            return self.agents[agent_id]["current_behavior"]
        return None

    def get_group_behavior(self, agent_id):
        """Get the dominant behavior of an agent's group.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary of behavior counts or None
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        neighbors = agent["neighbors"]
        if not neighbors:
            return {agent["current_behavior"]: 1}

        behavior_counts = {}
        for neighbor_id in neighbors:
            behavior = self.get_current_behavior(neighbor_id)
            if behavior:
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        # Include self
        behavior_counts[agent["current_behavior"]] = (
            behavior_counts.get(agent["current_behavior"], 0) + 1
        )

        return behavior_counts

    def get_behavior_patterns(self):
        """Get all available behavior patterns."""
        return self.behavior_patterns

    def add_behavior_pattern(self, name, pattern):
        """Add a new behavior pattern.

        Args:
            name: Name of the pattern
            pattern: Dictionary containing pattern details
        """
        self.behavior_patterns[name] = pattern

    def remove_behavior_pattern(self, name):
        """Remove a behavior pattern.

        Args:
            name: Name of the pattern to remove
        """
        if name in self.behavior_patterns:
            del self.behavior_patterns[name]

    def get_swarm_size_threshold(self):
        """Get the swarm size threshold."""
        return self.swarm_size_threshold

    def set_swarm_size_threshold(self, threshold):
        """Set the swarm size threshold.

        Args:
            threshold: New threshold value
        """
        self.swarm_size_threshold = threshold

    def get_coordination_radius(self):
        """Get the coordination radius."""
        return self.coordination_radius

    def set_coordination_radius(self, radius):
        """Set the coordination radius.

        Args:
            radius: New radius value
        """
        self.coordination_radius = radius


# Example usage
if __name__ == "__main__":
    # Create emergent behavior system
    ebs = EmergentBehaviorSystem()

    # Register agents
    ebs.register_agent("A1", (0, 0), {"stealth": 0.8, "speed": 0.6})
    ebs.register_agent("A2", (10, 10), {"stealth": 0.7, "speed": 0.7})
    ebs.register_agent("A3", (20, 20), {"stealth": 0.6, "speed": 0.8})
    ebs.register_agent("A4", (30, 30), {"stealth": 0.5, "speed": 0.9})
    ebs.register_agent("A5", (40, 40), {"stealth": 0.4, "speed": 1.0})

    # Update positions and detect neighbors
    ebs.update_agent_position("A1", (15, 15))
    ebs.update()

    # Get behavior information
    print("Agent A1 neighbors:", ebs.detect_neighbors("A1"))
    print("Agent A1 current behavior:", ebs.get_current_behavior("A1"))
    print("Group behavior for A1:", ebs.get_group_behavior("A1"))
    print("All behavior patterns:", ebs.get_behavior_patterns())
