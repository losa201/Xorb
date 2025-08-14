import logging
from datetime import datetime
from typing import Dict, List, Any

import aioredis

from xorb.shared.models import UnifiedAgent

# Intelligent Agent Orchestrator
class IntelligentAgentOrchestrator:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.active_agents = {}
        self.agent_performance = {}
        self.logger = logging.getLogger(__name__)

    async def register_agent(self, agent: UnifiedAgent) -> bool:
        """Register a new agent."""
        try:
            # Store agent in Redis
            await self.redis.setex(
                f"agent:{agent.id}",
                3600,  # 1 hour TTL
                agent.json()
            )

            # Add to active agents
            self.active_agents[agent.id] = agent

            self.logger.info(f"Registered agent {agent.id} ({agent.agent_type})")
            return True
        except Exception as e:
            self.logger.error(f"Error registering agent {agent.id}: {e}")
            return False

    async def assign_agent_to_target(self, agent_id: str, target_id: str, task: Dict[str, Any]) -> bool:
        """Assign agent to a specific target with task."""
        try:
            agent_data = await self.redis.get(f"agent:{agent_id}")
            if not agent_data:
                return False

            agent = UnifiedAgent.parse_raw(agent_data)
            agent.target_id = target_id
            agent.status = "assigned"
            agent.last_active = datetime.utcnow()

            # Store updated agent
            await self.redis.setex(f"agent:{agent_id}", 3600, agent.json())

            # Store task assignment
            await self.redis.setex(
                f"task:{agent_id}:{target_id}",
                7200,  # 2 hours
                str(task)
            )

            self.logger.info(f"Assigned agent {agent.id} to target {target_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error assigning agent {agent.id} to target {target_id}: {e}")
            return False

    async def get_optimal_agents(self, agent_types: List[str], count: int = 1) -> List[UnifiedAgent]:
        """Get optimal agents for given types based on performance."""
        optimal_agents = []

        for agent_type in agent_types:
            # Get all agents of this type
            agent_keys = await self.redis.keys(f"agent:*")
            candidates = []

            for key in agent_keys:
                agent_data = await self.redis.get(key)
                if agent_data:
                    agent = UnifiedAgent.parse_raw(agent_data)
                    if agent.agent_type == agent_type and agent.status == "idle":
                        candidates.append(agent)

            # Sort by performance metrics
            candidates.sort(key=lambda a: a.performance_metrics.get('success_rate', 0.5), reverse=True)

            # Take the best ones
            optimal_agents.extend(candidates[:count])

        return optimal_agents
