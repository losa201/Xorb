#!/usr/bin/env python3
"""
XORB Swarm Intelligence API Router
FastAPI endpoints for swarm intelligence control and monitoring
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from xorb_core.agents.xorb_agent_specialization_cluster import (
    XORBAgentSpecializationCluster,
    SwarmEvolutionSignal,
    AgentSpecializationRole
)


router = APIRouter(prefix="/api/swarm", tags=["swarm-intelligence"])
logger = logging.getLogger(__name__)

# Global swarm cluster instance (in production, this would be dependency injected)
_swarm_cluster: Optional[XORBAgentSpecializationCluster] = None


# ================================================================================
# PYDANTIC MODELS
# ================================================================================

class SwarmStatusResponse(BaseModel):
    """Swarm status response model"""
    cluster_id: str
    total_agents: int
    active_clusters: int
    global_fitness: float
    swarm_entropy: float
    adaptation_rate: float
    recent_evolutions: int
    healing_events_today: int
    clusters: Dict[str, Dict[str, Any]]


class AgentRoleResponse(BaseModel):
    """Agent role response model"""
    cluster_id: Optional[int]
    specialization_role: str
    specialization_scores: Dict[str, float]
    anomaly_score: float
    performance_metrics: Dict[str, float]
    last_updated: str


class EntropyMetricsResponse(BaseModel):
    """Entropy metrics response model"""
    swarm_entropy: float
    cluster_drift_scores: Dict[str, float]
    global_fitness: float
    adaptation_rate: float
    evolution_frequency: float


class EvolutionTriggerRequest(BaseModel):
    """Evolution trigger request model"""
    evolution_type: str = Field(
        default="manual",
        description="Type of evolution to trigger: manual, clustering, structure, healing"
    )
    force: bool = Field(default=False, description="Force evolution even if not needed")


class CoSynapseStateResponse(BaseModel):
    """XORB-CoSynapse state response model"""
    synapse_id: str
    swarm_entropy: float
    global_fitness: float
    active_conflicts: int
    learning_weights: Dict[str, Any]
    consciousness_shift_threshold: float
    total_agents_managed: int
    evolution_events_processed: int


# ================================================================================
# DEPENDENCY FUNCTIONS
# ================================================================================

async def get_swarm_cluster() -> XORBAgentSpecializationCluster:
    """Get swarm cluster dependency"""
    global _swarm_cluster

    if _swarm_cluster is None:
        raise HTTPException(
            status_code=503,
            detail="Swarm intelligence system not initialized"
        )

    return _swarm_cluster


async def initialize_swarm_cluster(agents: List[Any] = None) -> XORBAgentSpecializationCluster:
    """Initialize swarm cluster (called during startup)"""
    global _swarm_cluster

    if _swarm_cluster is None:
        _swarm_cluster = XORBAgentSpecializationCluster()

        if agents:
            await _swarm_cluster.initialize_swarm(agents)

        logger.info("🧬 Swarm intelligence system initialized")

    return _swarm_cluster


# ================================================================================
# API ENDPOINTS
# ================================================================================

@router.get("/status", response_model=SwarmStatusResponse)
async def get_swarm_status(
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> SwarmStatusResponse:
    """
    Get comprehensive swarm intelligence status

    Returns current state of the swarm including:
    - Total agent count and cluster distribution
    - Global fitness and entropy metrics
    - Recent evolution activity
    - Cluster-specific performance metrics
    """
    try:
        status = await swarm.get_swarm_status()
        return SwarmStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to get swarm status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/roles", response_model=Dict[str, AgentRoleResponse])
async def get_agent_roles(
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> Dict[str, AgentRoleResponse]:
    """
    Get current agent role specialization map

    Returns detailed role assignments for each agent including:
    - Cluster assignment and specialization role
    - Performance metrics and anomaly scores
    - Last update timestamp
    """
    try:
        roles = await swarm.get_agent_roles()

        return {
            agent_id: AgentRoleResponse(**role_data)
            for agent_id, role_data in roles.items()
        }

    except Exception as e:
        logger.error(f"Failed to get agent roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entropy", response_model=EntropyMetricsResponse)
async def get_entropy_metrics(
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> EntropyMetricsResponse:
    """
    Get swarm entropy and cluster drift scores

    Returns entropy metrics including:
    - Overall swarm entropy (diversity measure)
    - Per-cluster drift scores
    - Evolution frequency statistics
    """
    try:
        metrics = await swarm.get_entropy_metrics()
        return EntropyMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Failed to get entropy metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve", response_model=SwarmStatusResponse)
async def trigger_evolution(
    request: EvolutionTriggerRequest,
    background_tasks: BackgroundTasks,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> SwarmStatusResponse:
    """
    Manually trigger swarm evolution

    Supports different evolution types:
    - manual: General evolution trigger
    - clustering: Force re-clustering of agents
    - structure: Evolve swarm structure with mutations
    - healing: Trigger healing sequence for all agents
    """
    try:
        logger.info(f"🎯 Manual evolution triggered: {request.evolution_type}")

        # Trigger evolution in background for non-blocking response
        background_tasks.add_task(
            swarm.trigger_manual_evolution,
            request.evolution_type
        )

        # Return current status
        status = await swarm.get_swarm_status()
        return SwarmStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to trigger evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cosynapse/state", response_model=CoSynapseStateResponse)
async def get_cosynapse_state(
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> CoSynapseStateResponse:
    """
    Get XORB-CoSynapse meta-agent state snapshot

    Returns state of the meta-agent overseer including:
    - Consciousness shift threshold and learning weights
    - Active conflict resolution status
    - Agent management statistics
    """
    try:
        state = await swarm.get_cosynapse_state()
        return CoSynapseStateResponse(**state)

    except Exception as e:
        logger.error(f"Failed to get CoSynapse state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> Dict[str, Any]:
    """
    Get detailed metrics for a specific agent

    Returns comprehensive performance and behavioral metrics for the specified agent
    """
    try:
        roles = await swarm.get_agent_roles()

        if agent_id not in roles:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_data = roles[agent_id]

        # Add additional metrics if available
        return {
            "agent_id": agent_id,
            "role_data": agent_data,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent metrics for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}/agents")
async def get_cluster_agents(
    cluster_id: int,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> Dict[str, Any]:
    """
    Get all agents in a specific cluster

    Returns list of agents and their performance metrics within the specified cluster
    """
    try:
        status = await swarm.get_swarm_status()

        if str(cluster_id) not in status["clusters"]:
            raise HTTPException(status_code=404, detail="Cluster not found")

        cluster_info = status["clusters"][str(cluster_id)]

        # Get agent roles to find agents in this cluster
        roles = await swarm.get_agent_roles()
        cluster_agents = {
            agent_id: role_data
            for agent_id, role_data in roles.items()
            if role_data.cluster_id == cluster_id
        }

        return {
            "cluster_id": cluster_id,
            "cluster_info": cluster_info,
            "agents": cluster_agents,
            "agent_count": len(cluster_agents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster {cluster_id} agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evolution/history")
async def get_evolution_history(
    limit: int = 50,
    signal_type: Optional[str] = None,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> List[Dict[str, Any]]:
    """
    Get swarm evolution history

    Returns recent evolution events with optional filtering by signal type
    """
    try:
        # Access evolution history from swarm cluster
        history = list(swarm.evolution_history)

        # Filter by signal type if specified
        if signal_type:
            try:
                filter_signal = SwarmEvolutionSignal(signal_type)
                history = [event for event in history if event.signal_type == filter_signal]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid signal type: {signal_type}"
                )

        # Limit results and convert to dict
        limited_history = history[-limit:] if limit > 0 else history

        return [
            {
                "event_id": event.event_id,
                "signal_type": event.signal_type.value,
                "agent_id": event.agent_id,
                "cluster_id": event.cluster_id,
                "impact_score": event.impact_score,
                "metadata": event.metadata,
                "timestamp": event.timestamp.isoformat()
            }
            for event in limited_history
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evolution history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/healing/events")
async def get_healing_events(
    limit: int = 20,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> List[Dict[str, Any]]:
    """
    Get recent self-healing events

    Returns history of agent self-healing activities and their outcomes
    """
    try:
        # Get healing events from swarm cluster
        healing_events = swarm.healing_events[-limit:] if limit > 0 else swarm.healing_events

        return [
            {
                "agent_id": event["agent_id"],
                "timestamp": event["timestamp"].isoformat(),
                "diagnosis": event["diagnosis"],
                "success": event["success"]
            }
            for event in healing_events
        ]

    except Exception as e:
        logger.error(f"Failed to get healing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/heal")
async def trigger_agent_healing(
    agent_id: str,
    background_tasks: BackgroundTasks,
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> Dict[str, Any]:
    """
    Manually trigger healing for a specific agent

    Initiates self-healing sequence for the specified agent
    """
    try:
        if agent_id not in swarm.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Trigger healing in background
        background_tasks.add_task(
            swarm._trigger_agent_healing,
            agent_id,
            1.0  # Manual trigger with high impact score
        )

        return {
            "message": f"Healing triggered for agent {agent_id}",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger healing for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    swarm: XORBAgentSpecializationCluster = Depends(get_swarm_cluster)
) -> str:
    """
    Get Prometheus-formatted metrics

    Returns swarm intelligence metrics in Prometheus format for monitoring
    """
    try:
        status = await swarm.get_swarm_status()
        entropy_metrics = await swarm.get_entropy_metrics()

        # Generate Prometheus metrics format
        metrics = []

        # Global metrics
        metrics.append(f"xorb_swarm_global_fitness {status['global_fitness']}")
        metrics.append(f"xorb_swarm_entropy {status['swarm_entropy']}")
        metrics.append(f"xorb_swarm_adaptation_rate {status['adaptation_rate']}")
        metrics.append(f"xorb_swarm_total_agents {status['total_agents']}")
        metrics.append(f"xorb_swarm_active_clusters {status['active_clusters']}")
        metrics.append(f"xorb_swarm_recent_evolutions {status['recent_evolutions']}")
        metrics.append(f"xorb_swarm_healing_events_today {status['healing_events_today']}")

        # Cluster-specific metrics
        for cluster_id, cluster_info in status['clusters'].items():
            labels = f'cluster_id="{cluster_id}",role="{cluster_info["specialization_role"]}"'
            metrics.append(f"xorb_cluster_fitness{{{labels}}} {cluster_info['fitness_score']}")
            metrics.append(f"xorb_cluster_cohesion{{{labels}}} {cluster_info['cohesion_metric']}")
            metrics.append(f"xorb_cluster_agent_count{{{labels}}} {cluster_info['agent_count']}")
            metrics.append(f"xorb_cluster_adaptation_rate{{{labels}}} {cluster_info['adaptation_rate']}")

        return "\n".join(metrics)

    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================================
# HEALTH CHECK ENDPOINTS
# ================================================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for swarm intelligence system"""
    try:
        global _swarm_cluster

        if _swarm_cluster is None:
            return {
                "status": "unhealthy",
                "message": "Swarm intelligence system not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }

        status = await _swarm_cluster.get_swarm_status()

        # Determine health based on global fitness and entropy
        is_healthy = (
            status['global_fitness'] > 0.3 and
            status['swarm_entropy'] > 0.5 and
            status['total_agents'] > 0
        )

        return {
            "status": "healthy" if is_healthy else "degraded",
            "global_fitness": status['global_fitness'],
            "swarm_entropy": status['swarm_entropy'],
            "total_agents": status['total_agents'],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
