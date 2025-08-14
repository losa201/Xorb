"""
Red/Blue Agent Framework API Integration

FastAPI router for integrating the Red/Blue Agent Framework with the existing XORB PTaaS infrastructure.
Provides endpoints for mission management, agent orchestration, and real-time monitoring.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import XORB infrastructure components
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector
from ..infrastructure.cache import get_cache_client
from ..infrastructure.database import get_database_pool

# Import Red/Blue Agent Framework components
from ...services.red_blue_agents.core.agent_scheduler import AgentScheduler, Mission, MissionStatus
from ...services.red_blue_agents.core.capability_registry import CapabilityRegistry, Environment
from ...services.red_blue_agents.core.sandbox_orchestrator import SandboxOrchestrator
from ...services.red_blue_agents.telemetry.collector import TelemetryCollector
from ...services.red_blue_agents.learning.autonomous_explorer import AutonomousExplorer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Red/Blue Agents"], prefix="/red-blue-agents")

# Global framework components (initialized at startup)
agent_scheduler: Optional[AgentScheduler] = None
capability_registry: Optional[CapabilityRegistry] = None
sandbox_orchestrator: Optional[SandboxOrchestrator] = None
telemetry_collector: Optional[TelemetryCollector] = None
autonomous_explorer: Optional[AutonomousExplorer] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.mission_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Remove from mission subscriptions
        for mission_id, subscribers in self.mission_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)

    async def subscribe_to_mission(self, websocket: WebSocket, mission_id: str):
        if mission_id not in self.mission_subscribers:
            self.mission_subscribers[mission_id] = []
        self.mission_subscribers[mission_id].append(websocket)

    async def broadcast_mission_update(self, mission_id: str, data: Dict[str, Any]):
        if mission_id in self.mission_subscribers:
            for websocket in self.mission_subscribers[mission_id].copy():
                try:
                    await websocket.send_json(data)
                except:
                    self.disconnect(websocket)

connection_manager = ConnectionManager()

# Request/Response Models
class RedBlueTargetRequest(BaseModel):
    """Request model for red/blue team targets"""
    type: str = Field(..., description="Target type (web_app, network, host, cloud)")
    host: str = Field(..., description="Target hostname or IP address")
    ports: List[int] = Field(default=[], description="Target ports")
    services: List[str] = Field(default=[], description="Known services")
    technologies: List[str] = Field(default=[], description="Known technologies")
    credentials: Optional[Dict[str, str]] = Field(default=None, description="Test credentials")
    metadata: Dict[str, Any] = Field(default={}, description="Additional target metadata")

class RedBlueMissionRequest(BaseModel):
    """Request model for creating red/blue team missions"""
    name: str = Field(..., description="Mission name")
    description: str = Field(..., description="Mission description")
    environment: str = Field(..., description="Environment (production, staging, development, cyber_range)")
    objectives: List[str] = Field(..., description="Mission objectives")
    targets: List[RedBlueTargetRequest] = Field(..., description="Mission targets")
    red_team_config: Dict[str, Any] = Field(default={}, description="Red team configuration")
    blue_team_config: Dict[str, Any] = Field(default={}, description="Blue team configuration")
    constraints: Dict[str, Any] = Field(default={}, description="Mission constraints")
    timeout_seconds: int = Field(default=7200, description="Mission timeout in seconds")
    learning_enabled: bool = Field(default=True, description="Enable autonomous learning")

class RedBlueMissionResponse(BaseModel):
    """Response model for mission operations"""
    mission_id: str
    name: str
    status: str
    environment: str
    progress: Optional[Dict[str, Any]] = None
    agents: Optional[List[Dict[str, Any]]] = None
    results: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class AgentCapabilityResponse(BaseModel):
    """Response model for agent capabilities"""
    agent_type: str
    name: str
    description: str
    supported_categories: List[str]
    techniques: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    environments: List[str]

class TechniqueRecommendationRequest(BaseModel):
    """Request model for technique recommendations"""
    environment: str = Field(..., description="Target environment")
    target_type: str = Field(default="unknown", description="Target type")
    agent_type: str = Field(..., description="Agent type requesting recommendation")
    available_techniques: List[str] = Field(..., description="Available techniques")
    context: Dict[str, Any] = Field(default={}, description="Additional context")

class TechniqueRecommendationResponse(BaseModel):
    """Response model for technique recommendations"""
    technique_id: Optional[str]
    confidence: float
    reason: str
    strategy: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default=[])

# Dependency injection
async def get_agent_scheduler() -> AgentScheduler:
    """Get the agent scheduler instance"""
    if agent_scheduler is None:
        raise HTTPException(status_code=503, detail="Agent scheduler not initialized")
    return agent_scheduler

async def get_capability_registry() -> CapabilityRegistry:
    """Get the capability registry instance"""
    if capability_registry is None:
        raise HTTPException(status_code=503, detail="Capability registry not initialized")
    return capability_registry

async def get_telemetry_collector() -> TelemetryCollector:
    """Get the telemetry collector instance"""
    if telemetry_collector is None:
        raise HTTPException(status_code=503, detail="Telemetry collector not initialized")
    return telemetry_collector

async def get_autonomous_explorer() -> AutonomousExplorer:
    """Get the autonomous explorer instance"""
    if autonomous_explorer is None:
        raise HTTPException(status_code=503, detail="Autonomous explorer not initialized")
    return autonomous_explorer

# API Endpoints

@router.post("/missions", response_model=RedBlueMissionResponse)
async def create_red_blue_mission(
    request: RedBlueMissionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    scheduler: AgentScheduler = Depends(get_agent_scheduler)
):
    """
    Create a new red/blue team mission with comprehensive agent orchestration.
    
    This endpoint creates a mission that can include both offensive (red team) and
    defensive (blue team) operations, with autonomous learning capabilities.
    """
    try:
        # Convert request to mission configuration
        mission_config = {
            "name": request.name,
            "description": request.description,
            "environment": request.environment,
            "objectives": request.objectives,
            "targets": [target.dict() for target in request.targets],
            "red_team_config": request.red_team_config,
            "blue_team_config": request.blue_team_config,
            "constraints": request.constraints,
            "timeout_seconds": request.timeout_seconds,
            "tenant_id": str(tenant_id),
            "learning_enabled": request.learning_enabled
        }
        
        # Create mission
        mission_id = await scheduler.create_mission(mission_config)
        
        # Start mission in background
        background_tasks.add_task(scheduler.start_mission, mission_id)
        
        # Get mission details for response
        mission = await scheduler.get_mission_status(mission_id)
        if not mission:
            raise HTTPException(status_code=500, detail="Failed to retrieve mission status")
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("red_blue_mission_created", 1)
        
        # Add tracing context
        add_trace_context(
            operation="red_blue_mission_created",
            mission_id=mission_id,
            tenant_id=str(tenant_id),
            agent_count=len(mission.agents),
            environment=request.environment
        )
        
        logger.info(f"Created red/blue mission {mission_id} for tenant {tenant_id}")
        
        return RedBlueMissionResponse(
            mission_id=mission.mission_id,
            name=mission.name,
            status=mission.status.value,
            environment=mission.environment.value,
            created_at=mission.created_at.isoformat(),
            agents=[{
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status.value
            } for agent in mission.agents]
        )
        
    except ValueError as e:
        logger.error(f"Invalid request for red/blue mission: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create red/blue mission: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/missions/{mission_id}", response_model=RedBlueMissionResponse)
async def get_red_blue_mission_status(
    mission_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    scheduler: AgentScheduler = Depends(get_agent_scheduler)
):
    """
    Get detailed status of a red/blue team mission including agent progress and results.
    """
    try:
        mission = await scheduler.get_mission_status(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        
        # Calculate progress
        completed_agents = len([a for a in mission.agents if a.status.value == "completed"])
        total_agents = len(mission.agents)
        progress_percentage = (completed_agents / total_agents * 100) if total_agents > 0 else 0
        
        # Collect results from telemetry
        results = {}
        if telemetry_collector:
            mission_metrics = await telemetry_collector.get_mission_metrics(mission_id)
            results.update(mission_metrics)
        
        return RedBlueMissionResponse(
            mission_id=mission.mission_id,
            name=mission.name,
            status=mission.status.value,
            environment=mission.environment.value,
            progress={
                "completion_percentage": progress_percentage,
                "completed_agents": completed_agents,
                "total_agents": total_agents
            },
            agents=[{
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status.value,
                "sandbox_id": agent.sandbox_id,
                "tasks_completed": len([t for t in agent.tasks if t.task_id in agent.results]),
                "tasks_total": len(agent.tasks)
            } for agent in mission.agents],
            results=results,
            created_at=mission.created_at.isoformat(),
            started_at=mission.started_at.isoformat() if mission.started_at else None,
            completed_at=mission.completed_at.isoformat() if mission.completed_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mission status for {mission_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/missions", response_model=List[RedBlueMissionResponse])
async def list_red_blue_missions(
    tenant_id: UUID = Depends(get_current_tenant_id),
    status: Optional[str] = Query(None, description="Filter by mission status"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    limit: int = Query(50, le=100, description="Maximum number of missions to return"),
    offset: int = Query(0, description="Number of missions to skip"),
    scheduler: AgentScheduler = Depends(get_agent_scheduler)
):
    """
    List red/blue team missions with optional filtering and pagination.
    """
    try:
        missions = await scheduler.list_missions()
        
        # Apply filters
        if status:
            missions = [m for m in missions if m.status.value == status]
        if environment:
            missions = [m for m in missions if m.environment.value == environment]
        
        # Apply pagination
        total = len(missions)
        missions = missions[offset:offset + limit]
        
        # Convert to response format
        mission_responses = []
        for mission in missions:
            completed_agents = len([a for a in mission.agents if a.status.value == "completed"])
            total_agents = len(mission.agents)
            progress_percentage = (completed_agents / total_agents * 100) if total_agents > 0 else 0
            
            mission_responses.append(RedBlueMissionResponse(
                mission_id=mission.mission_id,
                name=mission.name,
                status=mission.status.value,
                environment=mission.environment.value,
                progress={"completion_percentage": progress_percentage},
                created_at=mission.created_at.isoformat(),
                started_at=mission.started_at.isoformat() if mission.started_at else None,
                completed_at=mission.completed_at.isoformat() if mission.completed_at else None
            ))
        
        return mission_responses
        
    except Exception as e:
        logger.error(f"Failed to list missions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/missions/{mission_id}/stop")
async def stop_red_blue_mission(
    mission_id: str,
    force: bool = Query(False, description="Force stop the mission"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    scheduler: AgentScheduler = Depends(get_agent_scheduler)
):
    """
    Stop a running red/blue team mission and cleanup all associated resources.
    """
    try:
        success = await scheduler.stop_mission(mission_id, force)
        if not success:
            raise HTTPException(status_code=404, detail="Mission not found or cannot be stopped")
        
        # Broadcast update to WebSocket subscribers
        await connection_manager.broadcast_mission_update(mission_id, {
            "type": "mission_stopped",
            "mission_id": mission_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Stopped red/blue mission {mission_id}")
        
        return {"message": "Mission stopped successfully", "mission_id": mission_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop mission {mission_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/capabilities/agents", response_model=List[AgentCapabilityResponse])
async def get_agent_capabilities(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    category: Optional[str] = Query(None, description="Filter by technique category"),
    registry: CapabilityRegistry = Depends(get_capability_registry),
    scheduler: AgentScheduler = Depends(get_agent_scheduler)
):
    """
    Get available agent capabilities and techniques for the specified environment.
    """
    try:
        # Get agent registry from scheduler
        agent_registry = scheduler.agent_registry
        
        capabilities = []
        for agent_type, agent_info in agent_registry.items():
            # Filter by environment if specified
            if environment:
                env = Environment(environment)
                # Check if agent has techniques available in this environment
                agent_techniques = []
                for tech_category in agent_info["categories"]:
                    techniques = await registry.get_allowed_techniques(env, tech_category)
                    agent_techniques.extend(techniques)
                
                if not agent_techniques:
                    continue
            else:
                agent_techniques = []
            
            capabilities.append(AgentCapabilityResponse(
                agent_type=agent_type,
                name=agent_info["name"],
                description=agent_info.get("description", ""),
                supported_categories=[cat.value for cat in agent_info["categories"]],
                techniques=[{
                    "technique_id": tech.id,
                    "name": tech.name,
                    "category": tech.category.value,
                    "risk_level": tech.risk_level,
                    "description": tech.description
                } for tech in agent_techniques],
                resource_requirements={
                    "cpu_cores": agent_info["resource_requirements"].cpu_cores,
                    "memory_mb": agent_info["resource_requirements"].memory_mb,
                    "disk_mb": agent_info["resource_requirements"].disk_mb
                },
                environments=["development", "staging", "cyber_range"]  # TODO: Get from config
            ))
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get agent capabilities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/capabilities/recommend", response_model=TechniqueRecommendationResponse)
async def recommend_technique(
    request: TechniqueRecommendationRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    explorer: AutonomousExplorer = Depends(get_autonomous_explorer)
):
    """
    Get AI-powered technique recommendations based on context and learning history.
    
    This endpoint uses the autonomous learning engine to suggest the most effective
    techniques for the given context.
    """
    try:
        # Prepare context for recommendation
        context = {
            "environment": request.environment,
            "target_type": request.target_type,
            "agent_type": request.agent_type,
            "available_techniques": request.available_techniques,
            "tenant_id": str(tenant_id),
            **request.context
        }
        
        # Get recommendation from autonomous explorer
        recommendation = await explorer.suggest_technique(context)
        
        # Get alternative recommendations
        alternatives = []
        if len(request.available_techniques) > 1:
            # Get top 3 alternatives
            for technique_id in request.available_techniques[:3]:
                if technique_id != recommendation["technique_id"]:
                    alt_context = context.copy()
                    alt_context["technique_id"] = technique_id
                    alt_rec = await explorer.suggest_technique(alt_context)
                    alternatives.append({
                        "technique_id": technique_id,
                        "confidence": alt_rec["confidence"],
                        "reason": alt_rec["reason"]
                    })
        
        return TechniqueRecommendationResponse(
            technique_id=recommendation["technique_id"],
            confidence=recommendation["confidence"],
            reason=recommendation["reason"],
            strategy=recommendation.get("strategy"),
            alternatives=alternatives
        )
        
    except Exception as e:
        logger.error(f"Failed to recommend technique: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/learning/feedback")
async def record_technique_feedback(
    technique_id: str,
    success: bool,
    execution_time: float,
    context: Dict[str, Any] = {},
    tenant_id: UUID = Depends(get_current_tenant_id),
    explorer: AutonomousExplorer = Depends(get_autonomous_explorer)
):
    """
    Record feedback from technique execution to improve learning algorithms.
    
    This endpoint allows agents to report back on technique execution results,
    enabling the autonomous learning system to improve future recommendations.
    """
    try:
        # Add tenant context
        context["tenant_id"] = str(tenant_id)
        context["timestamp"] = datetime.utcnow().isoformat()
        
        # Record result for learning
        await explorer.record_technique_result(technique_id, success, execution_time, context)
        
        logger.debug(f"Recorded technique feedback: {technique_id} - Success: {success}")
        
        return {"message": "Feedback recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record technique feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/learning/statistics")
async def get_learning_statistics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    explorer: AutonomousExplorer = Depends(get_autonomous_explorer)
):
    """
    Get comprehensive learning statistics and model performance metrics.
    """
    try:
        stats = await explorer.get_learning_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get learning statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/telemetry/mission/{mission_id}")
async def get_mission_telemetry(
    mission_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    time_range: int = Query(3600, description="Time range in seconds"),
    collector: TelemetryCollector = Depends(get_telemetry_collector)
):
    """
    Get telemetry data for a specific mission.
    """
    try:
        time_delta = timedelta(seconds=time_range)
        metrics = await collector.get_mission_metrics(mission_id)
        
        return {
            "mission_id": mission_id,
            "time_range_seconds": time_range,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get mission telemetry: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/telemetry/agent/{agent_id}")
async def get_agent_telemetry(
    agent_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    time_range: int = Query(3600, description="Time range in seconds"),
    collector: TelemetryCollector = Depends(get_telemetry_collector)
):
    """
    Get telemetry data for a specific agent.
    """
    try:
        time_delta = timedelta(seconds=time_range)
        metrics = await collector.get_agent_metrics(agent_id, time_delta)
        
        return {
            "agent_id": agent_id,
            "time_range_seconds": time_range,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent telemetry: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def get_red_blue_health():
    """
    Get health status of the red/blue agent framework components.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check agent scheduler
        if agent_scheduler:
            scheduler_stats = await agent_scheduler.get_statistics()
            health_status["components"]["agent_scheduler"] = {
                "status": "healthy",
                "active_missions": scheduler_stats["total_missions"],
                "total_agents": scheduler_stats["total_agents"]
            }
        else:
            health_status["components"]["agent_scheduler"] = {"status": "unavailable"}
        
        # Check capability registry
        if capability_registry:
            registry_stats = await capability_registry.get_statistics()
            health_status["components"]["capability_registry"] = {
                "status": "healthy",
                "total_techniques": registry_stats["total_techniques"],
                "environments": len(registry_stats["environments"])
            }
        else:
            health_status["components"]["capability_registry"] = {"status": "unavailable"}
        
        # Check sandbox orchestrator
        if sandbox_orchestrator:
            sandbox_stats = await sandbox_orchestrator.get_statistics()
            health_status["components"]["sandbox_orchestrator"] = {
                "status": "healthy",
                "active_sandboxes": sandbox_stats["total_sandboxes"]
            }
        else:
            health_status["components"]["sandbox_orchestrator"] = {"status": "unavailable"}
        
        # Check telemetry collector
        if telemetry_collector:
            health_status["components"]["telemetry_collector"] = {"status": "healthy"}
        else:
            health_status["components"]["telemetry_collector"] = {"status": "unavailable"}
        
        # Check autonomous explorer
        if autonomous_explorer:
            learning_stats = await autonomous_explorer.get_learning_statistics()
            health_status["components"]["autonomous_explorer"] = {
                "status": "healthy",
                "model_accuracy": learning_stats["learning_state"]["model_accuracy"],
                "total_techniques": learning_stats["performance_summary"]["total_techniques"]
            }
        else:
            health_status["components"]["autonomous_explorer"] = {"status": "unavailable"}
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# WebSocket endpoint for real-time updates
@router.websocket("/ws/{mission_id}")
async def mission_websocket(websocket: WebSocket, mission_id: str):
    """
    WebSocket endpoint for real-time mission updates.
    """
    await connection_manager.connect(websocket)
    await connection_manager.subscribe_to_mission(websocket, mission_id)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Handle client commands (ping, subscribe, etc.)
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# Startup/Shutdown Events
async def initialize_red_blue_framework():
    """Initialize the red/blue agent framework components"""
    global agent_scheduler, capability_registry, sandbox_orchestrator, telemetry_collector, autonomous_explorer
    
    try:
        logger.info("Initializing Red/Blue Agent Framework...")
        
        # Get infrastructure dependencies
        redis_client = await get_cache_client()
        postgres_pool = await get_database_pool()
        
        # Initialize components
        capability_registry = CapabilityRegistry(redis_client=redis_client)
        await capability_registry.initialize()
        
        sandbox_orchestrator = SandboxOrchestrator(redis_client=redis_client)
        await sandbox_orchestrator.initialize()
        
        telemetry_collector = TelemetryCollector(redis_client=redis_client, postgres_pool=postgres_pool)
        await telemetry_collector.initialize()
        
        autonomous_explorer = AutonomousExplorer(redis_client=redis_client, postgres_pool=postgres_pool)
        await autonomous_explorer.initialize()
        
        agent_scheduler = AgentScheduler(
            capability_registry=capability_registry,
            sandbox_orchestrator=sandbox_orchestrator,
            redis_client=redis_client
        )
        await agent_scheduler.initialize()
        
        logger.info("Red/Blue Agent Framework initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Red/Blue Agent Framework: {e}")
        raise

async def shutdown_red_blue_framework():
    """Shutdown the red/blue agent framework components"""
    global agent_scheduler, capability_registry, sandbox_orchestrator, telemetry_collector, autonomous_explorer
    
    try:
        logger.info("Shutting down Red/Blue Agent Framework...")
        
        if agent_scheduler:
            await agent_scheduler.shutdown()
        if sandbox_orchestrator:
            await sandbox_orchestrator.shutdown()
        if telemetry_collector:
            await telemetry_collector.shutdown()
        if autonomous_explorer:
            await autonomous_explorer.shutdown()
            
        logger.info("Red/Blue Agent Framework shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during Red/Blue Agent Framework shutdown: {e}")

# Background task for mission monitoring
async def monitor_missions():
    """Background task to monitor missions and send WebSocket updates"""
    while True:
        try:
            if agent_scheduler:
                missions = await agent_scheduler.list_missions()
                
                for mission in missions:
                    if mission.status in [MissionStatus.EXECUTING]:
                        # Send progress update
                        completed_agents = len([a for a in mission.agents if a.status.value == "completed"])
                        total_agents = len(mission.agents)
                        progress = (completed_agents / total_agents * 100) if total_agents > 0 else 0
                        
                        update_data = {
                            "type": "mission_progress",
                            "mission_id": mission.mission_id,
                            "progress": progress,
                            "completed_agents": completed_agents,
                            "total_agents": total_agents,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        await connection_manager.broadcast_mission_update(mission.mission_id, update_data)
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in mission monitoring: {e}")
            await asyncio.sleep(60)

# Auto-start monitoring task
asyncio.create_task(monitor_missions())