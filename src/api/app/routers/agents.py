"""
XORB Agent Management API Endpoints
Provides secure API for autonomous agent lifecycle management
"""
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from ..security import (
    SecurityContext,
    get_security_context,
    require_agent_management,
    require_permission,
    Permission
)
from ..auth.rbac_dependencies import require_permission as rbac_require_permission


class AgentType(str, Enum):
    """Types of autonomous agents"""
    SECURITY_ANALYST = "security_analyst"
    THREAT_HUNTER = "threat_hunter"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    COMPLIANCE_MONITOR = "compliance_monitor"
    INCIDENT_RESPONDER = "incident_responder"
    FORENSIC_ANALYZER = "forensic_analyzer"


class AgentStatus(str, Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(str, Enum):
    """Agent capabilities"""
    NETWORK_ANALYSIS = "network_analysis"
    MALWARE_DETECTION = "malware_detection"
    THREAT_INTELLIGENCE = "threat_intelligence"
    LOG_ANALYSIS = "log_analysis"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    COMPLIANCE_CHECKING = "compliance_checking"
    INCIDENT_CORRELATION = "incident_correlation"
    FORENSIC_COLLECTION = "forensic_collection"


# Pydantic Models
class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    uptime_seconds: int = 0
    last_heartbeat: datetime


class CreateAgentRequest(BaseModel):
    """Request to create a new agent"""
    name: str = Field(..., min_length=1, max_length=100)
    agent_type: AgentType
    capabilities: List[AgentCapability] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = Field(None, max_length=500)
    auto_start: bool = True


class UpdateAgentRequest(BaseModel):
    """Request to update agent configuration"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    capabilities: Optional[List[AgentCapability]] = None
    configuration: Optional[Dict[str, Any]] = None
    description: Optional[str] = Field(None, max_length=500)
    status: Optional[AgentStatus] = None


class Agent(BaseModel):
    """Agent information"""
    id: str
    name: str
    agent_type: AgentType
    status: AgentStatus
    capabilities: List[AgentCapability]
    configuration: Dict[str, Any]
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_active: Optional[datetime] = None
    metrics: AgentMetrics
    current_task_id: Optional[str] = None
    owner_id: str
    tags: Dict[str, str] = Field(default_factory=dict)


class AgentStatusResponse(BaseModel):
    """Agent status response"""
    id: str
    status: AgentStatus
    current_task_id: Optional[str]
    metrics: AgentMetrics
    health_check: Dict[str, Any]
    last_heartbeat: datetime


class AgentsListResponse(BaseModel):
    """Response for listing agents"""
    agents: List[Agent]
    total: int
    page: int
    per_page: int
    has_next: bool


class AgentCommandRequest(BaseModel):
    """Request to send command to agent"""
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)


class AgentCommandResponse(BaseModel):
    """Response from agent command"""
    command_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int


# In-memory agent storage (replace with database in production)
agents_store: Dict[str, Agent] = {}
agent_commands: Dict[str, AgentCommandResponse] = {}


router = APIRouter(prefix="/agents", tags=["Agent Management"])


@router.get("/", response_model=AgentsListResponse)
async def list_agents(
    context: SecurityContext = Depends(get_security_context),
    status: Optional[AgentStatus] = Query(None, description="Filter by status"),
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    owner_id: Optional[str] = Query(None, description="Filter by owner"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(100, ge=1, le=1000, description="Items per page"),
) -> AgentsListResponse:
    """List all agents with filtering and pagination"""
    
    # Filter agents based on permissions and filters
    filtered_agents = []
    for agent in agents_store.values():
        # Check if user can view this agent
        if Permission.AGENT_READ not in context.permissions:
            continue
            
        # Apply filters
        if status and agent.status != status:
            continue
        if agent_type and agent.agent_type != agent_type:
            continue
        if owner_id and agent.owner_id != owner_id:
            continue
            
        filtered_agents.append(agent)
    
    # Apply pagination
    total = len(filtered_agents)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_agents = filtered_agents[start_idx:end_idx]
    
    return AgentsListResponse(
        agents=page_agents,
        total=total,
        page=page,
        per_page=per_page,
        has_next=end_idx < total
    )


@router.post("/", response_model=Agent, status_code=201)
async def create_agent(
    request: CreateAgentRequest,
    context: SecurityContext = Depends(require_agent_management)
) -> Agent:
    """Create a new autonomous agent"""
    
    agent_id = str(uuid.uuid4())
    current_time = datetime.utcnow()
    
    # Create agent with initial metrics
    agent = Agent(
        id=agent_id,
        name=request.name,
        agent_type=request.agent_type,
        status=AgentStatus.INITIALIZING if request.auto_start else AgentStatus.IDLE,
        capabilities=request.capabilities,
        configuration=request.configuration,
        description=request.description,
        created_at=current_time,
        updated_at=current_time,
        owner_id=context.user_id,
        metrics=AgentMetrics(
            last_heartbeat=current_time
        )
    )
    
    # Store agent
    agents_store[agent_id] = agent
    
    # Simulate agent initialization
    if request.auto_start:
        asyncio.create_task(_initialize_agent(agent_id))
    
    return agent


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    context: SecurityContext = Depends(get_security_context)
) -> Agent:
    """Get detailed agent information"""
    
    if Permission.AGENT_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: agent:read")
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    current_user = Depends(rbac_require_permission("agent:update"))
) -> Agent:
    """Update agent configuration"""
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check ownership or admin permissions
    if agent.owner_id != context.user_id and Permission.SYSTEM_ADMIN not in context.permissions:
        raise HTTPException(status_code=403, detail="Not authorized to modify this agent")
    
    # Update fields
    if request.name is not None:
        agent.name = request.name
    if request.capabilities is not None:
        agent.capabilities = request.capabilities
    if request.configuration is not None:
        agent.configuration.update(request.configuration)
    if request.description is not None:
        agent.description = request.description
    if request.status is not None:
        agent.status = request.status
    
    agent.updated_at = datetime.utcnow()
    
    return agent


@router.delete("/{agent_id}")
async def terminate_agent(
    agent_id: str,
    current_user = Depends(rbac_require_permission("agent:delete"))
) -> Dict[str, str]:
    """Safely terminate an agent"""
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check ownership or admin permissions
    if agent.owner_id != context.user_id and Permission.SYSTEM_ADMIN not in context.permissions:
        raise HTTPException(status_code=403, detail="Not authorized to terminate this agent")
    
    # Prevent termination if agent is busy with critical tasks
    if agent.status == AgentStatus.BUSY and agent.current_task_id:
        raise HTTPException(
            status_code=409, 
            detail="Cannot terminate agent while processing critical task"
        )
    
    # Mark agent as terminated
    agent.status = AgentStatus.TERMINATED
    agent.updated_at = datetime.utcnow()
    
    # Schedule cleanup
    asyncio.create_task(_cleanup_agent(agent_id))
    
    return {"message": "Agent termination initiated", "agent_id": agent_id}


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    context: SecurityContext = Depends(get_security_context)
) -> AgentStatusResponse:
    """Get real-time agent status"""
    
    if Permission.AGENT_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: agent:read")
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Simulate health check
    health_check = {
        "connectivity": "healthy",
        "resources": "normal",
        "capabilities": "operational",
        "last_check": datetime.utcnow().isoformat()
    }
    
    return AgentStatusResponse(
        id=agent.id,
        status=agent.status,
        current_task_id=agent.current_task_id,
        metrics=agent.metrics,
        health_check=health_check,
        last_heartbeat=agent.metrics.last_heartbeat
    )


@router.post("/{agent_id}/commands", response_model=AgentCommandResponse)
async def send_agent_command(
    agent_id: str,
    request: AgentCommandRequest,
    current_user = Depends(rbac_require_permission("agent:update"))
) -> AgentCommandResponse:
    """Send command to agent"""
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.status not in [AgentStatus.ACTIVE, AgentStatus.IDLE]:
        raise HTTPException(
            status_code=409, 
            detail=f"Agent not available for commands (status: {agent.status})"
        )
    
    command_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Simulate command execution
        result = await _execute_agent_command(agent, request.command, request.parameters)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = AgentCommandResponse(
            command_id=command_id,
            status="completed",
            result=result,
            execution_time_ms=int(execution_time)
        )
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = AgentCommandResponse(
            command_id=command_id,
            status="failed",
            error=str(e),
            execution_time_ms=int(execution_time)
        )
    
    # Store command result
    agent_commands[command_id] = response
    
    return response


@router.get("/{agent_id}/logs")
async def get_agent_logs(
    agent_id: str,
    lines: int = Query(100, ge=1, le=10000, description="Number of log lines"),
    level: Optional[str] = Query(None, description="Log level filter"),
    context: SecurityContext = Depends(get_security_context)
) -> Dict[str, Any]:
    """Get agent logs"""
    
    if Permission.AGENT_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: agent:read")
    
    agent = agents_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Simulate log retrieval
    logs = []
    for i in range(min(lines, 50)):  # Simulate limited logs
        logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": f"Agent {agent.name} operational check {i+1}",
            "component": agent.agent_type.value
        })
    
    return {
        "agent_id": agent_id,
        "logs": logs,
        "total_lines": len(logs),
        "timestamp": datetime.utcnow().isoformat()
    }


# Helper functions
async def _initialize_agent(agent_id: str):
    """Initialize agent asynchronously"""
    await asyncio.sleep(2)  # Simulate initialization time
    
    if agent_id in agents_store:
        agent = agents_store[agent_id]
        agent.status = AgentStatus.ACTIVE
        agent.last_active = datetime.utcnow()
        agent.metrics.last_heartbeat = datetime.utcnow()


async def _cleanup_agent(agent_id: str):
    """Cleanup terminated agent"""
    await asyncio.sleep(5)  # Give time for graceful shutdown
    
    if agent_id in agents_store:
        del agents_store[agent_id]


async def _execute_agent_command(
    agent: Agent, 
    command: str, 
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute command on agent"""
    
    # Simulate command execution based on agent type and capabilities
    await asyncio.sleep(1)  # Simulate processing time
    
    if command == "status_check":
        return {
            "status": "healthy",
            "capabilities_online": len(agent.capabilities),
            "current_tasks": 0
        }
    elif command == "run_scan":
        target = parameters.get("target", "unknown")
        return {
            "scan_id": str(uuid.uuid4()),
            "target": target,
            "status": "initiated",
            "estimated_completion": "2 minutes"
        }
    elif command == "update_config":
        return {
            "config_updated": True,
            "restart_required": False
        }
    else:
        raise ValueError(f"Unknown command: {command}")


# Create some sample agents for testing
async def _initialize_sample_agents():
    """Initialize sample agents for testing"""
    if not agents_store:  # Only create if empty
        sample_agents = [
            CreateAgentRequest(
                name="Primary Security Analyst",
                agent_type=AgentType.SECURITY_ANALYST,
                capabilities=[
                    AgentCapability.THREAT_INTELLIGENCE,
                    AgentCapability.LOG_ANALYSIS,
                    AgentCapability.INCIDENT_CORRELATION
                ],
                description="Main security analysis agent"
            ),
            CreateAgentRequest(
                name="Threat Hunter Alpha",
                agent_type=AgentType.THREAT_HUNTER,
                capabilities=[
                    AgentCapability.NETWORK_ANALYSIS,
                    AgentCapability.MALWARE_DETECTION,
                    AgentCapability.FORENSIC_COLLECTION
                ],
                description="Advanced threat hunting agent"
            )
        ]
        
        for agent_req in sample_agents:
            agent_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            agent = Agent(
                id=agent_id,
                name=agent_req.name,
                agent_type=agent_req.agent_type,
                status=AgentStatus.ACTIVE,
                capabilities=agent_req.capabilities,
                configuration=agent_req.configuration,
                description=agent_req.description,
                created_at=current_time,
                updated_at=current_time,
                last_active=current_time,
                owner_id="system",
                metrics=AgentMetrics(
                    tasks_completed=42,
                    success_rate=0.95,
                    avg_response_time_ms=250.0,
                    cpu_usage_percent=15.0,
                    memory_usage_mb=512.0,
                    uptime_seconds=86400,
                    last_heartbeat=current_time
                )
            )
            
            agents_store[agent_id] = agent


# Initialize sample data (will be called during startup)
# Note: Actual initialization happens in FastAPI lifespan or startup event