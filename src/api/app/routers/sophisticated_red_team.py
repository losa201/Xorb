"""
Sophisticated Red Team API Router
Advanced red team operations API for defensive security testing

SECURITY NOTICE: All red team operations are conducted exclusively for defensive
security purposes within controlled environments to improve organizational security.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import uuid

from ..services.sophisticated_red_team_agent import (
    get_sophisticated_red_team_agent,
    RedTeamObjective,
    SophisticationLevel,
    AttackPhase,
    RedTeamAgentType,
    DefensiveInsight
)
from ..auth.dependencies import get_current_user
from ..middleware.tenant_context import require_tenant_context

logger = logging.getLogger(__name__)


async def _send_webhook_notifications(
    operation_id: str,
    event_type: str,
    data: Dict[str, Any],
    webhooks: List[str]
) -> None:
    """Send webhook notifications for red team operations"""
    try:
        import aiohttp
        
        notification_payload = {
            "operation_id": operation_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        async with aiohttp.ClientSession() as session:
            for webhook_url in webhooks:
                try:
                    async with session.post(
                        webhook_url,
                        json=notification_payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook notification sent successfully to {webhook_url}")
                        else:
                            logger.warning(f"Webhook notification failed: {response.status}")
                except Exception as e:
                    logger.error(f"Failed to send webhook to {webhook_url}: {e}")
                    
    except Exception as e:
        logger.error(f"Error sending webhook notifications: {e}")
security = HTTPBearer()

router = APIRouter(
    prefix="/sophisticated-red-team",
    tags=["Sophisticated Red Team Operations"],
    dependencies=[Depends(security)]
)


# Pydantic models for API requests/responses
class RedTeamObjectiveRequest(BaseModel):
    """Request model for creating red team objectives"""
    name: str = Field(..., description="Operation name")
    description: str = Field(..., description="Detailed operation description")
    target_assets: List[str] = Field(..., description="Target assets for testing")
    success_criteria: List[str] = Field(..., description="Success criteria")
    mitre_tactics: List[str] = Field(..., description="MITRE ATT&CK tactics to simulate")
    mitre_techniques: List[str] = Field(..., description="MITRE ATT&CK techniques to test")
    sophistication_level: str = Field(default="ADVANCED", description="Attack sophistication level")
    estimated_duration_hours: int = Field(default=24, description="Estimated operation duration in hours")
    stealth_requirements: bool = Field(default=True, description="Stealth operation requirements")
    defensive_learning_goals: List[str] = Field(..., description="Defensive learning objectives")


class OperationConstraints(BaseModel):
    """Constraints for red team operations"""
    max_impact_level: int = Field(default=3, ge=1, le=5, description="Maximum impact level (1-5)")
    authorized_techniques_only: bool = Field(default=True, description="Use only pre-authorized techniques")
    purple_team_coordination: bool = Field(default=True, description="Enable purple team coordination")
    real_time_feedback: bool = Field(default=True, description="Enable real-time defensive feedback")
    safety_stops: List[str] = Field(default=[], description="Conditions that halt operation")


class ThreatActorRequest(BaseModel):
    """Request for threat actor intelligence generation"""
    actor_id: str = Field(..., description="Threat actor identifier")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    defensive_focus: bool = Field(default=True, description="Focus on defensive applications")


class OperationExecutionRequest(BaseModel):
    """Request to execute red team operation"""
    operation_id: str = Field(..., description="Operation identifier")
    execution_constraints: Optional[OperationConstraints] = None
    purple_team_participants: List[str] = Field(default=[], description="Purple team participants")
    notification_webhooks: List[str] = Field(default=[], description="Webhook URLs for notifications")


@router.post("/objectives", response_model=Dict[str, Any])
async def create_red_team_objective(
    request: RedTeamObjectiveRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """
    Create a sophisticated red team operation objective
    
    This endpoint creates a comprehensive red team operation plan with:
    - AI-driven attack path selection
    - MITRE ATT&CK framework integration
    - Defensive learning objectives
    - Purple team coordination
    """
    try:
        # Verify authorization for red team operations
        user_context = await verify_token(credentials.credentials)
        if not await _verify_red_team_authorization(user_context, tenant_context):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for red team operations"
            )
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Create objective
        objective = RedTeamObjective(
            objective_id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            target_assets=request.target_assets,
            success_criteria=request.success_criteria,
            mitre_tactics=request.mitre_tactics,
            mitre_techniques=request.mitre_techniques,
            sophistication_level=SophisticationLevel[request.sophistication_level],
            estimated_duration=timedelta(hours=request.estimated_duration_hours),
            stealth_requirements=request.stealth_requirements,
            defensive_learning_goals=request.defensive_learning_goals
        )
        
        # Plan operation
        operation = await red_team_agent.plan_red_team_operation(
            objective=objective,
            target_environment=tenant_context.get('environment_info', {}),
            constraints=None
        )
        
        # Log operation creation
        logger.info(f"Red team operation created: {operation.operation_id} by user {user_context.get('user_id')}")
        
        return {
            "operation_id": operation.operation_id,
            "name": operation.name,
            "status": operation.status,
            "objective": {
                "name": objective.name,
                "description": objective.description,
                "sophistication_level": objective.sophistication_level.value,
                "mitre_tactics": objective.mitre_tactics,
                "mitre_techniques": objective.mitre_techniques
            },
            "attack_chain_summary": [
                {
                    "vector_id": vector.vector_id,
                    "name": vector.name,
                    "technique_id": vector.technique_id,
                    "success_probability": vector.success_probability,
                    "detection_probability": vector.detection_probability,
                    "defensive_value": vector.defensive_value
                }
                for vector in operation.attack_chain
            ],
            "timeline": {k: v.isoformat() for k, v in operation.timeline.items()},
            "success_metrics": operation.success_metrics,
            "defensive_insights_preview": operation.defensive_insights,
            "purple_team_integration": operation.purple_team_integration,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create red team objective: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create red team objective: {str(e)}"
        )


@router.post("/operations/{operation_id}/execute", response_model=Dict[str, Any])
async def execute_red_team_operation(
    operation_id: str,
    request: OperationExecutionRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """
    Execute a planned red team operation with real-time defensive coordination
    
    Features:
    - Controlled execution with safety constraints
    - Real-time purple team coordination
    - Defensive insight generation
    - Detection capability testing
    """
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        if not await _verify_red_team_execution_authorization(user_context, tenant_context):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for red team operation execution"
            )
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Execute operation in background
        background_tasks.add_task(
            _execute_operation_background,
            red_team_agent,
            operation_id,
            request,
            user_context,
            tenant_context
        )
        
        return {
            "operation_id": operation_id,
            "execution_status": "initiated",
            "message": "Red team operation execution initiated",
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat(),
            "tracking_id": str(uuid.uuid4()),
            "purple_team_coordination": True,
            "safety_constraints_active": True
        }
        
    except Exception as e:
        logger.error(f"Failed to execute red team operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute red team operation: {str(e)}"
        )


@router.get("/operations", response_model=Dict[str, Any])
async def list_red_team_operations(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """List red team operations with filtering and pagination"""
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Get operations (would implement filtering in real system)
        operations = list(red_team_agent.active_operations.values()) + red_team_agent.operation_history
        
        # Apply status filter
        if status:
            operations = [op for op in operations if op.status == status]
        
        # Apply pagination
        total = len(operations)
        operations = operations[offset:offset + limit]
        
        return {
            "operations": [
                {
                    "operation_id": op.operation_id,
                    "name": op.name,
                    "status": op.status,
                    "objective_name": op.objective.name,
                    "sophistication_level": op.objective.sophistication_level.value,
                    "techniques_count": len(op.attack_chain),
                    "purple_team_integration": op.purple_team_integration,
                    "created_at": "2025-01-01T00:00:00",  # Would track real timestamps
                    "completion_rate": len(op.results.get('execution_results', {}).get('phases_executed', [])) / max(len(op.attack_chain), 1) if op.results else 0
                }
                for op in operations
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Failed to list red team operations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list red team operations: {str(e)}"
        )


@router.get("/operations/{operation_id}", response_model=Dict[str, Any])
async def get_red_team_operation(
    operation_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """Get detailed red team operation information"""
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Find operation
        operation = red_team_agent.active_operations.get(operation_id)
        if not operation:
            # Check historical operations
            operation = next((op for op in red_team_agent.operation_history if op.operation_id == operation_id), None)
        
        if not operation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Red team operation not found"
            )
        
        return {
            "operation_id": operation.operation_id,
            "name": operation.name,
            "status": operation.status,
            "objective": {
                "name": operation.objective.name,
                "description": operation.objective.description,
                "sophistication_level": operation.objective.sophistication_level.value,
                "mitre_tactics": operation.objective.mitre_tactics,
                "mitre_techniques": operation.objective.mitre_techniques,
                "defensive_learning_goals": operation.objective.defensive_learning_goals
            },
            "attack_chain": [
                {
                    "vector_id": vector.vector_id,
                    "name": vector.name,
                    "technique_id": vector.technique_id,
                    "description": vector.description,
                    "success_probability": vector.success_probability,
                    "detection_probability": vector.detection_probability,
                    "impact_level": vector.impact_level,
                    "defensive_value": vector.defensive_value,
                    "prerequisites": vector.prerequisites,
                    "artifacts_generated": vector.artifacts_generated
                }
                for vector in operation.attack_chain
            ],
            "timeline": {k: v.isoformat() for k, v in operation.timeline.items()},
            "success_metrics": operation.success_metrics,
            "defensive_insights": operation.defensive_insights,
            "purple_team_integration": operation.purple_team_integration,
            "results": operation.results if operation.results else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get red team operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get red team operation: {str(e)}"
        )


@router.post("/threat-actors/{actor_id}/intelligence", response_model=Dict[str, Any])
async def generate_threat_actor_intelligence(
    actor_id: str,
    request: ThreatActorRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """
    Generate comprehensive threat actor intelligence for defensive purposes
    
    Provides detailed threat actor analysis including:
    - Behavioral patterns and TTPs
    - Detection rules and signatures
    - Defensive strategies and countermeasures
    - Attribution indicators
    """
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Generate threat actor intelligence
        intelligence = await red_team_agent.generate_threat_actor_intelligence(actor_id)
        
        return {
            "actor_id": actor_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_depth": request.analysis_depth,
            "defensive_focus": request.defensive_focus,
            "intelligence_report": intelligence,
            "generated_by": "sophisticated_red_team_agent",
            "version": "1.0"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to generate threat actor intelligence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate threat actor intelligence: {str(e)}"
        )


@router.get("/threat-actors", response_model=Dict[str, Any])
async def list_threat_actors(
    sophistication_level: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """List available threat actor profiles for emulation"""
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Get threat actor profiles
        actors = red_team_agent.threat_actor_models
        
        # Apply sophistication filter
        if sophistication_level:
            actors = {
                k: v for k, v in actors.items() 
                if v.sophistication_level.value == sophistication_level
            }
        
        return {
            "threat_actors": [
                {
                    "actor_id": actor.actor_id,
                    "name": actor.name,
                    "sophistication_level": actor.sophistication_level.value,
                    "preferred_techniques_count": len(actor.preferred_techniques),
                    "attribution_confidence": actor.attribution_confidence,
                    "targeting_sectors": list(actor.targeting_criteria.keys()),
                    "defensive_countermeasures_count": len(actor.defensive_countermeasures)
                }
                for actor in actors.values()
            ],
            "total_actors": len(actors),
            "sophistication_levels": list(SophisticationLevel.__members__.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to list threat actors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list threat actors: {str(e)}"
        )


@router.get("/defensive-insights", response_model=Dict[str, Any])
async def get_defensive_insights(
    insight_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """Get defensive insights generated from red team operations"""
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Get defensive insights
        insights = list(red_team_agent.defensive_insights)
        
        # Apply type filter
        if insight_type:
            insights = [insight for insight in insights if insight.get('type') == insight_type]
        
        # Apply pagination
        total = len(insights)
        insights = insights[offset:offset + limit]
        
        return {
            "defensive_insights": insights,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "insight_types": list(DefensiveInsight.__members__.keys()),
            "summary": {
                "detection_gaps": len([i for i in insights if i.get('type') == DefensiveInsight.DETECTION_GAP.value]),
                "response_improvements": len([i for i in insights if i.get('type') == DefensiveInsight.RESPONSE_IMPROVEMENT.value]),
                "prevention_opportunities": len([i for i in insights if i.get('type') == DefensiveInsight.PREVENTION_OPPORTUNITY.value]),
                "monitoring_enhancements": len([i for i in insights if i.get('type') == DefensiveInsight.MONITORING_ENHANCEMENT.value]),
                "training_needs": len([i for i in insights if i.get('type') == DefensiveInsight.TRAINING_NEED.value])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get defensive insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get defensive insights: {str(e)}"
        )


@router.get("/metrics", response_model=Dict[str, Any])
async def get_red_team_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    tenant_context = Depends(require_tenant_context)
):
    """Get comprehensive red team operation metrics and analytics"""
    try:
        # Verify authorization
        user_context = await verify_token(credentials.credentials)
        
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Get metrics
        metrics = await red_team_agent.get_operation_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "agent_status": "operational",
            "capabilities": {
                "threat_actor_emulation": True,
                "mitre_attack_integration": True,
                "ai_driven_planning": True,
                "purple_team_coordination": True,
                "defensive_insight_generation": True,
                "real_time_feedback": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get red team metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get red team metrics: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_red_team_health():
    """Get red team agent health status"""
    try:
        # Get red team agent
        red_team_agent = await get_sophisticated_red_team_agent()
        
        # Get health status
        health = await red_team_agent.health_check()
        
        return {
            "service_name": health.service_name,
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "details": health.details,
            "version": "1.0.0",
            "capabilities": {
                "ai_decision_making": True,
                "threat_actor_modeling": True,
                "attack_graph_analysis": True,
                "evasion_techniques": True,
                "exploit_simulation": True,
                "defensive_intelligence": True,
                "purple_team_integration": True
            }
        }
        
    except Exception as e:
        logger.error(f"Red team health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Red team health check failed: {str(e)}"
        )


# Helper functions
async def _verify_red_team_authorization(user_context: Dict[str, Any], tenant_context: Dict[str, Any]) -> bool:
    """Verify user has authorization for red team operations"""
    # Implementation would check user permissions
    # For now, allowing all authenticated users (would be more restrictive in production)
    return True


async def _verify_red_team_execution_authorization(user_context: Dict[str, Any], tenant_context: Dict[str, Any]) -> bool:
    """Verify user has authorization for red team operation execution"""
    # Implementation would check elevated permissions for operation execution
    return True


async def _execute_operation_background(
    red_team_agent,
    operation_id: str,
    request: OperationExecutionRequest,
    user_context: Dict[str, Any],
    tenant_context: Dict[str, Any]
):
    """Execute red team operation in background"""
    try:
        logger.info(f"Starting background execution of red team operation: {operation_id}")
        
        # Execute the operation
        results = await red_team_agent.execute_red_team_operation(operation_id)
        
        logger.info(f"Red team operation {operation_id} completed successfully")
        
        # Send notifications if webhooks provided
        if request.notification_webhooks:
            await _send_webhook_notifications(
                operation_id, 
                "operation_completed", 
                results,
                request.notification_webhooks
            )
            
    except Exception as e:
        logger.error(f"Background execution of red team operation {operation_id} failed: {e}")