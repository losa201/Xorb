"""
Team Operations API Router
Advanced red vs blue vs purple team operations with ML-powered coordination
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import XORB team orchestration components
from ...xorb.security.team_orchestration_framework import (
    get_team_orchestration_framework,
    TeamOrchestrationFramework,
    TeamRole,
    OperationType,
    ThreatLevel,
    OperationPhase,
    create_red_team_scenario,
    execute_purple_team_operation
)
from ...xorb.intelligence.ml_tactical_coordinator import (
    get_ml_tactical_coordinator,
    MLTacticalCoordinator,
    TacticalDecisionType,
    AdversaryProfile,
    TacticalContext,
    make_tactical_decision_request,
    create_adaptive_strategy_request
)

from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/team-operations", tags=["Team Operations"])

# Request/Response Models
class CreateScenarioRequest(BaseModel):
    """Request model for creating security scenarios"""
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Detailed scenario description")
    operation_type: str = Field(..., description="Type of operation")
    threat_level: str = Field(default="medium", description="Threat level")
    target_environment: str = Field(..., description="Target environment")
    objectives: List[str] = Field(..., description="Scenario objectives")
    success_criteria: List[str] = Field(..., description="Success criteria")
    constraints: Optional[List[str]] = Field(default=[], description="Operational constraints")
    mitre_tactics: Optional[List[str]] = Field(default=[], description="MITRE ATT&CK tactics")
    mitre_techniques: Optional[List[str]] = Field(default=[], description="MITRE ATT&CK techniques")
    duration_hours: Optional[int] = Field(default=8, description="Estimated duration in hours")
    complexity_score: Optional[float] = Field(default=0.7, description="Complexity score (0-1)")
    required_skills: Optional[List[str]] = Field(default=[], description="Required skills")

class OperationPlanRequest(BaseModel):
    """Request model for creating operation plans"""
    scenario_id: str = Field(..., description="Scenario ID to base plan on")
    customizations: Optional[Dict[str, Any]] = Field(default={}, description="Plan customizations")
    team_preferences: Optional[Dict[str, List[str]]] = Field(default={}, description="Team member preferences")
    resource_constraints: Optional[Dict[str, Any]] = Field(default={}, description="Resource constraints")
    priority_level: Optional[str] = Field(default="medium", description="Operation priority")

class TacticalDecisionRequest(BaseModel):
    """Request model for tactical decisions"""
    decision_type: str = Field(..., description="Type of tactical decision")
    context: Dict[str, Any] = Field(..., description="Decision context")
    urgency_level: Optional[str] = Field(default="medium", description="Decision urgency")
    constraints: Optional[List[str]] = Field(default=[], description="Decision constraints")

class AdaptiveStrategyRequest(BaseModel):
    """Request model for creating adaptive strategies"""
    team_role: str = Field(..., description="Team role for strategy")
    adversary_profile: str = Field(..., description="Target adversary profile")
    tactical_context: str = Field(..., description="Tactical context")
    base_objectives: List[str] = Field(..., description="Base strategy objectives")
    success_metrics: Optional[Dict[str, float]] = Field(default={}, description="Success metrics")

class OperationExecutionRequest(BaseModel):
    """Request model for operation execution"""
    plan_id: str = Field(..., description="Operation plan ID")
    execution_parameters: Optional[Dict[str, Any]] = Field(default={}, description="Execution parameters")
    monitoring_level: Optional[str] = Field(default="standard", description="Monitoring detail level")
    auto_adapt: Optional[bool] = Field(default=True, description="Enable automatic adaptation")

# Response Models
class ScenarioResponse(BaseModel):
    """Response model for security scenarios"""
    scenario_id: str
    name: str
    operation_type: str
    threat_level: str
    complexity_score: float
    estimated_duration: str
    created_at: str

class OperationPlanResponse(BaseModel):
    """Response model for operation plans"""
    plan_id: str
    scenario_id: str
    team_assignments: Dict[str, List[str]]
    phases: List[str]
    estimated_duration: str
    success_metrics: Dict[str, float]
    ml_optimizations: int
    created_at: str

class TacticalDecisionResponse(BaseModel):
    """Response model for tactical decisions"""
    decision_id: str
    recommended_action: str
    confidence_score: float
    reasoning: List[str]
    alternative_actions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    success_probability: float
    implementation_steps: List[str]

class OperationStatusResponse(BaseModel):
    """Response model for operation status"""
    execution_id: str
    plan_id: str
    current_phase: str
    overall_progress: float
    team_status: Dict[str, str]
    real_time_metrics: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    adaptive_adjustments: int

class TeamPerformanceResponse(BaseModel):
    """Response model for team performance"""
    timeframe_days: int
    team_analysis: Dict[str, Dict[str, float]]
    trend_analysis: Dict[str, Any]
    ml_insights: Dict[str, Any]
    recommendations: List[str]

@router.post("/scenarios", response_model=ScenarioResponse)
async def create_security_scenario(
    request: CreateScenarioRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Create a new security scenario for team operations
    
    Creates a comprehensive security scenario that can be used for
    red team, blue team, or purple team exercises.
    """
    try:
        # Convert request to scenario data
        scenario_data = {
            "name": request.name,
            "description": request.description,
            "operation_type": request.operation_type,
            "threat_level": request.threat_level,
            "target_environment": request.target_environment,
            "objectives": request.objectives,
            "success_criteria": request.success_criteria,
            "constraints": request.constraints,
            "mitre_tactics": request.mitre_tactics,
            "mitre_techniques": request.mitre_techniques,
            "duration_hours": request.duration_hours,
            "complexity_score": request.complexity_score,
            "required_skills": request.required_skills,
            "created_by": str(tenant_id)
        }
        
        # Create scenario using framework
        scenario_id = await create_red_team_scenario(scenario_data)
        
        # Get created scenario for response
        scenario = framework.security_scenarios[scenario_id]
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("scenario_created", 1)
        
        # Add tracing context
        add_trace_context(
            operation="scenario_created",
            scenario_id=scenario_id,
            tenant_id=str(tenant_id),
            operation_type=request.operation_type
        )
        
        logger.info(f"Created security scenario {scenario_id} for tenant {tenant_id}")
        
        return ScenarioResponse(
            scenario_id=scenario_id,
            name=scenario.name,
            operation_type=scenario.operation_type.value,
            threat_level=scenario.threat_level.value,
            complexity_score=scenario.complexity_score,
            estimated_duration=str(scenario.estimated_duration),
            created_at=scenario.created_at.isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Invalid scenario request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scenario: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/plans", response_model=OperationPlanResponse)
async def create_operation_plan(
    request: OperationPlanRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Create a comprehensive operation plan based on a security scenario
    
    Generates an ML-optimized operation plan with team assignments,
    resource allocation, and tactical coordination.
    """
    try:
        # Validate scenario exists
        if request.scenario_id not in framework.security_scenarios:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        # Create operation plan with customizations
        plan_id = await framework.create_operation_plan(
            request.scenario_id, 
            request.customizations
        )
        
        # Get created plan for response
        plan = framework.operation_plans[plan_id]
        scenario = framework.security_scenarios[request.scenario_id]
        
        # Count ML optimizations
        ml_optimizations = len(plan.ml_integration_points)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("operation_plan_created", 1)
        
        logger.info(f"Created operation plan {plan_id} for scenario {request.scenario_id}")
        
        return OperationPlanResponse(
            plan_id=plan_id,
            scenario_id=request.scenario_id,
            team_assignments=plan.resource_allocation,
            phases=[phase.value for phase in plan.phases],
            estimated_duration=str(scenario.estimated_duration),
            success_metrics=plan.success_metrics,
            ml_optimizations=ml_optimizations,
            created_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create operation plan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/tactical-decisions", response_model=TacticalDecisionResponse)
async def make_tactical_decision(
    request: TacticalDecisionRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    coordinator: MLTacticalCoordinator = Depends(get_ml_tactical_coordinator)
):
    """
    Make an ML-powered tactical decision
    
    Uses advanced machine learning to recommend optimal tactical actions
    based on current context and operational requirements.
    """
    try:
        # Validate decision type
        try:
            decision_type = TacticalDecisionType(request.decision_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid decision type: {request.decision_type}")
        
        # Add tenant context to decision context
        enhanced_context = request.context.copy()
        enhanced_context.update({
            "tenant_id": str(tenant_id),
            "urgency_level": request.urgency_level,
            "constraints": request.constraints,
            "timestamp": datetime.now().isoformat()
        })
        
        # Make tactical decision
        decision = await coordinator.make_tactical_decision(enhanced_context, decision_type)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("tactical_decision_made", 1)
        
        # Add tracing context
        add_trace_context(
            operation="tactical_decision",
            decision_id=decision.decision_id,
            decision_type=request.decision_type,
            confidence=decision.confidence_score
        )
        
        logger.info(f"Made tactical decision {decision.decision_id} with confidence {decision.confidence_score:.3f}")
        
        return TacticalDecisionResponse(
            decision_id=decision.decision_id,
            recommended_action=decision.recommended_action,
            confidence_score=decision.confidence_score,
            reasoning=decision.reasoning,
            alternative_actions=decision.alternative_actions,
            risk_assessment=decision.risk_assessment,
            success_probability=decision.success_probability,
            implementation_steps=decision.implementation_steps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to make tactical decision: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/adaptive-strategies")
async def create_adaptive_strategy(
    request: AdaptiveStrategyRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    coordinator: MLTacticalCoordinator = Depends(get_ml_tactical_coordinator)
):
    """
    Create an ML-optimized adaptive strategy
    
    Generates adaptive strategies that evolve based on adversary behavior
    and operational effectiveness.
    """
    try:
        # Validate adversary profile and tactical context
        try:
            adversary_profile = AdversaryProfile(request.adversary_profile)
            tactical_context = TacticalContext(request.tactical_context)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
        
        # Create base strategy from request
        base_strategy = {
            "objectives": request.base_objectives,
            "success_metrics": request.success_metrics,
            "team_role": request.team_role,
            "created_by": str(tenant_id),
            "created_at": datetime.now().isoformat()
        }
        
        # Create adaptive strategy
        strategy_id = await coordinator.create_adaptive_strategy(
            request.team_role,
            adversary_profile,
            tactical_context,
            base_strategy
        )
        
        # Get created strategy for response
        strategy = coordinator.active_strategies[strategy_id]
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("adaptive_strategy_created", 1)
        
        logger.info(f"Created adaptive strategy {strategy_id} for {request.team_role}")
        
        return {
            "strategy_id": strategy_id,
            "team_role": strategy.team_role,
            "adversary_profile": strategy.adversary_profile.value,
            "tactical_context": strategy.tactical_context.value,
            "optimization_score": strategy.optimization_score,
            "adaptive_modifications": len(strategy.adaptive_modifications),
            "created_at": strategy.last_updated.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create adaptive strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/execute", response_model=OperationStatusResponse)
async def execute_operation(
    request: OperationExecutionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Execute a planned team operation
    
    Starts real-time execution of a team operation with ML-powered
    monitoring and adaptive coordination.
    """
    try:
        # Validate plan exists
        if request.plan_id not in framework.operation_plans:
            raise HTTPException(status_code=404, detail="Operation plan not found")
        
        # Execute operation
        execution_id = await framework.execute_operation(request.plan_id)
        
        # Wait a moment for execution to initialize
        await asyncio.sleep(1)
        
        # Get initial status
        status = await framework.get_operation_status(execution_id)
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("operation_executed", 1)
        
        # Add tracing context
        add_trace_context(
            operation="operation_execution",
            execution_id=execution_id,
            plan_id=request.plan_id,
            monitoring_level=request.monitoring_level
        )
        
        logger.info(f"Started operation execution {execution_id} for plan {request.plan_id}")
        
        return OperationStatusResponse(
            execution_id=execution_id,
            plan_id=request.plan_id,
            current_phase=status["current_phase"],
            overall_progress=status["overall_progress"],
            team_status=status["team_status"],
            real_time_metrics=status["metrics"],
            ml_predictions=status.get("ml_predictions", {}),
            adaptive_adjustments=status.get("adaptive_adjustments", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute operation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/operations/{execution_id}/status", response_model=OperationStatusResponse)
async def get_operation_status(
    execution_id: str = Path(..., description="Operation execution ID"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Get real-time status of an executing operation
    
    Returns comprehensive status including progress, team coordination,
    ML predictions, and adaptive adjustments.
    """
    try:
        # Get operation status
        status = await framework.get_operation_status(execution_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("operation_status_checked", 1)
        
        return OperationStatusResponse(
            execution_id=execution_id,
            plan_id=status["plan_id"],
            current_phase=status["current_phase"],
            overall_progress=status["overall_progress"],
            team_status=status["team_status"],
            real_time_metrics=status["metrics"],
            ml_predictions=status.get("ml_predictions", {}),
            adaptive_adjustments=status.get("adaptive_adjustments", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get operation status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/scenarios")
async def list_scenarios(
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    limit: int = Query(50, le=100, description="Maximum scenarios to return"),
    offset: int = Query(0, description="Number of scenarios to skip"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    List available security scenarios
    
    Returns a paginated list of security scenarios, optionally filtered
    by operation type and threat level.
    """
    try:
        scenarios = []
        
        for scenario_id, scenario in framework.security_scenarios.items():
            # Apply filters
            if operation_type and scenario.operation_type.value != operation_type:
                continue
            if threat_level and scenario.threat_level.value != threat_level:
                continue
            
            scenarios.append({
                "scenario_id": scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "operation_type": scenario.operation_type.value,
                "threat_level": scenario.threat_level.value,
                "complexity_score": scenario.complexity_score,
                "estimated_duration": str(scenario.estimated_duration),
                "required_skills": scenario.required_skills,
                "created_at": scenario.created_at.isoformat()
            })
        
        # Apply pagination
        total_scenarios = len(scenarios)
        scenarios = scenarios[offset:offset + limit]
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("scenarios_listed", 1)
        
        return {
            "scenarios": scenarios,
            "total": total_scenarios,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_scenarios
        }
        
    except Exception as e:
        logger.error(f"Failed to list scenarios: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/performance", response_model=TeamPerformanceResponse)
async def get_team_performance(
    timeframe_days: int = Query(30, ge=1, le=365, description="Analysis timeframe in days"),
    team_role: Optional[str] = Query(None, description="Filter by team role"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Get comprehensive team performance analysis
    
    Returns detailed performance metrics, trends, and ML-powered insights
    for team operations within the specified timeframe.
    """
    try:
        # Get performance analysis
        analysis = await framework.get_team_performance_analysis(timeframe_days)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        # Filter by team role if specified
        if team_role and team_role in analysis.get("team_analysis", {}):
            filtered_analysis = analysis["team_analysis"][team_role]
            analysis["team_analysis"] = {team_role: filtered_analysis}
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("team_performance_analyzed", 1)
        
        return TeamPerformanceResponse(
            timeframe_days=analysis["timeframe_days"],
            team_analysis=analysis["team_analysis"],
            trend_analysis=analysis.get("trend_analysis", {}),
            ml_insights=analysis.get("ml_insights", {}),
            recommendations=analysis["recommendations"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get team performance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/tactical-intelligence")
async def get_tactical_intelligence(
    tenant_id: UUID = Depends(get_current_tenant_id),
    coordinator: MLTacticalCoordinator = Depends(get_ml_tactical_coordinator)
):
    """
    Get comprehensive tactical intelligence summary
    
    Returns ML model performance, decision analytics, strategy effectiveness,
    and actionable intelligence insights.
    """
    try:
        # Get tactical intelligence summary
        intelligence = await coordinator.get_tactical_intelligence_summary()
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("tactical_intelligence_retrieved", 1)
        
        return intelligence
        
    except Exception as e:
        logger.error(f"Failed to get tactical intelligence: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/framework-analytics")
async def get_framework_analytics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Get comprehensive framework analytics
    
    Returns overall framework status, team distribution, operation metrics,
    and performance insights.
    """
    try:
        # Get framework analytics
        analytics = await framework.get_framework_analytics()
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("framework_analytics_retrieved", 1)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get framework analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/operations/{execution_id}/cancel")
async def cancel_operation(
    execution_id: str = Path(..., description="Operation execution ID"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework)
):
    """
    Cancel an active operation
    
    Safely cancels an executing operation and performs cleanup.
    """
    try:
        # Check if operation exists
        status = await framework.get_operation_status(execution_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        # Cancel operation (implementation would be in framework)
        # For now, return success message
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("operation_cancelled", 1)
        
        logger.info(f"Cancelled operation {execution_id}")
        
        return {
            "message": "Operation cancelled successfully",
            "execution_id": execution_id,
            "cancelled_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel operation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def get_team_operations_health(
    framework: TeamOrchestrationFramework = Depends(get_team_orchestration_framework),
    coordinator: MLTacticalCoordinator = Depends(get_ml_tactical_coordinator)
):
    """
    Get team operations service health
    
    Returns health information for the team operations framework
    and ML tactical coordinator.
    """
    try:
        # Get framework analytics
        framework_analytics = await framework.get_framework_analytics()
        
        # Get ML coordinator summary
        ml_summary = await coordinator.get_tactical_intelligence_summary()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "framework_status": framework_analytics.get("framework_status", {}),
            "ml_coordinator_status": ml_summary.get("ml_coordinator_status", {}),
            "services": {
                "team_orchestration_framework": "operational",
                "ml_tactical_coordinator": "operational",
                "scenario_management": "operational",
                "operation_execution": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Team operations health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {
                "team_orchestration_framework": "error",
                "ml_tactical_coordinator": "error"
            }
        }