"""
Unified Intelligence PTaaS Router - Strategic Enhancement
Production-ready API integration between Unified Intelligence Command Center and PTaaS

This router provides seamless integration between the Unified Intelligence Command Center
and the existing PTaaS infrastructure, enabling autonomous intelligence-guided scanning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# XORB Intelligence imports
from ...xorb.intelligence.unified_intelligence_command_center import (
    get_unified_intelligence_command_center,
    UnifiedIntelligenceCommandCenter,
    UnifiedMission,
    MissionPriority,
    IntelligenceSource,
    OperationStatus
)

# Existing PTaaS imports
from ..services.ptaas_orchestrator_service import (
    get_ptaas_orchestrator, 
    PTaaSOrchestrator,
    PTaaSTarget,
    PTaaSSession
)
from ..services.intelligence_service import IntelligenceService, get_intelligence_service
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Unified Intelligence PTaaS"])

# Request/Response Models
class IntelligenceGuidedScanRequest(BaseModel):
    """Request model for intelligence-guided scanning"""
    mission_name: str = Field(..., description="Name of the intelligence mission")
    mission_description: str = Field(..., description="Description of the mission")
    priority: str = Field(default="medium", description="Mission priority (critical, high, medium, low)")
    
    # Intelligence requirements
    threat_analysis_required: bool = Field(default=True, description="Enable threat prediction analysis")
    behavioral_analysis: bool = Field(default=True, description="Enable behavioral pattern analysis")
    correlation_analysis: bool = Field(default=True, description="Enable intelligence correlation")
    
    # Scanning configuration
    targets: List[Dict[str, Any]] = Field(..., description="Targets for intelligent scanning")
    scan_profiles: List[str] = Field(default=["comprehensive"], description="Scan profiles to use")
    autonomous_adaptation: bool = Field(default=True, description="Enable autonomous scan adaptation")
    
    # Red team integration (optional)
    enable_red_team_simulation: bool = Field(default=False, description="Enable red team scenario simulation")
    red_team_scenarios: List[str] = Field(default_factory=list, description="Specific red team scenarios")
    
    # Safety and compliance
    safety_level: str = Field(default="high", description="Safety level for autonomous operations")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Required compliance frameworks")
    human_oversight_required: bool = Field(default=True, description="Require human oversight")
    
    # Mission parameters
    max_duration: Optional[int] = Field(default=3600, description="Maximum mission duration in seconds")
    success_criteria: List[str] = Field(default_factory=list, description="Mission success criteria")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class IntelligenceMissionResponse(BaseModel):
    """Response model for intelligence mission"""
    mission_id: str
    mission_name: str
    priority: str
    status: str
    created_at: str
    estimated_duration: Optional[str] = None
    
    # Intelligence components
    threat_prediction_tasks: List[Dict[str, Any]]
    ptaas_scans: List[Dict[str, Any]]
    red_team_operations: List[Dict[str, Any]]
    
    # Execution tracking
    progress_metrics: Dict[str, float]
    intelligence_assets_generated: int
    coordination_events: int
    
    # Results (when completed)
    results: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)

class IntelligenceAssetResponse(BaseModel):
    """Response model for intelligence assets"""
    asset_id: str
    asset_type: str
    source: str
    confidence_score: float
    timestamp: str
    data_summary: Dict[str, Any]
    correlated_assets: List[str]

class UnifiedAnalyticsResponse(BaseModel):
    """Response model for unified analytics"""
    analytics_id: str
    mission_count: int
    intelligence_fusion_results: List[Dict[str, Any]]
    threat_intelligence_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

@router.post("/intelligence-guided-scan", response_model=IntelligenceMissionResponse)
async def create_intelligence_guided_scan(
    request: IntelligenceGuidedScanRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center),
    ptaas_orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    intelligence_service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    Create intelligence-guided scanning mission
    
    This endpoint creates a unified mission that coordinates threat intelligence,
    autonomous red team operations, and PTaaS scanning for comprehensive
    security assessment with AI guidance.
    """
    try:
        logger.info(f"Creating intelligence-guided scan mission: {request.mission_name}")
        
        # Build mission specification
        mission_spec = {
            "name": request.mission_name,
            "description": request.mission_description,
            "priority": request.priority,
            "objectives": [
                "Conduct comprehensive threat intelligence analysis",
                "Perform intelligent security scanning",
                "Generate actionable security insights",
                "Provide strategic security recommendations"
            ],
            
            # Intelligence requirements
            "threat_analysis": request.threat_analysis_required,
            "behavioral_analysis": request.behavioral_analysis,
            "correlation_analysis": request.correlation_analysis,
            
            # PTaaS scanning configuration
            "ptaas_scans": [
                {
                    "targets": request.targets,
                    "scan_profiles": request.scan_profiles,
                    "autonomous_adaptation": request.autonomous_adaptation,
                    "tenant_id": str(tenant_id)
                }
            ],
            
            # Red team integration
            "red_team_operations": [
                {
                    "enabled": request.enable_red_team_simulation,
                    "scenarios": request.red_team_scenarios,
                    "safety_level": request.safety_level
                }
            ] if request.enable_red_team_simulation else [],
            
            # Safety and compliance
            "safety_level": request.safety_level,
            "compliance_requirements": request.compliance_frameworks,
            "human_oversight_required": request.human_oversight_required,
            
            # Mission parameters
            "max_duration": request.max_duration,
            "success_criteria": request.success_criteria or [
                "Complete threat intelligence analysis",
                "Execute comprehensive security scans",
                "Generate fusion intelligence report",
                "Provide actionable recommendations"
            ],
            "metadata": request.metadata or {}
        }
        
        # Plan unified mission
        mission = await command_center.plan_unified_mission(mission_spec)
        
        # Start mission execution in background
        background_tasks.add_task(
            _execute_intelligence_mission_background,
            command_center,
            mission.mission_id,
            ptaas_orchestrator,
            intelligence_service,
            tenant_id
        )
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("intelligence_guided_scan_created", 1)
        
        # Add tracing context
        add_trace_context(
            operation="intelligence_guided_scan_created",
            mission_id=mission.mission_id,
            tenant_id=str(tenant_id),
            targets_count=len(request.targets),
            priority=request.priority
        )
        
        logger.info(f"Intelligence-guided scan mission {mission.mission_id} created successfully")
        
        return IntelligenceMissionResponse(
            mission_id=mission.mission_id,
            mission_name=mission.name,
            priority=mission.priority.value,
            status=mission.status.value,
            created_at=mission.created_at.isoformat(),
            estimated_duration=str(mission.estimated_duration),
            threat_prediction_tasks=mission.threat_prediction_tasks,
            ptaas_scans=mission.ptaas_scans,
            red_team_operations=mission.red_team_operations,
            progress_metrics=mission.progress_metrics,
            intelligence_assets_generated=len(mission.generated_assets),
            coordination_events=0  # Will be updated during execution
        )
        
    except ValueError as e:
        logger.error(f"Invalid request for intelligence-guided scan: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create intelligence-guided scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/missions/{mission_id}", response_model=IntelligenceMissionResponse)
async def get_intelligence_mission_status(
    mission_id: str = Path(..., description="Mission ID"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center)
):
    """
    Get intelligence mission status and results
    
    Returns comprehensive information about the mission including execution status,
    progress metrics, generated intelligence assets, and results if completed.
    """
    try:
        # Get mission from command center
        if mission_id not in command_center.active_missions:
            # Check mission history
            historical_mission = None
            for mission in command_center.mission_history:
                if mission.mission_id == mission_id:
                    historical_mission = mission
                    break
            
            if not historical_mission:
                raise HTTPException(status_code=404, detail="Mission not found")
            
            mission = historical_mission
        else:
            mission = command_center.active_missions[mission_id]
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("intelligence_mission_status_checked", 1)
        
        return IntelligenceMissionResponse(
            mission_id=mission.mission_id,
            mission_name=mission.name,
            priority=mission.priority.value,
            status=mission.status.value,
            created_at=mission.created_at.isoformat(),
            estimated_duration=str(mission.estimated_duration),
            threat_prediction_tasks=mission.threat_prediction_tasks,
            ptaas_scans=mission.ptaas_scans,
            red_team_operations=mission.red_team_operations,
            progress_metrics=mission.progress_metrics,
            intelligence_assets_generated=len(mission.generated_assets),
            coordination_events=len(mission.performance_data.get("coordination_events", [])),
            results=mission.results if mission.status == OperationStatus.COMPLETED else None,
            recommendations=mission.recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mission status for {mission_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/missions/{mission_id}/intelligence-assets", response_model=List[IntelligenceAssetResponse])
async def get_mission_intelligence_assets(
    mission_id: str = Path(..., description="Mission ID"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    source: Optional[str] = Query(None, description="Filter by intelligence source"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center)
):
    """
    Get intelligence assets generated by a mission
    
    Returns all intelligence assets (threat predictions, scan results, correlations)
    generated during mission execution.
    """
    try:
        # Verify mission exists
        if mission_id not in command_center.active_missions:
            # Check mission history
            historical_mission = None
            for mission in command_center.mission_history:
                if mission.mission_id == mission_id:
                    historical_mission = mission
                    break
            
            if not historical_mission:
                raise HTTPException(status_code=404, detail="Mission not found")
            
            mission = historical_mission
        else:
            mission = command_center.active_missions[mission_id]
        
        # Get intelligence assets for this mission
        mission_assets = []
        for asset_id in mission.generated_assets:
            if asset_id in command_center.intelligence_assets:
                asset = command_center.intelligence_assets[asset_id]
                
                # Apply filters
                if asset_type and asset.asset_type != asset_type:
                    continue
                if source and asset.source.value != source:
                    continue
                
                # Create response object
                asset_response = IntelligenceAssetResponse(
                    asset_id=asset.asset_id,
                    asset_type=asset.asset_type,
                    source=asset.source.value,
                    confidence_score=asset.confidence_score,
                    timestamp=asset.timestamp.isoformat(),
                    data_summary={
                        "keys": list(asset.data.keys()),
                        "data_size": len(str(asset.data)),
                        "analysis_results": asset.analysis_results
                    },
                    correlated_assets=asset.correlated_assets
                )
                
                mission_assets.append(asset_response)
        
        return mission_assets
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get intelligence assets for mission {mission_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/missions/{mission_id}/correlate")
async def correlate_mission_intelligence(
    mission_id: str = Path(..., description="Mission ID"),
    correlation_parameters: Dict[str, Any],
    tenant_id: UUID = Depends(get_current_tenant_id),
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center)
):
    """
    Correlate intelligence assets from a mission
    
    Performs advanced correlation analysis on intelligence assets generated
    during mission execution to identify patterns and relationships.
    """
    try:
        # Verify mission exists
        if mission_id not in command_center.active_missions:
            historical_mission = None
            for mission in command_center.mission_history:
                if mission.mission_id == mission_id:
                    historical_mission = mission
                    break
            
            if not historical_mission:
                raise HTTPException(status_code=404, detail="Mission not found")
            
            mission = historical_mission
        else:
            mission = command_center.active_missions[mission_id]
        
        # Get mission intelligence assets
        mission_assets = [
            command_center.intelligence_assets[asset_id]
            for asset_id in mission.generated_assets
            if asset_id in command_center.intelligence_assets
        ]
        
        if not mission_assets:
            return {"message": "No intelligence assets found for correlation", "correlation_id": None}
        
        # Perform intelligence fusion
        fusion_method = correlation_parameters.get("method", "weighted_consensus")
        fusion_result = await command_center.fusion_engine.fuse_intelligence(
            mission_assets, fusion_method
        )
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("intelligence_correlation_performed", 1)
        
        return {
            "correlation_id": fusion_result.fusion_id,
            "assets_correlated": len(fusion_result.source_assets),
            "confidence_score": fusion_result.confidence_score,
            "correlation_strength": fusion_result.correlation_strength,
            "threat_level": fusion_result.threat_level,
            "recommended_actions": fusion_result.recommended_actions,
            "fusion_timestamp": fusion_result.fusion_timestamp.isoformat(),
            "expires_at": fusion_result.expires_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to correlate intelligence for mission {mission_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analytics/unified", response_model=UnifiedAnalyticsResponse)
async def get_unified_analytics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    time_range_hours: int = Query(24, description="Time range for analytics in hours"),
    include_predictions: bool = Query(True, description="Include threat predictions"),
    include_correlations: bool = Query(True, description="Include intelligence correlations"),
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center)
):
    """
    Get unified analytics across all intelligence operations
    
    Provides comprehensive analytics combining threat intelligence, PTaaS results,
    and autonomous operations for strategic insights.
    """
    try:
        # Get command center metrics
        metrics = await command_center.get_command_center_metrics()
        
        # Calculate time range
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Get recent fusion results
        recent_fusions = [
            {
                "fusion_id": result.fusion_id,
                "confidence_score": result.confidence_score,
                "correlation_strength": result.correlation_strength,
                "threat_level": result.threat_level,
                "timestamp": result.fusion_timestamp.isoformat()
            }
            for result in command_center.fusion_engine.fusion_history
            if result.fusion_timestamp >= cutoff_time
        ]
        
        # Get threat intelligence summary
        threat_intelligence_summary = {
            "total_assets": len(command_center.intelligence_assets),
            "recent_assets": len([
                asset for asset in command_center.intelligence_assets.values()
                if asset.timestamp >= cutoff_time
            ]),
            "average_confidence": sum(
                asset.confidence_score for asset in command_center.intelligence_assets.values()
            ) / max(len(command_center.intelligence_assets), 1),
            "source_distribution": {}
        }
        
        # Calculate source distribution
        for asset in command_center.intelligence_assets.values():
            source = asset.source.value
            threat_intelligence_summary["source_distribution"][source] = \
                threat_intelligence_summary["source_distribution"].get(source, 0) + 1
        
        # Generate strategic recommendations
        recommendations = await _generate_strategic_recommendations(
            command_center, metrics, recent_fusions
        )
        
        analytics_response = UnifiedAnalyticsResponse(
            analytics_id=str(uuid.uuid4()),
            mission_count=metrics["command_center_metrics"]["total_missions"],
            intelligence_fusion_results=recent_fusions,
            threat_intelligence_summary=threat_intelligence_summary,
            performance_metrics=metrics["performance_indicators"],
            recommendations=recommendations
        )
        
        return analytics_response
        
    except Exception as e:
        logger.error(f"Failed to get unified analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def get_unified_intelligence_health(
    command_center: UnifiedIntelligenceCommandCenter = Depends(get_unified_intelligence_command_center)
):
    """
    Get unified intelligence system health status
    
    Returns health information for the command center and all integrated components.
    """
    try:
        metrics = await command_center.get_command_center_metrics()
        
        return {
            "status": "healthy",
            "command_center_id": command_center.command_center_id,
            "timestamp": datetime.utcnow().isoformat(),
            "component_status": {
                "unified_command_center": "operational",
                "threat_prediction_engine": "operational" if command_center.threat_prediction_engine else "unavailable",
                "autonomous_red_team": "operational" if command_center.autonomous_red_team_engine else "unavailable",
                "payload_engine": "operational" if command_center.payload_engine else "unavailable",
                "intelligence_fusion": "operational"
            },
            "metrics_summary": {
                "active_missions": metrics["command_center_metrics"]["active_missions"],
                "intelligence_assets": metrics["command_center_metrics"]["intelligence_assets"],
                "success_rate": metrics["command_center_metrics"]["success_rate"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get unified intelligence health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Background task functions
async def _execute_intelligence_mission_background(
    command_center: UnifiedIntelligenceCommandCenter,
    mission_id: str,
    ptaas_orchestrator: PTaaSOrchestrator,
    intelligence_service: IntelligenceService,
    tenant_id: UUID
):
    """Execute intelligence mission in background"""
    try:
        logger.info(f"Starting background execution of mission {mission_id}")
        
        # Execute the unified mission
        results = await command_center.execute_unified_mission(mission_id)
        
        logger.info(f"Mission {mission_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background mission execution failed for {mission_id}: {e}")

async def _generate_strategic_recommendations(
    command_center: UnifiedIntelligenceCommandCenter,
    metrics: Dict[str, Any],
    recent_fusions: List[Dict[str, Any]]
) -> List[str]:
    """Generate strategic recommendations based on analytics"""
    recommendations = []
    
    try:
        # Success rate recommendations
        success_rate = metrics["command_center_metrics"]["success_rate"]
        if success_rate < 0.8:
            recommendations.append(
                f"Mission success rate is {success_rate:.1%}. Consider reviewing mission planning and execution processes."
            )
        
        # Intelligence fusion recommendations
        if len(recent_fusions) > 0:
            avg_confidence = sum(f["confidence_score"] for f in recent_fusions) / len(recent_fusions)
            if avg_confidence < 0.7:
                recommendations.append(
                    f"Average intelligence fusion confidence is {avg_confidence:.1%}. Consider improving data quality and source reliability."
                )
        
        # Component availability recommendations
        component_availability = metrics["component_availability"]
        unavailable_components = [k for k, v in component_availability.items() if not v]
        if unavailable_components:
            recommendations.append(
                f"The following components are unavailable: {', '.join(unavailable_components)}. Consider system maintenance."
            )
        
        # Default recommendation
        if not recommendations:
            recommendations.append("System operating optimally. Continue monitoring for any changes in performance metrics.")
        
    except Exception as e:
        logger.error(f"Failed to generate strategic recommendations: {e}")
        recommendations.append("Unable to generate specific recommendations due to analysis error.")
    
    return recommendations