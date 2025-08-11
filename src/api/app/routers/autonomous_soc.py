"""
Autonomous Security Operations Center API Router
Principal Auditor Implementation: Next-Generation SOC 3.0 API

This router provides enterprise-grade API endpoints for:
- Autonomous security operations management
- Real-time threat monitoring and response
- Predictive threat intelligence
- Self-healing infrastructure operations
- SOC performance metrics and analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Internal imports
from ...xorb.intelligence.autonomous_security_operations_center import (
    get_autonomous_soc, 
    AutonomousSecurityOperationsCenter,
    ThreatSeverity,
    OperationMode,
    SecurityThreat,
    AutonomousResponse
)
from ..middleware.auth import get_current_user
from ..middleware.rate_limiting import rate_limit

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/autonomous-soc", tags=["Autonomous SOC"])


# Pydantic models for API
class ThreatDetectionRequest(BaseModel):
    """Request model for manual threat detection"""
    indicators: List[Dict[str, Any]] = Field(..., description="Threat indicators to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    priority: Optional[str] = Field("medium", description="Detection priority level")


class SOCConfigurationRequest(BaseModel):
    """Request model for SOC configuration updates"""
    operation_mode: Optional[str] = Field(None, description="SOC operation mode")
    auto_response_threshold: Optional[float] = Field(None, description="Threshold for automatic responses")
    human_approval_required: Optional[bool] = Field(None, description="Require human approval for responses")


class ThreatResponseRequest(BaseModel):
    """Request model for manual threat response"""
    threat_id: str = Field(..., description="Threat ID to respond to")
    response_type: Optional[str] = Field("automated", description="Type of response to execute")
    additional_actions: Optional[List[Dict[str, Any]]] = Field(None, description="Additional response actions")


class PredictiveForecastRequest(BaseModel):
    """Request model for predictive threat forecasting"""
    horizon_hours: Optional[int] = Field(48, description="Forecast horizon in hours")
    threat_types: Optional[List[str]] = Field(None, description="Specific threat types to forecast")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold")


@router.get("/status", 
           summary="Get Autonomous SOC Status",
           description="Get comprehensive status and metrics for the Autonomous Security Operations Center")
@rate_limit("soc_status", max_calls=60, window_seconds=60)
async def get_soc_status(current_user: Dict = Depends(get_current_user)) -> JSONResponse:
    """Get comprehensive Autonomous SOC status and performance metrics"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Get comprehensive status
        status = await soc.get_soc_status()
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "soc_status": status,
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "endpoint": "autonomous_soc_status"
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get SOC status: {e}")
        raise HTTPException(status_code=500, detail=f"SOC status retrieval failed: {str(e)}")


@router.get("/threats",
           summary="Get Active Threats",
           description="Get all currently active security threats being monitored by the SOC")
@rate_limit("soc_threats", max_calls=120, window_seconds=60)
async def get_active_threats(
    severity: Optional[str] = Query(None, description="Filter by threat severity"),
    limit: int = Query(100, description="Maximum number of threats to return"),
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Get active security threats from the Autonomous SOC"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Get active threats
        active_threats = []
        for threat_id, threat in soc.active_threats.items():
            if severity and threat.severity.value != severity:
                continue
            
            threat_data = {
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type,
                "severity": threat.severity.value,
                "confidence_score": threat.confidence_score,
                "detection_timestamp": threat.detection_timestamp.isoformat(),
                "affected_systems": threat.affected_systems,
                "status": threat.status.value,
                "mitre_techniques": threat.mitre_techniques,
                "potential_impact": threat.potential_impact
            }
            active_threats.append(threat_data)
            
            if len(active_threats) >= limit:
                break
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "active_threats": active_threats,
            "total_count": len(soc.active_threats),
            "filtered_count": len(active_threats),
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "severity_filter": severity,
                "limit": limit
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get active threats: {e}")
        raise HTTPException(status_code=500, detail=f"Active threats retrieval failed: {str(e)}")


@router.post("/detect-threat",
            summary="Manual Threat Detection",
            description="Manually submit threat indicators for analysis by the Autonomous SOC")
@rate_limit("soc_detect", max_calls=30, window_seconds=60)
async def detect_threat(
    request: ThreatDetectionRequest,
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Manually submit threat indicators for autonomous analysis"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Process threat indicators
        threat_analysis = await soc._analyze_threat_indicators(
            indicators=request.indicators,
            context=request.context or {},
            priority=request.priority
        )
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "threat_analysis": threat_analysis,
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "indicators_count": len(request.indicators),
                "priority": request.priority
            }
        })
        
    except Exception as e:
        logger.error(f"Manual threat detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threat detection failed: {str(e)}")


@router.post("/respond-threat",
            summary="Execute Threat Response",
            description="Execute or approve autonomous response to a specific threat")
@rate_limit("soc_respond", max_calls=20, window_seconds=60)
async def respond_to_threat(
    request: ThreatResponseRequest,
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Execute or approve autonomous threat response"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Check if threat exists
        if request.threat_id not in soc.active_threats:
            raise HTTPException(status_code=404, detail=f"Threat {request.threat_id} not found")
        
        threat = soc.active_threats[request.threat_id]
        
        # Generate or execute response
        if request.threat_id in soc.active_responses:
            # Execute existing response
            response = soc.active_responses[request.threat_id]
            execution_result = await soc.incident_responder.execute_response(response)
        else:
            # Generate new response
            response = await soc.incident_responder.analyze_and_respond(threat)
            
            # Add any additional actions
            if request.additional_actions:
                response.actions.extend(request.additional_actions)
            
            # Execute response
            execution_result = await soc.incident_responder.execute_response(response)
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "threat_response": {
                "threat_id": request.threat_id,
                "response_id": response.response_id,
                "execution_result": execution_result
            },
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "response_type": request.response_type
            }
        })
        
    except Exception as e:
        logger.error(f"Threat response execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threat response failed: {str(e)}")


@router.get("/forecast",
           summary="Get Predictive Threat Forecast",
           description="Get predictive threat forecast from the Autonomous SOC")
@rate_limit("soc_forecast", max_calls=10, window_seconds=60)
async def get_threat_forecast(
    horizon_hours: int = Query(48, description="Forecast horizon in hours"),
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Get predictive threat forecast"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Generate forecast
        horizon = timedelta(hours=horizon_hours)
        forecast = await soc.predictive_engine.forecast_threats(horizon)
        
        forecast_data = {
            "forecast_id": forecast.forecast_id,
            "prediction_horizon_hours": horizon_hours,
            "predicted_threats": forecast.predicted_threats,
            "confidence_intervals": forecast.confidence_intervals,
            "recommended_preparations": forecast.recommended_preparations,
            "forecast_timestamp": forecast.forecast_timestamp.isoformat(),
            "expires_at": forecast.expires_at.isoformat()
        }
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "threat_forecast": forecast_data,
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "horizon_hours": horizon_hours
            }
        })
        
    except Exception as e:
        logger.error(f"Threat forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threat forecast failed: {str(e)}")


@router.post("/configure",
            summary="Configure SOC Operations",
            description="Update Autonomous SOC configuration and operation mode")
@rate_limit("soc_configure", max_calls=5, window_seconds=60)
async def configure_soc(
    config: SOCConfigurationRequest,
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Configure Autonomous SOC operation parameters"""
    try:
        # Verify admin privileges
        if not current_user.get("roles", []) or "admin" not in current_user.get("roles", []):
            raise HTTPException(status_code=403, detail="Admin privileges required for SOC configuration")
        
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        configuration_changes = {}
        
        # Update operation mode
        if config.operation_mode:
            try:
                new_mode = OperationMode(config.operation_mode)
                old_mode = soc.operation_mode
                soc.operation_mode = new_mode
                configuration_changes["operation_mode"] = {
                    "old": old_mode.value,
                    "new": new_mode.value
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation mode: {config.operation_mode}")
        
        # Update other configuration parameters
        if config.auto_response_threshold is not None:
            configuration_changes["auto_response_threshold"] = config.auto_response_threshold
        
        if config.human_approval_required is not None:
            configuration_changes["human_approval_required"] = config.human_approval_required
        
        # Log configuration change
        await soc.audit_logger.log_event({
            "event_type": "soc_configuration_changed",
            "component": "autonomous_soc_api",
            "user": current_user.get("username", "unknown"),
            "changes": configuration_changes,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "configuration_changes": configuration_changes,
            "current_operation_mode": soc.operation_mode.value,
            "request_info": {
                "user": current_user.get("username", "unknown"),
                "admin_action": True
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SOC configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"SOC configuration failed: {str(e)}")


@router.get("/metrics",
           summary="Get SOC Performance Metrics",
           description="Get detailed performance metrics and analytics for the Autonomous SOC")
@rate_limit("soc_metrics", max_calls=30, window_seconds=60)
async def get_soc_metrics(
    time_range_hours: int = Query(24, description="Time range for metrics in hours"),
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """Get comprehensive SOC performance metrics"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Get comprehensive metrics
        metrics = await soc._get_detailed_metrics(time_range_hours)
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "time_range_hours": time_range_hours,
            "request_info": {
                "user": current_user.get("username", "unknown")
            }
        })
        
    except Exception as e:
        logger.error(f"SOC metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"SOC metrics failed: {str(e)}")


@router.get("/health",
           summary="SOC Health Check",
           description="Get health status of all Autonomous SOC components")
async def soc_health_check() -> JSONResponse:
    """Health check endpoint for the Autonomous SOC"""
    try:
        # Get SOC instance
        soc = await get_autonomous_soc()
        
        # Perform health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "autonomous_soc": "healthy",
                "unified_command_center": "healthy" if soc.unified_command_center else "unavailable",
                "incident_responder": "healthy",
                "predictive_engine": "healthy",
                "self_healing": "healthy"
            },
            "metrics_summary": {
                "active_threats": len(soc.active_threats),
                "active_responses": len(soc.active_responses),
                "operation_mode": soc.operation_mode.value
            }
        }
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"SOC health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


# Additional utility endpoints
@router.get("/operation-modes",
           summary="Get Available Operation Modes",
           description="Get list of available operation modes for the Autonomous SOC")
async def get_operation_modes() -> JSONResponse:
    """Get available SOC operation modes"""
    return JSONResponse(content={
        "operation_modes": [mode.value for mode in OperationMode],
        "descriptions": {
            "full_autonomous": "Complete automation with no human intervention",
            "human_supervised": "Autonomous operations with human oversight",
            "manual_approval": "Human approval required for all actions",
            "monitoring_only": "Monitoring and alerting only, no automated actions"
        }
    })


@router.get("/threat-severities",
           summary="Get Threat Severity Levels", 
           description="Get available threat severity levels used by the SOC")
async def get_threat_severities() -> JSONResponse:
    """Get available threat severity levels"""
    return JSONResponse(content={
        "severity_levels": [severity.value for severity in ThreatSeverity],
        "response_times": {
            "critical": "Immediate response required",
            "high": "Response within 1 minute",
            "medium": "Response within 5 minutes", 
            "low": "Response within 15 minutes",
            "info": "Monitoring only"
        }
    })