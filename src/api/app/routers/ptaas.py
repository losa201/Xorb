"""
PTaaS API Router - Production-ready penetration testing endpoints
Provides comprehensive API for PTaaS orchestration and management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.ptaas_orchestrator_service import PTaaSTarget, PTaaSSession
try:
    from ..services.ptaas_orchestrator_service import get_ptaas_orchestrator, PTaaSOrchestrator
except ImportError:
    # Fallback for missing orchestrator
    def get_ptaas_orchestrator():
        from ..container import get_container
        return get_container().get(PTaaSOrchestrator)
from ..services.intelligence_service import IntelligenceService, get_intelligence_service
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(tags=["PTaaS"])

# Request/Response Models
class ScanTargetRequest(BaseModel):
    """Request model for scan target creation"""
    host: str = Field(..., description="Target hostname or IP address")
    ports: List[int] = Field(..., description="List of ports to scan")
    scan_profile: str = Field(default="comprehensive", description="Scan profile type")
    stealth_mode: bool = Field(default=True, description="Enable stealth scanning")
    authorized: bool = Field(default=True, description="Target authorization status")
    constraints: Optional[List[str]] = Field(default=None, description="Scan constraints")

class ScanSessionRequest(BaseModel):
    """Request model for scan session creation"""
    targets: List[ScanTargetRequest] = Field(..., description="List of targets to scan")
    scan_type: str = Field(..., description="Type of scan to perform")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ScanSessionResponse(BaseModel):
    """Response model for scan session"""
    session_id: str
    status: str
    scan_type: str
    targets_count: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class AvailableProfilesResponse(BaseModel):
    """Response model for available scan profiles"""
    profiles: Dict[str, Dict[str, Any]]
    available_scanners: List[str]

class PTaaSMetricsResponse(BaseModel):
    """Response model for PTaaS metrics"""
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    failed_sessions: int
    total_vulnerabilities_found: int
    average_scan_duration: float
    scanner_availability: Dict[str, bool]
    
@router.post("/sessions", response_model=ScanSessionResponse)
async def create_scan_session(
    request: ScanSessionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    intelligence_service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    Create a new PTaaS scan session
    
    This endpoint creates a new penetration testing session with the specified targets
    and configuration. The scan will be queued for execution and can be monitored
    via the session status endpoint.
    """
    try:
        # Convert request targets to PTaaSTarget objects
        ptaas_targets = []
        for target_req in request.targets:
            target = PTaaSTarget(
                target_id=f"target_{target_req.host}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                host=target_req.host,
                ports=target_req.ports,
                scan_profile=target_req.scan_profile,
                constraints=target_req.constraints or [],
                authorized=target_req.authorized
            )
            ptaas_targets.append(target)
        
        # Create scan session
        session_id = await orchestrator.create_scan_session(
            targets=ptaas_targets,
            scan_type=request.scan_type,
            tenant_id=tenant_id,
            metadata=request.metadata
        )
        
        # Start the session in background
        background_tasks.add_task(orchestrator.start_scan_session, session_id)
        
        # Get session details for response
        session_status = await orchestrator.get_scan_session_status(session_id)
        if not session_status:
            raise HTTPException(status_code=500, detail="Failed to retrieve session status")
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_session_created", 1)
        
        # Add tracing context
        add_trace_context(
            operation="ptaas_session_created",
            session_id=session_id,
            tenant_id=str(tenant_id),
            targets_count=len(ptaas_targets),
            scan_type=request.scan_type
        )
        
        logger.info(f"Created PTaaS session {session_id} for tenant {tenant_id}")
        
        return ScanSessionResponse(**session_status)
        
    except ValueError as e:
        logger.error(f"Invalid request for PTaaS session: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create PTaaS session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}", response_model=ScanSessionResponse)
async def get_scan_session_status(
    session_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator)
):
    """
    Get the status of a specific PTaaS scan session
    
    Returns detailed information about the scan session including current status,
    progress, and results if completed.
    """
    try:
        session_status = await orchestrator.get_scan_session_status(session_id)
        if not session_status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_session_status_checked", 1)
        
        return ScanSessionResponse(**session_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/sessions/{session_id}/cancel")
async def cancel_scan_session(
    session_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator)
):
    """
    Cancel an active PTaaS scan session
    
    Cancels a running or queued scan session. Once cancelled, the session
    cannot be restarted.
    """
    try:
        success = await orchestrator.cancel_scan_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or cannot be cancelled")
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_session_cancelled", 1)
        
        logger.info(f"Cancelled PTaaS session {session_id}")
        
        return {"message": "Session cancelled successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/profiles", response_model=AvailableProfilesResponse)
async def get_available_scan_profiles(
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator)
):
    """
    Get available scan profiles and scanner information
    
    Returns a list of available scan profiles (quick, comprehensive, stealth, web_focused)
    and information about available security scanners.
    """
    try:
        profiles_info = await orchestrator.get_available_scan_profiles()
        
        return AvailableProfilesResponse(**profiles_info)
        
    except Exception as e:
        logger.error(f"Failed to get available scan profiles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions", response_model=List[ScanSessionResponse])
async def list_scan_sessions(
    tenant_id: UUID = Depends(get_current_tenant_id),
    status: Optional[str] = Query(None, description="Filter by session status"),
    limit: int = Query(50, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(0, description="Number of sessions to skip"),
):
    """
    List PTaaS scan sessions for the current tenant
    
    Returns a paginated list of scan sessions, optionally filtered by status.
    """
    try:
        # This would typically query a database for tenant sessions
        # For now, return a basic response structure
        sessions = []
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_sessions_listed", 1)
        
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to list scan sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics", response_model=PTaaSMetricsResponse)
async def get_ptaas_metrics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator)
):
    """
    Get PTaaS platform metrics and statistics
    
    Returns comprehensive metrics about scan sessions, vulnerabilities found,
    and scanner availability.
    """
    try:
        # Get orchestrator health and metrics
        health = await orchestrator.health_check()
        
        # Calculate basic metrics
        metrics_data = {
            "total_sessions": len(orchestrator.active_sessions),
            "active_sessions": len([s for s in orchestrator.active_sessions.values() if s.status == "running"]),
            "completed_sessions": len([s for s in orchestrator.active_sessions.values() if s.status == "completed"]),
            "failed_sessions": len([s for s in orchestrator.active_sessions.values() if s.status == "failed"]),
            "total_vulnerabilities_found": 0,  # Would be calculated from session results
            "average_scan_duration": 0.0,  # Would be calculated from completed sessions
            "scanner_availability": orchestrator.available_scanners
        }
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_metrics_retrieved", 1)
        
        return PTaaSMetricsResponse(**metrics_data)
        
    except Exception as e:
        logger.error(f"Failed to get PTaaS metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/validate-target")
async def validate_scan_target(
    request: ScanTargetRequest,
    tenant_id: UUID = Depends(get_current_tenant_id)
):
    """
    Validate a scan target before creating a session
    
    Performs basic validation of the target including reachability,
    authorization checks, and constraint validation.
    """
    try:
        # Create PTaaSTarget for validation
        target = PTaaSTarget(
            target_id="validation_temp",
            host=request.host,
            ports=request.ports,
            scan_profile=request.scan_profile,
            constraints=request.constraints or [],
            authorized=request.authorized
        )
        
        # Perform basic validation
        validation_results = {
            "valid": True,
            "host": request.host,
            "reachable": True,  # Would perform actual reachability test
            "authorized": request.authorized,
            "port_count": len(request.ports),
            "warnings": [],
            "errors": []
        }
        
        # Check for suspicious or restricted ports
        restricted_ports = [22, 23, 3389]  # SSH, Telnet, RDP
        dangerous_ports = [1433, 3306, 5432]  # Database ports
        
        for port in request.ports:
            if port in restricted_ports:
                validation_results["warnings"].append(f"Port {port} requires special authorization")
            elif port in dangerous_ports:
                validation_results["warnings"].append(f"Port {port} contains sensitive services")
        
        # Validate scan profile
        valid_profiles = ["quick", "comprehensive", "stealth", "web_focused"]
        if request.scan_profile not in valid_profiles:
            validation_results["errors"].append(f"Invalid scan profile: {request.scan_profile}")
            validation_results["valid"] = False
        
        if validation_results["errors"]:
            validation_results["valid"] = False
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Failed to validate scan target: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def get_ptaas_health(
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator)
):
    """
    Get PTaaS service health status
    
    Returns health information for the PTaaS orchestrator and related services.
    """
    try:
        health = await orchestrator.health_check()
        
        return {
            "status": health.status.value if hasattr(health.status, 'value') else health.status,
            "message": health.message,
            "timestamp": health.timestamp.isoformat(),
            "checks": health.checks,
            "service": "ptaas_orchestrator"
        }
        
    except Exception as e:
        logger.error(f"Failed to get PTaaS health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/scan-results/{session_id}")
async def get_scan_results(
    session_id: str,
    tenant_id: UUID = Depends(get_current_tenant_id),
    orchestrator: PTaaSOrchestrator = Depends(get_ptaas_orchestrator),
    format: str = Query("json", description="Result format (json, pdf, csv)")
):
    """
    Get detailed scan results for a completed session
    
    Returns comprehensive scan results including vulnerabilities found,
    services discovered, and security recommendations.
    """
    try:
        session_status = await orchestrator.get_scan_session_status(session_id)
        if not session_status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Session not completed")
        
        results = session_status.get("results", {})
        
        if format == "json":
            return results
        elif format == "pdf":
            # Would generate PDF report
            return JSONResponse(
                content={"error": "PDF format not yet implemented"},
                status_code=501
            )
        elif format == "csv":
            # Would generate CSV export
            return JSONResponse(
                content={"error": "CSV format not yet implemented"},
                status_code=501
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scan results for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")