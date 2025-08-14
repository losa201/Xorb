"""
PTaaS API Router - Production-ready penetration testing as a service
Comprehensive API endpoints for real-world security testing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..auth.models import UserClaims
from ..auth.dependencies import get_current_user, require_permissions
from ..services.ptaas_orchestrator_service import (
    get_ptaas_orchestrator,
    PTaaSOrchestrator,
    PTaaSTarget,
    PTaaSSession
)
from ..services.intelligence_service import get_intelligence_service
from ..infrastructure.observability import add_trace_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ptaas", tags=["PTaaS - Penetration Testing"])

# Request/Response Models

class ScanTargetRequest(BaseModel):
    """Scan target specification"""
    host: str = Field(..., description="Target hostname or IP address")
    ports: List[int] = Field(..., description="List of ports to scan", min_items=1, max_items=1000)
    scan_profile: str = Field(default="comprehensive", description="Scan profile to use")
    constraints: Optional[List[str]] = Field(default=None, description="Scanning constraints")
    authorized: bool = Field(default=True, description="Confirmation that target is authorized")

    @validator('host')
    def validate_host(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Host cannot be empty")
        # Basic validation - in production, implement more comprehensive checks
        import re
        if not re.match(r'^[a-zA-Z0-9.-]+$', v.replace(':', '')):
            raise ValueError("Invalid host format")
        return v.strip()

    @validator('ports')
    def validate_ports(cls, v):
        for port in v:
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port: {port}")
        return v

class ScanSessionRequest(BaseModel):
    """Scan session creation request"""
    targets: List[ScanTargetRequest] = Field(..., min_items=1, max_items=100)
    scan_type: str = Field(..., description="Type of scan to perform")
    description: Optional[str] = Field(default=None, description="Session description")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    @validator('scan_type')
    def validate_scan_type(cls, v):
        valid_types = ["quick", "comprehensive", "stealth", "web_focused"]
        if v not in valid_types:
            raise ValueError(f"Invalid scan type. Must be one of: {valid_types}")
        return v

class ScanSessionResponse(BaseModel):
    """Scan session response"""
    session_id: str
    status: str
    scan_type: str
    targets_count: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class VulnerabilityResponse(BaseModel):
    """Vulnerability finding response"""
    name: str
    severity: str
    description: str
    port: Optional[int] = None
    service: Optional[str] = None
    scanner: str
    remediation: Optional[List[str]] = None
    references: Optional[List[str]] = None

class ScanResultResponse(BaseModel):
    """Scan result summary response"""
    target_id: str
    vulnerabilities_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    services_discovered: List[Dict[str, Any]]
    scan_duration: float
    tools_used: List[str]

class SessionSummaryResponse(BaseModel):
    """Session summary response"""
    session_id: str
    targets_scanned: int
    total_vulnerabilities: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    medium_vulnerabilities: int
    low_vulnerabilities: int
    risk_level: str
    tools_used: List[str]
    duration_seconds: Optional[float] = None

# API Endpoints

@router.get("/profiles",
    summary="Get Available Scan Profiles",
    description="Retrieve available PTaaS scan profiles and configurations")
async def get_scan_profiles(
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """Get available scan profiles and scanner information"""
    try:
        add_trace_context(
            operation="ptaas_get_profiles",
            user_id=str(current_user.user_id),
            tenant_id=str(current_user.tenant_id)
        )

        profiles = await ptaas.get_available_scan_profiles()

        return {
            "success": True,
            "data": profiles,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get scan profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scan profiles: {str(e)}")

@router.post("/sessions",
    response_model=ScanSessionResponse,
    summary="Create PTaaS Scan Session",
    description="Create a new penetration testing scan session with specified targets")
async def create_scan_session(
    request: ScanSessionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> ScanSessionResponse:
    """Create new PTaaS scan session"""
    try:
        # Check permissions
        await require_permissions(current_user, ["ptaas:scan"])

        # Convert request targets to PTaaS targets
        targets = []
        for i, target_req in enumerate(request.targets):
            target = PTaaSTarget(
                target_id=f"target_{i}",
                host=target_req.host,
                ports=target_req.ports,
                scan_profile=target_req.scan_profile,
                constraints=target_req.constraints or [],
                authorized=target_req.authorized
            )
            targets.append(target)

        # Create session
        session_id = await ptaas.create_scan_session(
            targets=targets,
            scan_type=request.scan_type,
            tenant_id=current_user.tenant_id,
            metadata={
                "description": request.description,
                "created_by": str(current_user.user_id),
                "user_metadata": request.metadata
            }
        )

        # Start session in background
        background_tasks.add_task(ptaas.start_scan_session, session_id)

        add_trace_context(
            operation="ptaas_session_created",
            session_id=session_id,
            user_id=str(current_user.user_id),
            tenant_id=str(current_user.tenant_id),
            targets_count=len(targets)
        )

        return ScanSessionResponse(
            session_id=session_id,
            status="created",
            scan_type=request.scan_type,
            targets_count=len(targets),
            created_at=datetime.utcnow().isoformat()
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scan session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create scan session: {str(e)}")

@router.get("/sessions/{session_id}",
    response_model=ScanSessionResponse,
    summary="Get Scan Session Status",
    description="Retrieve status and results of a PTaaS scan session")
async def get_scan_session(
    session_id: str,
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> ScanSessionResponse:
    """Get scan session status and results"""
    try:
        session_data = await ptaas.get_scan_session_status(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Scan session not found")

        # Verify tenant access
        # Note: In production, implement proper tenant isolation check

        return ScanSessionResponse(**session_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scan session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scan session: {str(e)}")

@router.post("/sessions/{session_id}/cancel",
    summary="Cancel Scan Session",
    description="Cancel an active PTaaS scan session")
async def cancel_scan_session(
    session_id: str,
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """Cancel active scan session"""
    try:
        # Check permissions
        await require_permissions(current_user, ["ptaas:cancel"])

        success = await ptaas.cancel_scan_session(session_id)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel session")

        add_trace_context(
            operation="ptaas_session_cancelled",
            session_id=session_id,
            user_id=str(current_user.user_id)
        )

        return {
            "success": True,
            "message": f"Scan session {session_id} cancelled",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel scan session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel scan session: {str(e)}")

@router.get("/sessions/{session_id}/results",
    summary="Get Detailed Scan Results",
    description="Retrieve detailed vulnerability findings and scan results")
async def get_scan_results(
    session_id: str,
    include_raw: bool = Query(False, description="Include raw scanner output"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity (critical,high,medium,low)"),
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """Get detailed scan results with optional filtering"""
    try:
        session_data = await ptaas.get_scan_session_status(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Scan session not found")

        if session_data["status"] not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Scan session not completed")

        results = session_data.get("results", {})
        scan_results = results.get("scan_results", [])

        # Process and filter results
        processed_results = []
        for result in scan_results:
            if "error" in result:
                processed_results.append(result)
                continue

            # Filter vulnerabilities by severity if requested
            vulnerabilities = result.get("raw_data", {}).get("vulnerabilities", [])
            if severity_filter:
                severity_levels = [s.strip().lower() for s in severity_filter.split(",")]
                vulnerabilities = [v for v in vulnerabilities if v.get("severity", "").lower() in severity_levels]

            processed_result = {
                "target_id": result.get("target_id"),
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities if not include_raw else vulnerabilities,
                "services_discovered": result.get("services_discovered", []),
                "scan_duration": result.get("scan_duration", 0),
                "tools_used": result.get("tools_used", [])
            }

            # Include raw data if requested
            if include_raw:
                processed_result["raw_data"] = result.get("raw_data", {})

            processed_results.append(processed_result)

        return {
            "success": True,
            "session_id": session_id,
            "results": processed_results,
            "summary": results.get("summary", {}),
            "filters_applied": {
                "severity_filter": severity_filter,
                "include_raw": include_raw
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scan results for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scan results: {str(e)}")

@router.get("/sessions/{session_id}/report",
    summary="Generate Scan Report",
    description="Generate comprehensive security report for scan session")
async def generate_scan_report(
    session_id: str,
    report_format: str = Query("json", description="Report format (json, pdf, html)"),
    include_executive_summary: bool = Query(True, description="Include executive summary"),
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """Generate comprehensive scan report"""
    try:
        # Check permissions
        await require_permissions(current_user, ["ptaas:report"])

        session_data = await ptaas.get_scan_session_status(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Scan session not found")

        if session_data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Scan session not completed")

        # Generate report based on format
        if report_format.lower() == "json":
            report = await _generate_json_report(session_data, include_executive_summary)
        elif report_format.lower() == "html":
            report = await _generate_html_report(session_data, include_executive_summary)
        elif report_format.lower() == "pdf":
            # PDF generation would require additional libraries
            raise HTTPException(status_code=501, detail="PDF format not yet implemented")
        else:
            raise HTTPException(status_code=400, detail="Invalid report format")

        add_trace_context(
            operation="ptaas_report_generated",
            session_id=session_id,
            report_format=report_format,
            user_id=str(current_user.user_id)
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/sessions",
    summary="List Scan Sessions",
    description="List PTaaS scan sessions for the current tenant")
async def list_scan_sessions(
    limit: int = Query(50, description="Maximum number of sessions to return", le=100),
    offset: int = Query(0, description="Number of sessions to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """List scan sessions for tenant"""
    try:
        # In production, this would query the database for sessions
        # For now, return active sessions from memory
        sessions = []
        for session_id, session in ptaas.active_sessions.items():
            if session.tenant_id != current_user.tenant_id:
                continue

            if status_filter and session.status != status_filter:
                continue

            session_info = {
                "session_id": session_id,
                "status": session.status,
                "scan_type": session.scan_type,
                "targets_count": len(session.targets),
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None
            }

            if session.results:
                session_info["summary"] = session.results.get("summary", {})

            sessions.append(session_info)

        # Apply pagination
        total_sessions = len(sessions)
        sessions = sessions[offset:offset + limit]

        return {
            "success": True,
            "sessions": sessions,
            "pagination": {
                "total": total_sessions,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_sessions
            },
            "filters": {
                "status_filter": status_filter
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to list scan sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list scan sessions: {str(e)}")

@router.get("/health",
    summary="PTaaS Service Health",
    description="Get PTaaS service health and status information")
async def get_ptaas_health(
    current_user: UserClaims = Depends(get_current_user),
    ptaas: PTaaSOrchestrator = Depends(lambda: get_ptaas_orchestrator(get_intelligence_service()))
) -> Dict[str, Any]:
    """Get PTaaS service health"""
    try:
        health = await ptaas.health_check()

        return {
            "success": True,
            "status": health.status.value if hasattr(health.status, 'value') else str(health.status),
            "message": health.message,
            "checks": health.checks,
            "timestamp": health.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"PTaaS health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

# Helper functions for report generation

async def _generate_json_report(session_data: Dict[str, Any], include_executive_summary: bool) -> Dict[str, Any]:
    """Generate JSON format report"""
    results = session_data.get("results", {})
    summary = results.get("summary", {})

    report = {
        "report_type": "ptaas_scan_report",
        "format": "json",
        "generated_at": datetime.utcnow().isoformat(),
        "session_info": {
            "session_id": session_data["session_id"],
            "scan_type": session_data["scan_type"],
            "targets_count": session_data["targets_count"],
            "created_at": session_data["created_at"],
            "completed_at": session_data["completed_at"]
        },
        "scan_summary": summary,
        "detailed_results": results.get("scan_results", [])
    }

    if include_executive_summary:
        report["executive_summary"] = _generate_executive_summary(summary)

    return report

async def _generate_html_report(session_data: Dict[str, Any], include_executive_summary: bool) -> Dict[str, Any]:
    """Generate HTML format report"""
    # HTML report generation would be implemented here
    # For now, return a simple structure
    return {
        "report_type": "ptaas_scan_report",
        "format": "html",
        "generated_at": datetime.utcnow().isoformat(),
        "content": "<html><body><h1>PTaaS Scan Report</h1><p>HTML report generation not yet implemented</p></body></html>",
        "session_id": session_data["session_id"]
    }

def _generate_executive_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary"""
    total_vulns = summary.get("total_vulnerabilities", 0)
    critical_vulns = summary.get("critical_vulnerabilities", 0)
    high_vulns = summary.get("high_vulnerabilities", 0)
    risk_level = summary.get("risk_level", "unknown")

    # Generate risk assessment
    if critical_vulns > 0:
        risk_assessment = "CRITICAL - Immediate action required"
        priority = "URGENT"
    elif high_vulns > 0:
        risk_assessment = "HIGH - Action required within 24 hours"
        priority = "HIGH"
    elif total_vulns > 0:
        risk_assessment = "MEDIUM - Address within 1 week"
        priority = "MEDIUM"
    else:
        risk_assessment = "LOW - Continue monitoring"
        priority = "LOW"

    # Generate recommendations
    recommendations = []
    if critical_vulns > 0:
        recommendations.append("Immediately patch critical vulnerabilities")
        recommendations.append("Consider isolating affected systems")
    if high_vulns > 0:
        recommendations.append("Prioritize high-severity vulnerability remediation")
    if total_vulns > 5:
        recommendations.append("Implement regular vulnerability scanning")
    recommendations.append("Review and update security policies")

    return {
        "risk_level": risk_level,
        "risk_assessment": risk_assessment,
        "priority": priority,
        "total_vulnerabilities": total_vulns,
        "critical_findings": critical_vulns,
        "high_findings": high_vulns,
        "key_recommendations": recommendations[:5],  # Top 5 recommendations
        "business_impact": _assess_business_impact(critical_vulns, high_vulns, total_vulns)
    }

def _assess_business_impact(critical: int, high: int, total: int) -> str:
    """Assess business impact of findings"""
    if critical > 0:
        return "HIGH - Critical vulnerabilities pose immediate risk to business operations and data security"
    elif high > 0:
        return "MEDIUM - High-severity vulnerabilities could lead to security incidents if exploited"
    elif total > 0:
        return "LOW - Vulnerabilities present manageable risk with proper monitoring"
    else:
        return "MINIMAL - No significant vulnerabilities detected"
