#!/usr/bin/env python3
"""
Enhanced PTaaS Router - Production-Ready Implementation
Principal Auditor Implementation: Comprehensive penetration testing and quantum-safe security

This router provides advanced PTaaS capabilities with quantum-safe security integration
and comprehensive compliance validation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from enum import Enum

# Internal imports
from ..services.quantum_integrated_ptaas_service import (
    QuantumIntegratedPTaaSService,
    QuantumSafeScanConfiguration,
    QuantumEnhancedScanResult
)
from ..services.enhanced_production_service_implementations import (
    EnhancedProductionPTaaSService,
    ComplianceFramework,
    ScanTarget,
    VulnerabilityFinding
)
from ...xorb.security.quantum_safe_security_engine import (
    PostQuantumAlgorithm,
    CryptographicMode,
    QuantumThreatLevel
)
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ptaas", tags=["Enhanced PTaaS"])


# Enums for API
class ScanTypeEnum(str, Enum):
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    STEALTH = "stealth"
    WEB_FOCUSED = "web_focused"
    QUANTUM_ASSESSMENT = "quantum_assessment"
    QUANTUM_SAFE_COMPLIANCE = "quantum_safe_compliance"
    FUTURE_PROOF_SECURITY = "future_proof_security"


class ComplianceFrameworkEnum(str, Enum):
    PCI_DSS = "PCI-DSS"
    HIPAA = "HIPAA"
    SOX = "SOX"
    ISO_27001 = "ISO-27001"
    GDPR = "GDPR"
    NIST = "NIST"


class ReportFormatEnum(str, Enum):
    JSON = "json"
    XML = "xml"
    HTML = "html"
    PDF = "pdf"


# Request Models
class ScanTargetRequest(BaseModel):
    """Enhanced scan target request"""
    host: str = Field(..., description="Target host or IP address")
    ports: List[int] = Field(default=[80, 443, 22], description="Ports to scan")
    scan_profile: Optional[str] = Field(default=None, description="Specific scan profile for this target")
    protocols: List[str] = Field(default=["tcp"], description="Protocols to test")
    authentication: Optional[Dict[str, Any]] = Field(default=None, description="Authentication credentials")
    compliance_requirements: List[str] = Field(default=[], description="Compliance requirements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional target metadata")

    @validator('host')
    def validate_host(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Host cannot be empty")
        return v.strip()

    @validator('ports')
    def validate_ports(cls, v):
        if not v:
            return [80, 443, 22]
        for port in v:
            if not 1 <= port <= 65535:
                raise ValueError(f"Port {port} is not valid (must be 1-65535)")
        return v


class QuantumConfigRequest(BaseModel):
    """Quantum-safe scan configuration request"""
    enable_quantum_safe: bool = Field(default=True, description="Enable quantum-safe analysis")
    post_quantum_algorithms: List[str] = Field(
        default=["kyber768", "dilithium3"], 
        description="Post-quantum algorithms to use"
    )
    cryptographic_mode: str = Field(default="hybrid", description="Cryptographic mode")
    key_rotation_interval: int = Field(default=24, description="Key rotation interval in hours")
    quantum_channel_validation: bool = Field(default=True, description="Enable quantum channel validation")
    threat_assessment_enabled: bool = Field(default=True, description="Enable threat assessment")
    compliance_quantum_requirements: List[str] = Field(
        default=["post_quantum_ready"], 
        description="Quantum compliance requirements"
    )


class PTaaSScanRequest(BaseModel):
    """Comprehensive PTaaS scan request"""
    targets: List[ScanTargetRequest] = Field(..., description="Targets to scan")
    scan_type: ScanTypeEnum = Field(..., description="Type of scan to perform")
    quantum_config: Optional[QuantumConfigRequest] = Field(default=None, description="Quantum-safe configuration")
    compliance_frameworks: List[ComplianceFrameworkEnum] = Field(
        default=[], 
        description="Compliance frameworks to validate"
    )
    scan_options: Dict[str, Any] = Field(default_factory=dict, description="Additional scan options")
    notification_settings: Dict[str, Any] = Field(default_factory=dict, description="Notification preferences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Scan metadata")

    @validator('targets')
    def validate_targets(cls, v):
        if not v:
            raise ValueError("At least one target must be specified")
        if len(v) > 50:
            raise ValueError("Maximum 50 targets allowed per scan")
        return v


class ComplianceScanRequest(BaseModel):
    """Compliance-specific scan request"""
    targets: List[str] = Field(..., description="Target hosts for compliance scanning")
    compliance_framework: ComplianceFrameworkEnum = Field(..., description="Compliance framework")
    include_quantum_assessment: bool = Field(default=True, description="Include quantum threat assessment")
    detailed_reporting: bool = Field(default=True, description="Generate detailed compliance report")
    remediation_guidance: bool = Field(default=True, description="Include remediation guidance")


class QuantumThreatAssessmentRequest(BaseModel):
    """Quantum threat assessment request"""
    targets: List[str] = Field(..., description="Targets for quantum threat assessment")
    assessment_depth: str = Field(default="comprehensive", description="Assessment depth")
    include_migration_roadmap: bool = Field(default=True, description="Include migration roadmap")
    compliance_impact_analysis: bool = Field(default=True, description="Analyze compliance impact")


# Response Models
class ScanSessionResponse(BaseModel):
    """Scan session creation response"""
    session_id: str
    status: str
    scan_type: str
    targets_count: int
    estimated_duration_minutes: int
    tools: List[str]
    quantum_enhanced: bool = False
    quantum_algorithms: List[str] = []
    cryptographic_mode: Optional[str] = None
    compliance_coverage: List[str] = []
    risk_level: str
    created_at: str


class ScanStatusResponse(BaseModel):
    """Scan status response"""
    session_id: str
    status: str
    progress: int
    phase: str
    scan_type: str
    targets_count: int
    created_at: str
    started_at: Optional[str] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[float] = None
    vulnerabilities_found: int = 0
    current_target: Optional[str] = None
    tools_status: Dict[str, str] = {}
    live_findings: List[Dict[str, Any]] = []
    quantum_features: Dict[str, Any] = {}


class ScanResultsResponse(BaseModel):
    """Comprehensive scan results response"""
    session_id: str
    scan_metadata: Dict[str, Any]
    summary: Dict[str, Any]
    vulnerabilities: List[Dict[str, Any]]
    services: List[Dict[str, Any]]
    compliance_status: Dict[str, Any] = {}
    quantum_assessment: Optional[Dict[str, Any]] = None
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    executive_summary: Dict[str, Any] = {}
    technical_details: Dict[str, Any] = {}


class ScanProfileResponse(BaseModel):
    """Scan profile information response"""
    id: str
    name: str
    description: str
    duration_minutes: int
    tools: List[str]
    risk_level: str
    compliance_coverage: List[str]
    capabilities: Dict[str, bool]
    quantum_features: Dict[str, bool] = {}
    output_formats: List[str]
    recommended_for: List[str]


class QuantumThreatAssessmentResponse(BaseModel):
    """Quantum threat assessment response"""
    assessment_id: str
    targets_assessed: int
    assessment_timestamp: str
    overall_quantum_readiness: Dict[str, Any]
    threat_assessments: List[Dict[str, Any]]
    migration_roadmap: Dict[str, Any]
    executive_summary: Dict[str, Any]
    recommendations: List[str]
    compliance_impact: Dict[str, Any]


# Dependency injection
async def get_ptaas_service() -> QuantumIntegratedPTaaSService:
    """Get PTaaS service instance"""
    service = QuantumIntegratedPTaaSService({})
    await service.initialize()
    return service


# API Endpoints
@router.post("/scans", response_model=ScanSessionResponse)
async def create_scan_session(
    request: PTaaSScanRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Create comprehensive PTaaS scan session with quantum-safe security analysis
    
    This endpoint creates advanced penetration testing sessions with optional
    quantum-safe security assessment and compliance validation.
    """
    try:
        # Convert request to internal format
        targets = [
            {
                "host": target.host,
                "ports": target.ports,
                "scan_profile": target.scan_profile or request.scan_type.value,
                "protocols": target.protocols,
                "authentication": target.authentication,
                "compliance_requirements": target.compliance_requirements,
                "metadata": target.metadata
            }
            for target in request.targets
        ]
        
        # Convert quantum configuration if provided
        quantum_config = None
        if request.quantum_config:
            from ..services.quantum_integrated_ptaas_service import QuantumSafeScanConfiguration
            quantum_config = QuantumSafeScanConfiguration(
                enable_quantum_safe=request.quantum_config.enable_quantum_safe,
                post_quantum_algorithms=[
                    PostQuantumAlgorithm(alg) for alg in request.quantum_config.post_quantum_algorithms
                ],
                cryptographic_mode=CryptographicMode(request.quantum_config.cryptographic_mode),
                key_rotation_interval=request.quantum_config.key_rotation_interval,
                quantum_channel_validation=request.quantum_config.quantum_channel_validation,
                threat_assessment_enabled=request.quantum_config.threat_assessment_enabled,
                compliance_quantum_requirements=request.quantum_config.compliance_quantum_requirements
            )
        
        # Add compliance frameworks to metadata
        enhanced_metadata = request.metadata.copy()
        if request.compliance_frameworks:
            enhanced_metadata["compliance_frameworks"] = [cf.value for cf in request.compliance_frameworks]
        
        # Create scan session
        if quantum_config or request.scan_type in [ScanTypeEnum.QUANTUM_ASSESSMENT, ScanTypeEnum.QUANTUM_SAFE_COMPLIANCE, ScanTypeEnum.FUTURE_PROOF_SECURITY]:
            scan_result = await ptaas_service.create_quantum_enhanced_scan(
                targets=targets,
                scan_type=request.scan_type.value,
                user={"id": str(tenant_id)},
                org={"id": str(tenant_id)},
                quantum_config=quantum_config,
                metadata=enhanced_metadata
            )
        else:
            scan_result = await ptaas_service.create_scan_session(
                targets=targets,
                scan_type=request.scan_type.value,
                user={"id": str(tenant_id)},
                org={"id": str(tenant_id)},
                metadata=enhanced_metadata
            )
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_api_request("ptaas_scan_created", 1)
        metrics.record_custom_metric("ptaas_targets_scanned", len(targets))
        
        # Add tracing context
        add_trace_context(
            operation="ptaas_scan_creation",
            session_id=scan_result["session_id"],
            tenant_id=str(tenant_id),
            scan_type=request.scan_type.value,
            targets_count=len(targets),
            quantum_enhanced=scan_result.get("quantum_enhanced", False)
        )
        
        logger.info(f"Created PTaaS scan session: {scan_result['session_id']}")
        
        return ScanSessionResponse(
            session_id=scan_result["session_id"],
            status=scan_result["status"],
            scan_type=scan_result["scan_type"],
            targets_count=scan_result["targets_count"],
            estimated_duration_minutes=scan_result["estimated_duration_minutes"],
            tools=scan_result["tools"],
            quantum_enhanced=scan_result.get("quantum_enhanced", False),
            quantum_algorithms=scan_result.get("quantum_algorithms", []),
            cryptographic_mode=scan_result.get("cryptographic_mode"),
            compliance_coverage=scan_result.get("compliance_coverage", []),
            risk_level=scan_result["risk_level"],
            created_at=scan_result["created_at"]
        )
        
    except ValueError as e:
        logger.error(f"Invalid scan request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scan session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/scans/{session_id}", response_model=ScanStatusResponse)
async def get_scan_status(
    session_id: str = Path(..., description="Scan session ID"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Get comprehensive scan session status with real-time progress updates
    
    Returns detailed information about scan progress, current phase,
    live findings, and quantum security analysis status.
    """
    try:
        status = await ptaas_service.get_scan_status(session_id, {"id": str(tenant_id)})
        
        return ScanStatusResponse(
            session_id=status["session_id"],
            status=status["status"],
            progress=status["progress"],
            phase=status["phase"],
            scan_type=status["scan_type"],
            targets_count=status["targets_count"],
            created_at=status["created_at"],
            started_at=status.get("started_at"),
            estimated_duration=status.get("estimated_duration"),
            actual_duration=status.get("actual_duration"),
            vulnerabilities_found=status["vulnerabilities_found"],
            current_target=status.get("current_target"),
            tools_status=status.get("tools_status", {}),
            live_findings=status.get("live_findings", []),
            quantum_features=status.get("quantum_features", {})
        )
        
    except ValueError as e:
        logger.error(f"Scan session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get scan status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/scans/{session_id}/results", response_model=ScanResultsResponse)
async def get_scan_results(
    session_id: str = Path(..., description="Scan session ID"),
    format: ReportFormatEnum = Query(ReportFormatEnum.JSON, description="Results format"),
    include_raw_data: bool = Query(False, description="Include raw scanner output"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Get comprehensive scan results with detailed vulnerability analysis
    
    Returns complete scan results including vulnerabilities, services,
    compliance status, quantum assessment, and actionable recommendations.
    """
    try:
        results = await ptaas_service.get_scan_results(session_id, {"id": str(tenant_id)})
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        # Handle different output formats
        if format == ReportFormatEnum.JSON:
            return ScanResultsResponse(
                session_id=results["session_id"],
                scan_metadata=results["scan_metadata"],
                summary=results["summary"],
                vulnerabilities=results["vulnerabilities"],
                services=results["services"],
                compliance_status=results.get("compliance_status", {}),
                quantum_assessment=results.get("quantum_assessment"),
                risk_assessment=results["risk_assessment"],
                recommendations=results["recommendations"],
                executive_summary=results.get("executive_summary", {}),
                technical_details=results.get("technical_details", {})
            )
        elif format in [ReportFormatEnum.HTML, ReportFormatEnum.PDF, ReportFormatEnum.XML]:
            # Generate formatted report
            report_file = await _generate_formatted_report(results, format.value)
            return FileResponse(
                path=report_file,
                filename=f"ptaas_scan_{session_id}.{format.value}",
                media_type=_get_media_type(format.value)
            )
        
    except ValueError as e:
        logger.error(f"Scan results not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get scan results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/scans/{session_id}")
async def cancel_scan(
    session_id: str = Path(..., description="Scan session ID"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Cancel an active scan session
    
    Safely terminates a running scan and preserves partial results.
    """
    try:
        success = await ptaas_service.cancel_scan(session_id, {"id": str(tenant_id)})
        
        if not success:
            raise HTTPException(status_code=404, detail="Scan session not found or already completed")
        
        return {"message": "Scan cancelled successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/profiles", response_model=List[ScanProfileResponse])
async def get_scan_profiles(
    include_quantum: bool = Query(True, description="Include quantum-enhanced profiles"),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Get available scan profiles with detailed capabilities
    
    Returns comprehensive information about available scan profiles,
    including quantum-enhanced options and compliance coverage.
    """
    try:
        profiles = await ptaas_service.get_available_scan_profiles()
        
        response_profiles = []
        for profile in profiles:
            # Filter quantum profiles if not requested
            if not include_quantum and "quantum" in profile["id"]:
                continue
                
            response_profiles.append(ScanProfileResponse(
                id=profile["id"],
                name=profile["name"],
                description=profile["description"],
                duration_minutes=profile["duration_minutes"],
                tools=profile["tools"],
                risk_level=profile.get("risk_level", "medium"),
                compliance_coverage=profile.get("compliance_coverage", []),
                capabilities=profile["capabilities"],
                quantum_features=profile.get("quantum_features", {}),
                output_formats=profile["output_formats"],
                recommended_for=profile["recommended_for"]
            ))
        
        return response_profiles
        
    except Exception as e:
        logger.error(f"Failed to get scan profiles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/compliance-scan", response_model=ScanSessionResponse)
async def create_compliance_scan(
    request: ComplianceScanRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Create compliance-specific scan with framework validation
    
    Performs specialized scans tailored to specific compliance frameworks
    with comprehensive reporting and remediation guidance.
    """
    try:
        scan_result = await ptaas_service.create_compliance_scan(
            targets=request.targets,
            compliance_framework=request.compliance_framework.value,
            user={"id": str(tenant_id)},
            org={"id": str(tenant_id)}
        )
        
        logger.info(f"Created compliance scan for {request.compliance_framework.value}: {scan_result['session_id']}")
        
        return ScanSessionResponse(
            session_id=scan_result["session_id"],
            status=scan_result["status"],
            scan_type=scan_result["scan_type"],
            targets_count=scan_result["targets_count"],
            estimated_duration_minutes=scan_result["estimated_duration_minutes"],
            tools=scan_result["tools"],
            compliance_coverage=scan_result.get("compliance_requirements", []),
            risk_level=scan_result["risk_level"],
            created_at=scan_result["created_at"]
        )
        
    except ValueError as e:
        logger.error(f"Invalid compliance scan request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create compliance scan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quantum-threat-assessment", response_model=QuantumThreatAssessmentResponse)
async def conduct_quantum_threat_assessment(
    request: QuantumThreatAssessmentRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Conduct comprehensive quantum threat assessment
    
    Performs detailed analysis of quantum threats, post-quantum readiness,
    and generates migration roadmaps for quantum-safe security.
    """
    try:
        assessment_result = await ptaas_service.conduct_quantum_threat_assessment(
            targets=request.targets,
            user={"id": str(tenant_id)},
            org={"id": str(tenant_id)}
        )
        
        logger.info(f"Completed quantum threat assessment: {assessment_result['assessment_id']}")
        
        return QuantumThreatAssessmentResponse(
            assessment_id=assessment_result["assessment_id"],
            targets_assessed=assessment_result["targets_assessed"],
            assessment_timestamp=assessment_result["assessment_timestamp"],
            overall_quantum_readiness=assessment_result["overall_quantum_readiness"],
            threat_assessments=assessment_result["threat_assessments"],
            migration_roadmap=assessment_result["migration_roadmap"],
            executive_summary=assessment_result["executive_summary"],
            recommendations=assessment_result["recommendations"],
            compliance_impact=assessment_result["compliance_impact"]
        )
        
    except Exception as e:
        logger.error(f"Quantum threat assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quantum-channels/{session_id}")
async def establish_quantum_channel(
    session_id: str = Path(..., description="Scan session ID"),
    participants: List[str] = Body(..., description="Channel participants"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Establish quantum-safe communication channel for secure scanning
    
    Creates quantum-safe communication channels between scan participants
    for enhanced security during penetration testing operations.
    """
    try:
        channel_result = await ptaas_service.establish_quantum_safe_scanning_channel(
            session_id=session_id,
            participants=participants
        )
        
        return {
            "message": "Quantum-safe channels established successfully",
            "session_id": session_id,
            "channels_established": channel_result["quantum_channels_established"],
            "security_verified": channel_result["security_verified"],
            "participants": channel_result["participants"]
        }
        
    except Exception as e:
        logger.error(f"Failed to establish quantum channel: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def get_ptaas_health(
    ptaas_service: QuantumIntegratedPTaaSService = Depends(get_ptaas_service)
):
    """
    Get PTaaS service health status
    
    Returns health information for the PTaaS service including
    quantum security engine status and scan capability availability.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "ptaas_engine": "healthy",
                "quantum_security": "healthy" if ptaas_service.quantum_engine else "unavailable",
                "ai_orchestrator": "healthy" if ptaas_service.ai_orchestrator else "unavailable"
            },
            "capabilities": {
                "standard_scanning": True,
                "quantum_enhanced_scanning": bool(ptaas_service.quantum_engine),
                "ai_orchestration": bool(ptaas_service.ai_orchestrator),
                "compliance_validation": True
            },
            "scan_profiles_available": len(ptaas_service.scan_profiles),
            "active_scans": len(ptaas_service.active_scans),
            "completed_scans": len(ptaas_service.scan_results)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get PTaaS health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper functions
async def _generate_formatted_report(results: Dict[str, Any], format: str) -> str:
    """Generate formatted report file"""
    # Mock implementation - would generate actual reports
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as f:
        if format == 'json':
            json.dump(results, f, indent=2)
        elif format == 'html':
            f.write(f"<html><body><h1>PTaaS Scan Report</h1><pre>{json.dumps(results, indent=2)}</pre></body></html>")
        elif format == 'xml':
            f.write(f"<?xml version='1.0'?><report>{json.dumps(results)}</report>")
        return f.name


def _get_media_type(format: str) -> str:
    """Get media type for format"""
    media_types = {
        'json': 'application/json',
        'html': 'text/html',
        'xml': 'application/xml',
        'pdf': 'application/pdf'
    }
    return media_types.get(format, 'application/octet-stream')


# Export router
__all__ = ["router"]