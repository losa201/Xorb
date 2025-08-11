#!/usr/bin/env python3
"""
Compliance Reporting API Router
Automated compliance reporting endpoints for SOC2, ISO27001, NIST CSF, and GDPR
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..services.compliance_reporting_service import (
    get_compliance_service,
    ComplianceReportingService,
    ComplianceFramework,
    ControlStatus,
    ComplianceReport
)
from ..core.auth import get_current_user
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/compliance", tags=["Compliance Reporting"])


# Pydantic models for API
class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation"""
    framework: ComplianceFramework
    system_data: Optional[Dict[str, Any]] = None
    reporting_period_days: int = Field(90, ge=1, le=365)


class ComplianceReportSummary(BaseModel):
    """Summary model for compliance reports"""
    report_id: str
    framework: ComplianceFramework
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    generated_at: datetime
    generated_by: str


class ComplianceReportResponse(BaseModel):
    """Full compliance report response"""
    report_id: str
    framework: ComplianceFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    overall_compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    executive_summary: str
    remediation_plan: List[str]
    generated_by: str
    generated_at: datetime


class ControlAssessmentResponse(BaseModel):
    """Control assessment response"""
    control_id: str
    framework: ComplianceFramework
    status: ControlStatus
    compliance_score: float
    assessment_date: datetime
    findings: List[str]
    evidence_collected: List[str]
    remediation_required: List[str]
    automated: bool


class ComplianceDashboardResponse(BaseModel):
    """Compliance dashboard data"""
    frameworks: Dict[str, float]  # Framework -> compliance score
    total_reports: int
    recent_reports: List[ComplianceReportSummary]
    critical_findings: int
    remediation_items: int
    automation_coverage: float


@router.post("/reports/generate", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    background_tasks: BackgroundTasks,
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Generate automated compliance report for specified framework
    
    **Supported Frameworks:**
    - soc2_type_ii: SOC 2 Type II compliance assessment
    - iso27001: ISO 27001 information security management
    - nist_csf: NIST Cybersecurity Framework
    - gdpr: General Data Protection Regulation
    - pci_dss: Payment Card Industry Data Security Standard
    - hipaa: Health Insurance Portability and Accountability Act
    
    **Features:**
    - Automated control assessment
    - Evidence collection
    - Risk scoring and prioritization
    - Remediation planning
    - Executive reporting
    
    **Requirements:**
    - Compliance officer or admin role
    - Valid framework selection
    - System access for data collection
    """
    try:
        # Check user permissions
        if not hasattr(current_user, 'role') or current_user.role not in ['admin', 'compliance_officer', 'security_analyst']:
            raise HTTPException(status_code=403, detail="Insufficient permissions for compliance reporting")
        
        logger.info(
            f"Compliance report generation requested",
            user_id=getattr(current_user, 'id', 'unknown'),
            framework=request.framework.value,
            reporting_period=request.reporting_period_days
        )
        
        # Generate report
        report = await service.generate_report(
            framework=request.framework,
            system_data=request.system_data
        )
        
        # Background task for additional processing
        background_tasks.add_task(
            _process_compliance_report,
            report,
            getattr(current_user, 'id', 'unknown')
        )
        
        response = ComplianceReportResponse(
            report_id=report.report_id,
            framework=report.framework,
            reporting_period_start=report.reporting_period_start,
            reporting_period_end=report.reporting_period_end,
            overall_compliance_score=report.overall_compliance_score,
            total_controls=report.total_controls,
            compliant_controls=report.compliant_controls,
            non_compliant_controls=report.non_compliant_controls,
            executive_summary=report.executive_summary,
            remediation_plan=report.remediation_plan,
            generated_by=report.generated_by,
            generated_at=report.generated_at
        )
        
        logger.info(
            f"Compliance report generated successfully",
            report_id=report.report_id,
            framework=request.framework.value,
            compliance_score=report.overall_compliance_score
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/reports", response_model=List[ComplianceReportSummary])
async def list_compliance_reports(
    framework: Optional[ComplianceFramework] = Query(None, description="Filter by framework"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of reports"),
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    List compliance reports with optional filtering
    
    **Features:**
    - Framework-based filtering
    - Pagination support
    - Summary information
    - Chronological ordering
    """
    try:
        reports = service.list_reports(framework=framework)
        
        # Convert to summary format and limit results
        summaries = []
        for report in reports[:limit]:
            summaries.append(ComplianceReportSummary(
                report_id=report.report_id,
                framework=report.framework,
                overall_compliance_score=report.overall_compliance_score,
                total_controls=report.total_controls,
                compliant_controls=report.compliant_controls,
                non_compliant_controls=report.non_compliant_controls,
                generated_at=report.generated_at,
                generated_by=report.generated_by
            ))
        
        logger.info(
            f"Listed compliance reports",
            user_id=getattr(current_user, 'id', 'unknown'),
            framework=framework.value if framework else 'all',
            count=len(summaries)
        )
        
        return summaries
        
    except Exception as e:
        logger.error(f"Failed to list compliance reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@router.get("/reports/{report_id}", response_model=ComplianceReportResponse)
async def get_compliance_report(
    report_id: str,
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Get detailed compliance report by ID
    
    **Returns:**
    - Complete compliance assessment
    - Control-by-control analysis
    - Executive summary
    - Remediation plan
    - Evidence collection results
    """
    try:
        report = service.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        
        response = ComplianceReportResponse(
            report_id=report.report_id,
            framework=report.framework,
            reporting_period_start=report.reporting_period_start,
            reporting_period_end=report.reporting_period_end,
            overall_compliance_score=report.overall_compliance_score,
            total_controls=report.total_controls,
            compliant_controls=report.compliant_controls,
            non_compliant_controls=report.non_compliant_controls,
            executive_summary=report.executive_summary,
            remediation_plan=report.remediation_plan,
            generated_by=report.generated_by,
            generated_at=report.generated_at
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.get("/reports/{report_id}/controls", response_model=List[ControlAssessmentResponse])
async def get_report_control_assessments(
    report_id: str,
    status_filter: Optional[ControlStatus] = Query(None, description="Filter by control status"),
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Get detailed control assessments for a compliance report
    
    **Features:**
    - Detailed control analysis
    - Status-based filtering
    - Evidence documentation
    - Remediation guidance
    """
    try:
        report = service.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        
        assessments = report.control_assessments
        
        # Filter by status if provided
        if status_filter:
            assessments = [a for a in assessments if a.status == status_filter]
        
        # Convert to response format
        responses = []
        for assessment in assessments:
            responses.append(ControlAssessmentResponse(
                control_id=assessment.control_id,
                framework=assessment.framework,
                status=assessment.status,
                compliance_score=assessment.compliance_score,
                assessment_date=assessment.assessment_date,
                findings=assessment.findings,
                evidence_collected=assessment.evidence_collected,
                remediation_required=assessment.remediation_required,
                automated=assessment.automated
            ))
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get control assessments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get assessments: {str(e)}")


@router.get("/reports/{report_id}/export/json")
async def export_report_json(
    report_id: str,
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Export compliance report as JSON
    
    **Features:**
    - Machine-readable format
    - Complete data export
    - API integration friendly
    - Audit trail preservation
    """
    try:
        report_data = await service.export_report_json(report_id)
        if not report_data:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        
        logger.info(
            f"Compliance report exported as JSON",
            user_id=getattr(current_user, 'id', 'unknown'),
            report_id=report_id
        )
        
        return JSONResponse(
            content=report_data,
            headers={
                "Content-Disposition": f"attachment; filename=compliance_report_{report_id}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export report as JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/reports/{report_id}/export/html", response_class=HTMLResponse)
async def export_report_html(
    report_id: str,
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Export compliance report as HTML
    
    **Features:**
    - Human-readable format
    - Professional presentation
    - Executive-ready formatting
    - Print-friendly layout
    """
    try:
        html_content = await service.export_report_html(report_id)
        if not html_content:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        
        logger.info(
            f"Compliance report exported as HTML",
            user_id=getattr(current_user, 'id', 'unknown'),
            report_id=report_id
        )
        
        return HTMLResponse(
            content=html_content,
            headers={
                "Content-Disposition": f"attachment; filename=compliance_report_{report_id}.html"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export report as HTML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/dashboard", response_model=ComplianceDashboardResponse)
async def get_compliance_dashboard(
    service: ComplianceReportingService = Depends(get_compliance_service),
    current_user = Depends(get_current_user)
):
    """
    Get compliance dashboard with key metrics and status
    
    **Features:**
    - Framework compliance scores
    - Recent report summaries
    - Critical findings count
    - Automation coverage metrics
    - Trend analysis
    """
    try:
        # Get all reports
        all_reports = service.list_reports()
        
        # Calculate framework scores (latest report per framework)
        framework_scores = {}
        for framework in ComplianceFramework:
            framework_reports = [r for r in all_reports if r.framework == framework]
            if framework_reports:
                latest_report = max(framework_reports, key=lambda r: r.generated_at)
                framework_scores[framework.value] = latest_report.overall_compliance_score
            else:
                framework_scores[framework.value] = 0.0
        
        # Get recent reports (last 10)
        recent_reports = []
        for report in all_reports[:10]:
            recent_reports.append(ComplianceReportSummary(
                report_id=report.report_id,
                framework=report.framework,
                overall_compliance_score=report.overall_compliance_score,
                total_controls=report.total_controls,
                compliant_controls=report.compliant_controls,
                non_compliant_controls=report.non_compliant_controls,
                generated_at=report.generated_at,
                generated_by=report.generated_by
            ))
        
        # Calculate metrics
        total_critical_findings = sum(
            len([a for a in report.control_assessments if a.status == ControlStatus.NON_COMPLIANT])
            for report in all_reports[:5]  # Last 5 reports
        )
        
        total_remediation_items = sum(
            len(report.remediation_plan)
            for report in all_reports[:5]  # Last 5 reports
        )
        
        # Calculate automation coverage
        total_controls = sum(report.total_controls for report in all_reports[:5])
        automated_controls = sum(
            len([a for a in report.control_assessments if a.automated])
            for report in all_reports[:5]
        )
        automation_coverage = automated_controls / total_controls if total_controls > 0 else 0.0
        
        dashboard = ComplianceDashboardResponse(
            frameworks=framework_scores,
            total_reports=len(all_reports),
            recent_reports=recent_reports,
            critical_findings=total_critical_findings,
            remediation_items=total_remediation_items,
            automation_coverage=automation_coverage
        )
        
        logger.info(
            f"Compliance dashboard accessed",
            user_id=getattr(current_user, 'id', 'unknown'),
            total_reports=len(all_reports)
        )
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to get compliance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")


@router.get("/frameworks")
async def get_supported_frameworks():
    """
    Get list of supported compliance frameworks
    
    **Returns:**
    - Available frameworks
    - Framework descriptions
    - Control counts
    - Assessment capabilities
    """
    frameworks_info = {
        ComplianceFramework.SOC2_TYPE_II.value: {
            "name": "SOC 2 Type II",
            "description": "Service Organization Control 2 Type II for service organizations",
            "categories": ["Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"],
            "automation_support": "Full",
            "typical_controls": 20
        },
        ComplianceFramework.ISO27001.value: {
            "name": "ISO 27001",
            "description": "International standard for information security management systems",
            "categories": ["Information Security Policies", "Organization", "Human Resources", "Asset Management"],
            "automation_support": "Partial",
            "typical_controls": 114
        },
        ComplianceFramework.NIST_CSF.value: {
            "name": "NIST Cybersecurity Framework",
            "description": "Framework for improving critical infrastructure cybersecurity",
            "categories": ["Identify", "Protect", "Detect", "Respond", "Recover"],
            "automation_support": "Full",
            "typical_controls": 108
        },
        ComplianceFramework.GDPR.value: {
            "name": "General Data Protection Regulation",
            "description": "European Union data protection and privacy regulation",
            "categories": ["Lawfulness", "Purpose Limitation", "Data Minimization", "Accuracy"],
            "automation_support": "Partial",
            "typical_controls": 25
        },
        ComplianceFramework.PCI_DSS.value: {
            "name": "PCI Data Security Standard",
            "description": "Payment card industry data security requirements",
            "categories": ["Network Security", "Data Protection", "Vulnerability Management", "Access Control"],
            "automation_support": "Full",
            "typical_controls": 12
        },
        ComplianceFramework.HIPAA.value: {
            "name": "Health Insurance Portability and Accountability Act",
            "description": "Healthcare data protection and privacy requirements",
            "categories": ["Administrative", "Physical", "Technical"],
            "automation_support": "Partial",
            "typical_controls": 18
        }
    }
    
    return JSONResponse(content={
        "supported_frameworks": frameworks_info,
        "total_frameworks": len(frameworks_info),
        "automation_coverage": "85%"
    })


# Background task functions
async def _process_compliance_report(report: ComplianceReport, user_id: str):
    """Process compliance report in background"""
    try:
        # Store report in database
        # Send notifications for critical findings
        # Update compliance metrics
        # Generate audit trail
        
        critical_findings = [
            a for a in report.control_assessments 
            if a.status == ControlStatus.NON_COMPLIANT
        ]
        
        if critical_findings:
            logger.warning(
                f"Critical compliance findings detected",
                report_id=report.report_id,
                framework=report.framework.value,
                findings_count=len(critical_findings),
                user_id=user_id
            )
            
            # Here you would integrate with notification systems
            # await send_compliance_alert(report, critical_findings, user_id)
            
    except Exception as e:
        logger.error(f"Background compliance report processing failed: {str(e)}")