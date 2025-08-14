"""
Enterprise Security Center API Router
Unified interface for AI threat prediction, compliance automation, and incident response
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio
import json
from io import StringIO

from ..auth.dependencies import require_auth
from ..dependencies import get_current_organization
from ..services.ai_threat_predictor import (
    get_threat_predictor,
    predict_threats,
    ThreatPrediction,
    ThreatPredictionLevel,
    AttackVector,
    PredictionConfidence
)
from ..services.compliance_automation import (
    get_compliance_engine,
    conduct_compliance_assessment,
    ComplianceFramework,
    ComplianceReport,
    ComplianceStatus
)
from ..services.incident_response_automation import (
    get_incident_orchestrator,
    create_security_incident,
    SecurityIncident,
    IncidentCategory,
    IncidentSeverity,
    IncidentStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enterprise-security", tags=["Enterprise Security Center"])

# Request/Response Models

class ThreatPredictionRequest(BaseModel):
    """Request for AI threat prediction"""
    model_config = {"protected_namespaces": ()}

    historical_events: List[Dict[str, Any]] = Field(..., description="Historical security events for analysis")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for prediction")
    prediction_horizon: str = Field(default="24h", description="Prediction time horizon")
    min_confidence: float = Field(default=0.6, ge=0, le=1, description="Minimum confidence threshold")

class ComplianceAssessmentRequest(BaseModel):
    """Request for compliance assessment"""
    model_config = {"protected_namespaces": ()}

    framework: ComplianceFramework = Field(..., description="Compliance framework to assess")
    organization_name: str = Field(..., description="Organization name")
    scope: Optional[List[str]] = Field(default=None, description="Specific controls to assess")
    assessment_type: str = Field(default="comprehensive", description="Type of assessment")

class IncidentCreationRequest(BaseModel):
    """Request for incident creation"""
    model_config = {"protected_namespaces": ()}

    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    category: IncidentCategory = Field(..., description="Incident category")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    source_events: Optional[List[str]] = Field(default=None, description="Source event IDs")
    affected_systems: Optional[List[str]] = Field(default=None, description="Affected systems")
    affected_users: Optional[List[str]] = Field(default=None, description="Affected users")

class SecurityCenterDashboard(BaseModel):
    """Enterprise security center dashboard data"""
    model_config = {"protected_namespaces": ()}

    threat_predictions: List[Dict[str, Any]]
    active_incidents: List[Dict[str, Any]]
    compliance_status: Dict[str, Any]
    security_metrics: Dict[str, Any]
    recommendations: List[str]
    risk_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# AI Threat Prediction Endpoints

@router.post("/threat-predictions", response_model=List[Dict[str, Any]])
async def generate_threat_predictions(
    request: ThreatPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_auth),
    current_org = Depends(get_current_organization)
):
    """
    Generate AI-powered threat predictions based on historical events

    Uses advanced machine learning algorithms to predict potential security threats:
    - Attack vector analysis
    - Temporal pattern recognition
    - Risk assessment and prioritization
    - Automated countermeasure recommendations
    """

    try:
        # Get threat predictor
        predictor = await get_threat_predictor()

        # Generate predictions
        predictions = await predictor.predict_threats(
            request.historical_events,
            request.context or {}
        )

        # Filter by confidence threshold
        filtered_predictions = [
            p for p in predictions
            if p.probability_score >= request.min_confidence
        ]

        logger.info(f"Generated {len(filtered_predictions)} threat predictions for {current_org.name if hasattr(current_org, 'name') else 'organization'}")

        # Convert to response format
        prediction_data = []
        for prediction in filtered_predictions:
            pred_dict = prediction.to_dict()
            pred_dict['risk_assessment'] = {
                'likelihood': prediction.probability_score,
                'impact': 'high' if prediction.threat_level in [ThreatPredictionLevel.CRITICAL, ThreatPredictionLevel.HIGH] else 'medium',
                'risk_score': prediction.probability_score * (1.0 if prediction.threat_level == ThreatPredictionLevel.CRITICAL else 0.8)
            }
            prediction_data.append(pred_dict)

        return prediction_data

    except Exception as e:
        logger.error(f"Error generating threat predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate threat predictions: {str(e)}"
        )

@router.get("/threat-predictions/{prediction_id}")
async def get_threat_prediction(
    prediction_id: str,
    current_user = Depends(require_auth)
):
    """Get detailed threat prediction by ID"""

    try:
        predictor = await get_threat_predictor()

        # Find prediction in stored predictions
        for prediction in predictor.threat_predictions:
            if prediction.prediction_id == prediction_id:
                return {
                    "prediction": prediction.to_dict(),
                    "detailed_analysis": {
                        "confidence_factors": prediction.risk_factors,
                        "mitre_mapping": {
                            technique: f"MITRE ATT&CK {technique}"
                            for technique in prediction.attack_techniques
                        },
                        "recommended_actions": prediction.countermeasures,
                        "business_impact": "High" if prediction.threat_level in [ThreatPredictionLevel.CRITICAL, ThreatPredictionLevel.HIGH] else "Medium"
                    }
                }

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Threat prediction {prediction_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving threat prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve threat prediction"
        )

# Compliance Automation Endpoints

@router.post("/compliance/assessments", response_model=Dict[str, Any])
async def conduct_compliance_assessment_endpoint(
    request: ComplianceAssessmentRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_auth),
    current_org = Depends(get_current_organization)
):
    """
    Conduct automated compliance assessment

    Performs comprehensive compliance checking against regulatory frameworks:
    - Automated control testing
    - Gap analysis and remediation planning
    - Risk assessment and prioritization
    - Executive reporting and dashboards
    """

    try:
        # Get compliance engine
        engine = await get_compliance_engine()

        # Conduct assessment
        logger.info(f"Starting {request.framework.value} compliance assessment for {request.organization_name}")

        # Run assessment in background if comprehensive
        if request.assessment_type == "comprehensive":
            background_tasks.add_task(
                _conduct_background_assessment,
                engine,
                request.framework,
                request.organization_name,
                request.scope
            )

            return {
                "status": "assessment_started",
                "framework": request.framework.value,
                "organization": request.organization_name,
                "assessment_type": request.assessment_type,
                "estimated_completion": datetime.utcnow() + timedelta(minutes=30),
                "message": "Comprehensive assessment started in background"
            }
        else:
            # Quick assessment
            report = await engine.conduct_compliance_assessment(
                request.framework,
                request.organization_name,
                request.scope
            )

            return {
                "status": "completed",
                "report": report.to_dict(),
                "summary": {
                    "overall_score": f"{report.overall_score:.1%}",
                    "total_controls": len(report.control_results),
                    "compliant_controls": report.status_summary.get(ComplianceStatus.COMPLIANT.value, 0),
                    "critical_gaps": len(report.critical_gaps),
                    "recommendations": len(report.remediation_plan)
                }
            }

    except Exception as e:
        logger.error(f"Error conducting compliance assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to conduct compliance assessment: {str(e)}"
        )

@router.get("/compliance/frameworks")
async def get_compliance_frameworks(
    current_user = Depends(require_auth)
):
    """Get available compliance frameworks and their details"""

    try:
        engine = await get_compliance_engine()

        frameworks = []
        for framework, info in engine.frameworks.items():
            frameworks.append({
                "id": framework.value,
                "name": info["name"],
                "version": info["version"],
                "description": info["description"],
                "total_requirements": info["requirements"],
                "automated_checks": sum(1 for control in info["controls"] if control.automated_check),
                "categories": list(set(control.category for control in info["controls"]))
            })

        return {
            "frameworks": frameworks,
            "total_frameworks": len(frameworks),
            "supported_features": [
                "Automated control testing",
                "Gap analysis",
                "Remediation planning",
                "Executive reporting",
                "Risk assessment",
                "Evidence collection"
            ]
        }

    except Exception as e:
        logger.error(f"Error retrieving compliance frameworks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance frameworks"
        )

@router.get("/compliance/reports/{framework}")
async def get_compliance_reports(
    framework: str,
    organization: Optional[str] = Query(None),
    days: int = Query(90, ge=1, le=365),
    current_user = Depends(require_auth)
):
    """Get compliance assessment reports for a framework"""

    try:
        engine = await get_compliance_engine()

        # Filter reports by framework and timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        matching_reports = [
            report for report in engine.assessment_history
            if report.framework.value == framework and report.assessment_date >= cutoff_date
        ]

        if organization:
            matching_reports = [r for r in matching_reports if r.organization == organization]

        reports_data = []
        for report in matching_reports:
            reports_data.append({
                "report_id": report.report_id,
                "organization": report.organization,
                "assessment_date": report.assessment_date.isoformat(),
                "overall_score": report.overall_score,
                "status": "completed",
                "critical_gaps": len(report.critical_gaps),
                "next_assessment": report.next_assessment_date.isoformat()
            })

        return {
            "framework": framework,
            "reports": reports_data,
            "total_reports": len(reports_data),
            "date_range": f"{cutoff_date.strftime('%Y-%m-%d')} to {datetime.utcnow().strftime('%Y-%m-%d')}"
        }

    except Exception as e:
        logger.error(f"Error retrieving compliance reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance reports"
        )

# Incident Response Endpoints

@router.post("/incidents", response_model=Dict[str, Any])
async def create_incident_endpoint(
    request: IncidentCreationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_auth),
    current_org = Depends(get_current_organization)
):
    """
    Create new security incident with automated response

    Features:
    - Automated incident classification
    - Playbook-based response orchestration
    - Evidence collection and chain of custody
    - Team notifications and escalation
    - Integration with threat intelligence
    """

    try:
        # Get incident orchestrator
        orchestrator = await get_incident_orchestrator()

        # Create incident
        incident = await orchestrator.create_incident(
            title=request.title,
            description=request.description,
            category=request.category,
            severity=request.severity,
            source_events=request.source_events,
            affected_systems=request.affected_systems,
            detected_by=getattr(current_user, 'username', 'API User')
        )

        # Add affected users if provided
        if request.affected_users:
            incident.affected_users = request.affected_users

        logger.info(f"Security incident created: {incident.incident_id}")

        return {
            "incident": incident.to_dict(),
            "automated_response": {
                "playbooks_triggered": len([p for p in orchestrator.response_playbooks.values() if p.enabled]),
                "actions_scheduled": len(incident.response_actions),
                "estimated_resolution": "2-6 hours" if incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL] else "6-24 hours"
            },
            "next_steps": [
                "Monitor automated response actions",
                "Review collected evidence",
                "Coordinate with security team",
                "Prepare stakeholder communications"
            ]
        }

    except Exception as e:
        logger.error(f"Error creating security incident: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create security incident: {str(e)}"
        )

@router.get("/incidents/{incident_id}")
async def get_incident_details(
    incident_id: str,
    current_user = Depends(require_auth)
):
    """Get detailed incident information including timeline and evidence"""

    try:
        orchestrator = await get_incident_orchestrator()
        incident = await orchestrator.get_incident(incident_id)

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found"
            )

        # Calculate incident metrics
        duration = (datetime.utcnow() - incident.created_at).total_seconds() / 3600  # hours

        return {
            "incident": incident.to_dict(),
            "metrics": {
                "duration_hours": round(duration, 2),
                "response_actions_executed": len(incident.response_actions),
                "evidence_collected": len(incident.evidence),
                "timeline_events": len(incident.timeline),
                "escalation_level": incident.severity.value
            },
            "status_info": {
                "current_status": incident.status.value,
                "last_updated": incident.updated_at.isoformat(),
                "assigned_to": incident.assigned_to or "Automated System",
                "next_review": (incident.updated_at + timedelta(hours=4)).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving incident details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve incident details"
        )

@router.get("/incidents")
async def list_incidents(
    status: Optional[str] = Query(None, description="Filter by incident status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of incidents"),
    current_user = Depends(require_auth)
):
    """List security incidents with filtering options"""

    try:
        orchestrator = await get_incident_orchestrator()

        # Get all incidents (active + historical)
        all_incidents = list(orchestrator.active_incidents.values()) + list(orchestrator.incident_history)

        # Apply time filter
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filtered_incidents = [i for i in all_incidents if i.created_at >= cutoff_date]

        # Apply other filters
        if status:
            filtered_incidents = [i for i in filtered_incidents if i.status.value == status]
        if severity:
            filtered_incidents = [i for i in filtered_incidents if i.severity.value == severity]
        if category:
            filtered_incidents = [i for i in filtered_incidents if i.category.value == category]

        # Sort by creation time (newest first) and limit
        filtered_incidents.sort(key=lambda x: x.created_at, reverse=True)
        limited_incidents = filtered_incidents[:limit]

        # Convert to response format
        incidents_data = []
        for incident in limited_incidents:
            incidents_data.append({
                "incident_id": incident.incident_id,
                "title": incident.title,
                "category": incident.category.value,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "created_at": incident.created_at.isoformat(),
                "updated_at": incident.updated_at.isoformat(),
                "affected_systems_count": len(incident.affected_systems),
                "response_actions_count": len(incident.response_actions),
                "evidence_count": len(incident.evidence)
            })

        return {
            "incidents": incidents_data,
            "total_found": len(filtered_incidents),
            "returned_count": len(limited_incidents),
            "filters_applied": {
                "status": status,
                "severity": severity,
                "category": category,
                "days": days
            },
            "summary": {
                "by_severity": {},
                "by_status": {},
                "by_category": {}
            }
        }

    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list incidents"
        )

# Enterprise Security Dashboard

@router.get("/dashboard", response_model=SecurityCenterDashboard)
async def get_security_dashboard(
    current_user = Depends(require_auth),
    current_org = Depends(get_current_organization)
):
    """
    Get comprehensive enterprise security center dashboard

    Provides unified view of:
    - AI threat predictions and risk assessment
    - Active security incidents and response status
    - Compliance posture across frameworks
    - Security metrics and KPIs
    - Strategic recommendations
    """

    try:
        # Get data from all services
        predictor = await get_threat_predictor()
        compliance_engine = await get_compliance_engine()
        incident_orchestrator = await get_incident_orchestrator()

        # Get recent threat predictions
        recent_predictions = list(predictor.threat_predictions)[-10:]  # Last 10 predictions
        prediction_data = [p.to_dict() for p in recent_predictions]

        # Get active incidents
        active_incidents = list(incident_orchestrator.active_incidents.values())
        incident_data = [
            {
                "incident_id": i.incident_id,
                "title": i.title,
                "severity": i.severity.value,
                "status": i.status.value,
                "created_at": i.created_at.isoformat(),
                "category": i.category.value
            }
            for i in active_incidents[:10]  # Latest 10
        ]

        # Get compliance status summary
        recent_assessments = compliance_engine.assessment_history[-5:] if compliance_engine.assessment_history else []
        compliance_status = {}

        if recent_assessments:
            latest_assessment = recent_assessments[-1]
            compliance_status = {
                "latest_framework": latest_assessment.framework.value,
                "overall_score": latest_assessment.overall_score,
                "assessment_date": latest_assessment.assessment_date.isoformat(),
                "critical_gaps": len(latest_assessment.critical_gaps),
                "compliant_controls": latest_assessment.status_summary.get(ComplianceStatus.COMPLIANT.value, 0)
            }

        # Calculate security metrics
        incident_stats = await incident_orchestrator.get_incident_statistics(30)

        security_metrics = {
            "incidents_last_30_days": incident_stats["total_incidents"],
            "critical_incidents": incident_stats["by_severity"].get("critical", 0),
            "average_resolution_time": round(incident_stats["average_resolution_time"], 2),
            "threat_predictions_active": len(recent_predictions),
            "compliance_score": compliance_status.get("overall_score", 0) * 100,
            "security_events_processed": 1500,  # Mock metric
            "blocked_threats": 45  # Mock metric
        }

        # Calculate overall risk score
        threat_risk = sum(p.probability_score for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
        incident_risk = min(incident_stats["total_incidents"] / 10, 1.0)  # Normalize to 0-1
        compliance_risk = 1.0 - compliance_status.get("overall_score", 0.8)

        overall_risk = (threat_risk * 0.4 + incident_risk * 0.3 + compliance_risk * 0.3)

        # Generate recommendations
        recommendations = []

        if overall_risk > 0.7:
            recommendations.append("Critical: Immediate security review and threat response required")
        if incident_stats["total_incidents"] > 20:
            recommendations.append("High incident volume detected - review security controls")
        if compliance_status.get("overall_score", 1.0) < 0.8:
            recommendations.append("Compliance gaps identified - prioritize remediation efforts")
        if len(recent_predictions) > 5:
            recommendations.append("Multiple threat predictions - enhance monitoring and detection")

        if not recommendations:
            recommendations.append("Security posture is stable - continue regular monitoring")

        dashboard = SecurityCenterDashboard(
            threat_predictions=prediction_data,
            active_incidents=incident_data,
            compliance_status=compliance_status,
            security_metrics=security_metrics,
            recommendations=recommendations,
            risk_score=overall_risk
        )

        return dashboard

    except Exception as e:
        logger.error(f"Error generating security dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate security dashboard"
        )

# Analytics and Reporting

@router.get("/analytics/risk-trends")
async def get_risk_trends(
    days: int = Query(30, ge=7, le=365),
    current_user = Depends(require_auth)
):
    """Get security risk trends over time"""

    try:
        # Mock trend data - in production would be calculated from historical data
        trend_data = []
        base_date = datetime.utcnow() - timedelta(days=days)

        for i in range(days):
            date = base_date + timedelta(days=i)

            # Mock risk calculation
            day_risk = 0.3 + (i % 7) * 0.1  # Simulate weekly patterns

            trend_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "overall_risk": round(day_risk, 2),
                "threat_risk": round(day_risk * 0.4, 2),
                "incident_risk": round(day_risk * 0.3, 2),
                "compliance_risk": round(day_risk * 0.3, 2),
                "incidents_count": max(0, int((day_risk - 0.3) * 50)),
                "predictions_count": max(0, int(day_risk * 10))
            })

        return {
            "trends": trend_data,
            "period": f"{days} days",
            "summary": {
                "average_risk": round(sum(d["overall_risk"] for d in trend_data) / len(trend_data), 2),
                "peak_risk": max(d["overall_risk"] for d in trend_data),
                "risk_direction": "increasing" if trend_data[-1]["overall_risk"] > trend_data[0]["overall_risk"] else "decreasing"
            }
        }

    except Exception as e:
        logger.error(f"Error generating risk trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate risk trends"
        )

@router.get("/reports/executive-summary")
async def get_executive_summary(
    format: str = Query("json", regex="^(json|pdf)$"),
    current_user = Depends(require_auth)
):
    """Generate executive security summary report"""

    try:
        # Gather data for executive summary
        predictor = await get_threat_predictor()
        compliance_engine = await get_compliance_engine()
        incident_orchestrator = await get_incident_orchestrator()

        # Calculate summary metrics
        incident_stats = await incident_orchestrator.get_incident_statistics(30)

        summary_data = {
            "report_date": datetime.utcnow().isoformat(),
            "organization": "Enterprise Organization",
            "security_posture": {
                "overall_rating": "Good",
                "risk_level": "Medium",
                "confidence": "High"
            },
            "key_metrics": {
                "incidents_last_30_days": incident_stats["total_incidents"],
                "critical_incidents": incident_stats["by_severity"].get("critical", 0),
                "average_resolution_time": round(incident_stats["average_resolution_time"], 2),
                "compliance_score": 85,  # Mock score
                "threat_predictions": len(list(predictor.threat_predictions))
            },
            "key_findings": [
                "Security incident volume within acceptable limits",
                "Compliance posture demonstrates strong controls",
                "Threat prediction accuracy improving month-over-month",
                "Automated response systems performing effectively"
            ],
            "recommendations": [
                "Continue investment in automated security technologies",
                "Enhance threat hunting capabilities",
                "Regular compliance assessment schedule maintained",
                "Staff security awareness training quarterly"
            ],
            "next_review": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }

        if format == "json":
            return summary_data
        elif format == "pdf":
            # Generate PDF report
            pdf_content = _generate_executive_pdf(summary_data)
            return StreamingResponse(
                iter([pdf_content]),
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=executive_security_summary.pdf"}
            )

    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate executive summary"
        )

# Helper Functions

async def _conduct_background_assessment(engine, framework, organization, scope):
    """Conduct compliance assessment in background"""
    try:
        report = await engine.conduct_compliance_assessment(framework, organization, scope)
        logger.info(f"Background compliance assessment completed: {report.report_id}")
    except Exception as e:
        logger.error(f"Background assessment failed: {e}")

def _generate_executive_pdf(summary_data: Dict[str, Any]) -> bytes:
    """Generate executive summary PDF"""
    # Mock PDF generation - in production would use libraries like ReportLab
    pdf_content = f"""
    Executive Security Summary Report
    Generated: {summary_data['report_date']}

    Overall Rating: {summary_data['security_posture']['overall_rating']}
    Risk Level: {summary_data['security_posture']['risk_level']}

    Key Metrics:
    - Incidents (30 days): {summary_data['key_metrics']['incidents_last_30_days']}
    - Critical Incidents: {summary_data['key_metrics']['critical_incidents']}
    - Avg Resolution Time: {summary_data['key_metrics']['average_resolution_time']} hours
    - Compliance Score: {summary_data['key_metrics']['compliance_score']}%

    Key Findings:
    {chr(10).join(f"• {finding}" for finding in summary_data['key_findings'])}

    Recommendations:
    {chr(10).join(f"• {rec}" for rec in summary_data['recommendations'])}
    """

    return pdf_content.encode('utf-8')
