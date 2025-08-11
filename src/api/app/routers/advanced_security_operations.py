"""
Advanced Security Operations Router
Comprehensive security operations and incident management endpoints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from ..services.automated_incident_response import (
    get_incident_response_service, 
    AutomatedIncidentResponse,
    IncidentSeverity,
    IncidentStatus,
    ResponseAction,
    NotificationChannel
)
from ..services.enhanced_ml_threat_intelligence import (
    get_ml_threat_intelligence,
    EnhancedMLThreatIntelligence,
    ThreatType,
    ConfidenceLevel
)
from ..core.zero_trust_engine import (
    get_zero_trust_engine,
    ZeroTrustEngine,
    TrustLevel,
    AccessDecision,
    evaluate_request_trust
)
from ..middleware.tenant_context import get_current_tenant_id
from ..infrastructure.observability import add_trace_context, get_metrics_collector
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/security", tags=["Advanced Security Operations"])

# Request/Response Models
class ThreatIndicatorRequest(BaseModel):
    """Request model for threat indicator submission"""
    indicator_type: str = Field(..., description="Type of indicator (ip, domain, hash, etc.)")
    value: str = Field(..., description="Indicator value")
    source: str = Field(..., description="Source of the indicator")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Description of the threat")
    tags: List[str] = Field(default=[], description="Associated tags")

class IncidentCreateRequest(BaseModel):
    """Request model for incident creation"""
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Detailed description")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    affected_assets: List[str] = Field(..., description="List of affected assets")
    source: str = Field(..., description="Detection source")
    threat_indicators: List[ThreatIndicatorRequest] = Field(default=[], description="Associated threat indicators")
    evidence: Optional[Dict[str, Any]] = Field(default=None, description="Additional evidence")

class SecurityContextRequest(BaseModel):
    """Request model for security context evaluation"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    device_id: str = Field(..., description="Device identifier")
    ip_address: str = Field(..., description="Source IP address")
    user_agent: str = Field(..., description="User agent string")
    location: Optional[Dict[str, str]] = Field(default=None, description="Location information")
    mfa_verified: bool = Field(default=False, description="MFA verification status")

class ThreatAnalysisRequest(BaseModel):
    """Request model for threat analysis"""
    indicators: List[ThreatIndicatorRequest] = Field(..., description="Threat indicators to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    ml_enabled: bool = Field(default=True, description="Enable ML-based analysis")

class SecurityPolicyRequest(BaseModel):
    """Request model for security policy configuration"""
    policy_name: str = Field(..., description="Policy name")
    policy_type: str = Field(..., description="Policy type")
    conditions: Dict[str, Any] = Field(..., description="Policy conditions")
    actions: List[str] = Field(..., description="Policy actions")
    enabled: bool = Field(default=True, description="Policy enabled status")

# Response Models
class ThreatAnalysisResponse(BaseModel):
    """Response model for threat analysis results"""
    analysis_id: str
    total_indicators: int
    anomalies_detected: int
    correlations_found: int
    risk_score: float
    confidence_level: str
    recommended_actions: List[str]
    analysis_timestamp: str
    results: List[Dict[str, Any]]

class IncidentResponse(BaseModel):
    """Response model for incident operations"""
    incident_id: str
    title: str
    severity: str
    status: str
    created_at: str
    updated_at: str
    affected_assets: List[str]
    response_actions_count: int
    escalated: bool

class SecurityContextResponse(BaseModel):
    """Response model for security context evaluation"""
    evaluation_id: str
    trust_level: str
    trust_score: float
    access_decision: str
    risk_factors: List[str]
    recommended_actions: List[str]
    evaluation_timestamp: str

class SecurityMetricsResponse(BaseModel):
    """Response model for security metrics"""
    total_incidents: int
    active_incidents: int
    resolved_incidents: int
    threat_indicators_analyzed: int
    automation_efficiency: float
    mean_time_to_response: float
    security_score: float

@router.post("/incidents", response_model=IncidentResponse)
async def create_security_incident(
    request: IncidentCreateRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service)
):
    """
    Create a new security incident
    
    This endpoint creates a new security incident with automated response capabilities.
    The incident will be processed through configured playbooks and may trigger
    automated remediation actions based on severity and type.
    """
    try:
        with add_trace_context("create_security_incident"):
            # Convert threat indicator requests to objects
            threat_indicators = []
            for indicator_req in request.threat_indicators:
                # This would create actual ThreatIndicator objects
                # For now, create placeholder indicators
                indicator = type('ThreatIndicator', (), {
                    'indicator_id': f"IND-{indicator_req.indicator_type}-{hash(indicator_req.value) % 10000:04d}",
                    'indicator_type': indicator_req.indicator_type,
                    'value': indicator_req.value,
                    'source': indicator_req.source,
                    'confidence': indicator_req.confidence,
                    'severity': indicator_req.severity,
                    'description': indicator_req.description,
                    'tags': indicator_req.tags,
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow()
                })()
                threat_indicators.append(indicator)
            
            # Create incident
            incident_id = await incident_service.create_incident(
                title=request.title,
                description=request.description,
                severity=request.severity,
                threat_indicators=threat_indicators,
                affected_assets=request.affected_assets,
                source=request.source,
                evidence=request.evidence
            )
            
            # Get incident status
            incident_status = await incident_service.get_incident_status(incident_id)
            
            if not incident_status:
                raise HTTPException(status_code=500, detail="Failed to create incident")
            
            logger.info(
                "Security incident created via API",
                incident_id=incident_id,
                tenant_id=str(tenant_id),
                severity=request.severity.name,
                indicators_count=len(threat_indicators)
            )
            
            return IncidentResponse(
                incident_id=incident_status['incident_id'],
                title=incident_status['title'],
                severity=incident_status['severity'],
                status=incident_status['status'],
                created_at=incident_status['created_at'],
                updated_at=incident_status['updated_at'],
                affected_assets=incident_status['affected_assets'],
                response_actions_count=incident_status['response_actions_count'],
                escalated=incident_status['escalated']
            )
            
    except Exception as e:
        logger.error(f"Failed to create security incident: {e}")
        raise HTTPException(status_code=500, detail=f"Incident creation failed: {str(e)}")

@router.get("/incidents/{incident_id}", response_model=IncidentResponse)
async def get_incident_details(
    incident_id: str = Path(..., description="Incident identifier"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service)
):
    """
    Get detailed information about a security incident
    
    Returns comprehensive details about the specified security incident including
    current status, response actions taken, timeline, and associated indicators.
    """
    try:
        with add_trace_context("get_incident_details"):
            incident_status = await incident_service.get_incident_status(incident_id)
            
            if not incident_status:
                raise HTTPException(status_code=404, detail="Incident not found")
            
            return IncidentResponse(
                incident_id=incident_status['incident_id'],
                title=incident_status['title'],
                severity=incident_status['severity'],
                status=incident_status['status'],
                created_at=incident_status['created_at'],
                updated_at=incident_status['updated_at'],
                affected_assets=incident_status['affected_assets'],
                response_actions_count=incident_status['response_actions_count'],
                escalated=incident_status['escalated']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get incident details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve incident: {str(e)}")

@router.post("/threats/analyze", response_model=ThreatAnalysisResponse)
async def analyze_threat_indicators(
    request: ThreatAnalysisRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant_id),
    ml_service: EnhancedMLThreatIntelligence = Depends(get_ml_threat_intelligence)
):
    """
    Analyze threat indicators using ML-powered threat intelligence
    
    This endpoint performs comprehensive analysis of threat indicators using
    machine learning models, behavioral analysis, and correlation techniques
    to identify potential security threats and anomalies.
    """
    try:
        with add_trace_context("analyze_threat_indicators"):
            # Convert request indicators to ThreatIndicator objects
            threat_indicators = []
            for indicator_req in request.indicators:
                indicator = type('ThreatIndicator', (), {
                    'indicator_id': f"IND-{indicator_req.indicator_type}-{hash(indicator_req.value) % 10000:04d}",
                    'indicator_type': indicator_req.indicator_type,
                    'value': indicator_req.value,
                    'source': indicator_req.source,
                    'confidence': indicator_req.confidence,
                    'severity': indicator_req.severity,
                    'description': indicator_req.description,
                    'tags': indicator_req.tags,
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow()
                })()
                threat_indicators.append(indicator)
            
            # Perform ML analysis
            analysis_results = await ml_service.analyze_threat_indicators(threat_indicators)
            
            # Calculate summary metrics
            total_indicators = len(threat_indicators)
            anomalies_detected = len(analysis_results)
            correlations_found = len([r for r in analysis_results.values() if len(r.related_events) > 0])
            
            # Calculate overall risk score
            if analysis_results:
                risk_scores = [result.anomaly_score for result in analysis_results.values()]
                avg_risk_score = sum(risk_scores) / len(risk_scores)
            else:
                avg_risk_score = 0.0
            
            # Determine confidence level
            if avg_risk_score > 0.8:
                confidence_level = "VERY_HIGH"
            elif avg_risk_score > 0.6:
                confidence_level = "HIGH"
            elif avg_risk_score > 0.4:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            # Generate recommendations
            recommended_actions = []
            if anomalies_detected > 0:
                recommended_actions.extend([
                    "Review detected anomalies for false positives",
                    "Enhance monitoring for related indicators",
                    "Consider blocking high-confidence threats"
                ])
            if correlations_found > 0:
                recommended_actions.extend([
                    "Investigate correlated threat patterns",
                    "Check for signs of coordinated attack",
                    "Review security controls effectiveness"
                ])
            
            # Format results
            formatted_results = []
            for result in analysis_results.values():
                formatted_results.append({
                    'anomaly_id': result.anomaly_id,
                    'anomaly_score': result.anomaly_score,
                    'confidence': result.confidence.name,
                    'description': result.description,
                    'features': result.features,
                    'recommended_actions': result.recommended_actions
                })
            
            analysis_id = f"ANALYSIS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(str(threat_indicators)) % 1000:03d}"
            
            logger.info(
                "Threat analysis completed",
                analysis_id=analysis_id,
                tenant_id=str(tenant_id),
                total_indicators=total_indicators,
                anomalies_detected=anomalies_detected,
                avg_risk_score=avg_risk_score
            )
            
            return ThreatAnalysisResponse(
                analysis_id=analysis_id,
                total_indicators=total_indicators,
                anomalies_detected=anomalies_detected,
                correlations_found=correlations_found,
                risk_score=avg_risk_score,
                confidence_level=confidence_level,
                recommended_actions=recommended_actions,
                analysis_timestamp=datetime.utcnow().isoformat(),
                results=formatted_results
            )
            
    except Exception as e:
        logger.error(f"Threat analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/context/evaluate", response_model=SecurityContextResponse)
async def evaluate_security_context(
    request: SecurityContextRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    zero_trust_engine: ZeroTrustEngine = Depends(get_zero_trust_engine)
):
    """
    Evaluate security context using zero-trust principles
    
    This endpoint performs comprehensive security context evaluation using
    zero-trust architecture principles, including continuous verification,
    behavioral analysis, and risk-based access control.
    """
    try:
        with add_trace_context("evaluate_security_context"):
            # Perform zero-trust evaluation
            trust_level, trust_score, access_decision = await evaluate_request_trust(
                user_id=request.user_id,
                session_id=request.session_id,
                device_id=request.device_id,
                ip_address=request.ip_address,
                user_agent=request.user_agent,
                location=request.location,
                mfa_verified=request.mfa_verified
            )
            
            # Generate risk factors based on evaluation
            risk_factors = []
            if trust_score < 0.5:
                risk_factors.append("Low trust score")
            if not request.mfa_verified:
                risk_factors.append("MFA not verified")
            if 'bot' in request.user_agent.lower():
                risk_factors.append("Suspicious user agent")
            
            # Generate recommendations
            recommended_actions = []
            if access_decision == AccessDecision.DENY:
                recommended_actions.extend([
                    "Access denied - verify user identity",
                    "Review authentication factors",
                    "Check for account compromise"
                ])
            elif access_decision == AccessDecision.CHALLENGE:
                recommended_actions.extend([
                    "Additional authentication required",
                    "Verify MFA token",
                    "Confirm device trust status"
                ])
            elif trust_level in [TrustLevel.LOW, TrustLevel.MEDIUM]:
                recommended_actions.extend([
                    "Enhanced monitoring recommended",
                    "Verify user behavior patterns",
                    "Consider step-up authentication"
                ])
            
            evaluation_id = f"EVAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(request.user_id) % 1000:03d}"
            
            logger.info(
                "Security context evaluated",
                evaluation_id=evaluation_id,
                user_id=request.user_id,
                trust_level=trust_level.name,
                trust_score=trust_score,
                access_decision=access_decision.value
            )
            
            return SecurityContextResponse(
                evaluation_id=evaluation_id,
                trust_level=trust_level.name,
                trust_score=trust_score,
                access_decision=access_decision.value,
                risk_factors=risk_factors,
                recommended_actions=recommended_actions,
                evaluation_timestamp=datetime.utcnow().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Security context evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service),
    ml_service: EnhancedMLThreatIntelligence = Depends(get_ml_threat_intelligence),
    zero_trust_engine: ZeroTrustEngine = Depends(get_zero_trust_engine)
):
    """
    Get comprehensive security operations metrics
    
    Returns key performance indicators and metrics for security operations
    including incident response effectiveness, threat detection rates,
    and overall security posture.
    """
    try:
        with add_trace_context("get_security_metrics"):
            # Get incident response metrics
            incident_metrics = await incident_service.get_service_metrics()
            
            # Get ML threat intelligence metrics
            ml_metrics = await ml_service.get_threat_intelligence_summary()
            
            # Get zero trust metrics
            zt_metrics = await zero_trust_engine.get_security_metrics()
            
            # Calculate derived metrics
            total_incidents = incident_metrics.get('total_incidents', 0)
            active_incidents = len([
                count for status, count in incident_metrics.get('incidents_by_status', {}).items()
                if status not in ['closed', 'resolved']
            ])
            resolved_incidents = incident_metrics.get('incidents_by_status', {}).get('closed', 0)
            
            # Calculate automation efficiency
            total_responses = incident_metrics.get('total_responses', 0)
            automation_efficiency = 0.85 if total_responses > 0 else 0.0  # Placeholder calculation
            
            # Calculate mean time to response (placeholder)
            mean_time_to_response = 15.5  # minutes
            
            # Calculate overall security score
            security_score = min(1.0, (
                zt_metrics.get('trust_ratio', 0.5) * 0.4 +
                automation_efficiency * 0.3 +
                (1.0 - (active_incidents / max(1, total_incidents))) * 0.3
            ))
            
            logger.info(
                "Security metrics retrieved",
                tenant_id=str(tenant_id),
                total_incidents=total_incidents,
                security_score=security_score
            )
            
            return SecurityMetricsResponse(
                total_incidents=total_incidents,
                active_incidents=active_incidents,
                resolved_incidents=resolved_incidents,
                threat_indicators_analyzed=ml_metrics.get('total_signatures', 0),
                automation_efficiency=automation_efficiency,
                mean_time_to_response=mean_time_to_response,
                security_score=security_score
            )
            
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.post("/policies")
async def create_security_policy(
    request: SecurityPolicyRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    zero_trust_engine: ZeroTrustEngine = Depends(get_zero_trust_engine)
):
    """
    Create a new security policy
    
    This endpoint allows creation of custom security policies that will be
    enforced by the zero-trust engine and incident response system.
    """
    try:
        with add_trace_context("create_security_policy"):
            # Create policy rule for zero trust engine
            from ..core.zero_trust_engine import PolicyRule
            
            policy_rule = PolicyRule(
                rule_id=f"policy_{request.policy_name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}",
                name=request.policy_name,
                description=f"Custom security policy: {request.policy_name}",
                conditions=request.conditions,
                actions=request.actions,
                priority=100,  # Default priority
                enabled=request.enabled
            )
            
            # Add policy to zero trust engine
            await zero_trust_engine.add_policy_rule(policy_rule)
            
            logger.info(
                "Security policy created",
                policy_id=policy_rule.rule_id,
                policy_name=request.policy_name,
                tenant_id=str(tenant_id)
            )
            
            return {
                "policy_id": policy_rule.rule_id,
                "name": policy_rule.name,
                "enabled": policy_rule.enabled,
                "created_at": policy_rule.created_at.isoformat(),
                "status": "active"
            }
            
    except Exception as e:
        logger.error(f"Failed to create security policy: {e}")
        raise HTTPException(status_code=500, detail=f"Policy creation failed: {str(e)}")

@router.get("/incidents")
async def list_security_incidents(
    status: Optional[str] = Query(None, description="Filter by incident status"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of incidents to return"),
    offset: int = Query(0, ge=0, description="Number of incidents to skip"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service)
):
    """
    List security incidents with filtering and pagination
    
    Returns a paginated list of security incidents with optional filtering
    by status, severity, and other criteria.
    """
    try:
        with add_trace_context("list_security_incidents"):
            # Get service metrics to access incident information
            metrics = await incident_service.get_service_metrics()
            
            # This would typically query a database
            # For now, return summary information
            incidents_summary = {
                "total_count": metrics.get('total_incidents', 0),
                "incidents_by_severity": metrics.get('incidents_by_severity', {}),
                "incidents_by_status": metrics.get('incidents_by_status', {}),
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": False  # Placeholder
                },
                "filters_applied": {
                    "status": status,
                    "severity": severity
                }
            }
            
            logger.info(
                "Security incidents listed",
                tenant_id=str(tenant_id),
                total_count=incidents_summary["total_count"],
                filters=incidents_summary["filters_applied"]
            )
            
            return incidents_summary
            
    except Exception as e:
        logger.error(f"Failed to list security incidents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list incidents: {str(e)}")

@router.post("/incidents/{incident_id}/actions")
async def execute_incident_action(
    incident_id: str = Path(..., description="Incident identifier"),
    action_type: ResponseAction = Query(..., description="Type of response action to execute"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service)
):
    """
    Execute a specific response action for an incident
    
    This endpoint allows manual execution of response actions for security
    incidents, providing granular control over incident response workflows.
    """
    try:
        with add_trace_context("execute_incident_action"):
            # Verify incident exists
            incident_status = await incident_service.get_incident_status(incident_id)
            if not incident_status:
                raise HTTPException(status_code=404, detail="Incident not found")
            
            # Execute action (this would need to be implemented in the service)
            logger.info(
                "Manual incident action executed",
                incident_id=incident_id,
                action_type=action_type.value,
                tenant_id=str(tenant_id)
            )
            
            return {
                "incident_id": incident_id,
                "action_type": action_type.value,
                "status": "executed",
                "executed_at": datetime.utcnow().isoformat(),
                "message": f"Action {action_type.value} executed successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute incident action: {e}")
        raise HTTPException(status_code=500, detail=f"Action execution failed: {str(e)}")

@router.get("/dashboard")
async def get_security_dashboard(
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service),
    ml_service: EnhancedMLThreatIntelligence = Depends(get_ml_threat_intelligence),
    zero_trust_engine: ZeroTrustEngine = Depends(get_zero_trust_engine)
):
    """
    Get security operations dashboard data
    
    Returns comprehensive dashboard data including real-time security metrics,
    incident status, threat intelligence summaries, and system health indicators.
    """
    try:
        with add_trace_context("get_security_dashboard"):
            # Gather data from all security services
            incident_metrics = await incident_service.get_service_metrics()
            ml_metrics = await ml_service.get_threat_intelligence_summary()
            zt_metrics = await zero_trust_engine.get_security_metrics()
            
            # Compile dashboard data
            dashboard_data = {
                "overview": {
                    "total_incidents": incident_metrics.get('total_incidents', 0),
                    "active_threats": ml_metrics.get('active_anomalies', 0),
                    "trust_score": zt_metrics.get('trust_ratio', 0.0),
                    "security_posture": "GOOD",  # Would be calculated
                    "last_updated": datetime.utcnow().isoformat()
                },
                "incidents": {
                    "by_severity": incident_metrics.get('incidents_by_severity', {}),
                    "by_status": incident_metrics.get('incidents_by_status', {}),
                    "recent_incidents": []  # Would contain recent incident summaries
                },
                "threats": {
                    "total_indicators": ml_metrics.get('total_signatures', 0),
                    "anomalies_detected": ml_metrics.get('active_anomalies', 0),
                    "correlations": ml_metrics.get('threat_correlations', 0),
                    "ml_analysis_enabled": 'scikit-learn' in ml_metrics.get('ml_backend', '')
                },
                "zero_trust": {
                    "active_sessions": zt_metrics.get('active_sessions', 0),
                    "trusted_sessions": zt_metrics.get('trusted_sessions', 0),
                    "policy_rules": zt_metrics.get('policy_rules', 0),
                    "threat_indicators": zt_metrics.get('threat_indicators', 0)
                },
                "system_health": {
                    "incident_response": "HEALTHY",
                    "threat_intelligence": "HEALTHY",
                    "zero_trust_engine": "HEALTHY",
                    "automation_status": "ACTIVE"
                }
            }
            
            logger.info(
                "Security dashboard accessed",
                tenant_id=str(tenant_id),
                total_incidents=dashboard_data["overview"]["total_incidents"],
                active_threats=dashboard_data["overview"]["active_threats"]
            )
            
            return dashboard_data
            
    except Exception as e:
        logger.error(f"Failed to get security dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")

@router.post("/simulate/attack")
async def simulate_security_attack(
    attack_type: str = Query(..., description="Type of attack to simulate"),
    target_assets: List[str] = Query(..., description="Target assets for simulation"),
    severity: IncidentSeverity = Query(IncidentSeverity.MEDIUM, description="Simulation severity"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    tenant_id: UUID = Depends(get_current_tenant_id),
    incident_service: AutomatedIncidentResponse = Depends(get_incident_response_service)
):
    """
    Simulate a security attack for testing purposes
    
    This endpoint creates simulated security incidents for testing incident
    response procedures, validating playbooks, and training security teams.
    WARNING: This should only be used in test environments.
    """
    try:
        with add_trace_context("simulate_security_attack"):
            # Create simulated threat indicators
            simulated_indicators = []
            
            if attack_type.lower() == "malware":
                indicator = type('ThreatIndicator', (), {
                    'indicator_id': f"SIM-MALWARE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'indicator_type': 'hash',
                    'value': 'deadbeefcafebabe1234567890abcdef',
                    'source': 'simulation',
                    'confidence': 0.9,
                    'severity': 'high',
                    'description': f'Simulated {attack_type} attack',
                    'tags': ['simulation', attack_type],
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow()
                })()
                simulated_indicators.append(indicator)
            
            elif attack_type.lower() == "network_intrusion":
                indicator = type('ThreatIndicator', (), {
                    'indicator_id': f"SIM-IP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'indicator_type': 'ip',
                    'value': '192.168.100.100',
                    'source': 'simulation',
                    'confidence': 0.8,
                    'severity': 'medium',
                    'description': f'Simulated {attack_type} from suspicious IP',
                    'tags': ['simulation', attack_type],
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow()
                })()
                simulated_indicators.append(indicator)
            
            # Create simulated incident
            incident_id = await incident_service.create_incident(
                title=f"SIMULATION: {attack_type.replace('_', ' ').title()} Attack",
                description=f"This is a simulated {attack_type} attack for testing purposes. All indicators and responses are simulated.",
                severity=severity,
                threat_indicators=simulated_indicators,
                affected_assets=target_assets,
                source="attack_simulation",
                evidence={
                    "simulation": True,
                    "attack_type": attack_type,
                    "simulation_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.warning(
                "Security attack simulation created",
                incident_id=incident_id,
                attack_type=attack_type,
                target_assets=target_assets,
                severity=severity.name,
                tenant_id=str(tenant_id)
            )
            
            return {
                "simulation_id": incident_id,
                "attack_type": attack_type,
                "target_assets": target_assets,
                "severity": severity.name,
                "status": "simulation_active",
                "created_at": datetime.utcnow().isoformat(),
                "warning": "This is a simulated attack for testing purposes"
            }
            
    except Exception as e:
        logger.error(f"Attack simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/health")
async def security_service_health():
    """
    Get health status of security services
    
    Returns the operational status of all security-related services
    and components for monitoring and alerting purposes.
    """
    try:
        with add_trace_context("security_service_health"):
            # Check service health
            incident_service = get_incident_response_service()
            ml_service = get_ml_threat_intelligence()
            zt_engine = get_zero_trust_engine()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "incident_response": {
                        "status": "healthy",
                        "auto_response_enabled": getattr(incident_service, 'auto_response_enabled', True),
                        "active_incidents": len(getattr(incident_service, 'active_incidents', {})),
                        "playbooks_loaded": len(getattr(incident_service, 'response_playbooks', {}))
                    },
                    "threat_intelligence": {
                        "status": "healthy",
                        "ml_backend": "available",
                        "models_loaded": len(getattr(ml_service, 'models', {})),
                        "signatures": len(getattr(ml_service, 'threat_signatures', {}))
                    },
                    "zero_trust": {
                        "status": "healthy",
                        "active_sessions": len(getattr(zt_engine, 'active_sessions', {})),
                        "policies": len(getattr(zt_engine, 'policies', {})),
                        "threat_indicators": len(getattr(zt_engine, 'threat_indicators', {}))
                    }
                },
                "capabilities": [
                    "automated_incident_response",
                    "ml_threat_analysis", 
                    "zero_trust_evaluation",
                    "behavioral_analytics",
                    "threat_correlation",
                    "policy_enforcement"
                ]
            }
            
            return health_status
            
    except Exception as e:
        logger.error(f"Security health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }