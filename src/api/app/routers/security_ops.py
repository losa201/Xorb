"""
XORB Security Operations API Endpoints
Provides security monitoring, threat detection, and incident response capabilities
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..security import (
    SecurityContext,
    get_security_context,
    require_security_ops,
    require_permission,
    Permission
)


class ThreatSeverity(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatStatus(str, Enum):
    """Threat handling status"""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class ThreatCategory(str, Enum):
    """Threat categories"""
    MALWARE = "malware"
    PHISHING = "phishing"
    NETWORK_INTRUSION = "network_intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    RECONNAISSANCE = "reconnaissance"
    DENIAL_OF_SERVICE = "denial_of_service"
    POLICY_VIOLATION = "policy_violation"


class ResponseAction(str, Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    QUARANTINE_HOST = "quarantine_host"
    ISOLATE_NETWORK = "isolate_network"
    DISABLE_ACCOUNT = "disable_account"
    COLLECT_FORENSICS = "collect_forensics"
    ALERT_TEAM = "alert_team"
    UPDATE_RULES = "update_rules"
    PATCH_SYSTEM = "patch_system"


class ComplianceStandard(str, Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    NIST_CSF = "nist_csf"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"


# Pydantic Models
class ThreatIndicator(BaseModel):
    """Threat indicator information"""
    indicator_type: str  # ip, hash, domain, url, etc.
    value: str
    confidence: float = Field(ge=0, le=1)
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str] = Field(default_factory=list)


class Threat(BaseModel):
    """Threat detection information"""
    id: str
    name: str
    description: str
    severity: ThreatSeverity
    status: ThreatStatus
    category: ThreatCategory
    
    # Detection details
    detected_at: datetime
    source_system: str
    detection_rule: Optional[str] = None
    
    # Affected assets
    affected_hosts: List[str] = Field(default_factory=list)
    affected_users: List[str] = Field(default_factory=list)
    affected_networks: List[str] = Field(default_factory=list)
    
    # Threat intelligence
    indicators: List[ThreatIndicator] = Field(default_factory=list)
    tactics: List[str] = Field(default_factory=list)  # MITRE ATT&CK tactics
    techniques: List[str] = Field(default_factory=list)  # MITRE ATT&CK techniques
    
    # Investigation
    assigned_analyst: Optional[str] = None
    investigation_notes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Response
    response_actions: List[str] = Field(default_factory=list)
    containment_status: Optional[str] = None
    
    # Metadata
    created_by: str = "system"
    updated_at: datetime
    tags: Dict[str, str] = Field(default_factory=dict)


class CreateThreatRequest(BaseModel):
    """Request to create a new threat"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    severity: ThreatSeverity
    category: ThreatCategory
    source_system: str
    detection_rule: Optional[str] = None
    affected_hosts: List[str] = Field(default_factory=list)
    affected_users: List[str] = Field(default_factory=list)
    indicators: List[ThreatIndicator] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)


class ThreatResponseRequest(BaseModel):
    """Request to respond to a threat"""
    actions: List[ResponseAction]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    auto_execute: bool = False
    notify_team: bool = True


class ThreatResponseResult(BaseModel):
    """Result of threat response"""
    response_id: str
    threat_id: str
    actions_taken: List[Dict[str, Any]]
    status: str
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None


class SecurityEvent(BaseModel):
    """Security event information"""
    id: str
    event_type: str
    source: str
    timestamp: datetime
    severity: str
    description: str
    details: Dict[str, Any]
    correlated_threats: List[str] = Field(default_factory=list)


class ComplianceCheck(BaseModel):
    """Compliance check result"""
    id: str
    standard: ComplianceStandard
    control_id: str
    control_name: str
    status: str  # compliant, non_compliant, not_applicable
    score: Optional[float] = Field(None, ge=0, le=100)
    findings: List[str] = Field(default_factory=list)
    remediation_actions: List[str] = Field(default_factory=list)
    checked_at: datetime
    next_check: Optional[datetime] = None


class SecurityMetrics(BaseModel):
    """Security operation metrics"""
    threats_detected_today: int
    threats_resolved_today: int
    average_response_time_minutes: float
    compliance_score: float
    active_investigations: int
    blocked_attacks: int
    false_positive_rate: float
    system_health_score: float


# In-memory storage (replace with database in production)
threats_store: Dict[str, Threat] = {}
security_events_store: Dict[str, SecurityEvent] = {}
compliance_checks_store: Dict[str, ComplianceCheck] = {}
response_actions_store: Dict[str, ThreatResponseResult] = {}


router = APIRouter(prefix="/security", tags=["Security Operations"])


@router.get("/threats")
async def list_threats(
    context: SecurityContext = Depends(get_security_context),
    severity: Optional[ThreatSeverity] = Query(None, description="Filter by severity"),
    status: Optional[ThreatStatus] = Query(None, description="Filter by status"),
    category: Optional[ThreatCategory] = Query(None, description="Filter by category"),
    assigned_analyst: Optional[str] = Query(None, description="Filter by assigned analyst"),
    hours_back: int = Query(24, ge=1, le=8760, description="Hours to look back"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
) -> Dict[str, Any]:
    """List security threats with filtering"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    
    # Filter threats
    filtered_threats = []
    for threat in threats_store.values():
        # Time filter
        if threat.detected_at < cutoff_time:
            continue
            
        # Apply other filters
        if severity and threat.severity != severity:
            continue
        if status and threat.status != status:
            continue
        if category and threat.category != category:
            continue
        if assigned_analyst and threat.assigned_analyst != assigned_analyst:
            continue
            
        filtered_threats.append(threat)
    
    # Sort by severity and detection time
    filtered_threats.sort(key=lambda t: (
        _severity_value(t.severity),
        t.detected_at
    ), reverse=True)
    
    # Apply limit
    if len(filtered_threats) > limit:
        filtered_threats = filtered_threats[:limit]
    
    return {
        "threats": filtered_threats,
        "total": len(filtered_threats),
        "filtered_from": len(threats_store),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/threats", response_model=Threat, status_code=201)
async def create_threat(
    request: CreateThreatRequest,
    context: SecurityContext = Depends(require_security_ops)
) -> Threat:
    """Create a new security threat"""
    
    threat_id = str(uuid.uuid4())
    current_time = datetime.utcnow()
    
    threat = Threat(
        id=threat_id,
        name=request.name,
        description=request.description,
        severity=request.severity,
        status=ThreatStatus.NEW,
        category=request.category,
        detected_at=current_time,
        source_system=request.source_system,
        detection_rule=request.detection_rule,
        affected_hosts=request.affected_hosts,
        affected_users=request.affected_users,
        indicators=request.indicators,
        created_by=context.user_id,
        updated_at=current_time,
        tags=request.tags
    )
    
    # Add initial investigation note
    threat.investigation_notes.append({
        "timestamp": current_time.isoformat(),
        "author": context.user_id,
        "note": f"Threat created from {request.source_system}",
        "type": "system"
    })
    
    threats_store[threat_id] = threat
    
    # Auto-assign high severity threats
    if request.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
        threat.assigned_analyst = context.user_id
        threat.investigation_notes.append({
            "timestamp": current_time.isoformat(),
            "author": "system",
            "note": f"Auto-assigned to {context.user_id} due to {request.severity.value} severity",
            "type": "assignment"
        })
    
    return threat


@router.get("/threats/{threat_id}", response_model=Threat)
async def get_threat(
    threat_id: str,
    context: SecurityContext = Depends(get_security_context)
) -> Threat:
    """Get detailed threat information"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    threat = threats_store.get(threat_id)
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    return threat


@router.put("/threats/{threat_id}/status")
async def update_threat_status(
    threat_id: str,
    status: ThreatStatus,
    notes: Optional[str] = None,
    context: SecurityContext = Depends(require_security_ops)
) -> Threat:
    """Update threat status"""
    
    threat = threats_store.get(threat_id)
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    old_status = threat.status
    threat.status = status
    threat.updated_at = datetime.utcnow()
    
    # Add investigation note
    note_text = f"Status changed from {old_status.value} to {status.value}"
    if notes:
        note_text += f"\nNotes: {notes}"
    
    threat.investigation_notes.append({
        "timestamp": datetime.utcnow().isoformat(),
        "author": context.user_id,
        "note": note_text,
        "type": "status_update"
    })
    
    return threat


@router.post("/threats/{threat_id}/respond", response_model=ThreatResponseResult)
async def respond_to_threat(
    threat_id: str,
    request: ThreatResponseRequest,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_security_ops)
) -> ThreatResponseResult:
    """Execute automated threat response"""
    
    threat = threats_store.get(threat_id)
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    response_id = str(uuid.uuid4())
    current_time = datetime.utcnow()
    
    # Create response result
    response_result = ThreatResponseResult(
        response_id=response_id,
        threat_id=threat_id,
        actions_taken=[],
        status="initiated",
        initiated_at=current_time
    )
    
    # Store response
    response_actions_store[response_id] = response_result
    
    # Add to threat's response actions
    for action in request.actions:
        threat.response_actions.append(action.value)
    
    # Add investigation note
    threat.investigation_notes.append({
        "timestamp": current_time.isoformat(),
        "author": context.user_id,
        "note": f"Response initiated: {', '.join([a.value for a in request.actions])}",
        "type": "response"
    })
    
    threat.updated_at = current_time
    
    # Execute response actions
    background_tasks.add_task(_execute_threat_response, response_id, request)
    
    return response_result


@router.get("/threats/{threat_id}/timeline")
async def get_threat_timeline(
    threat_id: str,
    context: SecurityContext = Depends(get_security_context)
) -> Dict[str, Any]:
    """Get threat investigation timeline"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    threat = threats_store.get(threat_id)
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    # Build timeline from investigation notes and related events
    timeline_events = []
    
    # Add threat creation
    timeline_events.append({
        "timestamp": threat.detected_at.isoformat(),
        "event_type": "threat_detected",
        "description": f"Threat detected: {threat.name}",
        "severity": threat.severity.value,
        "source": "detection_system"
    })
    
    # Add investigation notes
    for note in threat.investigation_notes:
        timeline_events.append({
            "timestamp": note["timestamp"],
            "event_type": "investigation_note",
            "description": note["note"],
            "author": note.get("author", "system"),
            "note_type": note.get("type", "general")
        })
    
    # Sort by timestamp
    timeline_events.sort(key=lambda x: x["timestamp"])
    
    return {
        "threat_id": threat_id,
        "timeline": timeline_events,
        "total_events": len(timeline_events),
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/events")
async def list_security_events(
    context: SecurityContext = Depends(get_security_context),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source: Optional[str] = Query(None, description="Filter by source"),
    hours_back: int = Query(1, ge=1, le=168, description="Hours to look back"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum results"),
) -> Dict[str, Any]:
    """List security events"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    
    # Filter events
    filtered_events = []
    for event in security_events_store.values():
        if event.timestamp < cutoff_time:
            continue
            
        if event_type and event.event_type != event_type:
            continue
        if severity and event.severity != severity:
            continue
        if source and event.source != source:
            continue
            
        filtered_events.append(event)
    
    # Sort by timestamp (newest first)
    filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
    
    # Apply limit
    if len(filtered_events) > limit:
        filtered_events = filtered_events[:limit]
    
    return {
        "events": filtered_events,
        "total": len(filtered_events),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/compliance")
async def get_compliance_status(
    context: SecurityContext = Depends(get_security_context),
    standard: Optional[ComplianceStandard] = Query(None, description="Filter by standard"),
    status: Optional[str] = Query(None, description="Filter by compliance status"),
) -> Dict[str, Any]:
    """Get compliance status and checks"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    # Filter compliance checks
    filtered_checks = []
    for check in compliance_checks_store.values():
        if standard and check.standard != standard:
            continue
        if status and check.status != status:
            continue
            
        filtered_checks.append(check)
    
    # Calculate overall compliance scores by standard
    compliance_scores = {}
    for std in ComplianceStandard:
        std_checks = [c for c in filtered_checks if c.standard == std]
        if std_checks:
            scores = [c.score for c in std_checks if c.score is not None]
            if scores:
                compliance_scores[std.value] = sum(scores) / len(scores)
            else:
                compliance_scores[std.value] = 0.0
    
    return {
        "compliance_checks": filtered_checks,
        "compliance_scores": compliance_scores,
        "total_checks": len(filtered_checks),
        "last_updated": datetime.utcnow().isoformat()
    }


@router.get("/metrics", response_model=SecurityMetrics)
async def get_security_metrics(
    context: SecurityContext = Depends(get_security_context)
) -> SecurityMetrics:
    """Get security operation metrics"""
    
    if Permission.SECURITY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: security:read")
    
    current_time = datetime.utcnow()
    today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate metrics
    threats_today = [
        t for t in threats_store.values() 
        if t.detected_at >= today_start
    ]
    
    resolved_today = [
        t for t in threats_today 
        if t.status == ThreatStatus.RESOLVED
    ]
    
    active_investigations = [
        t for t in threats_store.values() 
        if t.status in [ThreatStatus.INVESTIGATING, ThreatStatus.CONTAINED]
    ]
    
    # Calculate average response time
    completed_responses = [
        r for r in response_actions_store.values()
        if r.completed_at is not None
    ]
    
    if completed_responses:
        total_response_time = sum(
            (r.completed_at - r.initiated_at).total_seconds() / 60
            for r in completed_responses
        )
        avg_response_time = total_response_time / len(completed_responses)
    else:
        avg_response_time = 0.0
    
    # Simulate other metrics
    compliance_score = 85.5  # This would come from compliance checks
    blocked_attacks = len([t for t in threats_today if t.status == ThreatStatus.CONTAINED])
    false_positive_rate = 0.05  # 5%
    system_health_score = 92.3
    
    return SecurityMetrics(
        threats_detected_today=len(threats_today),
        threats_resolved_today=len(resolved_today),
        average_response_time_minutes=avg_response_time,
        compliance_score=compliance_score,
        active_investigations=len(active_investigations),
        blocked_attacks=blocked_attacks,
        false_positive_rate=false_positive_rate,
        system_health_score=system_health_score
    )


@router.post("/alerts")
async def create_security_alert(
    alert_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_security_ops)
) -> Dict[str, str]:
    """Create a security alert and potential threat"""
    
    alert_id = str(uuid.uuid4())
    
    # Process alert and potentially create threat
    background_tasks.add_task(_process_security_alert, alert_id, alert_data, context.user_id)
    
    return {
        "alert_id": alert_id,
        "status": "processing",
        "message": "Security alert received and processing initiated"
    }


# Helper functions
def _severity_value(severity: ThreatSeverity) -> int:
    """Convert severity to numeric value for sorting"""
    values = {
        ThreatSeverity.CRITICAL: 4,
        ThreatSeverity.HIGH: 3,
        ThreatSeverity.MEDIUM: 2,
        ThreatSeverity.LOW: 1
    }
    return values.get(severity, 1)


async def _execute_threat_response(response_id: str, request: ThreatResponseRequest):
    """Execute threat response actions"""
    response = response_actions_store.get(response_id)
    if not response:
        return
    
    try:
        response.status = "executing"
        
        # Execute each action
        for action in request.actions:
            action_result = await _execute_response_action(action, request.parameters)
            response.actions_taken.append(action_result)
            
            # Add delay between actions
            await asyncio.sleep(1)
        
        response.status = "completed"
        response.completed_at = datetime.utcnow()
        response.effectiveness_score = 0.85  # Simulate effectiveness calculation
        
    except Exception as e:
        response.status = "failed"
        response.actions_taken.append({
            "action": "error",
            "result": f"Execution failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })


async def _execute_response_action(action: ResponseAction, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single response action"""
    current_time = datetime.utcnow()
    
    # Simulate action execution
    await asyncio.sleep(2)
    
    action_results = {
        ResponseAction.BLOCK_IP: {
            "action": "block_ip",
            "result": f"Blocked IP {parameters.get('ip', 'unknown')} on firewall",
            "success": True
        },
        ResponseAction.QUARANTINE_HOST: {
            "action": "quarantine_host",
            "result": f"Quarantined host {parameters.get('host', 'unknown')}",
            "success": True
        },
        ResponseAction.ISOLATE_NETWORK: {
            "action": "isolate_network",
            "result": f"Isolated network segment {parameters.get('network', 'unknown')}",
            "success": True
        },
        ResponseAction.DISABLE_ACCOUNT: {
            "action": "disable_account",
            "result": f"Disabled account {parameters.get('account', 'unknown')}",
            "success": True
        },
        ResponseAction.COLLECT_FORENSICS: {
            "action": "collect_forensics",
            "result": "Initiated forensic data collection",
            "success": True,
            "artifacts": ["memory_dump.raw", "network_traffic.pcap", "system_logs.zip"]
        }
    }
    
    result = action_results.get(action, {
        "action": action.value,
        "result": f"Executed {action.value}",
        "success": True
    })
    
    result["timestamp"] = current_time.isoformat()
    return result


async def _process_security_alert(alert_id: str, alert_data: Dict[str, Any], user_id: str):
    """Process incoming security alert"""
    try:
        # Analyze alert data and determine if threat creation is needed
        severity = alert_data.get("severity", "medium")
        confidence = alert_data.get("confidence", 0.7)
        
        # Create threat if confidence is high enough
        if confidence > 0.6:
            threat_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            # Map alert severity to threat severity
            severity_mapping = {
                "low": ThreatSeverity.LOW,
                "medium": ThreatSeverity.MEDIUM,
                "high": ThreatSeverity.HIGH,
                "critical": ThreatSeverity.CRITICAL
            }
            
            threat = Threat(
                id=threat_id,
                name=alert_data.get("name", f"Alert {alert_id}"),
                description=alert_data.get("description", "Auto-generated from security alert"),
                severity=severity_mapping.get(severity, ThreatSeverity.MEDIUM),
                status=ThreatStatus.NEW,
                category=ThreatCategory.NETWORK_INTRUSION,  # Default category
                detected_at=current_time,
                source_system=alert_data.get("source", "alert_system"),
                created_by=user_id,
                updated_at=current_time,
                tags={"alert_id": alert_id, "auto_generated": "true"}
            )
            
            # Add initial investigation note
            threat.investigation_notes.append({
                "timestamp": current_time.isoformat(),
                "author": "system",
                "note": f"Threat auto-created from alert {alert_id} with {confidence:.2f} confidence",
                "type": "system"
            })
            
            threats_store[threat_id] = threat
            
    except Exception as e:
        # Log error but don't fail
        pass


# Initialize sample data
async def _initialize_sample_security_data():
    """Initialize sample security data for testing"""
    if not threats_store:  # Only create if empty
        current_time = datetime.utcnow()
        
        # Sample threats
        sample_threats = [
            {
                "name": "Suspected Malware Activity",
                "description": "Unusual network connections detected from workstation WS-001",
                "severity": ThreatSeverity.HIGH,
                "category": ThreatCategory.MALWARE,
                "source_system": "edr_system",
                "affected_hosts": ["WS-001"],
                "indicators": [
                    ThreatIndicator(
                        indicator_type="ip",
                        value="192.168.1.100",
                        confidence=0.85,
                        source="internal_monitoring",
                        first_seen=current_time - timedelta(hours=2),
                        last_seen=current_time - timedelta(minutes=30),
                        tags=["internal", "suspicious"]
                    )
                ]
            },
            {
                "name": "Failed Login Attempts",
                "description": "Multiple failed login attempts from external IP",
                "severity": ThreatSeverity.MEDIUM,
                "category": ThreatCategory.RECONNAISSANCE,
                "source_system": "auth_system",
                "affected_users": ["admin", "service_account"],
                "indicators": [
                    ThreatIndicator(
                        indicator_type="ip",
                        value="203.0.113.45",
                        confidence=0.75,
                        source="firewall_logs",
                        first_seen=current_time - timedelta(hours=1),
                        last_seen=current_time - timedelta(minutes=5),
                        tags=["external", "brute_force"]
                    )
                ]
            }
        ]
        
        for i, threat_data in enumerate(sample_threats):
            threat_id = str(uuid.uuid4())
            
            threat = Threat(
                id=threat_id,
                name=threat_data["name"],
                description=threat_data["description"],
                severity=threat_data["severity"],
                status=ThreatStatus.INVESTIGATING if i == 0 else ThreatStatus.NEW,
                category=threat_data["category"],
                detected_at=current_time - timedelta(hours=2-i),
                source_system=threat_data["source_system"],
                affected_hosts=threat_data.get("affected_hosts", []),
                affected_users=threat_data.get("affected_users", []),
                indicators=threat_data.get("indicators", []),
                created_by="system",
                updated_at=current_time,
                investigation_notes=[
                    {
                        "timestamp": (current_time - timedelta(hours=2-i)).isoformat(),
                        "author": "system",
                        "note": f"Threat detected by {threat_data['source_system']}",
                        "type": "detection"
                    }
                ]
            )
            
            if i == 0:  # Assign first threat to analyst
                threat.assigned_analyst = "analyst_001"
                threat.investigation_notes.append({
                    "timestamp": (current_time - timedelta(hours=1)).isoformat(),
                    "author": "analyst_001",
                    "note": "Beginning investigation of network anomaly",
                    "type": "investigation"
                })
            
            threats_store[threat_id] = threat


# Note: Sample data initialization handled during startup