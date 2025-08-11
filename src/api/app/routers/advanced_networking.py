"""
Advanced Networking API Router for XORB Platform
Principal Auditor Implementation: Enterprise networking endpoints with production capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..auth.dependencies import get_current_user, require_admin
from ..container import get_container
from ..services.advanced_network_security import AdvancedNetworkSecurityService
from ..domain.entities import User, Organization

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/networking", tags=["Advanced Networking"])


# Request/Response Models
class NetworkSegmentRequest(BaseModel):
    """Network microsegment creation request"""
    segment_name: str = Field(..., description="Name of the network segment")
    cidr_range: str = Field(..., description="CIDR range for the segment (e.g., 192.168.1.0/24)")
    security_level: str = Field(..., description="Security level: public, restricted, confidential, secret, top_secret")
    zone: str = Field(..., description="Network zone: dmz, internal, management, guest, quarantine, critical")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class AssetAddRequest(BaseModel):
    """Add asset to segment request"""
    asset_ip: str = Field(..., description="IP address of the asset")
    asset_type: str = Field(..., description="Type of asset (server, workstation, iot, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional asset metadata")


class TrafficRuleRequest(BaseModel):
    """Traffic rule creation request"""
    source_segment: str = Field(..., description="Source segment ID")
    destination_segment: str = Field(..., description="Destination segment ID")
    allowed_protocols: List[str] = Field(..., description="Allowed protocols (tcp, udp, icmp, etc.)")
    allowed_ports: List[int] = Field(..., description="Allowed ports (0 for any port)")
    conditions: Optional[Dict[str, Any]] = Field(default=None, description="Additional conditions")


class ZeroTrustAccessRequest(BaseModel):
    """Zero trust access evaluation request"""
    user_id: str = Field(..., description="User identifier")
    device_id: str = Field(..., description="Device identifier")
    resource: str = Field(..., description="Resource being accessed")
    context: Dict[str, Any] = Field(..., description="Access context (user, device, network info)")


class NetworkAssessmentRequest(BaseModel):
    """Network security assessment request"""
    target_networks: List[str] = Field(..., description="Target network ranges to assess")
    assessment_type: str = Field(default="comprehensive", description="Type of assessment: quick, comprehensive")


class SecurityPolicyRequest(BaseModel):
    """Network security policy creation request"""
    policy_name: str = Field(..., description="Name of the security policy")
    description: str = Field(..., description="Policy description")
    conditions: Dict[str, Any] = Field(..., description="Policy conditions")
    actions: List[str] = Field(..., description="Policy actions")
    priority: int = Field(default=100, description="Policy priority (lower = higher priority)")


# Networking Management Endpoints
@router.get("/status")
async def get_networking_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive networking service status"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        status = await networking_service.get_networking_status()
        
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get networking status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_networking(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Initialize advanced networking services"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        # Initialize in background
        background_tasks.add_task(networking_service.initialize)
        
        return {
            "success": True,
            "message": "Networking initialization started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize networking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Network Security Assessment
@router.post("/assessment")
async def perform_network_assessment(
    request: NetworkAssessmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Perform comprehensive network security assessment"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        # Start assessment
        assessment = await networking_service.perform_network_security_assessment(
            request.target_networks,
            request.assessment_type
        )
        
        return {
            "success": True,
            "assessment": assessment,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Network assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Microsegmentation Endpoints
@router.post("/microsegmentation/segments")
async def create_network_segment(
    request: NetworkSegmentRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Create a new network microsegment"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        result = await networking_service.create_network_microsegment(
            request.segment_name,
            request.cidr_range,
            request.security_level,
            request.zone
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "segment": result["segment"],
            "message": f"Network segment '{request.segment_name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create network segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microsegmentation/segments")
async def list_network_segments(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List all network microsegments"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        segments = networking_service.microsegmentation.segments
        
        return {
            "success": True,
            "segments": list(segments.values()),
            "total_segments": len(segments)
        }
        
    except Exception as e:
        logger.error(f"Failed to list network segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microsegmentation/segments/{segment_id}")
async def get_network_segment(
    segment_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a network segment"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        if segment_id not in networking_service.microsegmentation.segments:
            raise HTTPException(status_code=404, detail="Network segment not found")
        
        segment = networking_service.microsegmentation.segments[segment_id]
        analytics = await networking_service.microsegmentation.get_segment_analytics(segment_id)
        
        return {
            "success": True,
            "segment": segment,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get network segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/microsegmentation/segments/{segment_id}/assets")
async def add_asset_to_segment(
    segment_id: str,
    request: AssetAddRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Add an asset to a network segment"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        result = await networking_service.microsegmentation.add_asset_to_segment(
            segment_id,
            request.asset_ip,
            request.asset_type,
            request.metadata
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "asset": result["asset"],
            "message": f"Asset {request.asset_ip} added to segment {segment_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add asset to segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/microsegmentation/traffic-rules")
async def create_traffic_rule(
    request: TrafficRuleRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Create traffic rule between network segments"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        # Convert protocol strings to enum objects
        from ..infrastructure.advanced_networking import NetworkProtocol
        protocols = [NetworkProtocol(p.lower()) for p in request.allowed_protocols]
        
        result = await networking_service.microsegmentation.create_traffic_rule(
            request.source_segment,
            request.destination_segment,
            protocols,
            request.allowed_ports,
            request.conditions
        )
        
        return {
            "success": True,
            "rule": result["rule"],
            "message": "Traffic rule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create traffic rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Zero Trust Network Access
@router.post("/zero-trust/evaluate-access")
async def evaluate_zero_trust_access(
    request: ZeroTrustAccessRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Evaluate access request using zero trust principles"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        evaluation = await networking_service.evaluate_zero_trust_access(
            request.user_id,
            request.device_id,
            request.resource,
            request.context
        )
        
        return {
            "success": True,
            "evaluation": evaluation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Zero trust evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/zero-trust/policies")
async def list_zero_trust_policies(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List zero trust access policies"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        policies = networking_service.zero_trust.access_policies
        
        return {
            "success": True,
            "policies": [
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "description": policy.description,
                    "enabled": policy.enabled,
                    "priority": policy.priority,
                    "created_at": policy.created_at.isoformat()
                }
                for policy in policies.values()
            ],
            "total_policies": len(policies)
        }
        
    except Exception as e:
        logger.error(f"Failed to list zero trust policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/zero-trust/sessions")
async def list_active_sessions(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List active zero trust sessions"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        sessions = networking_service.zero_trust.access_sessions
        active_sessions = {k: v for k, v in sessions.items() if v["active"]}
        
        return {
            "success": True,
            "sessions": list(active_sessions.values()),
            "total_active_sessions": len(active_sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/zero-trust/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Revoke zero trust access session"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        success = await networking_service.zero_trust.revoke_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "message": f"Session {session_id} revoked successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Network Security Monitoring
@router.get("/security/alerts")
async def get_security_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get network security alerts"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        # Mock organization for interface compatibility
        org = Organization(id=current_user.id, name="default", plan_type="enterprise")
        
        alerts = await networking_service.get_security_alerts(
            org, severity, limit
        )
        
        return {
            "success": True,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "filters": {
                "severity": severity,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get security alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/policies")
async def create_security_policy(
    request: SecurityPolicyRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Create network security policy"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        # Mock organization for interface compatibility
        org = Organization(id=current_user.id, name="default", plan_type="enterprise")
        
        rule_definition = {
            "name": request.policy_name,
            "description": request.description,
            "conditions": request.conditions,
            "actions": request.actions,
            "priority": request.priority
        }
        
        result = await networking_service.create_alert_rule(
            rule_definition, org, current_user
        )
        
        return {
            "success": True,
            "policy": result["policy"],
            "rule_id": result["rule_id"],
            "message": f"Security policy '{request.policy_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create security policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/investigate/{incident_id}")
async def investigate_incident(
    incident_id: str,
    investigation_params: Dict[str, Any],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Investigate network security incident"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        investigation = await networking_service.investigate_incident(
            incident_id, investigation_params, current_user
        )
        
        if not investigation["success"]:
            raise HTTPException(status_code=404, detail=investigation["error"])
        
        return {
            "success": True,
            "investigation": investigation["investigation"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to investigate incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Network Analytics and Reporting
@router.get("/analytics/microsegmentation")
async def get_microsegmentation_analytics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get microsegmentation analytics"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        analysis = await networking_service._analyze_microsegmentation()
        
        return {
            "success": True,
            "analytics": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get microsegmentation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/zero-trust")
async def get_zero_trust_analytics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get zero trust posture analytics"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        analysis = await networking_service._analyze_zero_trust_posture()
        
        return {
            "success": True,
            "analytics": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get zero trust analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/threat-landscape")
async def get_threat_landscape_analytics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get threat landscape analytics"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        analysis = await networking_service._analyze_threat_landscape()
        
        return {
            "success": True,
            "analytics": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get threat landscape analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Firewall Management
@router.get("/firewall/status")
async def get_firewall_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get firewall status and statistics"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        firewall_manager = networking_service.enterprise_networking.firewall_manager
        status = await firewall_manager.get_firewall_status()
        
        return {
            "success": True,
            "firewall_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get firewall status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/firewall/rules")
async def create_firewall_rule(
    rule_name: str,
    source: str,
    destination: str,
    ports: List[int],
    protocol: str,
    action: str,
    priority: int = 100,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Create firewall rule"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        firewall_manager = networking_service.enterprise_networking.firewall_manager
        result = await firewall_manager.create_firewall_rule(
            rule_name, source, destination, ports, protocol, action, priority
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "rule": result["rule"],
            "message": f"Firewall rule '{rule_name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create firewall rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/firewall/evaluate")
async def evaluate_connection(
    source_ip: str,
    dest_ip: str,
    dest_port: int,
    protocol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Evaluate connection against firewall rules"""
    try:
        container = get_container()
        networking_service = container.get(AdvancedNetworkSecurityService)
        
        firewall_manager = networking_service.enterprise_networking.firewall_manager
        evaluation = await firewall_manager.evaluate_connection(
            source_ip, dest_ip, dest_port, protocol
        )
        
        return {
            "success": True,
            "evaluation": evaluation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))