"""
Security Monitoring API Router
Real-time security monitoring, threat detection, and incident response endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ..auth.dependencies import require_auth
from ..dependencies import get_current_organization
from ..services.security_monitoring_enhanced import (
    get_security_monitor,
    SecurityEvent,
    ThreatLevel,
    EventType,
    analyze_request_security,
    log_security_event
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["Security Monitoring"])

# Request/Response Models

class SecurityDashboardResponse(BaseModel):
    """Security monitoring dashboard data"""
    model_config = {"protected_namespaces": ()}
    
    summary: Dict[str, Any]
    event_types: Dict[str, int]
    threat_levels: Dict[str, int]
    top_source_ips: Dict[str, int]
    recent_critical_events: List[Dict[str, Any]]
    threat_intelligence_stats: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ThreatIntelligenceUpdate(BaseModel):
    """Threat intelligence update request"""
    model_config = {"protected_namespaces": ()}
    
    indicators: List[Dict[str, Any]] = Field(..., description="List of threat indicators")
    source: str = Field(..., description="Source of the intelligence")
    confidence: float = Field(ge=0, le=1, description="Confidence level")

class SecurityEventQuery(BaseModel):
    """Security event query parameters"""
    model_config = {"protected_namespaces": ()}
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    threat_levels: Optional[List[str]] = None
    source_ip: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)

class IncidentResponse(BaseModel):
    """Incident response configuration"""
    model_config = {"protected_namespaces": ()}
    
    incident_id: str
    severity: str
    status: str  # open, investigating, resolved, closed
    assigned_to: Optional[str] = None
    response_actions: List[str] = Field(default_factory=list)
    containment_measures: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

# Security Monitoring Endpoints

@router.get("/dashboard", response_model=SecurityDashboardResponse)
async def get_security_dashboard(
    current_user = Depends(require_auth),
    current_org = Depends(get_current_organization)
):
    """
    Get comprehensive security monitoring dashboard data
    
    Returns real-time security metrics including:
    - Event counts and trends
    - Threat level distribution
    - Top attacking source IPs
    - Recent critical security events
    - Threat intelligence statistics
    """
    
    try:
        monitor = await get_security_monitor()
        dashboard_data = await monitor.get_security_dashboard_data()
        
        return SecurityDashboardResponse(**dashboard_data)
        
    except Exception as e:
        logger.error(f"Failed to get security dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security dashboard data"
        )

@router.get("/events")
async def get_security_events(
    start_time: Optional[datetime] = Query(None, description="Start time for event query"),
    end_time: Optional[datetime] = Query(None, description="End time for event query"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    source_ip: Optional[str] = Query(None, description="Filter by source IP"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    current_user = Depends(require_auth)
):
    """
    Query security events with advanced filtering options
    
    Supports filtering by:
    - Time range
    - Event type (login_attempt, scan_detected, brute_force, etc.)
    - Threat level (critical, high, medium, low)
    - Source IP address
    """
    
    try:
        monitor = await get_security_monitor()
        
        # Filter events based on query parameters
        events = list(monitor.events_buffer)
        
        # Apply time filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Apply other filters
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        if threat_level:
            events = [e for e in events if e.threat_level.value == threat_level]
        if source_ip:
            events = [e for e in events if e.source_ip == source_ip]
        
        # Limit results
        events = events[-limit:]
        
        return {
            "events": [event.to_dict() for event in events],
            "total_count": len(events),
            "query_params": {
                "start_time": start_time,
                "end_time": end_time,
                "event_type": event_type,
                "threat_level": threat_level,
                "source_ip": source_ip,
                "limit": limit
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to query security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )

@router.post("/threat-intelligence")
async def update_threat_intelligence(
    update_request: ThreatIntelligenceUpdate,
    current_user = Depends(require_auth)
):
    """
    Update threat intelligence database with new indicators
    
    Accepts threat indicators including:
    - IP addresses
    - Domain names
    - File hashes
    - URLs
    
    Each indicator includes confidence level and source attribution.
    """
    
    try:
        monitor = await get_security_monitor()
        
        # Validate indicators
        valid_indicators = []
        for indicator in update_request.indicators:
            if 'indicator' in indicator and 'type' in indicator:
                # Add source and confidence from request
                indicator['source'] = update_request.source
                indicator['confidence'] = update_request.confidence
                valid_indicators.append(indicator)
        
        if not valid_indicators:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid indicators provided"
            )
        
        await monitor.update_threat_intelligence(valid_indicators)
        
        logger.info(f"Updated threat intelligence with {len(valid_indicators)} indicators from {update_request.source}")
        
        return {
            "status": "success",
            "message": f"Updated {len(valid_indicators)} threat indicators",
            "source": update_request.source,
            "indicators_processed": len(valid_indicators),
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update threat intelligence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update threat intelligence"
        )

@router.get("/threat-intelligence")
async def get_threat_intelligence(
    indicator_type: Optional[str] = Query(None, description="Filter by indicator type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(100, ge=1, le=1000),
    current_user = Depends(require_auth)
):
    """
    Retrieve threat intelligence indicators
    
    Returns current threat intelligence database with filtering options.
    """
    
    try:
        monitor = await get_security_monitor()
        
        # Get all threat intelligence
        all_intel = list(monitor.threat_intelligence.values())
        
        # Apply filters
        if indicator_type:
            all_intel = [intel for intel in all_intel if intel.indicator_type == indicator_type]
        if threat_level:
            all_intel = [intel for intel in all_intel if intel.threat_level.value == threat_level]
        if source:
            all_intel = [intel for intel in all_intel if intel.source == source]
        
        # Limit results
        limited_intel = all_intel[-limit:]
        
        return {
            "indicators": [
                {
                    "indicator": intel.indicator,
                    "type": intel.indicator_type,
                    "threat_level": intel.threat_level.value,
                    "confidence": intel.confidence,
                    "source": intel.source,
                    "description": intel.description,
                    "tags": intel.tags,
                    "first_seen": intel.first_seen.isoformat(),
                    "last_seen": intel.last_seen.isoformat()
                }
                for intel in limited_intel
            ],
            "total_indicators": len(all_intel),
            "returned_count": len(limited_intel),
            "filters_applied": {
                "indicator_type": indicator_type,
                "threat_level": threat_level,
                "source": source
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get threat intelligence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve threat intelligence"
        )

@router.post("/analyze-request")
async def analyze_request_endpoint(
    request_data: Dict[str, Any],
    current_user = Depends(require_auth)
):
    """
    Analyze a specific request for security threats
    
    Performs real-time threat analysis including:
    - Attack pattern detection
    - Threat intelligence correlation
    - Anomaly detection
    - Behavioral analysis
    """
    
    try:
        # Analyze the request
        security_event = await analyze_request_security(request_data)
        
        if security_event:
            # Log the security event
            await log_security_event(security_event)
            
            return {
                "threat_detected": True,
                "security_event": security_event.to_dict(),
                "recommendations": _get_threat_recommendations(security_event),
                "timestamp": datetime.utcnow()
            }
        else:
            return {
                "threat_detected": False,
                "message": "No security threats detected",
                "timestamp": datetime.utcnow()
            }
            
    except Exception as e:
        logger.error(f"Failed to analyze request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze request for security threats"
        )

@router.get("/statistics")
async def get_security_statistics(
    days: int = Query(7, ge=1, le=90, description="Number of days for statistics"),
    current_user = Depends(require_auth)
):
    """
    Get comprehensive security statistics and trends
    """
    
    try:
        monitor = await get_security_monitor()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Filter events for time range
        period_events = [
            e for e in monitor.events_buffer
            if start_time <= e.timestamp <= end_time
        ]
        
        # Calculate statistics
        stats = {
            "period": {
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat(),
                "days": days
            },
            "event_summary": {
                "total_events": len(period_events),
                "events_per_day": len(period_events) / days if days > 0 else 0,
                "unique_source_ips": len(set(e.source_ip for e in period_events)),
                "unique_targets": len(set(e.target for e in period_events))
            },
            "threat_breakdown": {},
            "event_type_breakdown": {},
            "top_attacking_ips": {},
            "top_targeted_endpoints": {},
            "daily_trend": {}
        }
        
        # Calculate breakdowns
        from collections import defaultdict
        
        threat_counts = defaultdict(int)
        event_type_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        target_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for event in period_events:
            threat_counts[event.threat_level.value] += 1
            event_type_counts[event.event_type.value] += 1
            ip_counts[event.source_ip] += 1
            target_counts[event.target] += 1
            day_key = event.timestamp.strftime('%Y-%m-%d')
            daily_counts[day_key] += 1
        
        stats["threat_breakdown"] = dict(threat_counts)
        stats["event_type_breakdown"] = dict(event_type_counts)
        stats["top_attacking_ips"] = dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        stats["top_targeted_endpoints"] = dict(sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        stats["daily_trend"] = dict(daily_counts)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get security statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security statistics"
        )

@router.post("/incidents")
async def create_security_incident(
    incident: IncidentResponse,
    current_user = Depends(require_auth)
):
    """
    Create a new security incident for tracking and response
    """
    
    try:
        # In a real implementation, this would store in database
        incident_data = {
            **incident.dict(),
            "created_at": datetime.utcnow(),
            "created_by": getattr(current_user, 'user_id', 'unknown'),
            "last_updated": datetime.utcnow()
        }
        
        logger.info(f"Security incident created: {incident.incident_id}")
        
        return {
            "status": "success",
            "message": "Security incident created successfully",
            "incident": incident_data,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to create security incident: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create security incident"
        )

# Helper Functions

def _get_threat_recommendations(security_event: SecurityEvent) -> List[str]:
    """Get recommendations based on security event type"""
    
    recommendations = {
        EventType.SQL_INJECTION: [
            "Implement parameterized queries",
            "Add input validation and sanitization",
            "Enable SQL injection protection in WAF",
            "Review database permissions"
        ],
        EventType.XSS_ATTEMPT: [
            "Implement output encoding",
            "Add Content Security Policy (CSP) headers",
            "Validate and sanitize user inputs",
            "Enable XSS protection in WAF"
        ],
        EventType.BRUTE_FORCE: [
            "Implement account lockout policies",
            "Add CAPTCHA after failed attempts",
            "Enable multi-factor authentication",
            "Consider IP-based blocking"
        ],
        EventType.SCAN_DETECTED: [
            "Monitor for additional scanning activity",
            "Consider blocking source IP",
            "Review exposed services",
            "Enable intrusion detection system"
        ],
        EventType.MALWARE_DETECTED: [
            "Isolate affected systems",
            "Run full antimalware scan",
            "Check for lateral movement",
            "Review backup integrity"
        ]
    }
    
    return recommendations.get(security_event.event_type, [
        "Monitor for additional suspicious activity",
        "Review security logs",
        "Consider implementing additional security controls"
    ])