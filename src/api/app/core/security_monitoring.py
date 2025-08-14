"""
Security Monitoring and Alerting System
Production-grade security monitoring for tenant isolation and SQL injection prevention
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque

from ..core.secure_tenant_context import TenantSecurityEvent, TenantContextViolationType
from ..core.logging import get_logger

logger = get_logger(__name__)


class SecurityThreatLevel(str, Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventCategory(str, Enum):
    """Categories of security events"""
    TENANT_ISOLATION = "tenant_isolation"
    SQL_INJECTION = "sql_injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM_SECURITY = "system_security"


@dataclass
class SecurityAlert:
    """Security alert with threat information"""
    alert_id: str
    event_category: SecurityEventCategory
    threat_level: SecurityThreatLevel
    title: str
    description: str
    affected_resources: List[str]
    source_events: List[Dict[str, Any]]
    first_seen: datetime
    last_seen: datetime
    occurrences: int
    ip_addresses: Set[str] = field(default_factory=set)
    user_ids: Set[str] = field(default_factory=set)
    tenant_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['ip_addresses'] = list(self.ip_addresses)
        data['user_ids'] = list(self.user_ids)
        data['tenant_ids'] = list(self.tenant_ids)
        return data


@dataclass
class SecurityMetrics:
    """Security metrics and statistics"""
    total_events: int = 0
    events_by_category: Dict[SecurityEventCategory, int] = field(default_factory=dict)
    events_by_threat_level: Dict[SecurityThreatLevel, int] = field(default_factory=dict)
    unique_attackers: int = 0
    blocked_attacks: int = 0
    escalated_alerts: int = 0
    response_time_avg: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class SecurityMonitor:
    """
    Production-grade security monitoring and alerting system
    
    Features:
    - Real-time security event processing
    - Intelligent alert correlation and deduplication
    - Threat pattern detection and scoring
    - Automated response triggers
    - Security metrics and reporting
    - Integration with external security tools
    """
    
    def __init__(
        self,
        alert_webhook_url: Optional[str] = None,
        email_alerts: Optional[List[str]] = None,
        retention_days: int = 90
    ):
        self.alert_webhook_url = alert_webhook_url
        self.email_alerts = email_alerts or []
        self.retention_days = retention_days
        
        # Event storage (in production, use persistent storage)
        self.events: deque = deque(maxlen=10000)
        self.alerts: Dict[str, SecurityAlert] = {}
        self.metrics = SecurityMetrics()
        
        # Threat detection configuration
        self.threat_patterns = {
            # Tenant isolation violations
            TenantContextViolationType.UNAUTHORIZED_ACCESS: {
                "threshold": 3,
                "window_minutes": 15,
                "threat_level": SecurityThreatLevel.HIGH,
                "category": SecurityEventCategory.TENANT_ISOLATION
            },
            TenantContextViolationType.CROSS_TENANT_ATTEMPT: {
                "threshold": 2,
                "window_minutes": 10,
                "threat_level": SecurityThreatLevel.CRITICAL,
                "category": SecurityEventCategory.TENANT_ISOLATION
            },
            TenantContextViolationType.HEADER_MANIPULATION: {
                "threshold": 1,
                "window_minutes": 5,
                "threat_level": SecurityThreatLevel.HIGH,
                "category": SecurityEventCategory.TENANT_ISOLATION
            },
            TenantContextViolationType.RLS_FAILURE: {
                "threshold": 1,
                "window_minutes": 1,
                "threat_level": SecurityThreatLevel.CRITICAL,
                "category": SecurityEventCategory.SYSTEM_SECURITY
            }
        }
        
        # Known attack patterns
        self.attack_patterns = {
            "sql_injection": {
                "patterns": [
                    r"union\s+select",
                    r"or\s+1\s*=\s*1",
                    r"drop\s+table",
                    r"exec\s+xp_",
                    r"information_schema",
                ],
                "threat_level": SecurityThreatLevel.CRITICAL,
                "category": SecurityEventCategory.SQL_INJECTION
            },
            "tenant_switching": {
                "patterns": [
                    r"x-tenant-id",
                    r"tenant-id",
                    r"organization-id"
                ],
                "threat_level": SecurityThreatLevel.HIGH,
                "category": SecurityEventCategory.TENANT_ISOLATION
            }
        }
        
        # Suspicious IP tracking
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Rate limiting for alerts
        self.alert_rate_limits: Dict[str, datetime] = {}
        self.min_alert_interval = timedelta(minutes=5)
        
        logger.info("Security monitor initialized")
    
    async def process_security_event(self, event: TenantSecurityEvent) -> None:
        """
        Process security event and generate alerts if needed
        
        Args:
            event: Security event to process
        """
        try:
            # Store event
            self.events.append(event)
            self.metrics.total_events += 1
            
            # Update metrics
            await self._update_metrics(event)
            
            # Analyze threat level
            threat_level = await self._analyze_threat_level(event)
            
            # Check for attack patterns
            await self._detect_attack_patterns(event)
            
            # Update suspicious IP tracking
            if event.ip_address:
                await self._track_suspicious_ip(event.ip_address, event)
            
            # Generate alerts if thresholds exceeded
            await self._check_alert_thresholds(event, threat_level)
            
            # Log security event
            await self._log_security_event(event, threat_level)
            
        except Exception as e:
            logger.error(f"Error processing security event: {e}")
    
    async def _update_metrics(self, event: TenantSecurityEvent) -> None:
        """Update security metrics"""
        # Determine category
        if event.violation_type in [
            TenantContextViolationType.UNAUTHORIZED_ACCESS,
            TenantContextViolationType.CROSS_TENANT_ATTEMPT,
            TenantContextViolationType.HEADER_MANIPULATION
        ]:
            category = SecurityEventCategory.TENANT_ISOLATION
        elif event.violation_type == TenantContextViolationType.RLS_FAILURE:
            category = SecurityEventCategory.SYSTEM_SECURITY
        else:
            category = SecurityEventCategory.DATA_ACCESS
        
        # Update category counts
        if category not in self.metrics.events_by_category:
            self.metrics.events_by_category[category] = 0
        self.metrics.events_by_category[category] += 1
        
        # Update unique attackers
        if event.ip_address and event.ip_address not in self.suspicious_ips:
            self.metrics.unique_attackers += 1
        
        self.metrics.last_updated = datetime.utcnow()
    
    async def _analyze_threat_level(self, event: TenantSecurityEvent) -> SecurityThreatLevel:
        """Analyze threat level of security event"""
        config = self.threat_patterns.get(event.violation_type)
        if not config:
            return SecurityThreatLevel.MEDIUM
        
        # Check for escalation factors
        threat_level = config["threat_level"]
        
        # Escalate if repeat offender
        if event.ip_address in self.suspicious_ips:
            ip_data = self.suspicious_ips[event.ip_address]
            if ip_data.get("violations", 0) > 5:
                threat_level = SecurityThreatLevel.CRITICAL
        
        # Escalate if multiple violation types
        if event.ip_address:
            recent_events = [
                e for e in self.events 
                if e.ip_address == event.ip_address 
                and (datetime.utcnow() - e.timestamp) < timedelta(hours=1)
            ]
            violation_types = set(e.violation_type for e in recent_events)
            if len(violation_types) > 2:
                threat_level = SecurityThreatLevel.CRITICAL
        
        return threat_level
    
    async def _detect_attack_patterns(self, event: TenantSecurityEvent) -> None:
        """Detect known attack patterns in event data"""
        event_data = json.dumps(event.details).lower()
        
        for pattern_name, pattern_config in self.attack_patterns.items():
            for pattern in pattern_config["patterns"]:
                if pattern in event_data:
                    await self._create_pattern_alert(
                        event, pattern_name, pattern_config
                    )
                    break
    
    async def _track_suspicious_ip(self, ip_address: str, event: TenantSecurityEvent) -> None:
        """Track suspicious IP activity"""
        if ip_address not in self.suspicious_ips:
            self.suspicious_ips[ip_address] = {
                "first_seen": event.timestamp,
                "violations": 0,
                "violation_types": set(),
                "targeted_tenants": set(),
                "user_agents": set()
            }
        
        ip_data = self.suspicious_ips[ip_address]
        ip_data["violations"] += 1
        ip_data["violation_types"].add(event.violation_type.value)
        ip_data["last_seen"] = event.timestamp
        
        if event.tenant_id:
            ip_data["targeted_tenants"].add(str(event.tenant_id))
        
        if event.details.get("user_agent"):
            ip_data["user_agents"].add(event.details["user_agent"])
        
        # Auto-block if threshold exceeded
        if ip_data["violations"] > 10:
            await self._auto_block_ip(ip_address, ip_data)
    
    async def _auto_block_ip(self, ip_address: str, ip_data: Dict[str, Any]) -> None:
        """Automatically block suspicious IP"""
        if ip_address not in self.blocked_ips:
            self.blocked_ips.add(ip_address)
            
            # Create critical alert
            alert = SecurityAlert(
                alert_id=f"auto_block_{ip_address}_{int(datetime.utcnow().timestamp())}",
                event_category=SecurityEventCategory.SYSTEM_SECURITY,
                threat_level=SecurityThreatLevel.CRITICAL,
                title=f"IP Address Auto-Blocked: {ip_address}",
                description=f"IP {ip_address} has been automatically blocked due to {ip_data['violations']} security violations",
                affected_resources=[ip_address],
                source_events=[],
                first_seen=ip_data["first_seen"],
                last_seen=ip_data["last_seen"],
                occurrences=ip_data["violations"],
                ip_addresses={ip_address},
                metadata={
                    "violation_types": list(ip_data["violation_types"]),
                    "targeted_tenants": list(ip_data["targeted_tenants"]),
                    "user_agents": list(ip_data["user_agents"])
                }
            )
            
            await self._send_alert(alert)
            
            logger.critical(f"Auto-blocked suspicious IP: {ip_address}")
    
    async def _check_alert_thresholds(
        self, 
        event: TenantSecurityEvent, 
        threat_level: SecurityThreatLevel
    ) -> None:
        """Check if alert thresholds are exceeded"""
        config = self.threat_patterns.get(event.violation_type)
        if not config:
            return
        
        threshold = config["threshold"]
        window_minutes = config["window_minutes"]
        
        # Count recent events of this type
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_events = [
            e for e in self.events
            if (e.violation_type == event.violation_type and
                e.timestamp >= cutoff_time and
                e.ip_address == event.ip_address)
        ]
        
        if len(recent_events) >= threshold:
            await self._create_threshold_alert(event, recent_events, config)
    
    async def _create_threshold_alert(
        self,
        event: TenantSecurityEvent,
        recent_events: List[TenantSecurityEvent],
        config: Dict[str, Any]
    ) -> None:
        """Create alert when threshold is exceeded"""
        alert_id = f"{event.violation_type.value}_{event.ip_address}_{int(datetime.utcnow().timestamp())}"
        
        # Check rate limiting
        if alert_id in self.alert_rate_limits:
            last_alert = self.alert_rate_limits[alert_id]
            if datetime.utcnow() - last_alert < self.min_alert_interval:
                return
        
        alert = SecurityAlert(
            alert_id=alert_id,
            event_category=config["category"],
            threat_level=config["threat_level"],
            title=f"Security Threshold Exceeded: {event.violation_type.value}",
            description=f"{len(recent_events)} instances of {event.violation_type.value} from {event.ip_address} in {config['window_minutes']} minutes",
            affected_resources=[event.ip_address or "unknown"],
            source_events=[asdict(e) for e in recent_events],
            first_seen=recent_events[0].timestamp,
            last_seen=recent_events[-1].timestamp,
            occurrences=len(recent_events),
            ip_addresses={event.ip_address} if event.ip_address else set(),
            user_ids={event.user_id} if event.user_id else set(),
            tenant_ids={str(event.tenant_id)} if event.tenant_id else set()
        )
        
        await self._send_alert(alert)
        self.alert_rate_limits[alert_id] = datetime.utcnow()
    
    async def _create_pattern_alert(
        self,
        event: TenantSecurityEvent,
        pattern_name: str,
        pattern_config: Dict[str, Any]
    ) -> None:
        """Create alert for detected attack pattern"""
        alert_id = f"pattern_{pattern_name}_{event.ip_address}_{int(datetime.utcnow().timestamp())}"
        
        alert = SecurityAlert(
            alert_id=alert_id,
            event_category=pattern_config["category"],
            threat_level=pattern_config["threat_level"],
            title=f"Attack Pattern Detected: {pattern_name}",
            description=f"Known attack pattern '{pattern_name}' detected from {event.ip_address}",
            affected_resources=[event.ip_address or "unknown"],
            source_events=[asdict(event)],
            first_seen=event.timestamp,
            last_seen=event.timestamp,
            occurrences=1,
            ip_addresses={event.ip_address} if event.ip_address else set(),
            user_ids={event.user_id} if event.user_id else set(),
            tenant_ids={str(event.tenant_id)} if event.tenant_id else set(),
            metadata={"pattern": pattern_name, "details": event.details}
        )
        
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: SecurityAlert) -> None:
        """Send security alert via configured channels"""
        try:
            # Store alert
            self.alerts[alert.alert_id] = alert
            self.metrics.escalated_alerts += 1
            
            # Log alert
            logger.warning(f"Security Alert: {alert.title} - {alert.description}")
            
            # Send webhook notification
            if self.alert_webhook_url:
                await self._send_webhook_alert(alert)
            
            # Send email notifications
            if self.email_alerts:
                await self._send_email_alert(alert)
            
            # Update threat level metrics
            if alert.threat_level not in self.metrics.events_by_threat_level:
                self.metrics.events_by_threat_level[alert.threat_level] = 0
            self.metrics.events_by_threat_level[alert.threat_level] += 1
            
        except Exception as e:
            logger.error(f"Error sending security alert: {e}")
    
    async def _send_webhook_alert(self, alert: SecurityAlert) -> None:
        """Send alert via webhook"""
        try:
            import aiohttp
            
            payload = {
                "alert_type": "security_incident",
                "severity": alert.threat_level.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.first_seen.isoformat(),
                "details": alert.to_dict()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.alert_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully: {alert.alert_id}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    async def _send_email_alert(self, alert: SecurityAlert) -> None:
        """Send alert via email (placeholder - implement with your email service)"""
        # This would integrate with your email service (SendGrid, SES, etc.)
        logger.info(f"Email alert would be sent to {self.email_alerts}: {alert.title}")
    
    async def _log_security_event(
        self,
        event: TenantSecurityEvent,
        threat_level: SecurityThreatLevel
    ) -> None:
        """Log security event with structured data"""
        log_data = {
            "event_type": "security_violation",
            "violation_type": event.violation_type.value,
            "threat_level": threat_level.value,
            "user_id": event.user_id,
            "tenant_id": str(event.tenant_id) if event.tenant_id else None,
            "ip_address": event.ip_address,
            "timestamp": event.timestamp.isoformat(),
            "details": event.details
        }
        
        if threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            logger.error(f"High-priority security event: {json.dumps(log_data)}")
        else:
            logger.warning(f"Security event: {json.dumps(log_data)}")
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        return self.metrics
    
    def get_recent_alerts(self, hours: int = 24) -> List[SecurityAlert]:
        """Get recent security alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts.values()
            if alert.last_seen >= cutoff_time
        ]
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP address is flagged as suspicious"""
        return ip_address in self.suspicious_ips
    
    async def cleanup_old_data(self) -> None:
        """Clean up old security data"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean up old events
        self.events = deque([
            event for event in self.events
            if event.timestamp >= cutoff_time
        ], maxlen=10000)
        
        # Clean up old alerts
        old_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.last_seen < cutoff_time
        ]
        for alert_id in old_alerts:
            del self.alerts[alert_id]
        
        # Clean up old suspicious IP data
        old_ips = [
            ip for ip, data in self.suspicious_ips.items()
            if data.get("last_seen", datetime.min) < cutoff_time
        ]
        for ip in old_ips:
            del self.suspicious_ips[ip]
        
        logger.info(f"Cleaned up old security data: {len(old_alerts)} alerts, {len(old_ips)} IPs")


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def initialize_security_monitor(
    alert_webhook_url: Optional[str] = None,
    email_alerts: Optional[List[str]] = None,
    retention_days: int = 90
) -> SecurityMonitor:
    """Initialize global security monitor"""
    global _security_monitor
    _security_monitor = SecurityMonitor(
        alert_webhook_url=alert_webhook_url,
        email_alerts=email_alerts,
        retention_days=retention_days
    )
    return _security_monitor