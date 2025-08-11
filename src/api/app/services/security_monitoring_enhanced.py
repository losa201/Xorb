"""
Enhanced Security Monitoring Service
Real-time threat detection, anomaly analysis, and security event correlation
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import ipaddress

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class EventType(Enum):
    """Security event types"""
    LOGIN_ATTEMPT = "login_attempt"
    API_ACCESS = "api_access"
    SCAN_DETECTED = "scan_detected"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    MALWARE_DETECTED = "malware_detected"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: EventType
    threat_level: ThreatLevel
    source_ip: str
    target: str
    timestamp: datetime
    user_agent: Optional[str] = None
    payload: Optional[str] = None
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['threat_level'] = self.threat_level.value
        return data

@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    indicator: str
    indicator_type: str  # ip, domain, hash, url
    threat_level: ThreatLevel
    confidence: float
    source: str
    description: str
    tags: List[str]
    first_seen: datetime
    last_seen: datetime
    
class EnhancedSecurityMonitor:
    """Enhanced security monitoring with real-time threat detection"""
    
    def __init__(self):
        self.events_buffer = deque(maxlen=10000)  # Ring buffer for recent events
        self.threat_intelligence = {}  # Threat intel database
        self.ip_reputation = {}  # IP reputation cache
        self.behavioral_profiles = {}  # User behavioral profiles
        self.correlation_rules = []  # Event correlation rules
        self.anomaly_detection_enabled = True
        self.threat_feeds_enabled = True
        
        # Initialize threat detection patterns
        self._initialize_detection_patterns()
        self._initialize_threat_intelligence()
    
    def _initialize_detection_patterns(self):
        """Initialize threat detection patterns"""
        self.detection_patterns = {
            'sql_injection': [
                r"(?i)(union.*select|select.*from|insert.*into|delete.*from)",
                r"(?i)(\'\s*or\s*\'|\'\s*and\s*\'|--\s*$|;\s*--)",
                r"(?i)(exec\s*\(|sp_executesql|xp_cmdshell)"
            ],
            'xss': [
                r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
                r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
                r"(?i)(<img.*src.*=.*javascript|<iframe.*src.*=)"
            ],
            'directory_traversal': [
                r"(?i)(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                r"(?i)(\/etc\/passwd|\/etc\/shadow|\/windows\/system32)"
            ],
            'command_injection': [
                r"(?i)(;|\||\&|\$\(|\`|nc\s+-l|bash\s+-i|sh\s+-i)",
                r"(?i)(wget\s+|curl\s+|python\s+-c|perl\s+-e)"
            ]
        }
    
    def _initialize_threat_intelligence(self):
        """Initialize threat intelligence database"""
        # Mock threat intelligence data (in production, load from external feeds)
        malicious_ips = [
            "192.168.1.100",  # Example malicious IP
            "10.0.0.50",      # Example suspicious IP
            "172.16.0.25"     # Example compromised IP
        ]
        
        for ip in malicious_ips:
            self.threat_intelligence[ip] = ThreatIntelligence(
                indicator=ip,
                indicator_type="ip",
                threat_level=ThreatLevel.HIGH,
                confidence=0.85,
                source="internal_honeypot",
                description="Known malicious IP from honeypot data",
                tags=["malware", "botnet", "scanner"],
                first_seen=datetime.utcnow() - timedelta(days=30),
                last_seen=datetime.utcnow() - timedelta(hours=2)
            )
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Analyze incoming request for security threats"""
        
        try:
            # Extract request details
            source_ip = request_data.get('source_ip', '')
            path = request_data.get('path', '')
            method = request_data.get('method', '')
            user_agent = request_data.get('user_agent', '')
            payload = request_data.get('payload', '')
            headers = request_data.get('headers', {})
            
            # Check threat intelligence
            threat_intel_result = await self._check_threat_intelligence(source_ip)
            if threat_intel_result:
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.SCAN_DETECTED,
                    threat_level=threat_intel_result.threat_level,
                    source_ip=source_ip,
                    target=path,
                    timestamp=datetime.utcnow(),
                    user_agent=user_agent,
                    metadata={
                        'threat_intel': asdict(threat_intel_result),
                        'method': method
                    }
                )
            
            # Check for attack patterns
            attack_result = await self._detect_attack_patterns(payload, path, headers)
            if attack_result:
                return attack_result
            
            # Check for brute force attempts
            brute_force_result = await self._detect_brute_force(source_ip, path)
            if brute_force_result:
                return brute_force_result
            
            # Check for anomalous behavior
            if self.anomaly_detection_enabled:
                anomaly_result = await self._detect_anomalies(request_data)
                if anomaly_result:
                    return anomaly_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            return None
    
    async def _check_threat_intelligence(self, ip: str) -> Optional[ThreatIntelligence]:
        """Check IP against threat intelligence database"""
        
        # Check internal threat intelligence
        if ip in self.threat_intelligence:
            intel = self.threat_intelligence[ip]
            intel.last_seen = datetime.utcnow()
            return intel
        
        # Check IP reputation (mock implementation)
        if await self._is_malicious_ip(ip):
            # Create new threat intelligence entry
            intel = ThreatIntelligence(
                indicator=ip,
                indicator_type="ip",
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.75,
                source="reputation_check",
                description="IP flagged by reputation service",
                tags=["suspicious"],
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
            self.threat_intelligence[ip] = intel
            return intel
        
        return None
    
    async def _is_malicious_ip(self, ip: str) -> bool:
        """Check if IP is malicious (mock implementation)"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check for private/local IPs
            if ip_obj.is_private or ip_obj.is_loopback:
                return False
            
            # Mock reputation check (in production, query external services)
            # For demonstration, flag certain IP patterns as suspicious
            if ip.endswith('.100') or ip.endswith('.50'):
                return True
                
            return False
            
        except ValueError:
            return False
    
    async def _detect_attack_patterns(self, payload: str, path: str, headers: Dict[str, str]) -> Optional[SecurityEvent]:
        """Detect attack patterns in request data"""
        
        import re
        
        combined_data = f"{payload} {path} {' '.join(headers.values())}"
        
        # Check SQL injection patterns
        for pattern in self.detection_patterns['sql_injection']:
            if re.search(pattern, combined_data):
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.SQL_INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    source_ip="",  # Will be filled by caller
                    target=path,
                    timestamp=datetime.utcnow(),
                    payload=payload[:1000],  # Truncate for storage
                    metadata={'pattern_matched': pattern, 'attack_type': 'sql_injection'}
                )
        
        # Check XSS patterns
        for pattern in self.detection_patterns['xss']:
            if re.search(pattern, combined_data):
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.XSS_ATTEMPT,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip="",
                    target=path,
                    timestamp=datetime.utcnow(),
                    payload=payload[:1000],
                    metadata={'pattern_matched': pattern, 'attack_type': 'xss'}
                )
        
        # Check directory traversal
        for pattern in self.detection_patterns['directory_traversal']:
            if re.search(pattern, combined_data):
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.SCAN_DETECTED,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip="",
                    target=path,
                    timestamp=datetime.utcnow(),
                    payload=payload[:1000],
                    metadata={'pattern_matched': pattern, 'attack_type': 'directory_traversal'}
                )
        
        # Check command injection
        for pattern in self.detection_patterns['command_injection']:
            if re.search(pattern, combined_data):
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.PRIVILEGE_ESCALATION,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip="",
                    target=path,
                    timestamp=datetime.utcnow(),
                    payload=payload[:1000],
                    metadata={'pattern_matched': pattern, 'attack_type': 'command_injection'}
                )
        
        return None
    
    async def _detect_brute_force(self, source_ip: str, path: str) -> Optional[SecurityEvent]:
        """Detect brute force attacks"""
        
        # Count recent requests from this IP
        now = datetime.utcnow()
        recent_events = [
            event for event in self.events_buffer
            if event.source_ip == source_ip and 
               (now - event.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        # Check for high request rate
        if len(recent_events) > 50:  # More than 50 requests in 5 minutes
            return SecurityEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                source_ip=source_ip,
                target=path,
                timestamp=datetime.utcnow(),
                metadata={
                    'request_count': len(recent_events),
                    'time_window': '5_minutes',
                    'rate': len(recent_events) / 5  # requests per minute
                }
            )
        
        return None
    
    async def _detect_anomalies(self, request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect anomalous behavior patterns"""
        
        source_ip = request_data.get('source_ip', '')
        user_agent = request_data.get('user_agent', '')
        path = request_data.get('path', '')
        
        # Detect unusual user agents
        suspicious_user_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'zap', 'burp',
            'python-requests', 'curl', 'wget'
        ]
        
        if any(agent.lower() in user_agent.lower() for agent in suspicious_user_agents):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.SCAN_DETECTED,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                target=path,
                timestamp=datetime.utcnow(),
                user_agent=user_agent,
                metadata={
                    'anomaly_type': 'suspicious_user_agent',
                    'detected_tool': next(agent for agent in suspicious_user_agents if agent.lower() in user_agent.lower())
                }
            )
        
        # Detect rapid scanning behavior
        if '/admin' in path or '/wp-admin' in path or '/.env' in path:
            return SecurityEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.SCAN_DETECTED,
                threat_level=ThreatLevel.LOW,
                source_ip=source_ip,
                target=path,
                timestamp=datetime.utcnow(),
                metadata={
                    'anomaly_type': 'sensitive_path_access',
                    'path_category': 'admin_or_config'
                }
            )
        
        return None
    
    async def correlate_events(self, events: List[SecurityEvent]) -> List[SecurityEvent]:
        """Correlate security events to identify attack campaigns"""
        
        correlated_events = []
        
        # Group events by source IP
        ip_groups = defaultdict(list)
        for event in events:
            ip_groups[event.source_ip].append(event)
        
        # Look for attack patterns
        for source_ip, ip_events in ip_groups.items():
            if len(ip_events) >= 3:  # Multiple events from same IP
                # Check if events span multiple attack types
                event_types = set(event.event_type for event in ip_events)
                if len(event_types) >= 2:
                    # Create correlation event
                    correlation_id = self._generate_correlation_id()
                    for event in ip_events:
                        event.correlation_id = correlation_id
                    
                    # Create summary event
                    correlated_event = SecurityEvent(
                        event_id=self._generate_event_id(),
                        event_type=EventType.ANOMALOUS_BEHAVIOR,
                        threat_level=ThreatLevel.HIGH,
                        source_ip=source_ip,
                        target="multiple_targets",
                        timestamp=datetime.utcnow(),
                        metadata={
                            'correlation_type': 'multi_vector_attack',
                            'event_count': len(ip_events),
                            'attack_types': list(event_types),
                            'time_span': (max(e.timestamp for e in ip_events) - min(e.timestamp for e in ip_events)).total_seconds(),
                            'correlated_events': [e.event_id for e in ip_events]
                        },
                        correlation_id=correlation_id
                    )
                    correlated_events.append(correlated_event)
        
        return correlated_events
    
    async def add_event(self, event: SecurityEvent):
        """Add security event to buffer for analysis"""
        self.events_buffer.append(event)
        
        # Trigger real-time correlation if buffer has enough events
        if len(self.events_buffer) >= 100:
            recent_events = list(self.events_buffer)[-100:]  # Last 100 events
            correlated = await self.correlate_events(recent_events)
            
            for corr_event in correlated:
                logger.warning(f"Correlated security event detected: {corr_event.event_type.value} from {corr_event.source_ip}")
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security monitoring dashboard"""
        
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)
        
        # Filter recent events
        recent_events = [e for e in self.events_buffer if e.timestamp >= last_24h]
        last_hour_events = [e for e in recent_events if e.timestamp >= last_hour]
        
        # Calculate statistics
        event_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        top_sources = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type.value] += 1
            threat_levels[event.threat_level.value] += 1
            top_sources[event.source_ip] += 1
        
        return {
            'summary': {
                'total_events_24h': len(recent_events),
                'events_last_hour': len(last_hour_events),
                'unique_source_ips': len(set(e.source_ip for e in recent_events)),
                'critical_events': threat_levels.get('critical', 0),
                'high_severity_events': threat_levels.get('high', 0)
            },
            'event_types': dict(event_counts),
            'threat_levels': dict(threat_levels),
            'top_source_ips': dict(sorted(top_sources.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_critical_events': [
                event.to_dict() for event in recent_events
                if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]
            ][-10:],  # Last 10 critical/high events
            'threat_intelligence_stats': {
                'total_indicators': len(self.threat_intelligence),
                'active_threats': len([
                    intel for intel in self.threat_intelligence.values()
                    if (now - intel.last_seen).days < 7
                ])
            }
        }
    
    async def update_threat_intelligence(self, indicators: List[Dict[str, Any]]):
        """Update threat intelligence database"""
        
        for indicator_data in indicators:
            indicator = indicator_data.get('indicator')
            if not indicator:
                continue
            
            intel = ThreatIntelligence(
                indicator=indicator,
                indicator_type=indicator_data.get('type', 'unknown'),
                threat_level=ThreatLevel(indicator_data.get('threat_level', 'medium')),
                confidence=indicator_data.get('confidence', 0.5),
                source=indicator_data.get('source', 'external_feed'),
                description=indicator_data.get('description', ''),
                tags=indicator_data.get('tags', []),
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
            
            self.threat_intelligence[indicator] = intel
        
        logger.info(f"Updated threat intelligence with {len(indicators)} indicators")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.utcnow()) % 10000:04d}"
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for related events"""
        return f"corr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.utcnow()) % 1000:03d}"

# Global security monitor instance
security_monitor = EnhancedSecurityMonitor()

async def get_security_monitor() -> EnhancedSecurityMonitor:
    """Get security monitor instance"""
    return security_monitor

# Integration functions for middleware
async def analyze_request_security(request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
    """Analyze request for security threats"""
    monitor = await get_security_monitor()
    return await monitor.analyze_request(request_data)

async def log_security_event(event: SecurityEvent):
    """Log security event"""
    monitor = await get_security_monitor()
    await monitor.add_event(event)
    
    # Log to standard logger
    logger.warning(
        f"Security event: {event.event_type.value} | "
        f"Level: {event.threat_level.value} | "
        f"Source: {event.source_ip} | "
        f"Target: {event.target}"
    )