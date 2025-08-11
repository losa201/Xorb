"""
Event normalization for SIEM processing
Converts parsed logs into standardized security events
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import ipaddress
import re

from .log_parser import ParsedLog

logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """Security event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization" 
    NETWORK_TRAFFIC = "network_traffic"
    SYSTEM_ACTIVITY = "system_activity"
    FILE_ACTIVITY = "file_activity"
    PROCESS_ACTIVITY = "process_activity"
    MALWARE_DETECTION = "malware_detection"
    INTRUSION_DETECTION = "intrusion_detection"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    VULNERABILITY = "vulnerability"
    COMPLIANCE = "compliance"
    OTHER = "other"


class EventSeverity(Enum):
    """Normalized event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventAction(Enum):
    """Normalized event actions"""
    ALLOW = "allow"
    DENY = "deny"
    BLOCK = "block"
    ALERT = "alert"
    LOG = "log"
    CREATE = "create"
    DELETE = "delete"
    MODIFY = "modify"
    ACCESS = "access"
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    EXECUTE = "execute"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    UNKNOWN = "unknown"


@dataclass
class NormalizedEvent:
    """Normalized security event"""
    # Core event fields
    event_id: str
    timestamp: datetime
    category: EventCategory
    severity: EventSeverity
    action: EventAction
    
    # Network fields
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    
    # Identity fields
    user: Optional[str] = None
    user_domain: Optional[str] = None
    source_user: Optional[str] = None
    destination_user: Optional[str] = None
    
    # System fields
    hostname: Optional[str] = None
    process_name: Optional[str] = None
    process_id: Optional[int] = None
    parent_process: Optional[str] = None
    command_line: Optional[str] = None
    
    # File fields
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    
    # HTTP fields
    http_method: Optional[str] = None
    http_status_code: Optional[int] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    
    # Message and metadata
    message: str = ""
    description: Optional[str] = None
    device_vendor: Optional[str] = None
    device_product: Optional[str] = None
    signature_id: Optional[str] = None
    
    # Risk scoring
    risk_score: float = 0.0
    confidence: float = 1.0
    
    # Enrichment fields
    source_geolocation: Optional[Dict[str, Any]] = None
    destination_geolocation: Optional[Dict[str, Any]] = None
    threat_intelligence: Optional[Dict[str, Any]] = None
    
    # Original data
    original_log: Optional[str] = None
    source_id: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_fields is None:
            self.custom_fields = {}


class EventNormalizer:
    """Normalizes parsed logs into security events"""
    
    def __init__(self):
        self.severity_mappings = self._build_severity_mappings()
        self.category_patterns = self._build_category_patterns()
        self.action_patterns = self._build_action_patterns()
        self.risk_weights = self._build_risk_weights()
        
        # Common ports for protocol detection
        self.common_ports = {
            20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet",
            25: "smtp", 53: "dns", 67: "dhcp", 68: "dhcp",
            69: "tftp", 80: "http", 110: "pop3", 143: "imap",
            443: "https", 993: "imaps", 995: "pop3s"
        }
    
    def normalize(self, parsed_log: ParsedLog) -> Optional[NormalizedEvent]:
        """Normalize a parsed log into a security event"""
        try:
            # Generate event ID
            event_id = self._generate_event_id(parsed_log)
            
            # Determine event category
            category = self._determine_category(parsed_log)
            
            # Normalize severity
            severity = self._normalize_severity(parsed_log.severity)
            
            # Determine action
            action = self._determine_action(parsed_log, category)
            
            # Extract network information
            source_ip = self._validate_ip(parsed_log.source_ip)
            destination_ip = self._validate_ip(parsed_log.destination_ip)
            
            # Extract user information
            user, user_domain = self._extract_user_info(parsed_log.user)
            
            # Extract process information
            process_name = self._extract_process_name(parsed_log.process)
            
            # Extract HTTP information
            http_info = self._extract_http_info(parsed_log)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(parsed_log, category, severity, action)
            
            # Extract tags
            tags = self._extract_tags(parsed_log, category)
            
            event = NormalizedEvent(
                event_id=event_id,
                timestamp=parsed_log.timestamp,
                category=category,
                severity=severity,
                action=action,
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=parsed_log.source_port,
                destination_port=parsed_log.destination_port,
                protocol=self._normalize_protocol(parsed_log.protocol),
                user=user,
                user_domain=user_domain,
                hostname=parsed_log.fields.get('hostname'),
                process_name=process_name,
                process_id=parsed_log.pid,
                http_method=http_info.get('method'),
                http_status_code=http_info.get('status_code'),
                url=http_info.get('url'),
                user_agent=http_info.get('user_agent'),
                referer=http_info.get('referer'),
                message=parsed_log.message or "",
                device_vendor=parsed_log.fields.get('device_vendor'),
                device_product=parsed_log.fields.get('device_product'),
                signature_id=parsed_log.fields.get('signature_id'),
                risk_score=risk_score,
                original_log=parsed_log.raw_log,
                source_id=parsed_log.metadata.get('source_id'),
                tags=tags,
                custom_fields=parsed_log.fields
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error normalizing event: {e}")
            return None
    
    def _generate_event_id(self, parsed_log: ParsedLog) -> str:
        """Generate unique event ID"""
        import hashlib
        
        # Create hash from key fields
        key_data = f"{parsed_log.timestamp}{parsed_log.message}{parsed_log.source_ip}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _determine_category(self, parsed_log: ParsedLog) -> EventCategory:
        """Determine event category based on log content"""
        message = (parsed_log.message or "").lower()
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern in message:
                    return category
        
        # Check fields for additional categorization
        fields = parsed_log.fields or {}
        
        # Network traffic indicators
        if parsed_log.source_ip or parsed_log.destination_ip or parsed_log.protocol:
            return EventCategory.NETWORK_TRAFFIC
        
        # Authentication indicators
        if any(field in fields for field in ['user', 'username', 'login']):
            return EventCategory.AUTHENTICATION
        
        # System activity indicators
        if parsed_log.process or parsed_log.pid:
            return EventCategory.SYSTEM_ACTIVITY
        
        return EventCategory.OTHER
    
    def _normalize_severity(self, severity: Optional[str]) -> EventSeverity:
        """Normalize severity level"""
        if not severity:
            return EventSeverity.INFO
        
        severity_lower = severity.lower()
        
        return self.severity_mappings.get(severity_lower, EventSeverity.INFO)
    
    def _determine_action(self, parsed_log: ParsedLog, category: EventCategory) -> EventAction:
        """Determine event action"""
        message = (parsed_log.message or "").lower()
        fields = parsed_log.fields or {}
        
        # Check for explicit action in fields
        if 'action' in fields:
            action_str = str(fields['action']).lower()
            for action, patterns in self.action_patterns.items():
                if action_str in patterns:
                    return action
        
        # Pattern matching in message
        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                if pattern in message:
                    return action
        
        # Category-based defaults
        if category == EventCategory.AUTHENTICATION:
            if 'fail' in message or 'error' in message:
                return EventAction.FAILED_LOGIN
            elif 'login' in message or 'auth' in message:
                return EventAction.LOGIN
            elif 'logout' in message:
                return EventAction.LOGOUT
        
        elif category == EventCategory.NETWORK_TRAFFIC:
            # Check HTTP status codes
            status_code = fields.get('status_code') or fields.get('status')
            if status_code:
                try:
                    code = int(status_code)
                    if 200 <= code < 300:
                        return EventAction.ALLOW
                    elif 400 <= code < 500:
                        return EventAction.DENY
                    elif code >= 500:
                        return EventAction.ALERT
                except (ValueError, TypeError):
                    pass
            
            return EventAction.CONNECT
        
        return EventAction.UNKNOWN
    
    def _validate_ip(self, ip_str: Optional[str]) -> Optional[str]:
        """Validate IP address"""
        if not ip_str:
            return None
        
        try:
            ipaddress.ip_address(ip_str)
            return ip_str
        except ValueError:
            return None
    
    def _extract_user_info(self, user_str: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Extract user and domain information"""
        if not user_str or user_str == '-':
            return None, None
        
        # Handle domain\user format
        if '\\' in user_str:
            domain, user = user_str.split('\\', 1)
            return user, domain
        
        # Handle user@domain format
        if '@' in user_str:
            user, domain = user_str.split('@', 1)
            return user, domain
        
        return user_str, None
    
    def _extract_process_name(self, process_str: Optional[str]) -> Optional[str]:
        """Extract process name from process string"""
        if not process_str:
            return None
        
        # Remove path information
        if '/' in process_str:
            return process_str.split('/')[-1]
        elif '\\' in process_str:
            return process_str.split('\\')[-1]
        
        return process_str
    
    def _extract_http_info(self, parsed_log: ParsedLog) -> Dict[str, Any]:
        """Extract HTTP-related information"""
        fields = parsed_log.fields or {}
        
        return {
            'method': fields.get('method'),
            'status_code': fields.get('status_code') or fields.get('status'),
            'url': fields.get('url') or fields.get('uri'),
            'user_agent': fields.get('user_agent'),
            'referer': fields.get('referer')
        }
    
    def _normalize_protocol(self, protocol: Optional[str]) -> Optional[str]:
        """Normalize protocol name"""
        if not protocol:
            return None
        
        protocol_map = {
            '6': 'tcp',
            '17': 'udp',
            '1': 'icmp',
            'TCP': 'tcp',
            'UDP': 'udp',
            'ICMP': 'icmp'
        }
        
        return protocol_map.get(protocol, protocol.lower())
    
    def _calculate_risk_score(self, parsed_log: ParsedLog, category: EventCategory, 
                            severity: EventSeverity, action: EventAction) -> float:
        """Calculate risk score for the event"""
        base_score = 0.0
        
        # Severity weight
        severity_weights = {
            EventSeverity.CRITICAL: 1.0,
            EventSeverity.HIGH: 0.8,
            EventSeverity.MEDIUM: 0.6,
            EventSeverity.LOW: 0.4,
            EventSeverity.INFO: 0.2
        }
        base_score += severity_weights.get(severity, 0.2)
        
        # Category weight
        category_weights = {
            EventCategory.MALWARE_DETECTION: 0.9,
            EventCategory.INTRUSION_DETECTION: 0.8,
            EventCategory.AUTHENTICATION: 0.7,
            EventCategory.AUTHORIZATION: 0.6,
            EventCategory.NETWORK_TRAFFIC: 0.5,
            EventCategory.SYSTEM_ACTIVITY: 0.4,
            EventCategory.FILE_ACTIVITY: 0.4,
            EventCategory.PROCESS_ACTIVITY: 0.4
        }
        base_score += category_weights.get(category, 0.2)
        
        # Action weight
        action_weights = {
            EventAction.DENY: 0.8,
            EventAction.BLOCK: 0.8,
            EventAction.ALERT: 0.7,
            EventAction.FAILED_LOGIN: 0.6,
            EventAction.DELETE: 0.5,
            EventAction.MODIFY: 0.4,
            EventAction.ALLOW: 0.1
        }
        base_score += action_weights.get(action, 0.3)
        
        # Normalize to 0-1 range
        risk_score = min(base_score / 3.0, 1.0)
        
        # Additional risk factors
        message = (parsed_log.message or "").lower()
        
        # High-risk keywords
        high_risk_keywords = [
            'attack', 'exploit', 'malware', 'virus', 'trojan',
            'backdoor', 'rootkit', 'breach', 'compromise'
        ]
        
        for keyword in high_risk_keywords:
            if keyword in message:
                risk_score = min(risk_score + 0.2, 1.0)
                break
        
        return round(risk_score, 2)
    
    def _extract_tags(self, parsed_log: ParsedLog, category: EventCategory) -> List[str]:
        """Extract tags for the event"""
        tags = []
        
        # Category tag
        tags.append(category.value)
        
        # Source type tag
        if parsed_log.metadata.get('source_id'):
            tags.append(f"source:{parsed_log.metadata['source_id']}")
        
        # Protocol tag
        if parsed_log.protocol:
            tags.append(f"protocol:{self._normalize_protocol(parsed_log.protocol)}")
        
        # Network tags
        if parsed_log.source_ip:
            if self._is_private_ip(parsed_log.source_ip):
                tags.append("internal_source")
            else:
                tags.append("external_source")
        
        if parsed_log.destination_ip:
            if self._is_private_ip(parsed_log.destination_ip):
                tags.append("internal_destination")
            else:
                tags.append("external_destination")
        
        # Port-based tags
        if parsed_log.destination_port in self.common_ports:
            service = self.common_ports[parsed_log.destination_port]
            tags.append(f"service:{service}")
        
        return tags
    
    def _is_private_ip(self, ip_str: str) -> bool:
        """Check if IP address is private"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False
    
    def _build_severity_mappings(self) -> Dict[str, EventSeverity]:
        """Build severity mapping dictionary"""
        return {
            # Standard severity levels
            'critical': EventSeverity.CRITICAL,
            'high': EventSeverity.HIGH,
            'medium': EventSeverity.MEDIUM,
            'low': EventSeverity.LOW,
            'info': EventSeverity.INFO,
            'information': EventSeverity.INFO,
            
            # Syslog severity levels
            'emergency': EventSeverity.CRITICAL,
            'alert': EventSeverity.CRITICAL,
            'error': EventSeverity.HIGH,
            'warning': EventSeverity.MEDIUM,
            'notice': EventSeverity.LOW,
            'debug': EventSeverity.INFO,
            
            # Numeric levels
            '0': EventSeverity.CRITICAL,  # Emergency
            '1': EventSeverity.CRITICAL,  # Alert
            '2': EventSeverity.CRITICAL,  # Critical
            '3': EventSeverity.HIGH,      # Error
            '4': EventSeverity.MEDIUM,    # Warning
            '5': EventSeverity.LOW,       # Notice
            '6': EventSeverity.INFO,      # Info
            '7': EventSeverity.INFO,      # Debug
        }
    
    def _build_category_patterns(self) -> Dict[EventCategory, List[str]]:
        """Build category pattern matching dictionary"""
        return {
            EventCategory.AUTHENTICATION: [
                'login', 'logout', 'auth', 'authentication', 'signin', 'signout',
                'credential', 'password', 'token', 'session', 'sso'
            ],
            EventCategory.AUTHORIZATION: [
                'authorization', 'permission', 'access control', 'privilege',
                'forbidden', 'unauthorized', 'denied'
            ],
            EventCategory.NETWORK_TRAFFIC: [
                'connection', 'traffic', 'packet', 'flow', 'tcp', 'udp',
                'icmp', 'firewall', 'proxy', 'load balancer'
            ],
            EventCategory.MALWARE_DETECTION: [
                'malware', 'virus', 'trojan', 'worm', 'spyware', 'adware',
                'ransomware', 'rootkit', 'backdoor', 'threat detected'
            ],
            EventCategory.INTRUSION_DETECTION: [
                'intrusion', 'attack', 'exploit', 'vulnerability', 'breach',
                'suspicious', 'anomaly', 'ips', 'ids'
            ],
            EventCategory.FILE_ACTIVITY: [
                'file', 'directory', 'folder', 'document', 'download',
                'upload', 'copy', 'move', 'rename'
            ],
            EventCategory.PROCESS_ACTIVITY: [
                'process', 'execution', 'command', 'script', 'binary',
                'executable', 'spawn', 'fork'
            ],
            EventCategory.SYSTEM_ACTIVITY: [
                'system', 'service', 'daemon', 'startup', 'shutdown',
                'boot', 'kernel', 'driver'
            ]
        }
    
    def _build_action_patterns(self) -> Dict[EventAction, List[str]]:
        """Build action pattern matching dictionary"""
        return {
            EventAction.ALLOW: ['allow', 'permit', 'accept', 'success', 'ok'],
            EventAction.DENY: ['deny', 'reject', 'refuse', 'block', 'drop'],
            EventAction.BLOCK: ['block', 'blocked', 'filtered'],
            EventAction.ALERT: ['alert', 'alarm', 'warning', 'detected'],
            EventAction.LOGIN: ['login', 'logon', 'signin', 'authenticated'],
            EventAction.LOGOUT: ['logout', 'logoff', 'signout', 'disconnected'],
            EventAction.FAILED_LOGIN: ['failed login', 'authentication failed', 'invalid password'],
            EventAction.CREATE: ['create', 'created', 'add', 'new'],
            EventAction.DELETE: ['delete', 'deleted', 'remove', 'removed'],
            EventAction.MODIFY: ['modify', 'modified', 'change', 'update', 'edit'],
            EventAction.ACCESS: ['access', 'accessed', 'read', 'open'],
            EventAction.EXECUTE: ['execute', 'executed', 'run', 'launch'],
            EventAction.CONNECT: ['connect', 'connected', 'establish'],
            EventAction.DISCONNECT: ['disconnect', 'disconnected', 'close', 'terminate']
        }
    
    def _build_risk_weights(self) -> Dict[str, float]:
        """Build risk scoring weights"""
        return {
            'categories': {
                EventCategory.MALWARE_DETECTION: 0.9,
                EventCategory.INTRUSION_DETECTION: 0.8,
                EventCategory.AUTHENTICATION: 0.7,
                EventCategory.AUTHORIZATION: 0.6
            },
            'actions': {
                EventAction.DENY: 0.8,
                EventAction.BLOCK: 0.8,
                EventAction.FAILED_LOGIN: 0.6
            }
        }