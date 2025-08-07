"""
Event normalizer for security events
Converts parsed logs into normalized security events with standardized fields
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .log_parser import ParsedLog


class EventCategory(Enum):
    """Security event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK_TRAFFIC = "network_traffic"
    FILE_ACTIVITY = "file_activity"
    PROCESS_ACTIVITY = "process_activity"
    SYSTEM_ACTIVITY = "system_activity"
    SECURITY_ALERT = "security_alert"
    MALWARE = "malware"
    VULNERABILITY = "vulnerability"
    DATA_ACCESS = "data_access"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE = "compliance"
    UNKNOWN = "unknown"


class EventSeverity(Enum):
    """Standardized event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventOutcome(Enum):
    """Event outcome status"""
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class NormalizedEvent:
    """Normalized security event"""
    
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event classification
    category: EventCategory = EventCategory.UNKNOWN
    subcategory: str = ""
    event_type: str = ""
    severity: EventSeverity = EventSeverity.INFO
    outcome: EventOutcome = EventOutcome.UNKNOWN
    
    # Source information
    source_system: str = ""
    source_host: str = ""
    source_ip: str = ""
    source_port: Optional[int] = None
    source_user: str = ""
    source_process: str = ""
    
    # Destination information
    dest_host: str = ""
    dest_ip: str = ""
    dest_port: Optional[int] = None
    dest_user: str = ""
    
    # Event details
    message: str = ""
    description: str = ""
    action: str = ""
    object_name: str = ""
    object_type: str = ""
    
    # Security context
    threat_name: str = ""
    threat_type: str = ""
    signature_id: str = ""
    rule_name: str = ""
    
    # Network information
    protocol: str = ""
    bytes_in: Optional[int] = None
    bytes_out: Optional[int] = None
    packets_in: Optional[int] = None
    packets_out: Optional[int] = None
    
    # File information
    file_name: str = ""
    file_path: str = ""
    file_hash: str = ""
    file_size: Optional[int] = None
    
    # Process information
    process_name: str = ""
    process_id: Optional[int] = None
    parent_process: str = ""
    command_line: str = ""
    
    # Original data
    raw_event: str = ""
    parsed_fields: Dict[str, Any] = field(default_factory=dict)
    vendor: str = ""
    product: str = ""
    version: str = ""
    
    # Enrichment data
    geo_location: Dict[str, str] = field(default_factory=dict)
    threat_intel: Dict[str, Any] = field(default_factory=dict)
    asset_info: Dict[str, Any] = field(default_factory=dict)
    
    # Correlation information
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Metadata
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            else:
                result[field_name] = field_value
        return result
    
    def get_fingerprint(self) -> str:
        """Generate event fingerprint for deduplication"""
        # Create fingerprint based on key fields
        fingerprint_data = f"{self.source_host}:{self.event_type}:{self.message}:{self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()


class EventNormalizer:
    """Normalizes parsed logs into standardized security events"""
    
    def __init__(self):
        self.severity_mappings = self._build_severity_mappings()
        self.category_mappings = self._build_category_mappings()
        self.outcome_mappings = self._build_outcome_mappings()
    
    def normalize(self, parsed_log: ParsedLog) -> NormalizedEvent:
        """Normalize a parsed log into a standard security event"""
        start_time = datetime.utcnow()
        
        event = NormalizedEvent(
            timestamp=parsed_log.timestamp,
            raw_event=parsed_log.raw_data,
            parsed_fields=parsed_log.parsed_fields,
            message=parsed_log.message,
            source_system=parsed_log.source,
            event_type=parsed_log.event_type,
            ingestion_timestamp=start_time
        )
        
        # Normalize based on format
        if parsed_log.metadata.get('format') == 'CEF':
            self._normalize_cef(parsed_log, event)
        elif parsed_log.metadata.get('format') == 'LEEF':
            self._normalize_leef(parsed_log, event)
        elif parsed_log.metadata.get('format') == 'JSON':
            self._normalize_json(parsed_log, event)
        elif 'Syslog' in parsed_log.metadata.get('format', ''):
            self._normalize_syslog(parsed_log, event)
        elif 'Windows Event' in parsed_log.metadata.get('format', ''):
            self._normalize_windows_event(parsed_log, event)
        
        # Apply common normalizations
        event.severity = self._normalize_severity(parsed_log.severity)
        event.category = self._categorize_event(event)
        event.outcome = self._determine_outcome(event)
        
        # Add processing metadata
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        event.processing_time_ms = processing_time
        
        return event
    
    def _normalize_cef(self, parsed_log: ParsedLog, event: NormalizedEvent):
        """Normalize CEF format events"""
        fields = parsed_log.parsed_fields
        
        # Vendor and product information
        event.vendor = fields.get('device_vendor', '')
        event.product = fields.get('device_product', '')
        event.version = fields.get('device_version', '')
        
        # Network information
        event.source_ip = fields.get('src', '')
        event.dest_ip = fields.get('dst', '')
        event.source_port = self._safe_int(fields.get('spt'))
        event.dest_port = self._safe_int(fields.get('dpt'))
        event.protocol = fields.get('proto', '')
        
        # User information
        event.source_user = fields.get('suser', fields.get('duser', ''))
        event.dest_user = fields.get('duser', '')
        
        # Host information
        event.source_host = fields.get('shost', fields.get('src', ''))
        event.dest_host = fields.get('dhost', fields.get('dst', ''))
        
        # File information
        event.file_name = fields.get('fname', '')
        event.file_path = fields.get('filePath', '')
        event.file_hash = fields.get('fileHash', '')
        event.file_size = self._safe_int(fields.get('fsize'))
        
        # Process information
        event.process_name = fields.get('sproc', '')
        event.process_id = self._safe_int(fields.get('spid'))
        
        # Action and outcome
        event.action = fields.get('act', '')
        event.outcome = self._cef_outcome_to_standard(fields.get('outcome', ''))
        
        # Threat information
        event.threat_name = fields.get('cs1', '')  # Often used for threat name
        event.signature_id = fields.get('cs2', fields.get('signature_id', ''))
        
        # Description
        event.description = fields.get('msg', event.message)
    
    def _normalize_leef(self, parsed_log: ParsedLog, event: NormalizedEvent):
        """Normalize LEEF format events"""
        fields = parsed_log.parsed_fields
        
        # Vendor and product
        event.vendor = fields.get('vendor', '')
        event.product = fields.get('product_name', '')
        event.version = fields.get('product_version', '')
        
        # Network information
        event.source_ip = fields.get('srcIP', fields.get('src', ''))
        event.dest_ip = fields.get('dstIP', fields.get('dst', ''))
        event.source_port = self._safe_int(fields.get('srcPort'))
        event.dest_port = self._safe_int(fields.get('dstPort'))
        event.protocol = fields.get('proto', '')
        
        # User information
        event.source_user = fields.get('usrName', '')
        
        # Host information
        event.source_host = fields.get('srcHostName', '')
        event.dest_host = fields.get('dstHostName', '')
        
        # Security information
        event.threat_name = fields.get('threat', '')
        event.signature_id = fields.get('sigId', '')
        
        # Action
        event.action = fields.get('cat', '')  # Category often contains action
    
    def _normalize_json(self, parsed_log: ParsedLog, event: NormalizedEvent):
        """Normalize JSON format events"""
        fields = parsed_log.parsed_fields
        
        # Try common JSON field mappings
        event.source_ip = fields.get('src_ip', fields.get('source_ip', fields.get('srcip', '')))
        event.dest_ip = fields.get('dest_ip', fields.get('destination_ip', fields.get('dstip', '')))
        event.source_port = self._safe_int(fields.get('src_port', fields.get('source_port')))
        event.dest_port = self._safe_int(fields.get('dest_port', fields.get('destination_port')))
        
        # User information
        event.source_user = fields.get('user', fields.get('username', fields.get('src_user', '')))
        
        # Host information
        event.source_host = fields.get('host', fields.get('hostname', fields.get('src_host', '')))
        event.dest_host = fields.get('dest_host', fields.get('destination_host', ''))
        
        # Process information
        event.process_name = fields.get('process', fields.get('process_name', ''))
        event.process_id = self._safe_int(fields.get('pid', fields.get('process_id')))
        
        # File information
        event.file_name = fields.get('file', fields.get('filename', fields.get('file_name', '')))
        event.file_path = fields.get('path', fields.get('file_path', fields.get('filepath', '')))
        
        # Action and outcome
        event.action = fields.get('action', fields.get('event_action', ''))
        
        # Additional fields
        event.description = fields.get('description', fields.get('details', ''))
    
    def _normalize_syslog(self, parsed_log: ParsedLog, event: NormalizedEvent):
        """Normalize Syslog format events"""
        fields = parsed_log.parsed_fields
        
        event.source_host = fields.get('hostname', '')
        event.source_system = fields.get('app_name', 'syslog')
        
        # Parse message for additional fields
        message = parsed_log.message
        if 'login' in message.lower():
            event.category = EventCategory.AUTHENTICATION
            if 'failed' in message.lower() or 'invalid' in message.lower():
                event.outcome = EventOutcome.FAILURE
            else:
                event.outcome = EventOutcome.SUCCESS
    
    def _normalize_windows_event(self, parsed_log: ParsedLog, event: NormalizedEvent):
        """Normalize Windows Event format"""
        fields = parsed_log.parsed_fields
        
        event.vendor = "Microsoft"
        event.product = "Windows"
        event.source_system = "Windows Event Log"
        
        event_id = fields.get('event_id', fields.get('EventID', ''))
        event.signature_id = str(event_id)
        
        # Categorize based on Event ID
        if event_id in ['4624', '4625']:  # Logon events
            event.category = EventCategory.AUTHENTICATION
            event.outcome = EventOutcome.SUCCESS if event_id == '4624' else EventOutcome.FAILURE
        elif event_id in ['4648']:  # Explicit credential use
            event.category = EventCategory.AUTHENTICATION
        elif event_id in ['4720', '4726']:  # Account management
            event.category = EventCategory.SYSTEM_ACTIVITY
        elif event_id in ['5156', '5158']:  # Network connections
            event.category = EventCategory.NETWORK_TRAFFIC
    
    def _normalize_severity(self, severity: str) -> EventSeverity:
        """Normalize severity to standard levels"""
        severity_lower = severity.lower()
        
        for pattern, std_severity in self.severity_mappings.items():
            if pattern in severity_lower:
                return std_severity
        
        return EventSeverity.INFO
    
    def _categorize_event(self, event: NormalizedEvent) -> EventCategory:
        """Categorize event based on various fields"""
        # Check message and event type for keywords
        text_to_check = f"{event.message} {event.event_type} {event.description}".lower()
        
        for keywords, category in self.category_mappings.items():
            if any(keyword in text_to_check for keyword in keywords):
                return category
        
        # Default categorization based on fields
        if event.source_user or event.dest_user:
            if any(word in text_to_check for word in ['login', 'logon', 'auth', 'password']):
                return EventCategory.AUTHENTICATION
        
        if event.source_ip and event.dest_ip:
            return EventCategory.NETWORK_TRAFFIC
        
        if event.file_name or event.file_path:
            return EventCategory.FILE_ACTIVITY
        
        if event.process_name or event.process_id:
            return EventCategory.PROCESS_ACTIVITY
        
        return EventCategory.UNKNOWN
    
    def _determine_outcome(self, event: NormalizedEvent) -> EventOutcome:
        """Determine event outcome"""
        if event.outcome != EventOutcome.UNKNOWN:
            return event.outcome
        
        text_to_check = f"{event.message} {event.description} {event.action}".lower()
        
        for pattern, outcome in self.outcome_mappings.items():
            if pattern in text_to_check:
                return outcome
        
        return EventOutcome.UNKNOWN
    
    def _build_severity_mappings(self) -> Dict[str, EventSeverity]:
        """Build severity mapping dictionary"""
        return {
            'critical': EventSeverity.CRITICAL,
            'emergency': EventSeverity.CRITICAL,
            'fatal': EventSeverity.CRITICAL,
            'high': EventSeverity.HIGH,
            'error': EventSeverity.HIGH,
            'alert': EventSeverity.HIGH,
            'medium': EventSeverity.MEDIUM,
            'warning': EventSeverity.MEDIUM,
            'warn': EventSeverity.MEDIUM,
            'notice': EventSeverity.MEDIUM,
            'low': EventSeverity.LOW,
            'info': EventSeverity.INFO,
            'information': EventSeverity.INFO,
            'debug': EventSeverity.INFO,
            'verbose': EventSeverity.INFO
        }
    
    def _build_category_mappings(self) -> Dict[tuple, EventCategory]:
        """Build category mapping dictionary"""
        return {
            ('login', 'logon', 'authenticate', 'auth', 'password', 'credential'): EventCategory.AUTHENTICATION,
            ('access', 'permission', 'authorize', 'privilege', 'role'): EventCategory.AUTHORIZATION,
            ('connection', 'network', 'traffic', 'packet', 'tcp', 'udp', 'http', 'https'): EventCategory.NETWORK_TRAFFIC,
            ('file', 'document', 'create', 'delete', 'modify', 'read', 'write'): EventCategory.FILE_ACTIVITY,
            ('process', 'execute', 'run', 'start', 'stop', 'kill', 'terminate'): EventCategory.PROCESS_ACTIVITY,
            ('system', 'service', 'boot', 'shutdown', 'restart', 'configuration'): EventCategory.SYSTEM_ACTIVITY,
            ('alert', 'alarm', 'security', 'violation', 'breach', 'incident'): EventCategory.SECURITY_ALERT,
            ('malware', 'virus', 'trojan', 'ransomware', 'backdoor', 'rootkit'): EventCategory.MALWARE,
            ('vulnerability', 'exploit', 'cve', 'patch', 'update'): EventCategory.VULNERABILITY,
            ('data', 'database', 'query', 'select', 'insert', 'update', 'delete'): EventCategory.DATA_ACCESS,
            ('policy', 'compliance', 'audit', 'regulation', 'standard'): EventCategory.COMPLIANCE
        }
    
    def _build_outcome_mappings(self) -> Dict[str, EventOutcome]:
        """Build outcome mapping dictionary"""
        return {
            'success': EventOutcome.SUCCESS,
            'successful': EventOutcome.SUCCESS,
            'allowed': EventOutcome.SUCCESS,
            'granted': EventOutcome.SUCCESS,
            'accepted': EventOutcome.SUCCESS,
            'completed': EventOutcome.SUCCESS,
            'passed': EventOutcome.SUCCESS,
            'failure': EventOutcome.FAILURE,
            'failed': EventOutcome.FAILURE,
            'denied': EventOutcome.FAILURE,
            'rejected': EventOutcome.FAILURE,
            'blocked': EventOutcome.FAILURE,
            'error': EventOutcome.FAILURE,
            'invalid': EventOutcome.FAILURE,
            'unauthorized': EventOutcome.FAILURE
        }
    
    def _cef_outcome_to_standard(self, cef_outcome: str) -> EventOutcome:
        """Convert CEF outcome to standard outcome"""
        if not cef_outcome:
            return EventOutcome.UNKNOWN
        
        cef_outcome = cef_outcome.lower()
        if 'success' in cef_outcome:
            return EventOutcome.SUCCESS
        elif 'failure' in cef_outcome or 'fail' in cef_outcome:
            return EventOutcome.FAILURE
        
        return EventOutcome.UNKNOWN
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None