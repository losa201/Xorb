"""
Log parser for various security log formats
Supports CEF, LEEF, JSON, Syslog, and custom formats
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass


class LogFormat(Enum):
    """Supported log formats"""
    CEF = "cef"
    LEEF = "leef"
    JSON = "json"
    SYSLOG = "syslog"
    WINDOWS_EVENT = "windows_event"
    APACHE = "apache"
    NGINX = "nginx"
    FIREWALL = "firewall"
    IDS_IPS = "ids_ips"


@dataclass
class ParsedLog:
    """Parsed log entry"""
    timestamp: datetime
    source: str
    event_type: str
    severity: str
    message: str
    raw_data: str
    parsed_fields: Dict[str, Any]
    metadata: Dict[str, Any]


class LogParser(ABC):
    """Abstract base class for log parsers"""
    
    @abstractmethod
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse a single log line"""
        pass
    
    @abstractmethod
    def can_parse(self, log_line: str) -> bool:
        """Check if this parser can handle the log format"""
        pass


class CEFParser(LogParser):
    """Common Event Format (CEF) parser"""
    
    CEF_PATTERN = re.compile(
        r'CEF:(\d+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)$'
    )
    
    def can_parse(self, log_line: str) -> bool:
        return log_line.strip().startswith('CEF:')
    
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        match = self.CEF_PATTERN.match(log_line.strip())
        if not match:
            return None
        
        version, device_vendor, device_product, device_version, \
        device_event_class_id, name, severity, extension = match.groups()
        
        # Parse extension fields
        extension_fields = self._parse_extension(extension)
        
        # Extract timestamp
        timestamp = self._extract_timestamp(extension_fields)
        
        return ParsedLog(
            timestamp=timestamp,
            source=f"{device_vendor} {device_product}",
            event_type=device_event_class_id,
            severity=severity,
            message=name,
            raw_data=log_line,
            parsed_fields={
                'version': version,
                'device_vendor': device_vendor,
                'device_product': device_product,
                'device_version': device_version,
                'device_event_class_id': device_event_class_id,
                'name': name,
                'severity': severity,
                **extension_fields
            },
            metadata={
                'format': 'CEF',
                'parser_version': '1.0'
            }
        )
    
    def _parse_extension(self, extension: str) -> Dict[str, str]:
        """Parse CEF extension fields"""
        fields = {}
        # CEF extension parsing logic
        parts = re.findall(r'(\w+)=([^=]+?)(?=\s+\w+=|$)', extension)
        for key, value in parts:
            fields[key] = value.strip()
        return fields
    
    def _extract_timestamp(self, fields: Dict[str, str]) -> datetime:
        """Extract timestamp from CEF fields"""
        # Try different timestamp fields
        for field in ['rt', 'end', 'start']:
            if field in fields:
                try:
                    # Parse epoch timestamp
                    if fields[field].isdigit():
                        return datetime.fromtimestamp(int(fields[field]) / 1000)
                    # Parse ISO format
                    return datetime.fromisoformat(fields[field].replace('Z', '+00:00'))
                except:
                    continue
        return datetime.utcnow()


class LEEFParser(LogParser):
    """Log Event Extended Format (LEEF) parser"""
    
    LEEF_PATTERN = re.compile(
        r'LEEF:(\d+\.\d+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)$'
    )
    
    def can_parse(self, log_line: str) -> bool:
        return log_line.strip().startswith('LEEF:')
    
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        match = self.LEEF_PATTERN.match(log_line.strip())
        if not match:
            return None
        
        version, vendor, product_name, product_version, event_id, attributes = match.groups()
        
        # Parse attributes
        attr_dict = self._parse_attributes(attributes)
        
        # Extract timestamp
        timestamp = self._extract_timestamp(attr_dict)
        
        return ParsedLog(
            timestamp=timestamp,
            source=f"{vendor} {product_name}",
            event_type=event_id,
            severity=attr_dict.get('sev', 'Medium'),
            message=attr_dict.get('devTime', ''),
            raw_data=log_line,
            parsed_fields={
                'version': version,
                'vendor': vendor,
                'product_name': product_name,
                'product_version': product_version,
                'event_id': event_id,
                **attr_dict
            },
            metadata={
                'format': 'LEEF',
                'parser_version': '1.0'
            }
        )
    
    def _parse_attributes(self, attributes: str) -> Dict[str, str]:
        """Parse LEEF attributes"""
        attrs = {}
        # Split by tabs or delimiter
        parts = attributes.split('\t')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                attrs[key.strip()] = value.strip()
        return attrs
    
    def _extract_timestamp(self, attrs: Dict[str, str]) -> datetime:
        """Extract timestamp from LEEF attributes"""
        for field in ['devTime', 'timestamp']:
            if field in attrs:
                try:
                    return datetime.fromisoformat(attrs[field])
                except:
                    continue
        return datetime.utcnow()


class JSONParser(LogParser):
    """JSON log parser"""
    
    def can_parse(self, log_line: str) -> bool:
        try:
            json.loads(log_line.strip())
            return True
        except:
            return False
    
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        try:
            data = json.loads(log_line.strip())
        except json.JSONDecodeError:
            return None
        
        # Extract common fields
        timestamp = self._extract_timestamp(data)
        source = data.get('source', data.get('host', 'unknown'))
        event_type = data.get('event_type', data.get('type', 'unknown'))
        severity = data.get('severity', data.get('level', 'info'))
        message = data.get('message', data.get('msg', str(data)))
        
        return ParsedLog(
            timestamp=timestamp,
            source=source,
            event_type=event_type,
            severity=severity,
            message=message,
            raw_data=log_line,
            parsed_fields=data,
            metadata={
                'format': 'JSON',
                'parser_version': '1.0'
            }
        )
    
    def _extract_timestamp(self, data: Dict[str, Any]) -> datetime:
        """Extract timestamp from JSON data"""
        for field in ['timestamp', '@timestamp', 'time', 'datetime', 'date']:
            if field in data:
                try:
                    ts_value = data[field]
                    if isinstance(ts_value, (int, float)):
                        return datetime.fromtimestamp(ts_value)
                    elif isinstance(ts_value, str):
                        # Try different formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                            try:
                                return datetime.strptime(ts_value, fmt)
                            except:
                                continue
                        # Try ISO format
                        return datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                except:
                    continue
        return datetime.utcnow()


class SyslogParser(LogParser):
    """RFC3164/RFC5424 Syslog parser"""
    
    RFC3164_PATTERN = re.compile(
        r'^<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.+)$'
    )
    
    RFC5424_PATTERN = re.compile(
        r'^<(\d+)>(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$'
    )
    
    def can_parse(self, log_line: str) -> bool:
        return log_line.strip().startswith('<') and '>' in log_line[:10]
    
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        line = log_line.strip()
        
        # Try RFC5424 first
        match = self.RFC5424_PATTERN.match(line)
        if match:
            return self._parse_rfc5424(match, log_line)
        
        # Try RFC3164
        match = self.RFC3164_PATTERN.match(line)
        if match:
            return self._parse_rfc3164(match, log_line)
        
        return None
    
    def _parse_rfc5424(self, match, raw_data: str) -> ParsedLog:
        priority, version, timestamp_str, hostname, app_name, proc_id, msg_id, message = match.groups()
        
        # Parse priority
        facility = int(priority) // 8
        severity_num = int(priority) % 8
        severity = self._severity_to_string(severity_num)
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.utcnow()
        
        return ParsedLog(
            timestamp=timestamp,
            source=hostname,
            event_type=app_name,
            severity=severity,
            message=message,
            raw_data=raw_data,
            parsed_fields={
                'priority': priority,
                'facility': facility,
                'severity_num': severity_num,
                'version': version,
                'hostname': hostname,
                'app_name': app_name,
                'proc_id': proc_id,
                'msg_id': msg_id
            },
            metadata={
                'format': 'Syslog RFC5424',
                'parser_version': '1.0'
            }
        )
    
    def _parse_rfc3164(self, match, raw_data: str) -> ParsedLog:
        priority, timestamp_str, hostname, message = match.groups()
        
        # Parse priority
        facility = int(priority) // 8
        severity_num = int(priority) % 8
        severity = self._severity_to_string(severity_num)
        
        # Parse timestamp (add current year)
        try:
            current_year = datetime.now().year
            timestamp = datetime.strptime(f"{current_year} {timestamp_str}", "%Y %b %d %H:%M:%S")
        except:
            timestamp = datetime.utcnow()
        
        return ParsedLog(
            timestamp=timestamp,
            source=hostname,
            event_type='syslog',
            severity=severity,
            message=message,
            raw_data=raw_data,
            parsed_fields={
                'priority': priority,
                'facility': facility,
                'severity_num': severity_num,
                'hostname': hostname
            },
            metadata={
                'format': 'Syslog RFC3164',
                'parser_version': '1.0'
            }
        )
    
    def _severity_to_string(self, severity_num: int) -> str:
        """Convert syslog severity number to string"""
        severity_map = {
            0: 'Emergency',
            1: 'Alert', 
            2: 'Critical',
            3: 'Error',
            4: 'Warning',
            5: 'Notice',
            6: 'Info',
            7: 'Debug'
        }
        return severity_map.get(severity_num, 'Unknown')


class WindowsEventParser(LogParser):
    """Windows Event Log parser"""
    
    def can_parse(self, log_line: str) -> bool:
        # Check for Windows Event XML or JSON format
        return '<Event xmlns=' in log_line or (
            self._is_json(log_line) and 'EventID' in log_line
        )
    
    def _is_json(self, log_line: str) -> bool:
        try:
            json.loads(log_line)
            return True
        except:
            return False
    
    def parse(self, log_line: str) -> Optional[ParsedLog]:
        if self._is_json(log_line):
            return self._parse_json_event(log_line)
        else:
            return self._parse_xml_event(log_line)
    
    def _parse_json_event(self, log_line: str) -> Optional[ParsedLog]:
        try:
            data = json.loads(log_line)
        except:
            return None
        
        # Extract Windows Event fields
        event_id = data.get('EventID', 'Unknown')
        level = data.get('Level', data.get('Keywords', 'Info'))
        source = data.get('Source', data.get('ProviderName', 'Windows'))
        message = data.get('Message', str(data))
        
        # Extract timestamp
        timestamp_str = data.get('TimeCreated', data.get('@timestamp'))
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.utcnow()
        
        return ParsedLog(
            timestamp=timestamp,
            source=source,
            event_type=f"Windows-{event_id}",
            severity=self._windows_level_to_severity(level),
            message=message,
            raw_data=log_line,
            parsed_fields=data,
            metadata={
                'format': 'Windows Event JSON',
                'parser_version': '1.0'
            }
        )
    
    def _parse_xml_event(self, log_line: str) -> Optional[ParsedLog]:
        # Simplified XML parsing - in production, use proper XML parser
        # For now, extract basic fields with regex
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(log_line)
            
            # Extract basic fields
            event_id = root.find('.//{http://schemas.microsoft.com/win/2004/08/events/event}EventID')
            level = root.find('.//{http://schemas.microsoft.com/win/2004/08/events/event}Level')
            provider = root.find('.//{http://schemas.microsoft.com/win/2004/08/events/event}Provider')
            time_created = root.find('.//{http://schemas.microsoft.com/win/2004/08/events/event}TimeCreated')
            
            event_id_val = event_id.text if event_id is not None else 'Unknown'
            level_val = level.text if level is not None else '4'
            provider_val = provider.get('Name') if provider is not None else 'Windows'
            
            # Parse timestamp
            if time_created is not None:
                timestamp_str = time_created.get('SystemTime')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()
            
            return ParsedLog(
                timestamp=timestamp,
                source=provider_val,
                event_type=f"Windows-{event_id_val}",
                severity=self._windows_level_to_severity(int(level_val)),
                message=f"Windows Event {event_id_val}",
                raw_data=log_line,
                parsed_fields={
                    'event_id': event_id_val,
                    'level': level_val,
                    'provider': provider_val
                },
                metadata={
                    'format': 'Windows Event XML',
                    'parser_version': '1.0'
                }
            )
        except ET.ParseError:
            return None
    
    def _windows_level_to_severity(self, level: Union[int, str]) -> str:
        """Convert Windows event level to severity"""
        level_map = {
            1: 'Critical',  # Critical
            2: 'Error',     # Error
            3: 'Warning',   # Warning
            4: 'Info',      # Information
            5: 'Debug',     # Verbose
        }
        
        if isinstance(level, str):
            level = int(level) if level.isdigit() else 4
        
        return level_map.get(level, 'Info')


class LogParserFactory:
    """Factory for creating appropriate log parsers"""
    
    def __init__(self):
        self.parsers = [
            CEFParser(),
            LEEFParser(),
            JSONParser(),
            SyslogParser(),
            WindowsEventParser()
        ]
    
    def get_parser(self, log_line: str) -> Optional[LogParser]:
        """Get appropriate parser for the log line"""
        for parser in self.parsers:
            if parser.can_parse(log_line):
                return parser
        return None
    
    def parse_log(self, log_line: str) -> Optional[ParsedLog]:
        """Parse log line with appropriate parser"""
        parser = self.get_parser(log_line)
        if parser:
            return parser.parse(log_line)
        return None
    
    def add_custom_parser(self, parser: LogParser):
        """Add custom parser to the factory"""
        self.parsers.insert(0, parser)  # Add at beginning for priority