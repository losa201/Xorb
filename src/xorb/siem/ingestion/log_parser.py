"""
Log parsing utilities for various log formats
Supports multiple log formats including syslog, JSON, CEF, and custom formats
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import ipaddress

logger = logging.getLogger(__name__)


class LogFormat(Enum):
    """Supported log formats"""
    SYSLOG = "syslog"
    JSON = "json"
    CEF = "cef"
    APACHE = "apache"
    NGINX = "nginx"
    IIS = "iis"
    FIREWALL = "firewall"
    CUSTOM = "custom"


@dataclass
class ParsedLog:
    """Parsed log entry with structured data"""
    timestamp: datetime
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    action: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    user: Optional[str] = None
    process: Optional[str] = None
    pid: Optional[int] = None
    facility: Optional[str] = None
    fields: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    raw_log: str = ""

    def __post_init__(self):
        if self.fields is None:
            self.fields = {}
        if self.metadata is None:
            self.metadata = {}


class BaseLogParser:
    """Base class for log parsers"""

    def __init__(self, format_type: LogFormat):
        self.format_type = format_type
        self.patterns = {}
        self.field_mappings = {}

    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse a log line into structured data"""
        try:
            # Create base parsed log with current timestamp as fallback
            parsed_log = ParsedLog(
                timestamp=datetime.now(),
                raw_log=log_line.strip()
            )

            # Extract basic information common to most logs
            self._extract_common_fields(parsed_log, log_line)

            return parsed_log

        except Exception as e:
            logger.warning(f"Failed to parse log line: {e}")
            return None

    def _extract_common_fields(self, parsed_log: ParsedLog, log_line: str):
        """Extract common fields from log line"""
        # Extract IP addresses
        ips = self.extract_ip_addresses(log_line)
        if len(ips) >= 1:
            parsed_log.source_ip = ips[0]
        if len(ips) >= 2:
            parsed_log.destination_ip = ips[1]

        # Extract common port patterns
        port_pattern = r':(\d{1,5})\b'
        ports = re.findall(port_pattern, log_line)
        if ports:
            try:
                parsed_log.source_port = int(ports[0])
                if len(ports) > 1:
                    parsed_log.destination_port = int(ports[1])
            except ValueError:
                pass

        # Extract severity keywords
        severity_keywords = {
            'emerg': 'emergency', 'alert': 'alert', 'crit': 'critical',
            'err': 'error', 'warn': 'warning', 'notice': 'notice',
            'info': 'info', 'debug': 'debug'
        }

        log_lower = log_line.lower()
        for keyword, severity in severity_keywords.items():
            if keyword in log_lower:
                parsed_log.severity = severity
                break

        # Extract protocol information
        protocols = ['tcp', 'udp', 'icmp', 'http', 'https', 'ftp', 'ssh']
        for protocol in protocols:
            if protocol.lower() in log_lower:
                parsed_log.protocol = protocol.upper()
                break

    def extract_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Extract timestamp from string"""
        timestamp_patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%b %d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%b/%Y:%H:%M:%S",
        ]

        for pattern in timestamp_patterns:
            try:
                return datetime.strptime(timestamp_str, pattern)
            except ValueError:
                continue

        return None

    def extract_ip_addresses(self, text: str) -> List[str]:
        """Extract IP addresses from text"""
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)

        # Validate IP addresses
        valid_ips = []
        for ip in ips:
            try:
                ipaddress.ip_address(ip)
                valid_ips.append(ip)
            except ValueError:
                continue

        return valid_ips


class SyslogParser(BaseLogParser):
    """RFC5424 Syslog parser"""

    def __init__(self):
        super().__init__(LogFormat.SYSLOG)
        # RFC5424 pattern
        self.patterns['rfc5424'] = re.compile(
            r'^<(?P<priority>\d+)>(?P<version>\d+)?\s+'
            r'(?P<timestamp>\S+)\s+'
            r'(?P<hostname>\S+)\s+'
            r'(?P<app_name>\S+)\s+'
            r'(?P<proc_id>\S+)\s+'
            r'(?P<msg_id>\S+)\s+'
            r'(?P<structured_data>\[.*?\])?\s*'
            r'(?P<message>.*)'
        )

        # BSD Syslog pattern
        self.patterns['bsd'] = re.compile(
            r'^<(?P<priority>\d+)>'
            r'(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'
            r'(?P<hostname>\S+)\s+'
            r'(?P<tag>[^:\[\]]+)(\[(?P<pid>\d+)\])?:\s*'
            r'(?P<message>.*)'
        )

    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse syslog format"""
        try:
            # Try RFC5424 first
            match = self.patterns['rfc5424'].match(log_line.strip())
            if not match:
                # Try BSD syslog
                match = self.patterns['bsd'].match(log_line.strip())

            if not match:
                return None

            groups = match.groupdict()

            # Extract priority and calculate facility/severity
            priority = int(groups.get('priority', 0))
            facility = priority // 8
            severity = priority % 8

            # Map severity to text
            severity_map = {
                0: "Emergency", 1: "Alert", 2: "Critical", 3: "Error",
                4: "Warning", 5: "Notice", 6: "Info", 7: "Debug"
            }

            # Parse timestamp
            timestamp_str = groups.get('timestamp', '')
            timestamp = self.extract_timestamp(timestamp_str)
            if not timestamp:
                timestamp = datetime.utcnow()

            # Extract process info
            process = groups.get('app_name') or groups.get('tag', '')
            pid = groups.get('proc_id') or groups.get('pid')
            if pid and pid != '-':
                try:
                    pid = int(pid)
                except ValueError:
                    pid = None
            else:
                pid = None

            message = groups.get('message', '').strip()

            # Extract IP addresses from message
            ips = self.extract_ip_addresses(message)
            source_ip = ips[0] if ips else None
            destination_ip = ips[1] if len(ips) > 1 else None

            parsed_log = ParsedLog(
                timestamp=timestamp,
                source_ip=source_ip,
                destination_ip=destination_ip,
                severity=severity_map.get(severity, "Unknown"),
                message=message,
                process=process,
                pid=pid,
                facility=f"facility{facility}",
                fields={
                    'hostname': groups.get('hostname'),
                    'version': groups.get('version'),
                    'msg_id': groups.get('msg_id'),
                    'structured_data': groups.get('structured_data'),
                    'priority': priority,
                    'facility': facility,
                    'severity_level': severity
                },
                raw_log=log_line
            )

            return parsed_log

        except Exception as e:
            logger.error(f"Error parsing syslog: {e}")
            return None


class JSONParser(BaseLogParser):
    """JSON log parser"""

    def __init__(self):
        super().__init__(LogFormat.JSON)

    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse JSON format"""
        try:
            data = json.loads(log_line.strip())

            # Extract common fields
            timestamp_str = data.get('timestamp') or data.get('time') or data.get('@timestamp')
            timestamp = self.extract_timestamp(timestamp_str) if timestamp_str else datetime.utcnow()

            message = data.get('message') or data.get('msg') or str(data)

            parsed_log = ParsedLog(
                timestamp=timestamp,
                source_ip=data.get('source_ip') or data.get('src_ip'),
                destination_ip=data.get('dest_ip') or data.get('dst_ip'),
                source_port=data.get('source_port') or data.get('src_port'),
                destination_port=data.get('dest_port') or data.get('dst_port'),
                protocol=data.get('protocol'),
                action=data.get('action'),
                severity=data.get('severity') or data.get('level'),
                message=message,
                user=data.get('user') or data.get('username'),
                process=data.get('process') or data.get('service'),
                pid=data.get('pid'),
                fields=data,
                raw_log=log_line
            )

            return parsed_log

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON log: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing JSON log: {e}")
            return None


class CEFParser(BaseLogParser):
    """Common Event Format (CEF) parser"""

    def __init__(self):
        super().__init__(LogFormat.CEF)
        self.cef_pattern = re.compile(
            r'^CEF:(?P<version>\d+)\|'
            r'(?P<device_vendor>[^|]*)\|'
            r'(?P<device_product>[^|]*)\|'
            r'(?P<device_version>[^|]*)\|'
            r'(?P<signature_id>[^|]*)\|'
            r'(?P<name>[^|]*)\|'
            r'(?P<severity>[^|]*)\|'
            r'(?P<extension>.*)'
        )

    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse CEF format"""
        try:
            match = self.cef_pattern.match(log_line.strip())
            if not match:
                return None

            groups = match.groupdict()

            # Parse extension fields
            extension = groups.get('extension', '')
            extension_fields = self._parse_cef_extension(extension)

            # Extract timestamp
            timestamp_str = extension_fields.get('rt') or extension_fields.get('end')
            timestamp = self.extract_timestamp(timestamp_str) if timestamp_str else datetime.utcnow()

            parsed_log = ParsedLog(
                timestamp=timestamp,
                source_ip=extension_fields.get('src'),
                destination_ip=extension_fields.get('dst'),
                source_port=self._safe_int(extension_fields.get('spt')),
                destination_port=self._safe_int(extension_fields.get('dpt')),
                protocol=extension_fields.get('proto'),
                action=extension_fields.get('act'),
                severity=groups.get('severity'),
                message=groups.get('name'),
                user=extension_fields.get('suser') or extension_fields.get('duser'),
                fields={
                    'device_vendor': groups.get('device_vendor'),
                    'device_product': groups.get('device_product'),
                    'device_version': groups.get('device_version'),
                    'signature_id': groups.get('signature_id'),
                    'cef_version': groups.get('version'),
                    **extension_fields
                },
                raw_log=log_line
            )

            return parsed_log

        except Exception as e:
            logger.error(f"Error parsing CEF log: {e}")
            return None

    def _parse_cef_extension(self, extension: str) -> Dict[str, str]:
        """Parse CEF extension fields"""
        fields = {}
        # Simple key=value parsing (should be more robust for production)
        for pair in extension.split(' '):
            if '=' in pair:
                key, value = pair.split('=', 1)
                fields[key] = value
        return fields

    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int"""
        try:
            return int(value) if value else None
        except (ValueError, TypeError):
            return None


class ApacheParser(BaseLogParser):
    """Apache access log parser"""

    def __init__(self):
        super().__init__(LogFormat.APACHE)
        # Common Log Format
        self.patterns['common'] = re.compile(
            r'^(?P<remote_host>\S+)\s+'
            r'(?P<remote_logname>\S+)\s+'
            r'(?P<remote_user>\S+)\s+'
            r'\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<request>[^"]*)"\s+'
            r'(?P<status>\d+)\s+'
            r'(?P<bytes_sent>\S+)'
        )

        # Combined Log Format
        self.patterns['combined'] = re.compile(
            r'^(?P<remote_host>\S+)\s+'
            r'(?P<remote_logname>\S+)\s+'
            r'(?P<remote_user>\S+)\s+'
            r'\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<request>[^"]*)"\s+'
            r'(?P<status>\d+)\s+'
            r'(?P<bytes_sent>\S+)\s+'
            r'"(?P<referer>[^"]*)"\s+'
            r'"(?P<user_agent>[^"]*)"'
        )

    def parse(self, log_line: str) -> Optional[ParsedLog]:
        """Parse Apache log format"""
        try:
            # Try combined first, then common
            match = self.patterns['combined'].match(log_line.strip())
            if not match:
                match = self.patterns['common'].match(log_line.strip())

            if not match:
                return None

            groups = match.groupdict()

            # Parse timestamp (Apache format: day/month/year:hour:minute:second zone)
            timestamp_str = groups.get('timestamp', '')
            timestamp = self.extract_timestamp(timestamp_str.split(' ')[0]) if timestamp_str else datetime.utcnow()

            # Extract request details
            request = groups.get('request', '')
            method, url, protocol = '', '', ''
            if request:
                parts = request.split(' ')
                method = parts[0] if len(parts) > 0 else ''
                url = parts[1] if len(parts) > 1 else ''
                protocol = parts[2] if len(parts) > 2 else ''

            parsed_log = ParsedLog(
                timestamp=timestamp,
                source_ip=groups.get('remote_host'),
                message=f"{method} {url} {groups.get('status')}",
                user=groups.get('remote_user') if groups.get('remote_user') != '-' else None,
                fields={
                    'method': method,
                    'url': url,
                    'protocol': protocol,
                    'status_code': self._safe_int(groups.get('status')),
                    'bytes_sent': self._safe_int(groups.get('bytes_sent')),
                    'referer': groups.get('referer'),
                    'user_agent': groups.get('user_agent'),
                    'remote_logname': groups.get('remote_logname')
                },
                raw_log=log_line
            )

            return parsed_log

        except Exception as e:
            logger.error(f"Error parsing Apache log: {e}")
            return None

    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int"""
        try:
            return int(value) if value and value != '-' else None
        except (ValueError, TypeError):
            return None


class LogParserFactory:
    """Factory for creating log parsers"""

    def __init__(self):
        self.parsers = {
            LogFormat.SYSLOG: SyslogParser(),
            LogFormat.JSON: JSONParser(),
            LogFormat.CEF: CEFParser(),
            LogFormat.APACHE: ApacheParser(),
        }
        self.auto_detect_patterns = [
            (LogFormat.JSON, lambda line: line.strip().startswith('{')),
            (LogFormat.CEF, lambda line: line.startswith('CEF:')),
            (LogFormat.SYSLOG, lambda line: re.match(r'^<\d+>', line.strip())),
            (LogFormat.APACHE, lambda line: '[' in line and '"' in line),
        ]

    def parse_log(self, log_line: str, format_hint: Optional[LogFormat] = None) -> Optional[ParsedLog]:
        """Parse log line with optional format hint"""
        if not log_line or not log_line.strip():
            return None

        # Use specific parser if format is provided
        if format_hint and format_hint in self.parsers:
            return self.parsers[format_hint].parse(log_line)

        # Auto-detect format
        detected_format = self.detect_format(log_line)
        if detected_format and detected_format in self.parsers:
            return self.parsers[detected_format].parse(log_line)

        # Fallback to simple parsing
        return self._simple_parse(log_line)

    def detect_format(self, log_line: str) -> Optional[LogFormat]:
        """Auto-detect log format"""
        for log_format, pattern_func in self.auto_detect_patterns:
            try:
                if pattern_func(log_line):
                    return log_format
            except Exception:
                continue

        return None

    def _simple_parse(self, log_line: str) -> ParsedLog:
        """Simple fallback parser for unknown formats"""
        # Extract timestamp from beginning if present
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', log_line)
        timestamp = None
        if timestamp_match:
            try:
                timestamp = datetime.fromisoformat(timestamp_match.group(1).replace('T', ' '))
            except ValueError:
                pass

        if not timestamp:
            timestamp = datetime.utcnow()

        # Extract severity/level
        severity = None
        for level in ['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG']:
            if level in log_line.upper():
                severity = level
                break

        # Extract IP addresses
        ips = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', log_line)
        source_ip = ips[0] if ips else None
        destination_ip = ips[1] if len(ips) > 1 else None

        return ParsedLog(
            timestamp=timestamp,
            source_ip=source_ip,
            destination_ip=destination_ip,
            severity=severity,
            message=log_line.strip(),
            raw_log=log_line
        )

    def add_parser(self, format_type: LogFormat, parser: BaseLogParser):
        """Add custom parser"""
        self.parsers[format_type] = parser

    def get_supported_formats(self) -> List[LogFormat]:
        """Get list of supported log formats"""
        return list(self.parsers.keys())
