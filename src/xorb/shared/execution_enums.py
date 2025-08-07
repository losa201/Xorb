from enum import Enum

class ScanType(str, Enum):
    DISCOVERY = "discovery"
    PORT_SCAN = "port_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_SCAN = "web_scan"
    STEALTH_SCAN = "stealth_scan"

class ExploitType(str, Enum):
    WEB_EXPLOIT = "web_exploit"
    NETWORK_EXPLOIT = "network_exploit"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"

class EvidenceType(str, Enum):
    SCREENSHOT = "screenshot"
    HTTP_RESPONSE = "http_response"
    COMMAND_OUTPUT = "command_output"
    FILE_CONTENT = "file_content"
    NETWORK_PACKET = "network_packet"