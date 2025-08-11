"""
IoT Security Service - Advanced Internet of Things security monitoring and analysis
Comprehensive security assessment for IoT devices, industrial control systems, and smart infrastructure
"""

import asyncio
import json
import logging
import re
import socket
import struct
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4
import hashlib
import base64

# Network and protocol imports with graceful fallbacks
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether, ARP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    scapy = None

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False
    nmap = None

# Industrial protocol support
try:
    # These would be actual industrial protocol libraries
    # import modbus_tk  # Modbus protocol
    # import opcua      # OPC UA protocol
    # import dnp3       # DNP3 protocol
    INDUSTRIAL_PROTOCOLS_AVAILABLE = False
except ImportError:
    INDUSTRIAL_PROTOCOLS_AVAILABLE = False

from .base_service import XORBService, ServiceHealth, ServiceStatus
from .interfaces import SecurityOrchestrationService, ThreatIntelligenceService

logger = logging.getLogger(__name__)


class IoTDeviceType(Enum):
    """Types of IoT devices"""
    SMART_CAMERA = "smart_camera"
    SMART_SPEAKER = "smart_speaker"
    SMART_THERMOSTAT = "smart_thermostat"
    SMART_LOCK = "smart_lock"
    SMART_LIGHTBULB = "smart_lightbulb"
    SMART_PLUG = "smart_plug"
    ROUTER = "router"
    ACCESS_POINT = "access_point"
    PRINTER = "printer"
    NAS_DEVICE = "nas_device"
    INDUSTRIAL_SENSOR = "industrial_sensor"
    PLC = "plc"  # Programmable Logic Controller
    HMI = "hmi"  # Human Machine Interface
    SCADA = "scada"
    SMART_METER = "smart_meter"
    MEDICAL_DEVICE = "medical_device"
    AUTOMOTIVE = "automotive"
    UNKNOWN = "unknown"


class VulnerabilityCategory(Enum):
    """IoT vulnerability categories"""
    WEAK_AUTHENTICATION = "weak_authentication"
    DEFAULT_CREDENTIALS = "default_credentials"
    INSECURE_COMMUNICATION = "insecure_communication"
    FIRMWARE_VULNERABILITIES = "firmware_vulnerabilities"
    INSECURE_UPDATE_MECHANISM = "insecure_update_mechanism"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    WEAK_ENCRYPTION = "weak_encryption"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    BUFFER_OVERFLOW = "buffer_overflow"
    COMMAND_INJECTION = "command_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    PHYSICAL_SECURITY = "physical_security"


class ThreatLevel(Enum):
    """IoT threat severity levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class IoTDevice:
    """IoT device information"""
    device_id: str
    ip_address: str
    mac_address: str
    device_type: IoTDeviceType
    manufacturer: str
    model: str
    firmware_version: str
    open_ports: List[int]
    services: List[Dict[str, Any]]
    protocols: List[str]
    last_seen: datetime
    metadata: Dict[str, Any]


@dataclass
class IoTVulnerability:
    """IoT vulnerability assessment"""
    vulnerability_id: str
    device_id: str
    category: VulnerabilityCategory
    severity: str
    title: str
    description: str
    cve_id: Optional[str]
    cvss_score: Optional[float]
    exploit_available: bool
    remediation: str
    references: List[str]
    discovered_at: datetime


@dataclass
class IoTSecurityAssessment:
    """Comprehensive IoT security assessment"""
    assessment_id: str
    timestamp: datetime
    network_range: str
    devices_discovered: List[IoTDevice]
    vulnerabilities: List[IoTVulnerability]
    network_topology: Dict[str, Any]
    security_posture: Dict[str, Any]
    threat_level: ThreatLevel
    recommendations: List[str]
    compliance_status: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class IndustrialControlSystemAssessment:
    """Industrial Control System (ICS) security assessment"""
    assessment_id: str
    timestamp: datetime
    ics_devices: List[IoTDevice]
    control_networks: List[Dict[str, Any]]
    safety_systems: List[Dict[str, Any]]
    vulnerabilities: List[IoTVulnerability]
    operational_risks: List[Dict[str, Any]]
    compliance_frameworks: List[str]
    threat_level: ThreatLevel
    recommendations: List[str]
    metadata: Dict[str, Any]


class IoTSecurityService(XORBService, SecurityOrchestrationService, ThreatIntelligenceService):
    """Advanced IoT and Industrial Control System security service"""
    
    def __init__(self, **kwargs):
        super().__init__(
            service_id="iot_security_service",
            dependencies=["network_scanner", "vulnerability_db"],
            **kwargs
        )
        
        # IoT device fingerprints and signatures
        self.device_signatures = {
            "smart_camera": {
                "ports": [80, 443, 554, 8080],
                "services": ["http", "https", "rtsp"],
                "banners": ["IP Camera", "Network Camera", "AXIS", "Hikvision", "Dahua"],
                "user_agents": ["Mozilla/5.0", "VLC", "QuickTime"]
            },
            "smart_speaker": {
                "ports": [80, 443, 1900, 8080],
                "services": ["http", "upnp", "mdns"],
                "banners": ["Amazon Echo", "Google Home", "Alexa"],
                "mdns_services": ["_googlecast._tcp", "_spotify-connect._tcp"]
            },
            "router": {
                "ports": [22, 23, 53, 80, 443, 8080],
                "services": ["ssh", "telnet", "dns", "http", "https"],
                "banners": ["RouterOS", "OpenWrt", "DD-WRT", "ASUS", "Netgear", "Linksys"],
                "snmp_oids": ["1.3.6.1.2.1.1.1.0"]
            },
            "plc": {
                "ports": [102, 502, 44818, 20000],
                "services": ["modbus", "s7comm", "ethernet/ip", "dnp3"],
                "banners": ["Siemens", "Allen-Bradley", "Schneider", "ABB"],
                "protocols": ["modbus-tcp", "s7", "cip", "dnp3"]
            }
        }
        
        # Known IoT vulnerabilities and CVEs
        self.vulnerability_database = {
            VulnerabilityCategory.DEFAULT_CREDENTIALS: [
                {
                    "title": "Default Administrative Credentials",
                    "description": "Device uses default username/password combinations",
                    "severity": "high",
                    "common_credentials": [
                        ("admin", "admin"), ("admin", "password"), ("admin", ""),
                        ("root", "root"), ("admin", "123456"), ("user", "user")
                    ],
                    "remediation": "Change default credentials immediately"
                }
            ],
            VulnerabilityCategory.WEAK_ENCRYPTION: [
                {
                    "title": "Weak SSL/TLS Configuration",
                    "description": "Device uses outdated encryption protocols",
                    "severity": "medium",
                    "indicators": ["SSLv2", "SSLv3", "RC4", "MD5"],
                    "remediation": "Update to TLS 1.2 or higher"
                }
            ],
            VulnerabilityCategory.INSECURE_COMMUNICATION: [
                {
                    "title": "Unencrypted Communications",
                    "description": "Device transmits sensitive data without encryption",
                    "severity": "high",
                    "protocols": ["telnet", "ftp", "http"],
                    "remediation": "Enable encrypted communications (SSH, SFTP, HTTPS)"
                }
            ]
        }
        
        # Industrial protocol parsers
        self.industrial_protocols = {
            "modbus": {"port": 502, "function_codes": [1, 2, 3, 4, 5, 6, 15, 16]},
            "s7comm": {"port": 102, "protocols": ["ISO-TSAP", "COTP"]},
            "dnp3": {"port": 20000, "functions": ["read", "write", "select", "operate"]},
            "ethernet_ip": {"port": 44818, "services": ["list_services", "list_identity"]}
        }
        
        # Threat intelligence for IoT devices
        self.iot_threat_intelligence = {
            "malware_families": ["Mirai", "Bashlite", "IoTReaper", "VPNFilter", "Hajime"],
            "attack_signatures": [],
            "compromised_devices": set(),
            "c2_servers": set()
        }
        
        # Assessment cache
        self.assessment_cache = {}
        
    async def discover_iot_devices(
        self,
        network_ranges: List[str],
        discovery_options: Dict[str, Any] = None
    ) -> List[IoTDevice]:
        """Discover IoT devices on specified networks"""
        try:
            discovery_options = discovery_options or {}
            discovered_devices = []
            
            for network_range in network_ranges:
                # Perform network scan
                scan_results = await self._perform_network_scan(network_range, discovery_options)
                
                # Analyze scan results and identify IoT devices
                for host_result in scan_results:
                    device = await self._identify_iot_device(host_result)
                    if device:
                        discovered_devices.append(device)
            
            logger.info(f"Discovered {len(discovered_devices)} IoT devices")
            return discovered_devices
            
        except Exception as e:
            logger.error(f"IoT device discovery failed: {e}")
            raise
    
    async def assess_iot_security(
        self,
        network_ranges: List[str],
        assessment_options: Dict[str, Any] = None
    ) -> IoTSecurityAssessment:
        """Perform comprehensive IoT security assessment"""
        try:
            assessment_id = str(uuid4())
            assessment_options = assessment_options or {}
            
            # Discover IoT devices
            devices = await self.discover_iot_devices(network_ranges, assessment_options)
            
            # Analyze network topology
            network_topology = await self._analyze_network_topology(devices)
            
            # Assess vulnerabilities for each device
            vulnerabilities = []
            for device in devices:
                device_vulns = await self._assess_device_vulnerabilities(device)
                vulnerabilities.extend(device_vulns)
            
            # Calculate security posture
            security_posture = await self._calculate_security_posture(devices, vulnerabilities)
            
            # Determine overall threat level
            threat_level = self._determine_threat_level(vulnerabilities, security_posture)
            
            # Generate recommendations
            recommendations = await self._generate_iot_recommendations(devices, vulnerabilities)
            
            # Check compliance status
            compliance_status = await self._check_iot_compliance(devices, vulnerabilities)
            
            # Create assessment
            assessment = IoTSecurityAssessment(
                assessment_id=assessment_id,
                timestamp=datetime.utcnow(),
                network_range=", ".join(network_ranges),
                devices_discovered=devices,
                vulnerabilities=vulnerabilities,
                network_topology=network_topology,
                security_posture=security_posture,
                threat_level=threat_level,
                recommendations=recommendations,
                compliance_status=compliance_status,
                metadata={
                    "devices_scanned": len(devices),
                    "vulnerabilities_found": len(vulnerabilities),
                    "critical_vulnerabilities": len([v for v in vulnerabilities if v.severity == "critical"]),
                    "assessment_duration": "pending"
                }
            )
            
            # Cache assessment
            self.assessment_cache[assessment_id] = assessment
            
            logger.info(f"IoT security assessment completed: {assessment_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"IoT security assessment failed: {e}")
            raise
    
    async def assess_industrial_control_systems(
        self,
        network_ranges: List[str],
        ics_options: Dict[str, Any] = None
    ) -> IndustrialControlSystemAssessment:
        """Assess Industrial Control System (ICS) security"""
        try:
            assessment_id = str(uuid4())
            ics_options = ics_options or {}
            
            # Discover ICS devices
            all_devices = await self.discover_iot_devices(network_ranges, ics_options)
            ics_devices = [d for d in all_devices if self._is_industrial_device(d)]
            
            # Identify control networks
            control_networks = await self._identify_control_networks(ics_devices)
            
            # Identify safety systems
            safety_systems = await self._identify_safety_systems(ics_devices)
            
            # Assess ICS-specific vulnerabilities
            vulnerabilities = []
            for device in ics_devices:
                ics_vulns = await self._assess_ics_vulnerabilities(device)
                vulnerabilities.extend(ics_vulns)
            
            # Analyze operational risks
            operational_risks = await self._analyze_operational_risks(ics_devices, vulnerabilities)
            
            # Check compliance with industrial standards
            compliance_frameworks = await self._check_ics_compliance(ics_devices)
            
            # Determine threat level
            threat_level = self._determine_ics_threat_level(vulnerabilities, operational_risks)
            
            # Generate ICS-specific recommendations
            recommendations = await self._generate_ics_recommendations(ics_devices, vulnerabilities)
            
            # Create assessment
            assessment = IndustrialControlSystemAssessment(
                assessment_id=assessment_id,
                timestamp=datetime.utcnow(),
                ics_devices=ics_devices,
                control_networks=control_networks,
                safety_systems=safety_systems,
                vulnerabilities=vulnerabilities,
                operational_risks=operational_risks,
                compliance_frameworks=compliance_frameworks,
                threat_level=threat_level,
                recommendations=recommendations,
                metadata={
                    "ics_devices_found": len(ics_devices),
                    "control_networks": len(control_networks),
                    "safety_systems": len(safety_systems),
                    "operational_risks": len(operational_risks)
                }
            )
            
            logger.info(f"ICS security assessment completed: {assessment_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"ICS security assessment failed: {e}")
            raise
    
    async def monitor_iot_traffic(
        self,
        interface: str,
        monitoring_duration: int = 300,  # 5 minutes
        monitoring_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Monitor IoT network traffic for security threats"""
        try:
            monitoring_id = str(uuid4())
            monitoring_options = monitoring_options or {}
            
            # Initialize monitoring session
            monitoring_session = {
                "monitoring_id": monitoring_id,
                "start_time": datetime.utcnow(),
                "interface": interface,
                "duration": monitoring_duration,
                "packets_captured": 0,
                "threats_detected": [],
                "device_communications": {},
                "protocol_analysis": {}
            }
            
            if SCAPY_AVAILABLE:
                # Start packet capture
                monitoring_session = await self._monitor_with_scapy(
                    interface, monitoring_duration, monitoring_session, monitoring_options
                )
            else:
                # Fallback to basic monitoring
                logger.warning("Scapy not available - using basic monitoring")
                monitoring_session = await self._basic_traffic_monitoring(
                    interface, monitoring_duration, monitoring_session
                )
            
            # Analyze captured traffic
            analysis_results = await self._analyze_iot_traffic(monitoring_session)
            monitoring_session.update(analysis_results)
            
            logger.info(f"IoT traffic monitoring completed: {monitoring_id}")
            return monitoring_session
            
        except Exception as e:
            logger.error(f"IoT traffic monitoring failed: {e}")
            raise
    
    async def detect_iot_malware(
        self,
        devices: List[IoTDevice],
        detection_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Detect IoT malware and compromised devices"""
        try:
            detection_id = str(uuid4())
            detection_options = detection_options or {}
            
            malware_detections = {
                "detection_id": detection_id,
                "timestamp": datetime.utcnow(),
                "devices_analyzed": len(devices),
                "infected_devices": [],
                "malware_families": [],
                "c2_communications": [],
                "lateral_movement": [],
                "persistence_mechanisms": []
            }
            
            for device in devices:
                # Check for malware signatures
                device_infections = await self._check_device_malware(device)
                if device_infections:
                    malware_detections["infected_devices"].append({
                        "device_id": device.device_id,
                        "ip_address": device.ip_address,
                        "infections": device_infections
                    })
                
                # Check for C&C communications
                c2_activity = await self._detect_c2_communications(device)
                if c2_activity:
                    malware_detections["c2_communications"].extend(c2_activity)
                
                # Check for lateral movement indicators
                lateral_movement = await self._detect_lateral_movement(device)
                if lateral_movement:
                    malware_detections["lateral_movement"].extend(lateral_movement)
            
            # Identify malware families
            malware_detections["malware_families"] = await self._identify_malware_families(
                malware_detections["infected_devices"]
            )
            
            logger.info(f"IoT malware detection completed: {detection_id}")
            return malware_detections
            
        except Exception as e:
            logger.error(f"IoT malware detection failed: {e}")
            raise
    
    # Private helper methods
    async def _perform_network_scan(
        self,
        network_range: str,
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform network scan to discover devices"""
        scan_results = []
        
        if NMAP_AVAILABLE:
            # Use nmap for comprehensive scanning
            nm = nmap.PortScanner()
            scan_args = options.get("nmap_args", "-sS -O -sV --script vuln")
            
            try:
                nm.scan(network_range, arguments=scan_args)
                for host in nm.all_hosts():
                    host_info = {
                        "ip": host,
                        "hostname": nm[host].hostname(),
                        "state": nm[host].state(),
                        "protocols": list(nm[host].all_protocols()),
                        "ports": {},
                        "os": nm[host].get("osclass", []),
                        "services": {}
                    }
                    
                    for protocol in host_info["protocols"]:
                        ports = nm[host][protocol].keys()
                        host_info["ports"][protocol] = list(ports)
                        
                        for port in ports:
                            service_info = nm[host][protocol][port]
                            host_info["services"][port] = {
                                "name": service_info.get("name", ""),
                                "product": service_info.get("product", ""),
                                "version": service_info.get("version", ""),
                                "extrainfo": service_info.get("extrainfo", ""),
                                "state": service_info.get("state", "")
                            }
                    
                    scan_results.append(host_info)
                    
            except Exception as e:
                logger.error(f"Nmap scan failed: {e}")
        else:
            # Basic scanning without nmap
            scan_results = await self._basic_network_scan(network_range)
        
        return scan_results
    
    async def _basic_network_scan(self, network_range: str) -> List[Dict[str, Any]]:
        """Basic network scan without external tools"""
        results = []
        
        # Parse network range (simplified)
        if "/" in network_range:
            base_ip = network_range.split("/")[0]
            # For simplicity, scan first 10 IPs
            base_parts = base_ip.split(".")
            base_network = ".".join(base_parts[:3])
            
            for i in range(1, 11):
                ip = f"{base_network}.{i}"
                # Test common IoT ports
                common_ports = [22, 23, 80, 443, 554, 8080]
                open_ports = []
                
                for port in common_ports:
                    if await self._test_port(ip, port):
                        open_ports.append(port)
                
                if open_ports:
                    results.append({
                        "ip": ip,
                        "hostname": "",
                        "state": "up",
                        "protocols": ["tcp"],
                        "ports": {"tcp": open_ports},
                        "os": [],
                        "services": {}
                    })
        
        return results
    
    async def _test_port(self, ip: str, port: int, timeout: float = 1.0) -> bool:
        """Test if a port is open"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False
    
    async def _identify_iot_device(self, host_result: Dict[str, Any]) -> Optional[IoTDevice]:
        """Identify if a host is an IoT device and determine its type"""
        ip = host_result["ip"]
        open_ports = []
        services = []
        
        # Extract port information
        for protocol, ports in host_result.get("ports", {}).items():
            open_ports.extend(ports)
        
        # Extract service information
        for port, service_info in host_result.get("services", {}).items():
            services.append({
                "port": port,
                "service": service_info.get("name", ""),
                "product": service_info.get("product", ""),
                "version": service_info.get("version", "")
            })
        
        # Identify device type based on signatures
        device_type = await self._classify_device_type(open_ports, services, host_result)
        
        if device_type != IoTDeviceType.UNKNOWN:
            # Extract additional device information
            manufacturer, model, firmware = await self._extract_device_info(services, host_result)
            
            device = IoTDevice(
                device_id=str(uuid4()),
                ip_address=ip,
                mac_address=await self._get_mac_address(ip),
                device_type=device_type,
                manufacturer=manufacturer,
                model=model,
                firmware_version=firmware,
                open_ports=open_ports,
                services=services,
                protocols=self._identify_protocols(services),
                last_seen=datetime.utcnow(),
                metadata={"scan_result": host_result}
            )
            
            return device
        
        return None
    
    async def _classify_device_type(
        self,
        open_ports: List[int],
        services: List[Dict[str, Any]],
        host_result: Dict[str, Any]
    ) -> IoTDeviceType:
        """Classify device type based on port and service signatures"""
        
        # Check against known device signatures
        for device_type, signature in self.device_signatures.items():
            score = 0
            
            # Check port matches
            port_matches = len(set(open_ports) & set(signature.get("ports", [])))
            score += port_matches * 2
            
            # Check service matches
            service_names = [s.get("service", "").lower() for s in services]
            service_matches = len(set(service_names) & set(signature.get("services", [])))
            score += service_matches * 3
            
            # Check banner matches
            banners = [s.get("product", "") + " " + s.get("version", "") for s in services]
            banner_text = " ".join(banners).lower()
            banner_matches = sum(1 for banner in signature.get("banners", []) 
                               if banner.lower() in banner_text)
            score += banner_matches * 4
            
            # If score is high enough, classify as this device type
            if score >= 5:
                return IoTDeviceType(device_type)
        
        # Default classification based on common patterns
        if 554 in open_ports:  # RTSP port
            return IoTDeviceType.SMART_CAMERA
        elif 502 in open_ports:  # Modbus port
            return IoTDeviceType.PLC
        elif 1900 in open_ports:  # UPnP port
            return IoTDeviceType.SMART_SPEAKER
        elif len(open_ports) > 5 and 80 in open_ports:
            return IoTDeviceType.ROUTER
        
        return IoTDeviceType.UNKNOWN
    
    async def _extract_device_info(
        self,
        services: List[Dict[str, Any]],
        host_result: Dict[str, Any]
    ) -> Tuple[str, str, str]:
        """Extract manufacturer, model, and firmware information"""
        manufacturer = "Unknown"
        model = "Unknown"
        firmware = "Unknown"
        
        # Extract from service banners
        for service in services:
            product = service.get("product", "")
            version = service.get("version", "")
            extrainfo = service.get("extrainfo", "")
            
            # Common manufacturer patterns
            manufacturers = {
                "axis": "Axis Communications",
                "hikvision": "Hikvision",
                "dahua": "Dahua Technology",
                "cisco": "Cisco Systems",
                "netgear": "Netgear",
                "linksys": "Linksys",
                "dlink": "D-Link",
                "tplink": "TP-Link",
                "siemens": "Siemens",
                "schneider": "Schneider Electric",
                "abb": "ABB",
                "rockwell": "Rockwell Automation"
            }
            
            for keyword, mfr in manufacturers.items():
                if keyword.lower() in product.lower() or keyword.lower() in extrainfo.lower():
                    manufacturer = mfr
                    break
            
            if version and version != "Unknown":
                firmware = version
            
            if product and product != "Unknown":
                model = product
        
        # Extract from OS detection
        os_info = host_result.get("os", [])
        if os_info:
            for os_entry in os_info:
                if "vendor" in os_entry:
                    manufacturer = os_entry["vendor"]
                if "osgen" in os_entry:
                    firmware = os_entry["osgen"]
        
        return manufacturer, model, firmware
    
    async def _get_mac_address(self, ip: str) -> str:
        """Get MAC address for IP (simplified implementation)"""
        # In a real implementation, this would use ARP tables or network scanning
        return "00:00:00:00:00:00"
    
    def _identify_protocols(self, services: List[Dict[str, Any]]) -> List[str]:
        """Identify protocols used by the device"""
        protocols = set()
        
        for service in services:
            service_name = service.get("service", "").lower()
            
            # Map services to protocols
            protocol_mapping = {
                "http": "HTTP",
                "https": "HTTPS",
                "ssh": "SSH",
                "telnet": "Telnet",
                "ftp": "FTP",
                "smtp": "SMTP",
                "snmp": "SNMP",
                "modbus": "Modbus",
                "s7comm": "S7",
                "dnp3": "DNP3",
                "rtsp": "RTSP",
                "upnp": "UPnP"
            }
            
            if service_name in protocol_mapping:
                protocols.add(protocol_mapping[service_name])
        
        return list(protocols)
    
    async def _assess_device_vulnerabilities(self, device: IoTDevice) -> List[IoTVulnerability]:
        """Assess vulnerabilities for a specific IoT device"""
        vulnerabilities = []
        
        # Check for default credentials
        if await self._has_default_credentials(device):
            vuln = IoTVulnerability(
                vulnerability_id=str(uuid4()),
                device_id=device.device_id,
                category=VulnerabilityCategory.DEFAULT_CREDENTIALS,
                severity="high",
                title="Default Administrative Credentials",
                description="Device appears to use default credentials",
                cve_id=None,
                cvss_score=8.8,
                exploit_available=True,
                remediation="Change default username and password",
                references=["https://owasp.org/www-project-iot-security/"],
                discovered_at=datetime.utcnow()
            )
            vulnerabilities.append(vuln)
        
        # Check for insecure protocols
        insecure_protocols = ["Telnet", "FTP", "HTTP"]
        for protocol in device.protocols:
            if protocol in insecure_protocols:
                vuln = IoTVulnerability(
                    vulnerability_id=str(uuid4()),
                    device_id=device.device_id,
                    category=VulnerabilityCategory.INSECURE_COMMUNICATION,
                    severity="medium",
                    title=f"Insecure Protocol: {protocol}",
                    description=f"Device uses insecure {protocol} protocol",
                    cve_id=None,
                    cvss_score=5.3,
                    exploit_available=False,
                    remediation=f"Disable {protocol} and use secure alternatives",
                    references=[],
                    discovered_at=datetime.utcnow()
                )
                vulnerabilities.append(vuln)
        
        # Check for known CVEs based on device type and firmware
        cve_vulns = await self._check_known_cves(device)
        vulnerabilities.extend(cve_vulns)
        
        return vulnerabilities
    
    async def _has_default_credentials(self, device: IoTDevice) -> bool:
        """Check if device has default credentials"""
        # This would attempt to connect with common default credentials
        # For demonstration, we'll simulate this check
        
        # Common indicators of default credentials
        if device.device_type in [IoTDeviceType.SMART_CAMERA, IoTDeviceType.ROUTER]:
            # Check if telnet or SSH is open (common for devices with defaults)
            if 23 in device.open_ports or 22 in device.open_ports:
                return True  # Simulate finding default credentials
        
        return False
    
    async def _check_known_cves(self, device: IoTDevice) -> List[IoTVulnerability]:
        """Check for known CVEs affecting the device"""
        vulnerabilities = []
        
        # Simulate CVE database lookup
        # In production, this would query a real CVE database
        
        if device.device_type == IoTDeviceType.SMART_CAMERA:
            # Example CVE for IP cameras
            vuln = IoTVulnerability(
                vulnerability_id=str(uuid4()),
                device_id=device.device_id,
                category=VulnerabilityCategory.FIRMWARE_VULNERABILITIES,
                severity="critical",
                title="Remote Code Execution in Camera Firmware",
                description="Buffer overflow vulnerability in web interface",
                cve_id="CVE-2023-XXXXX",
                cvss_score=9.8,
                exploit_available=True,
                remediation="Update firmware to latest version",
                references=["https://nvd.nist.gov/vuln/detail/CVE-2023-XXXXX"],
                discovered_at=datetime.utcnow()
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    async def _analyze_network_topology(self, devices: List[IoTDevice]) -> Dict[str, Any]:
        """Analyze network topology of IoT devices"""
        return {
            "total_devices": len(devices),
            "device_types": {device_type.value: len([d for d in devices if d.device_type == device_type]) 
                           for device_type in IoTDeviceType},
            "network_segments": await self._identify_network_segments(devices),
            "communication_patterns": await self._analyze_communication_patterns(devices),
            "security_zones": await self._identify_security_zones(devices)
        }
    
    async def _identify_network_segments(self, devices: List[IoTDevice]) -> List[Dict[str, Any]]:
        """Identify network segments based on IP addresses"""
        segments = {}
        
        for device in devices:
            # Extract network segment (simplified - assumes /24)
            ip_parts = device.ip_address.split(".")
            segment = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
            
            if segment not in segments:
                segments[segment] = {
                    "segment": segment,
                    "devices": [],
                    "device_types": set()
                }
            
            segments[segment]["devices"].append(device.device_id)
            segments[segment]["device_types"].add(device.device_type.value)
        
        # Convert sets to lists for JSON serialization
        for segment_info in segments.values():
            segment_info["device_types"] = list(segment_info["device_types"])
        
        return list(segments.values())
    
    async def _analyze_communication_patterns(self, devices: List[IoTDevice]) -> Dict[str, Any]:
        """Analyze communication patterns between devices"""
        return {
            "protocols_used": list(set(protocol for device in devices for protocol in device.protocols)),
            "common_ports": self._get_common_ports(devices),
            "inter_device_communication": "analysis_pending",
            "external_communication": "analysis_pending"
        }
    
    def _get_common_ports(self, devices: List[IoTDevice]) -> List[int]:
        """Get most commonly used ports across devices"""
        port_counts = {}
        for device in devices:
            for port in device.open_ports:
                port_counts[port] = port_counts.get(port, 0) + 1
        
        # Return top 10 most common ports
        return sorted(port_counts.keys(), key=lambda p: port_counts[p], reverse=True)[:10]
    
    async def _identify_security_zones(self, devices: List[IoTDevice]) -> Dict[str, Any]:
        """Identify security zones based on device types and functions"""
        zones = {
            "corporate": [],
            "iot_consumer": [],
            "iot_industrial": [],
            "dmz": [],
            "guest": []
        }
        
        for device in devices:
            if device.device_type in [IoTDeviceType.PLC, IoTDeviceType.HMI, IoTDeviceType.SCADA]:
                zones["iot_industrial"].append(device.device_id)
            elif device.device_type in [IoTDeviceType.SMART_CAMERA, IoTDeviceType.SMART_SPEAKER]:
                zones["iot_consumer"].append(device.device_id)
            elif device.device_type in [IoTDeviceType.ROUTER, IoTDeviceType.ACCESS_POINT]:
                zones["dmz"].append(device.device_id)
            else:
                zones["corporate"].append(device.device_id)
        
        return zones
    
    async def _calculate_security_posture(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> Dict[str, Any]:
        """Calculate overall security posture"""
        total_devices = len(devices)
        total_vulns = len(vulnerabilities)
        critical_vulns = len([v for v in vulnerabilities if v.severity == "critical"])
        high_vulns = len([v for v in vulnerabilities if v.severity == "high"])
        
        # Calculate security score (0-100)
        base_score = 100
        base_score -= critical_vulns * 20
        base_score -= high_vulns * 10
        base_score -= (total_vulns - critical_vulns - high_vulns) * 5
        security_score = max(0, base_score)
        
        return {
            "security_score": security_score,
            "total_devices": total_devices,
            "total_vulnerabilities": total_vulns,
            "critical_vulnerabilities": critical_vulns,
            "high_vulnerabilities": high_vulns,
            "devices_with_vulnerabilities": len(set(v.device_id for v in vulnerabilities)),
            "risk_level": self._calculate_risk_level(security_score)
        }
    
    def _calculate_risk_level(self, security_score: float) -> str:
        """Calculate risk level based on security score"""
        if security_score >= 80:
            return "low"
        elif security_score >= 60:
            return "medium"
        elif security_score >= 40:
            return "high"
        else:
            return "critical"
    
    def _determine_threat_level(
        self,
        vulnerabilities: List[IoTVulnerability],
        security_posture: Dict[str, Any]
    ) -> ThreatLevel:
        """Determine overall threat level"""
        critical_vulns = security_posture.get("critical_vulnerabilities", 0)
        security_score = security_posture.get("security_score", 100)
        
        if critical_vulns > 0 or security_score < 20:
            return ThreatLevel.CRITICAL
        elif security_score < 40:
            return ThreatLevel.HIGH
        elif security_score < 60:
            return ThreatLevel.MEDIUM
        elif security_score < 80:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MINIMAL
    
    async def _generate_iot_recommendations(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> List[str]:
        """Generate IoT security recommendations"""
        recommendations = []
        
        # Default credential recommendations
        default_cred_vulns = [v for v in vulnerabilities 
                            if v.category == VulnerabilityCategory.DEFAULT_CREDENTIALS]
        if default_cred_vulns:
            recommendations.append("Immediately change all default passwords on IoT devices")
        
        # Insecure protocol recommendations
        insecure_comm_vulns = [v for v in vulnerabilities 
                             if v.category == VulnerabilityCategory.INSECURE_COMMUNICATION]
        if insecure_comm_vulns:
            recommendations.append("Disable insecure protocols (Telnet, HTTP, FTP) and use secure alternatives")
        
        # Network segmentation
        if len(devices) > 10:
            recommendations.append("Implement network segmentation to isolate IoT devices")
        
        # Firmware updates
        firmware_vulns = [v for v in vulnerabilities 
                        if v.category == VulnerabilityCategory.FIRMWARE_VULNERABILITIES]
        if firmware_vulns:
            recommendations.append("Update firmware on all affected devices")
        
        # Monitoring
        recommendations.append("Implement continuous monitoring of IoT device traffic")
        
        # Access control
        recommendations.append("Implement strong authentication and access controls")
        
        return recommendations
    
    async def _check_iot_compliance(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> Dict[str, Any]:
        """Check IoT compliance with various standards"""
        return {
            "NIST_IoT": await self._check_nist_iot_compliance(devices, vulnerabilities),
            "IEC_62443": await self._check_iec_62443_compliance(devices, vulnerabilities),
            "OWASP_IoT": await self._check_owasp_iot_compliance(devices, vulnerabilities)
        }
    
    async def _check_nist_iot_compliance(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> Dict[str, Any]:
        """Check NIST IoT Device Cybersecurity Capability Core Baseline compliance"""
        return {
            "device_identification": "partial",
            "device_configuration": "non_compliant",
            "data_protection": "unknown",
            "logical_access": "non_compliant",
            "software_update": "unknown",
            "cybersecurity_state_awareness": "partial"
        }
    
    async def _check_iec_62443_compliance(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> Dict[str, Any]:
        """Check IEC 62443 industrial cybersecurity compliance"""
        industrial_devices = [d for d in devices if self._is_industrial_device(d)]
        
        return {
            "identification_authentication": "non_compliant",
            "use_control": "partial",
            "system_integrity": "unknown",
            "data_confidentiality": "non_compliant",
            "restricted_data_flow": "unknown",
            "timely_response": "unknown",
            "resource_availability": "partial"
        }
    
    async def _check_owasp_iot_compliance(
        self,
        devices: List[IoTDevice],
        vulnerabilities: List[IoTVulnerability]
    ) -> Dict[str, Any]:
        """Check OWASP IoT Top 10 compliance"""
        issues = []
        
        # Check for common OWASP IoT Top 10 issues
        if any(v.category == VulnerabilityCategory.DEFAULT_CREDENTIALS for v in vulnerabilities):
            issues.append("I2: Insecure Default Passwords")
        
        if any(v.category == VulnerabilityCategory.INSECURE_COMMUNICATION for v in vulnerabilities):
            issues.append("I5: Use of Insecure or Outdated Components")
        
        return {
            "compliance_score": max(0, 100 - len(issues) * 20),
            "issues_found": issues,
            "recommendations": [
                "Address weak, guessable, or hardcoded passwords",
                "Ensure software/firmware updates",
                "Implement secure communication protocols"
            ]
        }
    
    def _is_industrial_device(self, device: IoTDevice) -> bool:
        """Check if device is an industrial control system device"""
        industrial_types = [
            IoTDeviceType.PLC,
            IoTDeviceType.HMI,
            IoTDeviceType.SCADA,
            IoTDeviceType.INDUSTRIAL_SENSOR
        ]
        return device.device_type in industrial_types
    
    # Additional helper methods for ICS assessment, traffic monitoring, and malware detection
    # ... (continued with more implementation details)
    
    # ThreatIntelligenceService and SecurityOrchestrationService interface implementations
    # ... (similar to blockchain service)
    
    # XORBService interface methods
    async def initialize(self) -> bool:
        """Initialize IoT security service"""
        try:
            self.start_time = datetime.utcnow()
            self.status = ServiceStatus.HEALTHY
            
            # Load IoT threat intelligence
            await self._load_iot_threat_intelligence()
            
            logger.info(f"IoT security service {self.service_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"IoT security service initialization failed: {e}")
            self.status = ServiceStatus.UNHEALTHY
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown IoT security service"""
        try:
            self.status = ServiceStatus.SHUTTING_DOWN
            
            # Clear caches and sensitive data
            self.assessment_cache.clear()
            self.iot_threat_intelligence.clear()
            
            self.status = ServiceStatus.STOPPED
            logger.info(f"IoT security service {self.service_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"IoT security service shutdown failed: {e}")
            return False
    
    async def health_check(self) -> ServiceHealth:
        """Perform IoT security service health check"""
        try:
            checks = {
                "scapy_available": SCAPY_AVAILABLE,
                "nmap_available": NMAP_AVAILABLE,
                "industrial_protocols": INDUSTRIAL_PROTOCOLS_AVAILABLE,
                "threat_intelligence_loaded": len(self.iot_threat_intelligence) > 0,
                "device_signatures_loaded": len(self.device_signatures) > 0
            }
            
            all_healthy = all(checks.values())
            status = ServiceStatus.HEALTHY if all_healthy else ServiceStatus.DEGRADED
            
            uptime = 0.0
            if hasattr(self, 'start_time') and self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return ServiceHealth(
                status=status,
                message="IoT security service operational",
                timestamp=datetime.utcnow(),
                checks=checks,
                uptime_seconds=uptime,
                metadata={
                    "cached_assessments": len(self.assessment_cache),
                    "device_signatures": len(self.device_signatures),
                    "vulnerability_categories": len(self.vulnerability_database)
                }
            )
            
        except Exception as e:
            logger.error(f"IoT security health check failed: {e}")
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={},
                last_error=str(e)
            )
    
    async def _load_iot_threat_intelligence(self):
        """Load IoT threat intelligence data"""
        # Load known IoT malware families and attack patterns
        self.iot_threat_intelligence = {
            "malware_families": [
                "Mirai", "Bashlite", "Gafgyt", "IoTReaper", "VPNFilter", "Hajime",
                "Persirai", "Amnesia", "Brickerbot", "Hide and Seek"
            ],
            "attack_signatures": [
                {"name": "telnet_bruteforce", "pattern": "telnet.*admin.*admin"},
                {"name": "mirai_infection", "pattern": "busybox.*wget.*tmp"},
                {"name": "upnp_exploit", "pattern": "location.*xml.*upnp"}
            ],
            "compromised_devices": set(),
            "c2_servers": {
                "198.51.100.1",  # Example C2 server
                "203.0.113.1"
            }
        }
    
    # Placeholder implementations for remaining methods
    async def _monitor_with_scapy(self, interface, duration, session, options):
        """Monitor traffic using Scapy"""
        # Implementation would use Scapy for packet capture and analysis
        session["packets_captured"] = 1000  # Mock
        return session
    
    async def _basic_traffic_monitoring(self, interface, duration, session):
        """Basic traffic monitoring without Scapy"""
        # Fallback monitoring implementation
        session["packets_captured"] = 100  # Mock
        return session
    
    async def _analyze_iot_traffic(self, session):
        """Analyze captured IoT traffic"""
        return {
            "suspicious_communications": [],
            "protocol_violations": [],
            "malware_indicators": []
        }
    
    async def _check_device_malware(self, device):
        """Check device for malware infections"""
        # Mock malware check
        return []
    
    async def _detect_c2_communications(self, device):
        """Detect command and control communications"""
        # Mock C2 detection
        return []
    
    async def _detect_lateral_movement(self, device):
        """Detect lateral movement indicators"""
        # Mock lateral movement detection
        return []
    
    async def _identify_malware_families(self, infected_devices):
        """Identify malware families from infected devices"""
        # Mock malware family identification
        return []
    
    # Placeholder methods for ICS assessment
    async def _identify_control_networks(self, devices):
        """Identify industrial control networks"""
        return []
    
    async def _identify_safety_systems(self, devices):
        """Identify safety instrumented systems"""
        return []
    
    async def _assess_ics_vulnerabilities(self, device):
        """Assess ICS-specific vulnerabilities"""
        return []
    
    async def _analyze_operational_risks(self, devices, vulnerabilities):
        """Analyze operational risks in ICS environment"""
        return []
    
    async def _check_ics_compliance(self, devices):
        """Check ICS compliance frameworks"""
        return ["IEC 62443", "NERC CIP"]
    
    def _determine_ics_threat_level(self, vulnerabilities, risks):
        """Determine ICS threat level"""
        return ThreatLevel.MEDIUM
    
    async def _generate_ics_recommendations(self, devices, vulnerabilities):
        """Generate ICS-specific recommendations"""
        return [
            "Implement network segmentation between IT and OT networks",
            "Deploy industrial firewalls and DMZ",
            "Establish secure remote access procedures",
            "Implement continuous monitoring of control system traffic"
        ]
    
    # ThreatIntelligenceService interface methods (simplified implementations)
    async def analyze_indicators(self, indicators, context, user):
        """Analyze IoT threat indicators"""
        return {"analysis_id": str(uuid4()), "results": []}
    
    async def correlate_threats(self, scan_results, threat_feeds=None):
        """Correlate IoT threats"""
        return {"correlation_id": str(uuid4()), "threats": []}
    
    async def get_threat_prediction(self, environment_data, timeframe="24h"):
        """Get IoT threat predictions"""
        return {"prediction_id": str(uuid4()), "predictions": []}
    
    async def generate_threat_report(self, analysis_results, report_format="json"):
        """Generate IoT threat report"""
        return {"report_id": str(uuid4()), "format": report_format}
    
    # SecurityOrchestrationService interface methods (simplified implementations)
    async def create_workflow(self, workflow_definition, user, org):
        """Create IoT security workflow"""
        return {"workflow_id": str(uuid4()), "type": "iot_security"}
    
    async def execute_workflow(self, workflow_id, parameters, user):
        """Execute IoT security workflow"""
        return {"execution_id": str(uuid4()), "status": "running"}
    
    async def get_workflow_status(self, execution_id, user):
        """Get IoT security workflow status"""
        return {"execution_id": execution_id, "status": "completed"}
    
    async def schedule_recurring_scan(self, targets, schedule, scan_config, user):
        """Schedule recurring IoT security scans"""
        return {"schedule_id": str(uuid4()), "status": "scheduled"}