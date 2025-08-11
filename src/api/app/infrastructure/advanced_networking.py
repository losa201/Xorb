"""
Advanced Networking Infrastructure for XORB Platform
Principal Auditor Implementation: Enterprise-grade networking with production capabilities
"""

import asyncio
import socket
import ssl
import json
import logging
import ipaddress
import subprocess
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

logger = logging.getLogger(__name__)


class NetworkProtocol(Enum):
    """Network protocol types"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    HTTP = "http"
    HTTPS = "https"
    DNS = "dns"
    SSH = "ssh"
    FTP = "ftp"
    SMTP = "smtp"
    SNMP = "snmp"


class NetworkSecurityLevel(Enum):
    """Network security levels for classification"""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class NetworkZone(Enum):
    """Network zones for segmentation"""
    DMZ = "dmz"
    INTERNAL = "internal"
    MANAGEMENT = "management"
    GUEST = "guest"
    QUARANTINE = "quarantine"
    CRITICAL = "critical"


@dataclass
class NetworkEndpoint:
    """Represents a network endpoint with security context"""
    ip_address: str
    port: int
    protocol: NetworkProtocol
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    security_level: NetworkSecurityLevel = NetworkSecurityLevel.PUBLIC
    zone: NetworkZone = NetworkZone.DMZ
    last_seen: Optional[datetime] = None
    health_status: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()


@dataclass
class NetworkInterface:
    """Represents a network interface with monitoring capabilities"""
    interface_name: str
    ip_address: str
    netmask: str
    gateway: Optional[str] = None
    dns_servers: List[str] = None
    mac_address: Optional[str] = None
    mtu: int = 1500
    speed: Optional[int] = None  # Mbps
    duplex: str = "full"
    link_status: str = "up"
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    tx_packets: int = 0
    rx_errors: int = 0
    tx_errors: int = 0
    
    def __post_init__(self):
        if self.dns_servers is None:
            self.dns_servers = []


class NetworkTopologyMapper:
    """Advanced network topology discovery and mapping"""
    
    def __init__(self, max_concurrent_scans: int = 50):
        self.max_concurrent_scans = max_concurrent_scans
        self.discovered_hosts: Dict[str, Dict[str, Any]] = {}
        self.network_graph: Dict[str, List[str]] = {}
        self.subnet_mappings: Dict[str, List[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_scans)
        
    async def discover_network_topology(
        self, 
        target_ranges: List[str],
        discovery_methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive network topology discovery
        
        Args:
            target_ranges: List of IP ranges/subnets to scan
            discovery_methods: Methods to use ['ping', 'arp', 'port_scan', 'dns']
        """
        if discovery_methods is None:
            discovery_methods = ['ping', 'arp', 'port_scan']
        
        logger.info(f"Starting network topology discovery for {len(target_ranges)} ranges")
        
        topology = {
            "discovery_start": datetime.utcnow().isoformat(),
            "target_ranges": target_ranges,
            "methods_used": discovery_methods,
            "discovered_hosts": {},
            "network_segments": {},
            "routing_information": {},
            "security_analysis": {}
        }
        
        # Discover hosts using multiple methods
        if 'ping' in discovery_methods:
            await self._ping_sweep(target_ranges, topology)
        
        if 'arp' in discovery_methods:
            await self._arp_discovery(target_ranges, topology)
        
        if 'port_scan' in discovery_methods:
            await self._port_scan_discovery(topology)
        
        if 'dns' in discovery_methods:
            await self._dns_enumeration(topology)
        
        # Analyze network segments and routing
        await self._analyze_network_segments(topology)
        await self._analyze_routing_topology(topology)
        await self._security_topology_analysis(topology)
        
        topology["discovery_end"] = datetime.utcnow().isoformat()
        topology["total_hosts_discovered"] = len(topology["discovered_hosts"])
        
        return topology
    
    async def _ping_sweep(self, target_ranges: List[str], topology: Dict[str, Any]):
        """Perform ICMP ping sweep for host discovery"""
        logger.info("Performing ping sweep discovery")
        
        tasks = []
        for range_str in target_ranges:
            try:
                network = ipaddress.ip_network(range_str, strict=False)
                for ip in network.hosts():
                    if len(tasks) < self.max_concurrent_scans:
                        tasks.append(self._ping_host(str(ip)))
                    else:
                        # Process batch
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        self._process_ping_results(results, topology)
                        tasks = []
            except Exception as e:
                logger.error(f"Error processing range {range_str}: {e}")
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._process_ping_results(results, topology)
    
    async def _ping_host(self, ip_address: str) -> Dict[str, Any]:
        """Ping a single host"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1', ip_address,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                # Extract RTT from ping output
                output = stdout.decode()
                rtt = self._extract_ping_rtt(output)
                return {
                    "ip": ip_address,
                    "status": "up",
                    "rtt": rtt,
                    "method": "ping"
                }
            else:
                return {
                    "ip": ip_address,
                    "status": "down",
                    "method": "ping"
                }
        except Exception as e:
            return {
                "ip": ip_address,
                "status": "error",
                "error": str(e),
                "method": "ping"
            }
    
    def _extract_ping_rtt(self, ping_output: str) -> Optional[float]:
        """Extract RTT from ping output"""
        import re
        match = re.search(r'time=(\d+\.?\d*)', ping_output)
        if match:
            return float(match.group(1))
        return None
    
    def _process_ping_results(self, results: List, topology: Dict[str, Any]):
        """Process ping sweep results"""
        for result in results:
            if isinstance(result, dict) and result.get("status") == "up":
                ip = result["ip"]
                topology["discovered_hosts"][ip] = {
                    "ip_address": ip,
                    "status": "up",
                    "discovery_methods": ["ping"],
                    "rtt": result.get("rtt"),
                    "first_seen": datetime.utcnow().isoformat(),
                    "services": [],
                    "os_fingerprint": {},
                    "security_assessment": {}
                }
    
    async def _arp_discovery(self, target_ranges: List[str], topology: Dict[str, Any]):
        """ARP table analysis for local network discovery"""
        logger.info("Performing ARP table analysis")
        
        try:
            # Get ARP table
            proc = await asyncio.create_subprocess_exec(
                'arp', '-a',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                arp_entries = self._parse_arp_table(stdout.decode())
                
                for entry in arp_entries:
                    ip = entry["ip"]
                    if ip in topology["discovered_hosts"]:
                        topology["discovered_hosts"][ip]["mac_address"] = entry["mac"]
                        if "arp" not in topology["discovered_hosts"][ip]["discovery_methods"]:
                            topology["discovered_hosts"][ip]["discovery_methods"].append("arp")
                    else:
                        topology["discovered_hosts"][ip] = {
                            "ip_address": ip,
                            "status": "up",
                            "discovery_methods": ["arp"],
                            "mac_address": entry["mac"],
                            "first_seen": datetime.utcnow().isoformat(),
                            "services": [],
                            "os_fingerprint": {},
                            "security_assessment": {}
                        }
        except Exception as e:
            logger.error(f"ARP discovery failed: {e}")
    
    def _parse_arp_table(self, arp_output: str) -> List[Dict[str, str]]:
        """Parse ARP table output"""
        import re
        entries = []
        
        for line in arp_output.split('\n'):
            # Match lines like: hostname (192.168.1.1) at aa:bb:cc:dd:ee:ff [ether] on eth0
            match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\) at ([a-fA-F0-9:]{17})', line)
            if match:
                entries.append({
                    "ip": match.group(1),
                    "mac": match.group(2)
                })
        
        return entries
    
    async def _port_scan_discovery(self, topology: Dict[str, Any]):
        """Port scanning for service discovery"""
        logger.info("Performing port scan discovery")
        
        common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1723, 3389, 5900, 8080]
        
        tasks = []
        for ip in topology["discovered_hosts"]:
            tasks.append(self._scan_host_ports(ip, common_ports))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    ip = list(topology["discovered_hosts"].keys())[i]
                    topology["discovered_hosts"][ip]["services"] = result.get("open_ports", [])
                    if "port_scan" not in topology["discovered_hosts"][ip]["discovery_methods"]:
                        topology["discovered_hosts"][ip]["discovery_methods"].append("port_scan")
    
    async def _scan_host_ports(self, ip: str, ports: List[int]) -> Dict[str, Any]:
        """Scan specific ports on a host"""
        open_ports = []
        
        for port in ports:
            try:
                future = asyncio.open_connection(ip, port)
                reader, writer = await asyncio.wait_for(future, timeout=1.0)
                
                # Try to get service banner
                service_info = await self._get_service_banner(reader, writer, port)
                
                open_ports.append({
                    "port": port,
                    "protocol": "tcp",
                    "state": "open",
                    "service": service_info.get("service", "unknown"),
                    "banner": service_info.get("banner", ""),
                    "detected_at": datetime.utcnow().isoformat()
                })
                
                writer.close()
                await writer.wait_closed()
                
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                continue
            except Exception as e:
                logger.debug(f"Error scanning {ip}:{port}: {e}")
        
        return {"open_ports": open_ports}
    
    async def _get_service_banner(self, reader, writer, port: int) -> Dict[str, str]:
        """Attempt to get service banner"""
        service_info = {"service": "unknown", "banner": ""}
        
        try:
            # Send appropriate probe based on port
            if port == 80:
                writer.write(b"GET / HTTP/1.0\r\n\r\n")
                service_info["service"] = "http"
            elif port == 443:
                service_info["service"] = "https"
            elif port == 22:
                service_info["service"] = "ssh"
            elif port == 21:
                service_info["service"] = "ftp"
            elif port == 25:
                service_info["service"] = "smtp"
            
            await writer.drain()
            
            # Try to read banner (with timeout)
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
                service_info["banner"] = data.decode('utf-8', errors='ignore')[:200]
            except asyncio.TimeoutError:
                pass
                
        except Exception as e:
            logger.debug(f"Banner grab failed for port {port}: {e}")
        
        return service_info
    
    async def _dns_enumeration(self, topology: Dict[str, Any]):
        """DNS enumeration for hostname resolution"""
        logger.info("Performing DNS enumeration")
        
        for ip in topology["discovered_hosts"]:
            try:
                # Reverse DNS lookup
                hostname = await self._reverse_dns_lookup(ip)
                if hostname:
                    topology["discovered_hosts"][ip]["hostname"] = hostname
                    if "dns" not in topology["discovered_hosts"][ip]["discovery_methods"]:
                        topology["discovered_hosts"][ip]["discovery_methods"].append("dns")
            except Exception as e:
                logger.debug(f"DNS lookup failed for {ip}: {e}")
    
    async def _reverse_dns_lookup(self, ip: str) -> Optional[str]:
        """Perform reverse DNS lookup"""
        try:
            loop = asyncio.get_event_loop()
            hostname, _, _ = await loop.run_in_executor(
                None, socket.gethostbyaddr, ip
            )
            return hostname
        except Exception:
            return None
    
    async def _analyze_network_segments(self, topology: Dict[str, Any]):
        """Analyze network segmentation"""
        segments = {}
        
        for ip, host_info in topology["discovered_hosts"].items():
            try:
                ip_obj = ipaddress.ip_address(ip)
                
                # Determine likely subnet (assuming /24 for simplicity)
                network = ipaddress.ip_network(f"{ip}/24", strict=False)
                network_str = str(network.network_address) + "/24"
                
                if network_str not in segments:
                    segments[network_str] = {
                        "network": network_str,
                        "hosts": [],
                        "host_count": 0,
                        "services": set(),
                        "security_level": "unknown"
                    }
                
                segments[network_str]["hosts"].append(ip)
                segments[network_str]["host_count"] += 1
                
                # Aggregate services
                for service in host_info.get("services", []):
                    segments[network_str]["services"].add(service.get("service", "unknown"))
                
            except Exception as e:
                logger.debug(f"Error analyzing segment for {ip}: {e}")
        
        # Convert sets to lists for JSON serialization
        for segment in segments.values():
            segment["services"] = list(segment["services"])
        
        topology["network_segments"] = segments
    
    async def _analyze_routing_topology(self, topology: Dict[str, Any]):
        """Analyze routing and network topology"""
        routing_info = {
            "default_gateway": None,
            "routing_table": [],
            "network_hops": {},
            "network_latency": {}
        }
        
        try:
            # Get default gateway
            routing_info["default_gateway"] = await self._get_default_gateway()
            
            # Get routing table
            routing_info["routing_table"] = await self._get_routing_table()
            
            # Perform traceroute to sample destinations
            sample_ips = list(topology["discovered_hosts"].keys())[:5]
            for ip in sample_ips:
                hops = await self._traceroute(ip)
                if hops:
                    routing_info["network_hops"][ip] = hops
            
        except Exception as e:
            logger.error(f"Routing analysis failed: {e}")
        
        topology["routing_information"] = routing_info
    
    async def _get_default_gateway(self) -> Optional[str]:
        """Get default gateway"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ip', 'route', 'show', 'default',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                output = stdout.decode()
                import re
                match = re.search(r'via (\d+\.\d+\.\d+\.\d+)', output)
                if match:
                    return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to get default gateway: {e}")
        
        return None
    
    async def _get_routing_table(self) -> List[Dict[str, str]]:
        """Get routing table"""
        routes = []
        try:
            proc = await asyncio.create_subprocess_exec(
                'ip', 'route', 'show',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                for line in stdout.decode().split('\n'):
                    if line.strip():
                        routes.append({"route": line.strip()})
        except Exception as e:
            logger.debug(f"Failed to get routing table: {e}")
        
        return routes
    
    async def _traceroute(self, destination: str) -> List[Dict[str, Any]]:
        """Perform traceroute to destination"""
        hops = []
        try:
            proc = await asyncio.create_subprocess_exec(
                'traceroute', '-m', '10', destination,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                lines = stdout.decode().split('\n')[1:]  # Skip header
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('traceroute'):
                        hop_info = self._parse_traceroute_hop(line, i + 1)
                        if hop_info:
                            hops.append(hop_info)
        except Exception as e:
            logger.debug(f"Traceroute to {destination} failed: {e}")
        
        return hops
    
    def _parse_traceroute_hop(self, line: str, hop_number: int) -> Optional[Dict[str, Any]]:
        """Parse traceroute hop information"""
        import re
        
        # Match lines like: "  1  192.168.1.1 (192.168.1.1)  1.234 ms  1.456 ms  1.789 ms"
        match = re.search(r'(\d+\.\d+\.\d+\.\d+).*?(\d+\.\d+) ms', line)
        if match:
            return {
                "hop": hop_number,
                "ip": match.group(1),
                "rtt": float(match.group(2))
            }
        return None
    
    async def _security_topology_analysis(self, topology: Dict[str, Any]):
        """Perform security analysis of network topology"""
        security_analysis = {
            "exposed_services": [],
            "security_concerns": [],
            "network_zones": {},
            "firewall_detection": {},
            "vulnerability_indicators": []
        }
        
        # Analyze exposed services
        for ip, host_info in topology["discovered_hosts"].items():
            for service in host_info.get("services", []):
                if service.get("port") in [21, 23, 135, 139, 445]:  # Risky services
                    security_analysis["exposed_services"].append({
                        "ip": ip,
                        "port": service.get("port"),
                        "service": service.get("service"),
                        "risk_level": "high",
                        "reason": "Potentially insecure service exposed"
                    })
        
        # Detect potential network zones
        for segment_name, segment_info in topology.get("network_segments", {}).items():
            zone_classification = self._classify_network_zone(segment_info)
            security_analysis["network_zones"][segment_name] = zone_classification
        
        topology["security_analysis"] = security_analysis
    
    def _classify_network_zone(self, segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify network segment into security zones"""
        services = segment_info.get("services", [])
        host_count = segment_info.get("host_count", 0)
        
        classification = {
            "zone_type": "internal",
            "security_level": "medium",
            "rationale": []
        }
        
        # DMZ detection
        if "http" in services or "https" in services:
            classification["zone_type"] = "dmz"
            classification["rationale"].append("Web services detected")
        
        # Management network detection
        if "ssh" in services and host_count < 10:
            classification["zone_type"] = "management"
            classification["rationale"].append("SSH services with low host count")
        
        # Critical systems detection
        if "unknown" not in services and len(services) > 5:
            classification["security_level"] = "high"
            classification["rationale"].append("Multiple services indicating critical systems")
        
        return classification


class AdvancedFirewallManager:
    """Enterprise-grade firewall management and policy enforcement"""
    
    def __init__(self):
        self.firewall_rules: List[Dict[str, Any]] = []
        self.rule_groups: Dict[str, List[str]] = {}
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.blocked_connections: List[Dict[str, Any]] = []
        self.traffic_stats: Dict[str, int] = {}
        
    async def create_firewall_rule(
        self,
        rule_name: str,
        source: str,
        destination: str,
        ports: List[int],
        protocol: str,
        action: str,
        priority: int = 100
    ) -> Dict[str, Any]:
        """Create a comprehensive firewall rule"""
        
        rule = {
            "rule_id": f"rule_{len(self.firewall_rules) + 1:04d}",
            "name": rule_name,
            "source": source,
            "destination": destination,
            "ports": ports,
            "protocol": protocol.upper(),
            "action": action.upper(),  # ALLOW, DENY, DROP, REJECT
            "priority": priority,
            "created_at": datetime.utcnow().isoformat(),
            "enabled": True,
            "hit_count": 0,
            "last_hit": None,
            "metadata": {
                "created_by": "system",
                "category": "user_defined"
            }
        }
        
        # Validate rule
        validation_result = await self._validate_firewall_rule(rule)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "rule": rule
            }
        
        self.firewall_rules.append(rule)
        
        logger.info(f"Created firewall rule: {rule_name} ({rule['rule_id']})")
        
        return {
            "success": True,
            "rule": rule,
            "rule_id": rule["rule_id"]
        }
    
    async def _validate_firewall_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate firewall rule configuration"""
        
        # Check required fields
        required_fields = ["source", "destination", "ports", "protocol", "action"]
        for field in required_fields:
            if field not in rule or rule[field] is None:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate IP addresses/ranges
        try:
            if rule["source"] != "*":
                ipaddress.ip_network(rule["source"], strict=False)
            if rule["destination"] != "*":
                ipaddress.ip_network(rule["destination"], strict=False)
        except ValueError as e:
            return {
                "valid": False,
                "error": f"Invalid IP address/range: {e}"
            }
        
        # Validate ports
        for port in rule["ports"]:
            if not isinstance(port, int) or port < 1 or port > 65535:
                return {
                    "valid": False,
                    "error": f"Invalid port number: {port}"
                }
        
        # Validate protocol
        valid_protocols = ["TCP", "UDP", "ICMP", "ANY"]
        if rule["protocol"] not in valid_protocols:
            return {
                "valid": False,
                "error": f"Invalid protocol: {rule['protocol']}"
            }
        
        # Validate action
        valid_actions = ["ALLOW", "DENY", "DROP", "REJECT"]
        if rule["action"] not in valid_actions:
            return {
                "valid": False,
                "error": f"Invalid action: {rule['action']}"
            }
        
        return {"valid": True}
    
    async def evaluate_connection(
        self,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        protocol: str
    ) -> Dict[str, Any]:
        """Evaluate connection against firewall rules"""
        
        connection_id = f"{source_ip}:{dest_ip}:{dest_port}:{protocol}"
        
        evaluation = {
            "connection_id": connection_id,
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "dest_port": dest_port,
            "protocol": protocol.upper(),
            "decision": "DENY",  # Default deny
            "matched_rule": None,
            "evaluated_at": datetime.utcnow().isoformat(),
            "evaluation_time_ms": 0
        }
        
        start_time = time.time()
        
        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(self.firewall_rules, key=lambda x: x["priority"])
        
        for rule in sorted_rules:
            if not rule["enabled"]:
                continue
            
            if await self._rule_matches_connection(rule, source_ip, dest_ip, dest_port, protocol):
                evaluation["decision"] = rule["action"]
                evaluation["matched_rule"] = rule["rule_id"]
                
                # Update rule statistics
                rule["hit_count"] += 1
                rule["last_hit"] = datetime.utcnow().isoformat()
                
                break
        
        evaluation["evaluation_time_ms"] = (time.time() - start_time) * 1000
        
        # Log connection
        if evaluation["decision"] in ["ALLOW"]:
            self.active_connections[connection_id] = evaluation
        else:
            self.blocked_connections.append(evaluation)
        
        # Update traffic statistics
        self._update_traffic_stats(evaluation)
        
        logger.debug(f"Firewall evaluation: {connection_id} -> {evaluation['decision']}")
        
        return evaluation
    
    async def _rule_matches_connection(
        self,
        rule: Dict[str, Any],
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        protocol: str
    ) -> bool:
        """Check if firewall rule matches connection"""
        
        # Check protocol
        if rule["protocol"] != "ANY" and rule["protocol"] != protocol.upper():
            return False
        
        # Check source IP
        if rule["source"] != "*":
            try:
                source_network = ipaddress.ip_network(rule["source"], strict=False)
                if ipaddress.ip_address(source_ip) not in source_network:
                    return False
            except (ValueError, ipaddress.AddressValueError):
                return False
        
        # Check destination IP
        if rule["destination"] != "*":
            try:
                dest_network = ipaddress.ip_network(rule["destination"], strict=False)
                if ipaddress.ip_address(dest_ip) not in dest_network:
                    return False
            except (ValueError, ipaddress.AddressValueError):
                return False
        
        # Check port
        if dest_port not in rule["ports"] and 0 not in rule["ports"]:  # 0 means any port
            return False
        
        return True
    
    def _update_traffic_stats(self, evaluation: Dict[str, Any]):
        """Update traffic statistics"""
        decision = evaluation["decision"]
        protocol = evaluation["protocol"]
        
        stats_key = f"{decision}_{protocol}"
        self.traffic_stats[stats_key] = self.traffic_stats.get(stats_key, 0) + 1
        self.traffic_stats["total_connections"] = self.traffic_stats.get("total_connections", 0) + 1
    
    async def create_rule_group(self, group_name: str, rule_ids: List[str]) -> Dict[str, Any]:
        """Create a group of firewall rules for easier management"""
        
        # Validate rule IDs exist
        existing_rule_ids = {rule["rule_id"] for rule in self.firewall_rules}
        invalid_rule_ids = [rid for rid in rule_ids if rid not in existing_rule_ids]
        
        if invalid_rule_ids:
            return {
                "success": False,
                "error": f"Invalid rule IDs: {invalid_rule_ids}"
            }
        
        self.rule_groups[group_name] = rule_ids
        
        return {
            "success": True,
            "group_name": group_name,
            "rule_count": len(rule_ids)
        }
    
    async def enable_rule_group(self, group_name: str) -> Dict[str, Any]:
        """Enable all rules in a group"""
        if group_name not in self.rule_groups:
            return {"success": False, "error": "Rule group not found"}
        
        enabled_count = 0
        for rule in self.firewall_rules:
            if rule["rule_id"] in self.rule_groups[group_name]:
                rule["enabled"] = True
                enabled_count += 1
        
        return {
            "success": True,
            "enabled_rules": enabled_count
        }
    
    async def disable_rule_group(self, group_name: str) -> Dict[str, Any]:
        """Disable all rules in a group"""
        if group_name not in self.rule_groups:
            return {"success": False, "error": "Rule group not found"}
        
        disabled_count = 0
        for rule in self.firewall_rules:
            if rule["rule_id"] in self.rule_groups[group_name]:
                rule["enabled"] = False
                disabled_count += 1
        
        return {
            "success": True,
            "disabled_rules": disabled_count
        }
    
    async def get_firewall_status(self) -> Dict[str, Any]:
        """Get comprehensive firewall status"""
        
        total_rules = len(self.firewall_rules)
        enabled_rules = len([r for r in self.firewall_rules if r["enabled"]])
        
        return {
            "firewall_status": "active",
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "rule_groups": len(self.rule_groups),
            "active_connections": len(self.active_connections),
            "blocked_connections_today": len([
                b for b in self.blocked_connections 
                if datetime.fromisoformat(b["evaluated_at"]).date() == datetime.utcnow().date()
            ]),
            "traffic_stats": self.traffic_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def export_firewall_config(self) -> Dict[str, Any]:
        """Export firewall configuration"""
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "firewall_rules": self.firewall_rules,
            "rule_groups": self.rule_groups,
            "configuration_version": "1.0",
            "total_rules": len(self.firewall_rules)
        }
    
    async def import_firewall_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Import firewall configuration"""
        
        try:
            # Validate configuration format
            if "firewall_rules" not in config:
                return {"success": False, "error": "Invalid configuration format"}
            
            # Backup current configuration
            backup = {
                "rules": self.firewall_rules.copy(),
                "groups": self.rule_groups.copy()
            }
            
            # Import rules
            imported_rules = 0
            for rule in config["firewall_rules"]:
                validation_result = await self._validate_firewall_rule(rule)
                if validation_result["valid"]:
                    self.firewall_rules.append(rule)
                    imported_rules += 1
            
            # Import rule groups
            if "rule_groups" in config:
                self.rule_groups.update(config["rule_groups"])
            
            return {
                "success": True,
                "imported_rules": imported_rules,
                "backup_created": True
            }
            
        except Exception as e:
            logger.error(f"Failed to import firewall config: {e}")
            return {"success": False, "error": str(e)}


class NetworkSecurityMonitor:
    """Real-time network security monitoring and threat detection"""
    
    def __init__(self, firewall_manager: AdvancedFirewallManager):
        self.firewall_manager = firewall_manager
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[Dict[str, Any]] = []
        self.threat_indicators: Set[str] = set()
        self.monitoring_active = False
        
    async def start_network_monitoring(self, interfaces: List[str] = None) -> Dict[str, Any]:
        """Start comprehensive network security monitoring"""
        
        if interfaces is None:
            interfaces = await self._get_network_interfaces()
        
        self.monitoring_active = True
        
        monitor_config = {
            "start_time": datetime.utcnow().isoformat(),
            "interfaces": interfaces,
            "monitors": [
                "connection_monitoring",
                "anomaly_detection",
                "threat_detection",
                "bandwidth_monitoring"
            ]
        }
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_connections())
        asyncio.create_task(self._monitor_anomalies())
        asyncio.create_task(self._monitor_threats())
        asyncio.create_task(self._monitor_bandwidth())
        
        logger.info("Network security monitoring started")
        
        return {
            "success": True,
            "config": monitor_config,
            "status": "monitoring_active"
        }
    
    async def _get_network_interfaces(self) -> List[str]:
        """Get available network interfaces"""
        interfaces = []
        try:
            proc = await asyncio.create_subprocess_exec(
                'ip', 'link', 'show',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                for line in stdout.decode().split('\n'):
                    if ': ' in line and 'state UP' in line:
                        interface = line.split(':')[1].strip().split('@')[0]
                        interfaces.append(interface)
        except Exception as e:
            logger.error(f"Failed to get network interfaces: {e}")
            interfaces = ["eth0", "wlan0"]  # Fallback
        
        return interfaces
    
    async def _monitor_connections(self):
        """Monitor active network connections"""
        while self.monitoring_active:
            try:
                connections = await self._get_active_connections()
                
                for conn in connections:
                    # Analyze connection for security concerns
                    security_assessment = await self._assess_connection_security(conn)
                    
                    if security_assessment["risk_level"] == "high":
                        await self._create_security_event("suspicious_connection", {
                            "connection": conn,
                            "assessment": security_assessment
                        })
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _get_active_connections(self) -> List[Dict[str, Any]]:
        """Get active network connections"""
        connections = []
        try:
            proc = await asyncio.create_subprocess_exec(
                'netstat', '-tuln',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                for line in stdout.decode().split('\n')[2:]:  # Skip headers
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            connections.append({
                                "protocol": parts[0],
                                "local_address": parts[3],
                                "state": parts[5] if len(parts) > 5 else "LISTEN",
                                "detected_at": datetime.utcnow().isoformat()
                            })
        except Exception as e:
            logger.debug(f"Failed to get active connections: {e}")
        
        return connections
    
    async def _assess_connection_security(self, connection: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security risk of a connection"""
        
        assessment = {
            "risk_level": "low",
            "risk_factors": [],
            "recommendations": []
        }
        
        local_addr = connection.get("local_address", "")
        
        # Check for risky ports
        risky_ports = ["21", "23", "135", "139", "445", "1433", "3306"]
        for port in risky_ports:
            if f":{port}" in local_addr:
                assessment["risk_level"] = "high"
                assessment["risk_factors"].append(f"Risky service on port {port}")
                assessment["recommendations"].append(f"Consider securing or disabling service on port {port}")
        
        # Check for external-facing services
        if ":0.0.0.0:" in local_addr or ":*:" in local_addr:
            if assessment["risk_level"] != "high":
                assessment["risk_level"] = "medium"
            assessment["risk_factors"].append("Service exposed to external network")
            assessment["recommendations"].append("Verify if external exposure is necessary")
        
        return assessment
    
    async def _monitor_anomalies(self):
        """Monitor for network anomalies"""
        baseline_metrics = {}
        
        while self.monitoring_active:
            try:
                current_metrics = await self._collect_network_metrics()
                
                if baseline_metrics:
                    anomalies = await self._detect_anomalies(baseline_metrics, current_metrics)
                    
                    for anomaly in anomalies:
                        await self._create_security_event("network_anomaly", anomaly)
                
                # Update baseline (simple moving average)
                for metric, value in current_metrics.items():
                    if metric in baseline_metrics:
                        baseline_metrics[metric] = (baseline_metrics[metric] + value) / 2
                    else:
                        baseline_metrics[metric] = value
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Anomaly monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_network_metrics(self) -> Dict[str, float]:
        """Collect current network metrics"""
        metrics = {
            "connections_count": 0,
            "listening_ports": 0,
            "established_connections": 0
        }
        
        try:
            connections = await self._get_active_connections()
            metrics["connections_count"] = len(connections)
            metrics["listening_ports"] = len([c for c in connections if c.get("state") == "LISTEN"])
            metrics["established_connections"] = len([c for c in connections if c.get("state") == "ESTABLISHED"])
        except Exception as e:
            logger.debug(f"Failed to collect network metrics: {e}")
        
        return metrics
    
    async def _detect_anomalies(
        self, 
        baseline: Dict[str, float], 
        current: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in network metrics"""
        anomalies = []
        
        for metric, current_value in current.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Simple threshold-based anomaly detection
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    
                    if deviation > 0.5:  # 50% deviation threshold
                        anomalies.append({
                            "metric": metric,
                            "baseline_value": baseline_value,
                            "current_value": current_value,
                            "deviation_percent": deviation * 100,
                            "severity": "high" if deviation > 1.0 else "medium"
                        })
        
        return anomalies
    
    async def _monitor_threats(self):
        """Monitor for known threat indicators"""
        while self.monitoring_active:
            try:
                # Check for connections to known malicious IPs
                connections = await self._get_active_connections()
                
                for conn in connections:
                    # Extract IP from address
                    addr_parts = conn.get("local_address", "").split(":")
                    if len(addr_parts) >= 2:
                        ip = addr_parts[0]
                        
                        if ip in self.threat_indicators:
                            await self._create_security_event("threat_detected", {
                                "threat_type": "malicious_ip",
                                "ip_address": ip,
                                "connection": conn
                            })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Threat monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_bandwidth(self):
        """Monitor bandwidth usage"""
        previous_stats = {}
        
        while self.monitoring_active:
            try:
                current_stats = await self._get_interface_statistics()
                
                if previous_stats:
                    bandwidth_analysis = await self._analyze_bandwidth_usage(
                        previous_stats, current_stats
                    )
                    
                    if bandwidth_analysis["anomaly_detected"]:
                        await self._create_security_event("bandwidth_anomaly", bandwidth_analysis)
                
                previous_stats = current_stats
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Bandwidth monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _get_interface_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get network interface statistics"""
        stats = {}
        
        try:
            with open('/proc/net/dev', 'r') as f:
                lines = f.readlines()[2:]  # Skip headers
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 10:
                        interface = parts[0].rstrip(':')
                        stats[interface] = {
                            "rx_bytes": int(parts[1]),
                            "tx_bytes": int(parts[9]),
                            "rx_packets": int(parts[2]),
                            "tx_packets": int(parts[10])
                        }
        except Exception as e:
            logger.debug(f"Failed to get interface statistics: {e}")
        
        return stats
    
    async def _analyze_bandwidth_usage(
        self, 
        previous: Dict[str, Dict[str, int]], 
        current: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """Analyze bandwidth usage for anomalies"""
        
        analysis = {
            "anomaly_detected": False,
            "interface_analysis": {},
            "high_bandwidth_interfaces": [],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        for interface, current_stats in current.items():
            if interface in previous:
                prev_stats = previous[interface]
                
                # Calculate deltas
                rx_delta = current_stats["rx_bytes"] - prev_stats["rx_bytes"]
                tx_delta = current_stats["tx_bytes"] - prev_stats["tx_bytes"]
                
                # Convert to Mbps (assuming 30-second interval)
                rx_mbps = (rx_delta * 8) / (30 * 1024 * 1024)
                tx_mbps = (tx_delta * 8) / (30 * 1024 * 1024)
                
                interface_analysis = {
                    "rx_mbps": rx_mbps,
                    "tx_mbps": tx_mbps,
                    "total_mbps": rx_mbps + tx_mbps
                }
                
                # Detect high bandwidth usage (>100 Mbps threshold)
                if interface_analysis["total_mbps"] > 100:
                    analysis["anomaly_detected"] = True
                    analysis["high_bandwidth_interfaces"].append({
                        "interface": interface,
                        "bandwidth_mbps": interface_analysis["total_mbps"]
                    })
                
                analysis["interface_analysis"][interface] = interface_analysis
        
        return analysis
    
    async def _create_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Create a security event"""
        
        event = {
            "event_id": f"evt_{len(self.security_events) + 1:06d}",
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": self._calculate_event_severity(event_type, event_data),
            "data": event_data,
            "source": "network_security_monitor",
            "status": "new"
        }
        
        self.security_events.append(event)
        
        logger.warning(f"Security event created: {event_type} ({event['event_id']})")
        
        # Trigger automated response if high severity
        if event["severity"] == "critical":
            await self._trigger_automated_response(event)
    
    def _calculate_event_severity(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Calculate severity level for security event"""
        
        severity_map = {
            "threat_detected": "critical",
            "suspicious_connection": "high",
            "network_anomaly": "medium",
            "bandwidth_anomaly": "low"
        }
        
        base_severity = severity_map.get(event_type, "low")
        
        # Adjust based on event data
        if event_type == "network_anomaly":
            anomaly_severity = event_data.get("severity", "low")
            if anomaly_severity == "high":
                base_severity = "high"
        
        return base_severity
    
    async def _trigger_automated_response(self, event: Dict[str, Any]):
        """Trigger automated security response"""
        
        logger.critical(f"Triggering automated response for event: {event['event_id']}")
        
        event_type = event["event_type"]
        
        if event_type == "threat_detected":
            # Automatically block malicious IP
            threat_data = event["data"]
            if "ip_address" in threat_data:
                await self._auto_block_ip(threat_data["ip_address"])
        
        elif event_type == "suspicious_connection":
            # Increase monitoring for the connection
            conn_data = event["data"]["connection"]
            await self._increase_connection_monitoring(conn_data)
    
    async def _auto_block_ip(self, ip_address: str):
        """Automatically block malicious IP address"""
        
        try:
            # Create firewall rule to block IP
            rule_result = await self.firewall_manager.create_firewall_rule(
                rule_name=f"auto_block_{ip_address}",
                source=ip_address,
                destination="*",
                ports=[0],  # All ports
                protocol="ANY",
                action="DROP",
                priority=1  # High priority
            )
            
            if rule_result["success"]:
                logger.info(f"Automatically blocked malicious IP: {ip_address}")
            else:
                logger.error(f"Failed to auto-block IP {ip_address}: {rule_result['error']}")
                
        except Exception as e:
            logger.error(f"Auto-block failed for IP {ip_address}: {e}")
    
    async def _increase_connection_monitoring(self, connection: Dict[str, Any]):
        """Increase monitoring for suspicious connection"""
        
        conn_id = f"{connection.get('protocol', 'unknown')}_{connection.get('local_address', 'unknown')}"
        
        self.active_monitors[conn_id] = {
            "connection": connection,
            "monitoring_level": "high",
            "start_time": datetime.utcnow().isoformat(),
            "alerts_triggered": 0
        }
        
        logger.info(f"Increased monitoring for connection: {conn_id}")
    
    async def get_security_events(
        self, 
        limit: int = 100, 
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent security events"""
        
        events = self.security_events
        
        # Filter by severity if specified
        if severity:
            events = [e for e in events if e["severity"] == severity]
        
        # Sort by timestamp (most recent first) and limit
        events = sorted(events, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        return events
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop network security monitoring"""
        
        self.monitoring_active = False
        
        return {
            "success": True,
            "stop_time": datetime.utcnow().isoformat(),
            "events_collected": len(self.security_events),
            "status": "monitoring_stopped"
        }


class EnterpriseNetworkingService:
    """Comprehensive enterprise networking service"""
    
    def __init__(self):
        self.topology_mapper = NetworkTopologyMapper()
        self.firewall_manager = AdvancedFirewallManager()
        self.security_monitor = NetworkSecurityMonitor(self.firewall_manager)
        self.network_interfaces: Dict[str, NetworkInterface] = {}
        self.network_endpoints: Dict[str, NetworkEndpoint] = {}
        
    async def initialize_networking(self) -> Dict[str, Any]:
        """Initialize enterprise networking components"""
        
        logger.info("Initializing enterprise networking service")
        
        # Initialize network interfaces
        await self._discover_network_interfaces()
        
        # Set up default firewall rules
        await self._setup_default_firewall_rules()
        
        # Start security monitoring
        monitoring_result = await self.security_monitor.start_network_monitoring()
        
        initialization_result = {
            "success": True,
            "initialization_time": datetime.utcnow().isoformat(),
            "components_initialized": [
                "network_topology_mapper",
                "firewall_manager", 
                "security_monitor"
            ],
            "network_interfaces": len(self.network_interfaces),
            "default_firewall_rules": len(self.firewall_manager.firewall_rules),
            "monitoring_active": monitoring_result["success"]
        }
        
        logger.info("Enterprise networking service initialized successfully")
        
        return initialization_result
    
    async def _discover_network_interfaces(self):
        """Discover and catalog network interfaces"""
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'ip', 'addr', 'show',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                interfaces = self._parse_ip_addr_output(stdout.decode())
                for interface in interfaces:
                    self.network_interfaces[interface.interface_name] = interface
                    
        except Exception as e:
            logger.error(f"Failed to discover network interfaces: {e}")
    
    def _parse_ip_addr_output(self, output: str) -> List[NetworkInterface]:
        """Parse 'ip addr show' output"""
        interfaces = []
        current_interface = None
        
        for line in output.split('\n'):
            line = line.strip()
            
            # Interface line: "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500"
            if ': ' in line and '<' in line:
                parts = line.split(': ')
                if len(parts) >= 2:
                    interface_name = parts[1].split(':')[0]
                    
                    # Extract MTU
                    mtu = 1500
                    if 'mtu ' in line:
                        mtu_part = line.split('mtu ')[1].split()[0]
                        try:
                            mtu = int(mtu_part)
                        except ValueError:
                            pass
                    
                    current_interface = NetworkInterface(
                        interface_name=interface_name,
                        ip_address="",
                        netmask="",
                        mtu=mtu
                    )
            
            # IP address line: "inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0"
            elif line.startswith('inet ') and current_interface:
                ip_info = line.split()[1]
                if '/' in ip_info:
                    ip, prefix = ip_info.split('/')
                    current_interface.ip_address = ip
                    
                    # Convert CIDR prefix to netmask
                    try:
                        network = ipaddress.IPv4Network(f"0.0.0.0/{prefix}", strict=False)
                        current_interface.netmask = str(network.netmask)
                    except ValueError:
                        current_interface.netmask = "255.255.255.0"
                
                # Interface is complete, add to list
                if current_interface.ip_address:
                    interfaces.append(current_interface)
                    current_interface = None
        
        return interfaces
    
    async def _setup_default_firewall_rules(self):
        """Set up default enterprise firewall rules"""
        
        default_rules = [
            {
                "name": "Allow SSH",
                "source": "*",
                "destination": "*",
                "ports": [22],
                "protocol": "TCP",
                "action": "ALLOW",
                "priority": 100
            },
            {
                "name": "Allow HTTP",
                "source": "*", 
                "destination": "*",
                "ports": [80],
                "protocol": "TCP",
                "action": "ALLOW",
                "priority": 100
            },
            {
                "name": "Allow HTTPS",
                "source": "*",
                "destination": "*", 
                "ports": [443],
                "protocol": "TCP",
                "action": "ALLOW",
                "priority": 100
            },
            {
                "name": "Block Telnet",
                "source": "*",
                "destination": "*",
                "ports": [23],
                "protocol": "TCP", 
                "action": "DROP",
                "priority": 10
            },
            {
                "name": "Block FTP",
                "source": "*",
                "destination": "*",
                "ports": [21],
                "protocol": "TCP",
                "action": "DROP",
                "priority": 10
            }
        ]
        
        for rule_config in default_rules:
            await self.firewall_manager.create_firewall_rule(**rule_config)
    
    async def perform_network_assessment(
        self, 
        target_ranges: List[str],
        assessment_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Perform comprehensive network security assessment"""
        
        logger.info(f"Starting {assessment_type} network assessment")
        
        assessment = {
            "assessment_id": f"assess_{int(time.time())}",
            "assessment_type": assessment_type,
            "start_time": datetime.utcnow().isoformat(),
            "target_ranges": target_ranges,
            "topology_discovery": {},
            "security_analysis": {},
            "firewall_analysis": {},
            "recommendations": []
        }
        
        # Discover network topology
        discovery_methods = ["ping", "arp", "port_scan", "dns"]
        if assessment_type == "quick":
            discovery_methods = ["ping", "arp"]
        
        topology = await self.topology_mapper.discover_network_topology(
            target_ranges, discovery_methods
        )
        assessment["topology_discovery"] = topology
        
        # Perform security analysis
        security_analysis = await self._perform_security_analysis(topology)
        assessment["security_analysis"] = security_analysis
        
        # Analyze firewall configuration
        firewall_analysis = await self._analyze_firewall_configuration()
        assessment["firewall_analysis"] = firewall_analysis
        
        # Generate recommendations
        recommendations = await self._generate_security_recommendations(
            topology, security_analysis, firewall_analysis
        )
        assessment["recommendations"] = recommendations
        
        assessment["end_time"] = datetime.utcnow().isoformat()
        assessment["duration_seconds"] = (
            datetime.fromisoformat(assessment["end_time"]) - 
            datetime.fromisoformat(assessment["start_time"])
        ).total_seconds()
        
        logger.info(f"Network assessment completed: {assessment['assessment_id']}")
        
        return assessment
    
    async def _perform_security_analysis(self, topology: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed security analysis of discovered topology"""
        
        analysis = {
            "security_score": 0,
            "vulnerabilities": [],
            "exposed_services": [],
            "network_zones": {},
            "compliance_issues": [],
            "threat_indicators": []
        }
        
        discovered_hosts = topology.get("discovered_hosts", {})
        
        # Analyze each discovered host
        for ip, host_info in discovered_hosts.items():
            host_analysis = await self._analyze_host_security(ip, host_info)
            
            # Aggregate findings
            analysis["vulnerabilities"].extend(host_analysis.get("vulnerabilities", []))
            analysis["exposed_services"].extend(host_analysis.get("exposed_services", []))
            analysis["threat_indicators"].extend(host_analysis.get("threat_indicators", []))
        
        # Analyze network segments
        for segment_name, segment_info in topology.get("network_segments", {}).items():
            segment_analysis = await self._analyze_segment_security(segment_name, segment_info)
            analysis["network_zones"][segment_name] = segment_analysis
        
        # Calculate overall security score
        analysis["security_score"] = await self._calculate_security_score(analysis)
        
        return analysis
    
    async def _analyze_host_security(self, ip: str, host_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security of individual host"""
        
        host_analysis = {
            "ip": ip,
            "vulnerabilities": [],
            "exposed_services": [],
            "threat_indicators": [],
            "security_score": 100
        }
        
        services = host_info.get("services", [])
        
        # Check for insecure services
        insecure_services = {21: "FTP", 23: "Telnet", 135: "RPC", 139: "NetBIOS", 445: "SMB"}
        
        for service in services:
            port = service.get("port")
            if port in insecure_services:
                vulnerability = {
                    "type": "insecure_service",
                    "severity": "high",
                    "port": port,
                    "service": insecure_services[port],
                    "description": f"Insecure service {insecure_services[port]} detected on port {port}",
                    "recommendation": f"Disable or secure {insecure_services[port]} service"
                }
                host_analysis["vulnerabilities"].append(vulnerability)
                host_analysis["security_score"] -= 20
        
        # Check for exposed management services
        management_ports = {22: "SSH", 3389: "RDP", 5900: "VNC"}
        
        for service in services:
            port = service.get("port")
            if port in management_ports:
                exposed_service = {
                    "type": "management_service",
                    "port": port,
                    "service": management_ports[port],
                    "risk_level": "medium",
                    "description": f"Management service {management_ports[port]} exposed on port {port}"
                }
                host_analysis["exposed_services"].append(exposed_service)
                host_analysis["security_score"] -= 10
        
        return host_analysis
    
    async def _analyze_segment_security(self, segment_name: str, segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security of network segment"""
        
        segment_analysis = {
            "segment": segment_name,
            "security_classification": "internal",
            "isolation_score": 0,
            "access_controls": [],
            "recommendations": []
        }
        
        services = segment_info.get("services", [])
        host_count = segment_info.get("host_count", 0)
        
        # Classify segment based on services
        if "http" in services or "https" in services:
            segment_analysis["security_classification"] = "dmz"
            segment_analysis["recommendations"].append("Implement web application firewall")
        
        if "ssh" in services and host_count < 5:
            segment_analysis["security_classification"] = "management"
            segment_analysis["recommendations"].append("Implement strict access controls for management network")
        
        # Calculate isolation score based on service diversity
        unique_services = len(set(services))
        if unique_services > 5:
            segment_analysis["isolation_score"] = 30  # Poor isolation
            segment_analysis["recommendations"].append("Consider micro-segmentation to reduce service exposure")
        elif unique_services > 2:
            segment_analysis["isolation_score"] = 60  # Moderate isolation
        else:
            segment_analysis["isolation_score"] = 90  # Good isolation
        
        return segment_analysis
    
    async def _calculate_security_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate overall network security score"""
        
        base_score = 100
        
        # Deduct points for vulnerabilities
        critical_vulns = len([v for v in analysis["vulnerabilities"] if v.get("severity") == "critical"])
        high_vulns = len([v for v in analysis["vulnerabilities"] if v.get("severity") == "high"])
        medium_vulns = len([v for v in analysis["vulnerabilities"] if v.get("severity") == "medium"])
        
        base_score -= (critical_vulns * 30)
        base_score -= (high_vulns * 15)
        base_score -= (medium_vulns * 5)
        
        # Deduct points for exposed services
        exposed_count = len(analysis["exposed_services"])
        base_score -= (exposed_count * 5)
        
        # Deduct points for threat indicators
        threat_count = len(analysis["threat_indicators"])
        base_score -= (threat_count * 10)
        
        return max(0, base_score)
    
    async def _analyze_firewall_configuration(self) -> Dict[str, Any]:
        """Analyze current firewall configuration"""
        
        firewall_status = await self.firewall_manager.get_firewall_status()
        
        analysis = {
            "configuration_score": 100,
            "rule_analysis": {},
            "security_gaps": [],
            "rule_optimization": [],
            "compliance_status": {}
        }
        
        # Analyze rules
        total_rules = firewall_status["total_rules"]
        enabled_rules = firewall_status["enabled_rules"]
        
        if total_rules == 0:
            analysis["security_gaps"].append("No firewall rules configured")
            analysis["configuration_score"] -= 50
        elif enabled_rules < total_rules * 0.8:
            analysis["security_gaps"].append("Many firewall rules are disabled")
            analysis["configuration_score"] -= 20
        
        # Check for default deny policy
        rules = self.firewall_manager.firewall_rules
        has_default_deny = any(
            rule["action"] == "DENY" and rule["source"] == "*" and rule["destination"] == "*"
            for rule in rules
        )
        
        if not has_default_deny:
            analysis["security_gaps"].append("No default deny policy configured")
            analysis["configuration_score"] -= 30
        
        analysis["rule_analysis"] = {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "allow_rules": len([r for r in rules if r["action"] == "ALLOW"]),
            "deny_rules": len([r for r in rules if r["action"] in ["DENY", "DROP"]]),
            "has_default_deny": has_default_deny
        }
        
        return analysis
    
    async def _generate_security_recommendations(
        self,
        topology: Dict[str, Any],
        security_analysis: Dict[str, Any],
        firewall_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive security recommendations"""
        
        recommendations = []
        
        # Network topology recommendations
        discovered_hosts = topology.get("discovered_hosts", {})
        if len(discovered_hosts) > 50:
            recommendations.append({
                "category": "network_segmentation",
                "priority": "high",
                "title": "Implement Network Segmentation",
                "description": "Large number of hosts detected. Consider implementing network segmentation.",
                "implementation": "Deploy VLANs or micro-segmentation solution"
            })
        
        # Security analysis recommendations
        critical_vulns = len([v for v in security_analysis["vulnerabilities"] if v.get("severity") == "critical"])
        if critical_vulns > 0:
            recommendations.append({
                "category": "vulnerability_management",
                "priority": "critical",
                "title": "Address Critical Vulnerabilities",
                "description": f"{critical_vulns} critical vulnerabilities detected",
                "implementation": "Immediately patch or disable vulnerable services"
            })
        
        exposed_services = security_analysis.get("exposed_services", [])
        if len(exposed_services) > 10:
            recommendations.append({
                "category": "access_control",
                "priority": "high", 
                "title": "Reduce Service Exposure",
                "description": f"{len(exposed_services)} services exposed on network",
                "implementation": "Implement principle of least privilege and service hardening"
            })
        
        # Firewall recommendations
        if firewall_analysis["configuration_score"] < 70:
            recommendations.append({
                "category": "firewall_management",
                "priority": "high",
                "title": "Improve Firewall Configuration",
                "description": "Firewall configuration needs improvement",
                "implementation": "Review and optimize firewall rules, implement default deny policy"
            })
        
        # Zero-trust recommendations
        recommendations.append({
            "category": "zero_trust",
            "priority": "medium",
            "title": "Implement Zero Trust Architecture",
            "description": "Enhance security with zero trust principles",
            "implementation": "Deploy identity-based access controls and continuous monitoring"
        })
        
        return recommendations
    
    async def get_networking_status(self) -> Dict[str, Any]:
        """Get comprehensive networking service status"""
        
        firewall_status = await self.firewall_manager.get_firewall_status()
        security_events = await self.security_monitor.get_security_events(limit=10)
        
        return {
            "service_status": "active",
            "network_interfaces": len(self.network_interfaces),
            "network_endpoints": len(self.network_endpoints),
            "firewall_status": firewall_status,
            "recent_security_events": len(security_events),
            "monitoring_active": self.security_monitor.monitoring_active,
            "last_updated": datetime.utcnow().isoformat()
        }