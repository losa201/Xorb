#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Network Configuration & Telemetry Ports
Configure network policies, firewall rules, and telemetry services
"""

import asyncio
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import socket
import ipaddress
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortProtocol(Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    BOTH = "both"

class FirewallAction(Enum):
    """Firewall actions"""
    ALLOW = "allow"
    DENY = "deny"
    REJECT = "reject"

class NetworkZone(Enum):
    """Network security zones"""
    PUBLIC = "public"
    INTERNAL = "internal"
    MANAGEMENT = "management"
    DMZ = "dmz"

@dataclass
class PortConfiguration:
    """Port configuration for services"""
    service_name: str
    port: int
    protocol: PortProtocol
    description: str
    internal_only: bool = False
    health_check_path: Optional[str] = None
    required_for_telemetry: bool = False
    zone: NetworkZone = NetworkZone.INTERNAL

@dataclass
class FirewallRule:
    """Firewall rule configuration"""
    rule_id: str
    name: str
    action: FirewallAction
    protocol: PortProtocol
    port_range: str  # "80" or "8000-8010"
    source_ips: List[str]
    destination_ips: List[str]
    zone: NetworkZone
    priority: int = 100
    enabled: bool = True
    description: str = ""

@dataclass
class NetworkPolicy:
    """Network policy configuration"""
    policy_id: str
    name: str
    namespace: str
    ingress_rules: List[Dict[str, Any]]
    egress_rules: List[Dict[str, Any]]
    pod_selector: Dict[str, str]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class XORBNetworkConfigurator:
    """Network configuration and telemetry setup"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.configurator_id = str(uuid.uuid4())
        
        # Network configuration
        self.port_configurations: Dict[str, PortConfiguration] = {}
        self.firewall_rules: Dict[str, FirewallRule] = {}
        self.network_policies: Dict[str, NetworkPolicy] = {}
        
        # System information
        self.system_interfaces = []
        self.open_ports = []
        
        # Configuration paths
        self.firewall_config_path = self.config.get('firewall_config_path', '/tmp/xorb_firewall.conf')
        self.network_policy_path = self.config.get('network_policy_path', '/tmp/xorb_network_policies.yaml')
        
        # Initialize components
        self._initialize_xorb_service_ports()
        self._detect_system_interfaces()
        
        logger.info(f"Network Configurator initialized: {self.configurator_id}")
    
    def _initialize_xorb_service_ports(self):
        """Initialize XORB platform service port configurations"""
        try:
            # Core XORB services
            xorb_services = [
                PortConfiguration(
                    service_name="neural_orchestrator",
                    port=8003,
                    protocol=PortProtocol.TCP,
                    description="Neural AI Orchestrator Service",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="learning_service",
                    port=8004,
                    protocol=PortProtocol.TCP,
                    description="Autonomous Learning Service",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="threat_detection",
                    port=8005,
                    protocol=PortProtocol.TCP,
                    description="Neural Threat Detection Service",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="agent_cluster",
                    port=8006,
                    protocol=PortProtocol.TCP,
                    description="Agent Specialization Cluster",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="intelligence_fusion",
                    port=8007,
                    protocol=PortProtocol.TCP,
                    description="Intelligence Fusion Core",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="evolution_accelerator",
                    port=8008,
                    protocol=PortProtocol.TCP,
                    description="Autonomous Evolution Accelerator",
                    health_check_path="/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.INTERNAL
                ),
                
                # Infrastructure services
                PortConfiguration(
                    service_name="prometheus",
                    port=9090,
                    protocol=PortProtocol.TCP,
                    description="Prometheus Metrics Server",
                    health_check_path="/metrics",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="prometheus_autonomous",
                    port=9092,  
                    protocol=PortProtocol.TCP,
                    description="Prometheus Autonomous Metrics Server",
                    health_check_path="/metrics",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="prometheus_pushgateway",
                    port=9091,
                    protocol=PortProtocol.TCP,
                    description="Prometheus Push Gateway",
                    health_check_path="/metrics",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="grafana",
                    port=3000,
                    protocol=PortProtocol.TCP,
                    description="Grafana Visualization Dashboard",
                    health_check_path="/api/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="grafana_autonomous",
                    port=3002,
                    protocol=PortProtocol.TCP,
                    description="Grafana Autonomous Dashboard",
                    health_check_path="/api/health",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                
                # Database services
                PortConfiguration(
                    service_name="postgresql",
                    port=5432,
                    protocol=PortProtocol.TCP,
                    description="PostgreSQL Database",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="postgresql_autonomous",
                    port=5434,
                    protocol=PortProtocol.TCP,
                    description="PostgreSQL Autonomous Database",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="neo4j",
                    port=7474,
                    protocol=PortProtocol.TCP,
                    description="Neo4j Graph Database HTTP",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="neo4j_autonomous",
                    port=7476,
                    protocol=PortProtocol.TCP,
                    description="Neo4j Autonomous Graph Database HTTP",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="neo4j_bolt",
                    port=7687,
                    protocol=PortProtocol.TCP,
                    description="Neo4j Bolt Protocol",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="neo4j_autonomous_bolt",
                    port=7689,
                    protocol=PortProtocol.TCP,
                    description="Neo4j Autonomous Bolt Protocol",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="redis",
                    port=6379,
                    protocol=PortProtocol.TCP,
                    description="Redis Cache",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="redis_autonomous",
                    port=6381,
                    protocol=PortProtocol.TCP,
                    description="Redis Autonomous Cache",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                
                # Additional telemetry and management
                PortConfiguration(
                    service_name="alertmanager",
                    port=9093,
                    protocol=PortProtocol.TCP,
                    description="Prometheus Alertmanager",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="jaeger",
                    port=16686,
                    protocol=PortProtocol.TCP,
                    description="Jaeger Tracing UI",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                ),
                PortConfiguration(
                    service_name="elasticsearch",
                    port=9200,
                    protocol=PortProtocol.TCP,
                    description="Elasticsearch for Logging",
                    internal_only=True,
                    zone=NetworkZone.INTERNAL
                ),
                PortConfiguration(
                    service_name="kibana",
                    port=5601,
                    protocol=PortProtocol.TCP,
                    description="Kibana Log Visualization",
                    required_for_telemetry=True,
                    zone=NetworkZone.MANAGEMENT
                )
            ]
            
            # Store configurations
            for port_config in xorb_services:
                self.port_configurations[port_config.service_name] = port_config
            
            logger.info(f"Initialized {len(xorb_services)} service port configurations")
            
        except Exception as e:
            logger.error(f"Failed to initialize service ports: {e}")
    
    def _detect_system_interfaces(self):
        """Detect system network interfaces"""
        try:
            import netifaces
            
            interfaces = []
            for interface in netifaces.interfaces():
                if interface != 'lo':  # Skip loopback
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        for addr_info in addrs[netifaces.AF_INET]:
                            interfaces.append({
                                'interface': interface,
                                'ip': addr_info['addr'],
                                'netmask': addr_info['netmask']
                            })
            
            self.system_interfaces = interfaces
            logger.info(f"Detected {len(interfaces)} network interfaces")
            
        except ImportError:
            # Fallback method
            try:
                result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Network interfaces detected via ip command")
                else:
                    logger.warning("Could not detect network interfaces")
            except:
                logger.warning("Network interface detection failed")
        except Exception as e:
            logger.error(f"Failed to detect network interfaces: {e}")
    
    async def check_port_availability(self, port: int, host: str = 'localhost') -> bool:
        """Check if a port is available (not in use)"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port is available if connection fails
        except Exception as e:
            logger.error(f"Failed to check port {port}: {e}")
            return False
    
    async def scan_open_ports(self, host: str = 'localhost', port_range: Tuple[int, int] = (1, 65535)) -> List[int]:
        """Scan for open ports on the system"""
        try:
            open_ports = []
            start_port, end_port = port_range
            
            # Limit scan range for performance
            if end_port - start_port > 1000:
                logger.warning(f"Large port range ({end_port - start_port} ports), limiting scan")
                end_port = min(start_port + 1000, end_port)
            
            for port in range(start_port, end_port + 1):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.1)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        open_ports.append(port)
            
            self.open_ports = open_ports
            logger.info(f"Found {len(open_ports)} open ports")
            return open_ports
            
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            return []
    
    async def configure_firewall_rules(self) -> bool:
        """Configure firewall rules for XORB services"""
        try:
            # Generate firewall rules based on service configurations
            await self._generate_service_firewall_rules()
            
            # Apply firewall rules (platform-specific)
            success = await self._apply_firewall_rules()
            
            if success:
                logger.info("Firewall rules configured successfully")
                return True
            else:
                logger.error("Failed to apply firewall rules")
                return False
                
        except Exception as e:
            logger.error(f"Firewall configuration failed: {e}")
            return False
    
    async def _generate_service_firewall_rules(self):
        """Generate firewall rules for XORB services"""
        try:
            # Allow internal networks
            internal_networks = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '127.0.0.0/8']
            
            # Generate rules for each service
            for service_name, port_config in self.port_configurations.items():
                rule_id = f"allow_{service_name}"
                
                # Determine source IPs based on service type
                if port_config.zone == NetworkZone.PUBLIC:
                    source_ips = ['0.0.0.0/0']  # Allow from anywhere
                elif port_config.zone == NetworkZone.MANAGEMENT:
                    source_ips = internal_networks  # Management network only
                else:
                    source_ips = internal_networks  # Internal networks only
                
                # Create firewall rule
                rule = FirewallRule(
                    rule_id=rule_id,
                    name=f"Allow {service_name}",
                    action=FirewallAction.ALLOW,
                    protocol=port_config.protocol,
                    port_range=str(port_config.port),
                    source_ips=source_ips,
                    destination_ips=['0.0.0.0/0'],
                    zone=port_config.zone,
                    priority=100,
                    description=f"Allow access to {port_config.description}"
                )
                
                self.firewall_rules[rule_id] = rule
            
            # Add default deny rule
            deny_rule = FirewallRule(
                rule_id="default_deny",
                name="Default Deny",
                action=FirewallAction.DENY,
                protocol=PortProtocol.BOTH,
                port_range="1-65535",
                source_ips=['0.0.0.0/0'],
                destination_ips=['0.0.0.0/0'],
                zone=NetworkZone.PUBLIC,
                priority=999,
                description="Default deny all other traffic"
            )
            
            self.firewall_rules["default_deny"] = deny_rule
            
            logger.info(f"Generated {len(self.firewall_rules)} firewall rules")
            
        except Exception as e:
            logger.error(f"Failed to generate firewall rules: {e}")
    
    async def _apply_firewall_rules(self) -> bool:
        """Apply firewall rules to the system"""
        try:
            # Try different firewall systems
            
            # Check for ufw (Ubuntu/Debian)
            if await self._check_command_exists('ufw'):
                return await self._apply_ufw_rules()
            
            # Check for firewalld (CentOS/RHEL/Fedora)
            elif await self._check_command_exists('firewall-cmd'):
                return await self._apply_firewalld_rules()
            
            # Check for iptables
            elif await self._check_command_exists('iptables'):
                return await self._apply_iptables_rules()
            
            else:
                logger.warning("No supported firewall system found")
                # Create configuration file for manual application
                await self._export_firewall_config()
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply firewall rules: {e}")
            return False
    
    async def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists on the system"""
        try:
            result = subprocess.run(['which', command], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _apply_ufw_rules(self) -> bool:
        """Apply firewall rules using UFW"""
        try:
            # Enable UFW
            subprocess.run(['sudo', 'ufw', '--force', 'enable'], check=False)
            
            # Reset to default
            subprocess.run(['sudo', 'ufw', '--force', 'reset'], check=False)
            
            # Set default policies
            subprocess.run(['sudo', 'ufw', 'default', 'deny', 'incoming'], check=False)
            subprocess.run(['sudo', 'ufw', 'default', 'allow', 'outgoing'], check=False)
            
            # Apply service rules
            for rule in self.firewall_rules.values():
                if rule.enabled and rule.action == FirewallAction.ALLOW:
                    for source_ip in rule.source_ips:
                        if source_ip == '0.0.0.0/0':
                            cmd = ['sudo', 'ufw', 'allow', rule.port_range]
                        else:
                            cmd = ['sudo', 'ufw', 'allow', 'from', source_ip, 'to', 'any', 'port', rule.port_range]
                        
                        if rule.protocol != PortProtocol.BOTH:
                            cmd.extend(['proto', rule.protocol.value])
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.warning(f"UFW rule failed: {' '.join(cmd)} - {result.stderr}")
            
            logger.info("UFW firewall rules applied")
            return True
            
        except Exception as e:
            logger.error(f"UFW rule application failed: {e}")
            return False
    
    async def _apply_firewalld_rules(self) -> bool:
        """Apply firewall rules using firewalld"""
        try:
            # Start firewalld service
            subprocess.run(['sudo', 'systemctl', 'start', 'firewalld'], check=False)
            
            # Apply service rules
            for rule in self.firewall_rules.values():
                if rule.enabled and rule.action == FirewallAction.ALLOW:
                    # Add port
                    protocol = rule.protocol.value if rule.protocol != PortProtocol.BOTH else 'tcp'
                    cmd = ['sudo', 'firewall-cmd', '--permanent', '--add-port', f"{rule.port_range}/{protocol}"]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.warning(f"firewalld rule failed: {' '.join(cmd)} - {result.stderr}")
            
            # Reload firewall
            subprocess.run(['sudo', 'firewall-cmd', '--reload'], check=False)
            
            logger.info("firewalld rules applied")
            return True
            
        except Exception as e:
            logger.error(f"firewalld rule application failed: {e}")
            return False
    
    async def _apply_iptables_rules(self) -> bool:
        """Apply firewall rules using iptables"""
        try:
            # Flush existing rules
            subprocess.run(['sudo', 'iptables', '-F'], check=False)
            
            # Set default policies
            subprocess.run(['sudo', 'iptables', '-P', 'INPUT', 'DROP'], check=False)
            subprocess.run(['sudo', 'iptables', '-P', 'FORWARD', 'DROP'], check=False)
            subprocess.run(['sudo', 'iptables', '-P', 'OUTPUT', 'ACCEPT'], check=False)
            
            # Allow loopback
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-i', 'lo', '-j', 'ACCEPT'], check=False)
            
            # Allow established connections
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'], check=False)
            
            # Apply service rules
            for rule in self.firewall_rules.values():
                if rule.enabled and rule.action == FirewallAction.ALLOW:
                    protocol = rule.protocol.value if rule.protocol != PortProtocol.BOTH else 'tcp'
                    
                    for source_ip in rule.source_ips:
                        cmd = ['sudo', 'iptables', '-A', 'INPUT', '-p', protocol, '--dport', rule.port_range]
                        
                        if source_ip != '0.0.0.0/0':
                            cmd.extend(['-s', source_ip])
                        
                        cmd.extend(['-j', 'ACCEPT'])
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.warning(f"iptables rule failed: {' '.join(cmd)} - {result.stderr}")
            
            logger.info("iptables rules applied")
            return True
            
        except Exception as e:
            logger.error(f"iptables rule application failed: {e}")
            return False
    
    async def _export_firewall_config(self):
        """Export firewall configuration for manual application"""
        try:
            config_lines = []
            config_lines.append("# XORB Platform Firewall Configuration")
            config_lines.append(f"# Generated at: {datetime.now().isoformat()}")
            config_lines.append("")
            
            # UFW format
            config_lines.append("# UFW Commands:")
            config_lines.append("sudo ufw --force reset")
            config_lines.append("sudo ufw default deny incoming")
            config_lines.append("sudo ufw default allow outgoing")
            
            for rule in self.firewall_rules.values():
                if rule.enabled and rule.action == FirewallAction.ALLOW:
                    for source_ip in rule.source_ips:
                        if source_ip == '0.0.0.0/0':
                            config_lines.append(f"sudo ufw allow {rule.port_range}")
                        else:
                            config_lines.append(f"sudo ufw allow from {source_ip} to any port {rule.port_range}")
            
            config_lines.append("sudo ufw --force enable")
            config_lines.append("")
            
            # iptables format
            config_lines.append("# iptables Commands:")
            config_lines.append("sudo iptables -F")
            config_lines.append("sudo iptables -P INPUT DROP")
            config_lines.append("sudo iptables -P FORWARD DROP")
            config_lines.append("sudo iptables -P OUTPUT ACCEPT")
            config_lines.append("sudo iptables -A INPUT -i lo -j ACCEPT")
            config_lines.append("sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT")
            
            for rule in self.firewall_rules.values():
                if rule.enabled and rule.action == FirewallAction.ALLOW:
                    protocol = rule.protocol.value if rule.protocol != PortProtocol.BOTH else 'tcp'
                    for source_ip in rule.source_ips:
                        if source_ip == '0.0.0.0/0':
                            config_lines.append(f"sudo iptables -A INPUT -p {protocol} --dport {rule.port_range} -j ACCEPT")
                        else:
                            config_lines.append(f"sudo iptables -A INPUT -p {protocol} -s {source_ip} --dport {rule.port_range} -j ACCEPT")
            
            # Write configuration file
            with open(self.firewall_config_path, 'w') as f:
                f.write('\n'.join(config_lines))
            
            logger.info(f"Firewall configuration exported to: {self.firewall_config_path}")
            
        except Exception as e:
            logger.error(f"Failed to export firewall config: {e}")
    
    async def configure_network_policies(self) -> bool:
        """Configure Kubernetes network policies"""
        try:
            # Generate network policies for XORB services
            await self._generate_network_policies()
            
            # Export network policies to YAML
            await self._export_network_policies()
            
            logger.info("Network policies configured")
            return True
            
        except Exception as e:
            logger.error(f"Network policy configuration failed: {e}")
            return False
    
    async def _generate_network_policies(self):
        """Generate Kubernetes network policies"""
        try:
            # Create network policy for each zone
            zones = {
                NetworkZone.INTERNAL: "xorb-internal",
                NetworkZone.MANAGEMENT: "xorb-management", 
                NetworkZone.PUBLIC: "xorb-public"
            }
            
            for zone, namespace in zones.items():
                # Get services in this zone
                zone_services = [svc for svc in self.port_configurations.values() if svc.zone == zone]
                
                if not zone_services:
                    continue
                
                # Create ingress rules
                ingress_rules = []
                
                if zone == NetworkZone.PUBLIC:
                    # Public zone allows external traffic
                    for service in zone_services:
                        ingress_rules.append({
                            'from': [],  # Allow from anywhere
                            'ports': [{
                                'protocol': 'TCP',
                                'port': service.port
                            }]
                        })
                else:
                    # Internal/Management zones only allow internal traffic
                    ingress_rules.append({
                        'from': [
                            {'namespaceSelector': {'matchLabels': {'name': 'xorb-internal'}}},
                            {'namespaceSelector': {'matchLabels': {'name': 'xorb-management'}}},
                            {'podSelector': {'matchLabels': {'app': 'xorb-platform'}}}
                        ],
                        'ports': [{'protocol': 'TCP', 'port': svc.port} for svc in zone_services]
                    })
                
                # Create egress rules (allow all outbound for now)
                egress_rules = [{}]  # Allow all egress
                
                # Create network policy
                policy = NetworkPolicy(
                    policy_id=f"xorb-{zone.value}-policy",
                    name=f"xorb-{zone.value}-network-policy",
                    namespace=namespace,
                    ingress_rules=ingress_rules,
                    egress_rules=egress_rules,
                    pod_selector={'app': 'xorb-platform'}
                )
                
                self.network_policies[policy.policy_id] = policy
            
            logger.info(f"Generated {len(self.network_policies)} network policies")
            
        except Exception as e:
            logger.error(f"Failed to generate network policies: {e}")
    
    async def _export_network_policies(self):
        """Export network policies to Kubernetes YAML"""
        try:
            import yaml
            
            policies_yaml = []
            
            for policy in self.network_policies.values():
                policy_yaml = {
                    'apiVersion': 'networking.k8s.io/v1',
                    'kind': 'NetworkPolicy',
                    'metadata': {
                        'name': policy.name,
                        'namespace': policy.namespace,
                        'labels': {
                            'app': 'xorb-platform',
                            'component': 'network-security'
                        }
                    },
                    'spec': {
                        'podSelector': {
                            'matchLabels': policy.pod_selector
                        },
                        'policyTypes': ['Ingress', 'Egress'],
                        'ingress': policy.ingress_rules,
                        'egress': policy.egress_rules
                    }
                }
                
                policies_yaml.append(policy_yaml)
            
            # Write YAML file
            with open(self.network_policy_path, 'w') as f:
                yaml.dump_all(policies_yaml, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Network policies exported to: {self.network_policy_path}")
            
        except ImportError:
            # Fallback without yaml library
            policy_text = []
            for policy in self.network_policies.values():
                policy_text.append(f"# Network Policy: {policy.name}")
                policy_text.append(f"# Namespace: {policy.namespace}")
                policy_text.append(f"# Ingress rules: {len(policy.ingress_rules)}")
                policy_text.append(f"# Egress rules: {len(policy.egress_rules)}")
                policy_text.append("")
            
            with open(self.network_policy_path, 'w') as f:
                f.write('\n'.join(policy_text))
                
        except Exception as e:
            logger.error(f"Failed to export network policies: {e}")
    
    async def verify_telemetry_connectivity(self) -> Dict[str, Any]:
        """Verify connectivity to telemetry services"""
        try:
            connectivity_results = {}
            
            # Check telemetry services
            telemetry_services = [svc for svc in self.port_configurations.values() 
                                if svc.required_for_telemetry]
            
            for service in telemetry_services:
                try:
                    # Test TCP connection
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(5)
                        result = sock.connect_ex(('localhost', service.port))
                        
                        connectivity_results[service.service_name] = {
                            'port': service.port,
                            'reachable': result == 0,
                            'description': service.description,
                            'health_check_path': service.health_check_path
                        }
                        
                        # Test health check if available
                        if result == 0 and service.health_check_path:
                            try:
                                import requests
                                health_url = f"http://localhost:{service.port}{service.health_check_path}"
                                response = requests.get(health_url, timeout=5)
                                connectivity_results[service.service_name]['health_check'] = {
                                    'status_code': response.status_code,
                                    'healthy': response.status_code == 200
                                }
                            except:
                                connectivity_results[service.service_name]['health_check'] = {
                                    'status_code': None,
                                    'healthy': False
                                }
                                
                except Exception as e:
                    connectivity_results[service.service_name] = {
                        'port': service.port,
                        'reachable': False,
                        'error': str(e)
                    }
            
            # Summary statistics
            total_services = len(connectivity_results)
            reachable_services = len([r for r in connectivity_results.values() if r.get('reachable', False)])
            healthy_services = len([r for r in connectivity_results.values() 
                                  if r.get('health_check', {}).get('healthy', False)])
            
            return {
                'connectivity_results': connectivity_results,
                'summary': {
                    'total_telemetry_services': total_services,
                    'reachable_services': reachable_services,
                    'healthy_services': healthy_services,
                    'connectivity_percentage': (reachable_services / total_services * 100) if total_services > 0 else 0,
                    'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Telemetry connectivity verification failed: {e}")
            return {'error': str(e)}
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network configuration status"""
        try:
            # Get current open ports
            current_open_ports = await self.scan_open_ports(port_range=(8000, 9100))
            
            # Verify telemetry connectivity
            telemetry_status = await self.verify_telemetry_connectivity()
            
            # Service port analysis
            configured_ports = [svc.port for svc in self.port_configurations.values()]
            telemetry_ports = [svc.port for svc in self.port_configurations.values() if svc.required_for_telemetry]
            
            return {
                'configurator_id': self.configurator_id,
                'port_configurations': {
                    'total_services': len(self.port_configurations),
                    'telemetry_services': len([svc for svc in self.port_configurations.values() if svc.required_for_telemetry]),
                    'internal_only_services': len([svc for svc in self.port_configurations.values() if svc.internal_only]),
                    'configured_ports': configured_ports,
                    'telemetry_ports': telemetry_ports
                },
                'firewall_configuration': {
                    'total_rules': len(self.firewall_rules),
                    'enabled_rules': len([rule for rule in self.firewall_rules.values() if rule.enabled]),
                    'allow_rules': len([rule for rule in self.firewall_rules.values() if rule.action == FirewallAction.ALLOW]),
                    'config_file': self.firewall_config_path
                },
                'network_policies': {
                    'total_policies': len(self.network_policies),
                    'enabled_policies': len([policy for policy in self.network_policies.values() if policy.enabled]),
                    'policy_file': self.network_policy_path
                },
                'network_discovery': {
                    'system_interfaces': len(self.system_interfaces),
                    'open_ports_found': len(current_open_ports),
                    'open_ports': current_open_ports
                },
                'telemetry_status': telemetry_status,
                'configuration_files': {
                    'firewall_config': self.firewall_config_path,
                    'network_policies': self.network_policy_path
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {'error': str(e)}

# Example usage and testing
async def main():
    """Example usage of XORB Network Configurator"""
    try:
        print("🌐 XORB Network Configurator initializing...")
        
        # Initialize network configurator
        network_config = XORBNetworkConfigurator({
            'firewall_config_path': '/tmp/xorb_firewall.conf',
            'network_policy_path': '/tmp/xorb_network_policies.yaml'
        })
        
        print("✅ Network configurator initialized")
        
        # Scan for open ports
        print("\n🔍 Scanning for open ports...")
        open_ports = await network_config.scan_open_ports(port_range=(8000, 9100))
        print(f"✅ Found {len(open_ports)} open ports: {open_ports}")
        
        # Configure firewall rules
        print("\n🔥 Configuring firewall rules...")
        firewall_success = await network_config.configure_firewall_rules()
        if firewall_success:
            print("✅ Firewall rules configured")
        else:
            print("⚠️ Firewall rules exported for manual configuration")
        
        # Configure network policies
        print("\n📋 Configuring network policies...")
        policy_success = await network_config.configure_network_policies()
        if policy_success:
            print("✅ Network policies configured")
        
        # Verify telemetry connectivity
        print("\n📊 Verifying telemetry connectivity...")
        telemetry_status = await network_config.verify_telemetry_connectivity()
        
        if 'summary' in telemetry_status:
            summary = telemetry_status['summary']
            print(f"✅ Telemetry status:")
            print(f"  - Total services: {summary['total_telemetry_services']}")
            print(f"  - Reachable: {summary['reachable_services']} ({summary['connectivity_percentage']:.1f}%)")
            print(f"  - Healthy: {summary['healthy_services']} ({summary['health_percentage']:.1f}%)")
        
        # Get comprehensive status
        status = await network_config.get_network_status()
        print(f"\n📊 Network Configuration Status:")
        print(f"- Service Configurations: {status['port_configurations']['total_services']}")
        print(f"- Telemetry Services: {status['port_configurations']['telemetry_services']}")
        print(f"- Firewall Rules: {status['firewall_configuration']['total_rules']}")
        print(f"- Network Policies: {status['network_policies']['total_policies']}")
        print(f"- Open Ports Found: {status['network_discovery']['open_ports_found']}")
        
        print(f"\n📁 Configuration Files:")
        print(f"- Firewall Config: {status['configuration_files']['firewall_config']}")
        print(f"- Network Policies: {status['configuration_files']['network_policies']}")
        
        print(f"\n✅ XORB Network Configurator demonstration completed!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())