#!/usr/bin/env python3
"""
XORB Autonomous Self-Configuration System

Advanced DevOps AI system that automatically detects and configures XORB
for optimal performance on any host machine, from Raspberry Pi to EPYC servers.

Features:
- Hardware detection and profiling
- OS and runtime capability assessment  
- Dynamic resource allocation
- Adaptive service selection
- Configuration validation and reporting

Author: XORB DevOps AI
Version: 2.0.0
"""

import os
import sys
import json
import subprocess
import platform
import psutil
import docker
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemProfile(Enum):
    """System profile classifications"""
    RPI = "RPI"              # Raspberry Pi / ARM SBC
    CLOUD_MICRO = "CLOUD_MICRO"    # 1-2 cores, 2-4GB RAM
    CLOUD_SMALL = "CLOUD_SMALL"    # 2-4 cores, 4-8GB RAM
    CLOUD_MEDIUM = "CLOUD_MEDIUM"   # 4-8 cores, 8-16GB RAM
    BARE_METAL = "BARE_METAL"      # 16+ cores, 32+ GB RAM
    EPYC_SERVER = "EPYC_SERVER"    # 32+ cores, 64+ GB RAM


class XorbMode(Enum):
    """XORB deployment modes"""
    SIMPLE = "SIMPLE"        # Minimal services, low resource usage
    ENHANCED = "ENHANCED"    # Standard services with monitoring
    FULL = "FULL"           # Complete stack with all features


@dataclass
class SystemCapabilities:
    """Detected system capabilities"""
    os_type: str
    os_version: str
    architecture: str
    cpu_cores: int
    cpu_threads: int
    cpu_frequency: float
    ram_total_gb: float
    ram_available_gb: float
    disk_space_gb: float
    is_arm: bool
    is_virtualized: bool
    docker_version: str
    docker_buildkit: bool
    docker_compose_version: str
    podman_available: bool
    network_interfaces: List[str]
    dns_servers: List[str]
    system_load: float
    profile: SystemProfile


@dataclass
class XorbConfiguration:
    """Generated XORB configuration"""
    mode: XorbMode
    system_profile: SystemProfile
    agent_concurrency: int
    max_concurrent_missions: int
    worker_threads: int
    monitoring_enabled: bool
    memory_limit_mb: int
    cpu_limit: float
    services_enabled: List[str]
    resource_limits: Dict[str, Dict[str, Any]]
    environment_variables: Dict[str, str]


class XorbAutoConfigurator:
    """
    Advanced auto-configuration system for XORB ecosystem.
    
    Detects hardware, OS, and runtime capabilities to generate
    optimal configuration for any deployment environment.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Configuration templates
        self.service_profiles = {
            SystemProfile.RPI: {
                'services': ['postgres', 'redis', 'api', 'worker'],
                'monitoring': False,
                'agent_concurrency': 2,
                'max_missions': 1,
                'worker_threads': 2
            },
            SystemProfile.CLOUD_MICRO: {
                'services': ['postgres', 'redis', 'api', 'worker', 'orchestrator'],
                'monitoring': False,
                'agent_concurrency': 4,
                'max_missions': 2,
                'worker_threads': 2
            },
            SystemProfile.CLOUD_SMALL: {
                'services': ['postgres', 'redis', 'temporal', 'nats', 'api', 'worker', 'orchestrator'],
                'monitoring': True,
                'agent_concurrency': 8,
                'max_missions': 3,
                'worker_threads': 4
            },
            SystemProfile.CLOUD_MEDIUM: {
                'services': ['postgres', 'redis', 'temporal', 'nats', 'neo4j', 'qdrant', 
                           'api', 'worker', 'orchestrator', 'scanner-go', 'prometheus'],
                'monitoring': True,
                'agent_concurrency': 16,
                'max_missions': 5,
                'worker_threads': 6
            },
            SystemProfile.BARE_METAL: {
                'services': ['postgres', 'redis', 'temporal', 'nats', 'neo4j', 'qdrant',
                           'api', 'worker', 'orchestrator', 'scanner-go', 
                           'prometheus', 'grafana', 'tempo'],
                'monitoring': True,
                'agent_concurrency': 32,
                'max_missions': 10,
                'worker_threads': 8
            },
            SystemProfile.EPYC_SERVER: {
                'services': ['postgres', 'redis', 'temporal', 'nats', 'neo4j', 'qdrant',
                           'api', 'worker', 'orchestrator', 'scanner-go',
                           'prometheus', 'grafana', 'tempo', 'alertmanager'],
                'monitoring': True,
                'agent_concurrency': 64,
                'max_missions': 20,
                'worker_threads': 16
            }
        }
        
        logger.info("🤖 XORB Auto-Configurator initialized")
    
    def detect_system_capabilities(self) -> SystemCapabilities:
        """Comprehensive system capability detection"""
        logger.info("🔍 Detecting system capabilities...")
        
        try:
            # Basic system info
            os_type = platform.system()
            os_version = platform.release()
            architecture = platform.machine()
            
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            cpu_frequency = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # Memory information
            memory = psutil.virtual_memory()
            ram_total_gb = memory.total / (1024**3)
            ram_available_gb = memory.available / (1024**3)
            
            # Disk space
            disk = psutil.disk_usage('/')
            disk_space_gb = disk.free / (1024**3)
            
            # Architecture detection
            is_arm = architecture.lower() in ['arm64', 'aarch64', 'armv7l', 'armv6l']
            
            # Virtualization detection
            is_virtualized = self._detect_virtualization()
            
            # Docker capabilities
            docker_info = self._detect_docker_capabilities()
            
            # Network information
            network_interfaces = self._get_network_interfaces()
            dns_servers = self._get_dns_servers()
            
            # Current system load
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            # Determine system profile
            profile = self._classify_system_profile(
                cpu_cores, ram_total_gb, is_arm, is_virtualized
            )
            
            capabilities = SystemCapabilities(
                os_type=os_type,
                os_version=os_version,
                architecture=architecture,
                cpu_cores=cpu_cores,
                cpu_threads=cpu_threads,
                cpu_frequency=cpu_frequency,
                ram_total_gb=ram_total_gb,
                ram_available_gb=ram_available_gb,
                disk_space_gb=disk_space_gb,
                is_arm=is_arm,
                is_virtualized=is_virtualized,
                docker_version=docker_info['version'],
                docker_buildkit=docker_info['buildkit'],
                docker_compose_version=docker_info['compose_version'],
                podman_available=docker_info['podman_available'],
                network_interfaces=network_interfaces,
                dns_servers=dns_servers,
                system_load=system_load,
                profile=profile
            )
            
            logger.info(f"✅ System classified as: {profile.value}")
            logger.info(f"   CPU: {cpu_cores} cores ({cpu_threads} threads)")
            logger.info(f"   RAM: {ram_total_gb:.1f}GB total, {ram_available_gb:.1f}GB available")
            logger.info(f"   Architecture: {architecture}")
            logger.info(f"   Docker: {docker_info['version']}")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to detect system capabilities: {e}")
            raise
    
    def _detect_virtualization(self) -> bool:
        """Detect if running in virtualized environment"""
        try:
            # Check common virtualization indicators
            virt_indicators = [
                '/proc/vz',  # OpenVZ
                '/proc/xen',  # Xen
                '/sys/class/dmi/id/sys_vendor'  # Check vendor
            ]
            
            for indicator in virt_indicators:
                if os.path.exists(indicator):
                    if 'sys_vendor' in indicator:
                        try:
                            with open(indicator, 'r') as f:
                                vendor = f.read().strip().lower()
                                if any(v in vendor for v in ['vmware', 'virtualbox', 'qemu', 'kvm']):
                                    return True
                        except:
                            pass
                    else:
                        return True
            
            # Check for container environments
            if os.path.exists('/.dockerenv') or os.environ.get('container'):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_docker_capabilities(self) -> Dict[str, Any]:
        """Detect Docker and container runtime capabilities"""
        docker_info = {
            'version': 'unknown',
            'buildkit': False,
            'compose_version': 'unknown',
            'podman_available': False
        }
        
        try:
            # Docker version and BuildKit support
            client = docker.from_env()
            version_info = client.version()
            docker_info['version'] = version_info.get('Version', 'unknown')
            
            # Check BuildKit support
            try:
                buildkit_env = os.environ.get('DOCKER_BUILDKIT', '0')
                docker_info['buildkit'] = buildkit_env == '1' or 'buildkit' in str(version_info).lower()
            except:
                docker_info['buildkit'] = False
            
        except Exception as e:
            logger.warning(f"Docker detection failed: {e}")
        
        try:
            # Docker Compose version
            result = subprocess.run(['docker', 'compose', 'version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_info['compose_version'] = result.stdout.strip().split()[-1]
        except:
            try:
                # Try legacy docker-compose
                result = subprocess.run(['docker-compose', '--version'], 
                                     capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    docker_info['compose_version'] = result.stdout.strip().split()[-1]
            except:
                pass
        
        try:
            # Podman availability
            result = subprocess.run(['podman', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            docker_info['podman_available'] = result.returncode == 0
        except:
            pass
        
        return docker_info
    
    def _get_network_interfaces(self) -> List[str]:
        """Get available network interfaces"""
        try:
            interfaces = []
            for interface, addrs in psutil.net_if_addrs().items():
                if any(addr.family == 2 for addr in addrs):  # IPv4
                    interfaces.append(interface)
            return interfaces
        except:
            return ['eth0', 'lo']
    
    def _get_dns_servers(self) -> List[str]:
        """Get configured DNS servers"""
        try:
            dns_servers = []
            if os.path.exists('/etc/resolv.conf'):
                with open('/etc/resolv.conf', 'r') as f:
                    for line in f:
                        if line.startswith('nameserver'):
                            dns_servers.append(line.split()[1])
            return dns_servers[:3]  # Return first 3
        except:
            return ['8.8.8.8', '1.1.1.1']
    
    def _classify_system_profile(self, cpu_cores: int, ram_gb: float, 
                                is_arm: bool, is_virtualized: bool) -> SystemProfile:
        """Classify system into performance profile"""
        
        # ARM devices (likely Raspberry Pi or similar SBC)
        if is_arm:
            return SystemProfile.RPI
        
        # High-end server classification
        if cpu_cores >= 32 and ram_gb >= 64:
            return SystemProfile.EPYC_SERVER
        elif cpu_cores >= 16 and ram_gb >= 32:
            return SystemProfile.BARE_METAL
        
        # Cloud/VM classification
        if is_virtualized or cpu_cores <= 8:
            if cpu_cores <= 2 and ram_gb <= 4:
                return SystemProfile.CLOUD_MICRO
            elif cpu_cores <= 4 and ram_gb <= 8:
                return SystemProfile.CLOUD_SMALL
            else:
                return SystemProfile.CLOUD_MEDIUM
        
        # Default to bare metal for unclassified high-spec systems
        return SystemProfile.BARE_METAL
    
    def generate_configuration(self, capabilities: SystemCapabilities) -> XorbConfiguration:
        """Generate optimal XORB configuration based on capabilities"""
        logger.info("⚙️ Generating XORB configuration...")
        
        try:
            profile = capabilities.profile
            profile_config = self.service_profiles[profile]
            
            # Determine XORB mode
            if profile in [SystemProfile.RPI, SystemProfile.CLOUD_MICRO]:
                mode = XorbMode.SIMPLE
            elif profile in [SystemProfile.CLOUD_SMALL, SystemProfile.CLOUD_MEDIUM]:
                mode = XorbMode.ENHANCED
            else:
                mode = XorbMode.FULL
            
            # Calculate resource limits
            memory_limit_mb = max(512, int(capabilities.ram_available_gb * 1024 * 0.8))
            cpu_limit = min(capabilities.cpu_cores, capabilities.cpu_threads * 0.8)
            
            # Generate resource limits for services
            resource_limits = self._generate_resource_limits(capabilities, profile_config)
            
            # Generate environment variables
            env_vars = self._generate_environment_variables(capabilities, profile_config)
            
            config = XorbConfiguration(
                mode=mode,
                system_profile=profile,
                agent_concurrency=profile_config['agent_concurrency'],
                max_concurrent_missions=profile_config['max_missions'],
                worker_threads=profile_config['worker_threads'],
                monitoring_enabled=profile_config['monitoring'],
                memory_limit_mb=memory_limit_mb,
                cpu_limit=cpu_limit,
                services_enabled=profile_config['services'],
                resource_limits=resource_limits,
                environment_variables=env_vars
            )
            
            logger.info(f"✅ Generated configuration:")
            logger.info(f"   Mode: {mode.value}")
            logger.info(f"   Agent Concurrency: {config.agent_concurrency}")
            logger.info(f"   Services: {len(config.services_enabled)}")
            logger.info(f"   Monitoring: {config.monitoring_enabled}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to generate configuration: {e}")
            raise
    
    def _generate_resource_limits(self, capabilities: SystemCapabilities, 
                                profile_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate Docker resource limits for services"""
        
        # Base resource allocation percentages
        resource_allocation = {
            'postgres': {'memory_pct': 25, 'cpu_pct': 20},
            'redis': {'memory_pct': 15, 'cpu_pct': 10},
            'temporal': {'memory_pct': 20, 'cpu_pct': 15},
            'neo4j': {'memory_pct': 20, 'cpu_pct': 15},
            'qdrant': {'memory_pct': 15, 'cpu_pct': 10},
            'api': {'memory_pct': 15, 'cpu_pct': 20},
            'worker': {'memory_pct': 20, 'cpu_pct': 25},
            'orchestrator': {'memory_pct': 15, 'cpu_pct': 20},
            'scanner-go': {'memory_pct': 10, 'cpu_pct': 10},
            'prometheus': {'memory_pct': 10, 'cpu_pct': 5},
            'grafana': {'memory_pct': 5, 'cpu_pct': 5}
        }
        
        total_memory_mb = capabilities.ram_total_gb * 1024
        total_cpu = capabilities.cpu_cores
        
        resource_limits = {}
        
        for service in profile_config['services']:
            if service in resource_allocation:
                alloc = resource_allocation[service]
                
                memory_limit = int(total_memory_mb * (alloc['memory_pct'] / 100))
                memory_reservation = int(memory_limit * 0.7)
                cpu_limit = round(total_cpu * (alloc['cpu_pct'] / 100), 1)
                cpu_reservation = round(cpu_limit * 0.5, 1)
                
                resource_limits[service] = {
                    'deploy': {
                        'resources': {
                            'limits': {
                                'memory': f"{memory_limit}M",
                                'cpus': str(cpu_limit)
                            },
                            'reservations': {
                                'memory': f"{memory_reservation}M",
                                'cpus': str(cpu_reservation)
                            }
                        }
                    }
                }
        
        return resource_limits
    
    def _generate_environment_variables(self, capabilities: SystemCapabilities,
                                      profile_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate environment variables for XORB configuration"""
        
        env_vars = {
            # Core XORB configuration
            'XORB_MODE': capabilities.profile.value,
            'XORB_AGENT_CONCURRENCY': str(profile_config['agent_concurrency']),
            'XORB_MAX_CONCURRENT_MISSIONS': str(profile_config['max_missions']),
            'XORB_WORKER_THREADS': str(profile_config['worker_threads']),
            'XORB_MONITORING_ENABLED': str(profile_config['monitoring']).lower(),
            'XORB_SYSTEM_PROFILE': capabilities.profile.value,
            'XORB_MEMORY_LIMIT_MB': str(int(capabilities.ram_available_gb * 1024 * 0.8)),
            'XORB_CPU_LIMIT': str(min(capabilities.cpu_cores, capabilities.cpu_threads * 0.8)),
            
            # System-specific optimizations
            'XORB_IS_ARM': str(capabilities.is_arm).lower(),
            'XORB_IS_VIRTUALIZED': str(capabilities.is_virtualized).lower(),
            'XORB_CPU_CORES': str(capabilities.cpu_cores),
            'XORB_CPU_THREADS': str(capabilities.cpu_threads),
            'XORB_RAM_GB': str(int(capabilities.ram_total_gb)),
            
            # Docker configuration
            'DOCKER_BUILDKIT': '1' if capabilities.docker_buildkit else '0',
            'COMPOSE_DOCKER_CLI_BUILD': '1' if capabilities.docker_buildkit else '0',
            
            # Performance tuning
            'XORB_ORCHESTRATION_CYCLE_TIME': self._get_cycle_time(capabilities.profile),
            'XORB_DATABASE_POOL_SIZE': self._get_db_pool_size(capabilities.profile),
            'XORB_REDIS_POOL_SIZE': self._get_redis_pool_size(capabilities.profile),
            
            # Feature flags based on profile
            'XORB_PHASE_11_ENABLED': str(capabilities.profile != SystemProfile.RPI).lower(),
            'XORB_PLUGIN_DISCOVERY_ENABLED': str(capabilities.profile not in [SystemProfile.RPI, SystemProfile.CLOUD_MICRO]).lower(),
            'XORB_PI5_OPTIMIZATION': str(capabilities.is_arm and capabilities.cpu_cores >= 4).lower()
        }
        
        return env_vars
    
    def _get_cycle_time(self, profile: SystemProfile) -> str:
        """Get orchestration cycle time based on profile"""
        cycle_times = {
            SystemProfile.RPI: '800',
            SystemProfile.CLOUD_MICRO: '600',
            SystemProfile.CLOUD_SMALL: '400',
            SystemProfile.CLOUD_MEDIUM: '300',
            SystemProfile.BARE_METAL: '200',
            SystemProfile.EPYC_SERVER: '100'
        }
        return cycle_times.get(profile, '400')
    
    def _get_db_pool_size(self, profile: SystemProfile) -> str:
        """Get database pool size based on profile"""
        pool_sizes = {
            SystemProfile.RPI: '5',
            SystemProfile.CLOUD_MICRO: '10',
            SystemProfile.CLOUD_SMALL: '15',
            SystemProfile.CLOUD_MEDIUM: '20',
            SystemProfile.BARE_METAL: '30',
            SystemProfile.EPYC_SERVER: '50'
        }
        return pool_sizes.get(profile, '20')
    
    def _get_redis_pool_size(self, profile: SystemProfile) -> str:
        """Get Redis pool size based on profile"""
        pool_sizes = {
            SystemProfile.RPI: '5',
            SystemProfile.CLOUD_MICRO: '10',
            SystemProfile.CLOUD_SMALL: '15',
            SystemProfile.CLOUD_MEDIUM: '20',
            SystemProfile.BARE_METAL: '25',
            SystemProfile.EPYC_SERVER: '40'
        }
        return pool_sizes.get(profile, '20')
    
    def write_environment_file(self, config: XorbConfiguration) -> Path:
        """Write .xorb.env file with generated configuration"""
        logger.info("📝 Writing .xorb.env configuration file...")
        
        try:
            env_file = self.project_root / '.xorb.env'
            
            # Add timestamp and generation info
            content = [
                "# XORB Auto-Generated Environment Configuration",
                f"# Generated: {datetime.now().isoformat()}",
                f"# System Profile: {config.system_profile.value}",
                f"# XORB Mode: {config.mode.value}",
                "#",
                "# This file was automatically generated by XORB Auto-Configurator",
                "# Modify with caution - regenerate with: python autoconfigure.py",
                "",
            ]
            
            # Add all environment variables
            for key, value in sorted(config.environment_variables.items()):
                content.append(f"{key}={value}")
            
            # Add standard database passwords
            content.extend([
                "",
                "# Database Configuration",
                "POSTGRES_USER=xorb",
                "POSTGRES_PASSWORD=xorb_secure_2024",
                "POSTGRES_DB=xorb",
                "NEO4J_PASSWORD=xorb_neo4j_2024",
                "GRAFANA_ADMIN_PASSWORD=xorb_admin_2024",
                "",
                "# API Keys (configure as needed)",
                "OPENROUTER_API_KEY=",
                "CEREBRAS_API_KEY=",
                "",
                "# Container Configuration",
                "COMPOSE_PROJECT_NAME=xorb",
            ])
            
            env_file.write_text('\n'.join(content))
            
            logger.info(f"✅ Environment file written: {env_file}")
            return env_file
            
        except Exception as e:
            logger.error(f"Failed to write environment file: {e}")
            raise
    
    def generate_docker_compose(self, config: XorbConfiguration) -> Path:
        """Generate optimized Docker Compose file"""
        logger.info("🐳 Generating optimized Docker Compose configuration...")
        
        try:
            # Load base template
            base_compose_file = self.project_root / 'docker-compose.unified.yml'
            if not base_compose_file.exists():
                # Create minimal template if none exists
                base_compose = self._create_base_compose_template()
            else:
                with open(base_compose_file, 'r') as f:
                    base_compose = yaml.safe_load(f)
            
            # Filter services based on configuration
            filtered_services = {}
            for service_name in config.services_enabled:
                if service_name in base_compose.get('services', {}):
                    service_config = base_compose['services'][service_name].copy()
                    
                    # Apply resource limits
                    if service_name in config.resource_limits:
                        service_config.update(config.resource_limits[service_name])
                    
                    # Add profile-specific labels
                    if 'labels' not in service_config:
                        service_config['labels'] = []
                    
                    service_config['labels'].extend([
                        f"xorb.profile={config.system_profile.value}",
                        f"xorb.mode={config.mode.value}",
                        f"xorb.auto_configured=true"
                    ])
                    
                    filtered_services[service_name] = service_config
            
            # Update compose structure
            optimized_compose = {
                'version': base_compose.get('version', '3.8'),
                'networks': base_compose.get('networks', {}),
                'volumes': base_compose.get('volumes', {}),
                'services': filtered_services
            }
            
            # Write optimized compose file
            output_file = self.project_root / f'docker-compose.{config.system_profile.value.lower()}.yml'
            
            with open(output_file, 'w') as f:
                yaml.dump(optimized_compose, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Docker Compose file generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate Docker Compose file: {e}")
            raise
    
    def _create_base_compose_template(self) -> Dict[str, Any]:
        """Create a minimal base compose template"""
        return {
            'version': '3.8',
            'networks': {
                'xorb-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'postgres_data': {'driver': 'local'},
                'redis_data': {'driver': 'local'}
            },
            'services': {
                'postgres': {
                    'image': 'ankane/pgvector:latest',
                    'restart': 'unless-stopped',
                    'environment': [
                        'POSTGRES_USER=${POSTGRES_USER:-xorb}',
                        'POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-xorb_secure_2024}',
                        'POSTGRES_DB=${POSTGRES_DB:-xorb}'
                    ],
                    'volumes': ['postgres_data:/var/lib/postgresql/data'],
                    'ports': ['5432:5432'],
                    'networks': ['xorb-network']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'restart': 'unless-stopped',
                    'volumes': ['redis_data:/data'],
                    'ports': ['6379:6379'],
                    'networks': ['xorb-network']
                },
                'api': {
                    'build': {'context': '.', 'dockerfile': 'Dockerfile.api'},
                    'restart': 'unless-stopped',
                    'ports': ['8000:8000'],
                    'networks': ['xorb-network'],
                    'depends_on': ['postgres', 'redis']
                }
            }
        }
    
    def generate_bootstrap_report(self, capabilities: SystemCapabilities, 
                                config: XorbConfiguration) -> Path:
        """Generate comprehensive bootstrap report"""
        logger.info("📊 Generating bootstrap report...")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'system_capabilities': asdict(capabilities),
                'generated_configuration': asdict(config),
                'optimization_summary': {
                    'profile_selected': config.system_profile.value,
                    'mode_selected': config.mode.value,
                    'services_enabled': len(config.services_enabled),
                    'monitoring_enabled': config.monitoring_enabled,
                    'resource_optimization': {
                        'memory_limit_mb': config.memory_limit_mb,
                        'cpu_limit': config.cpu_limit,
                        'agent_concurrency': config.agent_concurrency,
                        'max_missions': config.max_concurrent_missions
                    }
                },
                'deployment_readiness': {
                    'docker_available': capabilities.docker_version != 'unknown',
                    'buildkit_enabled': capabilities.docker_buildkit,
                    'compose_available': capabilities.docker_compose_version != 'unknown',
                    'sufficient_resources': capabilities.ram_available_gb >= 2.0,
                    'disk_space_available': capabilities.disk_space_gb >= 10.0
                },
                'recommendations': self._generate_recommendations(capabilities, config)
            }
            
            report_file = self.logs_dir / 'bootstrap_report.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Bootstrap report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to generate bootstrap report: {e}")
            raise
    
    def _generate_recommendations(self, capabilities: SystemCapabilities,
                                config: XorbConfiguration) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if capabilities.ram_available_gb < 4:
            recommendations.append("Consider increasing RAM for better performance")
        
        if capabilities.cpu_cores < 4:
            recommendations.append("Limited CPU cores may affect concurrent agent execution")
        
        if not capabilities.docker_buildkit:
            recommendations.append("Enable Docker BuildKit for faster builds (export DOCKER_BUILDKIT=1)")
        
        if capabilities.disk_space_gb < 20:
            recommendations.append("Low disk space - monitor storage usage closely")
        
        if capabilities.is_virtualized and config.system_profile in [SystemProfile.BARE_METAL, SystemProfile.EPYC_SERVER]:
            recommendations.append("Performance may be limited in virtualized environment")
        
        if capabilities.profile == SystemProfile.RPI:
            recommendations.append("ARM optimization enabled - some features disabled for stability")
        
        if not config.monitoring_enabled:
            recommendations.append("Monitoring disabled due to resource constraints")
        
        return recommendations
    
    def validate_configuration(self, config: XorbConfiguration) -> bool:
        """Validate generated configuration"""
        logger.info("✅ Validating configuration...")
        
        try:
            # Check basic requirements
            if config.agent_concurrency < 1:
                logger.error("Invalid agent concurrency")
                return False
            
            if config.memory_limit_mb < 512:
                logger.error("Insufficient memory allocation")
                return False
            
            if not config.services_enabled:
                logger.error("No services enabled")
                return False
            
            # Check required services
            required_services = ['postgres', 'api']
            for service in required_services:
                if service not in config.services_enabled:
                    logger.error(f"Required service missing: {service}")
                    return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def print_summary(self, capabilities: SystemCapabilities, config: XorbConfiguration):
        """Print deployment summary"""
        print("\n" + "="*80)
        print("🚀 XORB AUTO-CONFIGURATION COMPLETE")
        print("="*80)
        
        print(f"\n📊 SYSTEM PROFILE: {config.system_profile.value}")
        print(f"   CPU: {capabilities.cpu_cores} cores ({capabilities.cpu_threads} threads)")
        print(f"   RAM: {capabilities.ram_total_gb:.1f}GB total, {capabilities.ram_available_gb:.1f}GB available")
        print(f"   Architecture: {capabilities.architecture}")
        print(f"   OS: {capabilities.os_type} {capabilities.os_version}")
        
        print(f"\n⚙️ XORB CONFIGURATION:")
        print(f"   Mode: {config.mode.value}")
        print(f"   Agent Concurrency: {config.agent_concurrency}")
        print(f"   Max Concurrent Missions: {config.max_concurrent_missions}")
        print(f"   Worker Threads: {config.worker_threads}")
        print(f"   Monitoring: {'Enabled' if config.monitoring_enabled else 'Disabled'}")
        print(f"   Memory Limit: {config.memory_limit_mb}MB")
        print(f"   CPU Limit: {config.cpu_limit}")
        
        print(f"\n🔧 SERVICES ENABLED ({len(config.services_enabled)}):")
        for i, service in enumerate(config.services_enabled, 1):
            print(f"   {i:2d}. {service}")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   • .xorb.env")
        print(f"   • docker-compose.{config.system_profile.value.lower()}.yml")
        print(f"   • logs/bootstrap_report.json")
        
        if config.system_profile == SystemProfile.RPI:
            print(f"\n🍓 RASPBERRY PI OPTIMIZATIONS:")
            print(f"   • ARM-specific container images")
            print(f"   • Reduced resource allocation")
            print(f"   • Simplified service stack")
        
        print("\n" + "="*80)
    
    def run_interactive_setup(self) -> bool:
        """Run interactive setup process"""
        try:
            print("\n🤖 XORB Autonomous Self-Configuration System")
            print("Detecting and optimizing for your hardware environment...\n")
            
            # Detect system capabilities
            capabilities = self.detect_system_capabilities()
            
            # Generate configuration
            config = self.generate_configuration(capabilities)
            
            # Validate configuration
            if not self.validate_configuration(config):
                logger.error("Configuration validation failed")
                return False
            
            # Generate files
            env_file = self.write_environment_file(config)
            compose_file = self.generate_docker_compose(config)
            report_file = self.generate_bootstrap_report(capabilities, config)
            
            # Print summary
            self.print_summary(capabilities, config)
            
            # Prompt for deployment
            response = input("\n🚀 Ready to deploy XORB with optimized configuration? [Y/n]: ").strip().lower()
            
            if response in ['', 'y', 'yes']:
                print("\n✅ Configuration complete! Use the following command to deploy:")
                print(f"   docker compose -f {compose_file.name} --env-file .xorb.env up -d")
                return True
            else:
                print("\n⏸️ Configuration saved. Deploy when ready with:")
                print(f"   docker compose -f {compose_file.name} --env-file .xorb.env up -d")
                return False
                
        except Exception as e:
            logger.error(f"Interactive setup failed: {e}")
            return False


def main():
    """Main entry point"""
    try:
        configurator = XorbAutoConfigurator()
        success = configurator.run_interactive_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏸️ Configuration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Auto-configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()