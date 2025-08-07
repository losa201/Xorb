#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Unified Deployment Script
Complete automated deployment, configuration, and verification system
"""

import asyncio
import json
import subprocess
import sys
import time
import logging
import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import psutil
import concurrent.futures

# Import XORB resilience components
try:
    from xorb_resilience_load_balancer import XORBResilienceLoadBalancer, LoadBalancingStrategy
    from xorb_resilience_circuit_breaker import XORBFaultToleranceManager
    from xorb_resilience_data_replication import XORBDataReplicationManager, ConsistencyLevel
    from xorb_resilience_enhanced_monitoring import XORBEnhancedMonitoring
    from xorb_resilience_performance_optimizer import XORBPerformanceOptimizer
    from xorb_resilience_security_hardening import XORBSecurityHardeningManager
    from xorb_resilience_network_config import XORBNetworkConfig
    XORB_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import XORB components: {e}")
    XORB_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/xorb_deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    """Deployment phases"""
    INITIALIZATION = "initialization"
    DEPENDENCY_CHECK = "dependency_check"
    INFRASTRUCTURE_SETUP = "infrastructure_setup"
    SERVICE_DEPLOYMENT = "service_deployment"
    CONFIGURATION = "configuration"
    SECURITY_HARDENING = "security_hardening"
    MONITORING_SETUP = "monitoring_setup"
    NETWORK_CONFIGURATION = "network_configuration"
    HEALTH_VERIFICATION = "health_verification"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TESTING = "testing"
    COMPLETION = "completion"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ServiceType(Enum):
    """Types of services to deploy"""
    CORE_PLATFORM = "core_platform"
    MONITORING = "monitoring"
    DATABASE = "database"
    SECURITY = "security"
    NETWORKING = "networking"

@dataclass
class DeploymentStep:
    """Individual deployment step"""
    step_id: str
    name: str
    phase: DeploymentPhase
    service_type: ServiceType
    command: Optional[str] = None
    function: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_attempts: int = 3
    critical: bool = True
    status: DeploymentStatus = DeploymentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: str = ""
    error_message: str = ""

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    platform_version: str = "1.0.0"
    environment: str = "production"
    namespace: str = "xorb-platform"
    docker_registry: str = "localhost:5000"
    enable_monitoring: bool = True
    enable_security: bool = True
    enable_backup: bool = True
    enable_scaling: bool = True
    resource_allocation: Dict[str, Any] = field(default_factory=lambda: {
        "cpu_limit": "2000m",
        "memory_limit": "4Gi",
        "storage_size": "10Gi"
    })
    network_config: Dict[str, Any] = field(default_factory=lambda: {
        "internal_network": "10.42.0.0/16",
        "service_mesh": True,
        "load_balancer": True
    })

class XORBUnifiedDeployment:
    """Unified deployment orchestrator for XORB Platform"""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig(
            deployment_id=str(uuid.uuid4()),
            platform_version="1.0.0",
            environment="production"
        )
        
        # Deployment state
        self.deployment_steps: Dict[str, DeploymentStep] = {}
        self.deployed_services: Dict[str, Dict[str, Any]] = {}
        self.deployment_start_time: Optional[datetime] = None
        self.deployment_end_time: Optional[datetime] = None
        
        # Component managers
        self.load_balancer = None
        self.fault_tolerance_manager = None
        self.data_replication_manager = None
        self.monitoring_manager = None
        self.performance_optimizer = None
        self.security_hardening = None
        self.network_configurator = None
        
        # Deployment paths
        self.deployment_root = Path("/tmp/xorb_deployment")
        self.config_dir = self.deployment_root / "config"
        self.scripts_dir = self.deployment_root / "scripts"
        self.logs_dir = self.deployment_root / "logs"
        
        self._initialize_deployment_structure()
        self._define_deployment_steps()
        
        logger.info(f"Unified Deployment initialized: {self.config.deployment_id}")
    
    def _initialize_deployment_structure(self):
        """Initialize deployment directory structure"""
        try:
            # Create deployment directories
            for directory in [self.deployment_root, self.config_dir, self.scripts_dir, self.logs_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create deployment manifest
            manifest = {
                'deployment_id': self.config.deployment_id,
                'created_at': datetime.now().isoformat(),
                'platform_version': self.config.platform_version,
                'environment': self.config.environment
            }
            
            with open(self.deployment_root / "deployment_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info("Deployment structure initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize deployment structure: {e}")
            raise e
    
    def _define_deployment_steps(self):
        """Define all deployment steps"""
        try:
            steps = [
                # Phase 1: Initialization
                DeploymentStep(
                    step_id="init_environment",
                    name="Initialize Environment",
                    phase=DeploymentPhase.INITIALIZATION,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="initialize_environment",
                    critical=True
                ),
                
                # Phase 2: Dependency Check
                DeploymentStep(
                    step_id="check_dependencies",
                    name="Check System Dependencies",
                    phase=DeploymentPhase.DEPENDENCY_CHECK,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="check_system_dependencies",
                    critical=True
                ),
                DeploymentStep(
                    step_id="check_resources",
                    name="Check System Resources",
                    phase=DeploymentPhase.DEPENDENCY_CHECK,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="check_system_resources",
                    critical=True
                ),
                
                # Phase 3: Infrastructure Setup
                DeploymentStep(
                    step_id="setup_docker_environment",
                    name="Setup Docker Environment",
                    phase=DeploymentPhase.INFRASTRUCTURE_SETUP,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="setup_docker_environment",
                    dependencies=["check_dependencies"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="create_docker_network",
                    name="Create Docker Networks",
                    phase=DeploymentPhase.INFRASTRUCTURE_SETUP,
                    service_type=ServiceType.NETWORKING,
                    function="create_docker_networks",
                    dependencies=["setup_docker_environment"],
                    critical=True
                ),
                
                # Phase 4: Database Services
                DeploymentStep(
                    step_id="deploy_postgresql",
                    name="Deploy PostgreSQL Database",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.DATABASE,
                    function="deploy_postgresql",
                    dependencies=["create_docker_network"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="deploy_redis",
                    name="Deploy Redis Cache",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.DATABASE,
                    function="deploy_redis",
                    dependencies=["create_docker_network"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="deploy_neo4j",
                    name="Deploy Neo4j Graph Database",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.DATABASE,
                    function="deploy_neo4j",
                    dependencies=["create_docker_network"],
                    critical=False
                ),
                
                # Phase 5: Monitoring Infrastructure
                DeploymentStep(
                    step_id="deploy_prometheus",
                    name="Deploy Prometheus Monitoring",
                    phase=DeploymentPhase.MONITORING_SETUP,
                    service_type=ServiceType.MONITORING,
                    function="deploy_prometheus",
                    dependencies=["deploy_postgresql"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="deploy_grafana",
                    name="Deploy Grafana Dashboard",
                    phase=DeploymentPhase.MONITORING_SETUP,
                    service_type=ServiceType.MONITORING,
                    function="deploy_grafana",
                    dependencies=["deploy_prometheus"],
                    critical=True
                ),
                
                # Phase 6: Core XORB Services
                DeploymentStep(
                    step_id="deploy_orchestrator",
                    name="Deploy Neural Orchestrator",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="deploy_neural_orchestrator",
                    dependencies=["deploy_postgresql", "deploy_redis"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="deploy_learning_service",
                    name="Deploy Learning Service",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="deploy_learning_service",
                    dependencies=["deploy_orchestrator"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="deploy_threat_detection",
                    name="Deploy Threat Detection",
                    phase=DeploymentPhase.SERVICE_DEPLOYMENT,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="deploy_threat_detection",
                    dependencies=["deploy_orchestrator"],
                    critical=True
                ),
                
                # Phase 7: Configuration
                DeploymentStep(
                    step_id="configure_load_balancer",
                    name="Configure Load Balancer",
                    phase=DeploymentPhase.CONFIGURATION,
                    service_type=ServiceType.NETWORKING,
                    function="configure_load_balancer",
                    dependencies=["deploy_orchestrator", "deploy_learning_service"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="configure_fault_tolerance",
                    name="Configure Fault Tolerance",
                    phase=DeploymentPhase.CONFIGURATION,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="configure_fault_tolerance",
                    dependencies=["deploy_orchestrator"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="configure_data_replication",
                    name="Configure Data Replication",
                    phase=DeploymentPhase.CONFIGURATION,
                    service_type=ServiceType.DATABASE,
                    function="configure_data_replication",
                    dependencies=["deploy_postgresql"],
                    critical=True
                ),
                
                # Phase 8: Security Hardening
                DeploymentStep(
                    step_id="setup_security_hardening",
                    name="Setup Security Hardening",
                    phase=DeploymentPhase.SECURITY_HARDENING,
                    service_type=ServiceType.SECURITY,
                    function="setup_security_hardening",
                    dependencies=["deploy_orchestrator"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="configure_network_policies",
                    name="Configure Network Policies",
                    phase=DeploymentPhase.NETWORK_CONFIGURATION,
                    service_type=ServiceType.NETWORKING,
                    function="configure_network_policies",
                    dependencies=["setup_security_hardening"],
                    critical=True
                ),
                
                # Phase 9: Performance Optimization
                DeploymentStep(
                    step_id="setup_performance_optimization",
                    name="Setup Performance Optimization",
                    phase=DeploymentPhase.PERFORMANCE_OPTIMIZATION,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="setup_performance_optimization",
                    dependencies=["configure_load_balancer"],
                    critical=False
                ),
                
                # Phase 10: Health Verification
                DeploymentStep(
                    step_id="verify_service_health",
                    name="Verify Service Health",
                    phase=DeploymentPhase.HEALTH_VERIFICATION,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="verify_all_services_health",
                    dependencies=["configure_network_policies", "setup_performance_optimization"],
                    critical=True
                ),
                DeploymentStep(
                    step_id="run_integration_tests",
                    name="Run Integration Tests",
                    phase=DeploymentPhase.TESTING,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="run_integration_tests",
                    dependencies=["verify_service_health"],
                    critical=False
                ),
                
                # Phase 11: Completion
                DeploymentStep(
                    step_id="generate_deployment_report",
                    name="Generate Deployment Report",
                    phase=DeploymentPhase.COMPLETION,
                    service_type=ServiceType.CORE_PLATFORM,
                    function="generate_deployment_report",
                    dependencies=["verify_service_health"],
                    critical=True
                )
            ]
            
            # Store steps
            for step in steps:
                self.deployment_steps[step.step_id] = step
            
            logger.info(f"Defined {len(steps)} deployment steps")
            
        except Exception as e:
            logger.error(f"Failed to define deployment steps: {e}")
            raise e
    
    async def execute_deployment(self) -> bool:
        """Execute the complete deployment process"""
        try:
            self.deployment_start_time = datetime.now()
            logger.info(f"Starting XORB Platform deployment: {self.config.deployment_id}")
            
            # Execute deployment phases in order
            phases = list(DeploymentPhase)
            
            for phase in phases:
                logger.info(f"Executing deployment phase: {phase.value}")
                
                # Get steps for this phase
                phase_steps = [step for step in self.deployment_steps.values() if step.phase == phase]
                
                if not phase_steps:
                    continue
                
                # Execute steps in dependency order
                success = await self._execute_phase_steps(phase_steps)
                
                if not success:
                    # Check if any critical steps failed
                    failed_critical_steps = [
                        step for step in phase_steps 
                        if step.status == DeploymentStatus.FAILED and step.critical
                    ]
                    
                    if failed_critical_steps:
                        logger.error(f"Critical steps failed in phase {phase.value}")
                        return False
                    else:
                        logger.warning(f"Non-critical steps failed in phase {phase.value}, continuing")
            
            self.deployment_end_time = datetime.now()
            deployment_duration = (self.deployment_end_time - self.deployment_start_time).total_seconds()
            
            # Check overall deployment success
            failed_critical_steps = [
                step for step in self.deployment_steps.values() 
                if step.status == DeploymentStatus.FAILED and step.critical
            ]
            
            if failed_critical_steps:
                logger.error(f"Deployment failed - {len(failed_critical_steps)} critical steps failed")
                return False
            
            logger.info(f"XORB Platform deployment completed successfully in {deployment_duration:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Deployment execution failed: {e}")
            return False

    def _get_all_completed_steps(self) -> set:
        """Get all completed step IDs across all phases"""
        return {step_id for step_id, step in self.deployment_steps.items() 
                if step.status == DeploymentStatus.COMPLETED}
    
    async def _execute_phase_steps(self, steps: List[DeploymentStep]) -> bool:
        """Execute steps in a phase, respecting dependencies"""
        try:
            # Build dependency graph
            remaining_steps = {step.step_id: step for step in steps}
            completed_steps = self._get_all_completed_steps()
            
            while remaining_steps:
                # Find steps with satisfied dependencies
                ready_steps = []
                
                for step_id, step in remaining_steps.items():
                    dependencies_satisfied = all(dep in completed_steps for dep in step.dependencies)
                    if dependencies_satisfied:
                        ready_steps.append(step)
                
                if not ready_steps:
                    # Check if there are steps without dependencies that can run
                    no_deps_steps = [step for step in remaining_steps.values() if not step.dependencies]
                    if no_deps_steps:
                        ready_steps = no_deps_steps
                    else:
                        logger.warning(f"Skipping steps with unsatisfied dependencies: {list(remaining_steps.keys())}")
                        # Mark remaining steps as skipped
                        for step in remaining_steps.values():
                            step.status = DeploymentStatus.SKIPPED
                        break
                
                # Execute ready steps in parallel (if safe)
                execution_tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(self._execute_deployment_step(step))
                    execution_tasks.append((step.step_id, task))
                
                # Wait for all tasks to complete
                for step_id, task in execution_tasks:
                    try:
                        await task
                        completed_steps.add(step_id)
                        del remaining_steps[step_id]
                    except Exception as e:
                        logger.error(f"Step {step_id} failed: {e}")
                        # Mark as failed but continue with non-critical steps
                        step = self.deployment_steps[step_id]
                        step.status = DeploymentStatus.FAILED
                        step.error_message = str(e)
                        
                        if step.critical:
                            logger.error(f"Critical step {step_id} failed, deployment may fail")
                        
                        completed_steps.add(step_id)
                        del remaining_steps[step_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Phase execution failed: {e}")
            return False
    
    async def _execute_deployment_step(self, step: DeploymentStep):
        """Execute a single deployment step"""
        try:
            step.start_time = datetime.now()
            step.status = DeploymentStatus.IN_PROGRESS
            
            logger.info(f"Executing step: {step.name}")
            
            # Execute based on type
            if step.function:
                # Execute Python function
                func = getattr(self, step.function, None)
                if func:
                    result = await func()
                    if not result:
                        raise Exception(f"Function {step.function} returned False")
                else:
                    raise Exception(f"Function {step.function} not found")
            
            elif step.command:
                # Execute shell command
                result = await self._execute_shell_command(step.command, step.timeout_seconds)
                step.output = result.get('output', '')
                if result['return_code'] != 0:
                    raise Exception(f"Command failed with code {result['return_code']}: {result.get('error', '')}")
            
            else:
                raise Exception("No function or command specified for step")
            
            # Mark as completed
            step.end_time = datetime.now()
            step.duration_seconds = (step.end_time - step.start_time).total_seconds()
            step.status = DeploymentStatus.COMPLETED
            
            logger.info(f"Step completed: {step.name} ({step.duration_seconds:.2f}s)")
            
        except Exception as e:
            step.end_time = datetime.now()
            step.duration_seconds = (step.end_time - step.start_time).total_seconds() if step.start_time else 0
            step.status = DeploymentStatus.FAILED
            step.error_message = str(e)
            
            logger.error(f"Step failed: {step.name} - {str(e)}")
            
            # Retry if configured
            if step.retry_attempts > 0:
                logger.info(f"Retrying step: {step.name} (attempts left: {step.retry_attempts})")
                step.retry_attempts -= 1
                await asyncio.sleep(5)  # Wait before retry
                await self._execute_deployment_step(step)
            else:
                raise e
    
    async def _execute_shell_command(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute shell command with timeout"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                'return_code': process.returncode,
                'output': stdout.decode() if stdout else '',
                'error': stderr.decode() if stderr else ''
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {command}")
            return {
                'return_code': -1,
                'output': '',
                'error': f'Command timed out after {timeout} seconds'
            }
        except Exception as e:
            logger.error(f"Command execution failed: {command} - {e}")
            return {
                'return_code': -1,
                'output': '',
                'error': str(e)
            }
    
    # Deployment step implementations
    async def initialize_environment(self) -> bool:
        """Initialize deployment environment"""
        try:
            logger.info("Initializing deployment environment")
            
            # Set environment variables
            os.environ['XORB_DEPLOYMENT_ID'] = self.config.deployment_id
            os.environ['XORB_ENVIRONMENT'] = self.config.environment
            os.environ['XORB_VERSION'] = self.config.platform_version
            
            # Create necessary directories
            for directory in ['/tmp/xorb_data', '/tmp/xorb_logs', '/tmp/xorb_config']:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("Environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            return False
    
    async def check_system_dependencies(self) -> bool:
        """Check required system dependencies"""
        try:
            logger.info("Checking system dependencies")
            
            required_commands = ['docker', 'docker-compose', 'python3', 'pip3']
            missing_commands = []
            
            for command in required_commands:
                result = await self._execute_shell_command(f"which {command}")
                if result['return_code'] != 0:
                    missing_commands.append(command)
            
            if missing_commands:
                logger.error(f"Missing required commands: {missing_commands}")
                return False
            
            # Check Python packages
            required_packages = ['psutil', 'asyncio', 'aiohttp', 'requests']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.warning(f"Optional package not available: {package}")
            
            logger.info("System dependencies check completed")
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    async def check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            logger.info("Checking system resources")
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            if cpu_cores < 2:
                logger.warning(f"Low CPU cores: {cpu_cores} (recommended: 4+)")
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                logger.warning(f"Low memory: {memory_gb:.1f}GB (recommended: 8GB+)")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            if disk_gb < 10:
                logger.warning(f"Low disk space: {disk_gb:.1f}GB (recommended: 50GB+)")
            
            logger.info(f"System resources: {cpu_cores} CPUs, {memory_gb:.1f}GB RAM, {disk_gb:.1f}GB free")
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False
    
    async def setup_docker_environment(self) -> bool:
        """Setup Docker environment"""
        try:
            logger.info("Setting up Docker environment")
            
            # Check if Docker is running
            result = await self._execute_shell_command("docker info")
            if result['return_code'] != 0:
                logger.error("Docker is not running")
                return False
            
            # Pull required images
            images = [
                'postgres:13',
                'redis:7-alpine',
                'neo4j:4.4',
                'prom/prometheus:latest',
                'grafana/grafana:latest'
            ]
            
            for image in images:
                logger.info(f"Pulling Docker image: {image}")
                result = await self._execute_shell_command(f"docker pull {image}")
                if result['return_code'] != 0:
                    logger.warning(f"Failed to pull image: {image}")
            
            logger.info("Docker environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Docker setup failed: {e}")
            return False
    
    async def create_docker_networks(self) -> bool:
        """Create Docker networks"""
        try:
            logger.info("Creating Docker networks")
            
            networks = [
                ('xorb-internal', '192.168.100.0/24'),
                ('xorb-management', '192.168.101.0/24'),
                ('xorb-public', '192.168.102.0/24')
            ]
            
            for network_name, subnet in networks:
                # Check if network exists
                result = await self._execute_shell_command(f"docker network ls --filter name={network_name} --format '{{{{.Name}}}}'")
                
                if network_name not in result['output']:
                    # Try to create network with subnet, fall back to default if subnet conflicts
                    create_cmd = f"docker network create --driver bridge --subnet {subnet} {network_name}"
                    result = await self._execute_shell_command(create_cmd)
                    
                    if result['return_code'] == 0:
                        logger.info(f"Created network: {network_name} with subnet {subnet}")
                    else:
                        # Try without subnet if there's a conflict
                        logger.warning(f"Failed to create network with subnet, trying without: {network_name}")
                        fallback_cmd = f"docker network create --driver bridge {network_name}"
                        fallback_result = await self._execute_shell_command(fallback_cmd)
                        
                        if fallback_result['return_code'] == 0:
                            logger.info(f"Created network: {network_name} (default subnet)")
                        else:
                            logger.error(f"Failed to create network: {network_name} - {fallback_result['error']}")
                            # Don't fail deployment for network issues, continue with existing networks
                            logger.warning(f"Continuing deployment with existing networks")
                else:
                    logger.info(f"Network already exists: {network_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Network creation failed: {e}")
            return False
    
    async def deploy_postgresql(self) -> bool:
        """Deploy PostgreSQL database"""
        try:
            logger.info("Deploying PostgreSQL database")
            
            # Create PostgreSQL configuration
            pg_config = {
                'container_name': 'xorb-postgresql',
                'image': 'postgres:13',
                'environment': {
                    'POSTGRES_DB': 'xorb_platform',
                    'POSTGRES_USER': 'xorb_user',
                    'POSTGRES_PASSWORD': 'xorb_secure_password',
                    'POSTGRES_INITDB_ARGS': '--auth-host=scram-sha-256'
                },
                'ports': ['5432:5432'],
                'volumes': ['/tmp/xorb_data/postgresql:/var/lib/postgresql/data'],
                'networks': ['xorb-internal']
            }
            
            # Stop existing container if running
            await self._execute_shell_command(f"docker stop {pg_config['container_name']} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {pg_config['container_name']} 2>/dev/null || true")
            
            # Build docker run command
            docker_cmd = f"docker run -d --name {pg_config['container_name']}"
            
            for key, value in pg_config['environment'].items():
                docker_cmd += f" -e {key}={value}"
            
            for port_mapping in pg_config['ports']:
                docker_cmd += f" -p {port_mapping}"
            
            for volume_mapping in pg_config['volumes']:
                docker_cmd += f" -v {volume_mapping}"
            
            for network in pg_config['networks']:
                docker_cmd += f" --network {network}"
            
            docker_cmd += f" {pg_config['image']}"
            
            # Execute deployment
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                # Wait for database to be ready
                await asyncio.sleep(10)
                logger.info("PostgreSQL deployed successfully")
                
                self.deployed_services['postgresql'] = {
                    'container_name': pg_config['container_name'],
                    'port': 5432,
                    'database': 'xorb_platform',
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"PostgreSQL deployment failed: {result['error']}")
                return False
            
        except Exception as e:
            logger.error(f"PostgreSQL deployment failed: {e}")
            return False
    
    async def deploy_redis(self) -> bool:
        """Deploy Redis cache"""
        try:
            logger.info("Deploying Redis cache")
            
            container_name = 'xorb-redis'
            
            # Stop existing container
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Deploy Redis (try custom network first, fall back to default)
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 6379:6379 \
                -v /tmp/xorb_data/redis:/data \
                redis:7-alpine redis-server --appendonly yes"""
            
            # Try with custom network first
            network_cmd = docker_cmd.replace("redis:7-alpine", "--network xorb-internal redis:7-alpine")
            result = await self._execute_shell_command(network_cmd)
            
            if result['return_code'] != 0:
                # Fall back to default network
                logger.warning("Custom network failed, using default network for Redis")
                result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(5)
                logger.info("Redis deployed successfully")
                
                self.deployed_services['redis'] = {
                    'container_name': container_name,
                    'port': 6379,
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Redis deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Redis deployment failed: {e}")
            return False
    
    async def deploy_neo4j(self) -> bool:
        """Deploy Neo4j graph database"""
        try:
            logger.info("Deploying Neo4j graph database")
            
            container_name = 'xorb-neo4j'
            
            # Stop existing container
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Deploy Neo4j
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 7474:7474 -p 7687:7687 \
                --network xorb-internal \
                -v /tmp/xorb_data/neo4j:/data \
                -e NEO4J_AUTH=neo4j/xorb_secure_password \
                neo4j:4.4"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(15)  # Neo4j takes longer to start
                logger.info("Neo4j deployed successfully")
                
                self.deployed_services['neo4j'] = {
                    'container_name': container_name,
                    'http_port': 7474,
                    'bolt_port': 7687,
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Neo4j deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Neo4j deployment failed: {e}")
            return False
    
    async def deploy_prometheus(self) -> bool:
        """Deploy Prometheus monitoring"""
        try:
            logger.info("Deploying Prometheus monitoring")
            
            # Create Prometheus configuration
            prometheus_config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'scrape_configs': [
                    {
                        'job_name': 'prometheus',
                        'static_configs': [{'targets': ['localhost:9090']}]
                    },
                    {
                        'job_name': 'xorb-services',
                        'static_configs': [{'targets': [
                            'localhost:8003',  # neural_orchestrator
                            'localhost:8004',  # learning_service
                            'localhost:8005',  # threat_detection
                        ]}]
                    }
                ]
            }
            
            # Write configuration file
            config_path = self.config_dir / 'prometheus.yml'
            with open(config_path, 'w') as f:
                yaml.dump(prometheus_config, f)
            
            container_name = 'xorb-prometheus'
            
            # Stop existing container
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Deploy Prometheus
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 9090:9090 \
                --network xorb-management \
                -v {config_path}:/etc/prometheus/prometheus.yml \
                -v /tmp/xorb_data/prometheus:/prometheus \
                prom/prometheus:latest"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(10)
                logger.info("Prometheus deployed successfully")
                
                self.deployed_services['prometheus'] = {
                    'container_name': container_name,
                    'port': 9090,
                    'config_path': str(config_path),
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Prometheus deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Prometheus deployment failed: {e}")
            return False
    
    async def deploy_grafana(self) -> bool:
        """Deploy Grafana dashboard"""
        try:
            logger.info("Deploying Grafana dashboard")
            
            container_name = 'xorb-grafana'
            
            # Stop existing container
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Deploy Grafana
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 3000:3000 \
                --network xorb-management \
                -v /tmp/xorb_data/grafana:/var/lib/grafana \
                -e GF_SECURITY_ADMIN_PASSWORD=xorb_admin_password \
                grafana/grafana:latest"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(15)
                logger.info("Grafana deployed successfully")
                
                self.deployed_services['grafana'] = {
                    'container_name': container_name,
                    'port': 3000,
                    'admin_password': 'xorb_admin_password',
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Grafana deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Grafana deployment failed: {e}")
            return False
    
    async def deploy_neural_orchestrator(self) -> bool:
        """Deploy Neural Orchestrator service"""
        try:
            logger.info("Deploying Neural Orchestrator service")
            
            # Create a simple orchestrator service script
            service_script = """#!/usr/bin/env python3
import asyncio
import logging
from aiohttp import web
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def health_check(request):
    return web.json_response({
        'status': 'healthy',
        'service': 'neural_orchestrator',
        'version': '1.0.0'
    })

async def orchestrate(request):
    return web.json_response({
        'status': 'success',
        'message': 'Neural orchestration active',
        'timestamp': str(datetime.now())
    })

async def init_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_post('/orchestrate', orchestrate)
    return app

if __name__ == '__main__':
    import datetime
    app = init_app()
    web.run_app(app, host='0.0.0.0', port=8003)
"""
            
            # Write service script
            script_path = self.scripts_dir / 'neural_orchestrator.py'
            with open(script_path, 'w') as f:
                f.write(service_script)
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            # Create Dockerfile
            dockerfile_content = f"""FROM python:3.9-slim
WORKDIR /app
RUN pip install aiohttp
COPY neural_orchestrator.py .
EXPOSE 8003
CMD ["python", "neural_orchestrator.py"]
"""
            
            dockerfile_path = self.scripts_dir / 'Dockerfile.orchestrator'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build Docker image
            build_cmd = f"cd {self.scripts_dir} && docker build -f Dockerfile.orchestrator -t xorb-neural-orchestrator ."
            result = await self._execute_shell_command(build_cmd)
            
            if result['return_code'] != 0:
                logger.error(f"Failed to build orchestrator image: {result['error']}")
                return False
            
            container_name = 'xorb-neural-orchestrator'
            
            # Stop existing container
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            # Deploy service
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 8003:8003 \
                --network xorb-internal \
                xorb-neural-orchestrator"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(5)
                logger.info("Neural Orchestrator deployed successfully")
                
                self.deployed_services['neural_orchestrator'] = {
                    'container_name': container_name,
                    'port': 8003,
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Neural Orchestrator deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Neural Orchestrator deployment failed: {e}")
            return False
    
    async def deploy_learning_service(self) -> bool:
        """Deploy Learning Service"""
        try:
            logger.info("Deploying Learning Service")
            
            # Similar implementation to neural orchestrator but on different port
            service_script = """#!/usr/bin/env python3
import asyncio
import logging
from aiohttp import web
import json
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def health_check(request):
    return web.json_response({
        'status': 'healthy',
        'service': 'learning_service',
        'version': '1.0.0'
    })

async def learn(request):
    return web.json_response({
        'status': 'success',
        'message': 'Learning process active',
        'timestamp': str(datetime.datetime.now())
    })

async def init_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_post('/learn', learn)
    return app

if __name__ == '__main__':
    app = init_app()
    web.run_app(app, host='0.0.0.0', port=8004)
"""
            
            script_path = self.scripts_dir / 'learning_service.py'
            with open(script_path, 'w') as f:
                f.write(service_script)
            
            os.chmod(script_path, 0o755)
            
            # Create Dockerfile
            dockerfile_content = f"""FROM python:3.9-slim
WORKDIR /app
RUN pip install aiohttp
COPY learning_service.py .
EXPOSE 8004
CMD ["python", "learning_service.py"]
"""
            
            dockerfile_path = self.scripts_dir / 'Dockerfile.learning'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build and deploy
            build_cmd = f"cd {self.scripts_dir} && docker build -f Dockerfile.learning -t xorb-learning-service ."
            result = await self._execute_shell_command(build_cmd)
            
            if result['return_code'] != 0:
                logger.error(f"Failed to build learning service image: {result['error']}")
                return False
            
            container_name = 'xorb-learning-service'
            
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 8004:8004 \
                --network xorb-internal \
                xorb-learning-service"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(5)
                logger.info("Learning Service deployed successfully")
                
                self.deployed_services['learning_service'] = {
                    'container_name': container_name,
                    'port': 8004,
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Learning Service deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Learning Service deployment failed: {e}")
            return False
    
    async def deploy_threat_detection(self) -> bool:
        """Deploy Threat Detection service"""
        try:
            logger.info("Deploying Threat Detection service")
            
            # Similar to other services but focused on threat detection
            service_script = """#!/usr/bin/env python3
import asyncio
import logging
from aiohttp import web
import json
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def health_check(request):
    return web.json_response({
        'status': 'healthy',
        'service': 'threat_detection',
        'version': '1.0.0'
    })

async def detect_threats(request):
    return web.json_response({
        'status': 'success',
        'message': 'Threat detection active',
        'threats_detected': 0,
        'timestamp': str(datetime.datetime.now())
    })

async def init_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_post('/detect', detect_threats)
    return app

if __name__ == '__main__':
    app = init_app()
    web.run_app(app, host='0.0.0.0', port=8005)
"""
            
            script_path = self.scripts_dir / 'threat_detection.py'
            with open(script_path, 'w') as f:
                f.write(service_script)
            
            os.chmod(script_path, 0o755)
            
            dockerfile_content = f"""FROM python:3.9-slim
WORKDIR /app
RUN pip install aiohttp
COPY threat_detection.py .
EXPOSE 8005
CMD ["python", "threat_detection.py"]
"""
            
            dockerfile_path = self.scripts_dir / 'Dockerfile.threat'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            build_cmd = f"cd {self.scripts_dir} && docker build -f Dockerfile.threat -t xorb-threat-detection ."
            result = await self._execute_shell_command(build_cmd)
            
            if result['return_code'] != 0:
                logger.error(f"Failed to build threat detection image: {result['error']}")
                return False
            
            container_name = 'xorb-threat-detection'
            
            await self._execute_shell_command(f"docker stop {container_name} 2>/dev/null || true")
            await self._execute_shell_command(f"docker rm {container_name} 2>/dev/null || true")
            
            docker_cmd = f"""docker run -d --name {container_name} \
                -p 8005:8005 \
                --network xorb-internal \
                xorb-threat-detection"""
            
            result = await self._execute_shell_command(docker_cmd)
            
            if result['return_code'] == 0:
                await asyncio.sleep(5)
                logger.info("Threat Detection deployed successfully")
                
                self.deployed_services['threat_detection'] = {
                    'container_name': container_name,
                    'port': 8005,
                    'status': 'running'
                }
                return True
            else:
                logger.error(f"Threat Detection deployment failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Threat Detection deployment failed: {e}")
            return False
    
    async def configure_load_balancer(self) -> bool:
        """Configure load balancer"""
        try:
            logger.info("Configuring load balancer")
            
            # Initialize load balancer
            try:
                self.load_balancer = XORBResilienceLoadBalancer()
                
                # Register services
                services = [
                    ('neural_orchestrator', 'localhost', 8003),
                    ('learning_service', 'localhost', 8004),
                    ('threat_detection', 'localhost', 8005)
                ]
                
                for service_name, host, port in services:
                    await self.load_balancer.register_service(service_name, host, port)
                
                logger.info("Load balancer configured successfully")
                return True
                
            except NameError:
                logger.warning("Load balancer component not available, creating basic configuration")
                
                # Create basic nginx configuration as fallback
                nginx_config = """upstream xorb_backend {
    server localhost:8003;
    server localhost:8004;
    server localhost:8005;
}

server {
    listen 8000;
    location / {
        proxy_pass http://xorb_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
"""
                
                config_path = self.config_dir / 'nginx.conf'
                with open(config_path, 'w') as f:
                    f.write(nginx_config)
                
                logger.info("Basic load balancer configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Load balancer configuration failed: {e}")
            return False
    
    async def configure_fault_tolerance(self) -> bool:
        """Configure fault tolerance"""
        try:
            logger.info("Configuring fault tolerance")
            
            try:
                # Initialize fault tolerance manager
                self.fault_tolerance_manager = XORBFaultToleranceManager()
                
                # Create circuit breakers for services
                services = ['neural_orchestrator', 'learning_service', 'threat_detection']
                
                for service in services:
                    self.fault_tolerance_manager.create_circuit_breaker(service)
                    self.fault_tolerance_manager.create_retry_mechanism(service)
                    self.fault_tolerance_manager.create_bulkhead(f"{service}_bulkhead")
                
                logger.info("Fault tolerance configured successfully")
                return True
                
            except NameError:
                logger.warning("Fault tolerance components not available, creating basic configuration")
                
                # Create basic health check configuration
                health_config = {
                    'services': [
                        {'name': 'neural_orchestrator', 'url': 'http://localhost:8003/health'},
                        {'name': 'learning_service', 'url': 'http://localhost:8004/health'},
                        {'name': 'threat_detection', 'url': 'http://localhost:8005/health'}
                    ],
                    'check_interval': 30,
                    'timeout': 10
                }
                
                config_path = self.config_dir / 'health_check.json'
                with open(config_path, 'w') as f:
                    json.dump(health_config, f, indent=2)
                
                logger.info("Basic fault tolerance configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Fault tolerance configuration failed: {e}")
            return False
    
    async def configure_data_replication(self) -> bool:
        """Configure data replication"""
        try:
            logger.info("Configuring data replication")
            
            try:
                # Initialize data replication manager
                self.data_replication_manager = XORBDataReplicationManager()
                
                # Setup replication nodes
                await self.data_replication_manager.add_replication_node(
                    "primary", "localhost", 5432, "primary"
                )
                
                logger.info("Data replication configured successfully")
                return True
                
            except NameError:
                logger.warning("Data replication components not available, creating basic configuration")
                
                # Create basic backup configuration
                backup_config = {
                    'database': {
                        'host': 'localhost',
                        'port': 5432,
                        'database': 'xorb_platform',
                        'user': 'xorb_user'
                    },
                    'backup_schedule': '0 2 * * *',  # Daily at 2 AM
                    'backup_path': '/tmp/xorb_data/backups',
                    'retention_days': 7
                }
                
                config_path = self.config_dir / 'backup.json'
                with open(config_path, 'w') as f:
                    json.dump(backup_config, f, indent=2)
                
                logger.info("Basic data replication configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Data replication configuration failed: {e}")
            return False
    
    async def setup_security_hardening(self) -> bool:
        """Setup security hardening"""
        try:
            logger.info("Setting up security hardening")
            
            try:
                # Initialize security hardening
                self.security_hardening = XORBSecurityHardeningManager()
                
                # Generate certificates
                if hasattr(self.security_hardening, 'setup_certificate_authority'):
                    await self.security_hardening.setup_certificate_authority()
                
                logger.info("Security hardening setup successfully")
                return True
                
            except (NameError, AttributeError):
                logger.warning("Security hardening components not available, creating basic configuration")
                
                # Create basic security configuration
                security_config = {
                    'authentication': {
                        'enabled': True,
                        'method': 'jwt',
                        'secret_key': 'xorb_security_secret_key'
                    },
                    'authorization': {
                        'enabled': True,
                        'default_role': 'user'
                    },
                    'encryption': {
                        'enabled': True,
                        'algorithm': 'AES-256-GCM'
                    }
                }
                
                config_path = self.config_dir / 'security.json'
                with open(config_path, 'w') as f:
                    json.dump(security_config, f, indent=2)
                
                logger.info("Basic security configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Security hardening setup failed: {e}")
            return False
    
    async def configure_network_policies(self) -> bool:
        """Configure network policies"""
        try:
            logger.info("Configuring network policies")
            
            try:
                # Initialize network configurator
                self.network_configurator = XORBNetworkConfig()
                
                # Configure firewall rules
                if hasattr(self.network_configurator, 'apply_firewall_rules'):
                    await self.network_configurator.apply_firewall_rules()
                
                # Configure network policies
                if hasattr(self.network_configurator, 'apply_k8s_network_policies'):
                    await self.network_configurator.apply_k8s_network_policies()
                
                logger.info("Network policies configured successfully")
                return True
                
            except (NameError, AttributeError):
                logger.warning("Network configuration components not available, creating basic configuration")
                
                # Create basic network configuration
                network_config = {
                    'allowed_ports': [8003, 8004, 8005, 9090, 3000, 5432, 6379],
                    'internal_networks': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'],
                    'firewall_enabled': True,
                    'rate_limiting': {
                        'enabled': True,
                        'requests_per_minute': 1000
                    }
                }
                
                config_path = self.config_dir / 'network.json'
                with open(config_path, 'w') as f:
                    json.dump(network_config, f, indent=2)
                
                logger.info("Basic network configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Network policies configuration failed: {e}")
            return False
    
    async def setup_performance_optimization(self) -> bool:
        """Setup performance optimization"""
        try:
            logger.info("Setting up performance optimization")
            
            try:
                # Initialize performance optimizer
                self.performance_optimizer = XORBPerformanceOptimizer()
                
                # Apply basic optimizations
                services = ['neural_orchestrator', 'learning_service', 'threat_detection']
                
                for service in services:
                    await self.performance_optimizer.apply_caching_optimization(service, "main_function")
                    await self.performance_optimizer.apply_compression_optimization(service)
                
                logger.info("Performance optimization setup successfully")
                return True
                
            except NameError:
                logger.warning("Performance optimization components not available, creating basic configuration")
                
                # Create basic performance configuration
                perf_config = {
                    'caching': {
                        'enabled': True,
                        'max_size_mb': 100,
                        'ttl_seconds': 3600
                    },
                    'compression': {
                        'enabled': True,
                        'algorithm': 'gzip',
                        'level': 6
                    },
                    'connection_pooling': {
                        'enabled': True,
                        'max_connections': 100
                    }
                }
                
                config_path = self.config_dir / 'performance.json'
                with open(config_path, 'w') as f:
                    json.dump(perf_config, f, indent=2)
                
                logger.info("Basic performance configuration created")
                return True
                
        except Exception as e:
            logger.error(f"Performance optimization setup failed: {e}")
            return False
    
    async def verify_all_services_health(self) -> bool:
        """Verify health of all deployed services"""
        try:
            logger.info("Verifying service health")
            
            health_results = {}
            
            # Check deployed services
            for service_name, service_info in self.deployed_services.items():
                if 'port' in service_info:
                    port = service_info['port']
                    
                    # Check if port is listening
                    result = await self._execute_shell_command(f"nc -z localhost {port}")
                    
                    if result['return_code'] == 0:
                        health_results[service_name] = 'healthy'
                        logger.info(f"Service {service_name} is healthy")
                    else:
                        health_results[service_name] = 'unhealthy'
                        logger.warning(f"Service {service_name} is unhealthy")
            
            # Check if critical services are healthy
            critical_services = ['postgresql', 'redis', 'neural_orchestrator']
            unhealthy_critical = [svc for svc in critical_services if health_results.get(svc) != 'healthy']
            
            if unhealthy_critical:
                logger.error(f"Critical services unhealthy: {unhealthy_critical}")
                return False
            
            logger.info("All critical services are healthy")
            return True
            
        except Exception as e:
            logger.error(f"Health verification failed: {e}")
            return False
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests"""
        try:
            logger.info("Running integration tests")
            
            # Basic integration tests
            test_results = {}
            
            # Test service connectivity
            services_to_test = [
                ('neural_orchestrator', 8003, '/health'),
                ('learning_service', 8004, '/health'),
                ('threat_detection', 8005, '/health')
            ]
            
            for service_name, port, endpoint in services_to_test:
                try:
                    # Use curl to test HTTP endpoints
                    curl_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}{endpoint}"
                    result = await self._execute_shell_command(curl_cmd)
                    
                    if result['output'].strip() == '200':
                        test_results[f"{service_name}_connectivity"] = "PASS"
                        logger.info(f"Integration test PASS: {service_name} connectivity")
                    else:
                        test_results[f"{service_name}_connectivity"] = "FAIL"
                        logger.warning(f"Integration test FAIL: {service_name} connectivity")
                        
                except Exception as e:
                    test_results[f"{service_name}_connectivity"] = "ERROR"
                    logger.error(f"Integration test ERROR: {service_name} - {e}")
            
            # Check if majority of tests passed
            passed_tests = len([result for result in test_results.values() if result == "PASS"])
            total_tests = len(test_results)
            
            if passed_tests >= total_tests * 0.8:  # 80% pass rate
                logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed")
                return True
            else:
                logger.warning(f"Integration tests completed with issues: {passed_tests}/{total_tests} passed")
                return False
                
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False
    
    async def generate_deployment_report(self) -> bool:
        """Generate comprehensive deployment report"""
        try:
            logger.info("Generating deployment report")
            
            deployment_duration = 0
            if self.deployment_start_time and self.deployment_end_time:
                deployment_duration = (self.deployment_end_time - self.deployment_start_time).total_seconds()
            
            # Collect step statistics
            step_stats = {
                'total_steps': len(self.deployment_steps),
                'completed_steps': len([s for s in self.deployment_steps.values() if s.status == DeploymentStatus.COMPLETED]),
                'failed_steps': len([s for s in self.deployment_steps.values() if s.status == DeploymentStatus.FAILED]),
                'skipped_steps': len([s for s in self.deployment_steps.values() if s.status == DeploymentStatus.SKIPPED])
            }
            
            # Phase statistics
            phase_stats = {}
            for phase in DeploymentPhase:
                phase_steps = [s for s in self.deployment_steps.values() if s.phase == phase]
                phase_stats[phase.value] = {
                    'total': len(phase_steps),
                    'completed': len([s for s in phase_steps if s.status == DeploymentStatus.COMPLETED]),
                    'failed': len([s for s in phase_steps if s.status == DeploymentStatus.FAILED])
                }
            
            # Service deployment status
            service_status = {}
            for service_name, service_info in self.deployed_services.items():
                service_status[service_name] = {
                    'status': service_info.get('status', 'unknown'),
                    'port': service_info.get('port', 'N/A'),
                    'container': service_info.get('container_name', 'N/A')
                }
            
            # Component status
            component_status = {
                'load_balancer': 'configured' if self.load_balancer else 'not_configured',
                'fault_tolerance': 'configured' if self.fault_tolerance_manager else 'not_configured',
                'data_replication': 'configured' if self.data_replication_manager else 'not_configured',
                'monitoring': 'configured' if self.monitoring_manager else 'not_configured',
                'performance_optimizer': 'configured' if self.performance_optimizer else 'not_configured',
                'security_hardening': 'configured' if self.security_hardening else 'not_configured',
                'network_configurator': 'configured' if self.network_configurator else 'not_configured'
            }
            
            # Create comprehensive report
            report = {
                'deployment_id': self.config.deployment_id,
                'platform_version': self.config.platform_version,
                'environment': self.config.environment,
                'deployment_start_time': self.deployment_start_time.isoformat() if self.deployment_start_time else None,
                'deployment_end_time': self.deployment_end_time.isoformat() if self.deployment_end_time else None,
                'total_deployment_duration_seconds': deployment_duration,
                'step_statistics': step_stats,
                'phase_statistics': phase_stats,
                'service_deployment_status': service_status,
                'component_status': component_status,
                'configuration_files': {
                    'deployment_root': str(self.deployment_root),
                    'config_directory': str(self.config_dir),
                    'scripts_directory': str(self.scripts_dir),
                    'logs_directory': str(self.logs_dir)
                },
                'step_details': [
                    {
                        'step_id': step.step_id,
                        'name': step.name,
                        'phase': step.phase.value,
                        'status': step.status.value,
                        'duration_seconds': step.duration_seconds,
                        'error_message': step.error_message if step.error_message else None
                    }
                    for step in self.deployment_steps.values()
                ],
                'kpis': {
                    'deployment_success_rate': (step_stats['completed_steps'] / step_stats['total_steps']) * 100,
                    'critical_services_deployed': len([s for s in self.deployed_services.values() if s.get('status') == 'running']),
                    'average_step_duration': sum(s.duration_seconds for s in self.deployment_steps.values()) / len(self.deployment_steps),
                    'components_configured': len([status for status in component_status.values() if status == 'configured'])
                },
                'recommendations': self._generate_deployment_recommendations(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Write report to file
            report_path = self.deployment_root / f"deployment_report_{self.config.deployment_id}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate summary report
            summary_path = self.deployment_root / "deployment_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"XORB Platform Deployment Report\n")
                f.write(f"================================\n\n")
                f.write(f"Deployment ID: {self.config.deployment_id}\n")
                f.write(f"Platform Version: {self.config.platform_version}\n")
                f.write(f"Environment: {self.config.environment}\n")
                f.write(f"Duration: {deployment_duration:.2f} seconds\n\n")
                f.write(f"Step Statistics:\n")
                f.write(f"  - Total Steps: {step_stats['total_steps']}\n")
                f.write(f"  - Completed: {step_stats['completed_steps']}\n")
                f.write(f"  - Failed: {step_stats['failed_steps']}\n")
                f.write(f"  - Success Rate: {report['kpis']['deployment_success_rate']:.1f}%\n\n")
                f.write(f"Services Deployed:\n")
                for service_name, status_info in service_status.items():
                    f.write(f"  - {service_name}: {status_info['status']} (port: {status_info['port']})\n")
                f.write(f"\nComponents Configured: {report['kpis']['components_configured']}/7\n")
                f.write(f"\nReport Generated: {datetime.now()}\n")
            
            logger.info(f"Deployment report generated: {report_path}")
            logger.info(f"Deployment summary generated: {summary_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment report generation failed: {e}")
            return False
    
    def _generate_deployment_recommendations(self) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []
        
        # Check failed steps
        failed_steps = [s for s in self.deployment_steps.values() if s.status == DeploymentStatus.FAILED]
        if failed_steps:
            recommendations.append(f"Review and resolve {len(failed_steps)} failed deployment steps")
        
        # Check component configuration
        unconfigured_components = [
            name for name, status in {
                'load_balancer': self.load_balancer,
                'fault_tolerance': self.fault_tolerance_manager,
                'monitoring': self.monitoring_manager,
                'security': self.security_hardening
            }.items() if not status
        ]
        
        if unconfigured_components:
            recommendations.append(f"Configure missing components: {', '.join(unconfigured_components)}")
        
        # Check service health
        unhealthy_services = [
            name for name, info in self.deployed_services.items() 
            if info.get('status') != 'running'
        ]
        
        if unhealthy_services:
            recommendations.append(f"Investigate unhealthy services: {', '.join(unhealthy_services)}")
        
        # Performance recommendations
        if len(self.deployed_services) > 3:
            recommendations.append("Consider implementing horizontal scaling for high-load services")
        
        # Security recommendations
        recommendations.append("Enable TLS/SSL certificates for production deployments")
        recommendations.append("Configure backup and disaster recovery procedures")
        recommendations.append("Set up monitoring alerts and dashboards")
        
        return recommendations

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        total_tasks = len(self.deployment_steps)
        completed_tasks = len([t for t in self.deployment_steps.values() if t.status == DeploymentStatus.COMPLETED])
        failed_tasks = len([t for t in self.deployment_steps.values() if t.status == DeploymentStatus.FAILED])
        in_progress_tasks = len([t for t in self.deployment_steps.values() if t.status == DeploymentStatus.IN_PROGRESS])
        skipped_tasks = len([t for t in self.deployment_steps.values() if t.status == DeploymentStatus.SKIPPED])
        
        return {
            "deployment_id": self.config.deployment_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "skipped_tasks": skipped_tasks,
            "progress_percentage": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            "current_phase": self._get_current_phase()
        }

    def _get_current_phase(self) -> str:
        """Get current deployment phase"""
        in_progress_task = next((t for t in self.deployment_steps.values() if t.status == DeploymentStatus.IN_PROGRESS), None)
        if in_progress_task:
            return in_progress_task.phase.value
        
        completed_phases = set(t.phase for t in self.deployment_steps.values() if t.status == DeploymentStatus.COMPLETED)
        for phase in DeploymentPhase:
            if phase not in completed_phases:
                return phase.value
        
        return DeploymentPhase.COMPLETION.value

# Example usage and testing
async def main():
    """Main deployment execution"""
    try:
        print("🚀 XORB Unified Deployment System initializing...")
        
        # Initialize deployment
        deployment_config = DeploymentConfig(
            deployment_id=str(uuid.uuid4()),
            platform_version="1.0.0",
            environment="production"
        )
        
        deployment = XORBUnifiedDeployment(deployment_config)
        
        print(f"✅ Deployment initialized: {deployment_config.deployment_id}")
        print(f"📁 Deployment root: {deployment.deployment_root}")
        
        # Execute deployment
        print(f"\n🎯 Starting deployment execution...")
        success = await deployment.execute_deployment()
        
        if success:
            print(f"\n✅ XORB Platform deployment completed successfully!")
            print(f"📊 Check deployment report: {deployment.deployment_root}/deployment_report_{deployment_config.deployment_id}.json")
        else:
            print(f"\n❌ XORB Platform deployment failed!")
            print(f"📋 Check deployment logs for details")
        
        # Print summary
        step_stats = {
            'total': len(deployment.deployment_steps),
            'completed': len([s for s in deployment.deployment_steps.values() if s.status == DeploymentStatus.COMPLETED]),
            'failed': len([s for s in deployment.deployment_steps.values() if s.status == DeploymentStatus.FAILED])
        }
        
        print(f"\n📈 Deployment Summary:")
        print(f"- Total Steps: {step_stats['total']}")
        print(f"- Completed: {step_stats['completed']}")
        print(f"- Failed: {step_stats['failed']}")
        print(f"- Success Rate: {(step_stats['completed'] / step_stats['total']) * 100:.1f}%")
        print(f"- Services Deployed: {len(deployment.deployed_services)}")
        
        return success
        
    except Exception as e:
        logger.error(f"Main deployment execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)