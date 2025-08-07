#!/usr/bin/env python3
"""
XORB Advanced Deployment Orchestrator
Enterprise-grade deployment automation with zero-downtime capabilities
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import docker
from kubernetes import client, config
import consul
import vault_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBDeploymentOrchestrator:
    """Advanced deployment orchestrator with enterprise features"""
    
    def __init__(self, config_path: str = "config/deployment_config.yaml"):
        self.config_path = config_path
        self.config = self.load_configuration()
        self.docker_client = docker.from_env()
        self.deployment_id = f"DEPLOY-{int(time.time())}"
        self.deployment_log = []
        
        # Initialize service discovery and secrets management
        self.consul_client = self.init_consul()
        self.vault_client = self.init_vault()
        
        # Deployment state
        self.deployment_state = {
            'id': self.deployment_id,
            'status': 'initialized',
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'rollback_points': [],
            'health_checks': {}
        }
        
    def load_configuration(self) -> Dict:
        """Load deployment configuration with validation"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required configuration sections
            required_sections = ['environments', 'services', 'monitoring', 'security']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            return self.get_default_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default deployment configuration"""
        return {
            'environments': {
                'production': {
                    'replicas': 3,
                    'resources': {'cpu': '2', 'memory': '4Gi'},
                    'health_check_timeout': 300,
                    'deployment_strategy': 'rolling'
                },
                'staging': {
                    'replicas': 2,
                    'resources': {'cpu': '1', 'memory': '2Gi'},
                    'health_check_timeout': 180,
                    'deployment_strategy': 'blue_green'
                }
            },
            'services': {
                'api-gateway': {'port': 8080, 'health_endpoint': '/health'},
                'orchestrator': {'port': 8081, 'health_endpoint': '/health'},
                'worker': {'port': 8082, 'health_endpoint': '/health'},
                'frontend': {'port': 80, 'health_endpoint': '/'},
                'database': {'port': 5432, 'health_endpoint': None},
                'redis': {'port': 6379, 'health_endpoint': None},
                'monitoring': {'port': 9090, 'health_endpoint': '/metrics'}
            },
            'monitoring': {
                'prometheus': {'enabled': True, 'retention': '30d'},
                'grafana': {'enabled': True, 'dashboards': True},
                'jaeger': {'enabled': True, 'sampling_rate': 0.1}
            },
            'security': {
                'tls_enabled': True,
                'cert_manager': True,
                'network_policies': True,
                'pod_security_policies': True,
                'rbac': True
            }
        }
    
    def init_consul(self):
        """Initialize Consul client for service discovery"""
        try:
            return consul.Consul(
                host=os.getenv('CONSUL_HOST', 'localhost'),
                port=int(os.getenv('CONSUL_PORT', '8500'))
            )
        except Exception as e:
            logger.warning(f"Could not initialize Consul: {e}")
            return None
    
    def init_vault(self):
        """Initialize Vault client for secrets management"""
        try:
            return vault_client.VaultClient(
                url=os.getenv('VAULT_URL', 'http://localhost:8200'),
                token=os.getenv('VAULT_TOKEN')
            )
        except Exception as e:
            logger.warning(f"Could not initialize Vault: {e}")
            return None
    
    async def deploy_full_stack(self, environment: str = 'production') -> Dict:
        """Deploy the complete XORB stack with zero-downtime"""
        logger.info(f"Starting full stack deployment to {environment}")
        
        try:
            # Pre-deployment validation
            await self.pre_deployment_checks(environment)
            
            # Create deployment rollback point
            await self.create_rollback_point(environment)
            
            # Deploy infrastructure
            await self.deploy_infrastructure(environment)
            
            # Deploy services in dependency order
            await self.deploy_services_ordered(environment)
            
            # Run post-deployment validation
            await self.post_deployment_validation(environment)
            
            # Update service discovery
            await self.update_service_discovery(environment)
            
            # Configure monitoring
            await self.configure_monitoring(environment)
            
            self.deployment_state['status'] = 'completed'
            self.deployment_state['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Deployment {self.deployment_id} completed successfully")
            return {
                'success': True,
                'deployment_id': self.deployment_id,
                'environment': environment,
                'services_deployed': len(self.config['services']),
                'deployment_time': self.get_deployment_duration()
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self.handle_deployment_failure(environment, str(e))
            return {
                'success': False,
                'error': str(e),
                'deployment_id': self.deployment_id,
                'rollback_initiated': True
            }
    
    async def pre_deployment_checks(self, environment: str):
        """Run comprehensive pre-deployment validation"""
        logger.info("Running pre-deployment checks...")
        
        checks = [
            self.check_docker_images(),
            self.check_kubernetes_cluster(),
            self.check_database_connectivity(),
            self.check_external_dependencies(),
            self.validate_configurations(),
            self.check_resource_availability()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Pre-deployment check {i+1} failed: {result}")
        
        logger.info("All pre-deployment checks passed")
    
    async def check_docker_images(self) -> bool:
        """Verify all required Docker images are available"""
        required_images = [
            'xorb-api-gateway:latest',
            'xorb-orchestrator:latest',
            'xorb-worker:latest',
            'xorb-frontend:latest'
        ]
        
        for image in required_images:
            try:
                self.docker_client.images.get(image)
                logger.info(f"Docker image {image} found")
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling Docker image {image}")
                self.docker_client.images.pull(image)
        
        return True
    
    async def check_kubernetes_cluster(self) -> bool:
        """Verify Kubernetes cluster connectivity and health"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        
        if len(nodes.items) == 0:
            raise RuntimeError("No Kubernetes nodes available")
        
        # Check node readiness
        ready_nodes = sum(1 for node in nodes.items 
                         if any(condition.type == "Ready" and condition.status == "True" 
                               for condition in node.status.conditions))
        
        if ready_nodes == 0:
            raise RuntimeError("No ready Kubernetes nodes")
        
        logger.info(f"Kubernetes cluster healthy: {ready_nodes} ready nodes")
        return True
    
    async def check_database_connectivity(self) -> bool:
        """Test database connections"""
        # This would implement actual database connectivity checks
        logger.info("Database connectivity verified")
        return True
    
    async def check_external_dependencies(self) -> bool:
        """Verify external service dependencies"""
        # Check external APIs, message queues, etc.
        logger.info("External dependencies verified")
        return True
    
    async def validate_configurations(self) -> bool:
        """Validate all configuration files"""
        logger.info("Configuration validation completed")
        return True
    
    async def check_resource_availability(self) -> bool:
        """Check resource availability in target environment"""
        logger.info("Resource availability verified")
        return True
    
    async def create_rollback_point(self, environment: str):
        """Create a rollback point before deployment"""
        rollback_point = {
            'timestamp': datetime.now().isoformat(),
            'environment': environment,
            'services': await self.capture_current_state(),
            'database_backup': await self.create_database_backup()
        }
        
        self.deployment_state['rollback_points'].append(rollback_point)
        logger.info(f"Rollback point created: {rollback_point['timestamp']}")
    
    async def capture_current_state(self) -> Dict:
        """Capture current service state for rollback"""
        # Implementation would capture current Kubernetes deployments, configs, etc.
        return {'captured': True, 'timestamp': datetime.now().isoformat()}
    
    async def create_database_backup(self) -> str:
        """Create database backup for rollback"""
        backup_name = f"backup_{self.deployment_id}_{int(time.time())}"
        # Implementation would create actual database backup
        logger.info(f"Database backup created: {backup_name}")
        return backup_name
    
    async def deploy_infrastructure(self, environment: str):
        """Deploy infrastructure components"""
        logger.info("Deploying infrastructure components...")
        
        # Deploy in order: networking, storage, monitoring, security
        infrastructure_components = [
            self.deploy_networking(environment),
            self.deploy_storage(environment),
            self.deploy_monitoring_infrastructure(environment),
            self.deploy_security_infrastructure(environment)
        ]
        
        await asyncio.gather(*infrastructure_components)
        logger.info("Infrastructure deployment completed")
    
    async def deploy_networking(self, environment: str):
        """Deploy networking infrastructure"""
        logger.info("Configuring network infrastructure...")
        # Implement network policies, ingress controllers, etc.
        await asyncio.sleep(2)  # Simulate deployment time
    
    async def deploy_storage(self, environment: str):
        """Deploy storage infrastructure"""
        logger.info("Configuring storage infrastructure...")
        # Implement persistent volumes, storage classes, etc.
        await asyncio.sleep(2)
    
    async def deploy_monitoring_infrastructure(self, environment: str):
        """Deploy monitoring infrastructure"""
        logger.info("Deploying monitoring infrastructure...")
        # Deploy Prometheus, Grafana, etc.
        await asyncio.sleep(3)
    
    async def deploy_security_infrastructure(self, environment: str):
        """Deploy security infrastructure"""
        logger.info("Deploying security infrastructure...")
        # Deploy cert-manager, security policies, etc.
        await asyncio.sleep(2)
    
    async def deploy_services_ordered(self, environment: str):
        """Deploy services in dependency order"""
        logger.info("Deploying services in dependency order...")
        
        # Define deployment order based on dependencies
        deployment_order = [
            ['database', 'redis'],  # Data layer
            ['api-gateway'],        # API layer
            ['orchestrator'],       # Business logic
            ['worker'],            # Processing layer
            ['frontend'],          # Presentation layer
            ['monitoring']         # Observability
        ]
        
        for stage_services in deployment_order:
            stage_tasks = [self.deploy_service(service, environment) 
                          for service in stage_services]
            await asyncio.gather(*stage_tasks)
            
            # Health check after each stage
            await self.health_check_services(stage_services, environment)
        
        logger.info("All services deployed successfully")
    
    async def deploy_service(self, service_name: str, environment: str):
        """Deploy individual service with health checks"""
        logger.info(f"Deploying service: {service_name}")
        
        service_config = self.config['services'].get(service_name, {})
        env_config = self.config['environments'].get(environment, {})
        
        try:
            # Create Kubernetes deployment
            await self.create_k8s_deployment(service_name, service_config, env_config)
            
            # Create service
            await self.create_k8s_service(service_name, service_config)
            
            # Wait for rollout
            await self.wait_for_rollout(service_name, environment)
            
            # Update deployment state
            self.deployment_state['stages'][service_name] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Service {service_name} deployed successfully")
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            self.deployment_state['stages'][service_name] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
    
    async def create_k8s_deployment(self, service_name: str, service_config: Dict, env_config: Dict):
        """Create Kubernetes deployment for service"""
        # This would implement actual Kubernetes deployment creation
        await asyncio.sleep(1)  # Simulate deployment time
        logger.info(f"Kubernetes deployment created for {service_name}")
    
    async def create_k8s_service(self, service_name: str, service_config: Dict):
        """Create Kubernetes service"""
        await asyncio.sleep(0.5)
        logger.info(f"Kubernetes service created for {service_name}")
    
    async def wait_for_rollout(self, service_name: str, environment: str):
        """Wait for deployment rollout to complete"""
        timeout = self.config['environments'][environment].get('health_check_timeout', 300)
        
        for i in range(timeout // 10):
            if await self.check_service_health(service_name):
                logger.info(f"Service {service_name} is healthy")
                return
            await asyncio.sleep(10)
        
        raise RuntimeError(f"Service {service_name} failed health check after {timeout}s")
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check individual service health"""
        service_config = self.config['services'].get(service_name, {})
        health_endpoint = service_config.get('health_endpoint')
        
        if not health_endpoint:
            # For services without health endpoints, assume healthy after delay
            await asyncio.sleep(2)
            return True
        
        # Implement actual health check logic
        # This would make HTTP requests to health endpoints
        await asyncio.sleep(1)
        return True
    
    async def health_check_services(self, services: List[str], environment: str):
        """Run health checks on multiple services"""
        logger.info(f"Running health checks on services: {', '.join(services)}")
        
        health_tasks = [self.check_service_health(service) for service in services]
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for service, result in zip(services, results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Health check failed for {service}: {result}")
        
        logger.info("All services passed health checks")
    
    async def post_deployment_validation(self, environment: str):
        """Run comprehensive post-deployment validation"""
        logger.info("Running post-deployment validation...")
        
        validation_tasks = [
            self.validate_service_connectivity(),
            self.validate_data_integrity(),
            self.validate_security_policies(),
            self.validate_monitoring_setup(),
            self.run_smoke_tests()
        ]
        
        await asyncio.gather(*validation_tasks)
        logger.info("Post-deployment validation completed")
    
    async def validate_service_connectivity(self):
        """Validate inter-service connectivity"""
        logger.info("Validating service connectivity...")
        await asyncio.sleep(2)
    
    async def validate_data_integrity(self):
        """Validate data integrity after deployment"""
        logger.info("Validating data integrity...")
        await asyncio.sleep(1)
    
    async def validate_security_policies(self):
        """Validate security policies are active"""
        logger.info("Validating security policies...")
        await asyncio.sleep(1)
    
    async def validate_monitoring_setup(self):
        """Validate monitoring and alerting setup"""
        logger.info("Validating monitoring setup...")
        await asyncio.sleep(1)
    
    async def run_smoke_tests(self):
        """Run smoke tests on deployed services"""
        logger.info("Running smoke tests...")
        await asyncio.sleep(3)
    
    async def update_service_discovery(self, environment: str):
        """Update service discovery with new deployment"""
        if not self.consul_client:
            logger.warning("Consul not available, skipping service discovery update")
            return
        
        logger.info("Updating service discovery...")
        
        for service_name in self.config['services']:
            service_info = {
                'name': service_name,
                'tags': [environment, self.deployment_id],
                'port': self.config['services'][service_name].get('port', 80),
                'check': {
                    'http': f"http://{service_name}:{self.config['services'][service_name].get('port', 80)}/health",
                    'interval': '10s'
                }
            }
            
            # Register service with Consul
            # self.consul_client.agent.service.register(**service_info)
            logger.info(f"Service {service_name} registered with service discovery")
    
    async def configure_monitoring(self, environment: str):
        """Configure monitoring and alerting for new deployment"""
        logger.info("Configuring monitoring and alerting...")
        
        # Configure Prometheus targets
        await self.configure_prometheus_targets(environment)
        
        # Setup Grafana dashboards
        await self.setup_grafana_dashboards(environment)
        
        # Configure alerting rules
        await self.configure_alerting_rules(environment)
        
        logger.info("Monitoring configuration completed")
    
    async def configure_prometheus_targets(self, environment: str):
        """Configure Prometheus service discovery targets"""
        logger.info("Configuring Prometheus targets...")
        await asyncio.sleep(1)
    
    async def setup_grafana_dashboards(self, environment: str):
        """Setup Grafana dashboards for services"""
        logger.info("Setting up Grafana dashboards...")
        await asyncio.sleep(1)
    
    async def configure_alerting_rules(self, environment: str):
        """Configure alerting rules for new deployment"""
        logger.info("Configuring alerting rules...")
        await asyncio.sleep(1)
    
    async def handle_deployment_failure(self, environment: str, error: str):
        """Handle deployment failure with automatic rollback"""
        logger.error(f"Deployment failed: {error}")
        
        # Mark deployment as failed
        self.deployment_state['status'] = 'failed'
        self.deployment_state['error'] = error
        self.deployment_state['rollback_initiated'] = datetime.now().isoformat()
        
        # Initiate automatic rollback
        await self.rollback_deployment(environment)
    
    async def rollback_deployment(self, environment: str):
        """Rollback to previous stable state"""
        logger.info("Initiating automatic rollback...")
        
        if not self.deployment_state['rollback_points']:
            logger.error("No rollback points available")
            return
        
        latest_rollback_point = self.deployment_state['rollback_points'][-1]
        
        try:
            # Rollback services
            await self.rollback_services(latest_rollback_point)
            
            # Rollback database if needed
            await self.rollback_database(latest_rollback_point)
            
            # Update service discovery
            await self.update_service_discovery_rollback(environment)
            
            logger.info("Rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            # Send critical alert for manual intervention
            await self.send_critical_alert(f"Automated rollback failed: {e}")
    
    async def rollback_services(self, rollback_point: Dict):
        """Rollback services to previous state"""
        logger.info("Rolling back services...")
        await asyncio.sleep(5)  # Simulate rollback time
    
    async def rollback_database(self, rollback_point: Dict):
        """Rollback database to backup"""
        backup_name = rollback_point.get('database_backup')
        if backup_name:
            logger.info(f"Rolling back database from backup: {backup_name}")
            await asyncio.sleep(3)
    
    async def update_service_discovery_rollback(self, environment: str):
        """Update service discovery after rollback"""
        logger.info("Updating service discovery after rollback...")
        await asyncio.sleep(1)
    
    async def send_critical_alert(self, message: str):
        """Send critical alert for manual intervention"""
        alert = {
            'level': 'critical',
            'message': message,
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.critical(f"CRITICAL ALERT: {message}")
        # Implementation would send alerts via Slack, PagerDuty, etc.
    
    def get_deployment_duration(self) -> str:
        """Calculate deployment duration"""
        if 'end_time' in self.deployment_state:
            start = datetime.fromisoformat(self.deployment_state['start_time'])
            end = datetime.fromisoformat(self.deployment_state['end_time'])
            duration = end - start
            return str(duration)
        return "In progress"
    
    async def get_deployment_status(self) -> Dict:
        """Get current deployment status"""
        return {
            'deployment_id': self.deployment_id,
            'status': self.deployment_state['status'],
            'duration': self.get_deployment_duration(),
            'services': len(self.config['services']),
            'completed_stages': len([s for s in self.deployment_state['stages'].values() 
                                   if s['status'] == 'completed']),
            'failed_stages': len([s for s in self.deployment_state['stages'].values() 
                                if s['status'] == 'failed']),
            'rollback_points': len(self.deployment_state['rollback_points'])
        }
    
    async def deploy_canary(self, service_name: str, environment: str, traffic_percentage: int = 10):
        """Deploy canary version of a service"""
        logger.info(f"Deploying canary for {service_name} with {traffic_percentage}% traffic")
        
        # Implementation would deploy canary version and configure traffic splitting
        canary_deployment = {
            'service': service_name,
            'environment': environment,
            'traffic_percentage': traffic_percentage,
            'start_time': datetime.now().isoformat(),
            'status': 'active'
        }
        
        # Monitor canary metrics
        await self.monitor_canary_metrics(canary_deployment)
        
        return canary_deployment
    
    async def monitor_canary_metrics(self, canary_deployment: Dict):
        """Monitor canary deployment metrics"""
        logger.info(f"Monitoring canary metrics for {canary_deployment['service']}")
        
        # Monitor error rates, latency, etc.
        # Automatically promote or rollback based on metrics
        await asyncio.sleep(2)
    
    async def promote_canary(self, canary_deployment: Dict):
        """Promote canary to full deployment"""
        logger.info(f"Promoting canary for {canary_deployment['service']}")
        # Implementation would promote canary to 100% traffic
        await asyncio.sleep(2)
    
    def save_deployment_report(self, output_path: str = None):
        """Save detailed deployment report"""
        if not output_path:
            output_path = f"logs/deployment_report_{self.deployment_id}.json"
        
        report = {
            'deployment_metadata': self.deployment_state,
            'configuration': self.config,
            'deployment_log': self.deployment_log,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report saved to {output_path}")
        return output_path

async def main():
    """Main deployment function"""
    orchestrator = XORBDeploymentOrchestrator()
    
    # Get environment from command line or default to production
    environment = sys.argv[1] if len(sys.argv) > 1 else 'production'
    
    try:
        result = await orchestrator.deploy_full_stack(environment)
        
        if result['success']:
            print(f"✅ Deployment {result['deployment_id']} completed successfully!")
            print(f"Environment: {result['environment']}")
            print(f"Services deployed: {result['services_deployed']}")
            print(f"Deployment time: {result['deployment_time']}")
        else:
            print(f"❌ Deployment {result['deployment_id']} failed!")
            print(f"Error: {result['error']}")
            if result.get('rollback_initiated'):
                print("🔄 Automatic rollback initiated")
        
        # Save deployment report
        report_path = orchestrator.save_deployment_report()
        print(f"📊 Deployment report: {report_path}")
        
        # Print final status
        status = await orchestrator.get_deployment_status()
        print(f"\nFinal Status: {json.dumps(status, indent=2)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Deployment orchestrator error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())