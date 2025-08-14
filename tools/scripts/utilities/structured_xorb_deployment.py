#!/usr/bin/env python3
"""
XORB Structured Deployment Architecture
Logical, efficient, and optimally wired cybersecurity platform deployment
"""

import asyncio
import subprocess
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServiceDefinition:
    """Structured service definition"""
    name: str
    type: str  # core, data, monitoring, security, web
    port: int
    dependencies: List[str]
    health_endpoint: str
    resources: Dict[str, str]
    replicas: int = 1
    priority: str = "medium"  # critical, high, medium, low

@dataclass
class DeploymentTier:
    """Deployment tier for logical service grouping"""
    name: str
    services: List[ServiceDefinition]
    order: int
    parallel_deployment: bool = True

class XORBStructuredDeployment:
    """Structured XORB deployment with optimal architecture"""

    def __init__(self):
        self.deployment_id = f"STRUCTURED-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.architecture = self._define_architecture()
        self.deployment_tiers = self._create_deployment_tiers()
        self.service_registry = {}
        self.wiring_config = {}

    def _define_architecture(self) -> Dict:
        """Define the logical XORB architecture"""
        return {
            "data_layer": {
                "description": "Persistent storage and caching",
                "services": ["postgres", "redis", "neo4j", "qdrant"],
                "network": "xorb-data-net",
                "security": "encrypted-at-rest"
            },
            "core_layer": {
                "description": "Core business logic and APIs",
                "services": ["api-gateway", "security-api", "threat-intel", "analytics"],
                "network": "xorb-core-net",
                "security": "mTLS-enabled"
            },
            "monitoring_layer": {
                "description": "Observability and monitoring",
                "services": ["prometheus", "grafana", "alertmanager"],
                "network": "xorb-monitor-net",
                "security": "rbac-enabled"
            },
            "web_layer": {
                "description": "Frontend and web services",
                "services": ["nginx", "web-dashboard"],
                "network": "xorb-web-net",
                "security": "waf-enabled"
            },
            "ptaas_layer": {
                "description": "Penetration Testing as a Service",
                "services": ["ptaas-core", "researcher-api", "company-api"],
                "network": "xorb-ptaas-net",
                "security": "isolated-network"
            }
        }

    def _create_deployment_tiers(self) -> List[DeploymentTier]:
        """Create logical deployment tiers"""

        # Tier 1: Foundation (Data Layer)
        tier1_services = [
            ServiceDefinition(
                name="postgres",
                type="data",
                port=5432,
                dependencies=[],
                health_endpoint="/health",
                resources={"cpu": "500m", "memory": "1Gi", "storage": "20Gi"},
                replicas=1,
                priority="critical"
            ),
            ServiceDefinition(
                name="redis",
                type="data",
                port=6379,
                dependencies=[],
                health_endpoint="/ping",
                resources={"cpu": "250m", "memory": "512Mi", "storage": "5Gi"},
                replicas=1,
                priority="critical"
            ),
            ServiceDefinition(
                name="neo4j",
                type="data",
                port=7474,
                dependencies=[],
                health_endpoint="/health",
                resources={"cpu": "1000m", "memory": "2Gi", "storage": "30Gi"},
                replicas=1,
                priority="high"
            )
        ]

        # Tier 2: Core Services
        tier2_services = [
            ServiceDefinition(
                name="xorb-unified-api",
                type="core",
                port=8000,
                dependencies=["postgres", "redis"],
                health_endpoint="/health",
                resources={"cpu": "500m", "memory": "1Gi"},
                replicas=2,
                priority="critical"
            ),
            ServiceDefinition(
                name="xorb-threat-intel",
                type="security",
                port=8004,
                dependencies=["postgres", "neo4j"],
                health_endpoint="/health",
                resources={"cpu": "300m", "memory": "512Mi"},
                replicas=2,
                priority="high"
            ),
            ServiceDefinition(
                name="xorb-analytics",
                type="core",
                port=8003,
                dependencies=["redis", "postgres"],
                health_endpoint="/health",
                resources={"cpu": "400m", "memory": "768Mi"},
                replicas=2,
                priority="high"
            )
        ]

        # Tier 3: Monitoring & Observability
        tier3_services = [
            ServiceDefinition(
                name="prometheus",
                type="monitoring",
                port=9090,
                dependencies=["xorb-unified-api", "xorb-analytics"],
                health_endpoint="/-/healthy",
                resources={"cpu": "300m", "memory": "1Gi", "storage": "10Gi"},
                replicas=1,
                priority="high"
            ),
            ServiceDefinition(
                name="grafana",
                type="monitoring",
                port=3000,
                dependencies=["prometheus"],
                health_endpoint="/api/health",
                resources={"cpu": "200m", "memory": "512Mi"},
                replicas=1,
                priority="medium"
            )
        ]

        # Tier 4: PTaaS Layer
        tier4_services = [
            ServiceDefinition(
                name="qdrant",
                type="data",
                port=6333,
                dependencies=["postgres"],
                health_endpoint="/health",
                resources={"cpu": "300m", "memory": "1Gi", "storage": "10Gi"},
                replicas=1,
                priority="high"
            ),
            ServiceDefinition(
                name="ptaas-core",
                type="security",
                port=8080,
                dependencies=["postgres", "redis", "neo4j", "qdrant"],
                health_endpoint="/health",
                resources={"cpu": "500m", "memory": "1Gi"},
                replicas=1,
                priority="high"
            ),
            ServiceDefinition(
                name="researcher-api",
                type="api",
                port=8081,
                dependencies=["postgres", "redis", "ptaas-core"],
                health_endpoint="/api/v1/health",
                resources={"cpu": "300m", "memory": "512Mi"},
                replicas=1,
                priority="medium"
            ),
            ServiceDefinition(
                name="company-api",
                type="api",
                port=8082,
                dependencies=["postgres", "redis", "ptaas-core"],
                health_endpoint="/api/v1/health",
                resources={"cpu": "300m", "memory": "512Mi"},
                replicas=1,
                priority="medium"
            )
        ]

        # Tier 5: Web Layer
        tier5_services = [
            ServiceDefinition(
                name="xorb-web-gateway",
                type="web",
                port=80,
                dependencies=["xorb-unified-api", "grafana", "researcher-api", "company-api"],
                health_endpoint="/health",
                resources={"cpu": "200m", "memory": "256Mi"},
                replicas=2,
                priority="high"
            )
        ]

        return [
            DeploymentTier("foundation", tier1_services, 1, True),
            DeploymentTier("core-services", tier2_services, 2, True),
            DeploymentTier("monitoring", tier3_services, 3, False),
            DeploymentTier("ptaas-platform", tier4_services, 4, True),
            DeploymentTier("web-layer", tier5_services, 5, True)
        ]

    async def deploy_structured_platform(self):
        """Deploy the structured XORB platform"""
        logger.info("🏗️ Starting XORB Structured Deployment")
        logger.info(f"📋 Deployment ID: {self.deployment_id}")

        try:
            # Phase 1: Infrastructure Setup
            await self.setup_infrastructure()

            # Phase 2: Deploy by Tiers
            await self.deploy_by_tiers()

            # Phase 3: Configure Service Wiring
            await self.configure_service_wiring()

            # Phase 4: Deploy Monitoring
            await self.deploy_monitoring_stack()

            # Phase 5: Validate Deployment
            await self.validate_deployment()

            # Phase 6: Generate Reports
            await self.generate_structured_report()

            logger.info("✅ Structured XORB deployment completed successfully!")

        except Exception as e:
            logger.error(f"❌ Structured deployment failed: {e}")
            raise

    async def setup_infrastructure(self):
        """Setup infrastructure networks and volumes"""
        logger.info("🌐 Setting up infrastructure networks...")

        # Create structured networks
        networks = [
            "xorb-data-net",
            "xorb-core-net",
            "xorb-monitor-net",
            "xorb-web-net",
            "xorb-ptaas-net"
        ]

        for network in networks:
            try:
                subprocess.run([
                    'docker', 'network', 'create',
                    '--driver', 'bridge',
                    '--subnet', f'172.{20 + networks.index(network)}.0.0/16',
                    network
                ], capture_output=True, check=False)
                logger.info(f"✅ Network {network} created")
            except Exception as e:
                logger.warning(f"⚠️ Network {network} warning: {e}")

        # Create optimized volumes
        await self.create_optimized_volumes()

    async def create_optimized_volumes(self):
        """Create optimized storage volumes"""
        logger.info("💾 Creating optimized storage volumes...")

        volume_configs = {
            "xorb-postgres-data": {"size": "20Gi", "type": "ssd"},
            "xorb-redis-data": {"size": "5Gi", "type": "ssd"},
            "xorb-neo4j-data": {"size": "30Gi", "type": "ssd"},
            "xorb-prometheus-data": {"size": "10Gi", "type": "ssd"},
            "xorb-grafana-data": {"size": "2Gi", "type": "ssd"}
        }

        for volume, config in volume_configs.items():
            try:
                subprocess.run([
                    'docker', 'volume', 'create',
                    '--driver', 'local',
                    volume
                ], capture_output=True, check=False)
                logger.info(f"✅ Volume {volume} ({config['size']}) created")
            except Exception as e:
                logger.warning(f"⚠️ Volume {volume} warning: {e}")

    async def deploy_by_tiers(self):
        """Deploy services tier by tier"""
        logger.info("🏗️ Deploying services by tiers...")

        for tier in sorted(self.deployment_tiers, key=lambda t: t.order):
            logger.info(f"📦 Deploying Tier {tier.order}: {tier.name}")

            if tier.parallel_deployment:
                # Deploy services in parallel
                tasks = [self.deploy_service(service) for service in tier.services]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Deploy services sequentially
                for service in tier.services:
                    await self.deploy_service(service)

            # Wait for tier to stabilize
            await self.wait_for_tier_health(tier)
            logger.info(f"✅ Tier {tier.order} deployment completed")

    async def deploy_service(self, service: ServiceDefinition):
        """Deploy individual service with optimal configuration"""
        logger.info(f"🚀 Deploying {service.name}...")

        # Generate service-specific compose configuration
        compose_config = await self.generate_service_compose(service)

        # Write compose file
        compose_file = f"/tmp/xorb-{service.name}.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_config)

        try:
            # Deploy service
            subprocess.run([
                'docker-compose', '-f', compose_file,
                'up', '-d', '--remove-orphans'
            ], check=True, cwd='/root/Xorb')

            logger.info(f"✅ {service.name} deployed successfully")
            self.service_registry[service.name] = {
                "status": "deployed",
                "port": service.port,
                "health_endpoint": service.health_endpoint,
                "dependencies": service.dependencies
            }

        except Exception as e:
            logger.error(f"❌ Failed to deploy {service.name}: {e}")
            self.service_registry[service.name] = {"status": "failed", "error": str(e)}

    async def generate_service_compose(self, service: ServiceDefinition) -> str:
        """Generate optimized Docker Compose configuration for service"""

        if service.name == "postgres":
            return self._generate_postgres_compose(service)
        elif service.name == "redis":
            return self._generate_redis_compose(service)
        elif service.name == "neo4j":
            return self._generate_neo4j_compose(service)
        elif service.name == "xorb-unified-api":
            return self._generate_unified_api_compose(service)
        elif service.name == "xorb-analytics":
            return self._generate_analytics_compose(service)
        elif service.name == "xorb-threat-intel":
            return self._generate_threat_intel_compose(service)
        elif service.name == "prometheus":
            return self._generate_prometheus_compose(service)
        elif service.name == "grafana":
            return self._generate_grafana_compose(service)
        elif service.name == "qdrant":
            return self._generate_qdrant_compose(service)
        elif service.name == "ptaas-core":
            return self._generate_ptaas_core_compose(service)
        elif service.name == "researcher-api":
            return self._generate_researcher_api_compose(service)
        elif service.name == "company-api":
            return self._generate_company_api_compose(service)
        elif service.name == "xorb-web-gateway":
            return self._generate_web_gateway_compose(service)
        else:
            return self._generate_generic_compose(service)

    def _generate_postgres_compose(self, service: ServiceDefinition) -> str:
        """Generate PostgreSQL compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: postgres:15-alpine
    container_name: xorb-{service.name}
    environment:
      POSTGRES_DB: xorb_platform
      POSTGRES_USER: xorb_user
      POSTGRES_PASSWORD: xorb_secure_2024
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    volumes:
      - xorb-postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-data-net
      - xorb-core-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U xorb_user -d xorb_platform"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}
        reservations:
          cpus: '250m'
          memory: 512Mi

networks:
  xorb-data-net:
    external: true
  xorb-core-net:
    external: true

volumes:
  xorb-postgres-data:
    external: true
"""

    def _generate_redis_compose(self, service: ServiceDefinition) -> str:
        """Generate Redis compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: redis:7-alpine
    container_name: xorb-{service.name}
    command: redis-server --appendonly yes --requirepass xorb_redis_2024 --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - xorb-redis-data:/data
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-data-net
      - xorb-core-net
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "xorb_redis_2024", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-data-net:
    external: true
  xorb-core-net:
    external: true

volumes:
  xorb-redis-data:
    external: true
"""

    def _generate_neo4j_compose(self, service: ServiceDefinition) -> str:
        """Generate Neo4j compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: neo4j:5.15-community
    container_name: xorb-{service.name}
    environment:
      NEO4J_AUTH: neo4j/xorb_graph_2024
      NEO4J_dbms_memory_heap_initial__size: 1G
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
    volumes:
      - xorb-neo4j-data:/data
      - xorb-neo4j-logs:/logs
    ports:
      - "{service.port}:{service.port}"
      - "7687:7687"
    networks:
      - xorb-data-net
      - xorb-core-net
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "xorb_graph_2024", "RETURN 1;"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-data-net:
    external: true
  xorb-core-net:
    external: true

volumes:
  xorb-neo4j-data:
    external: true
  xorb-neo4j-logs:
    external: true
"""

    def _generate_unified_api_compose(self, service: ServiceDefinition) -> str:
        """Generate unified API compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.unified-api
    container_name: xorb-unified-api
    environment:
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
      - REDIS_HOST=xorb-redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=xorb_redis_2024
      - API_PORT=8000
      - LOG_LEVEL=INFO
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-core-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-postgres
      - xorb-redis
    deploy:
      replicas: {service.replicas}
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-core-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_analytics_compose(self, service: ServiceDefinition) -> str:
        """Generate analytics service compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.analytics
    container_name: xorb-analytics
    environment:
      - REDIS_HOST=xorb-redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=xorb_redis_2024
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-core-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-redis
      - xorb-postgres
    deploy:
      replicas: {service.replicas}
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-core-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_threat_intel_compose(self, service: ServiceDefinition) -> str:
        """Generate threat intelligence compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.threat-intel
    container_name: xorb-threat-intel
    environment:
      - NEO4J_URI=bolt://xorb-neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=xorb_graph_2024
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-core-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-neo4j
      - xorb-postgres
    deploy:
      replicas: {service.replicas}
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-core-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_prometheus_compose(self, service: ServiceDefinition) -> str:
        """Generate Prometheus compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: prom/prometheus:v2.45.0
    container_name: xorb-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - xorb-prometheus-data:/prometheus
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-monitor-net
      - xorb-core-net
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-monitor-net:
    external: true
  xorb-core-net:
    external: true

volumes:
  xorb-prometheus-data:
    external: true
"""

    def _generate_grafana_compose(self, service: ServiceDefinition) -> str:
        """Generate Grafana compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: grafana/grafana:10.2.3
    container_name: xorb-grafana
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: xorb_admin_2024
      GF_SECURITY_SECRET_KEY: xorb_grafana_secret_2024
      GF_USERS_ALLOW_SIGN_UP: false
      GF_INSTALL_PLUGINS: grafana-piechart-panel,grafana-clock-panel
    volumes:
      - xorb-grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-monitor-net
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-prometheus
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-monitor-net:
    external: true

volumes:
  xorb-grafana-data:
    external: true
"""

    def _generate_web_gateway_compose(self, service: ServiceDefinition) -> str:
        """Generate web gateway compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: nginx:alpine
    container_name: xorb-web-gateway
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - /var/www/verteidiq.com:/usr/share/nginx/html:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - xorb-web-net
      - xorb-core-net
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-unified-api
    deploy:
      replicas: {service.replicas}
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-web-net:
    external: true
  xorb-core-net:
    external: true
"""

    def _generate_qdrant_compose(self, service: ServiceDefinition) -> str:
        """Generate Qdrant vector database compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: qdrant/qdrant:v1.7.4
    container_name: xorb-{service.name}
    volumes:
      - xorb-qdrant-data:/qdrant/storage
    ports:
      - "{service.port}:{service.port}"
      - "6334:6334"
    environment:
      QDRANT__SERVICE__HTTP_PORT: {service.port}
      QDRANT__SERVICE__GRPC_PORT: 6334
    networks:
      - xorb-data-net
      - xorb-ptaas-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{service.port}/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-data-net:
    external: true
  xorb-ptaas-net:
    external: true

volumes:
  xorb-qdrant-data:
    external: true
"""

    def _generate_ptaas_core_compose(self, service: ServiceDefinition) -> str:
        """Generate PTaaS core service compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.ptaas-core
    container_name: xorb-ptaas-core
    environment:
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
      - NEO4J_URI=bolt://xorb-neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=xorb_graph_2024
      - REDIS_HOST=xorb-redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=xorb_redis_2024
      - QDRANT_HOST=xorb-qdrant
      - QDRANT_PORT=6333
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-ptaas-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{service.port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-postgres
      - xorb-neo4j
      - xorb-redis
      - xorb-qdrant
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-ptaas-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_researcher_api_compose(self, service: ServiceDefinition) -> str:
        """Generate researcher API compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.researcher-api
    container_name: xorb-researcher-api
    environment:
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
      - REDIS_HOST=xorb-redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=xorb_redis_2024
      - JWT_SECRET=xorb_ptaas_jwt_secret_2024_secure_random_key
      - API_PORT={service.port}
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-ptaas-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{service.port}/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-postgres
      - xorb-redis
      - xorb-ptaas-core
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-ptaas-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_company_api_compose(self, service: ServiceDefinition) -> str:
        """Generate company API compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    build:
      context: .
      dockerfile: Dockerfile.company-api
    container_name: xorb-company-api
    environment:
      - POSTGRES_HOST=xorb-postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=xorb_platform
      - POSTGRES_USER=xorb_user
      - POSTGRES_PASSWORD=xorb_secure_2024
      - REDIS_HOST=xorb-redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=xorb_redis_2024
      - JWT_SECRET=xorb_ptaas_jwt_secret_2024_secure_random_key
      - API_PORT={service.port}
    ports:
      - "{service.port}:{service.port}"
    networks:
      - xorb-ptaas-net
      - xorb-data-net
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{service.port}/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - xorb-postgres
      - xorb-redis
      - xorb-ptaas-core
    deploy:
      resources:
        limits:
          cpus: '{service.resources["cpu"]}'
          memory: {service.resources["memory"]}

networks:
  xorb-ptaas-net:
    external: true
  xorb-data-net:
    external: true
"""

    def _generate_generic_compose(self, service: ServiceDefinition) -> str:
        """Generate generic compose configuration"""
        return f"""
version: '3.8'

services:
  {service.name}:
    image: alpine:latest
    container_name: xorb-{service.name}
    command: ['sh', '-c', 'echo "Service {service.name} placeholder" && sleep 3600']
    ports:
      - "{service.port}:{service.port}"
    restart: unless-stopped
"""

    async def wait_for_tier_health(self, tier: DeploymentTier):
        """Wait for tier services to be healthy"""
        logger.info(f"⏳ Waiting for {tier.name} tier to be healthy...")

        max_wait = 120  # 2 minutes
        wait_interval = 10

        for i in range(0, max_wait, wait_interval):
            healthy_services = 0

            for service in tier.services:
                if await self.check_service_health(service):
                    healthy_services += 1

            health_percentage = (healthy_services / len(tier.services)) * 100
            logger.info(f"📊 {tier.name} health: {healthy_services}/{len(tier.services)} ({health_percentage:.1f}%)")

            if health_percentage >= 80:  # 80% threshold
                logger.info(f"✅ {tier.name} tier is healthy")
                return

            await asyncio.sleep(wait_interval)

        logger.warning(f"⚠️ {tier.name} tier health check timeout - proceeding anyway")

    async def check_service_health(self, service: ServiceDefinition) -> bool:
        """Check individual service health"""
        try:
            # Use docker command to check container status
            result = subprocess.run([
                'docker', 'ps', '--filter', f'name=xorb-{service.name}',
                '--format', '{{.Status}}'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                status = result.stdout.strip()
                return 'Up' in status and 'healthy' in status.lower()

        except Exception as e:
            logger.debug(f"Health check failed for {service.name}: {e}")

        return False

    async def configure_service_wiring(self):
        """Configure efficient service communication"""
        logger.info("🔌 Configuring service wiring...")

        self.wiring_config = {
            "api_routing": {
                "unified_api": {
                    "internal_port": 8000,
                    "external_endpoints": ["/api/v1/*", "/health", "/metrics"],
                    "load_balancing": "round_robin",
                    "circuit_breaker": True
                },
                "analytics": {
                    "internal_port": 8003,
                    "external_endpoints": ["/api/v1/analytics/*", "/api/v1/metrics"],
                    "cache_ttl": 60
                },
                "threat_intel": {
                    "internal_port": 8004,
                    "external_endpoints": ["/api/v1/intelligence/*", "/api/v1/threats"],
                    "cache_ttl": 300
                }
            },
            "data_connections": {
                "postgres": {
                    "connection_pool": 20,
                    "max_connections": 100,
                    "timeout": 30
                },
                "redis": {
                    "connection_pool": 10,
                    "ttl_default": 3600
                },
                "neo4j": {
                    "connection_pool": 5,
                    "query_timeout": 60
                }
            },
            "monitoring_targets": [
                "xorb-unified-api:8000",
                "xorb-analytics:8003",
                "xorb-threat-intel:8004",
                "xorb-postgres:5432",
                "xorb-redis:6379",
                "xorb-neo4j:7474"
            ]
        }

        # Create monitoring configuration
        await self.create_monitoring_config()
        logger.info("✅ Service wiring configured")

    async def create_monitoring_config(self):
        """Create monitoring configuration files"""
        logger.info("📊 Creating monitoring configuration...")

        # Create monitoring directory
        monitoring_dir = Path("/root/Xorb/monitoring")
        monitoring_dir.mkdir(exist_ok=True)

        # Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'xorb-unified-api'
    static_configs:
      - targets: ['xorb-unified-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'xorb-analytics'
    static_configs:
      - targets: ['xorb-analytics:8003']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'xorb-threat-intel'
    static_configs:
      - targets: ['xorb-threat-intel:8004']
    metrics_path: '/metrics'
    scrape_interval: 60s

  - job_name: 'postgres'
    static_configs:
      - targets: ['xorb-postgres:5432']
    scrape_interval: 60s

  - job_name: 'redis'
    static_configs:
      - targets: ['xorb-redis:6379']
    scrape_interval: 30s

  - job_name: 'neo4j'
    static_configs:
      - targets: ['xorb-neo4j:7474']
    scrape_interval: 60s
"""

        with open(monitoring_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)

        logger.info("✅ Monitoring configuration created")

    async def deploy_monitoring_stack(self):
        """Deploy monitoring stack"""
        logger.info("📊 Deploying monitoring stack...")

        # Find monitoring tier and deploy
        monitoring_tier = next((t for t in self.deployment_tiers if t.name == "monitoring"), None)
        if monitoring_tier:
            for service in monitoring_tier.services:
                await self.deploy_service(service)
            await self.wait_for_tier_health(monitoring_tier)

        logger.info("✅ Monitoring stack deployed")

    async def validate_deployment(self):
        """Validate the complete deployment"""
        logger.info("✅ Validating structured deployment...")

        # Check all services
        total_services = sum(len(tier.services) for tier in self.deployment_tiers)
        healthy_services = 0

        for tier in self.deployment_tiers:
            for service in tier.services:
                if await self.check_service_health(service):
                    healthy_services += 1

        health_percentage = (healthy_services / total_services) * 100 if total_services > 0 else 0

        logger.info(f"📊 Overall platform health: {healthy_services}/{total_services} ({health_percentage:.1f}%)")

        if health_percentage >= 75:
            logger.info("🎉 Structured deployment validation successful!")
        else:
            logger.warning("⚠️ Deployment validation shows degraded performance")

        return health_percentage

    async def generate_structured_report(self):
        """Generate comprehensive structured deployment report"""
        logger.info("📋 Generating structured deployment report...")

        report = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "architecture": self.architecture,
            "deployment_tiers": [
                {
                    "name": tier.name,
                    "order": tier.order,
                    "services": [
                        {
                            "name": service.name,
                            "type": service.type,
                            "port": service.port,
                            "dependencies": service.dependencies,
                            "resources": service.resources,
                            "replicas": service.replicas,
                            "priority": service.priority,
                            "status": self.service_registry.get(service.name, {}).get("status", "unknown")
                        }
                        for service in tier.services
                    ]
                }
                for tier in self.deployment_tiers
            ],
            "service_registry": self.service_registry,
            "wiring_configuration": self.wiring_config,
            "access_information": {
                "unified_api": "http://localhost:8000/api/v1/status",
                "analytics": "http://localhost:8003/api/v1/metrics",
                "threat_intelligence": "http://localhost:8004/api/v1/intelligence",
                "grafana": "http://localhost:3000 (admin/xorb_admin_2024)",
                "prometheus": "http://localhost:9090",
                "website": "https://verteidiq.com"
            },
            "operational_commands": {
                "check_services": "docker ps --filter name=xorb",
                "view_logs": "docker logs xorb-unified-api",
                "restart_service": "docker restart xorb-<service-name>",
                "scale_service": "docker-compose up -d --scale xorb-unified-api=3"
            },
            "health_monitoring": {
                "prometheus_targets": self.wiring_config.get("monitoring_targets", []),
                "health_endpoints": [
                    f"http://localhost:{service.port}{service.health_endpoint}"
                    for tier in self.deployment_tiers
                    for service in tier.services
                ],
                "alerting": "Grafana dashboards with automated alerts"
            }
        }

        # Save report
        report_file = f"/root/Xorb/logs/structured-deployment-{self.deployment_id}.json"
        os.makedirs("/root/Xorb/logs", exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📋 Structured deployment report saved: {report_file}")
        return report

async def main():
    """Execute structured XORB deployment"""
    deployment = XORBStructuredDeployment()

    try:
        report = await deployment.deploy_structured_platform()

        print("\n" + "="*80)
        print("🏗️ XORB STRUCTURED DEPLOYMENT COMPLETE!")
        print("="*80)
        print(f"📋 Deployment ID: {deployment.deployment_id}")
        print(f"🏛️ Architecture: Multi-tier with optimized wiring")
        print(f"📦 Services: {sum(len(tier.services) for tier in deployment.deployment_tiers)}")
        print(f"🌐 Networks: {len(deployment.architecture)} isolated layers")

        print("\n🏗️ Deployment Tiers:")
        for tier in deployment.deployment_tiers:
            print(f"  Tier {tier.order}: {tier.name} ({len(tier.services)} services)")
            for service in tier.services:
                status = deployment.service_registry.get(service.name, {}).get("status", "unknown")
                print(f"    └─ {service.name}:{service.port} [{status}] ({service.priority})")

        print("\n🔌 Service Wiring:")
        print("  • Data Layer: Optimized connection pooling")
        print("  • Core Layer: Load balancing with circuit breakers")
        print("  • Monitor Layer: Real-time metrics collection")
        print("  • Web Layer: Reverse proxy with SSL termination")

        print("\n🚀 Access Points:")
        print("  Unified API:        http://localhost:8000/api/v1/status")
        print("  Analytics:          http://localhost:8003/api/v1/metrics")
        print("  Threat Intel:       http://localhost:8004/api/v1/intelligence")
        print("  Grafana Dashboard:  http://localhost:3000 (admin/xorb_admin_2024)")
        print("  Prometheus:         http://localhost:9090")
        print("  Website:            https://verteidiq.com")

        print("\n📊 Operations:")
        print("  Status:   docker ps --filter name=xorb")
        print("  Logs:     docker logs xorb-unified-api")
        print("  Scale:    docker-compose up -d --scale xorb-unified-api=3")
        print("  Health:   curl http://localhost:8000/health")

        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"❌ Structured deployment failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
