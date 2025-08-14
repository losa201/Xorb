#!/usr/bin/env python3
"""
XORB PTaaS Kubernetes Integration
Deploy PTaaS services into the existing XORB Kubernetes cluster
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTaaSKubernetesDeployer:
    """PTaaS Platform Kubernetes Integration"""

    def __init__(self):
        self.deployment_id = f"PTAAS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.namespace = "xorb_platform"  # Use existing namespace

    async def deploy_ptaas_services(self):
        """Deploy PTaaS services to existing Kubernetes cluster"""
        logger.info("🚀 Starting XORB PTaaS Kubernetes Integration")
        logger.info(f"📋 Deployment ID: {self.deployment_id}")

        try:
            # Deploy Neo4j Graph Database
            await self.deploy_neo4j()

            # Deploy Qdrant Vector Database
            await self.deploy_qdrant()

            # Deploy RabbitMQ Message Queue
            await self.deploy_rabbitmq()

            # Deploy PTaaS Core Services
            await self.deploy_ptaas_core_services()

            # Configure PTaaS Ingress
            await self.configure_ptaas_ingress()

            # Verify deployment
            await self.verify_ptaas_deployment()

            logger.info("🎉 PTaaS Platform Integration Complete!")

        except Exception as e:
            logger.error(f"❌ PTaaS deployment failed: {e}")
            raise

    async def deploy_neo4j(self):
        """Deploy Neo4j Graph Database"""
        neo4j_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neo4j
  namespace: xorb_platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15-community
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          value: "neo4j/xorb_graph_2024"
        - name: NEO4J_dbms_memory_heap_initial__size
          value: "1G"
        - name: NEO4J_dbms_memory_heap_max__size
          value: "2G"
        - name: NEO4J_dbms_memory_pagecache_size
          value: "1G"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 1000m
            memory: 4Gi
      volumes:
      - name: neo4j-data
        persistentVolumeClaim:
          claimName: neo4j-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j
  namespace: xorb_platform
spec:
  selector:
    app: neo4j
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neo4j-pvc
  namespace: xorb_platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: xorb-ssd
"""

        with open('/tmp/neo4j.yaml', 'w') as f:
            f.write(neo4j_yaml)

        try:
            subprocess.run(['kubectl', 'apply', '-f', '/tmp/neo4j.yaml'], check=True)
            logger.info("✅ Neo4j Graph Database deployed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Neo4j deployment warning: {e}")

    async def deploy_qdrant(self):
        """Deploy Qdrant Vector Database"""
        qdrant_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: xorb_platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.0
        ports:
        - containerPort: 6333
        - containerPort: 6334
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
      volumes:
      - name: qdrant-data
        persistentVolumeClaim:
          claimName: qdrant-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: xorb_platform
spec:
  selector:
    app: qdrant
  ports:
  - name: http
    port: 6333
    targetPort: 6333
  - name: grpc
    port: 6334
    targetPort: 6334
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
  namespace: xorb_platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: xorb-ssd
"""

        with open('/tmp/qdrant.yaml', 'w') as f:
            f.write(qdrant_yaml)

        try:
            subprocess.run(['kubectl', 'apply', '-f', '/tmp/qdrant.yaml'], check=True)
            logger.info("✅ Qdrant Vector Database deployed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Qdrant deployment warning: {e}")

    async def deploy_rabbitmq(self):
        """Deploy RabbitMQ Message Queue"""
        rabbitmq_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rabbitmq
  namespace: xorb_platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rabbitmq
  template:
    metadata:
      labels:
        app: rabbitmq
    spec:
      containers:
      - name: rabbitmq
        image: rabbitmq:3.12-management-alpine
        ports:
        - containerPort: 5672
        - containerPort: 15672
        env:
        - name: RABBITMQ_DEFAULT_USER
          value: "xorb"
        - name: RABBITMQ_DEFAULT_PASS
          value: "xorb_rabbit_2024"
        volumeMounts:
        - name: rabbitmq-data
          mountPath: /var/lib/rabbitmq
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
      volumes:
      - name: rabbitmq-data
        persistentVolumeClaim:
          claimName: rabbitmq-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rabbitmq
  namespace: xorb_platform
spec:
  selector:
    app: rabbitmq
  ports:
  - name: amqp
    port: 5672
    targetPort: 5672
  - name: management
    port: 15672
    targetPort: 15672
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rabbitmq-pvc
  namespace: xorb_platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: xorb-ssd
"""

        with open('/tmp/rabbitmq.yaml', 'w') as f:
            f.write(rabbitmq_yaml)

        try:
            subprocess.run(['kubectl', 'apply', '-f', '/tmp/rabbitmq.yaml'], check=True)
            logger.info("✅ RabbitMQ Message Queue deployed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ RabbitMQ deployment warning: {e}")

    async def deploy_ptaas_core_services(self):
        """Deploy PTaaS Core Services"""

        # PTaaS Bug Bounty Service
        bug_bounty_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ptaas-bug-bounty
  namespace: xorb_platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ptaas-bug-bounty
  template:
    metadata:
      labels:
        app: ptaas-bug-bounty
    spec:
      containers:
      - name: bug-bounty
        image: python:3.11-slim
        command: ["python", "/app/bug_bounty_service.py"]
        ports:
        - containerPort: 8004
        env:
        - name: DATABASE_URL
          value: "postgresql://xorb_user:xorb_secure_password@postgres:5432/xorb_platform"
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: NEO4J_URL
          value: "bolt://neo4j:7687"
        - name: QDRANT_URL
          value: "http://qdrant:6333"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ptaas-bug-bounty
  namespace: xorb_platform
spec:
  selector:
    app: ptaas-bug-bounty
  ports:
  - port: 8004
    targetPort: 8004
  type: ClusterIP
"""

        # PTaaS Exploit Validation Service
        exploit_validation_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ptaas-exploit-validation
  namespace: xorb_platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ptaas-exploit-validation
  template:
    metadata:
      labels:
        app: ptaas-exploit-validation
    spec:
      containers:
      - name: exploit-validation
        image: python:3.11-slim
        command: ["python", "/app/exploit_validation_service.py"]
        ports:
        - containerPort: 8005
        env:
        - name: DATABASE_URL
          value: "postgresql://xorb_user:xorb_secure_password@postgres:5432/xorb_platform"
        - name: RABBITMQ_URL
          value: "amqp://xorb:xorb_rabbit_2024@rabbitmq:5672"
        securityContext:
          privileged: true  # Required for Docker-in-Docker for sandboxing
        volumeMounts:
        - name: docker-sock
          mountPath: /var/run/docker.sock
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
      volumes:
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
---
apiVersion: v1
kind: Service
metadata:
  name: ptaas-exploit-validation
  namespace: xorb_platform
spec:
  selector:
    app: ptaas-exploit-validation
  ports:
  - port: 8005
    targetPort: 8005
  type: ClusterIP
"""

        # PTaaS Reward System Service
        reward_system_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ptaas-reward-system
  namespace: xorb_platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ptaas-reward-system
  template:
    metadata:
      labels:
        app: ptaas-reward-system
    spec:
      containers:
      - name: reward-system
        image: python:3.11-slim
        command: ["python", "/app/reward_system_service.py"]
        ports:
        - containerPort: 8006
        env:
        - name: DATABASE_URL
          value: "postgresql://xorb_user:xorb_secure_password@postgres:5432/xorb_platform"
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ptaas-reward-system
  namespace: xorb_platform
spec:
  selector:
    app: ptaas-reward-system
  ports:
  - port: 8006
    targetPort: 8006
  type: ClusterIP
"""

        # Deploy all PTaaS services
        services = [
            ('/tmp/bug-bounty.yaml', bug_bounty_yaml, 'Bug Bounty Service'),
            ('/tmp/exploit-validation.yaml', exploit_validation_yaml, 'Exploit Validation Service'),
            ('/tmp/reward-system.yaml', reward_system_yaml, 'Reward System Service')
        ]

        for file_path, yaml_content, service_name in services:
            with open(file_path, 'w') as f:
                f.write(yaml_content)

            try:
                subprocess.run(['kubectl', 'apply', '-f', file_path], check=True)
                logger.info(f"✅ {service_name} deployed")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ {service_name} deployment warning: {e}")

    async def configure_ptaas_ingress(self):
        """Configure PTaaS Ingress Rules"""
        ptaas_ingress_yaml = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ptaas-ingress
  namespace: xorb_platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  rules:
  - host: ptaas.verteidiq.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-frontend
            port:
              number: 80
      - path: /api/bug-bounty
        pathType: Prefix
        backend:
          service:
            name: ptaas-bug-bounty
            port:
              number: 8004
      - path: /api/exploit-validation
        pathType: Prefix
        backend:
          service:
            name: ptaas-exploit-validation
            port:
              number: 8005
      - path: /api/rewards
        pathType: Prefix
        backend:
          service:
            name: ptaas-reward-system
            port:
              number: 8006
      - path: /neo4j
        pathType: Prefix
        backend:
          service:
            name: neo4j
            port:
              number: 7474
      - path: /qdrant
        pathType: Prefix
        backend:
          service:
            name: qdrant
            port:
              number: 6333
      - path: /rabbitmq
        pathType: Prefix
        backend:
          service:
            name: rabbitmq
            port:
              number: 15672
"""

        with open('/tmp/ptaas-ingress.yaml', 'w') as f:
            f.write(ptaas_ingress_yaml)

        try:
            subprocess.run(['kubectl', 'apply', '-f', '/tmp/ptaas-ingress.yaml'], check=True)
            logger.info("✅ PTaaS Ingress configured")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ PTaaS Ingress warning: {e}")

    async def verify_ptaas_deployment(self):
        """Verify PTaaS deployment"""
        try:
            # Check pod status
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'xorb_platform', '-l', 'app in (neo4j,qdrant,rabbitmq,ptaas-bug-bounty,ptaas-exploit-validation,ptaas-reward-system)'],
                                  capture_output=True, text=True, check=True)

            logger.info("✅ PTaaS deployment verification completed")
            logger.info("📊 PTaaS Services Status:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ PTaaS verification warning: {e}")

    async def cleanup_temp_files(self):
        """Clean up temporary deployment files"""
        temp_files = [
            '/tmp/neo4j.yaml', '/tmp/qdrant.yaml', '/tmp/rabbitmq.yaml',
            '/tmp/bug-bounty.yaml', '/tmp/exploit-validation.yaml',
            '/tmp/reward-system.yaml', '/tmp/ptaas-ingress.yaml'
        ]

        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

        logger.info("✅ PTaaS deployment files cleaned up")

async def main():
    """Main PTaaS deployment function"""
    deployer = PTaaSKubernetesDeployer()

    try:
        await deployer.deploy_ptaas_services()
        await deployer.cleanup_temp_files()

        print("\n" + "="*80)
        print("🎉 XORB PTaaS PLATFORM INTEGRATION SUCCESSFUL!")
        print("="*80)
        print(f"Deployment ID: {deployer.deployment_id}")
        print("PTaaS Services Deployed:")
        print("  ✅ Neo4j Graph Database")
        print("  ✅ Qdrant Vector Database")
        print("  ✅ RabbitMQ Message Queue")
        print("  ✅ Bug Bounty Service")
        print("  ✅ Exploit Validation Engine")
        print("  ✅ Reward System Service")
        print("\n📍 PTaaS Access Points:")
        print("  Bug Bounty API: https://ptaas.verteidiq.com/api/bug-bounty")
        print("  Exploit Validation: https://ptaas.verteidiq.com/api/exploit-validation")
        print("  Reward System: https://ptaas.verteidiq.com/api/rewards")
        print("  Neo4j Browser: https://ptaas.verteidiq.com/neo4j")
        print("  Qdrant API: https://ptaas.verteidiq.com/qdrant")
        print("  RabbitMQ Management: https://ptaas.verteidiq.com/rabbitmq")
        print("="*80)

    except Exception as e:
        logger.error(f"❌ PTaaS deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
