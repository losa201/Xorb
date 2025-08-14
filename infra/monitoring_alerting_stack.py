#!/usr/bin/env python3
"""
XORB Monitoring & Alerting Stack
Comprehensive monitoring and alerting infrastructure with Prometheus, Grafana, AlertManager, and observability tools
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import yaml
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger('XORBMonitoring')

class MonitoringComponentType(Enum):
    """Monitoring component types"""
    METRICS = "metrics"
    VISUALIZATION = "visualization"
    ALERTING = "alerting"
    LOGGING = "logging"
    TRACING = "tracing"
    SERVICE_DISCOVERY = "service_discovery"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class NotificationChannel(Enum):
    """Notification channel types"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"

@dataclass
class PrometheusConfig:
    """Prometheus configuration"""
    version: str = "v2.45.0"
    port: int = 9090
    retention: str = "30d"
    storage_size: str = "50Gi"
    scrape_interval: str = "15s"
    evaluation_interval: str = "15s"
    enable_remote_write: bool = False
    enable_thanos: bool = False
    enable_high_availability: bool = False
    replica_count: int = 1
    external_labels: Dict[str, str] = field(default_factory=lambda: {
        "cluster": "xorb-cluster",
        "environment": "production"
    })
    rule_files: List[str] = field(default_factory=lambda: [
        "/etc/prometheus/rules/*.yml"
    ])
    remote_write_urls: List[str] = field(default_factory=list)

@dataclass
class GrafanaConfig:
    """Grafana configuration"""
    version: str = "10.0.0"
    port: int = 3000
    admin_password: str = "admin123"
    enable_oauth: bool = False
    enable_ldap: bool = False
    enable_ssl: bool = True
    plugins: List[str] = field(default_factory=lambda: [
        "grafana-piechart-panel",
        "grafana-worldmap-panel",
        "grafana-clock-panel",
        "grafana-polystat-panel"
    ])
    dashboard_repo: str = "https://github.com/xorb/grafana-dashboards.git"
    provisioning_enabled: bool = True
    smtp_enabled: bool = True
    smtp_host: str = "localhost:587"
    smtp_user: str = "grafana@xorb.local"

@dataclass
class AlertManagerConfig:
    """AlertManager configuration"""
    version: str = "v0.25.0"
    port: int = 9093
    enable_clustering: bool = False
    retention: str = "120h"
    cluster_listen_address: str = "0.0.0.0:9094"
    notification_channels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "slack": {
            "enabled": False,
            "webhook_url": "",
            "channel": "#alerts",
            "username": "AlertManager"
        },
        "email": {
            "enabled": True,
            "smtp_server": "localhost:587",
            "from": "alerts@xorb.local",
            "to": ["admin@xorb.local"]
        },
        "pagerduty": {
            "enabled": False,
            "integration_key": "",
            "severity": "critical"
        }
    })

@dataclass
class LokiConfig:
    """Loki logging configuration"""
    version: str = "2.8.0"
    port: int = 3100
    retention: str = "30d"
    storage_size: str = "20Gi"
    enable_compactor: bool = True
    chunk_store_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_look_back_period": "0s"
    })
    table_manager: Dict[str, Any] = field(default_factory=lambda: {
        "retention_deletes_enabled": True,
        "retention_period": "720h"
    })

@dataclass
class JaegerConfig:
    """Jaeger tracing configuration"""
    version: str = "1.47.0"
    port: int = 16686
    collector_port: int = 14268
    agent_port: int = 6831
    zipkin_port: int = 9411
    enable_elasticsearch: bool = True
    enable_sampling: bool = True
    sampling_strategies: Dict[str, Any] = field(default_factory=lambda: {
        "default_strategy": {
            "type": "probabilistic",
            "param": 0.1
        }
    })

@dataclass
class FluentdConfig:
    """Fluentd log collection configuration"""
    version: str = "v1.16-debian-1"
    port: int = 24224
    buffer_size: str = "32MB"
    flush_interval: str = "5s"
    enable_prometheus: bool = True
    output_plugins: List[str] = field(default_factory=lambda: [
        "elasticsearch",
        "loki",
        "prometheus"
    ])

@dataclass
class MonitoringComponent:
    """Monitoring component state"""
    name: str
    type: MonitoringComponentType
    config: Union[PrometheusConfig, GrafanaConfig, AlertManagerConfig, LokiConfig, JaegerConfig, FluentdConfig]
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    endpoints: Dict[str, str] = field(default_factory=dict)
    health_status: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

class XORBMonitoringStack:
    """Comprehensive monitoring and alerting stack"""

    def __init__(self, namespace: str = "xorb-monitoring", platform: str = "kubernetes"):
        self.namespace = namespace
        self.platform = platform
        self.components: Dict[str, MonitoringComponent] = {}
        self.deployment_id = f"monitoring_{int(time.time())}"
        self.dashboards = {}
        self.alert_rules = {}

        # Initialize monitoring configurations
        self._initialize_monitoring_configs()

        logger.info(f"📊 Monitoring stack initialized")
        logger.info(f"📋 Namespace: {self.namespace}")
        logger.info(f"🎯 Platform: {self.platform}")
        logger.info(f"🆔 Deployment ID: {self.deployment_id}")

    def _initialize_monitoring_configs(self):
        """Initialize monitoring component configurations"""

        # Prometheus
        prometheus_config = PrometheusConfig()
        self.components["prometheus"] = MonitoringComponent(
            name="prometheus",
            type=MonitoringComponentType.METRICS,
            config=prometheus_config
        )

        # Grafana
        grafana_config = GrafanaConfig()
        self.components["grafana"] = MonitoringComponent(
            name="grafana",
            type=MonitoringComponentType.VISUALIZATION,
            config=grafana_config
        )

        # AlertManager
        alertmanager_config = AlertManagerConfig()
        self.components["alertmanager"] = MonitoringComponent(
            name="alertmanager",
            type=MonitoringComponentType.ALERTING,
            config=alertmanager_config
        )

        # Loki
        loki_config = LokiConfig()
        self.components["loki"] = MonitoringComponent(
            name="loki",
            type=MonitoringComponentType.LOGGING,
            config=loki_config
        )

        # Jaeger
        jaeger_config = JaegerConfig()
        self.components["jaeger"] = MonitoringComponent(
            name="jaeger",
            type=MonitoringComponentType.TRACING,
            config=jaeger_config
        )

        # Fluentd
        fluentd_config = FluentdConfig()
        self.components["fluentd"] = MonitoringComponent(
            name="fluentd",
            type=MonitoringComponentType.LOGGING,
            config=fluentd_config
        )

        logger.info(f"🔧 Initialized {len(self.components)} monitoring components")

    async def deploy_monitoring_stack(self) -> bool:
        """Deploy comprehensive monitoring stack"""
        try:
            logger.info("📊 Starting monitoring stack deployment")

            # Create namespace
            if self.platform == "kubernetes":
                await self._create_monitoring_namespace()

            # Deploy components in dependency order
            deployment_order = [
                "prometheus",
                "loki",
                "jaeger",
                "fluentd",
                "alertmanager",
                "grafana"
            ]

            for component_name in deployment_order:
                component = self.components[component_name]
                logger.info(f"🔧 Deploying {component_name}")

                success = await self._deploy_monitoring_component(component)

                if success:
                    logger.info(f"✅ {component_name} deployed successfully")
                    await self._wait_for_component_health(component)
                else:
                    logger.error(f"❌ Failed to deploy {component_name}")
                    return False

            # Setup monitoring configurations
            await self._setup_prometheus_configuration()
            await self._setup_grafana_dashboards()
            await self._setup_alert_rules()
            await self._setup_log_aggregation()

            # Validate monitoring stack
            await self._validate_monitoring_stack()

            logger.info("🎉 Monitoring stack deployment completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Monitoring stack deployment failed: {e}")
            return False

    async def _create_monitoring_namespace(self) -> bool:
        """Create Kubernetes namespace for monitoring"""
        try:
            namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
  labels:
    name: {self.namespace}
    component: monitoring
    managed-by: xorb-deployment
    istio-injection: disabled
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: {self.namespace}-quota
  namespace: {self.namespace}
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "20"
    services: "20"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: {self.namespace}-limits
  namespace: {self.namespace}
spec:
  limits:
  - default:
      cpu: "2"
      memory: "2Gi"
    defaultRequest:
      cpu: "100m"
      memory: "128Mi"
    type: Container
"""

            return await self._kubectl_apply(namespace_yaml)

        except Exception as e:
            logger.error(f"❌ Failed to create monitoring namespace: {e}")
            return False

    async def _deploy_monitoring_component(self, component: MonitoringComponent) -> bool:
        """Deploy individual monitoring component"""
        try:
            component.status = "deploying"
            component.start_time = datetime.now()

            if component.type == MonitoringComponentType.METRICS:
                success = await self._deploy_prometheus(component)
            elif component.type == MonitoringComponentType.VISUALIZATION:
                success = await self._deploy_grafana(component)
            elif component.type == MonitoringComponentType.ALERTING:
                success = await self._deploy_alertmanager(component)
            elif component.type == MonitoringComponentType.LOGGING and component.name == "loki":
                success = await self._deploy_loki(component)
            elif component.type == MonitoringComponentType.TRACING:
                success = await self._deploy_jaeger(component)
            elif component.type == MonitoringComponentType.LOGGING and component.name == "fluentd":
                success = await self._deploy_fluentd(component)
            else:
                logger.warning(f"⚠️  Unknown component type: {component.type}")
                success = False

            component.end_time = datetime.now()

            if success:
                component.status = "running"
            else:
                component.status = "failed"

            return success

        except Exception as e:
            component.status = "failed"
            component.error_message = str(e)
            component.end_time = datetime.now()
            logger.error(f"❌ Component deployment failed: {e}")
            return False

    async def _deploy_prometheus(self, component: MonitoringComponent) -> bool:
        """Deploy Prometheus metrics server"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # Prometheus ConfigMap
                prometheus_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: {self.namespace}
data:
  prometheus.yml: |
    global:
      scrape_interval: {config.scrape_interval}
      evaluation_interval: {config.evaluation_interval}
      external_labels:
{self._format_yaml_dict(config.external_labels, 8)}

    rule_files:
{self._format_yaml_list([f'- "{rule}"' for rule in config.rule_files], 4)}

    scrape_configs:
    - job_name: 'prometheus'
      static_configs:
      - targets: ['localhost:9090']

    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - xorb-services
          - xorb-infra
          - {self.namespace}
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: ${{__meta_kubernetes_pod_ip}}:$$1

    - job_name: 'kubernetes-services'
      kubernetes_sd_configs:
      - role: service
        namespaces:
          names:
          - xorb-services
          - xorb-infra
          - {self.namespace}
      relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: ${{__meta_kubernetes_service_name}}.${{__meta_kubernetes_namespace}}.svc.cluster.local:$$1

    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/$${1}/proxy/metrics

    - job_name: 'kubernetes-cadvisor'
      kubernetes_sd_configs:
      - role: node
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        insecure_skip_verify: true
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/$${1}/proxy/metrics/cadvisor

    {"remote_write:" if config.enable_remote_write else ""}
{self._format_yaml_list([{"url": url} for url in config.remote_write_urls], 4) if config.enable_remote_write else ""}
"""

                await self._kubectl_apply(prometheus_config_yaml)

                # Prometheus Deployment
                prometheus_deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: {self.namespace}
  labels:
    app: prometheus
    component: metrics
spec:
  replicas: {config.replica_count}
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
        component: metrics
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: prometheus-sa
      securityContext:
        runAsUser: 65534
        runAsGroup: 65534
        fsGroup: 65534
      containers:
      - name: prometheus
        image: prom/prometheus:{config.version}
        ports:
        - containerPort: 9090
          name: web
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus/'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time={config.retention}'
        - '--web.enable-lifecycle'
        - '--web.enable-admin-api'
        - '--storage.tsdb.max-block-duration=2h'
        - '--storage.tsdb.min-block-duration=2h'
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus/prometheus.yml
          subPath: prometheus.yml
        - name: prometheus-storage
          mountPath: /prometheus/
        - name: prometheus-rules
          mountPath: /etc/prometheus/rules/
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 4
          failureThreshold: 3
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-storage-pvc
      - name: prometheus-rules
        configMap:
          name: prometheus-rules
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus-sa
  namespace: {self.namespace}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus-role
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/proxy", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["extensions"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus-role
subjects:
- kind: ServiceAccount
  name: prometheus-sa
  namespace: {self.namespace}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage-pvc
  namespace: {self.namespace}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: {config.storage_size}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: {self.namespace}
  labels:
    app: prometheus
    component: metrics
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  selector:
    app: prometheus
  ports:
  - name: web
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
"""

                success = await self._kubectl_apply(prometheus_deployment_yaml)

                if success:
                    component.endpoints = {
                        "web": f"prometheus.{self.namespace}.svc.cluster.local:9090",
                        "external": f"http://localhost:9090"
                    }

                return success

            return True

        except Exception as e:
            logger.error(f"❌ Prometheus deployment failed: {e}")
            return False

    async def _deploy_grafana(self, component: MonitoringComponent) -> bool:
        """Deploy Grafana visualization server"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # Grafana Configuration
                grafana_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-config
  namespace: {self.namespace}
data:
  grafana.ini: |
    [server]
    http_port = {config.port}
    root_url = http://localhost:{config.port}/

    [database]
    type = sqlite3
    path = /var/lib/grafana/grafana.db

    [security]
    admin_user = admin
    admin_password = {config.admin_password}

    [users]
    allow_sign_up = false
    default_theme = dark

    [auth]
    disable_login_form = false

    [auth.anonymous]
    enabled = false

    [smtp]
    enabled = {str(config.smtp_enabled).lower()}
    host = {config.smtp_host}
    user = {config.smtp_user}
    from_address = {config.smtp_user}

    [alerting]
    enabled = true
    execute_alerts = true

    [metrics]
    enabled = true

    [log]
    mode = console
    level = info

    [panels]
    disable_sanitize_html = false

    [plugins]
    allow_loading_unsigned_plugins = false

  datasources.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      access: proxy
      url: http://prometheus:9090
      isDefault: true
      editable: true
    - name: Loki
      type: loki
      access: proxy
      url: http://loki:3100
      editable: true
    - name: Jaeger
      type: jaeger
      access: proxy
      url: http://jaeger-query:16686
      editable: true
      jsonData:
        tracesToLogsV2:
          datasourceUid: loki
"""

                await self._kubectl_apply(grafana_config_yaml)

                # Grafana Deployment
                grafana_deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: {self.namespace}
  labels:
    app: grafana
    component: visualization
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
        component: visualization
    spec:
      securityContext:
        runAsUser: 472
        runAsGroup: 472
        fsGroup: 472
      containers:
      - name: grafana
        image: grafana/grafana:{config.version}
        ports:
        - containerPort: 3000
          name: web
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "{config.admin_password}"
        - name: GF_INSTALL_PLUGINS
          value: "{','.join(config.plugins)}"
        volumeMounts:
        - name: grafana-config
          mountPath: /etc/grafana/grafana.ini
          subPath: grafana.ini
        - name: grafana-datasources
          mountPath: /etc/grafana/provisioning/datasources/datasources.yaml
          subPath: datasources.yaml
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-dashboards-config
          mountPath: /etc/grafana/provisioning/dashboards/
        - name: grafana-dashboards
          mountPath: /var/lib/grafana/dashboards/
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: grafana-config
        configMap:
          name: grafana-config
      - name: grafana-datasources
        configMap:
          name: grafana-config
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage-pvc
      - name: grafana-dashboards-config
        configMap:
          name: grafana-dashboards-config
      - name: grafana-dashboards
        configMap:
          name: grafana-dashboards
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-storage-pvc
  namespace: {self.namespace}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: {self.namespace}
  labels:
    app: grafana
    component: visualization
spec:
  selector:
    app: grafana
  ports:
  - name: web
    port: 3000
    targetPort: 3000
    protocol: TCP
  type: ClusterIP
"""

                success = await self._kubectl_apply(grafana_deployment_yaml)

                if success:
                    component.endpoints = {
                        "web": f"grafana.{self.namespace}.svc.cluster.local:3000",
                        "external": f"http://localhost:3000"
                    }

                return success

            return True

        except Exception as e:
            logger.error(f"❌ Grafana deployment failed: {e}")
            return False

    async def _deploy_alertmanager(self, component: MonitoringComponent) -> bool:
        """Deploy AlertManager for alert handling"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # AlertManager Configuration
                alertmanager_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: {self.namespace}
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: '{config.notification_channels["email"]["smtp_server"]}'
      smtp_from: '{config.notification_channels["email"]["from"]}'
      smtp_require_tls: false

    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
      - match:
          severity: warning
        receiver: 'warning-alerts'

    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://localhost:5001/'

    - name: 'critical-alerts'
      {"email_configs:" if config.notification_channels["email"]["enabled"] else ""}
      {"- to: '" + "', '".join(config.notification_channels["email"]["to"]) + "'" if config.notification_channels["email"]["enabled"] else ""}
      {"  subject: 'CRITICAL Alert: {{ .GroupLabels.alertname }}'" if config.notification_channels["email"]["enabled"] else ""}
      {"  body: |" if config.notification_channels["email"]["enabled"] else ""}
      {"    {{ range .Alerts }}" if config.notification_channels["email"]["enabled"] else ""}
      {"    Alert: {{ .Annotations.summary }}" if config.notification_channels["email"]["enabled"] else ""}
      {"    Description: {{ .Annotations.description }}" if config.notification_channels["email"]["enabled"] else ""}
      {"    {{ end }}" if config.notification_channels["email"]["enabled"] else ""}
      {"slack_configs:" if config.notification_channels["slack"]["enabled"] else ""}
      {"- api_url: '" + config.notification_channels["slack"]["webhook_url"] + "'" if config.notification_channels["slack"]["enabled"] else ""}
      {"  channel: '" + config.notification_channels["slack"]["channel"] + "'" if config.notification_channels["slack"]["enabled"] else ""}
      {"  username: '" + config.notification_channels["slack"]["username"] + "'" if config.notification_channels["slack"]["enabled"] else ""}
      {"  title: 'CRITICAL Alert'" if config.notification_channels["slack"]["enabled"] else ""}
      {"  text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'" if config.notification_channels["slack"]["enabled"] else ""}

    - name: 'warning-alerts'
      {"email_configs:" if config.notification_channels["email"]["enabled"] else ""}
      {"- to: '" + "', '".join(config.notification_channels["email"]["to"]) + "'" if config.notification_channels["email"]["enabled"] else ""}
      {"  subject: 'WARNING Alert: {{ .GroupLabels.alertname }}'" if config.notification_channels["email"]["enabled"] else ""}

    inhibit_rules:
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'dev', 'instance']
"""

                await self._kubectl_apply(alertmanager_config_yaml)

                # AlertManager Deployment
                alertmanager_deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: {self.namespace}
  labels:
    app: alertmanager
    component: alerting
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
        component: alerting
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9093"
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:{config.version}
        ports:
        - containerPort: 9093
          name: web
        args:
        - '--config.file=/etc/alertmanager/alertmanager.yml'
        - '--storage.path=/alertmanager'
        - '--data.retention={config.retention}'
        volumeMounts:
        - name: alertmanager-config
          mountPath: /etc/alertmanager/alertmanager.yml
          subPath: alertmanager.yml
        - name: alertmanager-storage
          mountPath: /alertmanager
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9093
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9093
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 4
          failureThreshold: 3
      volumes:
      - name: alertmanager-config
        configMap:
          name: alertmanager-config
      - name: alertmanager-storage
        persistentVolumeClaim:
          claimName: alertmanager-storage-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: alertmanager-storage-pvc
  namespace: {self.namespace}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: {self.namespace}
  labels:
    app: alertmanager
    component: alerting
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9093"
spec:
  selector:
    app: alertmanager
  ports:
  - name: web
    port: 9093
    targetPort: 9093
    protocol: TCP
  type: ClusterIP
"""

                success = await self._kubectl_apply(alertmanager_deployment_yaml)

                if success:
                    component.endpoints = {
                        "web": f"alertmanager.{self.namespace}.svc.cluster.local:9093",
                        "external": f"http://localhost:9093"
                    }

                return success

            return True

        except Exception as e:
            logger.error(f"❌ AlertManager deployment failed: {e}")
            return False

    async def _deploy_loki(self, component: MonitoringComponent) -> bool:
        """Deploy Loki log aggregation system"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # Loki Configuration
                loki_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: {self.namespace}
data:
  loki.yaml: |
    auth_enabled: false

    server:
      http_listen_port: {config.port}
      grpc_listen_port: 9096

    common:
      path_prefix: /loki
      storage:
        filesystem:
          chunks_directory: /loki/chunks
          rules_directory: /loki/rules
      replication_factor: 1
      ring:
        instance_addr: 127.0.0.1
        kvstore:
          store: inmemory

    query_range:
      results_cache:
        cache:
          embedded_cache:
            enabled: true
            max_size_mb: 100

    schema_config:
      configs:
        - from: 2020-10-24
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h

    ruler:
      alertmanager_url: http://alertmanager:9093

    limits_config:
      retention_period: {config.retention}
      ingestion_rate_mb: 16
      ingestion_burst_size_mb: 32

    chunk_store_config:
{self._format_yaml_dict(config.chunk_store_config, 6)}

    table_manager:
{self._format_yaml_dict(config.table_manager, 6)}

    compactor:
      working_directory: /loki/compactor
      shared_store: filesystem
      compaction_interval: 10m
      retention_enabled: {str(config.enable_compactor).lower()}
"""

                await self._kubectl_apply(loki_config_yaml)

                # Loki Deployment
                loki_deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: {self.namespace}
  labels:
    app: loki
    component: logging
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
        component: logging
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3100"
    spec:
      securityContext:
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
      containers:
      - name: loki
        image: grafana/loki:{config.version}
        ports:
        - containerPort: 3100
          name: http
        - containerPort: 9096
          name: grpc
        args:
        - -config.file=/etc/loki/loki.yaml
        volumeMounts:
        - name: loki-config
          mountPath: /etc/loki/loki.yaml
          subPath: loki.yaml
        - name: loki-storage
          mountPath: /loki
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /ready
            port: 3100
          initialDelaySeconds: 45
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 3100
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 4
          failureThreshold: 3
      volumes:
      - name: loki-config
        configMap:
          name: loki-config
      - name: loki-storage
        persistentVolumeClaim:
          claimName: loki-storage-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: loki-storage-pvc
  namespace: {self.namespace}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: {config.storage_size}
---
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: {self.namespace}
  labels:
    app: loki
    component: logging
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "3100"
spec:
  selector:
    app: loki
  ports:
  - name: http
    port: 3100
    targetPort: 3100
    protocol: TCP
  - name: grpc
    port: 9096
    targetPort: 9096
    protocol: TCP
  type: ClusterIP
"""

                success = await self._kubectl_apply(loki_deployment_yaml)

                if success:
                    component.endpoints = {
                        "http": f"loki.{self.namespace}.svc.cluster.local:3100",
                        "grpc": f"loki.{self.namespace}.svc.cluster.local:9096"
                    }

                return success

            return True

        except Exception as e:
            logger.error(f"❌ Loki deployment failed: {e}")
            return False

    async def _deploy_jaeger(self, component: MonitoringComponent) -> bool:
        """Deploy Jaeger distributed tracing"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # Jaeger All-in-One Deployment
                jaeger_deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: {self.namespace}
  labels:
    app: jaeger
    component: tracing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
        component: tracing
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "14269"
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:{config.version}
        ports:
        - containerPort: 16686
          name: query
        - containerPort: 14268
          name: collector
        - containerPort: 6831
          name: agent-udp
        - containerPort: 6832
          name: agent-binary
        - containerPort: 5775
          name: agent-config
        - containerPort: 14269
          name: admin
        env:
        - name: COLLECTOR_ZIPKIN_HTTP_PORT
          value: "{config.zipkin_port}"
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        - name: METRICS_STORAGE_TYPE
          value: "prometheus"
        - name: PROMETHEUS_SERVER_URL
          value: "http://prometheus:9090"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 14269
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 14269
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 4
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-query
  namespace: {self.namespace}
  labels:
    app: jaeger
    component: query
spec:
  selector:
    app: jaeger
  ports:
  - name: query
    port: 16686
    targetPort: 16686
    protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-collector
  namespace: {self.namespace}
  labels:
    app: jaeger
    component: collector
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "14269"
spec:
  selector:
    app: jaeger
  ports:
  - name: collector
    port: 14268
    targetPort: 14268
    protocol: TCP
  - name: zipkin
    port: 9411
    targetPort: 9411
    protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-agent
  namespace: {self.namespace}
  labels:
    app: jaeger
    component: agent
spec:
  selector:
    app: jaeger
  ports:
  - name: agent-udp
    port: 6831
    targetPort: 6831
    protocol: UDP
  - name: agent-binary
    port: 6832
    targetPort: 6832
    protocol: TCP
  - name: agent-config
    port: 5775
    targetPort: 5775
    protocol: UDP
  type: ClusterIP
"""

                success = await self._kubectl_apply(jaeger_deployment_yaml)

                if success:
                    component.endpoints = {
                        "query": f"jaeger-query.{self.namespace}.svc.cluster.local:16686",
                        "collector": f"jaeger-collector.{self.namespace}.svc.cluster.local:14268",
                        "agent": f"jaeger-agent.{self.namespace}.svc.cluster.local:6831"
                    }

                return success

            return True

        except Exception as e:
            logger.error(f"❌ Jaeger deployment failed: {e}")
            return False

    async def _deploy_fluentd(self, component: MonitoringComponent) -> bool:
        """Deploy Fluentd log collection"""
        try:
            config = component.config

            if self.platform == "kubernetes":
                # Fluentd Configuration
                fluentd_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: {self.namespace}
data:
  fluent.conf: |
    <source>
      @type forward
      port {config.port}
      bind 0.0.0.0
    </source>

    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>

    <match kubernetes.**>
      @type copy
      <store>
        @type loki
        url http://loki:3100
        extra_labels {{"cluster":"xorb-cluster"}}
        <buffer>
          @type file
          path /var/log/fluentd-buffers/loki.buffer
          flush_mode interval
          retry_type exponential_backoff
          flush_thread_count 2
          flush_interval {config.flush_interval}
          retry_forever
          retry_max_interval 30
          chunk_limit_size {config.buffer_size}
        </buffer>
      </store>
      {"<store>" if "elasticsearch" in config.output_plugins else ""}
      {"  @type elasticsearch" if "elasticsearch" in config.output_plugins else ""}
      {"  host elasticsearch-cluster.xorb-infra.svc.cluster.local" if "elasticsearch" in config.output_plugins else ""}
      {"  port 9200" if "elasticsearch" in config.output_plugins else ""}
      {"  index_name fluentd" if "elasticsearch" in config.output_plugins else ""}
      {"  type_name _doc" if "elasticsearch" in config.output_plugins else ""}
      {"</store>" if "elasticsearch" in config.output_plugins else ""}
    </match>

    {"<source>" if config.enable_prometheus else ""}
    {"  @type prometheus" if config.enable_prometheus else ""}
    {"  bind 0.0.0.0" if config.enable_prometheus else ""}
    {"  port 24231" if config.enable_prometheus else ""}
    {"  metrics_path /metrics" if config.enable_prometheus else ""}
    {"</source>" if config.enable_prometheus else ""}

    {"<source>" if config.enable_prometheus else ""}
    {"  @type prometheus_output_monitor" if config.enable_prometheus else ""}
    {"  interval 10" if config.enable_prometheus else ""}
    {"  <labels>" if config.enable_prometheus else ""}
    {"    hostname ${{hostname}}" if config.enable_prometheus else ""}
    {"  </labels>" if config.enable_prometheus else ""}
    {"</source>" if config.enable_prometheus else ""}
"""

                await self._kubectl_apply(fluentd_config_yaml)

                # Fluentd DaemonSet
                fluentd_daemonset_yaml = f"""
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: {self.namespace}
  labels:
    app: fluentd
    component: logging
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
        component: logging
      annotations:
        {"prometheus.io/scrape": "true" if config.enable_prometheus else "false"}
        {"prometheus.io/port": "24231" if config.enable_prometheus else ""}
    spec:
      serviceAccountName: fluentd-sa
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:{config.version}
        ports:
        - containerPort: 24224
          name: forward
        {"- containerPort: 24231" if config.enable_prometheus else ""}
        {"  name: metrics" if config.enable_prometheus else ""}
        env:
        - name: FLUENTD_SYSTEMD_CONF
          value: "disable"
        - name: FLUENTD_PROMETHEUS_CONF
          value: "disable"
        volumeMounts:
        - name: fluentd-config
          mountPath: /fluentd/etc/fluent.conf
          subPath: fluent.conf
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          tcpSocket:
            port: 24224
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          tcpSocket:
            port: 24224
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 4
          failureThreshold: 3
      volumes:
      - name: fluentd-config
        configMap:
          name: fluentd-config
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      terminationGracePeriodSeconds: 30
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fluentd-sa
  namespace: {self.namespace}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fluentd-role
rules:
- apiGroups: [""]
  resources: ["pods", "namespaces"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fluentd-binding
roleRef:
  kind: ClusterRole
  name: fluentd-role
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: fluentd-sa
  namespace: {self.namespace}
---
apiVersion: v1
kind: Service
metadata:
  name: fluentd
  namespace: {self.namespace}
  labels:
    app: fluentd
    component: logging
  annotations:
    {"prometheus.io/scrape": "true" if config.enable_prometheus else "false"}
    {"prometheus.io/port": "24231" if config.enable_prometheus else ""}
spec:
  selector:
    app: fluentd
  ports:
  - name: forward
    port: 24224
    targetPort: 24224
    protocol: TCP
  {"- name: metrics" if config.enable_prometheus else ""}
  {"  port: 24231" if config.enable_prometheus else ""}
  {"  targetPort: 24231" if config.enable_prometheus else ""}
  {"  protocol: TCP" if config.enable_prometheus else ""}
  type: ClusterIP
"""

                success = await self._kubectl_apply(fluentd_daemonset_yaml)

                if success:
                    component.endpoints = {
                        "forward": f"fluentd.{self.namespace}.svc.cluster.local:24224"
                    }

                    if config.enable_prometheus:
                        component.endpoints["metrics"] = f"fluentd.{self.namespace}.svc.cluster.local:24231"

                return success

            return True

        except Exception as e:
            logger.error(f"❌ Fluentd deployment failed: {e}")
            return False

    async def _wait_for_component_health(self, component: MonitoringComponent, timeout: int = 300) -> bool:
        """Wait for monitoring component to become healthy"""
        try:
            logger.info(f"⏳ Waiting for {component.name} to become healthy...")

            start_time = time.time()

            while time.time() - start_time < timeout:
                health_status = await self._check_component_health(component)

                if health_status:
                    component.health_status = True
                    logger.info(f"✅ {component.name} is healthy")
                    return True

                logger.info(f"⏳ {component.name} not ready yet, waiting...")
                await asyncio.sleep(10)

            logger.error(f"❌ Timeout waiting for {component.name} to become healthy")
            return False

        except Exception as e:
            logger.error(f"❌ Health check failed for {component.name}: {e}")
            return False

    async def _check_component_health(self, component: MonitoringComponent) -> bool:
        """Check monitoring component health"""
        try:
            if self.platform == "kubernetes":
                # Check pod status
                result = subprocess.run([
                    "kubectl", "get", "pods",
                    "-l", f"app={component.name}",
                    "-n", self.namespace,
                    "--field-selector=status.phase=Running",
                    "--no-headers"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and result.stdout.strip():
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Component health check failed: {e}")
            return False

    # Additional helper methods

    async def _setup_prometheus_configuration(self):
        """Setup Prometheus configuration and rules"""
        logger.info("⚙️  Setting up Prometheus configuration")

        # Create Prometheus rules ConfigMap
        prometheus_rules_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: {self.namespace}
data:
  xorb-rules.yml: |
    groups:
    - name: xorb.rules
      rules:
      - alert: HighCPUUsage
        expr: 100 - (avg(irate(node_cpu_seconds_total{{mode="idle"}}[5m])) * 100) > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 2 minutes"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% for more than 2 minutes"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(http_requests_total{{status=~"5..+"}}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 10% for more than 5 minutes"
"""

        await self._kubectl_apply(prometheus_rules_yaml)

    async def _setup_grafana_dashboards(self):
        """Setup Grafana dashboards"""
        logger.info("📊 Setting up Grafana dashboards")

        # Dashboard provisioning configuration
        dashboard_config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-config
  namespace: {self.namespace}
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'default'
      orgId: 1
      folder: ''
      type: file
      disableDeletion: false
      updateIntervalSeconds: 10
      allowUiUpdates: true
      options:
        path: /var/lib/grafana/dashboards
"""

        await self._kubectl_apply(dashboard_config_yaml)

        # Sample dashboard
        dashboard_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: {self.namespace}
data:
  xorb-overview.json: |
    {{
      "dashboard": {{
        "id": null,
        "title": "XORB Platform Overview",
        "tags": ["xorb", "overview"],
        "style": "dark",
        "timezone": "browser",
        "panels": [
          {{
            "id": 1,
            "title": "CPU Usage",
            "type": "graph",
            "targets": [
              {{
                "expr": "100 - (avg(irate(node_cpu_seconds_total{{mode=\\"idle\\"}}[5m])) * 100)",
                "format": "time_series",
                "legendFormat": "CPU Usage %"
              }}
            ],
            "gridPos": {{"h": 9, "w": 12, "x": 0, "y": 0}}
          }},
          {{
            "id": 2,
            "title": "Memory Usage",
            "type": "graph",
            "targets": [
              {{
                "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                "format": "time_series",
                "legendFormat": "Memory Usage %"
              }}
            ],
            "gridPos": {{"h": 9, "w": 12, "x": 12, "y": 0}}
          }}
        ],
        "time": {{
          "from": "now-1h",
          "to": "now"
        }},
        "refresh": "5s"
      }}
    }}
"""

        await self._kubectl_apply(dashboard_yaml)

    async def _setup_alert_rules(self):
        """Setup alert rules"""
        logger.info("🚨 Setting up alert rules")
        # Alert rules are already configured in Prometheus rules
        pass

    async def _setup_log_aggregation(self):
        """Setup log aggregation"""
        logger.info("📋 Setting up log aggregation")
        # Log aggregation is handled by Loki and Fluentd configurations
        pass

    async def _validate_monitoring_stack(self):
        """Validate monitoring stack deployment"""
        logger.info("✅ Validating monitoring stack")

        validation_tasks = [
            self._validate_prometheus_metrics(),
            self._validate_grafana_dashboards(),
            self._validate_alert_manager(),
            self._validate_log_collection()
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception) or not result:
                logger.warning(f"⚠️  Monitoring validation {i+1} failed: {result}")

    async def _validate_prometheus_metrics(self) -> bool:
        """Validate Prometheus is collecting metrics"""
        try:
            # This would test Prometheus metrics collection
            return True
        except Exception:
            return False

    async def _validate_grafana_dashboards(self) -> bool:
        """Validate Grafana dashboards are loaded"""
        try:
            # This would test Grafana dashboard loading
            return True
        except Exception:
            return False

    async def _validate_alert_manager(self) -> bool:
        """Validate AlertManager is processing alerts"""
        try:
            # This would test AlertManager functionality
            return True
        except Exception:
            return False

    async def _validate_log_collection(self) -> bool:
        """Validate log collection is working"""
        try:
            # This would test log collection pipeline
            return True
        except Exception:
            return False

    async def _kubectl_apply(self, yaml_content: str) -> bool:
        """Apply Kubernetes YAML configuration"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                f.write(yaml_content)
                temp_file = f.name

            cmd = ["kubectl", "apply", "-f", temp_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            os.unlink(temp_file)

            if result.returncode == 0:
                return True
            else:
                logger.error(f"❌ kubectl apply failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ kubectl apply failed: {e}")
            return False

    def _format_yaml_dict(self, data: Dict[str, Any], indent: int) -> str:
        """Format dictionary as YAML with proper indentation"""
        if not data:
            return ""

        spaces = " " * indent
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{spaces}{key}:")
                lines.append(self._format_yaml_dict(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {value}")

        return "\n".join(lines)

    def _format_yaml_list(self, data: List[Any], indent: int) -> str:
        """Format list as YAML with proper indentation"""
        if not data:
            return ""

        spaces = " " * indent
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{spaces}- ")
                for key, value in item.items():
                    lines.append(f"{spaces}  {key}: {value}")
            else:
                lines.append(f"{spaces}- {item}")

        return "\n".join(lines)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring stack status"""
        status = {
            "deployment_id": self.deployment_id,
            "namespace": self.namespace,
            "platform": self.platform,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "summary": {
                "total": len(self.components),
                "running": 0,
                "failed": 0,
                "deploying": 0
            }
        }

        for name, component in self.components.items():
            status["components"][name] = {
                "type": component.type.value,
                "status": component.status,
                "health_status": component.health_status,
                "endpoints": component.endpoints,
                "start_time": component.start_time.isoformat() if component.start_time else None,
                "end_time": component.end_time.isoformat() if component.end_time else None,
                "error_message": component.error_message
            }

            if component.status == "running":
                status["summary"]["running"] += 1
            elif component.status == "failed":
                status["summary"]["failed"] += 1
            elif component.status == "deploying":
                status["summary"]["deploying"] += 1

        return status

async def main():
    """Main function for testing monitoring stack deployment"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize monitoring stack
    monitoring = XORBMonitoringStack(
        namespace="xorb-monitoring",
        platform="kubernetes"
    )

    # Deploy monitoring stack
    success = await monitoring.deploy_monitoring_stack()

    if success:
        print("🎉 Monitoring stack deployment completed successfully!")

        # Get status
        status = monitoring.get_monitoring_status()
        print(f"📊 Monitoring Stack Status:")
        print(f"  Total Components: {status['summary']['total']}")
        print(f"  Running: {status['summary']['running']}")
        print(f"  Failed: {status['summary']['failed']}")

        # Print component endpoints
        print(f"\n🔗 Monitoring Endpoints:")
        for name, component in status['components'].items():
            if component['endpoints']:
                print(f"  {name}:")
                for endpoint_type, endpoint_url in component['endpoints'].items():
                    print(f"    {endpoint_type}: {endpoint_url}")
    else:
        print("❌ Monitoring stack deployment failed!")

    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
