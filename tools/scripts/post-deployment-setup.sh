#!/bin/bash
set -euo pipefail

# ===================================================================
# XORB Autonomous Orchestrator - Post-Deployment Setup Script
# Configure production environment after successful deployment
# ===================================================================

# Configuration
NAMESPACE="${NAMESPACE:-xorb-system}"
DOMAIN="${DOMAIN:-xorb.local}"
ENABLE_INGRESS="${ENABLE_INGRESS:-true}"
ENABLE_TLS="${ENABLE_TLS:-true}"
SETUP_ALERTS="${SETUP_ALERTS:-true}"
CONFIGURE_BACKUPS="${CONFIGURE_BACKUPS:-true}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

step() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] 🚀${NC} $1"
}

# Check if deployment exists
check_deployment() {
    step "🔍 Verifying XORB deployment"
    
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        error "Namespace $NAMESPACE not found. Please run deployment first."
        exit 1
    fi
    
    if ! kubectl get deployment xorb-autonomous-orchestrator -n "$NAMESPACE" &>/dev/null; then
        error "XORB orchestrator deployment not found. Please run deployment first."
        exit 1
    fi
    
    local running_pods=$(kubectl get pods -n "$NAMESPACE" -l app=xorb-orchestrator --field-selector=status.phase=Running --no-headers | wc -l)
    if [ "$running_pods" -eq 0 ]; then
        error "No running XORB orchestrator pods found."
        exit 1
    fi
    
    log "✅ XORB deployment verified - $running_pods pods running"
}

# Setup external access
setup_ingress() {
    if [ "$ENABLE_INGRESS" = "false" ]; then
        warn "⏭️ Skipping ingress setup (ENABLE_INGRESS=false)"
        return 0
    fi
    
    step "🌐 Setting up external access"
    
    # Create ingress for orchestrator API
    info "📝 Creating ingress configuration"
    kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: xorb-orchestrator-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/rate-limit-rps: "10"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.$DOMAIN
    secretName: xorb-orchestrator-tls
  rules:
  - host: api.$DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: xorb-orchestrator
            port:
              number: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: xorb-metrics-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: xorb-metrics-auth
spec:
  tls:
  - hosts:
    - metrics.$DOMAIN
    secretName: xorb-metrics-tls
  rules:
  - host: metrics.$DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: xorb-orchestrator
            port:
              number: 9090
EOF
    
    # Create basic auth for metrics
    info "🔐 Setting up metrics authentication"
    htpasswd -cb /tmp/auth admin "$(openssl rand -base64 12)"
    kubectl create secret generic xorb-metrics-auth \
        --from-file=auth=/tmp/auth \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    rm -f /tmp/auth
    
    log "✅ Ingress configured for api.$DOMAIN and metrics.$DOMAIN"
}

# Setup TLS certificates
setup_tls() {
    if [ "$ENABLE_TLS" = "false" ]; then
        warn "⏭️ Skipping TLS setup (ENABLE_TLS=false)"
        return 0
    fi
    
    step "🔒 Setting up TLS certificates"
    
    # Create ClusterIssuer for Let's Encrypt
    info "📜 Creating Let's Encrypt ClusterIssuer"
    kubectl apply -f - << EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@$DOMAIN
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    # Create certificate for orchestrator
    info "🔐 Creating SSL certificate"
    kubectl apply -f - << EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: xorb-orchestrator-cert
  namespace: $NAMESPACE
spec:
  secretName: xorb-orchestrator-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.$DOMAIN
  - metrics.$DOMAIN
  - xorb-orchestrator.$NAMESPACE.svc.cluster.local
EOF
    
    log "✅ TLS certificates configured"
}

# Setup monitoring alerts
setup_alerts() {
    if [ "$SETUP_ALERTS" = "false" ]; then
        warn "⏭️ Skipping alerts setup (SETUP_ALERTS=false)"
        return 0
    fi
    
    step "🚨 Setting up monitoring alerts"
    
    # Create comprehensive alerting rules
    info "📊 Creating Prometheus alerting rules"
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: xorb-orchestrator-alerts
  namespace: $NAMESPACE
  labels:
    app: xorb-orchestrator
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
  - name: xorb.orchestrator.rules
    interval: 30s
    rules:
    - alert: XORBOrchestratorDown
      expr: up{job="xorb-orchestrator"} == 0
      for: 5m
      labels:
        severity: critical
        component: orchestrator
      annotations:
        summary: "XORB Orchestrator is down"
        description: "Orchestrator {{ \$labels.instance }} has been down for more than 5 minutes"
        runbook_url: "https://docs.xorb.company.com/runbooks/orchestrator-down"
    
    - alert: XORBHighMemoryUsage
      expr: (container_memory_usage_bytes{pod=~"xorb-autonomous-orchestrator-.*"} / container_spec_memory_limit_bytes) > 0.85
      for: 10m
      labels:
        severity: warning
        component: orchestrator
      annotations:
        summary: "XORB Orchestrator high memory usage"
        description: "Pod {{ \$labels.pod }} memory usage is above 85% for 10 minutes"
    
    - alert: XORBHighCPUUsage
      expr: (rate(container_cpu_usage_seconds_total{pod=~"xorb-autonomous-orchestrator-.*"}[5m]) / container_spec_cpu_quota * container_spec_cpu_period) > 0.8
      for: 15m
      labels:
        severity: warning
        component: orchestrator
      annotations:
        summary: "XORB Orchestrator high CPU usage"
        description: "Pod {{ \$labels.pod }} CPU usage is above 80% for 15 minutes"
    
    - alert: XORBTaskFailureRate
      expr: rate(xorb_orchestrator_tasks_failed_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
        component: tasks
      annotations:
        summary: "High task failure rate detected"
        description: "Task failure rate is {{ \$value }} per second"
    
    - alert: XORBMLModelAccuracy
      expr: xorb_ml_model_accuracy < 0.75
      for: 10m
      labels:
        severity: warning
        component: ml
      annotations:
        summary: "ML model accuracy degraded"
        description: "Model {{ \$labels.model_type }} accuracy is {{ \$value }}"
    
    - alert: XORBRedisDown
      expr: up{job="redis"} == 0
      for: 2m
      labels:
        severity: critical
        component: redis
      annotations:
        summary: "Redis is down"
        description: "Redis instance has been down for more than 2 minutes"
    
    - alert: XORBAPIErrorRate
      expr: rate(xorb_api_requests_total{status=~"5.."}[5m]) / rate(xorb_api_requests_total[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
        component: api
      annotations:
        summary: "High API error rate"
        description: "API error rate is {{ \$value | humanizePercentage }} over the last 5 minutes"
    
    - alert: XORBConsensusIssues
      expr: increase(xorb_consensus_leader_elections_failed_total[10m]) > 2
      for: 1m
      labels:
        severity: critical
        component: consensus
      annotations:
        summary: "Consensus leader election failures"
        description: "Multiple leader election failures detected"
EOF
    
    log "✅ Monitoring alerts configured"
}

# Setup automated backups
setup_backups() {
    if [ "$CONFIGURE_BACKUPS" = "false" ]; then
        warn "⏭️ Skipping backup setup (CONFIGURE_BACKUPS=false)"
        return 0
    fi
    
    step "💾 Setting up automated backups"
    
    # Create backup script ConfigMap
    info "📝 Creating backup script"
    kubectl create configmap xorb-backup-script \
        --namespace="$NAMESPACE" \
        --from-literal=backup.sh='#!/bin/bash
set -euo pipefail

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/xorb-$BACKUP_DATE"
mkdir -p "$BACKUP_DIR"

echo "Starting XORB backup at $(date)"

# Backup Redis
echo "Backing up Redis..."
redis-cli -h redis-service BGSAVE
sleep 10
redis-cli -h redis-service LASTSAVE > "$BACKUP_DIR/redis-lastsave.txt"
redis-cli -h redis-service --rdb "$BACKUP_DIR/redis-dump.rdb"

# Backup Kubernetes resources
echo "Backing up Kubernetes resources..."
kubectl get all,configmaps,secrets,pvc -n xorb-system -o yaml > "$BACKUP_DIR/k8s-resources.yaml"

# Create metadata
cat > "$BACKUP_DIR/metadata.json" << EOF
{
  "backup_date": "$(date -Iseconds)",
  "backup_type": "automated",
  "namespace": "xorb-system",
  "components": ["redis", "kubernetes-resources"]
}
EOF

# Compress backup
tar -czf "/backups/xorb-backup-$BACKUP_DATE.tar.gz" -C /backups "xorb-$BACKUP_DATE"
rm -rf "$BACKUP_DIR"

echo "Backup completed: xorb-backup-$BACKUP_DATE.tar.gz"

# Cleanup old backups (keep last 7 days)
find /backups -name "xorb-backup-*.tar.gz" -mtime +7 -delete

echo "Backup process finished at $(date)"
' --dry-run=client -o yaml | kubectl apply -f -
    
    # Create backup CronJob
    info "⏰ Creating backup CronJob"
    kubectl apply -f - << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: xorb-backup
  namespace: $NAMESPACE
  labels:
    app: xorb-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: redis:7-alpine
            command:
            - /bin/sh
            - /scripts/backup.sh
            env:
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: xorb-orchestrator-secrets
                  key: redis-password
            volumeMounts:
            - name: backup-script
              mountPath: /scripts
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-script
            configMap:
              name: xorb-backup-script
              defaultMode: 0755
          - name: backup-storage
            persistentVolumeClaim:
              claimName: xorb-backup-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: xorb-backup-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
EOF
    
    log "✅ Automated backups configured (daily at 2 AM)"
}

# Create operational scripts
create_operational_scripts() {
    step "📋 Creating operational scripts"
    
    local scripts_dir="/opt/xorb/scripts"
    mkdir -p "$scripts_dir"
    
    # Health check script
    info "🏥 Creating health check script"
    cat > "$scripts_dir/health-check.sh" << 'EOF'
#!/bin/bash
# XORB Health Check Script

NAMESPACE=${NAMESPACE:-xorb-system}

echo "🏥 XORB Autonomous Orchestrator Health Check"
echo "============================================"

# Check pods
echo "📋 Pod Status:"
kubectl get pods -n "$NAMESPACE" -o wide

# Check services
echo -e "\n🌐 Service Status:"
kubectl get services -n "$NAMESPACE"

# Check orchestrator health endpoint
echo -e "\n🤖 API Health Check:"
kubectl port-forward -n "$NAMESPACE" svc/xorb-orchestrator 8080:8080 &
PF_PID=$!
sleep 2

if curl -s -f http://localhost:8080/health >/dev/null; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
fi

kill $PF_PID 2>/dev/null || true

# Check Redis
echo -e "\n⚡ Redis Connectivity:"
REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
if kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli ping | grep -q PONG; then
    echo "✅ Redis connectivity check passed"
else
    echo "❌ Redis connectivity check failed"
fi

echo -e "\n✅ Health check completed"
EOF
    
    # Log collection script
    info "📝 Creating log collection script"
    cat > "$scripts_dir/collect-logs.sh" << 'EOF'
#!/bin/bash
# XORB Log Collection Script

NAMESPACE=${NAMESPACE:-xorb-system}
OUTPUT_DIR="/tmp/xorb-logs-$(date +%Y%m%d_%H%M%S)"

echo "📝 Collecting XORB logs..."
mkdir -p "$OUTPUT_DIR"

# Collect pod logs
echo "Collecting pod logs..."
kubectl logs -n "$NAMESPACE" -l app=xorb-orchestrator --tail=1000 > "$OUTPUT_DIR/orchestrator-logs.txt"
kubectl logs -n "$NAMESPACE" -l app=redis --tail=1000 > "$OUTPUT_DIR/redis-logs.txt"

# Collect events
echo "Collecting events..."
kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' > "$OUTPUT_DIR/events.txt"

# Collect resource status
echo "Collecting resource status..."
kubectl describe pods -n "$NAMESPACE" > "$OUTPUT_DIR/pod-descriptions.txt"
kubectl get all -n "$NAMESPACE" -o yaml > "$OUTPUT_DIR/resources.yaml"

# Create archive
tar -czf "${OUTPUT_DIR}.tar.gz" -C /tmp "$(basename "$OUTPUT_DIR")"
rm -rf "$OUTPUT_DIR"

echo "✅ Logs collected: ${OUTPUT_DIR}.tar.gz"
EOF
    
    # Performance monitoring script
    info "📊 Creating performance monitoring script"
    cat > "$scripts_dir/performance-monitor.sh" << 'EOF'
#!/bin/bash
# XORB Performance Monitoring Script

NAMESPACE=${NAMESPACE:-xorb-system}

echo "📊 XORB Performance Monitoring"
echo "=============================="

# Resource usage
echo "💾 Resource Usage:"
kubectl top pods -n "$NAMESPACE" --use-protocol-buffers

# Metrics snapshot
echo -e "\n📈 Key Metrics:"
kubectl port-forward -n "$NAMESPACE" svc/xorb-orchestrator 9090:9090 &
PF_PID=$!
sleep 2

echo "Orchestrator Status:"
curl -s http://localhost:9090/metrics | grep -E "xorb_orchestrator_(agents|tasks)_total" || echo "Metrics not available"

echo -e "\nAPI Request Rate:"
curl -s http://localhost:9090/metrics | grep -E "xorb_api_requests_total" | head -5 || echo "API metrics not available"

kill $PF_PID 2>/dev/null || true

echo -e "\n✅ Performance monitoring completed"
EOF
    
    # Make scripts executable
    chmod +x "$scripts_dir"/*.sh
    
    log "✅ Operational scripts created in $scripts_dir"
}

# Setup log aggregation
setup_logging() {
    step "📝 Setting up centralized logging"
    
    info "🔍 Creating log aggregation configuration"
    kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-fluent-bit-config
  namespace: $NAMESPACE
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf
        HTTP_Server   On
        HTTP_Listen   0.0.0.0
        HTTP_Port     2020

    [INPUT]
        Name              tail
        Path              /var/log/containers/*xorb*.log
        Parser            docker
        Tag               kube.*
        Refresh_Interval  5
        Mem_Buf_Limit     50MB
        Skip_Long_Lines   On

    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log           On
        K8S-Logging.Parser  On
        K8S-Logging.Exclude Off

    [OUTPUT]
        Name  stdout
        Match *

  parsers.conf: |
    [PARSER]
        Name   docker
        Format json
        Time_Key time
        Time_Format %Y-%m-%dT%H:%M:%S.%L
        Time_Keep   On
EOF
    
    log "✅ Logging configuration created"
}

# Generate final report
generate_post_deployment_report() {
    step "📋 Generating post-deployment report"
    
    local report_file="/var/log/xorb/post-deployment-$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
  "post_deployment_setup": {
    "timestamp": "$(date -Iseconds)",
    "domain": "$DOMAIN",
    "namespace": "$NAMESPACE"
  },
  "external_access": {
    "ingress_enabled": $ENABLE_INGRESS,
    "tls_enabled": $ENABLE_TLS,
    "api_endpoint": "https://api.$DOMAIN",
    "metrics_endpoint": "https://metrics.$DOMAIN"
  },
  "monitoring": {
    "alerts_configured": $SETUP_ALERTS,
    "prometheus_rules": "$(kubectl get prometheusrules -n $NAMESPACE --no-headers | wc -l)",
    "alert_rules_count": "$(kubectl get prometheusrules xorb-orchestrator-alerts -n $NAMESPACE -o jsonpath='{.spec.groups[0].rules}' | jq length 2>/dev/null || echo 0)"
  },
  "backup_system": {
    "automated_backups": $CONFIGURE_BACKUPS,
    "backup_schedule": "0 2 * * *",
    "retention_policy": "7 days"
  },
  "operational_tools": {
    "health_check_script": "/opt/xorb/scripts/health-check.sh",
    "log_collection_script": "/opt/xorb/scripts/collect-logs.sh",
    "performance_monitor_script": "/opt/xorb/scripts/performance-monitor.sh"
  },
  "next_steps": [
    "Configure DNS records for $DOMAIN",
    "Set up external monitoring integration",
    "Configure log shipping to external systems",
    "Set up notification channels for alerts",
    "Review and customize alert thresholds",
    "Set up disaster recovery procedures"
  ]
}
EOF
    
    log "✅ Post-deployment report generated: $report_file"
    echo "$report_file"
}

# Print summary
print_summary() {
    echo ""
    echo "==============================================="
    echo -e "${CYAN}🎉 XORB POST-DEPLOYMENT SETUP COMPLETE${NC}"
    echo "==============================================="
    echo ""
    echo -e "${GREEN}✅ CONFIGURATION COMPLETED:${NC}"
    echo ""
    echo -e "${BLUE}🌐 External Access:${NC}"
    if [ "$ENABLE_INGRESS" = "true" ]; then
        echo "  📡 API Endpoint: https://api.$DOMAIN"
        echo "  📊 Metrics Endpoint: https://metrics.$DOMAIN"
        echo "  🔐 Authentication: Basic auth for metrics"
    else
        echo "  📡 Access via port-forward only"
    fi
    echo ""
    echo -e "${BLUE}🔒 Security:${NC}"
    if [ "$ENABLE_TLS" = "true" ]; then
        echo "  🔐 TLS/SSL: Let's Encrypt certificates configured"
        echo "  📜 Certificate Issuer: letsencrypt-prod"
    else
        echo "  ⚠️  TLS/SSL: Not configured"
    fi
    echo ""
    echo -e "${BLUE}🚨 Monitoring:${NC}"
    if [ "$SETUP_ALERTS" = "true" ]; then
        echo "  📊 Prometheus Rules: Configured"
        echo "  🔔 Alert Categories: System, Performance, Security"
        echo "  ⚠️  Threshold Monitoring: CPU, Memory, Error Rates"
    else
        echo "  📊 Alerts: Not configured"
    fi
    echo ""
    echo -e "${BLUE}💾 Backups:${NC}"
    if [ "$CONFIGURE_BACKUPS" = "true" ]; then
        echo "  ⏰ Schedule: Daily at 2:00 AM"
        echo "  📦 Components: Redis data, Kubernetes resources"
        echo "  🔄 Retention: 7 days"
    else
        echo "  💾 Automated backups: Not configured"
    fi
    echo ""
    echo -e "${BLUE}🛠️ Operational Tools:${NC}"
    echo "  🏥 Health Check: /opt/xorb/scripts/health-check.sh"
    echo "  📝 Log Collection: /opt/xorb/scripts/collect-logs.sh"
    echo "  📊 Performance Monitor: /opt/xorb/scripts/performance-monitor.sh"
    echo ""
    echo -e "${BLUE}📋 Useful Commands:${NC}"
    echo "  🏥 Health Check: /opt/xorb/scripts/health-check.sh"
    echo "  📊 View Metrics: kubectl port-forward -n $NAMESPACE svc/xorb-orchestrator 9090:9090"
    echo "  📝 Collect Logs: /opt/xorb/scripts/collect-logs.sh"
    echo "  🔍 View Alerts: kubectl get prometheusrules -n $NAMESPACE"
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT NEXT STEPS:${NC}"
    echo "  1. Configure DNS records for $DOMAIN to point to your ingress"
    echo "  2. Set up notification channels (Slack, email, PagerDuty)"
    echo "  3. Configure external log shipping if needed"
    echo "  4. Review and customize alert thresholds"
    echo "  5. Test backup and restore procedures"
    echo "  6. Set up monitoring dashboards"
    echo ""
    echo -e "${GREEN}🎯 XORB is now fully configured for production operations!${NC}"
    echo "==============================================="
}

# Main execution
main() {
    echo -e "${CYAN}🚀 XORB Autonomous Orchestrator - Post-Deployment Setup${NC}"
    echo "======================================================"
    echo ""
    
    check_deployment
    setup_ingress
    setup_tls
    setup_alerts
    setup_backups
    create_operational_scripts
    setup_logging
    
    local report_file=$(generate_post_deployment_report)
    print_summary
    
    echo ""
    log "🎉 Post-deployment setup completed successfully!"
    log "📋 Detailed report: $report_file"
}

# Execute main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi