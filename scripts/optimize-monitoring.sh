#!/bin/bash

# XORB Monitoring Stack Optimization Script
# Optimizes Prometheus/Grafana connectivity and performance

set -euo pipefail

echo "🔧 XORB Monitoring Stack Optimization"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Stop existing monitoring services
echo "🛑 Stopping monitoring services..."
docker stop xorb_prometheus_1 xorb_grafana_1 2>/dev/null || true

# Create optimized Prometheus configuration
echo "📊 Creating optimized Prometheus configuration..."
mkdir -p monitoring/optimized

cat > monitoring/optimized/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'xorb-production'
    environment: 'production'

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    
  - job_name: 'xorb-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 10s
    
  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['host.docker.internal:8080']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'xorb-worker'
    static_configs:
      - targets: ['host.docker.internal:9000']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

storage:
  tsdb:
    retention.time: 15d
    retention.size: 10GB
    wal-compression: true
EOF

# Create alert rules
mkdir -p monitoring/optimized/rules
cat > monitoring/optimized/rules/xorb-alerts.yml << 'EOF'
groups:
  - name: xorb-platform
    rules:
      - alert: XORBServiceDown
        expr: up{job=~"xorb-.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "XORB service {{ $labels.job }} is down"
          description: "Service {{ $labels.job }} has been down for more than 1 minute"
          
      - alert: XORBHighResponseTime
        expr: http_request_duration_seconds{job=~"xorb-.*"} > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time for {{ $labels.job }}"
          description: "Response time is {{ $value }}s for {{ $labels.job }}"
          
      - alert: XORBHighMemoryUsage
        expr: process_resident_memory_bytes{job=~"xorb-.*"} / 1024 / 1024 > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage for {{ $labels.job }}"
          description: "Memory usage is {{ $value }}MB for {{ $labels.job }}"
EOF

# Create optimized Grafana configuration
echo "📈 Creating optimized Grafana configuration..."
mkdir -p monitoring/optimized/grafana/provisioning/{datasources,dashboards}

cat > monitoring/optimized/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: 15s
      queryTimeout: 60s
      httpMethod: POST
EOF

cat > monitoring/optimized/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'XORB Dashboards'
    orgId: 1
    folder: 'XORB Platform'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create optimized Docker Compose override
echo "🐳 Creating optimized monitoring stack..."
cat > docker-compose.monitoring-optimized.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: xorb_prometheus_optimized
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/optimized/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/optimized/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--storage.tsdb.retention.time=15d'
      - '--storage.tsdb.wal-compression'
    networks:
      - xorb_network
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  grafana:
    image: grafana/grafana:10.0.0
    container_name: xorb_grafana_optimized
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/optimized/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=xorb_admin_secure_2024
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
      - GF_FEATURE_TOGGLES_ENABLE=ngalert
      - GF_ALERTING_ENABLED=true
      - GF_UNIFIED_ALERTING_ENABLED=true
      - GF_SERVER_ROOT_URL=http://localhost:3000
    networks:
      - xorb_network
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  xorb_network:
    external: true
EOF

# Start optimized monitoring stack
echo "🚀 Starting optimized monitoring stack..."
docker-compose -f docker-compose.monitoring-optimized.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Test connectivity
echo "🔍 Testing connectivity..."
for i in {1..12}; do
    if curl -s http://localhost:9090/-/healthy >/dev/null; then
        log_info "Prometheus is healthy"
        break
    fi
    if [ $i -eq 12 ]; then
        log_error "Prometheus failed to start properly"
        exit 1
    fi
    sleep 5
done

for i in {1..12}; do
    if curl -s http://localhost:3000/api/health >/dev/null; then
        log_info "Grafana is healthy"
        break
    fi
    if [ $i -eq 12 ]; then
        log_warn "Grafana took longer than expected to start"
    fi
    sleep 5
done

# Test Prometheus targets
echo "🎯 Testing Prometheus targets..."
sleep 10
if curl -s "http://localhost:9090/api/v1/targets" | grep -q "up"; then
    log_info "Prometheus targets are being scraped"
else
    log_warn "Prometheus targets may need additional configuration"
fi

echo ""
echo "✅ Monitoring optimization complete!"
echo ""
echo "🔗 Access URLs:"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana: http://localhost:3000 (admin/xorb_admin_secure_2024)"
echo ""
echo "📊 Next steps:"
echo "   1. Configure Grafana dashboards for XORB services"
echo "   2. Set up alerting rules for production monitoring"
echo "   3. Test metrics collection from all XORB services"